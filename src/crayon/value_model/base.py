"""
Base classes for plasma parameter models.
"""

# Standard imports.
import abc
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.shared.constants import CoordinateSystem
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.dimensions import Dimension, Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.io import IONetcdf, write_netcdf_variable
from crayon.shared.types import Array, BooleanArray, FloatArray

logger = logging.getLogger(__name__)


class ModelType(CrayonEnum):
    """
    Type of value model.

    Attributes
    ----------
    CONSTANT
        Constant value.
    RAMP
        Value ramp.
    QUADRATIC_CHANNEL
        Quadratic channel.
    QUADRATIC_BOWL
        Quadratic bowl.
    QUADRATIC_WELL
        Quadratic well.
    SPLINE
        Spline fit.
    AXISYMMETRIC_MAGNETIC_FIELD
        Axisymmetric cylindrical magnetic field.
    """

    CONSTANT = 0
    RAMP = 1
    QUADRATIC_CHANNEL = 2
    QUADRATIC_BOWL = 3
    QUADRATIC_WELL = 4
    SPLINE = 5
    AXISYMMETRIC_MAGNETIC_FIELD = 6


class ValueCache:
    """
    Cache for value and derivatives of model at a single position.

    Attributes
    ----------
    hessian : np.array[float]
        Second derivative of value.
    jacobian : np.array[float]
        First derivative of value.
    jerk : np.array[float]
        Third derivative of value.
    snap : np.array[float]
        Fourth derivative of value.
    value : np.array[float]
        Value.
    """

    __slots__ = ("hessian", "jacobian", "jerk", "snap", "value")

    def __init__(
        self,
        input_dimension: Dimension,
        output_dimensions: list[Dimension],
        dtype: type,
    ):
        """
        Inits ValueCache.

        Parameters
        ----------
        input_dimension : Dimension
            Input value dimension.
        output_dimensions : list[Dimension]
            Output value dimensions.
        dtype : type
            Datatype of value.
        """
        input_size = input_dimension.size

        if output_dimensions:
            output_shape = tuple(d.size for d in output_dimensions)
        else:
            output_shape = ()

        self.value = np.zeros(output_shape, dtype=dtype)
        self.jacobian = np.zeros((*output_shape, input_size), dtype=dtype)
        self.hessian = np.zeros(
            (*output_shape, input_size, input_size), dtype=dtype
        )
        self.jerk = np.zeros(
            (*output_shape, input_size, input_size, input_size), dtype=dtype
        )
        self.snap = np.zeros(
            (*output_shape, input_size, input_size, input_size, input_size),
            dtype=dtype,
        )

    @classmethod
    def for_model(cls, model: "ValueModelBase") -> "ValueCache":
        """
        Construct a cache for a given model.

        Returns
        -------
        cache : ValueCache
            Cache for a given model.
        """
        return cls(model.input_dimension, model.output_dimensions, model.dtype)


class ValueModelBase(IONetcdf):
    """
    Base class for plasma parameter model.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        Coordinate system model is defined in.
    dtype : type
        Datatype of model value.
    input_bounds : np.array[float]
        Limits on components of input value.
    input_dimension : Dimension
        Dimension of input value.
    input_size : int
        Size of input value.
    output_dimensions : tuple[Dimension]
        Dimensions of output value.
    output_shape : tuple[int]
        Shape of output value.
    scale_factor : float
        Scale factor of value.
    units : str
        Units of value.
    """

    __slots__ = (
        "coordinate_system",
        "dtype",
        "input_bounds",
        "input_dimension",
        "input_size",
        "output_dimensions",
        "output_shape",
        "scale_factor",
        "units",
    )

    model_type: ModelType = NotImplemented

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        /,
        *,
        scale_factor: float = 1.0,
        input_bounds: FloatArray = None,
        dtype: type = float,
    ):
        """
        Inits ValueModelBase.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system model is defined in.
        input_dimension : Dimension
            Dimension of input value.
        output_dimensions : tuple[Dimension]
            Dimensions of output value.
        units : str
            Units of value.
        scale_factor : float, optional
            Scale factor of value.
        input_bounds : np.array[float], optional
            Limits on components of input value.
        dtype : type, optional
            Datatype of model value.
        """
        self.coordinate_system = CoordinateSystem.parse(coordinate_system)
        self.dtype = dtype
        self.scale_factor = float(scale_factor)
        self.units = units
        self.input_dimension = input_dimension
        self.input_size = self.input_dimension.size

        # Bounds on input values.
        input_bounds_shape = (self.input_size, 2)

        if input_bounds is None:
            self.input_bounds = np.empty(input_bounds_shape, dtype=self.dtype)
            self.input_bounds[..., 0] = -np.inf
            self.input_bounds[..., 1] = np.inf
        else:
            self.input_bounds = (
                np
                .ravel(input_bounds)
                .astype(dtype=self.dtype)
                .reshape(input_bounds_shape)
            )

        # Shape of output value.
        self.output_dimensions = output_dimensions

        if self.output_dimensions:
            self.output_shape = tuple(d.size for d in self.output_dimensions)
        else:
            self.output_shape = ()

    def _parse_abscissa(self, abscissa: FloatArray) -> BooleanArray:
        """
        Parse array of input values in place.

        Parameters
        ----------
        abscissa : np.array[float]
            Array of input values. Should have shape (n_points, input_size).

        Returns
        -------
        out_of_bounds : np.array[bool]
            Boolean array if input data in bounds of model.
        """
        out_of_bounds = np.any(
            np.logical_or(
                abscissa <= self.input_bounds[:, 0],
                abscissa >= self.input_bounds[:, 1],
            ),
            axis=1,
        )

        # Clip input values between bounds.
        np.clip(
            abscissa,
            self.input_bounds[..., 0],
            self.input_bounds[..., 1],
            out=abscissa,
        )

        return out_of_bounds

    def value(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: Array = None,
    ) -> Array:
        """
        Calculate model value.

        Parameters
        ----------
        abscissa : np.array[float]
            Array of input values.
        return_array : np.array
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        value : np.array
            Model value.
        """
        return self.__call__(abscissa, nu=0, return_array=return_array)

    def jacobian(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: Array = None,
    ) -> Array:
        """
        Calculate model first derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Array of input values.
        return_array : np.array
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        jacobian : np.array
            Model first derivative.
        """
        return self.__call__(abscissa, nu=1, return_array=return_array)

    def hessian(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: Array = None,
    ) -> Array:
        """
        Calculate model second derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Array of input values.
        return_array : np.array
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        hessian : np.array
            Model second derivative.
        """
        return self.__call__(abscissa, nu=2, return_array=return_array)

    def jerk(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: Array = None,
    ) -> Array:
        """
        Calculate model third derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Array of input values.
        return_array : np.array
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        jacobian : np.array
            Model third derivative.
        """
        return self.__call__(abscissa, nu=3, return_array=return_array)

    def snap(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: Array = None,
    ) -> Array:
        """
        Calculate model fourth derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Array of input values.
        return_array : np.array
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        jacobian : np.array
            Model fourth derivative.
        """
        return self.__call__(abscissa, nu=4, return_array=return_array)

    @abc.abstractmethod
    def value_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Calculate model value.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """

    @abc.abstractmethod
    def jacobian_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate first derivative of model value with respect to input.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """

    @abc.abstractmethod
    def hessian_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate second derivative of model value with respect to input.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """

    @abc.abstractmethod
    def jerk_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate third derivative of model value with respect to input.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """

    def snap_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate fourth derivative of model value with respect to input.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        raise NotImplementedError

    def __call__(
        self,
        abscissa: FloatArray,
        /,
        *,
        nu: int = 0,
        return_array: FloatArray = None,
    ):
        """
        Evaluate model value or derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Input value.
        nu : int, optional
            Order of derivative to evaluate.
        return_array : np.array
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        value : np.array
            Model value or derivative

        Raises
        ------
        ValueError
            Abscissa has incorrect shape.
            nu < 0 or > 4.
        """
        if nu < 0 or nu > 4:  # noqa: PLR2004
            raise ValueError(nu)

        # Calculate size of input.
        _abscissa = np.copy(abscissa).astype(float)

        # Figure out extra dimensions of input.
        if len(_abscissa.shape) > 0:
            final_dim = _abscissa.shape[-1]
            extra_dims = _abscissa.shape[:-1]
        else:
            final_dim = 1
            extra_dims = ()

        # np.prod(()) = 1
        _n = np.prod(extra_dims).astype(int)

        if final_dim != self.input_size:
            raise ValueError(
                "abscissa has incorrect shape. "
                f"Expected last dimension to have size {self.input_size}, "
                f"got shape {_abscissa.shape}"
            )

        # Flatten abscissa over extra dimensions and clip to input bounds.
        _abscissa = _abscissa.reshape((_n, self.input_size))
        out_of_bounds = self._parse_abscissa(_abscissa)

        output_shape = (
            *self.output_shape,
            *(self.input_size for _ in range(nu)),
        )

        # Flatten return array over extra dimensions if it exists.
        internal_shape = (_n, *output_shape)
        return_shape = (*extra_dims, *output_shape)

        if return_array is not None:
            return_array = return_array.reshape(internal_shape)

        return_array = get_return_array(return_array, internal_shape, float)

        if nu == 0:
            self.value_function(_abscissa, out_of_bounds, return_array)
        elif nu == 1:
            self.jacobian_function(_abscissa, out_of_bounds, return_array)
        elif nu == 2:  # noqa: PLR2004
            self.hessian_function(_abscissa, out_of_bounds, return_array)
        elif nu == 3:  # noqa: PLR2004
            self.jerk_function(_abscissa, out_of_bounds, return_array)
        elif nu == 4:  # noqa: PLR2004
            self.snap_function(_abscissa, out_of_bounds, return_array)
        else:
            raise NotImplementedError(nu)

        # Apply scale factor.
        return_array *= self.scale_factor

        # Unflatten extra dimensions if required.
        return np.reshape(return_array, return_shape)

    def get_cache(self) -> ValueCache:
        """
        Get cache for model.

        Returns
        -------
        value_cache : ValueCache
            Cache for model.
        """
        return ValueCache.for_model(self)

    def fill_cache(
        self, cache: ValueCache, abscissa: FloatArray, /, *, derivatives: int
    ):
        """
        Evaluate up to nu-th derivative of model value for a single input.

        Parameters
        ----------
        cache : ValueCache
            Cache to store results in.
        abscissa : np.array[float]
            Input values to evaluate model at.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            Abscissa has incorrect shape.
        """
        # Calculate size of input.
        _abscissa = np.copy(abscissa).astype(float)

        # Check input shape.
        if _abscissa.shape != (self.input_size,):
            raise ValueError(
                "abscissa has incorrect shape. "
                f"Expected ({self.input_size},), "
                f"got {_abscissa.shape}"
            )

        # Reshape to correct size for value functions.
        _abscissa = _abscissa.reshape((1, self.input_size))

        # Parse abscissa.
        out_of_bounds = self._parse_abscissa(_abscissa)

        # Create a view onto the target array with correct shape for the
        # value function (add 1 sized dimension at front).
        # This will also raise if it cannot do so without making a copy which
        # prevents any issues setting data in a copy of the array instead of
        # the array itself.
        _value = cache.value.view().reshape((1, *cache.value.shape))
        _jacobian = cache.jacobian.view().reshape((1, *cache.jacobian.shape))
        _hessian = cache.hessian.view().reshape((1, *cache.hessian.shape))
        _jerk = cache.jerk.view().reshape((1, *cache.jerk.shape))
        _snap = cache.snap.view().reshape((1, *cache.snap.shape))

        # Fill cache.
        self._fill_cache(
            _abscissa,
            out_of_bounds,
            derivatives,
            _value,
            _jacobian,
            _hessian,
            _jerk,
            _snap,
        )

        # Apply scale factor.
        _value *= self.scale_factor
        _jacobian *= self.scale_factor
        _hessian *= self.scale_factor
        _jerk *= self.scale_factor
        _snap *= self.scale_factor

    def _fill_cache(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        derivative: int,
        value: FloatArray,
        jacobian: FloatArray,
        hessian: FloatArray,
        jerk: FloatArray,
        snap: FloatArray,
    ):
        """
        Evaluate up to nu-th derivative of model value for a single input.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        derivative : int
            Order of derivative. Must be > 0 or <= 3.
        value : np.array[float]
            Array to store value result.
        jacobian : np.array[float]
            Array to store first derivative result.
        hessian : np.array[float]
            Array to store second derivative result.
        jerk : np.array[float]
            Array to store third derivative result.
        snap : np.array[float]
            Array to store fourth derivative result.

        Raises
        ------
        ValueError
            derivative < 0 or >= 5.
        """
        if derivative < 0 or derivative >= 5:  # noqa: PLR2004
            raise ValueError(derivative)

        if derivative >= 0:
            self.value_function(abscissa, out_of_bounds, value)

        if derivative >= 1:
            self.jacobian_function(abscissa, out_of_bounds, jacobian)

        if derivative >= 2:  # noqa: PLR2004
            self.hessian_function(abscissa, out_of_bounds, hessian)

        if derivative >= 3:  # noqa: PLR2004
            self.jerk_function(abscissa, out_of_bounds, jerk)

        if derivative >= 4:  # noqa: PLR2004
            self.snap_function(abscissa, out_of_bounds, snap)

    def write_netcdf(self, group: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group.setncattr("model_type", self.model_type.name)
        group.setncattr("coordinate_system", self.coordinate_system.name)
        group.setncattr("input_dimension", self.input_dimension.name)
        group.setncattr(
            "output_dimensions",
            ", ".join(d.name for d in self.output_dimensions),
        )
        group.setncattr("units", self.units)

        write_netcdf_variable(
            group,
            "scale_factor",
            (),
            self.scale_factor,
            "Scale factor for model value",
            "",
        )

        write_netcdf_variable(
            group,
            "input_bounds",
            (self.input_dimension, Dimensions.two),
            self.input_bounds,
            "Min / max values for input values",
            "",
        )

    @staticmethod
    def read_shared_netcdf(
        group: nc4.Group,
    ) -> tuple[
        CoordinateSystem, Dimension, tuple[Dimension], str, float, FloatArray
    ]:
        """
        Read common values from netCDF4 file.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        coordinate_system : CoordinateSystem
            Coordinate system model is defined in.
        input_dimension : Dimension
            Dimension of input value.
        output_dimensions : tuple[Dimension]
            Dimensions of output value.
        units : str
            Units of value.
        scale_factor : float, optional
            Scale factor of value.
        input_bounds : np.array[float], optional
            Limits on components of input value.
        """
        coordinate_system = CoordinateSystem.parse(
            group.getncattr("coordinate_system")
        )
        input_dimension = Dimensions.get_dim(
            group.getncattr("input_dimension")
        )
        output_dimensions = tuple(
            Dimensions.get_dim(name)
            for name in group.getncattr("output_dimensions").split(", ")
            if name
        )
        units = group.getncattr("units")

        scale_factor = group["scale_factor"][...].item()
        input_bounds = group["input_bounds"][:, :].data

        return (
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            input_bounds,
        )
