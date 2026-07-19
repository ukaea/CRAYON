"""
Spline fit models.
"""

# Standard imports
import itertools
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np
from scipy import interpolate

# Local imports
from crayon.shared.constants import CoordinateSystem
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.dimensions import Dimension
from crayon.shared.io import write_netcdf_variable
from crayon.shared.types import BooleanArray, FloatArray
from crayon.value_model.base import ModelType, ValueModelBase

logger = logging.getLogger(__name__)


class SplineMethod(CrayonEnum):
    """
    Method for spline fit.

    Attributes
    ----------
    LINEAR
        Linear fit
    CUBIC
        Cubic fit
    QUINTIC
        Quintic fit
    """

    LINEAR = 1
    CUBIC = 3
    QUINTIC = 5


_COLON = slice(None)


class SplineBase(ValueModelBase):
    """
    Base class for spline model.
    """

    __slots__ = (
        "_abscissas",
        "_data",
        "_dependent_components",
        "_dim",
        "_index_map",
        "_interpolator",
        "method",
    )

    model_type = ModelType.SPLINE

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        method: str | SplineMethod,
        dependent_components: tuple[bool],
        dtype: type,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Inits SplineBase.

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
        method : str | SplineMethod
            Method of spline fit.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        dtype : type
            Datatype of model value.
        scale_factor : float, optional
            Scale factor of value.

        Raises
        ------
        ValueError
            dependent_components has incorrect shape for input dimension.
        """
        # Initialise super class.
        super().__init__(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor=scale_factor,
            dtype=dtype,
        )

        # Convert to int then bool so it does it correctly.
        self._dependent_components = np.asarray(
            dependent_components, dtype=int
        ).astype(bool)

        if self._dependent_components.size != self.input_size:
            raise ValueError(
                "dependent_components has incorrect size compared to "
                "input_dimension. "
                f"Expected {self.input_size}, "
                f"got {self._dependent_components.size}"
            )

        # Dimension of interpolator.
        self._dim = sum(self._dependent_components.astype(int))

        # Map from dependent coordinate index to dimension index.
        j = 0

        self._index_map = {}
        for i, dependent in enumerate(self._dependent_components):
            if dependent:
                self._index_map[j] = i
                j += 1

        # Set interpolator method.
        self.method = SplineMethod.parse(method)

        # FITPACK cannot handle piecewise continuous splines so will have to
        # force order > 3 for now.
        self.method = SplineMethod.QUINTIC

    @property
    def abscissas(self) -> tuple[FloatArray]:
        """
        Abscissas for fitted data.

        Returns
        -------
        abscissas : tuple[np.array[float]]
            Abscissas.
        """
        return self._abscissas

    @property
    def data(self) -> FloatArray:
        """
        Data spline is fitted to.

        Returns
        -------
        data : np.array[float]
            Data.
        """
        return self._data

    def _parse_abscissa(
        self,
        abscissa: FloatArray,
    ) -> bool:
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
        # abscissa has shape (n, x)
        # input_bounds has shape (x, 2)
        out_of_bounds = np.any(
            np.logical_or(
                abscissa <= self.input_bounds[..., 0],
                abscissa >= self.input_bounds[..., 1],
            ),
            axis=1,
        )

        # Clip input values between bounds.
        for coordinate_index in self._index_map:
            min_value, max_value = self.input_bounds[coordinate_index]
            np.clip(
                abscissa[:, coordinate_index],
                min_value,
                max_value,
                out=abscissa[:, coordinate_index],
            )

        return out_of_bounds

    def write_netcdf(self, group: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        # Write common values for all value models.
        super().write_netcdf(group)

        # Write spline method.
        group.setncattr("dim", self._dim)
        group.setncattr("method", self.method.name)

        # Write abscissa.
        dimensions = []

        for i, data in enumerate(self._abscissas, start=1):
            _dim = Dimension(f"abscissa_{i}", data.size)

            group.createDimension(_dim.name, _dim.size)
            write_netcdf_variable(
                group,
                _dim.name,
                (_dim,),
                data,
                f"Coordinate component {i}",
                "",
            )

            dimensions.append(_dim)

        # Write dependent components.
        write_netcdf_variable(
            group,
            "dependent_components",
            (self.input_dimension,),
            self._dependent_components.astype(int),
            "Which coordinate components the data depends on",
            "",
        )

        # Write data.
        write_netcdf_variable(
            group,
            "data",
            (*dimensions, *self.output_dimensions),
            self._data,
            "Input data",
            self.units,
        )

    @staticmethod
    def read_shared_netcdf(group: nc4.Group):
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
        abscissas : tuple[np.array[float]]
            Array of input value components in each dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        method : str | SplineMethod
            Method of spline fit.
        """
        (
            coordinate,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            input_bounds,
        ) = ValueModelBase.read_shared_netcdf(group)

        abscissas = tuple(
            group[d][:].data
            for d in group["data"].dimensions
            if d.startswith("abscissa")
        )
        data = group["data"][...].data
        dependent_components = group["dependent_components"][:].data.astype(
            bool
        )
        method = SplineMethod.parse(group.getncattr("method"))

        return (
            coordinate,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            input_bounds,
            abscissas,
            data,
            dependent_components,
            method,
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "SplineBase":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        spline_base : SplineBase
            Spline base object.

        Raises
        ------
        ValueError
            Unsupported spline dimension.
        """
        dim = group.getncattr("dim")

        if dim == 1:
            return Spline1D.read_netcdf(group)
        if dim == 2:  # noqa: PLR2004
            return Spline2D.read_netcdf(group)
        if dim == 3:  # noqa: PLR2004
            return Spline3D.read_netcdf(group)

        raise ValueError(dim)


class Spline1D(SplineBase):
    """
    1D spline fit.
    """

    __slots__ = ()

    model_type = ModelType.SPLINE

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        abscissa: FloatArray,
        data: FloatArray,
        dependent_components: tuple[bool],
        /,
        *,
        scale_factor: float = 1.0,
        method: SplineMethod = SplineMethod.CUBIC,
    ):
        """
        Inits Spline1D.

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
        abscissa : np.array[float]
            Input value components in first dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        scale_factor : float, optional
            Scale factor of value.
        method : str | SplineMethod, optional
            Method of spline fit.

        Raises
        ------
        ValueError
            Abscissa not 1D.
            dependent_components has incorrect shape for input dimension.
        """
        # Parse data into numpy arrays.
        self._abscissas = (np.asarray(abscissa),)
        self._data = np.asarray(data)

        # Check shapes.
        if self._abscissas[0].ndim > 1:
            raise ValueError(
                f"abscissa must be 1D: {self._abscissas[0].shape}"
            )

        super().__init__(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            method,
            dependent_components,
            self._data.dtype,
            scale_factor=scale_factor,
        )

        if self._dim != 1:
            raise ValueError(
                "Incorrect number of dependent coordinate "
                f"components for 1d spline: {self._dependent_components}"
            )

        # Set input bounds to limits of input abscissas.
        for i, j in self._index_map.items():
            self.input_bounds[j, 0] = self._abscissas[i][0]
            self.input_bounds[j, 1] = self._abscissas[i][-1]

        # Create interpolator (BSpline).
        self._interpolator = interpolate.make_interp_spline(
            self._abscissas[0],
            self._data,
            k=self.method.value,
        )

    def value_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model value.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        idx_1 = self._index_map[0]
        return_array[:, ...] = self._interpolator(abscissa[:, idx_1])

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
        idx_1 = self._index_map[0]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if np.any(in_bounds):
            return_array[in_bounds, ..., idx_1] = self._interpolator(
                abscissa[in_bounds, idx_1], nu=1
            )

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
        idx_1 = self._index_map[0]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if np.any(in_bounds):
            return_array[in_bounds, ..., idx_1, idx_1] = self._interpolator(
                abscissa[in_bounds, idx_1], nu=2
            )

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
        idx_1 = self._index_map[0]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if np.any(in_bounds):
            return_array[in_bounds, ..., idx_1, idx_1, idx_1] = (
                self._interpolator(abscissa[in_bounds, idx_1], nu=3)
            )

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
        idx_1 = self._index_map[0]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if np.any(in_bounds):
            return_array[in_bounds, ..., idx_1, idx_1, idx_1, idx_1] = (
                self._interpolator(abscissa[in_bounds, idx_1], nu=4)
            )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Spline1D":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        spline_1d : Spline1D
            1D spline fit.
        """
        (
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            _,
            abscissas,
            data,
            dependent_components,
            method,
        ) = SplineBase.read_shared_netcdf(group)

        return cls(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            abscissas[0],
            data,
            dependent_components,
            scale_factor=scale_factor,
            method=method,
        )


class Spline2D(SplineBase):
    """
    2D spline fit.
    """

    __slots__ = ()

    model_type = ModelType.SPLINE

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        abscissa_1: FloatArray,
        abscissa_2: FloatArray,
        data: FloatArray,
        dependent_components: tuple[bool],
        /,
        *,
        scale_factor: float = 1.0,
        method: SplineMethod = SplineMethod.CUBIC,
    ):
        """
        Inits Spline2D.

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
        abscissa_1 : np.array[float]
            Input value components in first dimension.
        abscissa_2 : np.array[float]
            Input value components in second dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        scale_factor : float, optional
            Scale factor of value.
        method : str | SplineMethod, optional
            Method of spline fit.

        Raises
        ------
        ValueError
            abscissa_1 or abscissa_2 not 1D.
            data incorrect shape.
            dependent_components has incorrect shape for input dimension.
        """
        # Parse data into numpy arrays.
        self._abscissas = (np.asarray(abscissa_1), np.asarray(abscissa_2))
        self._data = np.asarray(data)

        # Check shapes.
        if self._abscissas[0].ndim > 1:
            raise ValueError(
                f"abscissa_1 must be 1D: {self._abscissas[0].shape}"
            )
        if self._abscissas[1].ndim > 1:
            raise ValueError(
                f"abscissa_2 must be 1D: {self._abscissas[1].shape}"
            )

        _expected_shape = (self._abscissas[0].size, self._abscissas[1].size)

        if self._data.shape[:2] != _expected_shape:
            raise ValueError(
                "First two dimensions of data must be "
                f"{_expected_shape}, got shape {data.shape}"
            )

        super().__init__(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            method,
            dependent_components,
            self._data.dtype,
            scale_factor=scale_factor,
        )

        if self._dim != 2:  # noqa: PLR2004
            raise ValueError(
                "Incorrect number of dependent coordinate "
                f"components for 2d spline: {self._dependent_components}"
            )

        # Set input bounds to limits of input abscissas.
        for i, j in self._index_map.items():
            self.input_bounds[j, 0] = self._abscissas[i][0]
            self.input_bounds[j, 1] = self._abscissas[i][-1]

        # Create interpolator (Bivariate BSpline).
        # RectBivariateSpline can only handle scalar valued _data unlike the
        # routines used in 1D and 3D splines. For convenience, create a spline
        # over each component of data, held in a numpy array.
        self._interpolator = np.empty(self.output_shape, dtype=object)

        for indicies in itertools.product(
            *(range(d) for d in self.output_shape)
        ):
            self._interpolator[indicies] = interpolate.RectBivariateSpline(
                self._abscissas[0],
                self._abscissas[1],
                self._data[(_COLON, _COLON, *indicies)],
                kx=self.method.value,
                ky=self.method.value,
            )

    def value_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model value.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]

        for indicies in itertools.product(
            *(range(d) for d in self.output_shape)
        ):
            _interpolator = self._interpolator[indicies]

            return_array[(_COLON, *indicies)] = _interpolator.ev(
                abscissa[:, idx_1],
                abscissa[:, idx_2],
            )

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
        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        arg = (abscissa[in_bounds, idx_1], abscissa[in_bounds, idx_2])

        for indicies in itertools.product(
            *(range(d) for d in self.output_shape)
        ):
            _interpolator = self._interpolator[indicies]

            for (i,) in itertools.combinations_with_replacement(
                (idx_1, idx_2), 1
            ):
                dx = int(i == idx_1)
                dy = 1 - dx

                return_array[(in_bounds, *indicies, i)] = _interpolator.ev(
                    *arg, dx=dx, dy=dy
                )

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
        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        arg = (abscissa[in_bounds, idx_1], abscissa[in_bounds, idx_2])

        for indicies in itertools.product(
            *(range(d) for d in self.output_shape)
        ):
            _interpolator = self._interpolator[indicies]

            for i, j in itertools.combinations_with_replacement(
                (idx_1, idx_2), 2
            ):
                dx = int(i == idx_1) + int(j == idx_1)
                dy = 2 - dx

                return_array[(in_bounds, *indicies, i, j)] = _interpolator.ev(
                    *arg, dx=dx, dy=dy
                )

        # Second order partial derivatives commute.
        for i, j in itertools.combinations_with_replacement((idx_1, idx_2), 2):
            # Get all other permutations of indicies.
            permutations = iter(set(itertools.permutations((i, j))))

            for _i, _j in permutations:
                if (_i, _j) == (i, j):
                    continue

                return_array[:, ..., _i, _j] = return_array[:, ..., i, j]

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
        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        arg = (abscissa[in_bounds, idx_1], abscissa[in_bounds, idx_2])

        for indicies in itertools.product(
            *(range(d) for d in self.output_shape)
        ):
            _interpolator = self._interpolator[indicies]

            for i, j, k in itertools.combinations_with_replacement(
                (idx_1, idx_2), 3
            ):
                dx = int(i == idx_1) + int(j == idx_1) + int(k == idx_1)
                dy = 3 - dx

                return_array[(in_bounds, *indicies, i, j, k)] = (
                    _interpolator.ev(*arg, dx=dx, dy=dy)
                )

        # Third order partial derivatives commute.
        for i, j, k in itertools.combinations_with_replacement(
            (idx_1, idx_2), 3
        ):
            # Get all other permutations of indicies.
            permutations = iter(set(itertools.permutations((i, j, k))))

            for _i, _j, _k in permutations:
                if (_i, _j, _k) == (i, j, k):
                    continue

                return_array[:, ..., _i, _j, _k] = return_array[
                    :, ..., i, j, k
                ]

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
        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]

        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        arg = (abscissa[in_bounds, idx_1], abscissa[in_bounds, idx_2])

        for indicies in itertools.product(
            *(range(d) for d in self.output_shape)
        ):
            _interpolator = self._interpolator[indicies]

            for i, j, k, m in itertools.combinations_with_replacement(
                (idx_1, idx_2), 4
            ):
                dx = (
                    int(i == idx_1)
                    + int(j == idx_1)
                    + int(k == idx_1)
                    + int(m == idx_1)
                )
                dy = 4 - dx

                return_array[(in_bounds, *indicies, i, j, k, m)] = (
                    _interpolator.ev(*arg, dx=dx, dy=dy)
                )

        # Fourth order partial derivatives commute.
        for i, j, k, m in itertools.combinations_with_replacement(
            (idx_1, idx_2), 4
        ):
            # Get all other permutations of indicies.
            permutations = iter(set(itertools.permutations((i, j, k, m))))

            for _i, _j, _k, _m in permutations:
                if (_i, _j, _k, _m) == (i, j, k, m):
                    continue

                return_array[:, ..., _i, _j, _k, _m] = return_array[
                    :, ..., i, j, k, m
                ]

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Spline2D":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        spline_2d : Spline2D
            2D spline fit.
        """
        (
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            _,
            abscissas,
            data,
            dependent_components,
            method,
        ) = SplineBase.read_shared_netcdf(group)

        return cls(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            abscissas[0],
            abscissas[1],
            data,
            dependent_components,
            scale_factor=scale_factor,
            method=method,
        )


class Spline3D(SplineBase):
    """
    3D spline fit.
    """

    __slots__ = ()

    model_type = ModelType.SPLINE

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        abscissa_1: FloatArray,
        abscissa_2: FloatArray,
        abscissa_3: FloatArray,
        data: FloatArray,
        dependent_components: tuple[bool],
        /,
        *,
        scale_factor: float = 1.0,
        method: str = "cubic",
    ):
        """
        Inits Spline3D.

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
        abscissa_1 : np.array[float]
            Input value components in first dimension.
        abscissa_2 : np.array[float]
            Input value components in second dimension.
        abscissa_3 : np.array[float]
            Input value components in third dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        scale_factor : float, optional
            Scale factor of value.
        method : str | SplineMethod, optional
            Method of spline fit.

        Raises
        ------
        ValueError
            abscissa_1, abscissa_2 or abscissa_3 not 1D.
            data incorrect shape.
            dependent_components has incorrect shape for input dimension.
        """
        # Parse data into numpy arrays.
        self._abscissas = (
            np.asarray(abscissa_1),
            np.asarray(abscissa_2),
            np.asarray(abscissa_3),
        )
        self._data = np.asarray(data)

        # Check shapes.
        if self._abscissas[0].ndim > 1:
            raise ValueError(
                f"abscissa_1 must be 1D: {self._abscissas[0].shape}"
            )

        if self._abscissas[1].ndim > 1:
            raise ValueError(
                f"abscissa_2 must be 1D: {self._abscissas[1].shape}"
            )

        if self._abscissas[2].ndim > 1:
            raise ValueError(
                f"abscissa_3 must be 1D: {self._abscissas[2].shape}"
            )

        _expected_shape = (
            self._abscissas[0].size,
            self._abscissas[1].size,
            self._abscissas[2].size,
        )

        if self._data.shape[:3] != _expected_shape:
            raise ValueError(
                "First three dimensions of data must be "
                f"{_expected_shape}, got shape {data.shape}"
            )

        super().__init__(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            method,
            dependent_components,
            self._data.dtype,
            scale_factor=scale_factor,
        )

        if self._dim != 3:  # noqa: PLR2004
            raise ValueError(
                "Incorrect number of dependent coordinate "
                f"components for 3d spline: {self._dependent_components}"
            )

        # Set input bounds to limits of input abscissas.
        for i, j in self._index_map.items():
            self.input_bounds[j, 0] = self._abscissas[i][0]
            self.input_bounds[j, 1] = self._abscissas[i][-1]

        # Create interpolator (Tensor Product BSpline).
        self._interpolator = interpolate.RegularGridInterpolator(
            self._abscissas, self._data, method=self.method.name.lower()
        )

    def value_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model value.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]
        idx_3 = self._index_map[2]

        arg = (abscissa[:, idx_1], abscissa[:, idx_2], abscissa[:, idx_3])

        return_array[:, ...] = self._interpolator(arg)

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
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]
        idx_3 = self._index_map[2]

        arg = (
            abscissa[in_bounds, idx_1],
            abscissa[in_bounds, idx_2],
            abscissa[in_bounds, idx_3],
        )

        _n = self.input_dimension.size
        nu = np.empty(_n, dtype=int)

        for (i,) in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 1
        ):
            nu[0] = int(i == idx_1)
            nu[1] = int(i == idx_2)
            nu[2] = int(i == idx_3)

            return_array[in_bounds, ..., i] = self._interpolator(arg, nu=nu)

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
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]
        idx_3 = self._index_map[2]

        arg = (
            abscissa[in_bounds, idx_1],
            abscissa[in_bounds, idx_2],
            abscissa[in_bounds, idx_3],
        )

        # nu is partial derivative order in each dimension.
        _n = self.input_dimension.size
        nu = np.empty(_n, dtype=int)

        for i, j in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 2
        ):
            nu[0] = int(i == idx_1) + int(j == idx_1)
            nu[1] = int(i == idx_2) + int(j == idx_2)
            nu[2] = 2 - nu[0] - nu[1]

            return_array[in_bounds, ..., i, j] = self._interpolator(arg, nu=nu)

        # Second order partial derivatives commute.
        for i, j in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 2
        ):
            # Get all other permutations of indicies.
            permutations = iter(set(itertools.permutations((i, j))))

            for _i, _j in permutations:
                if (_i, _j) == (i, j):
                    continue

                return_array[:, ..., _i, _j] = return_array[:, ..., i, j]

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
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]
        idx_3 = self._index_map[2]

        arg = (
            abscissa[in_bounds, idx_1],
            abscissa[in_bounds, idx_2],
            abscissa[in_bounds, idx_3],
        )

        # nu is partial derivative order in each dimension.
        _n = self.input_dimension.size
        nu = np.empty(_n, dtype=int)

        for i, j, k in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 3
        ):
            nu[0] = int(i == idx_1) + int(j == idx_1) + int(k == idx_1)
            nu[1] = int(i == idx_2) + int(j == idx_2) + int(k == idx_2)
            nu[2] = 3 - nu[0] - nu[1]

            return_array[in_bounds, ..., i, j, k] = self._interpolator(
                arg, nu=nu
            )

        # Third order partial derivatives commute.
        for i, j, k in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 3
        ):
            # Get all other permutations of indicies.
            permutations = iter(set(itertools.permutations((i, j, k))))

            for _i, _j, _k in permutations:
                if (_i, _j, _k) == (i, j, k):
                    continue

                return_array[:, ..., _i, _j, _k] = return_array[
                    :, ..., i, j, k
                ]

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
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        idx_1 = self._index_map[0]
        idx_2 = self._index_map[1]
        idx_3 = self._index_map[2]

        arg = (
            abscissa[in_bounds, idx_1],
            abscissa[in_bounds, idx_2],
            abscissa[in_bounds, idx_3],
        )

        # nu is partial derivative order in each dimension.
        _n = self.input_dimension.size
        nu = np.empty(_n, dtype=int)

        for i, j, k, m in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 4
        ):
            nu[0] = (
                int(i == idx_1)
                + int(j == idx_1)
                + int(k == idx_1)
                + int(m == idx_1)
            )
            nu[1] = (
                int(i == idx_2)
                + int(j == idx_2)
                + int(k == idx_2)
                + int(m == idx_2)
            )
            nu[2] = 4 - nu[0] - nu[1]

            return_array[in_bounds, ..., i, j, k, m] = self._interpolator(
                arg, nu=nu
            )

        # Third order partial derivatives commute.
        for i, j, k, m in itertools.combinations_with_replacement(
            (idx_1, idx_2, idx_3), 4
        ):
            # Get all other permutations of indicies.
            permutations = iter(set(itertools.permutations((i, j, k, m))))

            for _i, _j, _k, _m in permutations:
                if (_i, _j, _k, _m) == (i, j, k, m):
                    continue

                return_array[:, ..., _i, _j, _k, _m] = return_array[
                    :, ..., i, j, k, m
                ]

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Spline3D":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        spline_3d : Spline3D
            3D spline fit.
        """
        (
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            _,
            abscissas,
            data,
            dependent_components,
            method,
        ) = SplineBase.read_shared_netcdf(group)

        return cls(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            abscissas[0],
            abscissas[1],
            abscissas[2],
            data,
            dependent_components,
            scale_factor=scale_factor,
            method=method,
        )
