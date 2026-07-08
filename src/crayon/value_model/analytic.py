"""
Analytic plasma parameter models.
"""

# Standard imports
import abc
import itertools
import logging
import typing

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.shared.constants import CoordinateSystem
from crayon.shared.dimensions import Dimension, Dimensions
from crayon.shared.io import write_netcdf_variable
from crayon.shared.types import BooleanArray, FloatArray
from crayon.value_model.base import ModelType, ValueModelBase

logger = logging.getLogger(__name__)


class Constant(ValueModelBase):
    """
    A model with a constant value.

    Attributes
    ----------
    constant_value : np.array
        Model constant value.

    Methods
    -------
    write_netcdf
        Write to netCDF4 dataset.
    read_netcdf
        Load from netCDF4 dataset.
    """

    __slots__ = ("constant_value",)

    model_type = ModelType.CONSTANT

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        constant_value: FloatArray,
        /,
        *,
        scale_factor: float = 1.0,
        input_bounds: FloatArray = None,
    ):
        """
        Inits Constant.

        Parameters
        ----------
        coordinate_system: CoordinateSystem
            Coordinate system model is defined for.
        input_dimension: Dimension
            Dimension for input value to model.
        output_dimensions: tuple[Dimension]
            Dimensions for output value from model.
        units: str
            Units of model.
        constant_value: np.array[float]
            Constant value of model.
        scale_factor: float, optional
            Scale factor for model.
        input_bounds: np.array[float], optional
            Bounds on coordinate components

        Raises
        ------
        ValueError
            Constant value has incorrect shape for output dimensions.
        """
        self.constant_value = np.asarray(constant_value)

        # Initialise super class.
        super().__init__(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor=scale_factor,
            input_bounds=input_bounds,
            dtype=self.constant_value.dtype,
        )

        if self.constant_value.shape != self.output_shape:
            raise ValueError(
                "constant_value has incorrect shape."
                f"Expected {self.output_shape}, "
                f"got {self.constant_value.shape}."
            )

    def value_function(
        self,
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model value.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array[:, ...] = self.constant_value

    @staticmethod
    def jacobian_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate first derivative of model value with respect to input.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array.fill(0.0)

    @staticmethod
    def hessian_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate second derivative of model value with respect to input.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array.fill(0.0)

    @staticmethod
    def jerk_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate third derivative of model value with respect to input.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array.fill(0.0)

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

        # Write model specific values.
        write_netcdf_variable(
            group,
            "constant_value",
            self.output_dimensions,
            self.constant_value,
            "Constant output value",
            self.units,
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Constant":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        constant : Constant
            Model loaded from netCDF4 data.
        """
        (
            coordinate,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            input_bounds,
        ) = ValueModelBase.read_shared_netcdf(group)

        constant_value = group["constant_value"][...].data

        return cls(
            coordinate,
            input_dimension,
            output_dimensions,
            units,
            constant_value,
            scale_factor=scale_factor,
            input_bounds=input_bounds,
        )


class CnRamp(ValueModelBase):
    """
    A model with a value ramp of given smoothness class.

    Attributes
    ----------
    direction : np.array[float]
        Direction of value ramp.
    metric : np.array[float]
        Metric tensor at origin.
    origin : np.array[float]
        Origin of value ramp.
    ramp_width : np.array[float]
        Distance over which value ramps from y0 to y1.
    y0 : np.array[float]
        Value at bottom of ramp i.e. at origin.
    y1 : np.array[float]
        Value at top of ramp.
    smoothness : int
        Smoothness class of ramp.

    Methods
    -------
    with_smoothness
        Create ramp model with given smoothness class.
    set_metric
        Set value of metric tensor at origin.
    ramp
        Calculate ramp function value.
    ramp_ds
        Calculate first derivative of ramp function.
    ramp_ds2
        Calculate second derivative of ramp function.
    ramp_ds3
        Calculate third derivative of ramp function.
    write_netcdf
        Write to netCDF4 dataset.
    read_netcdf
        Load from netCDF4 dataset.
    """

    __slots__ = (
        "_ramp_height",
        "direction",
        "metric",
        "origin",
        "ramp_width",
        "y0",
        "y1",
    )

    model_type = ModelType.RAMP
    smoothness = NotImplemented

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Inits CnRamp.

        Parameters
        ----------
        coordinate_system: CoordinateSystem
            Coordinate system model is defined for.
        input_dimension: Dimension
            Dimension for input value to model.
        output_dimensions: tuple[Dimension]
            Dimensions for output value from model.
        units: str
            Units of model.
        origin : np.array[float]
            Point at which value is y0.
        direction : np.array[float]
            Direction parallel to the value gradient.
        y0 : np.array[float]
            Minimum value of the function.
        y1 : np.array[float]
            Maximum value of the function.
        ramp_width : float
           Distance over which value varies from y0 and y1. Must be > 0.
        scale_factor: float, optional
            Scale factor for model.

        Raises
        ------
        ValueError
            y0 or y1 has incorrect shape for output dimensions.
            Origin or direction has incorrect shape for input dimension.
            Ramp width is negative.
        """
        self.y0 = np.asarray(y0)
        self.y1 = np.asarray(y1)

        super().__init__(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor=scale_factor,
            dtype=self.y0.dtype,
        )

        if self.y0.shape != self.output_shape:
            raise ValueError(
                "y0 has incorrect shape. "
                f"Expected {self.output_shape}, "
                f"got {self.y0.shape}."
            )

        if self.y0.shape != self.y1.shape:
            raise ValueError(
                "y0 and y1 must have same shape: "
                f"{self.y0.shape}, {self.y1.shape}"
            )

        self.origin = np.array(origin)
        self.direction = np.array(direction)

        expected_shape = (input_dimension.size,)
        if self.origin.shape != expected_shape:
            raise ValueError(
                "origin has incorrect shape. "
                f"Expected {self.origin.shape}, got {expected_shape}."
            )

        if self.direction.shape != expected_shape:
            raise ValueError(
                "direction has incorrect shape. "
                f"Expected {self.direction.shape}, got {expected_shape}."
            )

        if self.direction.size > 1:
            norm = np.linalg.norm(self.direction)
            if np.isclose(norm, 0.0):
                raise ValueError("direction length is zero")
            self.direction /= norm

        if coordinate_system == CoordinateSystem.CARTESIAN:
            self.set_metric(np.eye(input_dimension.size))
        else:
            self.metric = None

        self._ramp_height = self.y1 - self.y0
        self.ramp_width = float(ramp_width)

        if self.ramp_width <= 0.0:
            raise ValueError(f"ramp_width must be positive: {self.ramp_width}")

    @classmethod
    def with_smoothness(
        cls,
        coordinate_system: CoordinateSystem,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        smoothness: int,
        /,
        *,
        scale_factor: float = 1.0,
    ) -> typing.Union["C0Ramp", "C1Ramp", "C2Ramp"]:
        """
        Create ramp model with given smoothness class.

        Parameters
        ----------
        coordinate_system: CoordinateSystem
            Coordinate system model is defined for.
        input_dimension: Dimension
            Dimension for input value to model.
        output_dimensions: tuple[Dimension]
            Dimensions for output value from model.
        units: str
            Units of model.
        origin : np.array[float]
            Point at which value is y0.
        direction : np.array[float]
            Direction parallel to the value gradient.
        y0 : np.array[float]
            Minimum value of the function.
        y1 : np.array[float]
            Maximum value of the function.
        ramp_width : float
           Distance over which value varies from y0 and y1. Must be > 0.
        smoothness : int
            Smoothness class of ramp.

        Returns
        -------
        ramp : C0Ramp | C1Ramp | C2Ramp
            Value ramp model.

        Raises
        ------
        ValueError
            Unsupported smoothness class.
        """
        if smoothness == 0:
            return C0Ramp(
                coordinate_system,
                input_dimension,
                output_dimensions,
                units,
                origin,
                direction,
                y0,
                y1,
                ramp_width,
                scale_factor=scale_factor,
            )

        if smoothness == 1:
            return C1Ramp(
                coordinate_system,
                input_dimension,
                output_dimensions,
                units,
                origin,
                direction,
                y0,
                y1,
                ramp_width,
                scale_factor=scale_factor,
            )

        if smoothness == 2:  # noqa: PLR2004
            return C2Ramp(
                coordinate_system,
                input_dimension,
                output_dimensions,
                units,
                origin,
                direction,
                y0,
                y1,
                ramp_width,
                scale_factor=scale_factor,
            )

        raise ValueError(smoothness)

    def set_metric(self, metric: FloatArray):
        """
        Set value of metric tensor at origin.

        Parameters
        ----------
        metric : np.array[float]
            Metric tensor at origin.

        Raises
        ------
        ValueError
            metric has incorrect shape for input dimension.
        """
        self.metric = np.asarray(metric)

        if self.metric.shape != (
            self.input_dimension.size,
            self.input_dimension.size,
        ):
            raise ValueError(
                "metric has incorrect shape. "
                f"Expected ({self.origin.size}, {self.origin.size}), "
                f"got {self.metric.shape}"
            )

    def normalised_position(
        self,
        abscissa: FloatArray,
    ) -> FloatArray:
        """
        Calculate normalised position along ramp s. At origin s = 0 while at
        origin + direction s = 1.

        abscissa : np.array[float]
            Input value.

        Returns
        -------
        normalised_position : np.array[float]
            Normalised position along ramp.
        """
        return_array = np.empty(abscissa.shape[0])

        np.einsum(
            "ij, ...i, j -> ...",
            self.metric,
            abscissa - self.origin,
            self.direction,
            out=return_array,
        )

        return_array /= self.ramp_width

        # Reshape to allow multiplication by something of shape y0 and still
        # broadcast correctly.
        return return_array.reshape((
            abscissa.shape[0],
            *itertools.repeat(1, self.y0.ndim),
        ))

    @abc.abstractmethod
    def ramp(self, s: FloatArray) -> FloatArray:
        """
        Calculate ramp function value. This equals 0 at s = 0 and 1 at s = 1
        with intermediate values determined by the smoothness class.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function value.
        """

    @abc.abstractmethod
    def ramp_ds(self, s: FloatArray) -> FloatArray:
        """
        Calculate first derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function first derivative.
        """

    @abc.abstractmethod
    def ramp_ds2(self, s: FloatArray) -> FloatArray:
        """
        Calculate second derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function second derivative.
        """

    @abc.abstractmethod
    def ramp_ds3(self, s: FloatArray) -> FloatArray:
        """
        Calculate third derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function third derivative.
        """

    def value_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        s: FloatArray = None,
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
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        if s is None:
            s = self.normalised_position(abscissa)

        return_array[:, ...] = self.y0 + self._ramp_height * self.ramp(
            np.clip(s, 0, 1)
        )

    def jacobian_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        s: FloatArray = None,
    ):
        """
        Evaluate model first derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        if s is None:
            s = self.normalised_position(abscissa)

        return_array.fill(0.0)
        _mask = np.logical_and(s >= 0, s <= 1)

        if np.any(_mask):
            _y = self.y0.shape
            _y_dim = self.y0.ndim
            _x = self.input_dimension.size

            f_ds = self._ramp_height * self.ramp_ds(s[_mask])
            s_dx = self.direction / self.ramp_width

            # Tricky broadcast as 3 different shapes being combined:
            #   First dimension is ~ of abscissas provided.
            #   Dimensions of the value y
            #   Dimensions of the direction
            mask = np.squeeze(_mask)
            return_array[mask, ..., :] = (
                # Broadcast across first and last dimension.
                np.reshape(f_ds, (1, *_y, 1))
                # Broadcast across all but last dimension.
                * np.reshape(s_dx, (1, *itertools.repeat(1, _y_dim), _x))
            )

    def hessian_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        s: FloatArray = None,
    ):
        """
        Evaluate model second derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        if s is None:
            s = self.normalised_position(abscissa)

        return_array.fill(0.0)
        _mask = np.logical_and(s >= 0, s <= 1)

        if np.any(_mask):
            f_ds2 = self._ramp_height * self.ramp_ds2(s[_mask])
            s_dx = self.direction / self.ramp_width

            _n = np.sum(_mask)
            _y = self.y0.shape
            _y_dim = self.y0.ndim
            _x = self.input_dimension.size

            # Reshape to make numpy broadcast the arrays correctly.
            mask = np.squeeze(_mask)
            return_array[mask, ..., :, :] = np.reshape(
                f_ds2, (_n, *_y, 1, 1)
            ) * np.reshape(
                np.outer(s_dx, s_dx), (1, *itertools.repeat(1, _y_dim), _x, _x)
            )

    def jerk_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        s: FloatArray = None,
    ):
        """
        Evaluate model third derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        if s is None:
            s = self.normalised_position(abscissa)

        return_array.fill(0.0)
        _mask = np.logical_and(s >= 0, s <= 1)

        if np.any(_mask):
            f_ds3 = self._ramp_height * self.ramp_ds3(s[_mask])
            s_dx = self.direction / self.ramp_width

            _n = np.sum(_mask)
            _y = self.y0.shape
            _y_dim = self.y0.ndim
            _x = self.input_dimension.size

            # Reshape to make numpy broadcast the arrays correctly.
            mask = np.squeeze(_mask)
            return_array[mask, ..., :, :] = np.reshape(
                f_ds3, (_n, *_y, 1, 1, 1)
            ) * np.reshape(
                np.einsum("i, j, k -> ijk", s_dx, s_dx, s_dx),
                (1, *itertools.repeat(1, _y_dim), _x, _x, _x),
            )

    def _fill_cache(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        derivative: int,
        value: FloatArray,
        jacobian: FloatArray,
        hessian: FloatArray,
        jerk: FloatArray,
        _snap: FloatArray,
    ):
        """
        Evaluate up to nu-th derivative of model value.

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
        _snap : np.array[float]
            Array to store fourth derivative result.

        Raises
        ------
        ValueError
            nu is < 0 or >= 4.
        """
        if derivative < 0 or derivative >= 4:  # noqa: PLR2004
            raise ValueError(derivative)

        s = self.normalised_position(abscissa)

        if derivative >= 0:
            self.value_function(abscissa, out_of_bounds, value, s=s)

        if derivative >= 1:
            self.jacobian_function(abscissa, out_of_bounds, jacobian, s=s)

        if derivative >= 2:  # noqa: PLR2004
            self.hessian_function(abscissa, out_of_bounds, hessian, s=s)

        if derivative >= 3:  # noqa: PLR2004
            self.jerk_function(abscissa, out_of_bounds, jerk, s=s)

    def write_netcdf(self, group: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        # Write common values for all value models.
        super().write_netcdf(group)

        # Write values specific to this model.
        write_netcdf_variable(
            group,
            "origin",
            (self.input_dimension,),
            self.origin,
            "Origin of step function",
            "",
        )

        write_netcdf_variable(
            group,
            "direction",
            (self.input_dimension,),
            self.direction,
            "Direction normal to transition in step function",
            "",
        )

        write_netcdf_variable(
            group,
            "y0",
            self.output_dimensions,
            self.y0,
            "Left hand value",
            self.units,
        )

        write_netcdf_variable(
            group,
            "y1",
            self.output_dimensions,
            self.y1,
            "Right hand value",
            self.units,
        )

        write_netcdf_variable(
            group,
            "metric",
            (self.input_dimension, self.input_dimension),
            self.metric,
            "Metric at origin",
            "",
        )

        group.setncattr("smoothness", self.smoothness)

        write_netcdf_variable(
            group,
            "ramp_width",
            (),
            self.ramp_width,
            "Width of ramp from y0 to y1",
            "m",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group):
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        cn_ramp : CnRamp
            Model loaded from netCDF4 data.
        """
        (
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            scale_factor,
            _,
        ) = ValueModelBase.read_shared_netcdf(group)

        origin = group["origin"][:].data
        direction = group["direction"][:].data
        y0 = group["y0"][...].data
        y1 = group["y1"][...].data
        metric = group["metric"][:, :].data
        ramp_width = group["ramp_width"][...].item()
        smoothness = group.getncattr("smoothness")

        obj = cls.with_smoothness(
            coordinate_system,
            input_dimension,
            output_dimensions,
            units,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            smoothness,
            scale_factor=scale_factor,
        )
        obj.set_metric(metric)

        return obj


class C0Ramp(CnRamp):
    """
    A model with a C0 continuous value ramp. This means the value is
    continuous at the ramp start and end but first order derivatives and
    higher are not.
    """

    __slots__ = ()

    smoothness = 0

    @staticmethod
    def ramp(s):
        """
        Calculate ramp function value. This equals 0 at s = 0 and 1 at s = 1
        with intermediate values determined by the smoothness class.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function value.
        """
        return s

    @staticmethod
    def ramp_ds(s):
        """
        Calculate first derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function first derivative.
        """
        return np.ones_like(s)

    @staticmethod
    def ramp_ds2(s):
        """
        Calculate second derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function second derivative.
        """
        return np.zeros_like(s)

    @staticmethod
    def ramp_ds3(s):
        """
        Calculate third derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function third derivative.
        """
        return np.zeros_like(s)

    @staticmethod
    def hessian_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        **_args,
    ):
        """
        Evaluate model second derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        return_array.fill(0.0)

    @staticmethod
    def jerk_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        **_args,
    ):
        """
        Evaluate model third derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        return_array.fill(0.0)


class C1Ramp(CnRamp):
    """
    A model with a C1 continuous value ramp. This means the value and first
    derivative are continuous at the ramp start and end but second order
    derivatives and higher are not.
    """

    __slots__ = ()

    smoothness = 1

    @staticmethod
    def ramp(s):
        """
        Calculate ramp function value. This equals 0 at s = 0 and 1 at s = 1
        with intermediate values determined by the smoothness class.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function value.
        """
        return (3 - 2 * s) * np.square(s)

    @staticmethod
    def ramp_ds(s):
        """
        Calculate first derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function first derivative.
        """
        return 6 * s * (1 - s)

    @staticmethod
    def ramp_ds2(s):
        """
        Calculate second derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function second derivative.
        """
        return 6 * (1 - 2 * s)

    @staticmethod
    def ramp_ds3(s):
        """
        Calculate third derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function third derivative.
        """
        return np.full(s.shape, -12.0)


class C2Ramp(C1Ramp):
    """
    A model with a C2 continuous value ramp. This means the value, first
    derivative and second derivative are continuous at the ramp start and end
    but third order derivatives and higher are not.
    """

    __slots__ = ()

    smoothness = 2

    @staticmethod
    def ramp(s):
        """
        Calculate ramp function value. This equals 0 at s = 0 and 1 at s = 1
        with intermediate values determined by the smoothness class.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function value.
        """
        s2 = np.square(s)
        s3 = s * s2
        return (10 - 15 * s + 6 * s2) * s3

    @staticmethod
    def ramp_ds(s):
        """
        Calculate first derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function first derivative.
        """
        return 30 * np.square(s) * np.square(1 - s)

    @staticmethod
    def ramp_ds2(s):
        """
        Calculate second derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function second derivative.
        """
        return 60 * s * (1 - s) * (1 - 2 * s)

    @staticmethod
    def ramp_ds3(s):
        """
        Calculate third derivative of ramp function.

        Parameters
        ----------
        s : np.array[float]
            Normalised position.

        Returns
        -------
        ramp : np.array[float]
            Ramp function third derivative.
        """
        return 60 * (6 * s * (s - 1) + 1)


class QuadraticChannel(ValueModelBase):
    """
    Cartesian quadratic channel model.

    Value is y(s) = y0 + (y1 - y0) * (s / L)**2 where s is the distance
    perpendicular from a line passing through a given origin in a given
    direction i.e. value is y0 along this line. L is the ramp width.

    Attributes
    ----------
    direction : np.array[float]
        Direction of value ramp.
    origin : np.array[float]
        Origin of value ramp.
    ramp_width : np.array[float]
        Distance over which value ramps from y0 to y1.
    y0 : np.array[float]
        Value at bottom of ramp i.e. at origin.
    y1 : np.array[float]
        Value at top of ramp.

    Methods
    -------
    write_netcdf
        Write to netCDF4 dataset.
    read_netcdf
        Load from netCDF4 dataset.
    """

    __slots__ = (
        "_ramp_height",
        "direction",
        "origin",
        "ramp_width",
        "y0",
        "y1",
    )

    model_type = ModelType.QUADRATIC_CHANNEL
    metric = np.eye(Dimensions.x.size)

    def __init__(
        self,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Inits QuadraticChannel.

        Parameters
        ----------
        input_dimension: Dimension
            Dimension for input value to model.
        output_dimensions: tuple[Dimension]
            Dimensions for output value from model.
        units: str
            Units of model.
        origin : np.array[float]
            Point at which value is y0.
        direction : np.array[float]
            Direction parallel to the value gradient.
        y0 : np.array[float]
            Minimum value of the function.
        y1 : np.array[float]
            Maximum value of the function.
        ramp_width : float
           Distance over which value varies from y0 and y1. Must be > 0.
        scale_factor : float, optional
            Scale factor for model.

        Raises
        ------
        ValueError
            y0 or y1 has incorrect shape for output dimensions.
            Origin or direction has incorrect shape for input dimension.
            Ramp width is negative.
        """
        self.y0 = np.asarray(y0)
        self.y1 = np.asarray(y1)

        super().__init__(
            CoordinateSystem.CARTESIAN,
            input_dimension,
            output_dimensions,
            units,
            scale_factor=scale_factor,
            dtype=self.y0.dtype,
        )

        if self.y0.shape != self.output_shape:
            raise ValueError(
                "y0 has incorrect shape. "
                f"Expected {self.output_shape}, "
                f"got {self.y0.shape}."
            )

        if self.y0.shape != self.y1.shape:
            raise ValueError(
                "y0 and y1 must have same shape: "
                f"{self.y0.shape}, {self.y1.shape}"
            )

        self.origin = np.array(origin)
        self.direction = np.array(direction)

        expected_shape = (input_dimension.size,)
        if self.origin.shape != expected_shape:
            raise ValueError(
                "origin has incorrect shape. "
                f"Expected {self.origin.shape}, got {expected_shape}."
            )

        if self.direction.shape != expected_shape:
            raise ValueError(
                "direction has incorrect shape. "
                f"Expected {self.direction.shape}, got {expected_shape}."
            )

        if self.direction.size > 1:
            norm = np.linalg.norm(self.direction)

            if np.isclose(norm, 0.0):
                raise ValueError("direction length is zero")

            self.direction /= norm

        self._ramp_height = self.y1 - self.y0
        self.ramp_width = float(ramp_width)

        if self.ramp_width <= 0.0:
            raise ValueError(f"ramp_width must be positive: {self.ramp_width}")

    def normalised_position2(
        self,
        abscissa: FloatArray,
    ):
        """
        Calculate squared normalised position along ramp s**2. At origin s = 0
        while at origin + direction s = 1.

        abscissa : np.array[float]
            Input value.

        Returns
        -------
        normalised_position : np.array[float]
            Normalised position along ramp.
        """
        return_array = np.empty(abscissa.shape[0])

        _dx = abscissa - self.origin

        return_array[:] = (
            np.sum(np.square(_dx), axis=1)
            - np.square(np.matmul(_dx, self.direction))
        ) / (self.ramp_width * self.ramp_width)

        # Reshape to allow multiplication by something of shape y0 and still
        # broadcast correctly.
        return return_array.reshape((
            abscissa.shape[0],
            *itertools.repeat(1, self.y0.ndim),
        ))

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
        s2 = self.normalised_position2(abscissa)

        return_array[:, ...] = self.y0 + self._ramp_height * s2

    def jacobian_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model first derivative.

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
        _n = abscissa.shape[0]
        _y = self.y0.shape
        _y_dim = self.y0.ndim
        _x = self.input_dimension.size

        # f_ds2 has shape (_n, *_y).
        f_ds2 = np.tile(self._ramp_height, (_n, *itertools.repeat(1, _y_dim)))

        # s2_dx has shape (_n, _x) with dimension 2.
        _dx = abscissa - self.origin
        s2_dx = (
            (
                _dx
                - np.matmul(_dx, self.direction).reshape((_n, 1))
                * np.reshape(self.direction, (1, _x))
            )
            * 2
            / (self.ramp_width * self.ramp_width)
        )

        # Reshape to make numpy broadcast the arrays correctly.
        return_array[:, ..., :] = np.reshape(f_ds2, (_n, *_y, 1)) * np.reshape(
            s2_dx, (_n, *itertools.repeat(1, _y_dim), _x)
        )

    def hessian_function(
        self,
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model second derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        _y = self.y0.shape
        _y_dim = self.y0.ndim
        _x = self.input_dimension.size

        # f_ds2 has shape _y
        f_ds2 = self._ramp_height

        # s2_dx2 has shape (_x, _x).
        s2_dx2 = (
            (np.identity(_x) - np.outer(self.direction, self.direction))
            * 2
            / (self.ramp_width * self.ramp_width)
        )

        # Reshape to make numpy broadcast the arrays correctly.
        # As value is independent of position broadcast over first dimension.
        return_array[:, ..., :, :] = np.reshape(
            f_ds2, (1, *_y, 1, 1)
        ) * np.reshape(s2_dx2, (1, *itertools.repeat(1, _y_dim), _x, _x))

    @staticmethod
    def jerk_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model third derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array.fill(0.0)

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

        # Write values specific to this model.
        write_netcdf_variable(
            group,
            "origin",
            (self.input_dimension,),
            self.origin,
            "Origin of step function",
            "",
        )

        write_netcdf_variable(
            group,
            "direction",
            (self.input_dimension,),
            self.direction,
            "Direction normal to transition in step function",
            "",
        )

        write_netcdf_variable(
            group,
            "y0",
            self.output_dimensions,
            self.y0,
            "Left hand value",
            self.units,
        )

        write_netcdf_variable(
            group,
            "y1",
            self.output_dimensions,
            self.y1,
            "Right hand value",
            self.units,
        )

        # Write values specific to this model.
        write_netcdf_variable(
            group,
            "ramp_width",
            (),
            self.ramp_width,
            "Width of ramp from y0 to y1",
            "m",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group):
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        quadratic_channel : QuadraticChannel
            Model loaded from netCDF4 data.
        """
        (_, input_dimension, output_dimensions, units, scale_factor, _) = (
            ValueModelBase.read_shared_netcdf(group)
        )

        origin = group["origin"][:].data
        direction = group["direction"][:].data
        y0 = group["y0"][...].data
        y1 = group["y1"][...].data
        ramp_width = group["ramp_width"][...].item()

        return cls(
            input_dimension,
            output_dimensions,
            units,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            scale_factor=scale_factor,
        )


class QuadraticBowl(QuadraticChannel):
    """
    Cartesian quadratic bowl model.

    Value y(s) = y0 + (y1 - y0) * (s / L)**2 where s is the distance from a
    given origin parallel to a given direction. L is the ramp width.

    Notes
    -----
    This differs from QuadraticChannel as the function only varies in the given
    direction compared to varying only NOT in the given direction.
    """

    __slots__ = ()

    model_type = ModelType.QUADRATIC_BOWL

    normalised_position = CnRamp.normalised_position

    def value_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        s: FloatArray = None,
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
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        if s is None:
            s = self.normalised_position(abscissa)

        return_array[:, ...] = self.y0 + self._ramp_height * np.square(s)

    def jacobian_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        s: FloatArray = None,
    ):
        """
        Evaluate model first derivative.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        s : np.array[float], optional
            Normalised position. If not provided it is calculated.
        """
        if s is None:
            s = self.normalised_position(abscissa)

        _n = abscissa.shape[0]
        _y = self.y0.shape
        _y_dim = self.y0.ndim
        _x = self.input_dimension.size

        f_ds = self._ramp_height * 2.0 * s
        s_dx = self.direction / self.ramp_width

        # Tricky broadcast as 3 different shapes being combined:
        #   First dimension is ~ of abscissas provided.
        #   Dimensions of the value y
        #   Dimensions of the direction
        return_array[:, ..., :] = (
            # Broadcast across first and last dimension.
            np.reshape(f_ds, (_n, *_y, 1))
            # Broadcast across all but last dimension.
            * np.reshape(s_dx, (_n, *itertools.repeat(1, _y_dim), _x))
        )

    def hessian_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model second derivative.

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
        f_ds2 = self._ramp_height * 2.0
        s_dx = self.direction / self.ramp_width

        _n = abscissa.shape[0]
        _y = self.y0.shape
        _y_dim = self.y0.ndim
        _x = self.input_dimension.size

        # Reshape to make numpy broadcast the arrays correctly.
        return_array[:, ..., :, :] = np.reshape(
            f_ds2, (_n, *_y, 1, 1)
        ) * np.reshape(
            np.outer(s_dx, s_dx), (_n, *itertools.repeat(1, _y_dim), _x, _x)
        )

    @staticmethod
    def jerk_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model third derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array.fill(0.0)

    def _fill_cache(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        derivative: int,
        value: FloatArray,
        jacobian: FloatArray,
        hessian: FloatArray,
        jerk: FloatArray,
        _snap: FloatArray,
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
        _snap : np.array[float]
            Array to store fourth derivative result.

        Raises
        ------
        ValueError
            derivatives is < 0 or >= 4.
        """
        if derivative >= 4:  # noqa: PLR2004
            raise ValueError(derivative)

        s = self.normalised_position(abscissa)

        if derivative >= 0:
            self.value_function(abscissa, out_of_bounds, value, s=s)

        if derivative >= 1:
            self.jacobian_function(abscissa, out_of_bounds, jacobian, s=s)

        if derivative >= 2:  # noqa: PLR2004
            self.hessian_function(abscissa, out_of_bounds, hessian)

        if derivative >= 3:  # noqa: PLR2004
            self.jerk_function(abscissa, out_of_bounds, jerk)


class QuadraticWell(ValueModelBase):
    """
    Cartesian quadratic well model.

    Value is y(s) = y0 + (y1 - y0) * (s / L)**2 where s is the distance from a
    given origin. L is the ramp width.

    Attributes
    ----------
    origin : np.array[float]
        Origin of value ramp.
    ramp_width : np.array[float]
        Distance over which value ramps from y0 to y1.
    y0 : np.array[float]
        Value at bottom of ramp i.e. at origin.
    y1 : np.array[float]
        Value at top of ramp.
    """

    __slots__ = ("_ramp_height", "origin", "ramp_width", "y0", "y1")

    model_type = ModelType.QUADRATIC_WELL

    def __init__(
        self,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
        origin: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Inits QuadraticWell.

        Parameters
        ----------
        input_dimension: Dimension
            Dimension for input value to model.
        output_dimensions: tuple[Dimension]
            Dimensions for output value from model.
        units: str
            Units of model.
        origin : np.array[float]
            Point at which value is y0.
        y0 : np.array[float]
            Minimum value of the function.
        y1 : np.array[float]
            Maximum value of the function.
        ramp_width : float
           Distance over which value varies from y0 and y1. Must be > 0.
        scale_factor : float, optional
            Scale factor for model.

        Raises
        ------
        ValueError
            y0 or y1 has incorrect shape for output dimensions.
        """
        self.y0 = np.asarray(y0)
        self.y1 = np.asarray(y1)

        super().__init__(
            CoordinateSystem.CARTESIAN,
            input_dimension,
            output_dimensions,
            units,
            scale_factor=scale_factor,
            dtype=self.y0.dtype,
        )

        if self.y0.shape != self.y1.shape:
            raise ValueError(
                "y0 and y1 must have same shape: "
                f"{self.y0.shape}, {self.y1.shape}"
            )

        if self.y0.shape != self.output_shape:
            raise ValueError(
                "y0 has incorrect shape. "
                f"Expected {self.y0.shape}, got {self.output_shape}."
            )

        self._ramp_height = self.y1 - self.y0
        self.ramp_width = float(ramp_width)
        self.origin = np.array(origin)

    def normalised_position(self, abscissa: FloatArray) -> FloatArray:
        """
        Calculate normalised position along ramp s. At origin s = 0 while at
        origin + direction s = 1.

        abscissa : np.array[float]
            Input value.

        Returns
        -------
        normalised_position : np.array[float]
            Normalised position along ramp.
        """
        return_array = np.empty(abscissa.shape[0])

        _dx = abscissa - self.origin

        return_array[:] = np.sum(np.square(_dx), axis=1) / (
            self.ramp_width * self.ramp_width
        )

        # Reshape to allow multiplication by something of shape y0 and still
        # broadcast correctly.
        return return_array.reshape((
            abscissa.shape[0],
            *itertools.repeat(1, self.y0.ndim),
        ))

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
        s2 = self.normalised_position(abscissa)
        return_array[...] = self.y0 + s2 * self._ramp_height

    def jacobian_function(
        self,
        abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model first derivative.

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
        _n = abscissa.shape[0]
        _y = self.y0.shape
        _y_dim = self.y0.ndim
        _x = self.input_dimension.size

        # f_ds2 has shape (_n, *_y).
        f_ds2 = np.tile(self._ramp_height, (_n, *itertools.repeat(1, _y_dim)))

        # s2_dx has shape (_n, _x) with dimension 2.
        _dx = abscissa - self.origin
        s2_dx = _dx * 2 / (self.ramp_width * self.ramp_width)

        # Reshape to make numpy broadcast the arrays correctly.
        return_array[:, ..., :] = np.reshape(f_ds2, (_n, *_y, 1)) * np.reshape(
            s2_dx, (_n, *itertools.repeat(1, _y_dim), _x)
        )

    def hessian_function(
        self,
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model second derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        _y = self.y0.shape
        _y_dim = self.y0.ndim
        _x = self.input_dimension.size

        # f_ds2 has shape _y
        f_ds2 = self._ramp_height

        # s2_dx2 has shape (_x, _x).
        s2_dx2 = np.identity(_x) * 2 / (self.ramp_width * self.ramp_width)

        # Reshape to make numpy broadcast the arrays correctly.
        # As value is independent of position broadcast over first dimension.
        return_array[:, ..., :, :] = np.reshape(
            f_ds2, (1, *_y, 1, 1)
        ) * np.reshape(s2_dx2, (1, *itertools.repeat(1, _y_dim), _x, _x))

    @staticmethod
    def jerk_function(
        _abscissa: FloatArray,
        _out_of_bounds: BooleanArray,
        return_array: FloatArray,
    ):
        """
        Evaluate model third derivative.

        Parameters
        ----------
        _abscissa : np.array[float]
            Input values to evaluate model at.
        _out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        return_array : np.array[float]
            Array into which the result is stored.
        """
        return_array.fill(0.0)

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

        # Write values specific to this model.
        write_netcdf_variable(
            group,
            "origin",
            (self.input_dimension,),
            self.origin,
            "Origin of step function",
            "",
        )

        write_netcdf_variable(
            group,
            "y0",
            self.output_dimensions,
            self.y0,
            "Left hand value",
            self.units,
        )

        write_netcdf_variable(
            group,
            "y1",
            self.output_dimensions,
            self.y1,
            "Right hand value",
            self.units,
        )

        write_netcdf_variable(
            group,
            "ramp_width",
            (),
            self.ramp_width,
            "Width of ramp from y0 to y1",
            "m",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group):
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        quadratic_well : QuadraticWell
            Model loaded from netCDF4 data.
        """
        (_, input_dimension, output_dimensions, units, scale_factor, _) = (
            ValueModelBase.read_shared_netcdf(group)
        )

        origin = group["origin"][:].data
        y0 = group["y0"][...].data
        y1 = group["y1"][...].data
        ramp_width = group["ramp_width"][...].item()

        return cls(
            input_dimension,
            output_dimensions,
            units,
            origin,
            y0,
            y1,
            ramp_width,
            scale_factor=scale_factor,
        )
