"""
Convenience classes for constructing plasma parameter models.
"""

# Standard imports
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.shared.dimensions import Dimension, Dimensions
from crayon.shared.types import BooleanArray, FloatArray
from crayon.value_model.analytic import (
    CnRamp,
    Constant,
    QuadraticBowl,
    QuadraticChannel,
    QuadraticWell,
)
from crayon.value_model.base import ModelType, ValueModelBase
from crayon.value_model.magnetic_field import AxisymmetricMagneticField
from crayon.value_model.splines import (
    Spline1D,
    Spline2D,
    Spline3D,
    SplineBase,
    SplineMethod,
)

logger = logging.getLogger(__name__)


class ValueModel:
    """
    Plasma parameter model.

    Attributes
    ----------
    input_dimension : Dimension
        Dimension of input value.
    name : str
        Name of value.
    output_dimensions : tuple[Dimension]
        Dimensions of output value.
    units : str
        Units of value.
    """

    __slots__ = ("input_dimension", "name", "output_dimensions", "units")

    def __init__(
        self,
        name: str,
        input_dimension: Dimension,
        output_dimensions: tuple[Dimension],
        units: str,
    ):
        """
        Inits ValueModel.

        Parameters
        ----------
        name : str
            Name of value.
        input_dimension : Dimension
            Dimension of input value.
        output_dimensions : tuple[Dimension]
            Dimensions of output value.
        units : str
            Units of value.
        """
        self.name = name
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.units = units

    def spline_1d(
        self,
        coordinate_system: CoordinateSystem,
        abscissa_1: FloatArray,
        data: FloatArray,
        dependent_components: BooleanArray,
        /,
        *,
        scale_factor: float = 1.0,
        method: SplineMethod = SplineMethod.CUBIC,
    ) -> Spline1D:
        """
        Create 1D spline fit.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system model is defined in.
        abscissa_1 : np.array[float]
            Abscissa in first dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        scale_factor : float, optional
            Scale factor for model.
        method : SplineMethod, optional
            Spline order.

        Returns
        -------
        spline_1d : Spline1D
            1D spline fit.
        """
        return Spline1D(
            coordinate_system,
            self.input_dimension,
            self.output_dimensions,
            self.units,
            abscissa_1,
            data,
            dependent_components,
            scale_factor=scale_factor,
            method=method,
        )

    def spline_2d(
        self,
        coordinate_system: CoordinateSystem,
        abscissa_1: FloatArray,
        abscissa_2: FloatArray,
        data: FloatArray,
        dependent_components: BooleanArray,
        /,
        *,
        scale_factor: float = 1.0,
        method: SplineMethod = SplineMethod.CUBIC,
    ) -> Spline2D:
        """
        Create 2D spline fit.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system model is defined in.
        abscissa_1 : np.array[float]
            Abscissa in first dimension.
        abscissa_2 : np.array[float]
            Abscissa in second dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        scale_factor : float, optional
            Scale factor for model.
        method : SplineMethod, optional
            Spline order.

        Returns
        -------
        spline_2d : Spline2D
            2D spline fit.
        """
        return Spline2D(
            coordinate_system,
            self.input_dimension,
            self.output_dimensions,
            self.units,
            abscissa_1,
            abscissa_2,
            data,
            dependent_components,
            scale_factor=scale_factor,
            method=method,
        )

    def spline_3d(
        self,
        coordinate_system: CoordinateSystem,
        abscissa_1: FloatArray,
        abscissa_2: FloatArray,
        abscissa_3: FloatArray,
        data: FloatArray,
        dependent_components: BooleanArray,
        /,
        *,
        scale_factor: float = 1.0,
        method: SplineMethod = SplineMethod.CUBIC,
    ) -> Spline3D:
        """
        Create 3D spline fit.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system model is defined in.
        abscissa_1 : np.array[float]
            Abscissa in first dimension.
        abscissa_2 : np.array[float]
            Abscissa in second dimension.
        abscissa_3 : np.array[float]
            Abscissa in third dimension.
        data : np.array
            Data to fit spline to.
        dependent_components : tuple[bool]
            Flag if value depends on input coordinate component.
        scale_factor : float, optional
            Scale factor for model.
        method : SplineMethod, optional
            Spline order.

        Returns
        -------
        spline_3d : Spline3D
            3D spline fit.
        """
        return Spline3D(
            coordinate_system,
            self.input_dimension,
            self.output_dimensions,
            self.units,
            abscissa_1,
            abscissa_2,
            abscissa_3,
            data,
            dependent_components,
            scale_factor=scale_factor,
            method=method,
        )

    def constant(
        self,
        coordinate_system: CoordinateSystem,
        constant_value: FloatArray,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Create Constant model.

        Parameters
        ----------
        coordinate_system: CoordinateSystem
            Coordinate system model is defined for.
        constant_value: np.array[float]
            Constant value of model.
        scale_factor: float, optional
            Scale factor for model.

        Returns
        -------
        constant : Constant
            Constant model.
        """
        return Constant(
            coordinate_system,
            self.input_dimension,
            self.output_dimensions,
            self.units,
            constant_value,
            scale_factor=scale_factor,
        )

    def ramp(
        self,
        coordinate_system: CoordinateSystem,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        smoothness: int,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        A model with a value ramp of given smoothness class.

        Parameters
        ----------
        coordinate_system: CoordinateSystem
            Coordinate system model is defined for.
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
        scale_factor: float, optional
            Scale factor for model.

        Returns
        -------
        ramp : CnRamp
            Ramp model.
        """
        return CnRamp.with_smoothness(
            coordinate_system,
            self.input_dimension,
            self.output_dimensions,
            self.units,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            smoothness,
            scale_factor=scale_factor,
        )

    def quadratic_channel(
        self,
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
        Cartesian quadratic channel model.

        Parameters
        ----------
        origin : np.array[float]
            Origin of value ramp.
        direction : np.array[float]
            Direction of value ramp.
        y0 : np.array[float]
            Value at bottom of ramp i.e. at origin.
        y1 : np.array[float]
            Value at top of ramp.
        ramp_width : np.array[float]
            Distance over which value ramps from y0 to y1.
        scale_factor : float, optional
            Scale factor for model.

        Returns
        -------
        quadratic_channel : QuadraticChannel
            Quadratic channel model.
        """
        return QuadraticChannel(
            self.input_dimension,
            self.output_dimensions,
            self.units,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            scale_factor=scale_factor,
        )

    def quadratic_bowl(
        self,
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
        Cartesian quadratic bowl model.

        Parameters
        ----------
        origin : np.array[float]
            Origin of value ramp.
        direction : np.array[float]
            Direction of value ramp.
        y0 : np.array[float]
            Value at bottom of ramp i.e. at origin.
        y1 : np.array[float]
            Value at top of ramp.
        ramp_width : np.array[float]
            Distance over which value ramps from y0 to y1.
        scale_factor : float, optional
            Scale factor for model.

        Returns
        -------
        quadratic_bowl : QuadraticBowl
            Quadratic bowl model.
        """
        return QuadraticBowl(
            self.input_dimension,
            self.output_dimensions,
            self.units,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            scale_factor=scale_factor,
        )

    def quadratic_well(
        self,
        origin: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Cartesian quadratic well model.

        Parameters
        ----------
        origin : np.array[float]
            Origin of value ramp.
        y0 : np.array[float]
            Value at bottom of ramp i.e. at origin.
        y1 : np.array[float]
            Value at top of ramp.
        ramp_width : np.array[float]
            Distance over which value ramps from y0 to y1.
        scale_factor : float, optional
            Scale factor for model.

        Returns
        -------
        quadratic_well : QuadraticWell
            Quadratic well model.
        """
        return QuadraticWell(
            self.input_dimension,
            self.output_dimensions,
            self.units,
            origin,
            y0,
            y1,
            ramp_width,
            scale_factor=scale_factor,
        )

    @staticmethod
    def read_netcdf(group: nc4.Group) -> ValueModelBase:
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        model : ValueModelBase
            Model loaded from netCDF4 data.
        """
        model_type = ModelType.parse(group.getncattr("model_type"))

        if model_type == ModelType.CONSTANT:
            return Constant.read_netcdf(group)
        if model_type == ModelType.SPLINE:
            return SplineBase.read_netcdf(group)
        if model_type == ModelType.RAMP:
            return CnRamp.read_netcdf(group)
        if model_type == ModelType.QUADRATIC_CHANNEL:
            return QuadraticChannel.read_netcdf(group)
        if model_type == ModelType.QUADRATIC_BOWL:
            return QuadraticBowl.read_netcdf(group)
        if model_type == ModelType.QUADRATIC_WELL:
            return QuadraticWell.read_netcdf(group)
        raise NotImplementedError(model_type.name)

    @classmethod
    def electron_density_per_m3(cls) -> "ValueModel":
        """
        Construct value model metadata for electron density [m^-3].

        Returns
        -------
        electron_density : ValueModel
            Electron density model metadata.
        """
        return cls("electron_density", Dimensions.x, (), "m^-3")

    @classmethod
    def electron_temperature_ev(cls) -> "ValueModel":
        """
        Construct value model metadata for electron temperature [eV].

        Returns
        -------
        electron_temperature : ValueModel
            Electron temperature model metadata.
        """
        return cls("electron_temperature", Dimensions.x, (), "eV")

    @classmethod
    def effective_charge(cls):
        """
        Construct value model metadata for effective charge [].

        Returns
        -------
        effective_charge : ValueModel
            Effective charge model metadata.
        """
        return cls("effective_charge", Dimensions.x, (), "")

    @classmethod
    def magnetic_field_t(cls):
        """
        Construct value model metadata for magnetic field [T].

        Returns
        -------
        magnetic_field : ValueModel
            Magnetic field model metadata.
        """
        return MagneticField(
            "magnetic_field", Dimensions.x, (Dimensions.x,), "T"
        )

    @classmethod
    def root_normalised_poloidal_flux_1d(cls):
        """
        Construct value model metadata for root normalised poloidal flux as
        a 1D function of another scalar value.

        Returns
        -------
        rho_poloidal : ValueModel
            Root normalised poloidal flux model metadata.
        """
        return cls("root_normalised_poloidal_flux", Dimensions.one, (), "")

    @classmethod
    def root_normalised_poloidal_flux(cls):
        """
        Construct value model metadata for root normalised poloidal flux as
        a 1D function of position.

        Returns
        -------
        rho_poloidal : ValueModel
            Root normalised poloidal flux model metadata.
        """
        return cls("root_normalised_poloidal_flux", Dimensions.x, (), "")

    @classmethod
    def root_normalised_toroidal_flux_1d(cls):
        """
        Construct value model metadata for root normalised toroidal flux as
        a 1D function of another scalar value.

        Returns
        -------
        rho_toroidal : ValueModel
            Root normalised toroidal flux model metadata.
        """
        return cls("root_normalised_toroidal_flux", Dimensions.one, (), "")

    @classmethod
    def root_normalised_toroidal_flux(cls):
        """
        Construct value model metadata for root normalised toroidal flux as
        a 1D function of position.

        Returns
        -------
        rho_toroidal : ValueModel
            Root normalised toroidal flux model metadata.
        """
        return cls("root_normalised_toroidal_flux", Dimensions.x, (), "")

    @classmethod
    def f_toroidal_tm(cls):
        """
        Construct value model metadata for diamagnetic function F [T.m].

        Returns
        -------
        f_toroidal : ValueModel
            Diamagnetic function model metadata.
        """
        return cls("toroidal_function", Dimensions.one, (), "T.m")

    @classmethod
    def cross_sectional_area_m2(cls):
        """
        Construct value model metadata for flux surface cross sectional
        area [m^-2].

        Returns
        -------
        cross_sectional_area : ValueModel
            Cross sectional area model metadata.
        """
        return cls("cross_sectional_area", Dimensions.one, (), "m^2")

    @classmethod
    def volume_m3(cls):
        """
        Construct value model metadata for flux tube volume [m^-3].

        Returns
        -------
        volume : ValueModel
            Volume model metadata.
        """
        return cls("volume", Dimensions.one, (), "m^3")

    @classmethod
    def fsa_1_over_r(cls):
        """
        Construct value model metadata for flux surface averaged 1 / r [m^-1].

        Returns
        -------
        fsa_1_over_r : ValueModel
            Flux surface averaged 1 / r model metadata.
        """
        return cls("fsa_1_over_r", Dimensions.one, (), "m^-1")

    @classmethod
    def fsa_1_over_r2(cls):
        """
        Construct value model metadata for flux surface averaged 1 / r**2
        [m^-2].

        Returns
        -------
        fsa_1_over_r2 : ValueModel
            Flux surface averaged 1 / r**2 model metadata.
        """
        return cls("fsa_1_over_r2", Dimensions.one, (), "m^-2")

    @classmethod
    def fsa_b(cls):
        """
        Construct value model metadata for flux surface averaged magnetic
        field strength [T].

        Returns
        -------
        fsa_b : ValueModel
            Flux surface averaged magnetic field strength model metadata.
        """
        return cls("fsa_b", Dimensions.one, (), "T")

    @classmethod
    def b_max(cls):
        """
        Construct value model metadata for maximum magnetic field strength on
        flux surface [T].

        Returns
        -------
        b_max : ValueModel
            Maximum magnetic field strength model metadata.
        """
        return cls("max_b", Dimensions.one, (), "T")

    @classmethod
    def trapped_particle_fraction(cls):
        """
        Construct value model metadata for trapped particle fraction on flux
        surface.

        Returns
        -------
        trapped_particle_fraction : ValueModel
            Trapped particle fraction model metadata.
        """
        return cls("trapped_particle_fraction", Dimensions.one, (), "")


class MagneticField(ValueModel):
    """
    Magnetic field models.
    """

    __slots__ = ()

    @staticmethod
    def axisymmetric(
        rho_poloidal_model: Spline2D,
        f_toroidal_model: Spline1D,
        total_poloidal_flux_wb: float,
        /,
        *,
        scale_factor: float = 1.0,
    ) -> AxisymmetricMagneticField:
        """
        Cylindrical axisymmetric magnetic field.

        Parameters
        ----------
        rho_poloidal_model : Spline2D
            Spline of root normalised poloidal flux rho (r, z) as function of
            cylindrical position.
        f_toroidal_model : ValueModelBase
            Model for diamagnetic function F.
        total_poloidal_flux_wb : float
            Total poloidal flux between magnetic axis and separatrix. This is
            total flux i.e. NOT flux / 2 pi.
        scale_factor : float, optional
            Scale factor for model.

        Returns
        -------
        axisymmetric_magnetic_field : AxisymmetricMagneticField
            Cylindrical axisymmetric magnetic field.
        """
        return AxisymmetricMagneticField(
            rho_poloidal_model,
            f_toroidal_model,
            total_poloidal_flux_wb,
            scale_factor=scale_factor,
        )
