"""
Objects for constructing initial conditions of ray trajectories.
"""

# Standard imports
import abc
import logging
import typing

# Third party imports
import cerberus
import netCDF4 as nc4  # noqa: N813
import numpy as np
import toml

# Local imports
from crayon.calculus import TensorType, rotation_a_onto_b
from crayon.ray_tracing.caches import CoordinateCache, PlasmaCache
from crayon.shared.constants import (
    C_M_PER_NS,
    DEG_TO_RAD,
    AngleFormat,
    CoordinateSystem,
    WaveMode,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.io import (
    IONetcdf,
    IOToml,
    TomlValidator,
    write_netcdf_variable,
)
from crayon.shared.types import ComplexArray, ComplexType, FloatArray

logger = logging.getLogger(__name__)


class InitialConditions(IONetcdf):
    """
    Initial conditions of ray trajectory.

    Attributes
    ----------
    adiabatic_phase_rad : float
        Adiabatic contribution to wave phase.
    beam_waist_radius_m
        1 / e electric field waist radius of beam.
    bundle : bool
        Flag if the ray is part of a bundle of rays.
    eikonal_phase_rad : float
        Eikonal contribution to wave phase.
    frequency_ghz : float
        Frequency [GHz].
    intensity_w_per_m2 : float
        Intensity [W.m^-2].
    name : str
        Name of ray.
    polarisation_stix : np.array[complex]
        Complex polarisation in Stix frame.
    position_cartesian : np.array[float]
        Cartesian position [m].
    power_w : float
        Power [W].
    reflections : int
        Number of reflections accrued.
    refractive_index_cartesian : np.array[float]
        Cartesian refractive index vector.
    time_ns : float
        Time [ns].
    wave_mode : WaveMode
        Mode of wave.

    Methods
    -------
    clone
        Create copy of object.
    write_netcdf
        Write contents to netCDF format.
    read_netcdf
        Construct from netCDF format.
    """

    __slots__ = (
        "adiabatic_phase_rad",
        "beam_waist_radius_m",
        "bundle",
        "eikonal_phase_rad",
        "frequency_ghz",
        "intensity_w_per_m2",
        "name",
        "polarisation_stix",
        "position_cartesian",
        "power_w",
        "reflections",
        "refractive_index_cartesian",
        "time_ns",
        "wave_mode",
    )

    def __init__(
        self,
        name: str,
        time_ns: float,
        frequency_ghz: float,
        position_cartesian: FloatArray,
        refractive_index_cartesian: FloatArray,
        eikonal_phase_rad: float,
        adiabatic_phase_rad: float,
        polarisation_stix: ComplexArray,
        wave_mode: WaveMode,
        power_w: float,
        intensity_w_per_m2: float,
        beam_waist_radius_m: float,
        /,
        *,
        bundle: bool,
    ):
        """
        Inits InitialConditions

        Parameters
        ----------
        name : str
            Name of ray.
        time_ns : float
            Time [ns].
        frequency_ghz : float
            Frequency [GHz].
        position_cartesian : np.array[float]
            Cartesian position [m].
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.
        eikonal_phase_rad : float
            Eikonal contribution to wave phase.
        adiabatic_phase_rad : float
            Adiabatic contribution to wave phase.
        polarisation_stix : np.array[complex]
            Complex polarisation in Stix frame.
        wave_mode : WaveMode
            Mode of wave.
        power_w : float
            Power [W].
        intensity_w_per_m2 : float
            Intensity [W.m^-2].
        """
        self.name = name
        self.time_ns = time_ns
        self.frequency_ghz = frequency_ghz
        self.position_cartesian = np.asarray(position_cartesian).reshape((
            Dimensions.x.size,
        ))
        self.refractive_index_cartesian = np.asarray(
            refractive_index_cartesian
        ).reshape((Dimensions.x.size,))
        self.eikonal_phase_rad = eikonal_phase_rad
        self.adiabatic_phase_rad = adiabatic_phase_rad
        self.polarisation_stix = np.asarray(
            polarisation_stix, dtype=complex
        ).reshape((Dimensions.x.size,))
        self.wave_mode = wave_mode
        self.power_w = power_w
        self.intensity_w_per_m2 = intensity_w_per_m2
        self.beam_waist_radius_m = float(beam_waist_radius_m)
        self.bundle = bundle

        self.reflections = 0

    def clone(self) -> "InitialConditions":
        """
        Clone initial conditions.

        Returns
        -------
        clone : InitialConditions
            A independent copy of the initial conditions.
        """
        return InitialConditions(
            self.name,
            self.time_ns,
            self.frequency_ghz,
            self.position_cartesian.copy(),
            self.refractive_index_cartesian.copy(),
            self.eikonal_phase_rad,
            self.adiabatic_phase_rad,
            self.polarisation_stix.copy(),
            self.wave_mode,
            self.power_w,
            self.intensity_w_per_m2,
            self.beam_waist_radius_m,
            self.bundle,
        )

    def write_netcdf(self, group: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 group to write data to.
        """
        group.setncattr("time_ns", self.time_ns)
        group.setncattr("frequency_ghz", self.frequency_ghz)
        group.setncattr("eikonal_phase_rad", self.eikonal_phase_rad)
        group.setncattr("adiabatic_phase_rad", self.adiabatic_phase_rad)
        group.setncattr("wave_mode", self.wave_mode.name)
        group.setncattr("power_w", self.power_w)
        group.setncattr("intensity_w_per_m2", self.intensity_w_per_m2)
        group.setncattr("beam_waist_radius_m", self.beam_waist_radius_m)
        group.setncattr("bundle", int(self.bundle))

        write_netcdf_variable(
            group,
            "position_cartesian",
            (Dimensions.x,),
            self.position_cartesian,
            "Initial Cartesian position of ray",
            "m",
        )

        write_netcdf_variable(
            group,
            "refractive_index_cartesian",
            (Dimensions.k,),
            self.refractive_index_cartesian,
            "Initial Cartesian refractive index vector of ray",
            "m",
        )

        write_netcdf_variable(
            group,
            "polarisation_stix",
            (Dimensions.x,),
            self.polarisation_stix,
            (
                "Cartesian complex polarisation of electric field in Stix "
                "basis (Nperp // x, B // z)"
            ),
            "m",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "InitialConditions":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        initial_conditions : InitialConditions
            Ray initial conditions object.
        """
        name = group.name
        time_ns = group.getncattr("time_ns")
        frequency_ghz = group.getncattr("frequency_ghz")
        position_cartesian = group["position_cartesian"][:].data
        refractive_index_cartesian = group["refractive_index_cartesian"][
            :
        ].data
        eikonal_phase_rad = group.getncattr("eikonal_phase_rad")
        adiabatic_phase_rad = group.getncattr("adiabatic_phase_rad")
        polarisation_stix = group["polarisation_stix"][:].data
        wave_mode = WaveMode.parse(group.getncattr("wave_mode"))
        power_w = group.getncattr("power_w")
        intensity_w_per_m2 = group.getncattr("intensity_w_per_m2")
        beam_waist_radius_m = group.getncattr("beam_waist_radius_m")
        bundle = group.getncattr("bundle") != 0

        return cls(
            name,
            time_ns,
            frequency_ghz,
            position_cartesian,
            refractive_index_cartesian,
            eikonal_phase_rad,
            adiabatic_phase_rad,
            polarisation_stix,
            wave_mode,
            power_w,
            intensity_w_per_m2,
            beam_waist_radius_m,
            bundle=bundle,
        )


class InitialConditionsTomlValidator(TomlValidator):
    """
    Custom cerberus validator for enums used to define ray initial conditions.
    """

    types_mapping = TomlValidator.types_mapping.copy()

    types_mapping["CoordinateSystem"] = cerberus.TypeDefinition(
        "CoordinateSystem", (CoordinateSystem,), ()
    )
    types_mapping["AngleFormat"] = cerberus.TypeDefinition(
        "AngleFormat", (AngleFormat,), ()
    )


def coerce_float_ndarray(x) -> FloatArray:
    """
    Coerce object to numpy array with float datatype.

    Parameters
    ----------
    x : any
        Object to be coerced.

    Returns
    -------
    float_array
        x as a numpy array with float datatype.
    """
    return np.asarray(x, dtype=float)


class InitialRefractiveIndexBase(IOToml):
    """
    Base class for initial refractive index definitions.

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate Cartesian refractive index vector.

    """

    __slots__ = ()

    @abc.abstractmethod
    def clone(self) -> "InitialRefractiveIndexBase":
        """
        Create copy of self.

        Returns
        -------
        clone : InitialRefractiveIndexBase
            Initial refractve index data.
        """

    @abc.abstractmethod
    def unpack(self) -> FloatArray:
        """
        Generate Cartesian refractive index vector.

        Returns
        -------
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.
        """


class RefractiveIndexComponents(InitialRefractiveIndexBase):
    """
    Initial refractive index provided as vector components. The magnitude of
    the refractive index is calculated by root finding the dispersion relation.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        Coordinate system components are provided in.
    holonomic : bool
        If components are given in holonomic basis.
    refractive_index : np.array[float]
        Refractive index vector components.

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate Cartesian refractive index vector.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ("coordinate_system", "holonomic", "refractive_index")

    schema: typing.ClassVar[dict] = {
        "refractive_index": {
            "required": True,
            "type": "ndarray",
            "coerce": coerce_float_ndarray,
        },
        "coordinate_system_refractive_index": {
            "required": True,
            "type": "CoordinateSystem",
            "coerce": CoordinateSystem.parse,
        },
        "holonomic": {
            "required": True,
            "type": "boolean",
        },
    }

    def __init__(
        self,
        refractive_index: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        holonomic: bool,
    ):
        """
        Inits RefractiveIndexComponents.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system components are provided in.
        holonomic : bool
            If components are given in holonomic basis.
        refractive_index : np.array[float]
            Refractive index vector components.
        """
        self.refractive_index = np.asarray(refractive_index).reshape((
            Dimensions.x.size,
        ))
        self.coordinate_system = coordinate_system
        self.holonomic = bool(holonomic)

    def clone(self) -> "RefractiveIndexComponents":
        """
        Create copy of self.

        Returns
        -------
        clone : RefractiveIndexComponents
            Independent copy of self.
        """
        return RefractiveIndexComponents(
            self.refractive_index.copy(),
            self.coordinate_system,
            self.holonomic,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "refractive_index": self.refractive_index,
            "coordinate_system_refractive_index": self.coordinate_system.name,
            "holonomic": self.holonomic,
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "RefractiveIndexComponents":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        refractive_index_definition : RefractiveIndexComponents
            Initial refractive index using vector components.
        """
        validator = InitialConditionsTomlValidator()
        validator.validate(d, cls.schema, allow_unknown=True)

        return cls(
            validator.document["refractive_index"],
            validator.document["coordinate_system_refractive_index"],
            holonomic=validator.document["holonomic"],
        )

    def unpack(self, coordinate_cache: CoordinateCache) -> FloatArray:
        """
        Generate Cartesian refractive index vector.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.

        Returns
        -------
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.
        """
        if (
            self.coordinate_system == CoordinateSystem.CARTESIAN
            or self.holonomic
        ):
            components_holonomic = self.refractive_index
        else:
            # Convert to holonomic basis.
            components_holonomic = coordinate_cache.transform_basis(
                self.coordinate_system,
                self.refractive_index,
                TensorType.COVECTOR,
                to_holonomic=True,
            )

        # Convert refractive index to Cartesian.
        return coordinate_cache.transform_tensor_field(
            self.coordinate_system,
            CoordinateSystem.CARTESIAN,
            components_holonomic,
            TensorType.COVECTOR,
        )


class RefractiveIndexLaunchAnglesGeometric(InitialRefractiveIndexBase):
    """
    Initial refractive index provided as launch angles using a geometric
    definition. The magnitude of the refractive index is calculated by root
    finding the dispersion relation.

    Attributes
    ----------
    poloidal_angle_rad : float
        Poloidal launch angle [radians]. Positive means point in positive z
        direction.
    toroidal_angle_rad : float
        Toroidal launch angle [radians]. Positive means point in
        counter-clockwise from above direction.

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate Cartesian refractive index vector.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ("poloidal_angle_rad", "toroidal_angle_rad")

    schema: typing.ClassVar[dict] = {
        "toroidal_angle": {
            "required": True,
            "type": "float",
            "coerce": float,
        },
        "poloidal_angle": {
            "required": True,
            "type": "float",
            "coerce": float,
        },
    }

    def __init__(
        self, toroidal_angle: float, poloidal_angle: float, /, *, radians: bool
    ):
        """
        Inits RefractiveIndexLaunchAnglesGeometric.

        Parameters
        ----------
        toroidal_angle : float
            Toroidal launch angle. Must be in [-pi, pi].
        poloidal_angle : float
            Poloidal launch angle. Must be in [-pi / 2, pi / 2].
        radians : bool
            If True, provided angles are in radians. If False, provided angles
            are in degrees.
        """
        if not radians:
            toroidal_angle *= DEG_TO_RAD
            poloidal_angle *= DEG_TO_RAD

        self.toroidal_angle_rad = np.clip(float(toroidal_angle), -np.pi, np.pi)
        self.poloidal_angle_rad = np.clip(
            float(poloidal_angle), -0.5 * np.pi, 0.5 * np.pi
        )

        if self.toroidal_angle_rad != toroidal_angle:
            logger.warning(
                "toroidal_angle has been clipped to [-pi, pi]: %s -> %s",
                toroidal_angle,
                self.toroidal_angle_rad,
            )

        if self.poloidal_angle_rad != poloidal_angle:
            logger.warning(
                "poloidal_angle has been clipped to [-pi/2, pi/2]: %s -> %s",
                poloidal_angle,
                self.poloidal_angle_rad,
            )

    def clone(self):
        """
        Create copy of self.

        Returns
        -------
        clone : RefractiveIndexLaunchAnglesGeometric
            Independent copy of self.
        """
        return RefractiveIndexLaunchAnglesGeometric(
            self.toroidal_angle_rad, self.poloidal_angle_rad, radians=True
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "toroidal_angle": self.toroidal_angle_rad,
            "poloidal_angle": self.poloidal_angle_rad,
        }

    @classmethod
    def from_dict_toml(
        cls, d: dict, /, *, radians: bool
    ) -> "RefractiveIndexLaunchAnglesGeometric":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        refractive_index_definition : RefractiveIndexLaunchAnglesGeometric
            Initial refractive index using launch angles.
        """
        validator = InitialConditionsTomlValidator()
        validator.validate(d, cls.schema, allow_unknown=True)

        return cls(
            validator.document["toroidal_angle"],
            validator.document["poloidal_angle"],
            radians=radians,
        )

    def unpack(self, coordinate_cache: CoordinateCache) -> FloatArray:
        """
        Generate Cartesian refractive index vector.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.

        Returns
        -------
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.

        Notes
        -----
        Defines a cylindrical refractive index vector in terms of the toroidal
        angle alpha and poloidal angle beta as
            N_r = -cos(beta) * cos(alpha)
            N_phi = cos(beta) * sin(alpha)
            N_z = sin(beta)

        alpha controls the toroidal (left / right) direction where alpha > 0
        makes N_phi point counter clockwise from above.

        beta controls the poloidal (up / down) direction where beta > 0 makes
        N_z point upwards.

        The cylindrical refractive index is then transformed to Cartesian.
        """
        # Transform from launch angles to physical refractive index.
        alpha, beta = self.toroidal_angle_rad, self.poloidal_angle_rad
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        sin_beta, cos_beta = np.sin(beta), np.cos(beta)

        value = np.empty(Dimensions.x.size)
        value[0] = -cos_beta * cos_alpha
        # No minus sign so alpha > 0 => launch counter clockwise.
        value[1] = cos_beta * sin_alpha
        value[2] = sin_beta

        # Transform to holonomic basis.
        value_holonomic = coordinate_cache.transform_basis(
            CoordinateSystem.CYLINDRICAL,
            value,
            TensorType.COVECTOR,
            to_holonomic=True,
        )

        # Convert refractive index to Cartesian.
        return coordinate_cache.transform_tensor_field(
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.CARTESIAN,
            value_holonomic,
            TensorType.COVECTOR,
        )


class RefractiveIndexLaunchAnglesImas(RefractiveIndexLaunchAnglesGeometric):
    """
    Initial refractive index provided as launch angles using IMAS definition.
    The magnitude of the refractive index is calculated by root finding the
    dispersion relation.

    Attributes
    ----------
    poloidal_angle_rad : float
        Poloidal launch angle [radians]. Positive means point in positive z
        direction.
    toroidal_angle_rad : float
        Toroidal launch angle [radians]. Positive means point in
        counter-clockwise from above direction.

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate Cartesian refractive index vector.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ()

    def unpack(self, coordinate_cache: CoordinateCache) -> FloatArray:
        """
        Generate Cartesian refractive index vector.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.

        Returns
        -------
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.

        Notes
        -----
        Defines a cylindrical refractive index vector in terms of the toroidal
        angle alpha and poloidal angle beta as
            N_r = -cos(alpha) * cos(beta)
            N_phi = sin(alpha)
            N_z = cos(alpha) * sin(beta)

        alpha controls the toroidal (left / right) direction where alpha > 0
        makes N_phi point counter clockwise from above.

        beta controls the poloidal (up / down) direction where beta > 0 makes
        N_z point upwards.

        The cylindrical refractive index is then transformed to Cartesian.
        """
        # Transform from launch angles to physical refractive index.
        alpha, beta = self.toroidal_angle_rad, self.poloidal_angle_rad
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        sin_beta, cos_beta = np.sin(beta), np.cos(beta)

        value = np.empty(Dimensions.x.size)
        value[0] = -cos_alpha * cos_beta
        value[1] = sin_alpha
        value[2] = -cos_alpha * sin_beta

        # Transform to holonomic basis.
        value_holonomic = coordinate_cache.transform_basis(
            CoordinateSystem.CYLINDRICAL,
            value,
            TensorType.COVECTOR,
            to_holonomic=True,
        )

        # Convert refractive index to Cartesian.
        return coordinate_cache.transform_tensor_field(
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.CARTESIAN,
            value_holonomic,
            TensorType.COVECTOR,
        )


class RefractiveIndexNparallel(InitialRefractiveIndexBase):
    """
    Initial refractive index defined using parallel refractive index component
    and angle between perpendicular refractive index component. The magnitude
    of the perpendicular refractive index is calculated by root finding the
    dispersion relation.

    Attributes
    ----------
    angle_perp_rad : float
        Angle defining component of refractive index perpendicular to magnetic
        field [radians]. In frame where n_parallel // z
            n_x = n_perp * cos(angle_perp)
            n_y = n_perp * sin(angle_perp)
    n_parallel : float
        Refractive index component parallel to magnetic field.

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate Cartesian refractive index vector.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ("angle_perp_rad", "n_parallel")

    schema: typing.ClassVar[dict] = {
        "n_parallel": {
            "required": True,
            "type": "float",
            "coerce": float,
        },
        "angle_perp": {
            "required": True,
            "type": "float",
            "coerce": float,
        },
    }

    def __init__(
        self, n_parallel: float, angle_perp: float, /, *, radians: bool
    ):
        """
        Inits RefractiveIndexNParallel.

        Parameters
        ----------
        n_parallel : float
            Refractive index component parallel to magnetic field.
        angle_perp : float
            Angle defining refractive index component perpendicular to magnetic
            field. In frame where n_parallel // z
                n_x = n_perp * cos(angle_perp)
                n_y = n_perp * sin(angle_perp)
        radians : bool
            If True, provided angles are in radians. If False, provided angles
            are in degrees.
        """
        if not radians:
            angle_perp *= DEG_TO_RAD

        self.n_parallel = float(n_parallel)
        self.angle_perp_rad = np.clip(float(angle_perp), -np.pi, np.pi)

        if self.angle_perp_rad != angle_perp:
            logger.warning(
                "angle_perp has been clipped to [-pi, pi]: %s -> %s",
                angle_perp,
                self.angle_perp_rad,
            )

    def clone(self):
        """
        Create copy of self.

        Returns
        -------
        clone : RefractiveIndexNparallel
            Independent copy of self.
        """
        return RefractiveIndexNparallel(
            self.n_parallel, self.angle_perp_rad, radians=True
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "n_parallel": self.n_parallel,
            "angle_perp": self.angle_perp_rad,
        }

    @classmethod
    def from_dict_toml(
        cls, d: dict, /, *, radians: bool
    ) -> "RefractiveIndexNparallel":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        refractive_index_definition : RefractiveIndexNparallel
            Initial refractive index using parallel refractive index.
        """
        validator = InitialConditionsTomlValidator()
        validator.validate(d, cls.schema, allow_unknown=True)

        return cls(
            validator.document["n_parallel"],
            validator.document["angle_perp"],
            radians=radians,
        )

    def unpack(self, plasma_cache: PlasmaCache) -> FloatArray:
        """
        Generate Cartesian refractive index vector.

        Parameters
        ----------
        plasma_cache : CoordinateCache
            Cache containing plasma parameter data.

        Returns
        -------
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.

        Raises
        ------
        ValueError
            n_parallel > 1.

        Notes
        -----
        Constructs a Cartesian vector
            n_x = n_perp * cos(angle_perp)
            n_y = n_perp * sin(angle_perp)
            n_z = n_parallel

        This is then rotated so n_z // magnetic field vector.
        """
        if abs(self.n_parallel) > 1:
            raise ValueError(
                "Cannot find root (N_perp): "
                f"n_parallel = {self.n_parallel} > 1"
            )

        # Generate Cartesian refractive index in basis where B // z.
        # This is not exactly the Stix frame where also Nperp // x.
        n_perp = np.sqrt(1.0 - self.n_parallel * self.n_parallel)

        value = np.empty(Dimensions.x.size)
        value[0] = n_perp * np.cos(self.angle_perp_rad)
        value[1] = n_perp * np.sin(self.angle_perp_rad)
        value[2] = self.n_parallel

        # Rotate into global Cartesian frame.
        rotation_z_onto_b = rotation_a_onto_b(
            np.array([0.0, 0.0, 1.0]), plasma_cache.magnetic_field_unit.value
        )

        return rotation_z_onto_b @ value


class RefractiveIndexOptimal(InitialRefractiveIndexBase):
    """
    Initial refractive index defined as optimal launch angles for OX
    conversion. The magnitude of the refractive index is calculated by root
    finding the dispersion relation.

    Attributes
    ----------
    n_parallel_positive : bool
        If parallel refractive index component at the OX conversion is
        positive.

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate Cartesian refractive index vector.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ("n_parallel_positive",)

    schema: typing.ClassVar[dict] = {
        "n_parallel_positive": {
            "required": True,
            "type": "boolean",
        },
    }

    def __init__(self, /, *, n_parallel_positive: bool):
        """
        Inits RefractiveIndexOptimal.

        Parameters
        ----------
        n_parallel_positive : bool
            If parallel refractive index component at the OX conversion is
            positive.
        """
        self.n_parallel_positive = bool(n_parallel_positive)

    def clone(self):
        """
        Create copy of self.

        Returns
        -------
        clone : RefractiveIndexOptimal
            Independent copy of self.
        """
        return RefractiveIndexOptimal(
            n_parallel_positive=self.n_parallel_positive
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "n_parallel_positive": self.n_parallel_positive,
        }

    @classmethod
    def from_dict_toml(cls, d: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        refractive_index_definition : RefractiveIndexOptimal
            Initial refractive index using optimal launch angles for OX
            conversion.
        """
        validator = InitialConditionsTomlValidator()
        validator.validate(d, cls.schema, allow_unknown=True)

        return cls(
            n_parallel_positive=validator.document["n_parallel_positive"],
        )

    def unpack(self, optimal_refractive_index: list[FloatArray]) -> FloatArray:
        """
        Generate Cartesian refractive index vector.

        Parameters
        ----------
        optimal_refractive_index : list[np.array[float]]
            List of 2 optimal refractive index vectors obtained from optimal
            OX conversion optimiser. The first value is for positive n_parallel
            while the second is for negative n_parallel.

        Returns
        -------
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector.
        """
        if self.n_parallel_positive:
            refractive_index = optimal_refractive_index[0]
        else:
            refractive_index = optimal_refractive_index[1]

        return refractive_index


InitialRefractiveIndexType = (
    RefractiveIndexComponents
    | RefractiveIndexLaunchAnglesGeometric
    | RefractiveIndexLaunchAnglesImas
    | RefractiveIndexNparallel
    | RefractiveIndexOptimal
)


class PolarisationWaveMode(IOToml):
    """
    Initial electric field polarisation defined as being a perfect eigenmode
    at the plasma boundary.

    Attributes
    ----------
    wave_mode : WaveMode
        Plasma eigenmode of wave.

    Methods
    -------
    clone
        Create copy of self.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ("wave_mode",)

    def __init__(self, wave_mode: WaveMode):
        """
        Inits PolarisationWaveMode.

        Parameters
        ----------
        wave_mode : WaveMode
            Desired wave mode.

        Raises
        ------
        ValueError
            wave_mode is neither O or X.
        """
        self.wave_mode = wave_mode

        if wave_mode == WaveMode.ANY:
            raise ValueError("wave_mode must be O or X.")

    def clone(self):
        """
        Create copy of self.

        Returns
        -------
        clone : PolarisationWaveMode
            Independent copy of self.
        """
        return PolarisationWaveMode(self.wave_mode)

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {"wave_mode": self.wave_mode.name}

    @classmethod
    def from_dict_toml(cls, d: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        polarisation_definition : PolarisationWaveMode
            Initial polarisation using desired wave mode in plasma.
        """
        return cls(WaveMode.parse(d["wave_mode"]))


class PolarisationEllipseAngles(IOToml):
    """
    Initial electric field polarisation defined using polarisation ellipse.

    Attributes
    ----------
    ellipticity_angle_rad : float
        Polarisation ellipticity angle [radians].
    orientation_angle_rad : float
        Polarisation orientation angle [radians].

    Methods
    -------
    clone
        Create copy of self.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = ("ellipticity_angle_rad", "orientation_angle_rad")

    def __init__(
        self,
        orientation_angle: float,
        ellipticity_angle: float,
        /,
        *,
        radians: bool,
    ):
        """
        Inits PolarisationEllipseAngles.

        Parameters
        ----------
        orientation_angle : float
            Polarisation orientation angle.
        ellipticity_angle : float
            Polarisation ellipticity angle.
        radians : bool
            If True, provided angles are in radians. If False, provided angles
            are in degrees.
        """
        if not radians:
            orientation_angle *= DEG_TO_RAD
            ellipticity_angle *= DEG_TO_RAD

        self.orientation_angle_rad = np.clip(
            float(orientation_angle), -np.pi, np.pi
        )
        self.ellipticity_angle_rad = np.clip(
            float(ellipticity_angle), -0.25 * np.pi, 0.25 * np.pi
        )

        if self.orientation_angle_rad != orientation_angle:
            logger.warning(
                "orientation_angle has been clipped to [-pi, pi]: %s -> %s",
                orientation_angle,
                self.orientation_angle_rad,
            )

        if self.ellipticity_angle_rad != ellipticity_angle:
            logger.warning(
                "ellipticity_angle has been clipped to [-pi/4, pi/4]: "
                "%s -> %s",
                ellipticity_angle,
                self.ellipticity_angle_rad,
            )

    def clone(self):
        """
        Create copy of self.

        Returns
        -------
        clone : PolarisationEllipseAngles
            Independent copy of self.
        """
        return PolarisationEllipseAngles(
            self.orientation_angle_rad,
            self.ellipticity_angle_rad,
            radians=True,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "orientation_angle": self.orientation_angle_rad,
            "ellipticity_angle": self.ellipticity_angle_rad,
        }

    @classmethod
    def from_dict_toml(
        cls, d: dict, /, *, radians: bool
    ) -> "PolarisationEllipseAngles":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        polarisation_definition : PolarisationEllipseAngles
            Initial polarisation using ellipse angles.
        """
        return cls(
            d["orientation_angle"], d["ellipticity_angle"], radians=radians
        )


InitialPolarisationType = PolarisationWaveMode | PolarisationEllipseAngles

REFRACTIVE_INDEX_SOURCE = "refractive_index_source"
COMPONENTS = "components"
LAUNCH_ANGLES_GEOMETRIC = "launch_angles_geometric"
LAUNCH_ANGLES_IMAS = "launch_angles_imas"
N_PARALLEL = "n_parallel"
OPTIMAL_OX = "optimal_ox"

POLARISATION_SOURCE = "polarisation_source"
WAVE_MODE = "wave_mode"
ELLIPSE_ANGLES = "ellipse_angles"


class InitialConditionsSchema(IOToml):
    """
    Object defining construction of initial conditions for ray trajectory.

    Attributes
    ----------
    beam_waist_radius_m : float
        1/e electric field radius as waist of Gaussian beam [m].
    coordinate_system_position : CoordinateSystem
        Coordinate system position components are provided in.
    distance_to_focus_m : float
        Distance of position from focal point of beam.
    divergence_angle_rad : float
        Divergence angle of Gaussian beam [radians].
    frequency_ghz : float
        Frequency [GHz].
    n_radial_zones : int
        Number of radial zones of rays used to represent beam. The number of
        azimuthal rays is determined automatically.
    name : str
        Name of ray.
    polarisation : InitialPolarisationType
        Definition of initial polarisation.
    position : np.array[float]
        Initial position.
    power_w : float
        Initial power [W].
    refractive_index : InitialRefractiveIndexType
        Definition for initial refractive index.
    time_ns : float
        Initial time [ns].

    Methods
    -------
    clone
        Create copy of self.
    unpack
        Generate all ray InitialConditions.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    """

    __slots__ = (
        "beam_waist_radius_m",
        "coordinate_system_position",
        "distance_to_focus_m",
        "divergence_angle_rad",
        "frequency_ghz",
        "n_radial_zones",
        "name",
        "polarisation",
        "position",
        "power_w",
        "refractive_index",
        "time_ns",
    )

    schema: typing.ClassVar[dict] = {
        "time_ns": {
            "required": False,
            "type": "float",
            "coerce": float,
            "default": 0.0,
        },
        "frequency_ghz": {
            "required": True,
            "type": "float",
            "coerce": float,
            "min": 0.0,
        },
        "position": {
            "required": True,
            "type": "ndarray",
            "coerce": coerce_float_ndarray,
        },
        "coordinate_system_position": {
            "required": True,
            "type": "CoordinateSystem",
            "coerce": CoordinateSystem.parse,
        },
        REFRACTIVE_INDEX_SOURCE: {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
            "allowed": (
                COMPONENTS,
                LAUNCH_ANGLES_GEOMETRIC,
                LAUNCH_ANGLES_IMAS,
                N_PARALLEL,
                OPTIMAL_OX,
            ),
        },
        POLARISATION_SOURCE: {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
            "allowed": (WAVE_MODE, ELLIPSE_ANGLES),
        },
        "power_w": {
            "required": True,
            "type": "float",
            "coerce": float,
            "min": 0.0,
        },
        "beam_waist_radius_m": {
            "required": True,
            "type": "float",
            "coerce": float,
            "min": 0.0,
        },
        "n_radial_zones": {
            "required": True,
            "type": "integer",
            "coerce": int,
            "min": 0,
        },
    }

    rays_scaling = 6

    def __init__(
        self,
        name: str,
        time_ns: float,
        frequency_ghz: float,
        position: FloatArray,
        coordinate_system_position: CoordinateSystem,
        refractive_index: InitialRefractiveIndexType,
        polarisation: InitialPolarisationType,
        power_w: float,
        beam_waist_radius_m: float,
        n_radial_zones: int,
    ):
        """
        Inits InitialConditionsSchema.

        Parameters
        ----------
        name : str
            Name of ray. Must not contain a hyphen.
        time_ns : float
            Initial time [ns].
        frequency_ghz : float
            Frequency [GHz].
        position : np.array[float]
            Initial position.
        coordinate_system_position : CoordinateSystem
            Coordinate system position components are provided in.
        refractive_index : InitialRefractiveIndexType
            Definition for initial refractive index.
        polarisation : InitialPolarisationType
            Definition of initial polarisation.
        power_w : float
            Initial power [W].
        beam_waist_radius_m : float
            1/e electric field radius as waist of Gaussian beam [m].
        n_radial_zones : int
            Number of radial zones of rays used to represent beam.

        Raises
        ------
        ValueError
            Ray name contains hyphen.
        """
        self.name = name
        self.time_ns = float(time_ns)
        self.frequency_ghz = float(frequency_ghz)
        self.position = np.asarray(position).reshape((Dimensions.x.size,))
        self.coordinate_system_position = coordinate_system_position
        self.refractive_index = refractive_index
        self.polarisation = polarisation
        self.power_w = float(power_w)
        self.beam_waist_radius_m = float(beam_waist_radius_m)
        self.distance_to_focus_m = 0.0
        self.n_radial_zones = max(1, int(n_radial_zones))

        self.divergence_angle_rad = (
            0.5 * C_M_PER_NS / (frequency_ghz * beam_waist_radius_m)
        )

        if "-" in self.name:
            raise ValueError(
                f"Hyphens are not allowed in ray names: '{self.name}'"
            )

    def clone(self):
        """
        Create copy of self.

        Returns
        -------
        clone : InitialConditionsSchema
            Independent copy of self.
        """
        return InitialConditionsSchema(
            self.name,
            self.time_ns,
            self.frequency_ghz,
            self.position.copy(),
            self.coordinate_system_position,
            self.refractive_index.clone(),
            self.polarisation.clone(),
            self.power_w,
            self.beam_waist_radius_m,
            self.n_radial_zones,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = {
            "time_ns": self.time_ns,
            "frequency_ghz": self.frequency_ghz,
            "position": self.position,
            "coordinate_system_position": self.coordinate_system_position.name,
        }

        # Refractive index.
        _type = type(self.refractive_index)

        if _type is RefractiveIndexComponents:
            d[REFRACTIVE_INDEX_SOURCE] = COMPONENTS
        elif _type is RefractiveIndexLaunchAnglesGeometric:
            d[REFRACTIVE_INDEX_SOURCE] = LAUNCH_ANGLES_GEOMETRIC
        elif _type is RefractiveIndexLaunchAnglesImas:
            d[REFRACTIVE_INDEX_SOURCE] = LAUNCH_ANGLES_IMAS
        elif _type is RefractiveIndexNparallel:
            d[REFRACTIVE_INDEX_SOURCE] = N_PARALLEL
        elif _type is RefractiveIndexOptimal:
            d[REFRACTIVE_INDEX_SOURCE] = OPTIMAL_OX
        else:
            raise NotImplementedError(_type)

        d.update(self.refractive_index.to_dict_toml())

        # Polarisation.
        _type = type(self.polarisation)

        if _type is PolarisationWaveMode:
            d[POLARISATION_SOURCE] = WAVE_MODE
        elif _type is PolarisationEllipseAngles:
            d[POLARISATION_SOURCE] = ELLIPSE_ANGLES
        else:
            raise NotImplementedError(type(self.polarisation))

        d.update(self.polarisation.to_dict_toml())

        # Power and intensity.
        d["power_w"] = self.power_w

        # Beam shape.
        d["beam_waist_radius_m"] = self.beam_waist_radius_m
        d["n_radial_zones"] = self.n_radial_zones

        return d

    @classmethod
    def from_dict_toml(
        cls, document: dict, name: str, /, *, radians: bool
    ) -> "InitialConditionsSchema":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.
        name : str
            Name of ray.
        radians : bool
            If True, provided angles are in radians. If False, provided angles
            are in degrees.

        Returns
        -------
        initial_conditions : InitialConditionsSchema
            Initial conditions definition.
        """
        validator = InitialConditionsTomlValidator()
        validator.validate(document, cls.schema, allow_unknown=True)

        time_ns = validator.document["time_ns"]
        frequency_ghz = validator.document["frequency_ghz"]
        position = validator.document["position"]
        coordinate_system_position = validator.document[
            "coordinate_system_position"
        ]
        power_w = validator.document["power_w"]
        beam_waist_radius_m = validator.document["beam_waist_radius_m"]
        n_radial_zones = validator.document["n_radial_zones"]

        # Get refractive index.
        refractive_index_source = validator.document[REFRACTIVE_INDEX_SOURCE]

        if refractive_index_source == COMPONENTS:
            refractive_index = RefractiveIndexComponents.from_dict_toml(
                document
            )
        elif refractive_index_source == LAUNCH_ANGLES_GEOMETRIC:
            refractive_index = (
                RefractiveIndexLaunchAnglesGeometric.from_dict_toml(
                    document, radians=radians
                )
            )
        elif refractive_index_source == LAUNCH_ANGLES_IMAS:
            refractive_index = RefractiveIndexLaunchAnglesImas.from_dict_toml(
                document, radians=radians
            )
        elif refractive_index_source == N_PARALLEL:
            refractive_index = RefractiveIndexNparallel.from_dict_toml(
                document, radians=radians
            )
        elif refractive_index_source == OPTIMAL_OX:
            refractive_index = RefractiveIndexOptimal.from_dict_toml(document)
        else:
            raise NotImplementedError(refractive_index_source)

        # Get polarisation.
        polarisation_source = validator.document[POLARISATION_SOURCE]

        if polarisation_source == WAVE_MODE:
            polarisation = PolarisationWaveMode.from_dict_toml(document)
        elif polarisation_source == ELLIPSE_ANGLES:
            polarisation = PolarisationEllipseAngles.from_dict_toml(
                document, radians=radians
            )
        else:
            raise NotImplementedError(polarisation_source)

        return cls(
            name,
            time_ns,
            frequency_ghz,
            position,
            coordinate_system_position,
            refractive_index,
            polarisation,
            power_w,
            beam_waist_radius_m,
            n_radial_zones,
        )

    def unpack(
        self,
        coordinate_cache: CoordinateCache,
        plasma_cache: PlasmaCache,
        optimal_refractive_index: list[FloatArray],
    ) -> typing.Iterable[InitialConditions]:
        """
        Generate all ray InitialConditions.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        plasma_cache : PlasmaCache
            Cache containing plasma parameter data.
        optimal_refractive_index : list[np.array[float]]
            List of 2 optimal refractive index vectors obtained from optimal
            OX conversion optimiser. The first value is for positive n_parallel
            while the second is for negative n_parallel.

        Yields
        ------
        initial_condition : InitialConditions
            InitialConditions for generated ray.

        Raises
        ------
        ValueError
            Not a vacuum at initial position.
            Refractive index for optimal OX conversion but no data provided.
            Initial refractive index vector has norm zero.
        """
        # Calculate position.
        coordinate_cache.set_position(
            self.coordinate_system_position, self.position
        )

        # Calculate plasma parameters.
        plasma_cache.set_frequency(self.frequency_ghz)
        plasma_cache.calculate(coordinate_cache, derivatives=0)

        if not plasma_cache.vacuum:
            raise ValueError(
                "Must be a vacuum at initial position: "
                f"X = {plasma_cache.normalised_electron_density.value.item()}"
            )

        # Get Cartesian refractive index.
        if (
            type(self.refractive_index) is RefractiveIndexComponents
            or (
                type(self.refractive_index)
                is RefractiveIndexLaunchAnglesGeometric
            )
            or type(self.refractive_index) is RefractiveIndexLaunchAnglesImas
        ):
            refractive_index_cartesian = self.refractive_index.unpack(
                coordinate_cache
            )
        elif type(self.refractive_index) is RefractiveIndexNparallel:
            refractive_index_cartesian = self.refractive_index.unpack(
                plasma_cache
            )
        elif type(self.refractive_index) is RefractiveIndexOptimal:
            if len(optimal_refractive_index) == 0:
                raise ValueError(
                    "Optimal OX launch conditions data not available. "
                    "Did you run optimise?"
                )

            refractive_index_cartesian = self.refractive_index.unpack(
                optimal_refractive_index
            )
        else:
            raise ValueError("Unknown initial refractive index type.")

        norm = np.linalg.norm(refractive_index_cartesian)

        if np.isclose(norm, 0.0, atol=1e-16):
            raise ValueError("Initial refractive index has norm zero.")

        # Normalise refractive index.
        refractive_index_cartesian /= norm

        # Get initial polarisation.
        if type(self.polarisation) is PolarisationWaveMode:
            # The polarisation for the desired mode will be calculated.
            polarisation_stix = np.zeros(Dimensions.x.size, dtype=ComplexType)
            wave_mode = self.polarisation.wave_mode

        elif type(self.polarisation) is PolarisationEllipseAngles:
            if not plasma_cache.vacuum:
                x = plasma_cache.normalised_electron_density.value.item()

                raise ValueError(
                    "Polarisation defined using ellipse angles only valid "
                    "for waves in vacuum."
                    f"Initial position is not in vacuum (X = {x})"
                )

            psi = self.polarisation.orientation_angle_rad
            chi = self.polarisation.ellipticity_angle_rad

            sin_psi, cos_psi = np.sin(psi), np.cos(psi)
            sin_chi, cos_chi = np.sin(chi), np.cos(chi)

            # Components of polarisation in local orthonormal frame.
            e1 = cos_chi * cos_psi + 1.0j * sin_psi * sin_chi
            e2 = cos_chi * sin_psi - 1.0j * cos_psi * sin_chi

            # Calculate rotation that maps x axis onto refractive index.
            rot = rotation_a_onto_b(
                np.array([1.0, 0.0, 0.0]),
                refractive_index_cartesian,
            )

            # Matrix multiplication with (0, 1, 0) and (0, 0, 1) just gives
            # the columns of the rotation matrix.
            ey, ez = rot[:, 1], rot[:, 2]

            # Calculate polarisation in Cartesian frame.
            polarisation_cartesian = e1 * ey + e2 * ez

            # Transform polarisation into Stix basis.
            polarisation_stix = plasma_cache.cartesian_to_stix(
                polarisation_cartesian
            )
            wave_mode = WaveMode.ANY
        else:
            raise NotImplementedError(self.polarisation)

        # Calculate intensity for a fundamental mode Gaussian beam.
        beam_radius2 = np.square(self.beam_waist_radius_m) + np.square(
            self.divergence_angle_rad * self.distance_to_focus_m
        )
        peak_intensity_w_per_m2 = 2.0 * self.power_w / (np.pi * beam_radius2)

        # Calculate power split for each of the zones defining the beam.
        power_contained = np.empty(self.n_radial_zones)
        radii_m = np.empty(self.n_radial_zones)
        bundle = self.n_radial_zones > 1

        if bundle:
            # Split beam power over N zones with boundaries at fractions of
            # the total beam power
            power_fraction_bins = np.linspace(
                0.5 / self.n_radial_zones, 1.0, self.n_radial_zones
            )

            # Calculate fraction of total beam power contained in each bin.
            power_contained[0] = power_fraction_bins[0]
            power_contained[1:] = (
                power_fraction_bins[1:] - power_fraction_bins[:-1]
            )

            # Only consider beam inside 2.15 * beam width which contains
            # ~0.9999 of total power.
            r_max = 2.15
            fraction_power_captured = 1 - np.exp(-2.0 * r_max * r_max)

            # Calculate radii of bin centres i.e. radii containing X percent
            # of total power in beam.
            if self.n_radial_zones == 2:  # noqa: PLR2004
                bin_centres = (
                    0.5
                    * fraction_power_captured
                    * (power_contained[0] + 0.5 * power_contained[1])
                )
            else:
                bin_centres = (
                    0.5
                    * fraction_power_captured
                    * (power_contained[1:-1] + power_contained[2:])
                )

            radii_m[0] = 0.0
            radii_m[1:] = np.sqrt(
                0.5 * beam_radius2 * np.log(1.0 / (1.0 - bin_centres))
            )

            # Scale to total power.
            power_contained *= self.power_w
        else:
            # Central ray only containing all power..
            power_contained[0] = self.power_w
            radii_m[0] = 0.0

        yield InitialConditions(
            f"{self.name}-0",
            self.time_ns,
            self.frequency_ghz,
            np.copy(coordinate_cache.position_cartesian),
            refractive_index_cartesian,
            0.0,
            0.0,
            polarisation_stix,
            wave_mode,
            power_contained[0],
            peak_intensity_w_per_m2,
            self.beam_waist_radius_m,
            bundle=bundle,
        )

        if not bundle:
            return

        # Define basis vectors in beam frame.
        # Calculate rotation that maps x axis onto refractive index.
        rot = rotation_a_onto_b(
            np.array([1.0, 0.0, 0.0]),
            refractive_index_cartesian,
        )

        # Find 2 orthogonal vectors to refractive index.
        # Matrix multiplication with (0, 1, 0) and (0, 0, 1) just gives
        # the columns of the rotation matrix.
        ey, ez = rot[:, 1], rot[:, 2]

        n = 1
        for i in range(1, self.n_radial_zones):
            # Calculate number of rays in this zone. Ensuring its even ensures
            # distribution of rays stays ~ even in angle.
            n_rays = 2 * int(
                np.ceil(0.5 * self.rays_scaling * radii_m[i] / radii_m[1])
            )

            # Drop off in intensity away from beam axis.
            power_w = power_contained[i] / n_rays
            intensity_w_per_m2 = peak_intensity_w_per_m2 * np.exp(
                -2 * np.square(radii_m[i - 1]) / beam_radius2
            )

            theta = 0.0
            dtheta = 2 * np.pi / n_rays

            # Offset neighbouring zones to keep ray distribution more even.
            if i % 2 == 0:
                theta += 0.5 * dtheta

            for _ in range(n_rays):
                # Calculate position offset from beam axis.
                _position_cartesian = (
                    coordinate_cache.position_cartesian
                    + np.cos(theta) * radii_m[i] * ey
                    + np.sin(theta) * radii_m[i] * ez
                )

                # For now do non-diverging beam bundle.
                _refractive_index_cartesian = refractive_index_cartesian

                yield InitialConditions(
                    f"{self.name}-{n}",
                    self.time_ns,
                    self.frequency_ghz,
                    _position_cartesian,
                    _refractive_index_cartesian,
                    0.0,
                    0.0,
                    polarisation_stix,
                    wave_mode,
                    power_w,
                    intensity_w_per_m2,
                    self.beam_waist_radius_m,
                    bundle=True,
                )

                # Increment angle.
                theta += dtheta

                # Increment ray count.
                n += 1


ANGLE_FORMAT = "angle_format"

schema_toml_file = {
    ANGLE_FORMAT: {
        "required": True,
        "type": "AngleFormat",
        "coerce": AngleFormat.parse,
    },
}

schema_unknown = {
    "type": "dict",
}


def read_initial_conditions_toml(
    fh: typing.TextIO,
) -> list[InitialConditionsSchema]:
    """
    Read all ray initial conditions from TOML file.

    Parameters
    ----------
    fh : TextIO
        Handle to TOML file to read from.

    Returns
    -------
    initial_conditions : list[InitialConditionsSchema]
        List of initial condition definions.
    """
    document = toml.load(fh)
    validator = InitialConditionsTomlValidator()
    validator.validate(document, schema_toml_file, allow_unknown=True)

    initial_conditions = []
    radians = validator.document.pop(ANGLE_FORMAT) == AngleFormat.RADIANS

    for name, d in validator.document.items():
        initial_conditions.append(
            InitialConditionsSchema.from_dict_toml(d, name, radians=radians)
        )

    return initial_conditions


def write_initial_conditions_toml(
    fh: typing.TextIO, schemas: list[InitialConditionsSchema]
):
    """
    Write all ray initial conditions to TOML file.

    Parameters
    ----------
    fh : TextIO
        Handle to TOML file to write to.
    schemas : list[InitialConditionsSchema]
        List of initial condition definions.
    """
    d = {ANGLE_FORMAT: AngleFormat.RADIANS.name}

    for schema in schemas:
        d[schema.name] = schema.to_dict_toml()

    toml.dump(d, fh, encoder=IOToml.numpy_encoder)
