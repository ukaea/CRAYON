"""
Classes for constructing models of system plasma parameters.
"""

# Standard imports
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np
from scipy import integrate

# Local imports
from crayon.coordinates import (
    AxisymmetricFluxCoordinate,
    AxisymmetricFluxCoordinateRebase,
    CoordinateCoordinator,
    CoordinateSystem,
    Toroidal,
)
from crayon.imas import DBEntry, IDSEmptyError, ids_empty, imasdef
from crayon.shared.io import IONetcdf, IOToml
from crayon.shared.types import FloatArray
from crayon.system_data.limiter import (
    CappedCone,
    Cylinder,
    Disk,
    Limiter,
    LimiterEffect,
    Plane,
)
from crayon.system_data.schemas import (
    COORDINATES,
    DATA_SOURCES,
    EFFECTIVE_CHARGE,
    ELECTRON_DENSITY_PER_M3,
    ELECTRON_TEMPERATURE_EV,
    IMAS,
    KINETIC,
    LIMITERS,
    MAGNETIC_FIELD_T,
    NETCDF,
    SIMPLE,
    STELLARATOR,
    TOKAMAK,
    VMEC,
    CoordinateType,
    DataSourceImas,
    DataSourceNetcdf,
    DataSourceType,
    DataSourceVmec,
    KineticModelType,
    LimiterAnalyticBoundingBox2D,
    LimiterAnalyticBoundingBox3D,
    LimiterImas2D,
    LimiterImas3D,
    LimiterImasBoundingBox2D,
    LimiterNetcdf,
    LimiterSchemaType,
    MagneticModelStellarator,
    MagneticModelTokamak,
    MagneticModelType,
    ModelAnalyticConstant,
    ModelAnalyticQuadraticBowl,
    ModelAnalyticQuadraticChannel,
    ModelAnalyticQuadraticWell,
    ModelAnalyticRamp,
    ModelAnalyticType,
    ModelImas,
    ModelNetcdf,
    parse_schema,
)
from crayon.value_model.base import ValueModelBase
from crayon.value_model.models import ValueModel
from crayon.value_model.splines import (
    Spline1D,
    Spline2D,
    Spline3D,
    SplineMethod,
)

logger = logging.getLogger(__name__)


class Kinetic(IONetcdf):
    """
    Kinetic plasma parameter models.

    Attributes
    ----------
    effective_charge : ValueModelBase
        Effective charge model.
    electron_density_per_m3 : ValueModelBase
        Electron density model [m^-3].
    electron_temperature_ev : ValueModelBase
        Electron temperature model [eV].

    Methods
    -------
    from_imas_profiles_1d
        Create 1D spline fit of profile data from IMAS IDS core_profiles.
    from_netcdf

    read_netcdf
        Load from netCDF4 dataset.
    write_netcdf
        Write to netCDF4 dataset.
    """

    __slots__ = (
        "effective_charge",
        "electron_density_per_m3",
        "electron_temperature_ev",
    )

    def __init__(
        self,
        electron_density_per_m3: ValueModelBase,
        electron_temperature_ev: ValueModelBase,
        effective_charge: ValueModelBase,
    ):
        """
        Inits Kinetic.

        Attributes
        ----------
        electron_density_per_m3 : ValueModelBase
            Electron density model [m^-3].
        electron_temperature_ev : ValueModelBase
            Electron temperature model [eV].
        effective_charge : ValueModelBase
            Effective charge model.
        """
        self.electron_density_per_m3 = electron_density_per_m3
        self.electron_temperature_ev = electron_temperature_ev
        self.effective_charge = effective_charge

    @staticmethod
    def from_imas_profiles_1d(
        model_name: str,
        ids_core_profiles,
        time_index: int,
        scale_factor: float,
    ) -> Spline1D:
        """
        Create 1d spline from IMAS IDS timeslice core_profiles.profiles_1d.

        Parameters
        ----------
        model_name : str
            Name of kinetic model to load.
        ids_core_profiles
            IMAS IDS core_profiles.
        time_index : int
            Index of slice in ids_core_profiles to load from.
        scale_factor : float
            Scale factor for model value.

        Returns
        -------
        spline : Spline1D
            Spline fit of loaded data.

        Raises
        ------
        IDSEmptyError
            Neither rho_pol_norm nor rho_tor_norm set in ids_core_profiles.
            Kinetic profile data not set in ids_core_profiles.
        """
        # Load radial grid. Prefer rho poloidal grid but accept rho toroidal.
        profiles_1d = ids_core_profiles.profiles_1d[time_index]

        if not ids_empty(profiles_1d.grid.rho_pol_norm):
            radial_grid = profiles_1d.grid.rho_pol_norm
            coordinate_system = CoordinateSystem.RHO_POLOIDAL
        elif not ids_empty(profiles_1d.grid.rho_tor_norm):
            radial_grid = profiles_1d.grid.rho_tor_norm
            coordinate_system = CoordinateSystem.RHO_TOROIDAL
        else:
            raise IDSEmptyError(
                "In IDS neither core_profiles.profiles_1d.rho_pol_norm nor "
                "core_profiles.profiles_1d.rho_tor_norm set."
            )

        # Load 1d profile.
        if model_name == ELECTRON_DENSITY_PER_M3:
            data = profiles_1d.electrons.density
            model = ValueModel.electron_density_per_m3()

            if ids_empty(data):
                raise IDSEmptyError(
                    "In IDS core_profiles.profiles_1d.electrons.density "
                    "not set."
                )

        elif model_name == ELECTRON_TEMPERATURE_EV:
            data = profiles_1d.electrons.temperature
            model = ValueModel.electron_temperature_ev()

            if ids_empty(data):
                raise IDSEmptyError(
                    "In IDS core_profiles.profiles_1d.electrons.temperature "
                    "not set."
                )

        elif model_name == EFFECTIVE_CHARGE:
            data = profiles_1d.zeff
            model = ValueModel.effective_charge()

            if ids_empty(data):
                raise IDSEmptyError(
                    "In IDS core_profiles.profiles_1d.zeff not set."
                )

        else:
            raise NotImplementedError(model_name)

        return model.spline_1d(
            coordinate_system,
            radial_grid,
            data,
            (True, False, False),
            scale_factor=scale_factor,
            method=SplineMethod.QUINTIC,
        )

    @classmethod
    def from_netcdf(
        cls,
        model_name: str,
        time_s: float,
        dset: nc4.Group,
        scale_factor: float,
    ) -> Spline1D | Spline2D | Spline3D:
        """
        Create 1d spline from netCDF4 file data.

        Parameters
        ----------
        model_name : str
            Name of kinetic model to load.
        time_s : float
            Time to load [s].
        dset : netCDF4.Dataset
            netCDF4 dataset or group to read data from.
        scale_factor : float
            Scale factor for model value.

        Returns
        -------
        spline : Spline1D | Spline2D | Spline3D
            Spline fit to data.

        Raises
        ------
        ValueError
            Data has incorrect shape.
            Unsupported spline dimension.
        """
        # Get time index of slice.
        time = dset["time_s"][:].data
        time_index = np.argmin(abs(time - time_s))

        # Load data.
        signal = dset[model_name]
        signal_data = signal["data"][time_index, ...].data

        # Get coordinate system.
        coordinate_system = CoordinateSystem.parse(
            signal.getncattr("coordinate_system")
        )

        # Load abscissas.
        dependent_components = signal["dependent_components"][:].data
        ndim = sum(dependent_components)
        dependent_components = dependent_components.astype(bool)

        dimensions = signal["data"].dimensions
        if len(dimensions) != ndim + 1:
            raise ValueError(
                f"{model_name} has wrong shape. "
                f"Expected {ndim + 1} dimensions (1 time, {ndim} space) "
                f"but got {len(dimensions)}."
            )

        abscissas = tuple(dset[dim][:].data for dim in dimensions[1:])

        # Load 1d profile.
        if model_name == ELECTRON_DENSITY_PER_M3:
            model = ValueModel.electron_density_per_m3()
        elif model_name == ELECTRON_TEMPERATURE_EV:
            model = ValueModel.electron_temperature_ev()
        elif model_name == EFFECTIVE_CHARGE:
            model = ValueModel.effective_charge()
        else:
            raise NotImplementedError(model_name)

        if ndim == 1:
            return model.spline_1d(
                coordinate_system,
                abscissas[0],
                signal_data,
                dependent_components,
                scale_factor=scale_factor,
                method=SplineMethod.QUINTIC,
            )
        if ndim == 2:  # noqa: PLR2004
            return model.spline_2d(
                coordinate_system,
                abscissas[0],
                abscissas[1],
                signal_data,
                dependent_components,
                scale_factor=scale_factor,
                method=SplineMethod.QUINTIC,
            )
        if ndim == 3:  # noqa: PLR2004
            raise NotImplementedError("scipy version issue")
            return model.spline_3d(
                coordinate_system,
                abscissas[0],
                abscissas[1],
                abscissas[2],
                signal_data,
                dependent_components,
                scale_factor=scale_factor,
                method=SplineMethod.QUINTIC,
            )

        raise ValueError(ndim)

    @classmethod
    def read_netcdf(cls, dset: nc4.Group) -> "Kinetic":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        kinetic : Kinetic
            Kinetic plasma parameter models.
        """
        electron_density_per_m3 = ValueModel.read_netcdf(
            dset[ELECTRON_DENSITY_PER_M3]
        )
        electron_temperature_ev = ValueModel.read_netcdf(
            dset[ELECTRON_TEMPERATURE_EV]
        )
        effective_charge = ValueModel.read_netcdf(dset[EFFECTIVE_CHARGE])

        return cls(
            electron_density_per_m3, electron_temperature_ev, effective_charge
        )

    def write_netcdf(self, dset: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        self.electron_density_per_m3.write_netcdf(
            dset.createGroup(ELECTRON_DENSITY_PER_M3)
        )
        self.electron_temperature_ev.write_netcdf(
            dset.createGroup(ELECTRON_TEMPERATURE_EV)
        )
        self.effective_charge.write_netcdf(dset.createGroup(EFFECTIVE_CHARGE))


class Magnetic(IONetcdf):
    """
    Magnetic field data model.

    Attributes
    ----------
    magnetic_field_t : ValueModelBase
        Magnetic field vector model [T].
    """

    __slots__ = ("magnetic_field_t",)

    topology = SIMPLE

    def __init__(self, magnetic_field_t: ValueModelBase):
        """
        Inits Magnetic.

        Parameters
        ----------
        magnetic_field_t : ValueModelBase
            Magnetic field vector model [T].
        """
        self.magnetic_field_t = magnetic_field_t

    @classmethod
    def from_imas(
        cls, ids_equilibrium, time_index: int, scale_factor: float
    ) -> "Magnetic":
        """
        Load cylindrical magnetic field components from equilibrium IDS.

        Parameters
        ----------
        ids_equilibrium
            IMAS IDS equilibrium.
        time_index : int
            Index of timeslice to load from equilibrium.
        scale_factor : float
            Scale factor for magnetic field vector.

        Returns
        -------
        magnetic : Magnetic
            Magnetic field model.

        Raises
        ------
        IDSEmptyError
            No poloidal flux vs (r, z) in equilibrium_ids.profiles_2d.
            Grid dimensions or magnetic field components missing.
        """
        # Find 2D poloidal flux grid against (r, z).
        time_slice = ids_equilibrium.time_slice[time_index]

        found_profiles_2d = False

        for _profiles_2d in time_slice.profiles_2d:
            if _profiles_2d.grid_type.index == 1:
                found_profiles_2d = True

        if not found_profiles_2d:
            raise IDSEmptyError(
                "Cannot find poloidal flux on r, z grid in "
                "equilibrium.profiles_2d IDS."
            )

        r = _profiles_2d.grid.dim1
        z = _profiles_2d.grid.dim2
        br = _profiles_2d.b_field_r
        bphi = _profiles_2d.b_field_phi
        bz = _profiles_2d.b_field_z

        if ids_empty(r):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.grid.dim1 not set."
            )

        if ids_empty(z):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.grid.dim2 not set."
            )

        if ids_empty(br):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.b_field_r not set."
            )

        if ids_empty(bphi):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.b_field_phi not set."
            )

        if ids_empty(bz):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.b_field_z not set."
            )

        # Transform magnetic field to holonomic basis.
        magnetic_field_data = np.empty((r.size, z.size, 3))
        magnetic_field_data[:, :, 0] = br
        magnetic_field_data[:, :, 1] = bphi / r[:, np.newaxis]
        magnetic_field_data[:, :, 2] = bz

        # Create spline.
        magnetic_field_t = ValueModel.magnetic_field_t().spline_2d(
            CoordinateSystem.CYLINDRICAL,
            r,
            z,
            magnetic_field_data,
            (True, False, True),
            scale_factor=scale_factor,
            method=SplineMethod.CUBIC,
        )

        return cls(magnetic_field_t)

    @classmethod
    def from_netcdf(
        cls, time_s: float, dset: nc4.Group, scale_factor: float
    ) -> "Magnetic":
        """
        Load magnetic field components from netCDF4 file.

        Parameters
        ----------
        time_s: float
            Time to load data [s].
        dset : netCDF4.Group
            netCDF4 dataset or group to read data from.
        scale_factor: float
            Scale factor for magnetic field model.

        Returns
        -------
        magnetic : Magnetic
            Magnetic field model.

        Raises
        ------
        ValueError
            Incorrect data shape.
            Unsupported spline dimension.
        """
        # Get time index of slice.
        time = dset["time_s"][:].data
        time_index = np.argmin(abs(time - time_s))

        # Load data.
        signal = dset[MAGNETIC_FIELD_T]
        signal_data = signal["data"][time_index, ...].data

        # Get coordinate system.
        coordinate_system = CoordinateSystem.parse(
            signal.getncattr("coordinate_system")
        )

        # Load abscissas.
        dependent_components = signal["dependent_components"][:].data
        ndim = sum(dependent_components)
        dependent_components = dependent_components.astype(bool)

        dimensions = signal["data"].dimensions
        if len(dimensions) != ndim + 2:
            raise ValueError(
                f"{MAGNETIC_FIELD_T} has wrong shape. "
                f"Expected {ndim + 1} dimensions "
                f"(1 time, {ndim} space, 1 components) "
                f"but got {len(dimensions)}."
            )

        abscissas = tuple(dset[dim][:].data for dim in dimensions[1:-1])

        model = ValueModel.magnetic_field_t()

        if ndim == 1:
            magnetic_field_t = model.spline_1d(
                coordinate_system,
                abscissas[0],
                signal_data,
                dependent_components,
                scale_factor=scale_factor,
                method=SplineMethod.QUINTIC,
            )
        elif ndim == 2:  # noqa: PLR2004
            magnetic_field_t = model.spline_2d(
                coordinate_system,
                abscissas[0],
                abscissas[1],
                signal_data,
                dependent_components,
                scale_factor=scale_factor,
                method=SplineMethod.QUINTIC,
            )
        elif ndim == 3:  # noqa: PLR2004
            raise NotImplementedError("scipy version issue")
            magnetic_field_t = model.spline_3d(
                coordinate_system,
                abscissas[0],
                abscissas[1],
                abscissas[2],
                signal_data,
                dependent_components,
                scale_factor=scale_factor,
                method=SplineMethod.QUINTIC,
            )
        else:
            raise ValueError(ndim)

        return cls(magnetic_field_t)

    def register_flux_coordinates(self, *_args, **_kwargs):
        """
        Register flux coordinate system.
        """
        # No flux coordinates to register.

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "Magnetic":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        magnetic : Magnetic
            Magnetic field models.
        """
        magnetic_field_t = ValueModel.read_netcdf(dset[MAGNETIC_FIELD_T])

        return cls(magnetic_field_t)

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        dset.setncattr("topology", self.topology)
        self.magnetic_field_t.write_netcdf(dset.createGroup(MAGNETIC_FIELD_T))


class MagneticTokamak(Magnetic):
    """
    Tokamak magnetic field data model.

    Attributes
    ----------
    b_max : Spline1D
        Maximum magnetic field strength on flux surface  vs radial flux
        coordinate [T].
    cross_sectional_area_m2 : Spline1D
        Flux surface cross sectional area vs radial flux coordinate [m^-2].
    f_toroidal_tm : Spline1D
        Diamagnetic function F vs radial flux coordinate [T.m].
    fsa_1_over_r : Spline1D
        Flux surface averaged 1 / r vs radial flux coordinate [m^-1].
    fsa_1_over_r2 : Spline1D
        Flux surface averaged 1 / r**2 vs radial flux coordinate [m^-2].
    fsa_b : Spline1D
        Flux surface averaged magnetic field strength vs radial flux
        coordinate [T].
    magnetic_axis_m : tuple[float, float]
        Cylindrical (r, z) position of magnetic axis [m].
    rho_poloidal_2d : Spline2D
        Root normalised poloidal flux rho as a function of cylindrical (r, z).
    rho_poloidal_coordinate : AxisymmetricFluxCoordinate
        Root normalised poloidal flux coordinate system.
    rho_poloidal_to_toroidal_1d : Spline1D
        Rho toroidal as a function of rho poloidal.
    rho_toroidal_coordinate : AxisymmetricFluxCoordinateRebase
        Root normalised toroidal flux coordinate system.
    rho_toroidal_to_poloidal_1d : Spline1D
        Rho poloidal as a function of rho toroidal.
    total_poloidal_flux_wb : float
        Total poloidal flux between magnetic axis and separatrix [Wb].
    total_toroidal_flux_wb : float
        Total toroidal flux between magnetic axis and separatrix [Wb].
    trapped_particle_fraction : Spline1D
        Trapped particle fraction vs radial flux coordinate.
    volume_m3 : Spline1D
        Flux tube volume vs radial flux coordinate [m^-3].

    Methods
    -------
    calculate_flux_surface_integrals
        Calculate MagneticTokamak evaluating flux surface integrals.
    from_imas
        Calculate MagneticTokamak from IMAS data.
    register_flux_coordinates
        Register flux coordinate systems.
    read_netcdf
        Load from netCDF4 dataset.
    set_magnetic_field
        Set magnetic field vector model and create flux coordinate systems.
    construct_flux_function_splines
        Create spline fits of flux functions.
    write_netcdf
        Write to netCDF4 dataset.
    """

    __slots__ = (
        "b_max",
        "cross_sectional_area_m2",
        "f_toroidal_tm",
        "fsa_1_over_r",
        "fsa_1_over_r2",
        "fsa_b",
        "magnetic_axis_m",
        "rho_poloidal_2d",
        "rho_poloidal_coordinate",
        "rho_poloidal_to_toroidal_1d",
        "rho_toroidal_coordinate",
        "rho_toroidal_to_poloidal_1d",
        "total_poloidal_flux_wb",
        "total_toroidal_flux_wb",
        "trapped_particle_fraction",
        "volume_m3",
    )

    topology = TOKAMAK

    def __init__(
        self,
        rho_poloidal_2d: Spline2D,
        magnetic_axis_m: tuple[float, float],
        rho_poloidal_to_toroidal_1d: Spline1D,
        rho_toroidal_to_poloidal_1d: Spline1D,
        total_poloidal_flux_wb: float,
        total_toroidal_flux_wb: float,
        f_toroidal_tm: Spline1D,
        cross_sectional_area_m2: Spline1D,
        volume_m3: Spline1D,
        fsa_1_over_r: Spline1D,
        fsa_1_over_r2: Spline1D,
        fsa_b: Spline1D,
        b_max: Spline1D,
        trapped_particle_fraction: Spline1D,
        /,
        *,
        scale_factor_magnetic_field: float = 1.0,
        boundary_rz: tuple[FloatArray, FloatArray] | None = None,
    ):
        """
        Inits MagneticTokamak.

        Parameters
        ----------
        rho_poloidal_2d : Spline2D
            Root normalised poloidal flux rho as a function of cylindrical
            (r, z).
        magnetic_axis_m : tuple[float, float]
            Cylindrical (r, z) position of magnetic axis [m].
        rho_poloidal_to_toroidal_1d : Spline1D
            Rho toroidal as a function of rho poloidal.
        rho_toroidal_to_poloidal_1d : Spline1D
            Rho poloidal as a function of rho toroidal.
        total_poloidal_flux_wb : float
            Total poloidal flux between magnetic axis and separatrix [Wb].
        total_toroidal_flux_wb : float
            Total toroidal flux between magnetic axis and separatrix [Wb].
        f_toroidal_tm : Spline1D
            Diamagnetic function F vs radial flux coordinate [T.m].
        cross_sectional_area_m2  : Spline1D
            Flux surface cross sectional area vs radial flux coordinate [m^-2].
        volume_m3 : Spline1D
            Flux tube volume vs radial flux coordinate [m^-3].
        fsa_1_over_r : Spline1D
            Flux surface averaged 1 / r**2 vs radial flux coordinate [m^-1].
        fsa_1_over_r2 : Spline1D
            Flux surface averaged 1 / r**2 vs radial flux coordinate [m^-2].
        fsa_b : Spline1D
            Flux surface averaged magnetic field strength vs radial flux
            coordinate [T].
        b_max : Spline1D
            Maximum magnetic field strength vs radial flux coordinate [T].
        trapped_particle_fraction : Spline1D
            Trapped particle fraction vs radial flux coordinate.
        scale_factor_magnetic_field : float, optional
            Scale factor for magnetic field vector.
        boundary_rz : tuple[np.array[float], np.array[float]], optional
            Cylindrical (r, z) coordinates of separatrix.
        """
        self.set_magnetic_field(
            rho_poloidal_2d,
            magnetic_axis_m,
            rho_poloidal_to_toroidal_1d,
            rho_toroidal_to_poloidal_1d,
            total_poloidal_flux_wb,
            total_toroidal_flux_wb,
            f_toroidal_tm,
            scale_factor_magnetic_field=scale_factor_magnetic_field,
            boundary_rz=boundary_rz,
        )

        self.cross_sectional_area_m2 = cross_sectional_area_m2
        self.volume_m3 = volume_m3
        self.fsa_1_over_r = fsa_1_over_r
        self.fsa_1_over_r2 = fsa_1_over_r2
        self.fsa_b = fsa_b
        self.b_max = b_max
        self.trapped_particle_fraction = trapped_particle_fraction

    def set_magnetic_field(
        self,
        rho_poloidal_2d: Spline2D,
        magnetic_axis_m: tuple[float, float],
        rho_poloidal_to_toroidal_1d: Spline1D,
        rho_toroidal_to_poloidal_1d: Spline1D,
        total_poloidal_flux_wb: float,
        total_toroidal_flux_wb: float,
        f_toroidal_tm: Spline1D,
        /,
        *,
        scale_factor_magnetic_field: float = 1.0,
        boundary_rz: tuple[FloatArray, FloatArray] | None = None,
    ):
        """
        Set magnetic field vector model and create flux coordinate systems.

        Parameters
        ----------
        rho_poloidal_2d : Spline2D
            Root normalised poloidal flux rho as a function of cylindrical
            (r, z).
        magnetic_axis_m : tuple[float, float]
            Cylindrical (r, z) position of magnetic axis [m].
        rho_poloidal_to_toroidal_1d : Spline1D
            Rho toroidal as a function of rho poloidal.
        rho_toroidal_to_poloidal_1d : Spline1D
            Rho poloidal as a function of rho toroidal.
        total_poloidal_flux_wb : float
            Total poloidal flux between magnetic axis and separatrix [Wb].
        total_toroidal_flux_wb : float
            Total toroidal flux between magnetic axis and separatrix [Wb].
        f_toroidal_tm : Spline1D
            Diamagnetic function F vs radial flux coordinate [T.m].
        scale_factor_magnetic_field : float, optional
            Scale factor for magnetic field vector.
        boundary_rz : tuple[np.array[float], np.array[float]], optional
            Cylindrical (r, z) coordinates of separatrix.
        """
        self.rho_poloidal_2d = rho_poloidal_2d
        self.rho_poloidal_to_toroidal_1d = rho_poloidal_to_toroidal_1d
        self.rho_toroidal_to_poloidal_1d = rho_toroidal_to_poloidal_1d
        self.total_poloidal_flux_wb = float(total_poloidal_flux_wb)
        self.total_toroidal_flux_wb = float(total_toroidal_flux_wb)
        self.f_toroidal_tm = f_toroidal_tm

        self.magnetic_axis_m = np.asarray(
            magnetic_axis_m, dtype=float
        ).reshape(2)

        self.rho_poloidal_coordinate = (
            AxisymmetricFluxCoordinate.find_contours(
                CoordinateSystem.RHO_POLOIDAL,
                self.rho_poloidal_2d,
                self.magnetic_axis_m,
                boundary_rz=boundary_rz,
            )
        )

        self.rho_toroidal_coordinate = AxisymmetricFluxCoordinateRebase(
            self.rho_poloidal_coordinate,
            CoordinateSystem.RHO_TOROIDAL,
            self.rho_poloidal_to_toroidal_1d,
            self.rho_toroidal_to_poloidal_1d,
        )

        magnetic_field_t = ValueModel.magnetic_field_t().axisymmetric(
            self.rho_poloidal_2d,
            self.f_toroidal_tm,
            self.total_poloidal_flux_wb,
            scale_factor=scale_factor_magnetic_field,
        )

        super().__init__(magnetic_field_t)

    def construct_flux_function_splines(
        self,
        rho_poloidal: FloatArray,
        cross_sectional_area_m2: FloatArray,
        volume_m3: FloatArray,
        fsa_1_over_r: FloatArray,
        fsa_1_over_r2: FloatArray,
        fsa_b: FloatArray,
        b_max: FloatArray,
        trapped_particle_fraction: FloatArray,
    ):
        """
        Create spline fits of flux functions.

        Parameters
        ----------
        rho_poloidal: np.array[float]
            Root normalised poloidal flux.
        cross_sectional_area_m2: np.array[float]
            Flux surface cross sectional area [m^2].
        volume_m3: np.array[float]
            Flux tube volume area [m^2].
        fsa_1_over_r: np.array[float]
            Flux surface averaged 1 / r [m^-1].
        fsa_1_over_r2: np.array[float]
            Flux surface averaged 1 / r**2 [m^-2].
        fsa_b: np.array[float]
            Flux surface averaged magnetic field strength [T].
        b_max: np.array[float]
            Maximum magnetic field strength on flux surface [T].
        trapped_particle_fraction: np.array[float]
            Trapped particle fraction.
        """
        self.cross_sectional_area_m2 = (
            ValueModel.cross_sectional_area_m2().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal,
                cross_sectional_area_m2,
                (True,),
            )
        )

        self.volume_m3 = ValueModel.volume_m3().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal,
            volume_m3,
            (True,),
        )

        self.fsa_1_over_r = ValueModel.fsa_1_over_r().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal,
            fsa_1_over_r,
            (True,),
        )

        self.fsa_1_over_r2 = ValueModel.fsa_1_over_r2().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal,
            fsa_1_over_r2,
            (True,),
        )

        self.fsa_b = ValueModel.fsa_b().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal,
            fsa_b,
            (True,),
        )

        self.b_max = ValueModel.b_max().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal,
            b_max,
            (True,),
        )

        self.trapped_particle_fraction = (
            ValueModel.trapped_particle_fraction().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal,
                trapped_particle_fraction,
                (True,),
            )
        )

    @staticmethod
    def _perimeter(
        theta: FloatArray,
        polar_radius2: FloatArray,
        polar_radius_dtheta: FloatArray,
    ) -> float:
        """
        Calculate perimeter of a convex closed curve.

        Parameters
        ----------
        theta : np.array[float]
            Polar angle with respect to the winding point along curve.
        polar_radius2 : np.array[float]
            Square of the polar radius from the winding point along curve.
        polar_radius_dtheta : np.array[float]
            First derivative of polar radius with respect to polar angle
            along curve.

        Returns
        -------
        perimeter: float
            Perimeter of curve.
        """
        return integrate.trapezoid(
            np.sqrt(polar_radius2 + np.square(polar_radius_dtheta)), x=theta
        )

    @staticmethod
    def _cross_sectional_area(
        theta: FloatArray, polar_radius2: FloatArray
    ) -> float:
        """
        Calculate enclosed area of a convex closed curve.

        Parameters
        ----------
        theta : np.array[float]
            Polar angle with respect to the winding point along curve.
        polar_radius2 : np.array[float]
            Square of the polar radius from the winding point along curve.

        Returns
        -------
        cross_sectional_area: float
            Area enclosed by curve.
        """
        return 0.5 * integrate.trapezoid(polar_radius2, x=theta)

    @staticmethod
    def _calculate_area_volume(
        theta: FloatArray, radius: FloatArray, polar_radius: FloatArray
    ) -> tuple[float, float, float, float]:
        """
        Return cross sectional area [m^2] and volume of solid of revolution
        [m^3] of a convex closed curve.

        Parameters
        ----------
        theta : np.array[float]
            Poloidal angle grid for r and z values
        radius : np.array[float]
            Radius r along curve
        polar_radius : np.array[float]
            Radius along curve relative to magnetic axis.

        Returns
        -------
        cross_sectional_area_m2 : float
            Area enclosed by curve.
        volume_m3 : float
            Volume enclosed by volume of revolution of curve.

        Notes
        -----
        The curve gives x and y as a function of geometric poloidal angle
        theta about the winding point x0, y0.

        The perimeter P can therefore be found as
            P = int sqrt(r**2 + (dr/dtheta)**2) dtheta

        The cross sectional area A can be found as
            A = int r**2 dtheta

        The surface area of the surface of revolution S and the volume of the
        volume of revolution V from rotation about the y axis is found using
        Pappus' theorems. For this we need the x position of the geometric
        centroid <x>
            2 * np.pi * <x> = int x dtheta

        We then have
            S = 2 * np.pi * <x> * P
            V = 2 * np.pi * <x> * A
        """
        # Calculate radius of contour.
        polar_radius2 = np.square(polar_radius)

        # Calculate area enclosed by contour.
        cross_sectional_area_m2 = MagneticTokamak._cross_sectional_area(
            theta, polar_radius2
        )

        # r position of centroid multiplied by 2 pi
        r_centroid_times_2pi = integrate.trapezoid(radius, x=theta)

        # Evaluate volume using Pappus' second theorem.
        volume_m3 = r_centroid_times_2pi * cross_sectional_area_m2

        return cross_sectional_area_m2, volume_m3

    @staticmethod
    def _get_fsa_kernel_denominator(
        theta: FloatArray,
        radius: FloatArray,
        polar_radius: FloatArray,
        grad_rho: FloatArray,
    ) -> tuple[FloatArray, float]:
        """
        Calculate integration kernel and normalising denominator for flux
        surface averages.

        The flux surface average

        Parameters
        ----------
        theta : np.array[float]
            Poloidal angle along flux surface.
        radius : np.array[float]
            Cylindrical major radius along flux surface.
        polar_radius : np.array[float]
            Radius from magnetic axis along flux surface
        grad_rho : np.array[float]
            Magnitude of gradient in root normalised poloidal flux along
            flux surface.

        Returns
        -------
        fsa_kernel : np.array[float]
            Kernel for flux surface average integral vs poloidal angle.
        fsa_denominator : float
            Denominator of flux surface average (integral of fsa_kernal
            over poloidal angle).
        """
        polar_radius2 = np.square(polar_radius)
        dradius_dtheta = np.gradient(polar_radius, theta)

        fsa_kernel = (
            # Jacobian for transformation to flux coordinates.
            (2 * np.pi * radius / grad_rho)
            # Jacobian for transformation from arclength to polar angle.
            * np.sqrt(polar_radius2 + np.square(dradius_dtheta))
        )
        fsa_denominator = integrate.trapezoid(fsa_kernel, x=theta)

        return fsa_kernel, fsa_denominator

    @staticmethod
    def _flux_surface_average(
        theta: FloatArray,
        fsa_kernel: FloatArray,
        quantity: FloatArray,
        fsa_denominator: float,
    ):
        """
        Flux surface average quantity over flux surface.

        Parameters
        ----------
        theta : np.array[float]
            Poloidal angle along flux surface.
        fsa_kernel : np.array[float]
            Kernel for flux surface average integral vs poloidal angle.
        quantity : np.array[float]
            Quantity to flux surface average.
        fsa_denominator : float
            Denominator of flux surface average (integral of fsa_kernal
            over poloidal angle).

        Returns
        -------
        flux_surface_average : float
            Flux surface averaged value.
        """
        return (
            integrate.trapezoid(fsa_kernel * quantity, x=theta)
            / fsa_denominator
        )

    @staticmethod
    def _get_circulating_particle_fraction(
        theta: FloatArray,
        fsa_kernel: FloatArray,
        fsa_denominator: float,
        h: FloatArray,
        fsa_h2: float,
        /,
        *,
        n_lambda: int = 51,
    ) -> float:
        """
        Calculate neoclassical circulating particle fraction f_c.

        Parameters
        ----------
        theta : np.array[float]
            Poloidal angle along flux surface.
        fsa_kernel : np.array[float]
            Kernel for flux surface average integral vs poloidal angle.
        fsa_denominator : float
            Denominator of flux surface average (integral of fsa_kernal
            over poloidal angle).
        h : np.array[float]
            Magnetic field divided by maximum value on flux surface.
        fsa_h2 : float
            Flux surface averaged h**2.
        n_lambda : int
            Number of pitch parameter lambda values used in integral.

        Returns
        -------
        trapped_particle_fraction : float
            Trapped particle fraction.

        Notes
        -----
        f_c = 0.75 <h**2> int_0^1{l / <sqrt(1 - l * h)>} dl
        This formula has an integrable singularity for h = 1 and lambda = 1 so
        if h >= 0.999 then 1 is returned.
        """
        if np.min(h) > 0.999:  # noqa: PLR2004
            return 1.0

        _lambda = np.linspace(0, 1, n_lambda)
        integrand = np.empty_like(_lambda)

        for i, _l in enumerate(_lambda):
            integrand[i] = MagneticTokamak._flux_surface_average(
                theta,
                fsa_kernel,
                np.sqrt(np.clip(1 - h * _l, 0.0, None)),
                fsa_denominator,
            )

        return (
            0.75 * fsa_h2 * integrate.trapezoid(_lambda / integrand, _lambda)
        )

    @staticmethod
    def _flux_surface_integrals(
        theta: FloatArray,
        radius: FloatArray,
        polar_radius: FloatArray,
        grad_rho: FloatArray,
        magnetic_field_strength: FloatArray,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """
        Calculate flux surface averaged quantities.

        Parameters
        ----------
        theta : np.array[float]
            Poloidal angle along flux surface.
        radius : np.array[float]
            Cylindrical major radius along flux surface.
        polar_radius : np.array[float]
            Radius from magnetic axis along flux surface
        grad_rho : np.array[float]
            Magnitude of gradient in root normalised poloidal flux along
            flux surface.
        magnetic_field_strength : np.array[float]
            Magnetic field strength on flux surface.

        Returns
        -------
        cross_sectional_area_m2: np.array[float]
            Flux surface cross sectional area [m^2].
        volume_m3: np.array[float]
            Flux tube volume area [m^2].
        fsa_1_over_r: np.array[float]
            Flux surface averaged 1 / r [m^-1].
        fsa_1_over_r2: np.array[float]
            Flux surface averaged 1 / r**2 [m^-2].
        fsa_b: np.array[float]
            Flux surface averaged magnetic field strength [T].
        b_max: np.array[float]
            Maximum magnetic field strength on flux surface [T].
        trapped_particle_fraction: np.array[float]
            Trapped particle fraction.
        """
        # Calculate cross sectional area and volume.
        cross_sectional_area_m2, volume_m3 = (
            MagneticTokamak._calculate_area_volume(theta, radius, polar_radius)
        )

        # Flux surface averages.
        fsa_kernel, fsa_denominator = (
            MagneticTokamak._get_fsa_kernel_denominator(
                theta, radius, polar_radius, grad_rho
            )
        )

        # Geometric.
        fsa_1_over_r = MagneticTokamak._flux_surface_average(
            theta, fsa_kernel, 1 / radius, fsa_denominator
        )
        fsa_1_over_r2 = MagneticTokamak._flux_surface_average(
            theta, fsa_kernel, 1 / np.square(radius), fsa_denominator
        )

        # Magnetic.
        fsa_b = MagneticTokamak._flux_surface_average(
            theta, fsa_kernel, abs(magnetic_field_strength), fsa_denominator
        )
        b_max = np.max(abs(magnetic_field_strength))

        # Trapped particle fraction.
        h = abs(magnetic_field_strength / b_max)
        fsa_h2 = MagneticTokamak._flux_surface_average(
            theta, fsa_kernel, np.square(h), fsa_denominator
        )

        trapped_particle_fraction = 1 - (
            MagneticTokamak._get_circulating_particle_fraction(
                theta,
                fsa_kernel,
                fsa_denominator,
                h,
                fsa_h2,
            )
        )

        return (
            cross_sectional_area_m2,
            volume_m3,
            fsa_1_over_r,
            fsa_1_over_r2,
            fsa_b,
            b_max,
            trapped_particle_fraction,
        )

    @classmethod
    def calculate_flux_surface_integrals(
        cls,
        rho_poloidal_2d: Spline2D,
        magnetic_axis_m: tuple[float, float],
        rho_poloidal_to_toroidal_1d: Spline1D,
        rho_toroidal_to_poloidal_1d: Spline1D,
        total_poloidal_flux_wb: float,
        total_toroidal_flux_wb: float,
        f_toroidal_tm: Spline1D,
        /,
        *,
        scale_factor_magnetic_field: float = 1.0,
        boundary_rz: tuple[FloatArray, FloatArray] | None = None,
    ) -> "MagneticTokamak":
        """
        Calculate MagneticTokamak evaluating flux surface integrals.

        Parameters
        ----------
        rho_poloidal_2d : Spline2D
            Root normalised poloidal flux rho as a function of cylindrical
            (r, z).
        magnetic_axis_m : tuple[float, float]
            Cylindrical (r, z) position of magnetic axis [m].
        rho_poloidal_to_toroidal_1d : Spline1D
            Rho toroidal as a function of rho poloidal.
        rho_toroidal_to_poloidal_1d : Spline1D
            Rho poloidal as a function of rho toroidal.
        total_poloidal_flux_wb : float
            Total poloidal flux between magnetic axis and separatrix [Wb].
        total_toroidal_flux_wb : float
            Total toroidal flux between magnetic axis and separatrix [Wb].
        f_toroidal_tm : Spline1D
            Diamagnetic function F vs radial flux coordinate [T.m].
        scale_factor_magnetic_field : float, optional
            Scale factor for magnetic field vector.
        boundary_rz : tuple[np.array[float], np.array[float]], optional
            Cylindrical (r, z) coordinates of separatrix.

        Returns
        -------
        magnetic_tokamak : MagneticTokamak
            Magnetic field model for tokamak.
        """
        # Need partial init to get magnetic field for flux surface averages.
        obj = object.__new__(cls)
        obj.set_magnetic_field(
            rho_poloidal_2d,
            magnetic_axis_m,
            rho_poloidal_to_toroidal_1d,
            rho_toroidal_to_poloidal_1d,
            total_poloidal_flux_wb,
            total_toroidal_flux_wb,
            f_toroidal_tm,
            scale_factor_magnetic_field=scale_factor_magnetic_field,
            boundary_rz=boundary_rz,
        )

        isocontours_rz = obj.rho_poloidal_coordinate.isocontours_rz

        _levels = np.empty(isocontours_rz.shape[0] + 1)
        _levels[0] = 0.0
        _levels[1:] = obj.rho_poloidal_coordinate.rho_1d

        _cross_sectional_area_m2 = np.empty_like(_levels)
        _volume_m3 = np.empty_like(_levels)
        _fsa_1_over_r = np.empty_like(_levels)
        _fsa_1_over_r2 = np.empty_like(_levels)
        _fsa_b = np.empty_like(_levels)
        _b_max = np.empty_like(_levels)
        _trapped_particle_fraction = np.empty_like(_levels)

        # Analytic values at zero.
        r0, z0 = obj.magnetic_axis_m
        b_magnetic_axis = obj.magnetic_field_t.value([r0, 0.0, z0])
        b_strength_magnetic_axis = np.sqrt(
            np.square(b_magnetic_axis[0])
            + np.square(r0 * b_magnetic_axis[1])
            + np.square(b_magnetic_axis[2])
        )

        _cross_sectional_area_m2[0] = 0.0
        _volume_m3[0] = 0.0
        _fsa_1_over_r[0] = 1 / r0
        _fsa_1_over_r2[0] = 1 / (r0 * r0)
        _fsa_b[0] = b_strength_magnetic_axis
        _b_max[0] = b_strength_magnetic_axis
        _trapped_particle_fraction[0] = 0.0

        # Calculate integrals.
        theta = obj.rho_poloidal_coordinate.theta_1d
        position_cylindrical = np.zeros((theta.size, 3))

        for i, rz in enumerate(isocontours_rz, start=1):
            r, z = rz[:, 0], rz[:, 1]
            polar_radius2 = np.square(r - r0) + np.square(z - z0)
            polar_radius = np.sqrt(polar_radius2)

            position_cylindrical[:, 0] = r
            position_cylindrical[:, 2] = z

            rho_jacobian = obj.rho_poloidal_coordinate.rho_spline.jacobian(
                position_cylindrical
            )

            grad_rho = np.sqrt(
                np.square(rho_jacobian[:, 0]) + np.square(rho_jacobian[:, 2])
            )

            magnetic_field_t = obj.magnetic_field_t.value(position_cylindrical)
            magnetic_field_strength = np.sqrt(
                np.square(magnetic_field_t[:, 0])
                + np.square(r * magnetic_field_t[:, 1])
                + np.square(magnetic_field_t[:, 2])
            )

            (
                _cross_sectional_area_m2[i],
                _volume_m3[i],
                _fsa_1_over_r[i],
                _fsa_1_over_r2[i],
                _fsa_b[i],
                _b_max[i],
                _trapped_particle_fraction[i],
            ) = cls._flux_surface_integrals(
                theta, r, polar_radius, grad_rho, magnetic_field_strength
            )

        obj.construct_flux_function_splines(
            _levels,
            _cross_sectional_area_m2,
            _volume_m3,
            _fsa_1_over_r,
            _fsa_1_over_r2,
            _fsa_b,
            _b_max,
            _trapped_particle_fraction,
        )

        return obj

    @staticmethod
    def imas_load_f_vacuum(ids_equilibrium, time_index: int) -> float:
        """
        Load magnetic axis location from IMAS.

        Parameters
        ----------
        ids_equilibrium
            IDS equilibrium.
        time_index : int
            Index of time slice in IDS.

        Returns
        -------
        r_mag, z_mag : tuple[float, float]
            Magnetic axis location [m].

        Raises
        ------
        IDSEmptyError
            r0 or b0 not set in IDS equilibrum.
        """
        r0 = ids_equilibrium.vacuum_toroidal_field.r0
        b0 = ids_equilibrium.vacuum_toroidal_field.b0[time_index]

        if ids_empty(r0):
            raise IDSEmptyError(
                "In IDS equilibrium.vacuum_toroidal_field.r0 not set."
            )

        if ids_empty(b0):
            raise IDSEmptyError(
                "In IDS equilibrium.vacuum_toroidal_field.b0 not set."
            )

        return r0 * b0

    @staticmethod
    def imas_load_flux_surface_integrals(
        profiles_1d,
    ) -> tuple[
        bool,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """
        Load flux surface integrals from IMAS.

        Parameters
        ----------
        profiles_1d
            Time slice from IDS equilibrium.profiles_1d

        Returns
        -------
        success : bool
            If add required flux surface integrals are in profiles_1d.
        cross_sectional_area_m2
            Flux surface cross sectional area [m^2].
        volume_m3
            Flux tube volume [m^3].
        fsa_1_over_r
            Flux surface averaged 1 / r [m^-1].
        fsa_1_over_r2
            Flux surface averaged 1 / r**2 [m^-2].
        fsa_b
            Flux surface averaged magnetic field strength [T].
        b_max
            Maximum magnetic field strength on flux surface [T].
        trapped_particle_fraction
            Trapped particle fraction.
        """
        success = True

        # Cross sectional area.
        cross_sectional_area_m2 = profiles_1d.area

        if ids_empty(cross_sectional_area_m2):
            success = False
            logger.warning("In IDS equilibrium.profiles_1d.area not set.")

        # Flux tube volume.
        volume_m3 = profiles_1d.volume

        if ids_empty(volume_m3):
            success = False
            logger.warning("In IDS equilibrium.profiles_1d.volume not set.")

        # Flux surface averaged 1 / R.
        fsa_1_over_r = profiles_1d.gm9

        if ids_empty(fsa_1_over_r):
            success = False
            logger.warning(
                "In IDS equilibrium.profiles_1d.gm9 "
                "(flux surface averaged 1 / R) not set."
            )

        # Flux surface averaged 1 / R^2.
        fsa_1_over_r2 = profiles_1d.gm1

        if ids_empty(fsa_1_over_r2):
            success = False
            logger.warning(
                "In IDS equilibrium.profiles_1d.gm1 "
                "(flux surface averaged 1 / R^2) not set."
            )

        # Flux surface averaged |B|.
        fsa_b = profiles_1d.b_field_average

        if ids_empty(fsa_b):
            success = False
            logger.warning(
                "In IDS equilibrium.profiles_1d.b_field_average not set."
            )

        # Maximum |B| on flux surface.
        b_max = profiles_1d.b_field_max

        if ids_empty(b_max):
            success = False
            logger.warning(
                "In IDS equilibrium.profiles_1d.b_field_max not set."
            )

        # Fraction of trapped particles on flux surface.
        trapped_particle_fraction = profiles_1d.trapped_fraction

        if ids_empty(trapped_particle_fraction):
            success = False
            logger.warning(
                "In IDS equilibrium.profiles_1d.trapped_fraction not set."
            )

        return (
            success,
            cross_sectional_area_m2,
            volume_m3,
            fsa_1_over_r,
            fsa_1_over_r2,
            fsa_b,
            b_max,
            trapped_particle_fraction,
        )

    @classmethod
    def from_imas(
        cls, ids_equilibrium, time_index: int, scale_factor: float
    ) -> "MagneticTokamak":
        """
        Calculate MagneticTokamak from IMAS data.

        Parameters
        ----------
        ids_equilibrium
            IMAS IDS equilibrium.
        time_index : int
            Index of timeslice to load from equilibrium.
        scale_factor : float
            Scale factor for magnetic field vector.

        Returns
        -------
        magnetic_tokamak : MagneticTokamak
            Magnetic field model for tokamak.

        Raises
        ------
        IDSEmptyError
            No poloidal flux vs (r, z) in equilibrium_ids.profiles_2d.
            Grid dimensions or poloidal flux data missing.

        Notes
        -----
        If flux surface averaged values are available in the IMAS database
        they will be used, otherwise they are calculated.
        """
        # Find 2D poloidal flux grid against (r, z).
        time_slice = ids_equilibrium.time_slice[time_index]

        found_profiles_2d = False

        for _profiles_2d in time_slice.profiles_2d:
            if _profiles_2d.grid_type.index == 1:
                found_profiles_2d = True

        if not found_profiles_2d:
            raise IDSEmptyError(
                "Cannot find poloidal flux on r, z grid in "
                "equilibrium.profiles_2d IDS."
            )

        r = _profiles_2d.grid.dim1
        z = _profiles_2d.grid.dim2
        psi = _profiles_2d.psi

        if ids_empty(r):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.grid.dim1 not set."
            )

        if ids_empty(z):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_2d.grid.dim2 not set."
            )

        if ids_empty(r):
            raise IDSEmptyError("In IDS equilibrium.profiles_2d.psi not set.")

        # Load poloidal flux at boundaries.
        global_quantities = ids_equilibrium.time_slice[
            time_index
        ].global_quantities

        psi_axis = global_quantities.psi_axis
        psi_separatrix = global_quantities.psi_boundary
        total_poloidal_flux_wb = psi_separatrix - psi_axis

        if ids_empty(psi_axis):
            raise IDSEmptyError(
                "In IDS equilibrium.global_quantities.psi_magnetic_axis "
                "not set."
            )

        if ids_empty(psi_separatrix):
            raise IDSEmptyError(
                "In IDS equilibrium.global_quantities.psi_boundary not set."
            )

        if np.isclose(total_poloidal_flux_wb, 0.0):
            raise IDSEmptyError(
                "In IDS equilibrium.global_quantities.psi_magnetic_axis "
                "and equilibrium.global_quantities.psi_boundary are equal "
            )

        # Load magnetic axis.
        r_axis = time_slice.global_quantities.magnetic_axis.r
        z_axis = time_slice.global_quantities.magnetic_axis.z

        if ids_empty(r_axis) or ids_empty(z_axis):
            logger.warning(
                "IDS equilibrium.global_quantities.magnetic_axis "
                "is not set. Magnetic axis location will be approximated."
            )

            idx_r, idx_y = np.unravel_index(
                np.argmin(abs(psi - psi_axis)), psi.shape
            )
            magnetic_axis_m = (r[idx_r], z[idx_y])
        else:
            magnetic_axis_m = (r_axis, z_axis)

        # Create rho poloidal spline.
        rho_poloidal = np.sqrt(
            np.clip((psi - psi_axis) / total_poloidal_flux_wb, 0.0, np.inf)
        )

        rho_poloidal_spline_2d = (
            ValueModel.root_normalised_poloidal_flux().spline_2d(
                CoordinateSystem.CYLINDRICAL,
                r,
                z,
                rho_poloidal,
                (True, False, True),
                scale_factor=1.0,
                method=SplineMethod.QUINTIC,
            )
        )

        # Generate mapping between rho toroidal and rho poloidal.
        _profiles_1d = ids_equilibrium.time_slice[time_index].profiles_1d

        psi_norm_poloidal = _profiles_1d.psi_norm
        rho_toroidal = _profiles_1d.rho_tor_norm
        toroidal_flux_wb = _profiles_1d.phi

        if ids_empty(psi_norm_poloidal):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_1d.psi_norm not set."
            )

        rho_poloidal = np.sqrt(np.clip(psi_norm_poloidal, 0.0, np.inf))

        if ids_empty(rho_toroidal) or ids_empty(toroidal_flux_wb):
            # Toroidal flux coordinates not provided.
            safety_factor = _profiles_1d.q

            if ids_empty(safety_factor):
                raise IDSEmptyError(
                    "In IDS equilibrium.profiles_1d.rho_tor_norm and/or "
                    "equilibrium.profiles_1d.phi not set and "
                    "equilibrium.profiles_1d.q not set."
                )

            toroidal_flux_wb = (
                total_poloidal_flux_wb
                * integrate.cumulative_trapezoid(
                    safety_factor, x=psi_norm_poloidal, initial=0.0
                )
            )

            total_toroidal_flux_wb = toroidal_flux_wb[-1]

            if np.isclose(total_toroidal_flux_wb, 0.0):
                raise IDSEmptyError(
                    "Total calculated toroidal flux is zero. "
                    "Likely q profile is zero."
                )

            rho_toroidal = np.sqrt(
                np.clip(toroidal_flux_wb / total_toroidal_flux_wb, 0.0, np.inf)
            )
        else:
            total_toroidal_flux_wb = toroidal_flux_wb[-1]

        rho_poloidal_to_toroidal_1d = (
            ValueModel.root_normalised_poloidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal,
                rho_toroidal,
                (True,),
                scale_factor=1.0,
                method=SplineMethod.QUINTIC,
            )
        )

        rho_toroidal_to_poloidal_1d = (
            ValueModel.root_normalised_toroidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_TOROIDAL,
                rho_toroidal,
                rho_poloidal,
                (True,),
                scale_factor=1.0,
                method=SplineMethod.QUINTIC,
            )
        )

        # Load vacuum diamagnetic function F.
        f0 = MagneticTokamak.imas_load_f_vacuum(ids_equilibrium, time_index)

        # Load flux function poloidal flux grid.
        psi_norm_poloidal = _profiles_1d.psi_norm

        if ids_empty(psi_norm_poloidal):
            raise IDSEmptyError(
                "In IDS equilibrium.profiles_1d.psi_norm not set."
            )

        rho_poloidal_1d = np.sqrt(np.clip(psi_norm_poloidal, 0.0, np.inf))

        # Load F toroidal.
        _f_toroidal_tm = _profiles_1d.f

        if ids_empty(_f_toroidal_tm):
            raise IDSEmptyError("In IDS equilibrium.profiles_1d, f not set.")

        # Force f[-1] equal to R0 * B0. Should be anyway.
        _f_toroidal_tm[-1] = f0

        f_toroidal_tm = ValueModel.f_toroidal_tm().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal_1d,
            _f_toroidal_tm,
            (True,),
        )

        # Load boundary contour.
        _boundary = time_slice.boundary
        boundary_rz = (_boundary.outline.r, _boundary.outline.z)

        if ids_empty(boundary_rz[0]) or ids_empty(boundary_rz[1]):
            boundary_rz = None

        # Load flux surface integrals from IMAS.
        (
            _success,
            _cross_sectional_area_m2,
            _volume_m3,
            _fsa_1_over_r,
            _fsa_1_over_r2,
            _fsa_b,
            _b_max,
            _trapped_particle_fraction,
        ) = MagneticTokamak.imas_load_flux_surface_integrals(_profiles_1d)

        if _success:
            obj = object.__new__(cls)

            obj.set_magnetic_field(
                rho_poloidal_spline_2d,
                magnetic_axis_m,
                rho_poloidal_to_toroidal_1d,
                rho_toroidal_to_poloidal_1d,
                total_poloidal_flux_wb,
                total_toroidal_flux_wb,
                f_toroidal_tm,
                scale_factor_magnetic_field=scale_factor,
                boundary_rz=boundary_rz,
            )

            obj.construct_flux_function_splines(
                rho_poloidal_1d,
                _cross_sectional_area_m2,
                _volume_m3,
                _fsa_1_over_r,
                _fsa_1_over_r2,
                _fsa_b,
                _b_max,
                _trapped_particle_fraction,
            )

            return obj

        # Calculate flux surface integrals.
        return cls.calculate_flux_surface_integrals(
            rho_poloidal_spline_2d,
            magnetic_axis_m,
            rho_poloidal_to_toroidal_1d,
            rho_toroidal_to_poloidal_1d,
            total_poloidal_flux_wb,
            total_toroidal_flux_wb,
            f_toroidal_tm,
            scale_factor_magnetic_field=scale_factor,
            boundary_rz=boundary_rz,
        )

    def register_flux_coordinates(
        self, coordinate_coordinator: CoordinateCoordinator
    ):
        """
        Register flux coordinate systems.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Object holding coordinate system data.
        """
        coordinate_coordinator.register_coordinate(
            self.rho_poloidal_coordinate
        )
        coordinate_coordinator.register_coordinate(
            self.rho_toroidal_coordinate
        )

    @classmethod
    def read_netcdf(
        cls,
        dset: nc4.Group,
        rho_poloidal_2d: Spline2D,
        magnetic_axis_m: tuple[float, float],
        rho_poloidal_to_toroidal_1d: Spline1D,
        rho_toroidal_to_poloidal_1d: Spline1D,
    ):
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to read data from.
        rho_poloidal_2d : Spline2D
            Root normalised poloidal flux rho as a function of cylindrical
            (r, z).
        magnetic_axis_m : tuple[float, float]
            Cylindrical (r, z) position of magnetic axis [m].
        rho_poloidal_to_toroidal_1d : Spline1D
            Rho toroidal as a function of rho poloidal.
        rho_toroidal_to_poloidal_1d : Spline1D
            Rho poloidal as a function of rho toroidal.

        Returns
        -------
        magnetic_tokamak : MagneticTokamak
            Magnetic field model for tokamak.
        """
        # Rho poloidal and toroidal splines are available from coordinate.
        total_poloidal_flux_wb = dset.getncattr("total_poloidal_flux_wb")
        total_toroidal_flux_wb = dset.getncattr("total_toroidal_flux_wb")
        scale_factor = dset.getncattr("scale_factor")

        f_toroidal_tm = Spline1D.read_netcdf(dset["f_toroidal"])
        cross_sectional_area_m2 = Spline1D.read_netcdf(
            dset["cross_sectional_area"]
        )
        volume_m3 = Spline1D.read_netcdf(dset["volume"])
        fsa_1_over_r = Spline1D.read_netcdf(dset["fsa_1_over_r"])
        fsa_1_over_r2 = Spline1D.read_netcdf(dset["fsa_1_over_r2"])
        fsa_b = Spline1D.read_netcdf(dset["fsa_b"])
        b_max = Spline1D.read_netcdf(dset["b_max"])
        trapped_particle_fraction = Spline1D.read_netcdf(
            dset["trapped_particle_fraction"]
        )

        return cls(
            rho_poloidal_2d,
            magnetic_axis_m,
            rho_poloidal_to_toroidal_1d,
            rho_toroidal_to_poloidal_1d,
            total_poloidal_flux_wb,
            total_toroidal_flux_wb,
            f_toroidal_tm,
            cross_sectional_area_m2,
            volume_m3,
            fsa_1_over_r,
            fsa_1_over_r2,
            fsa_b,
            b_max,
            trapped_particle_fraction,
            scale_factor_magnetic_field=scale_factor,
        )

    def write_netcdf(self, dset: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        dset.setncattr("topology", self.topology)

        # Don't write rho poloidal and toroidal as they are saved as part
        # of the coordinate coordinator.
        dset.setncattr("total_poloidal_flux_wb", self.total_poloidal_flux_wb)
        dset.setncattr("total_toroidal_flux_wb", self.total_toroidal_flux_wb)
        dset.setncattr("scale_factor", self.magnetic_field_t.scale_factor)

        self.f_toroidal_tm.write_netcdf(dset.createGroup("f_toroidal"))
        self.cross_sectional_area_m2.write_netcdf(
            dset.createGroup("cross_sectional_area")
        )
        self.volume_m3.write_netcdf(dset.createGroup("volume"))
        self.fsa_1_over_r.write_netcdf(dset.createGroup("fsa_1_over_r"))
        self.fsa_1_over_r2.write_netcdf(dset.createGroup("fsa_1_over_r2"))
        self.fsa_b.write_netcdf(dset.createGroup("fsa_b"))
        self.b_max.write_netcdf(dset.createGroup("b_max"))
        self.trapped_particle_fraction.write_netcdf(
            dset.createGroup("trapped_particle_fraction")
        )


class MagneticStellarator(Magnetic):
    """
    Stellarator magnetic field data model.

    Attributes
    ----------
    rho_toroidal_3d : Spline3D
        Root normalised toroidal flux rho_t(r, phi, z) as a function of
        cylindrical position.

    Methods
    -------
    from_vmec
        Create stellarator magnetic field model from VMEC output.
    register_flux_coordinates
        Register flux coordinate systems.
    """

    __slots__ = ("rho_toroidal_3d",)

    topology = STELLARATOR

    def __init__(self):
        """
        Inits MagneticStellarator.
        """
        raise NotImplementedError

    @classmethod
    def from_vmec(cls, fh):
        """
        Create stellarator magnetic field model from VMEC output.

        Parameters
        ----------
        fh : TextIO
            File handle to read data from.
        """
        raise NotImplementedError

    def register_flux_coordinates(
        self, coordinate_coordinator: CoordinateCoordinator
    ):
        """
        Register flux coordinate systems.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Object holding coordinate system data.
        """
        raise NotImplementedError


class Limiters(IONetcdf):
    """
    Collection of limiter models.

    Attributes
    ----------
    limiters : dict[str, LimiterType]
        Dictionary of all limiters.

    Methods
    -------
    set_element_idx
        Set unique index on each limiter element.
    intersects
        Determine if line connecting point_1 to point_2 any elements.
    """

    __slots__ = ("limiters",)

    def __init__(self, limiters: dict[str, Limiter]):
        """
        Inits Limiters.

        Parameters
        ----------
        limiters : dict[str, LimiterType]
            Dictionary of all limiters.
        """
        self.limiters = limiters

    def intersects(
        self,
        ray_origin: FloatArray,
        ray_direction: FloatArray,
        /,
        *,
        ignore: tuple[str, int] = ("", -1),
    ) -> tuple[bool, str, int, float, FloatArray, LimiterEffect, float]:
        """
        Determine if line connecting point_1 to point_2 any elements.

        Parameters
        ----------
        ray_origin : np.array[float]
            Starting point of ray.
        ray_direction : np.array[float]
            Direction of ray.
        ignore : tuple[str, int], optinonal
            If provided, ignore intersections on named limiter at given index.

        Returns
        -------
        intersects : bool
            Flag if the element is intersected.
        name : str
            Name of intersected limiter.
        idx : int
            Index of intersected element.
        s: float
            Normalised distance along line between point_1 and point_2 the
            intersection occurs.
        normal : array
            2D vector giving normal vector for applying reflections.
        effect : LimiterEffect
            Effect applied to ray on intersection.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].
        """
        intersects = False
        name = ""
        idx = -1
        s = np.inf
        normal = np.empty(3)
        effect = LimiterEffect.STOP
        extinction_coefficient_nepers = 0.0

        for _name, limiter in self.limiters.items():
            _intersects, _idx, _s, _normal, _effect, _extinction = (
                limiter.intersects(
                    ray_origin,
                    ray_direction,
                    ignore_idx=ignore[1] if _name == ignore[0] else None,
                )
            )

            if _intersects and _s < s:
                intersects = True
                name = _name
                idx = _idx
                s = _s
                normal[:] = _normal
                effect = _effect
                extinction_coefficient_nepers = _extinction

        return (
            intersects,
            name,
            idx,
            s,
            normal,
            effect,
            extinction_coefficient_nepers,
        )

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        for name, limiter in self.limiters.items():
            group = dset.createGroup(name)
            group.setncattr("limiter_type", limiter.__class__.__name__)
            limiter.write_netcdf(group)

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Limiters":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        limiters : Limiters
            Collection of limiters.
        """
        limiters = {}

        for name, g in group.groups.items():
            limiters[name] = Limiter.read_netcdf(g)

        return cls(limiters)

    def add_limiter(self, name: str, limiter: Limiter):
        """
        Add limiter to collection.

        Parameters
        ----------
        name : str
            Name of limiter.
        limiter : LimiterType
            Collection of limiter elements.

        Raises
        ------
        ValueError
            Limiter already exists with provided name.
        """
        if name in self.limiters:
            raise ValueError(f"Limiter already exists with name '{name}'")

        self.limiters[name] = limiter

    def add_from_imas_bounding_box(
        self,
        name: str,
        ids_equilibrium,
        time_index: int,
        effect: LimiterEffect,
        extinction_coefficient_nepers: float,
    ):
        """
        Create cylindrical (r, z) bounding box from magnetic equilibrium data.

        Parameters
        ----------
        name : str
            Limiter name.
        ids_equilibrium
            IMAS IDS equilibrium.
        time_index : int
            Index of timeslice to load from equilibrium.
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        IDSEmptyError
            IDS equilibrium is empty.
        """
        time_slice = ids_equilibrium.time_slice[time_index]

        for _profiles_2d in time_slice.profiles_2d:
            if _profiles_2d.grid_type.index == 1:
                found_profiles_2d = True

        if not found_profiles_2d:
            raise IDSEmptyError(
                "Cannot find poloidal flux on r, z grid in "
                "equilibrium.profiles_2d IDS."
            )

        r = _profiles_2d.grid.dim1
        z = _profiles_2d.grid.dim2
        r0, r1, z0, z1 = r[0], r[-1], z[0], z[-1]

        r_min, r_max = min(r0, r1), max(r0, r1)
        z_min, z_max = min(z0, z1), max(z0, z1)

        elements = [
            Disk(z_min, r_min, r_max, effect),
            Cylinder(r_min, z_min, z_max, effect),
            Disk(z_max, r_min, r_max, effect),
            Cylinder(r_max, z_min, z_max, effect),
        ]

        limiter = Limiter(elements, extinction_coefficient_nepers)
        self.add_limiter(name, limiter)

    def add_from_imas_2d(
        self,
        name: str,
        ids_wall,
        effect: LimiterEffect,
        extinction_coefficient_nepers: float,
    ):
        """
        Load cylindrical (r, z) limiter from IDS wall.description_2d.limiter

        Parameters
        ----------
        name : str
            Limiter name.
        ids_wall
            IMAS IDS wall.
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        IDSEmptyError
            IDS wall missing data.
        """
        elements = []

        for i, _description_2d in enumerate(ids_wall.description_2d):
            limiter_units = _description_2d.limiter.unit

            if len(limiter_units) == 0:
                raise IDSEmptyError(
                    f"In wall IDS, description_2d[{i}].limiter.units is empty."
                )

            elements = []
            for j, _unit in enumerate(limiter_units):
                r = _unit.outline.r
                z = _unit.outline.z

                if ids_empty(r):
                    raise IDSEmptyError(
                        f"In wall IDS, description_2d[{i}].limiter "
                        f".units[{j}].r is empty."
                    )

                if ids_empty(z):
                    raise IDSEmptyError(
                        f"In wall IDS, description_2d[{i}].limiter "
                        f".units[{j}].z is empty."
                    )

                elements.extend(
                    CappedCone.from_rz(
                        (r[k], z[k]),
                        (r[k + 1] - r[k], z[k + 1] - z[k]),
                        effect,
                    )
                    for k in range(r.size - 1)
                )

        limiter = Limiter(elements, extinction_coefficient_nepers)
        self.add_limiter(name, limiter)

    def add_from_netcdf(
        self,
        name: str,
        group: nc4.Group,
    ):
        """
        Add limiter from netCDF4 file.

        Parameters
        ----------
        name : str
            Name of limiter.
        group : netCDF4.Dataset
            netCDF4 dataset or group to read data from.
        """
        limiter = Limiter.read_netcdf(group)
        self.add_limiter(name, limiter)

    def add_bounding_box_2d(
        self,
        name: str,
        coordinate: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        effect: LimiterEffect,
        extinction_coefficient_nepers: float,
    ):
        """
        Add 2D bounding box limiter.

        Parameters
        ----------
        name : str
            Name of limiter.
        coordinate : literal['xy', 'xz', 'yz', 'rz']
            Coordinate pair being used.
        x_limits : tuple[float, float]
            Minimum and maximum x coordinate value.
        y_limits : tuple[float, float]
            Minimum and maximum y coordinate value.
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        ValueError
            Unknown coordinate.
        """
        if coordinate == LimiterAnalyticBoundingBox2D.RZ:
            r0, r1 = x_limits
            z0, z1 = y_limits

            r_min, r_max = min(r0, r1), max(r0, r1)
            z_min, z_max = min(z0, z1), max(z0, z1)

            elements = [
                Disk(z_min, r_min, r_max, effect),
                Cylinder(r_min, z_min, z_max, effect),
                Disk(z_max, r_min, r_max, effect),
                Cylinder(r_max, z_min, z_max, effect),
            ]

        elif coordinate in {
            LimiterAnalyticBoundingBox2D.XY,
            LimiterAnalyticBoundingBox2D.XZ,
            LimiterAnalyticBoundingBox2D.YZ,
        }:
            x0, x1 = x_limits
            y0, y1 = y_limits

            if coordinate == LimiterAnalyticBoundingBox2D.XY:
                element = Plane.xy
            elif coordinate == LimiterAnalyticBoundingBox2D.XZ:
                element = Plane.xz
            elif coordinate == LimiterAnalyticBoundingBox2D.YZ:
                element = Plane.yz
            else:
                raise ValueError(coordinate)

            elements = [
                element((x0, y0), (x1 - x0, 0.0), effect),
                element((x1, y0), (0.0, y1 - y0), effect),
                element((x1, y1), (x0 - x1, 0.0), effect),
                element((x0, y1), (0.0, y0 - y1), effect),
            ]

        else:
            raise ValueError(coordinate)

        limiter = Limiter(elements, extinction_coefficient_nepers)
        self.add_limiter(name, limiter)

    def add_bounding_box_3d(
        self,
        name: str,
        coordinate: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        z_limits: tuple[float, float],
        effect: LimiterEffect,
        extinction_coefficient_nepers: float,
    ):
        """
        Add 3D bounding box limiter.

        Parameters
        ----------
        name : str
            Name of limiter.
        coordinate : literal['xy', 'xz', 'yz', 'rz']
            Coordinate pair being used.
        x_limits : tuple[float, float]
            Minimum and maximum x coordinate value.
        y_limits : tuple[float, float]
            Minimum and maximum y coordinate value.
        z_limits : tuple[float, float]
            Minimum and maximum z coordinate value.
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        ValueError
            Unknown coordinate.
        """
        if coordinate == LimiterAnalyticBoundingBox3D.XYZ:
            x0, x1 = x_limits
            y0, y1 = y_limits
            z0, z1 = z_limits

            elements = [
                Plane.xyz(
                    (x0, y0, z0),
                    (x1 - x0, 0.0, 0.0),
                    (0.0, y1 - y0, 0.0),
                    effect,
                ),
                Plane.xyz(
                    (x0, y0, z0),
                    (x1 - x0, 0.0, 0.0),
                    (0.0, 0.0, z1 - z0),
                    effect,
                ),
                Plane.xyz(
                    (x0, y0, z0),
                    (0.0, y1 - y0, 0.0),
                    (0.0, 0.0, z1 - z0),
                    effect,
                ),
                Plane.xyz(
                    (x1, y1, z1),
                    (x0 - x1, 0.0, 0.0),
                    (0.0, y0 - y1, 0.0),
                    effect,
                ),
                Plane.xyz(
                    (x1, y1, z1),
                    (x0 - x1, 0.0, 0.0),
                    (0.0, 0.0, z0 - z1),
                    effect,
                ),
                Plane.xyz(
                    (x1, y1, z1),
                    (0.0, y0 - y1, 0.0),
                    (0.0, 0.0, z0 - z1),
                    effect,
                ),
            ]
        else:
            raise ValueError(coordinate)

        limiter = Limiter(elements, extinction_coefficient_nepers)
        self.add_limiter(name, limiter)


class SystemData(IONetcdf):
    """
    Object providing information about plasma system.

    Attributes
    ----------
    coordinate_coordinator : CoordinateCoordinator
        Object holding coordinate system data.
    kinetic : Kinetic
        Kinetic plasma parameter models.
    limiters : Limiters
        Collection of limiters.
    magnetic : Magnetic
        Magnetic field data model.

    Methods
    -------
    read_netcdf
        Load from netCDF4 dataset.
    write_netcdf
        Write to netCDF4 dataset.
    """

    __slots__ = ("coordinate_coordinator", "kinetic", "limiters", "magnetic")

    def __init__(
        self,
        coordinate_coordinator: CoordinateCoordinator,
        kinetic: Kinetic,
        magnetic: Magnetic,
        limiters: Limiters,
    ):
        """
        Inits SystemData.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Object holding coordinate system data.
        kinetic : Kinetic
            Kinetic plasma parameter models.
        magnetic : Magnetic
            Magnetic field data model.
        limiters : Limiters
            Collection of limiters.

        Raises
        ------
        ValueError
            coordinate_coordinator does not hold coordinate systems required
            for all models.
        """
        self.coordinate_coordinator = coordinate_coordinator
        self.kinetic = kinetic
        self.magnetic = magnetic
        self.limiters = limiters

        # Check there is a coordinate system defined for each model.
        all_coordinates = self.coordinate_coordinator.coordinates
        for model in (
            self.kinetic.electron_density_per_m3,
            self.kinetic.electron_temperature_ev,
            self.kinetic.effective_charge,
        ):
            if model.coordinate_system not in all_coordinates:
                raise ValueError(
                    "Missing coordinate system: "
                    f"{model.coordinate_system.name}"
                )

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "SystemData":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to read data from.

        Returns
        -------
        system_data : SystemData
            Object providing information about plasma system.

        Raises
        ------
        ValueError
            Unknown magnetic field topology.
        """
        # Load coordinate coordinator.
        dset_coordinate = dset["coordinate_coordinator"]
        coordinate_coordinator = CoordinateCoordinator.read_netcdf(
            dset_coordinate
        )

        # Load kinetic.
        dset_kinetic = dset["kinetic"]
        kinetic = Kinetic.read_netcdf(dset_kinetic)

        # Load magnetic.
        dset_magnetic = dset["magnetic"]
        magnetic_topology = dset_magnetic.getncattr("topology")

        if magnetic_topology == SIMPLE:
            magnetic = Magnetic.read_netcdf(dset_magnetic)
        elif magnetic_topology == TOKAMAK:
            rho_poloidal_coordinate = coordinate_coordinator.coordinates[
                CoordinateSystem.RHO_POLOIDAL
            ]
            rho_toroidal_coordinate = coordinate_coordinator.coordinates[
                CoordinateSystem.RHO_TOROIDAL
            ]

            magnetic = MagneticTokamak.read_netcdf(
                dset_magnetic,
                rho_poloidal_coordinate.rho_spline,
                rho_poloidal_coordinate.magnetic_axis_m,
                rho_toroidal_coordinate.rho_spline_1_to_2,
                rho_toroidal_coordinate.rho_spline_2_to_1,
            )
            magnetic.register_flux_coordinates(coordinate_coordinator)
        elif magnetic_topology == STELLARATOR:
            magnetic = MagneticStellarator.read_netcdf(dset_magnetic)
            magnetic.register_flux_coordinates(coordinate_coordinator)
        else:
            raise ValueError(
                f"Unknown magnetic field type '{magnetic_topology}'"
            )

        # Calculate coordinate conversion paths.
        coordinate_coordinator.calculate_conversion_paths()

        # Load limiters.
        dset_limiters = dset["limiters"]
        limiters = Limiters.read_netcdf(dset_limiters)

        return cls(coordinate_coordinator, kinetic, magnetic, limiters)

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        # Write coordinate coordinator.
        group_coordinate = dset.createGroup("coordinate_coordinator")
        self.coordinate_coordinator.write_netcdf(group_coordinate)

        # Write kinetic.
        group_kinetic = dset.createGroup("kinetic")
        self.kinetic.write_netcdf(group_kinetic)

        # Write magnetic.
        group_magnetic = dset.createGroup("magnetic")
        self.magnetic.write_netcdf(group_magnetic)

        # Write limiters.
        group_limiters = dset.createGroup("limiters")
        self.limiters.write_netcdf(group_limiters)


class SystemDataProvider(IOToml):
    """
    Object which constructs objects containing information about
    plasma system data.

    Attributes
    ----------
    coordinates : dict[CoordinateSystem, CoordinateType]
        Coordinate system definition.
    data_sources : dict[str, DataSourceType]
        Data sources definition.
    effective_charge : KineticModelType
        Effective charge model definition.
    electron_density_per_m3 : KineticModelType
        Electron density model definition.
    electron_temperature_ev : KineticModelType
        Electron temperature model definition.
    limiters : dict[str, LimiterSchemaType]
        Limiters model definition.
    magnetic_field_t : MagneticModelType
        Magnetic field model definition.

    Methods
    -------
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    from_dict_toml
        Create object from dictionary of data read from toml file.
    open_data_sources
        Open all data sources.
    close_data_sources
        Close all data sources.
    get_coordinate_coordinator
        Construct coordinate coordinator.
    get_analytic_model
        Construct analytic plasma parameter model.
    get_kinetic_model
        Construct kinetic plasma parameter model.
    get_kinetic
        Construct all kinetic parameter models.
    get_magnetic
        Construct magnetic parameter models.
    get_limiters
        Construct limiter models.
    build
        Generate system data object from sources at given time.
    """

    __slots__ = (
        "_data_source_handles",
        "coordinates",
        "data_sources",
        "effective_charge",
        "electron_density_per_m3",
        "electron_temperature_ev",
        "limiters",
        "magnetic_field_t",
    )

    def __init__(
        self,
        data_sources: dict[str, DataSourceType],
        coordinates: dict[str, CoordinateType],
        electron_density_per_m3: KineticModelType,
        electron_temperature_ev: KineticModelType,
        effective_charge: KineticModelType,
        magnetic_field_t: MagneticModelType,
        limiters: dict[str, LimiterSchemaType],
    ):
        """
        Inits SystemDataProvider.

        Attributes
        ----------
        data_sources : dict[str, DataSourceType]
            Data sources definition.
        coordinates : dict[CoordinateSystem, CoordinateType]
            Coordinate system definition.
        electron_density_per_m3 : KineticModelType
            Electron density model definition.
        electron_temperature_ev : KineticModelType
            Electron temperature model definition.
        effective_charge : KineticModelType
            Effective charge model definition.
        magnetic_field_t : MagneticModelType
            Magnetic field model definition.
        limiters : dict[str, LimiterSchemaType]
            Limiters model definition.

        Raises
        ------
        ValueError
            Models use unknown data sources.
        """
        self.data_sources = data_sources
        self.coordinates = coordinates
        self.electron_density_per_m3 = electron_density_per_m3
        self.electron_temperature_ev = electron_temperature_ev
        self.effective_charge = effective_charge
        self.magnetic_field_t = magnetic_field_t
        self.limiters = limiters

        self._data_source_handles = {}

        # Check requested data sources exist.
        for name, model in (
            (electron_density_per_m3, "electron_density_per_m3"),
            (electron_temperature_ev, "electron_temperature_ev"),
            (effective_charge, "effective_charge"),
        ):
            if (
                type(model) in {ModelImas, ModelNetcdf}
                and model.source not in data_sources
            ):
                raise ValueError(
                    f"{KINETIC}.{name} uses undefined data source "
                    f"'{model.source}'"
                )

        # Don't use isinstance as it considers inheritance.
        if (
            type(magnetic_field_t)
            in {MagneticModelTokamak, MagneticModelStellarator}
            and magnetic_field_t.source not in data_sources
        ):
            raise ValueError(
                f"{MAGNETIC_FIELD_T} uses undefined data source "
                f"'{magnetic_field_t.source}'"
            )

        for name, model in limiters.items():
            if (
                type(model)
                in {
                    LimiterImasBoundingBox2D,
                    LimiterImas2D,
                    LimiterImas3D,
                    LimiterNetcdf,
                }
                and model.source not in data_sources
            ):
                raise ValueError(
                    f"limiter.{name} uses undefined data source "
                    f"'{model.source}'"
                )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        # Ensures topology set correctly for simple magnetic fields.
        magnetic_field = {"topology": SIMPLE}
        magnetic_field.update(self.magnetic_field_t.to_dict_toml())

        ne = self.electron_density_per_m3
        te = self.electron_temperature_ev
        zeff = self.effective_charge

        return {
            DATA_SOURCES: {
                name: data_source.to_dict_toml()
                for name, data_source in self.data_sources.items()
            },
            COORDINATES: {
                name.name: coordinate.to_dict_toml()
                for name, coordinate in self.coordinates.items()
            },
            KINETIC: {
                ELECTRON_DENSITY_PER_M3: ne.to_dict_toml(),
                ELECTRON_TEMPERATURE_EV: te.to_dict_toml(),
                EFFECTIVE_CHARGE: zeff.to_dict_toml(),
            },
            MAGNETIC_FIELD_T: magnetic_field,
            LIMITERS: {
                name: limiter.to_dict_toml()
                for name, limiter in self.limiters.items()
            },
        }

    @classmethod
    def from_dict_toml(cls, document: dict) -> "SystemDataProvider":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        (
            data_sources,
            coordinates,
            electron_density_per_m3,
            electron_temperature_ev,
            effective_charge,
            magnetic_field_t,
            limiters,
        ) = parse_schema(document)

        return cls(
            data_sources,
            coordinates,
            electron_density_per_m3,
            electron_temperature_ev,
            effective_charge,
            magnetic_field_t,
            limiters,
        )

    def open_data_sources(self):
        """
        Open all data sources.

        Raises
        ------
        TypeError
            Unknown data source type.
        """
        self._data_source_handles.clear()

        for model in (
            self.electron_density_per_m3,
            self.electron_density_per_m3,
            self.effective_charge,
            self.magnetic_field_t,
        ):
            if (
                type(model)
                in {
                    ModelImas,
                    ModelNetcdf,
                    MagneticModelTokamak,
                    MagneticModelStellarator,
                }
                and model.source not in self._data_source_handles
            ):
                self._data_source_handles[model.source] = None

        for model in self.limiters.values():
            if (
                type(model)
                in {
                    LimiterImasBoundingBox2D,
                    LimiterImas2D,
                    LimiterImas3D,
                    LimiterNetcdf,
                }
                and model.source not in self._data_source_handles
            ):
                self._data_source_handles[model.source] = None

        # Open all data sources.
        for name in self._data_source_handles:
            data_source = self.data_sources[name]

            if isinstance(data_source, DataSourceImas):
                self._data_source_handles[name] = DBEntry(data_source.uri, "r")
            elif isinstance(data_source, DataSourceNetcdf):
                self._data_source_handles[name] = nc4.Dataset(
                    data_source.filepath, "r"
                )
            elif isinstance(data_source, DataSourceVmec):
                raise NotImplementedError
            else:
                raise TypeError(data_source.__class__.__name__)

    def close_data_sources(self):
        """
        Close all data sources.
        """
        for handle in self._data_source_handles.values():
            handle.close()

    def get_coordinate_coordinator(self) -> CoordinateCoordinator:
        """
        Construct coordinate coordinator.

        Returns
        -------
        coordinate_coordinator : CoordinateCoordinator
            Object holding coordinate system data.
        """
        coordinate_coordinator = CoordinateCoordinator()

        if CoordinateSystem.TOROIDAL in self.coordinates:
            schema = self.coordinates[CoordinateSystem.TOROIDAL]
            toroidal = Toroidal((schema.r0, schema.z0))
            coordinate_coordinator.register_coordinate(toroidal)

        return coordinate_coordinator

    @staticmethod
    def get_analytic_model(
        schema: ModelAnalyticType,
        value_model: ValueModel,
    ) -> ValueModelBase:
        """
        Construct analytic plasma parameter model.

        Parameters
        ----------
        schema : ModelAnalyticType
            Definition of model.
        value_model : ValueModel
            Value model type to construct.

        Returns
        -------
        value_model : ValueModelBase
            Analytic plasma parameter model.

        Raises
        ------
        ValueError
            Unknown model type.
        """
        if type(schema) is ModelAnalyticConstant:
            return value_model.constant(
                schema.coordinate_system,
                schema.constant_value,
                scale_factor=schema.scale_factor,
            )

        if type(schema) is ModelAnalyticRamp:
            return value_model.ramp(
                schema.coordinate_system,
                schema.origin,
                schema.direction,
                schema.y0,
                schema.y1,
                schema.ramp_width,
                schema.smoothness,
                scale_factor=schema.scale_factor,
            )
        if type(schema) is ModelAnalyticQuadraticWell:
            return value_model.quadratic_well(
                schema.origin,
                schema.y0,
                schema.y1,
                schema.ramp_width,
                scale_factor=schema.scale_factor,
            )
        if type(schema) is ModelAnalyticQuadraticChannel:
            return value_model.quadratic_channel(
                schema.origin,
                schema.direction,
                schema.y0,
                schema.y1,
                schema.ramp_width,
                scale_factor=schema.scale_factor,
            )
        if type(schema) is ModelAnalyticQuadraticBowl:
            return value_model.quadratic_bowl(
                schema.origin,
                schema.direction,
                schema.y0,
                schema.y1,
                schema.ramp_width,
                scale_factor=schema.scale_factor,
            )

        raise ValueError(schema.__class__.__name__)

    def get_kinetic_model(
        self, time_s: float, model_name: str
    ) -> ValueModelBase:
        """
        Construct kinetic plasma parameter model.

        Parameters
        ----------
        time_s : float
            Time to load model at [s].
        model_name : str
            Name of model.

        Returns
        -------
        value_model : ValueModelBase
            Analytic plasma parameter model.

        Raises
        ------
        ValueError
            Unknown model type.
        """
        if model_name == ELECTRON_DENSITY_PER_M3:
            schema = self.electron_density_per_m3
            value_model = ValueModel.electron_density_per_m3()
        elif model_name == ELECTRON_TEMPERATURE_EV:
            schema = self.electron_temperature_ev
            value_model = ValueModel.electron_temperature_ev()
        elif model_name == EFFECTIVE_CHARGE:
            schema = self.effective_charge
            value_model = ValueModel.effective_charge()
        else:
            raise NotImplementedError(model_name)

        if type(schema) is ModelImas:
            source = schema.source
            data_source = self.data_sources[source]
            handle = self._data_source_handles[source]

            ids_core_profiles = handle.get_slice(
                "core_profiles",
                time_s,
                imasdef.CLOSEST_INTERP,
                occurrence=data_source.occurrence_core_profiles,
            )

            return Kinetic.from_imas_profiles_1d(
                model_name, ids_core_profiles, 0, schema.scale_factor
            )

        if type(schema) is ModelNetcdf:
            source = schema.source
            data_source = self.data_sources[source]
            handle = self._data_source_handles[source]

            return Kinetic.from_netcdf(
                model_name, time_s, handle, schema.scale_factor
            )

        return self.get_analytic_model(schema, value_model)

    def get_kinetic(self, time_s: float) -> Kinetic:
        """
        Construct all kinetic parameter models.

        Parameters
        ----------
        time_s : float
            Time to load model at [s].

        Returns
        -------
        kinetic : Kinetic
            Kinetic plasma parameter models.
        """
        electron_density_per_m3 = self.get_kinetic_model(
            time_s, ELECTRON_DENSITY_PER_M3
        )
        electron_temperature_ev = self.get_kinetic_model(
            time_s, ELECTRON_TEMPERATURE_EV
        )
        effective_charge = self.get_kinetic_model(time_s, EFFECTIVE_CHARGE)

        return Kinetic(
            electron_density_per_m3, electron_temperature_ev, effective_charge
        )

    def get_magnetic(self, time_s: float) -> Magnetic:
        """
        Construct magnetic parameter models.

        Parameters
        ----------
        time_s : float
            Time to load model at [s].

        Returns
        -------
        magnetic : Magnetic
            Magnetic parameter models.

        Raises
        ------
        ValueError
            Unknown magnetic model type.
        """
        if type(self.magnetic_field_t) is MagneticModelTokamak:
            data_source_name = self.magnetic_field_t.source
            data_source = self.data_sources[data_source_name]

            if data_source_name.startswith(IMAS):
                ids_equilibrium = self._data_source_handles[
                    self.magnetic_field_t.source
                ].get_slice(
                    "equilibrium",
                    time_s,
                    imasdef.CLOSEST_INTERP,
                    occurrence=data_source.occurrence_equilibrium,
                )

                return MagneticTokamak.from_imas(
                    ids_equilibrium, 0, self.magnetic_field_t.scale_factor
                )

            raise ValueError(self.magnetic_field_t.source)

        if type(self.magnetic_field_t) is MagneticModelStellarator:
            data_source_name = self.magnetic_field_t.source
            data_source = self.data_sources[data_source_name]

            if self.magnetic_field_t.source.startswith(VMEC):
                return MagneticStellarator.from_vmec()

            raise ValueError(self.magnetic_field_t.source)

        data_source_name = self.magnetic_field_t.source

        if data_source_name.startswith(NETCDF):
            dset = self._data_source_handles[self.magnetic_field_t.source]

            return Magnetic.from_netcdf(
                time_s, dset, self.magnetic_field_t.scale_factor
            )

        return Magnetic(
            self.get_analytic_model(
                self.magnetic_field_t, ValueModel.magnetic_field_t()
            )
        )

    def get_limiters(self, time_s: float) -> Limiters:
        """
        Construct limiter models.

        Parameters
        ----------
        time_s : float
            Time to load model at [s].

        Returns
        -------
        limiters : Limiters
            Limiter models.

        Raises
        ------
        ValueError
            Unknown limiter model.
        """
        limiters = Limiters({})

        for name, schema in self.limiters.items():
            if type(schema) in {
                LimiterImasBoundingBox2D,
                LimiterImas2D,
                LimiterImas3D,
            }:
                data_source_name = schema.source
                data_source = self.data_sources[data_source_name]

                if type(schema) is LimiterImasBoundingBox2D:
                    # Load equilibrium IDS.
                    ids_equilibrium = self._data_source_handles[
                        data_source_name
                    ].get_slice(
                        "equilibrium",
                        time_s,
                        imasdef.CLOSEST_INTERP,
                        occurrence=data_source.occurrence_equilibrium,
                    )

                    limiters.add_from_imas_bounding_box(
                        name,
                        ids_equilibrium,
                        0,
                        schema.effect,
                        schema.extinction_coefficient_nepers,
                    )
                elif type(schema) is LimiterImas2D:
                    # Load wall IDS.
                    ids_wall = self._data_source_handles[
                        data_source_name
                    ].get_slice(
                        "wall",
                        time_s,
                        imasdef.CLOSEST_INTERP,
                        occurrence=data_source.occurrence_wall,
                    )

                    limiters.add_from_imas_2d(
                        name,
                        ids_wall,
                        schema.effect,
                        schema.extinction_coefficient_nepers,
                    )
                else:
                    raise ValueError(type(schema))
            elif type(schema) is LimiterNetcdf:
                limiters.add_from_netcdf(
                    name,
                    self._data_source_handles[schema.source][schema.group],
                )
            elif type(schema) is LimiterAnalyticBoundingBox2D:
                limiters.add_bounding_box_2d(
                    name,
                    schema.coordinate,
                    schema.x_limits,
                    schema.y_limits,
                    schema.effect,
                    schema.extinction_coefficient_nepers,
                )
            elif type(schema) is LimiterAnalyticBoundingBox3D:
                limiters.add_bounding_box_3d(
                    name,
                    schema.coordinate,
                    schema.x_limits,
                    schema.y_limits,
                    schema.z_limits,
                    schema.effect,
                    schema.extinction_coefficient_nepers,
                )
            else:
                raise ValueError(type(schema).__name__)

        return limiters

    def build(self, time_s: float) -> SystemData:
        """
        Generate system data object from sources at given time.

        Parameters
        ----------
        time_s : float
            Time to load system data at [s].

        Returns
        -------
        system_data : SystemData
            Object providing information about plasma system.
        """
        # Get all used data sources.
        self.open_data_sources()

        # Load coordinate models.
        coordinate_coordinator = self.get_coordinate_coordinator()

        # Load kinetic models.
        kinetic = self.get_kinetic(time_s)

        # Load magnetic models.
        magnetic = self.get_magnetic(time_s)

        # Register any flux coordinates.
        if isinstance(magnetic, (MagneticTokamak, MagneticStellarator)):
            magnetic.register_flux_coordinates(coordinate_coordinator)

        # Calculate coordinate conversion paths.
        coordinate_coordinator.calculate_conversion_paths()

        # Load limiter models.
        limiters = self.get_limiters(time_s)

        # Close all data sources.
        self.close_data_sources()

        return SystemData(coordinate_coordinator, kinetic, magnetic, limiters)
