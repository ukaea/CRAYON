"""
Ray tracing algorithm.
"""

# Standard imports
import copy
import logging
import time
import typing

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.calculus import second_tensor_invariant_3x3
from crayon.coordinates import CoordinateSystem
from crayon.dispersion import (
    polarisation,
    vacuum_stix_polarisation,
)
from crayon.ray_tracing.caches import (
    CausticCache,
    CoordinateCache,
    HamiltonianCache,
    LimiterCache,
    ModeConversionCache,
    PlasmaCache,
    PlasmaVacuumBoundaryCache,
    State,
    StateDt,
    TunnellingCache,
    WaveCache,
    mjolhus_gaussian_beam,
    mjolhus_plane_wave,
)
from crayon.ray_tracing.initial_conditions import InitialConditions
from crayon.ray_tracing.integrator import (
    OptionsIntegrator,
    get_integrator_from_options,
)
from crayon.ray_tracing.options import OptionsRayTracing
from crayon.ray_tracing.ray import Ray
from crayon.shared.constants import (
    MIN_NORM_ELECTRON_DENSITY,
    SYMPLECTIC_MATRIX_J6,
    WaveMode,
)
from crayon.shared.dimensions import Dimension, Dimensions
from crayon.shared.io import IONetcdf, write_netcdf_variable
from crayon.shared.types import (
    ComplexArray,
    ComplexType,
    FloatArray,
    FloatType,
)
from crayon.system_data import SystemData
from crayon.system_data.limiter import LimiterEffect, reflect_specular

logger = logging.getLogger(__name__)

_x = Dimensions.slice_x
_k = Dimensions.slice_k
_xk = Dimensions.slice_xk


class RayTracingOutput(IONetcdf):
    """
    Output from ray tracing calculations. Selected values for each steo along
    the ray.

    Attributes
    ----------
    adiabatic_phase_rad : np.array[float]
        Adiabatic contribution to wave phase [radians].
    arc_length_m : np.array[float]
        Arc length along ray [m].
    cumulative_damped_power_collisional_w : np.array[float]
        Cumulative damped power due to collisional damping [W].
    cumulative_damped_power_external_w : np.array[float]
        Cumulative damped power due to external damping [W].
    cumulative_damped_power_resonance_w : np.array[float]
        Cumulative damped power due to resonant damping [W].
    cumulative_damped_power_w : np.array[float]
        Cumulative damped power [W].
    damping_fraction_collisional : np.array[float]
        Fraction of initial power lost due to collisional damping.
    damping_fraction_external : np.array[float]
        Fraction of initial power lost due to external damping.
    damping_fraction_resonance : np.array[float]
        Fraction of initial power lost due to resonant damping.
    damping_rate : np.array[float]
        Rate of power loss [W.s^-1].
    determinant : np.array[complex]
        Determinant of dispersion tensor.
    determinant_error_frequency : np.array[float]
        Deviation of dispersion tensor determinant real part from zero
        expressed as frequency [GHz].
    effective_charge : np.array[float]
        Effective charge.
    eigenvalue : np.array[complex]
        Eigenvalue of dispersion tensor for mode.
    eigenvalue_error_frequency : np.array[float]
        Deviation of mode eigenvalue real part from zero expressed as
        frequency [GHz].
    eikonal_phase_rad : np.array[float]
        Eikonal contribution to wave phase [radians].
    electron_density_per_m3 : np.array[float]
        Electron density [m^-3].
    electron_temperature_ev : np.array[float]
        Electron temperature [eV].
    focusing_tensor_x : np.array[float]
        Focusing tensor in x representation.
    frequency_ghz : float
        Wave frequency [GHz].
    initial_power_w : float
        Initial power in ray [W].
    intensity_amplification : np.array[float]
        Amplification in intensity compared to vacuum.
    k_parallel : np.array[float]
        Parallel wavenumber [m^-1].
    k_perp : np.array[float]
        Perpendicular wavenumber [m^-1].
    magnetic_field_strength_t : np.array[float]
        Magnetic field strength [T].
    magnetic_field_t : np.array[float]
        Magnetic field vector (Cartesian) [T].
    magnification_x : np.array[float]
        Magnification factor for intensity in x representation.
    mode_conversion_alarm : np.array[float]
        Mode conversion alarm.
    n_parallel : np.array[float]
        Parallel refractive index.
    n_perp : np.array[float]
        Perpendicular refractive index.
    normalised_collision_rate : np.array[float]
        Normalised electron-ion collision rate aka Stix Z.
    normalised_electron_density : np.array[float]
        Normalised electron density aka Stix X.
    normalised_electron_temperature : np.array[float]
        Normalised electron temperature theta.
    normalised_em_energy_density : np.array[float]
        Normalised electromagnetic energy density.
    normalised_magnetic_field_strength : np.array[float]
        Normalised magnetic field strength aka Stix Y.
    optical_depth : np.array[float]
        Optical depth [nepers].
    optical_depth_external : np.array[float]
        Optical depth due to external damping [nepers].
    optical_depth_internal : np.array[float]
        Optical depth due to internal dampnig (collisional, resonant)[nepers].
    phase_rad : np.array[float]
        Wave phase [radians].
    polarisation_cartesian : np.array[complex]
        Cartesian electric field polarisation.
    polarisation_left_handed : np.array[float]
        Left handed electric field polarisation fraction.
    polarisation_parallel : np.array[float]
        Parallel electric field polarisation fraction.
    polarisation_right_handed : np.array[float]
        Right handed electric field polarisation fraction.
    polarisation_stix : np.array[float]
        Stix frame pelectric field olarisation.
    position : np.array[float]
        Position (Cartesian) [m].
    power_w : np.array[float]
        Power in ray [W].
    refractive_index : np.array[float]
        Refractive index vector.
    time_ns : np.array[float]
        Time [ns].
    vacuum_wavenumber_per_m : float
        Vacuum wavenumber [m^-1].
    velocity : np.array[float]
        Ray velocity magitude in x space [m.s^-1].
    velocity_k : np.array[float]
        Ray velocity in k space [m^-1.s^-1].
    velocity_x : np.array[float]
        Ray velocity in x space [m.s^-1].
    wavevector_per_m : np.array[float]
        Wavevector [m^-1].
    wkb_validity : np.array[float]
        Validity of WKB approximation for each coordinate component.
    """

    __slots__ = (
        "adiabatic_phase_rad",
        "arc_length_m",
        "closest_to_conversion",
        "cumulative_damped_power_collisional_w",
        "cumulative_damped_power_external_w",
        "cumulative_damped_power_resonance_w",
        "cumulative_damped_power_w",
        "damping_fraction_collisional",
        "damping_fraction_external",
        "damping_fraction_resonance",
        "damping_rate",
        "determinant",
        "determinant_error_frequency",
        "effective_charge",
        "eigenvalue",
        "eigenvalue_error_frequency",
        "eikonal_phase_rad",
        "electron_density_per_m3",
        "electron_temperature_ev",
        "focusing_tensor_x",
        "frequency_ghz",
        "initial_power_w",
        "intensity_amplification",
        "k0ln_saddle",
        "k_parallel",
        "k_perp",
        "magnetic_field_strength_t",
        "magnetic_field_t",
        "magnification_x",
        "mode_conversion_alarm",
        "n_parallel",
        "n_parallel_at_conversion",
        "n_perp",
        "n_y_at_conversion",
        "normalised_collision_rate",
        "normalised_electron_density",
        "normalised_electron_temperature",
        "normalised_em_energy_density",
        "normalised_magnetic_field_strength",
        "optical_depth",
        "optical_depth_external",
        "optical_depth_internal",
        "osculating_plane_basis",
        "phase_rad",
        "polarisation_cartesian",
        "polarisation_left_handed",
        "polarisation_parallel",
        "polarisation_right_handed",
        "polarisation_stix",
        "position",
        "power_w",
        "refractive_index",
        "saddle_at_conversion",
        "time_ns",
        "vacuum_wavenumber_per_m",
        "velocity",
        "velocity_k",
        "velocity_x",
        "wavevector_per_m",
        "wkb_validity",
        "y_saddle",
    )

    def __init__(
        self,
        coordinate_systems: typing.Iterable[CoordinateSystem],
        /,
        *,
        max_ray_nodes: int = Dimensions.max_ray_nodes.size,
        max_mode_conversions: int = Dimensions.max_ray_children.size,
    ):
        """
        Inits RayTracingOutput.

        Parameters
        ----------
        coordinate_systems
            All coordinate systems in use.
        """
        # Fixed values.
        self.frequency_ghz = 0.0
        self.vacuum_wavenumber_per_m = 0.0
        self.initial_power_w = 0.0

        # Array sizes.
        n = max_ray_nodes
        nx = Dimensions.x.size
        nk = Dimensions.k.size
        n_xk = Dimensions.xk.size
        n_mc = max_mode_conversions

        # Coordinate.
        self.position: dict[CoordinateSystem, FloatArray] = {
            c: np.zeros((n, nx), dtype=FloatType) for c in coordinate_systems
        }

        # Wavevector / refractive index.
        self.wavevector_per_m = np.zeros((n, nk), dtype=FloatType)
        self.refractive_index = np.zeros((n, nk), dtype=FloatType)
        self.k_perp = np.zeros(n, dtype=FloatType)
        self.k_parallel = np.zeros(n, dtype=FloatType)
        self.n_perp = np.zeros(n, dtype=FloatType)
        self.n_parallel = np.zeros(n, dtype=FloatType)

        # Ray trajectory.
        self.velocity = np.zeros(n, dtype=FloatType)
        self.velocity_x = np.zeros((n, nx), dtype=FloatType)
        self.velocity_k = np.zeros((n, nk), dtype=FloatType)

        # Plasma data.
        self.electron_density_per_m3 = np.zeros(n, dtype=FloatType)
        self.electron_temperature_ev = np.zeros(n, dtype=FloatType)
        self.effective_charge = np.zeros(n, dtype=FloatType)
        self.magnetic_field_t = np.zeros((n, nx), dtype=FloatType)
        self.magnetic_field_strength_t = np.zeros(n, dtype=FloatType)

        self.normalised_electron_density = np.zeros(n, dtype=FloatType)
        self.normalised_electron_temperature = np.zeros(n, dtype=FloatType)
        self.normalised_collision_rate = np.zeros(n, dtype=FloatType)
        self.normalised_magnetic_field_strength = np.zeros(n, dtype=FloatType)

        # Hamiltonian data.
        self.eigenvalue = np.zeros(n, dtype=ComplexType)
        self.determinant = np.zeros(n, dtype=ComplexType)
        self.eigenvalue_error_frequency = np.zeros(n, dtype=FloatType)
        self.determinant_error_frequency = np.zeros(n, dtype=FloatType)

        # State vector data.
        self.time_ns = np.zeros(n, dtype=FloatType)
        self.arc_length_m = np.zeros(n, dtype=FloatType)
        self.eikonal_phase_rad = np.zeros(n, dtype=FloatType)
        self.adiabatic_phase_rad = np.zeros(n, dtype=FloatType)
        self.phase_rad = np.zeros(n, dtype=FloatType)

        self.optical_depth = np.zeros(n, dtype=FloatType)
        self.optical_depth_internal = np.zeros(n, dtype=FloatType)
        self.optical_depth_external = np.zeros(n, dtype=FloatType)
        self.damping_rate = np.zeros(n, dtype=FloatType)
        self.power_w = np.zeros(n, dtype=FloatType)

        self.damping_fraction_resonance = np.zeros(n, dtype=FloatType)
        self.damping_fraction_collisional = np.zeros(n, dtype=FloatType)
        self.damping_fraction_external = np.zeros(n, dtype=FloatType)

        self.cumulative_damped_power_w = np.zeros(n, dtype=FloatType)
        self.cumulative_damped_power_resonance_w = np.zeros(n, dtype=FloatType)
        self.cumulative_damped_power_collisional_w = np.zeros(
            n, dtype=FloatType
        )
        self.cumulative_damped_power_external_w = np.zeros(n, dtype=FloatType)

        self.polarisation_cartesian = np.zeros((n, nx), dtype=ComplexType)
        self.polarisation_stix = np.zeros((n, nx), dtype=ComplexType)
        self.polarisation_right_handed = np.zeros(n, dtype=FloatType)
        self.polarisation_left_handed = np.zeros(n, dtype=FloatType)
        self.polarisation_parallel = np.zeros(n, dtype=FloatType)

        self.normalised_em_energy_density = np.zeros(n, dtype=FloatType)

        self.magnification_x = np.zeros(n, dtype=FloatType)
        self.focusing_tensor_x = np.zeros((n, nx, nx), dtype=FloatType)
        self.intensity_amplification = np.zeros(n, dtype=FloatType)

        # Mode conversion parameters.
        self.wkb_validity = np.zeros((n, nx), dtype=FloatType)
        self.mode_conversion_alarm = np.zeros(n, dtype=FloatType)

        self.k0ln_saddle = np.zeros(n_mc, dtype=float)
        self.y_saddle = np.zeros(n_mc, dtype=float)
        self.osculating_plane_basis = np.zeros((n_mc, 2, n_xk), dtype=float)
        self.closest_to_conversion = np.zeros((n_mc, n_xk), dtype=float)
        self.saddle_at_conversion = np.zeros((n_mc, n_xk), dtype=float)
        self.n_parallel_at_conversion = np.zeros(n_mc, dtype=float)
        self.n_y_at_conversion = np.zeros(n_mc, dtype=float)

    def set_initial(
        self,
        frequency_ghz: float,
        vacuum_wavenumber_per_m: float,
        initial_power_w: float,
    ):
        """
        Set initial values which don't vary along ray.

        Parameters
        ----------
        frequency_ghz : float
            Wave frequency [GHz].
        vacuum_wavenumber_per_m : float
            Vacuum wavenumber [m^-1].
        initial_power_w : float
            Initial power [W].
        """
        self.frequency_ghz = frequency_ghz
        self.vacuum_wavenumber_per_m = vacuum_wavenumber_per_m
        self.initial_power_w = initial_power_w

    def save_ray_step(
        self,
        index: int,
        coordinate_cache: CoordinateCache,
        hamiltonian_cache: HamiltonianCache,
        plasma_cache: PlasmaCache,
        wave_cache: WaveCache,
        state: State,
        state_dt: StateDt,
        mode_conversion_cache: ModeConversionCache,
        caustic_cache: CausticCache,
    ):
        """
        Save contents of caches into output.

        Parameters
        ----------
        index : int
            Index of ray element.
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        hamiltonian_cache : HamiltonianCache
            Cache containing ray Hamiltonian data.
        plasma_cache : PlasmaCache
            Cache containing plasma parameter data.
        wave_cache : WaveCache
            Cache containing wave parameter data.
        state : State
            Integrator state vector.
        state_dt : StateDt
            Integrator state vector time derivative.
        mode_conversion_cache : ModeConversionCache
            Cache containing mode conversion data.
        caustic_cache : CausticCache
            Cache containing data about ray caustics.
        """
        i = index

        if i >= self.velocity.size:
            logger.warning("Discarding save_ray_step as exceeded max elements")
            return

        # Spatial position.
        for coordinate_system in self.position:
            self.position[coordinate_system][i, :] = coordinate_cache.position[
                coordinate_system
            ]

        # Wavevector / refractive index.
        self.wavevector_per_m[i, :] = wave_cache.wavevector_per_m
        self.refractive_index[i, :] = wave_cache.refractive_index
        self.k_perp[i] = wave_cache.k_perp.value
        self.k_parallel[i] = wave_cache.k_parallel.value
        self.n_perp[i] = hamiltonian_cache.arguments.value[
            Dimensions.IDX_N_PERP
        ]
        self.n_parallel[i] = hamiltonian_cache.arguments.value[
            Dimensions.IDX_N_PARALLEL
        ]

        # Ray trajectory.
        self.velocity[i] = state_dt.velocity
        self.velocity_x[i, :] = state_dt.velocity_xk[Dimensions.slice_x]
        self.velocity_k[i, :] = state_dt.velocity_xk[Dimensions.slice_k]

        # Plasma data.
        self.electron_density_per_m3[i] = (
            plasma_cache.electron_density_per_m3.value.cartesian
        )
        self.electron_temperature_ev[i] = (
            plasma_cache.electron_temperature_ev.value.cartesian
        )
        self.effective_charge[i] = (
            plasma_cache.effective_charge.value.cartesian
        )
        self.magnetic_field_t[i, :] = (
            plasma_cache.magnetic_field_t.value.cartesian
        )
        self.magnetic_field_strength_t[i] = (
            plasma_cache.magnetic_field_strength_t.value
        )
        self.normalised_electron_density[i] = (
            plasma_cache.normalised_electron_density.value
        )
        self.normalised_electron_temperature[i] = (
            plasma_cache.normalised_electron_temperature.value
        )
        self.normalised_collision_rate[i] = (
            plasma_cache.normalised_collision_rate.value
        )
        self.normalised_magnetic_field_strength[i] = (
            plasma_cache.normalised_magnetic_field_strength.value
        )

        # Hamiltonian data.
        self.eigenvalue[i] = (
            hamiltonian_cache.eigenvalue.real
            + 1.0j * hamiltonian_cache.eigenvalue.imag
        )
        self.determinant[i] = (
            hamiltonian_cache.determinant.real
            + 1.0j * hamiltonian_cache.determinant.imag
        )
        self.eigenvalue_error_frequency[i] = (
            hamiltonian_cache.eigenvalue_error_frequency
        )
        self.determinant_error_frequency[i] = (
            hamiltonian_cache.determinant_error_frequency
        )

        # State vector data.
        self.time_ns[i] = state.time_ns
        self.arc_length_m[i] = state.arc_length_m
        self.eikonal_phase_rad[i] = state.eikonal_phase_x_rad
        self.adiabatic_phase_rad[i] = state.adiabatic_phase_rad
        self.phase_rad[i] = state.phase_rad

        self.optical_depth_internal[i] = state.optical_depth_internal
        self.optical_depth_external[i] = state.optical_depth_external
        self.optical_depth[i] = state.optical_depth

        # Power in ray.
        self.damping_rate[i] = state_dt.damping_rate
        self.power_w[i] = self.initial_power_w * np.exp(-self.optical_depth[i])

        # Fraction of initial power damped due to cyclotron resonance or
        # collisional damping.
        self.damping_fraction_resonance[i] = state.damping_fraction_resonance
        self.damping_fraction_collisional[i] = (
            state.damping_fraction_collisional
        )
        self.damping_fraction_external[i] = state.damping_fraction_external

        # Cumulative damped power.
        self.cumulative_damped_power_resonance_w[i] = (
            self.initial_power_w * self.damping_fraction_resonance[i]
        )
        self.cumulative_damped_power_collisional_w[i] = (
            self.initial_power_w * self.damping_fraction_collisional[i]
        )
        self.cumulative_damped_power_external_w[i] = (
            self.initial_power_w * self.damping_fraction_external[i]
        )
        self.cumulative_damped_power_w[i] = (
            self.initial_power_w - self.power_w[i]
        )

        # Polarisation.
        self.polarisation_stix[i, :] = (
            hamiltonian_cache.polarisation.stix.value
        )
        self.polarisation_cartesian[i, :] = (
            hamiltonian_cache.polarisation.cartesian
        )

        self.polarisation_right_handed[i] = abs(
            self.polarisation_stix[i, 0] + 1.0j * self.polarisation_stix[i, 1]
        )
        self.polarisation_left_handed[i] = abs(
            self.polarisation_stix[i, 0] - 1.0j * self.polarisation_stix[i, 1]
        )
        self.polarisation_parallel[i] = abs(self.polarisation_stix[i, 2])

        # Normalised electromagnetic energy density i.e. fraction of wave
        # energy contained in electromagnetic fields.
        self.normalised_em_energy_density[i] = (
            hamiltonian_cache.normalised_em_energy_density
        )

        # Intensity.
        self.magnification_x[i] = state.magnification_x
        self.focusing_tensor_x[i, :, :] = state.focusing_tensor_x
        self.intensity_amplification[i] = np.exp(self.magnification_x[i])

        # Validity of wkb approximation.
        self.wkb_validity[i, :] = caustic_cache.wkb_validity

        # Proximity to mode conversion point.
        self.mode_conversion_alarm[i] = mode_conversion_cache.alarm_history[0]

        if mode_conversion_cache.save_conversion:
            j = mode_conversion_cache.conversions - 1

            self.k0ln_saddle[j] = mode_conversion_cache.k0ln_saddle
            self.y_saddle[j] = mode_conversion_cache.y_saddle
            self.n_parallel_at_conversion[j] = mode_conversion_cache.n_parallel
            self.n_y_at_conversion[j] = mode_conversion_cache.n_y
            self.closest_to_conversion[j, :] = mode_conversion_cache.xk_closest
            self.saddle_at_conversion[j, :] = mode_conversion_cache.xk_saddle
            self.osculating_plane_basis[j, :, :] = (
                mode_conversion_cache.xk_osculating_plane
            )

    def backfill_polarisation(self, index: int):
        """
        Set polarisation for all steps before given index to index. Used when
        a pure mode is launched but the optimal polarisation is not known until
        the ray enters the plasma.

        Parameters
        ----------
        index : int
            Ray index to fill before.
        """
        i = index
        self.polarisation_stix[:i, :] = self.polarisation_stix[i]
        self.polarisation_cartesian[:i, :] = self.polarisation_cartesian[i]
        self.polarisation_right_handed[:i] = self.polarisation_right_handed[i]
        self.polarisation_left_handed[:i] = self.polarisation_left_handed[i]
        self.polarisation_parallel[:i] = self.polarisation_parallel[i]

    def write_netcdf(self, group: nc4.Group, n_nodes: int, n_conversions: int):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Group
            netCDF4 dataset or group to write data to.
        n_nodes : int
            Number of ray nodes to write.
        """
        # Create dimensions for the number of ray nodes.
        n = n_nodes
        ray_node_dim = Dimension(Dimensions.ray_node, n)
        group.createDimension(ray_node_dim.name, ray_node_dim.size)

        n_c = n_conversions
        mode_conversion_dim = Dimension(Dimensions.mode_conversion, n_c)
        group.createDimension(
            mode_conversion_dim.name, mode_conversion_dim.size
        )

        # Write constants from wave cache.
        write_netcdf_variable(
            group,
            "frequency",
            (),
            self.frequency_ghz,
            "Wave frequency (not angular)",
            "GHz",
        )

        write_netcdf_variable(
            group,
            "vacuum_wavenumber",
            (),
            self.vacuum_wavenumber_per_m,
            "Vacuum wavenumber k0",
            "m^-1",
        )

        # Spatial position.
        _position = group.createGroup("position")

        for coordinate_system, position in self.position.items():
            _name = coordinate_system.name

            write_netcdf_variable(
                _position,
                _name,
                (ray_node_dim, Dimensions.x),
                position[:n, :],
                f"Spatial position ({_name})",
                coordinate_system.units,
            )

        # Wavevector / refractive index.#
        write_netcdf_variable(
            group,
            "wavevector",
            (ray_node_dim, Dimensions.k),
            self.wavevector_per_m[:n, :],
            f"Wavevector ({CoordinateSystem.CARTESIAN.name})",
            "m^-1",
        )

        write_netcdf_variable(
            group,
            "refractive_index",
            (ray_node_dim, Dimensions.k),
            self.refractive_index[:n, :],
            f"Refractive index vector ({CoordinateSystem.CARTESIAN.name})",
            "",
        )

        write_netcdf_variable(
            group,
            "k_perp",
            (ray_node_dim,),
            self.k_perp[:n],
            "Wavevector component perpendicular to magnetic field",
            "m^-1",
        )

        write_netcdf_variable(
            group,
            "k_parallel",
            (ray_node_dim,),
            self.k_parallel[:n],
            "Wavevector component parallel to magnetic field",
            "m^-1",
        )

        write_netcdf_variable(
            group,
            "n_perp",
            (ray_node_dim,),
            self.n_perp[:n],
            "Refractive index component perpendicular to magnetic field",
            "",
        )

        write_netcdf_variable(
            group,
            "n_parallel",
            (ray_node_dim,),
            self.n_parallel[:n],
            "Refractive index component parallel to magnetic field",
            "",
        )

        write_netcdf_variable(
            group,
            "velocity",
            (ray_node_dim,),
            self.velocity[:n],
            "Magnitude of ray velocity in real space (CARTESIAN)",
            "m.ns^-1",
        )

        write_netcdf_variable(
            group,
            "velocity_x",
            (ray_node_dim, Dimensions.x),
            self.velocity_x[:n, :],
            "Ray velocity in real space (CARTESIAN)",
            "m.ns^-1",
        )

        write_netcdf_variable(
            group,
            "velocity_k",
            (ray_node_dim, Dimensions.k),
            self.velocity_k[:n, :],
            "Ray velocity in wavenumber space (CARTESIAN)",
            "m^-1.ns^-1",
        )

        # Plasma data.
        write_netcdf_variable(
            group,
            "electron_density",
            (ray_node_dim,),
            self.electron_density_per_m3[:n],
            "Electron density",
            "m^-3",
        )

        write_netcdf_variable(
            group,
            "electron_temperature",
            (ray_node_dim,),
            self.electron_temperature_ev[:n],
            "Electron temperature",
            "eV",
        )

        write_netcdf_variable(
            group,
            "effective_charge",
            (ray_node_dim,),
            self.effective_charge[:n],
            "Effective charge",
            "",
        )

        write_netcdf_variable(
            group,
            "magnetic_field",
            (ray_node_dim, Dimensions.x),
            self.magnetic_field_t[:n, :],
            f"Magnetic field vector ({CoordinateSystem.CARTESIAN.name})",
            "T",
        )

        write_netcdf_variable(
            group,
            "magnetic_field_strength",
            (ray_node_dim,),
            self.magnetic_field_strength_t[:n],
            "Magnetic field strength",
            "T",
        )

        write_netcdf_variable(
            group,
            "normalised_electron_density",
            (ray_node_dim,),
            self.normalised_electron_density[:n],
            "Normalised electron density X = (f_pe / f)**2",
            "",
        )

        write_netcdf_variable(
            group,
            "normalised_electron_temperature",
            (ray_node_dim,),
            self.normalised_electron_temperature[:n],
            (
                "Normalised electron temperature theta = Te / electron rest "
                "mass energy"
            ),
            "",
        )

        write_netcdf_variable(
            group,
            "normalised_magnetic_field_strength",
            (ray_node_dim,),
            self.normalised_magnetic_field_strength[:n],
            ("Normalised magnetic field strength Y = f_ce / f"),
            "",
        )

        write_netcdf_variable(
            group,
            "normalised_collision_rate",
            (ray_node_dim,),
            self.normalised_collision_rate[:n],
            ("Normalised electron-ion collision rate Z = nu_ei / f"),
            "",
        )

        # Hamiltonian data.
        write_netcdf_variable(
            group,
            "eigenvalue",
            (ray_node_dim,),
            self.eigenvalue[:n],
            ("Eigenvalue of dispersion tensor for traced mode."),
            "",
        )

        write_netcdf_variable(
            group,
            "determinant",
            (ray_node_dim,),
            self.determinant[:n],
            "Determinant of dispersion tensor for traced mode.",
            "",
        )

        write_netcdf_variable(
            group,
            "eigenvalue_error_frequency",
            (ray_node_dim,),
            self.eigenvalue_error_frequency[:n],
            "Error in eigenvalue expressed as frequency",
            "GHz",
        )

        write_netcdf_variable(
            group,
            "determinant_error_frequency",
            (ray_node_dim,),
            self.determinant_error_frequency[:n],
            "Error in determinant expressed as frequency",
            "GHz",
        )

        # State vector data.
        write_netcdf_variable(
            group, "time", (ray_node_dim,), self.time_ns[:n], "Time", "ns"
        )

        write_netcdf_variable(
            group,
            "arc_length",
            (ray_node_dim,),
            self.arc_length_m[:n],
            "Arc length along ray",
            "m",
        )

        write_netcdf_variable(
            group,
            "eikonal_phase",
            (ray_node_dim,),
            self.eikonal_phase_rad[:n],
            "Eikonal phase of wave",
            "rad",
        )

        write_netcdf_variable(
            group,
            "adiabatic_phase",
            (ray_node_dim,),
            self.adiabatic_phase_rad[:n],
            "Adiabatic phase of wave",
            "rad",
        )

        write_netcdf_variable(
            group,
            "phase",
            (ray_node_dim,),
            self.phase_rad[:n],
            "Total phase of wave",
            "rad",
        )

        # Optical depth.
        write_netcdf_variable(
            group,
            "optical_depth_internal",
            (ray_node_dim,),
            self.optical_depth_internal[:n],
            (
                "Optical depth due to internal plasma processes e.g. "
                "cyclotron damping, collisional damping, etc."
            ),
            "nepers",
        )

        write_netcdf_variable(
            group,
            "optical_depth_external",
            (ray_node_dim,),
            self.optical_depth_external[:n],
            (
                "Optical depth due to external processes e.g. "
                "mode conversion, tunnelling, damping at reflections, etc."
            ),
            "nepers",
        )

        write_netcdf_variable(
            group,
            "optical_depth",
            (ray_node_dim,),
            self.optical_depth[:n],
            "Total optical depth",
            "nepers",
        )

        write_netcdf_variable(
            group,
            "damping_rate",
            (ray_node_dim,),
            self.damping_rate[:n],
            "Damping rate alpha i.e. dP/dt = alpha * P",
            "ns^-1",
        )

        # Power.
        write_netcdf_variable(
            group,
            "initial_power",
            (),
            self.initial_power_w,
            "Initial power in ray",
            "W",
        )

        write_netcdf_variable(
            group,
            "power",
            (ray_node_dim,),
            self.power_w[:n],
            "Power in ray",
            "W",
        )

        write_netcdf_variable(
            group,
            "cumulative_damped_power",
            (ray_node_dim,),
            self.cumulative_damped_power_w[:n],
            "Cumulative power damped along ray",
            "W",
        )

        write_netcdf_variable(
            group,
            "damping_fraction_resonance",
            (ray_node_dim,),
            self.damping_fraction_resonance[:n],
            "Fraction of initial power damped due to cyclotron damping.",
            "",
        )

        write_netcdf_variable(
            group,
            "damping_fraction_collisional",
            (ray_node_dim,),
            self.damping_fraction_collisional[:n],
            "Fraction of initial power damped due to collisional damping.",
            "",
        )

        write_netcdf_variable(
            group,
            "damping_fraction_external",
            (ray_node_dim,),
            self.damping_fraction_external[:n],
            (
                "Fraction of initial power damped due to external losses "
                "e.g. ray splitting, absorption on reflections, etc."
            ),
            "",
        )

        write_netcdf_variable(
            group,
            "cumulative_damped_power_resonance",
            (ray_node_dim,),
            self.cumulative_damped_power_resonance_w[:n],
            "Cumulative power damped along ray due to cyclotron damping",
            "W",
        )

        write_netcdf_variable(
            group,
            "cumulative_damped_power_collisional",
            (ray_node_dim,),
            self.cumulative_damped_power_collisional_w[:n],
            "Cumulative power damped along ray due to collisional damping",
            "W",
        )

        write_netcdf_variable(
            group,
            "cumulative_damped_power_external",
            (ray_node_dim,),
            self.cumulative_damped_power_external_w[:n],
            "Cumulative power damped along ray due to external damping",
            "W",
        )

        # Polarisation.
        write_netcdf_variable(
            group,
            "polarisation_cartesian",
            (ray_node_dim, Dimensions.x),
            self.polarisation_cartesian[:n, :],
            f"Electric field polarisation ({CoordinateSystem.CARTESIAN.name})",
            "",
        )

        write_netcdf_variable(
            group,
            "polarisation_stix",
            (ray_node_dim, Dimensions.x),
            self.polarisation_stix[:n, :],
            "Electric field polarisation (Stix frame)",
            "",
        )

        write_netcdf_variable(
            group,
            "polarisation_right_handed",
            (ray_node_dim,),
            self.polarisation_right_handed[:n],
            "Right handed component of polarisation",
            "",
        )

        write_netcdf_variable(
            group,
            "polarisation_left_handed",
            (ray_node_dim,),
            self.polarisation_left_handed[:n],
            "Left handed component of polarisation",
            "",
        )

        write_netcdf_variable(
            group,
            "polarisation_parallel",
            (ray_node_dim,),
            self.polarisation_parallel[:n],
            "Parallel component of polarisation",
            "",
        )

        write_netcdf_variable(
            group,
            "normalised_em_energy_density",
            (ray_node_dim,),
            self.normalised_em_energy_density[:n],
            "Fraction of wave energy contained in electromagnetic field",
            "",
        )

        write_netcdf_variable(
            group,
            "magnification",
            (ray_node_dim,),
            self.magnification_x[:n],
            ("Wave intensity focusing factor f i.e. I = I0 * exp(f)"),
            "",
        )

        write_netcdf_variable(
            group,
            "focusing_tensor",
            (ray_node_dim, Dimensions.x, Dimensions.x),
            self.focusing_tensor_x[:n, :, :],
            (
                "Focusing tensor aka Hessian of Eikonal phase in x "
                "representation. Diverges at caustics."
            ),
            "rad.m^-2",
        )

        write_netcdf_variable(
            group,
            "intensity_amplification",
            (ray_node_dim,),
            self.intensity_amplification[:n],
            (
                "Wave intensity divided by initial value i.e. "
                "how much the wave intensity has amplified"
            ),
            "",
        )

        write_netcdf_variable(
            group,
            "wkb_validity",
            (ray_node_dim, Dimensions.x),
            self.wkb_validity[:n, :],
            (
                "Value representing validity of WKB approximation"
                "dk_i / dx_i / k_i**2 for each position variable i"
            ),
            "",
        )

        # Mode conversion variables.
        write_netcdf_variable(
            group,
            "mode_conversion_alarm",
            (ray_node_dim,),
            self.mode_conversion_alarm[:n],
            "Absolute value of 2nd tensor invariant of dispersion tensor",
            "",
        )

        write_netcdf_variable(
            group,
            "k0ln_saddle",
            (mode_conversion_dim,),
            self.k0ln_saddle[:n_c],
            "Vacuum wavenumber k0 * density gradient scale length at saddle",
            "",
        )

        write_netcdf_variable(
            group,
            "y_saddle",
            (mode_conversion_dim,),
            self.y_saddle[:n_c],
            "Normalised magnetic field strength at saddle",
            "",
        )

        write_netcdf_variable(
            group,
            "n_parallel_at_conversion",
            (mode_conversion_dim,),
            self.n_parallel_at_conversion[:n_c],
            "Parallel refractive index at conversion",
            "",
        )

        write_netcdf_variable(
            group,
            "n_y_at_conversion",
            (mode_conversion_dim,),
            self.n_y_at_conversion[:n_c],
            (
                "Refractive index component perpendicular to magnetic field "
                "and density gradient at conversion."
            ),
            "",
        )

        write_netcdf_variable(
            group,
            "xk_closest_to_conversion",
            (mode_conversion_dim, Dimensions.xk),
            self.closest_to_conversion[:n_c, :],
            "Phase space position at conversion",
            "",
        )

        write_netcdf_variable(
            group,
            "xk_saddle_to_conversion",
            (mode_conversion_dim, Dimensions.xk),
            self.saddle_at_conversion[:n_c, :],
            "Phase space position of saddle point.",
            "",
        )

        write_netcdf_variable(
            group,
            "osculating_plane_basis",
            (mode_conversion_dim, Dimensions.two, Dimensions.xk),
            self.osculating_plane_basis[:n_c, :, :],
            "Phase space osculating plane basis vectors.",
            "",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "RayTracingOutput":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        group : RayTracingOutput
            Ray tracing output.
        """
        # Get ray node dimension size.
        size = group.dimensions[Dimensions.ray_node].size
        mc_size = group.dimensions[Dimensions.mode_conversion].size

        # Read all coordinate systems from position.
        coordinate_systems = tuple(
            CoordinateSystem.parse(name)
            for name in group["position"].variables
        )

        obj = cls(
            coordinate_systems,
            max_ray_nodes=size,
            max_mode_conversions=mc_size,
        )

        # Load fixed values.
        obj.frequency_ghz = group["frequency"][...].item()
        obj.vacuum_wavenumber_per_m = group["vacuum_wavenumber"][...].item()
        obj.initial_power_w = group["initial_power"][...].item()

        # Spatial position.
        _position = group["position"]
        for coordinate_system in coordinate_systems:
            obj.position[coordinate_system][:, :] = _position[
                coordinate_system.name
            ][:, :].data

        # Wavevector / refractive index.
        obj.wavevector_per_m[:, :] = group["wavevector"][:, :].data
        obj.refractive_index[:, :] = group["refractive_index"][:, :].data
        obj.k_perp[:] = group["k_perp"][:].data
        obj.k_parallel[:] = group["k_parallel"][:].data
        obj.n_perp[:] = group["n_perp"][:].data
        obj.n_parallel[:] = group["n_parallel"][:].data

        obj.velocity[:] = group["velocity"][:].data
        obj.velocity_x[:] = group["velocity_x"][:].data
        obj.velocity_k[:] = group["velocity_k"][:].data

        # Plasma data.
        obj.electron_density_per_m3[:] = group["electron_density"][:].data
        obj.electron_temperature_ev[:] = group["electron_temperature"][:].data
        obj.effective_charge[:] = group["effective_charge"][:].data
        obj.magnetic_field_t[:, :] = group["magnetic_field"][:, :].data
        obj.magnetic_field_strength_t[:] = group["magnetic_field_strength"][
            :
        ].data
        obj.normalised_electron_density[:] = group[
            "normalised_electron_density"
        ][:].data
        obj.normalised_electron_temperature[:] = group[
            "normalised_electron_temperature"
        ][:].data
        obj.normalised_collision_rate[:] = group["normalised_collision_rate"][
            :
        ].data
        obj.normalised_magnetic_field_strength[:] = group[
            "normalised_magnetic_field_strength"
        ][:].data

        # Hamiltonian data.
        obj.eigenvalue[:] = group["eigenvalue"][:].data
        obj.determinant[:] = group["determinant"][:].data
        obj.eigenvalue_error_frequency[:] = group[
            "eigenvalue_error_frequency"
        ][:].data
        obj.determinant_error_frequency[:] = group[
            "determinant_error_frequency"
        ][:].data

        # State vector data
        obj.time_ns[:] = group["time"][:].data
        obj.arc_length_m[:] = group["arc_length"][:].data
        obj.eikonal_phase_rad[:] = group["eikonal_phase"][:].data
        obj.adiabatic_phase_rad[:] = group["adiabatic_phase"][:].data
        obj.phase_rad[:] = group["phase"][:].data

        # Optical depth.
        obj.optical_depth_internal[:] = group["optical_depth_internal"][:].data
        obj.optical_depth_external[:] = group["optical_depth_external"][:].data
        obj.optical_depth[:] = group["optical_depth"][:].data
        obj.damping_rate[:] = group["damping_rate"][:].data

        # Power.
        obj.power_w[:] = group["power"][:].data

        obj.damping_fraction_resonance[:] = group[
            "damping_fraction_resonance"
        ][:].data
        obj.damping_fraction_collisional[:] = group[
            "damping_fraction_collisional"
        ][:].data
        obj.damping_fraction_external[:] = group["damping_fraction_external"][
            :
        ].data
        obj.cumulative_damped_power_w[:] = group["cumulative_damped_power"][
            :
        ].data
        obj.cumulative_damped_power_resonance_w[:] = group[
            "cumulative_damped_power_resonance"
        ][:].data
        obj.cumulative_damped_power_collisional_w[:] = group[
            "cumulative_damped_power_collisional"
        ][:].data
        obj.cumulative_damped_power_external_w[:] = group[
            "cumulative_damped_power_external"
        ][:].data

        # Polarisation.
        obj.polarisation_cartesian[:, :] = group["polarisation_cartesian"][
            :, :
        ].data
        obj.polarisation_stix[:, :] = group["polarisation_stix"][:, :].data
        obj.polarisation_right_handed[:] = group["polarisation_right_handed"][
            :
        ].data
        obj.polarisation_left_handed[:] = group["polarisation_left_handed"][
            :
        ].data
        obj.polarisation_parallel[:] = group["polarisation_parallel"][:].data
        obj.normalised_em_energy_density[:] = group[
            "normalised_em_energy_density"
        ][:].data

        # Wave intensity.
        obj.magnification_x[:] = group["magnification"][:].data
        obj.intensity_amplification[:] = group["intensity_amplification"][
            :
        ].data
        obj.focusing_tensor_x[:, :, :] = group["focusing_tensor"][:].data

        obj.wkb_validity[:, :] = group["wkb_validity"][:, :].data

        # Mode conversion variables.
        obj.mode_conversion_alarm[:] = group["mode_conversion_alarm"][:].data

        obj.k0ln_saddle[:] = group["k0ln_saddle"][:].data
        obj.y_saddle[:] = group["y_saddle"][:].data
        obj.n_parallel_at_conversion[:] = group["n_parallel_at_conversion"][
            :
        ].data
        obj.n_y_at_conversion[:] = group["n_y_at_conversion"][:].data
        obj.closest_to_conversion[:] = group["xk_closest_to_conversion"][
            :
        ].data
        obj.saddle_at_conversion[:] = group["xk_saddle_to_conversion"][:].data
        obj.osculating_plane_basis[:] = group["osculating_plane_basis"][:].data

        return obj


MAX_ATTEMPTS = 32


class RayTracer:
    """
    Main ray tracing algorithm.

    Attributes
    ----------
    children : list[InitialConditions]
        Initial conditions of child rays.
    integrator : SolverBase
        Integrator.
    output : RayTracingOutput
        Output from ray tracing.
    ray : Ray
        Ray being traced.
    system_data : SystemData
        Object containing information about the system.
    """

    __slots__ = (
        "_caustic_cache",
        "_enable_mode_conversion",
        "_enable_tunnelling",
        "_limiter_cache",
        "_mode_conversion_cache",
        "_options_integrator",
        "_options_ray_tracing",
        "_plasma_vacuum_boundary_cache",
        "_tunnelling_cache",
        "children",
        "integrator",
        "output",
        "ray",
        "system_data",
    )

    def __init__(
        self,
        system_data: SystemData,
        options_ray_tracing: OptionsRayTracing,
        options_integrator: OptionsIntegrator,
    ):
        """
        Inits RayTracer.

        Parameters
        ----------
        system_data : SystemData
            Object containing information about the system.
        options_ray_tracing : OptionsRayTracing
            Options for ray tracing.
        options_integrator : OptionsIntegrator
            Options for integrator.
        """
        self.system_data = system_data
        self._options_ray_tracing = options_ray_tracing
        self._options_integrator = options_integrator

        # Additional ray tracing modules.
        self.ray = None
        self._plasma_vacuum_boundary_cache = PlasmaVacuumBoundaryCache()
        self._limiter_cache = LimiterCache()
        self._caustic_cache = CausticCache()
        self._tunnelling_cache = TunnellingCache()
        self._mode_conversion_cache = ModeConversionCache()

        # Initial conditions of child rays.
        self.children: list[InitialConditions] = []

        # Flags to control functionality.
        self._enable_mode_conversion = True
        self._enable_tunnelling = True

        # Create integrator.
        self.integrator = get_integrator_from_options(
            State.STATE_VECTOR_SIZE,
            self._options_integrator,
            is_complex=False,
            primary_size=State.IDX_TAU + 1,
        )

    def synchronise_integrator(self):
        """
        Synchonise state held in integrator with ray wave. This can happen in
        initialisation but also if the state is changed outside the
        integration loop e.g. reflections, caustics, mode conversion, etc.
        """
        # Pack current wave state contents into state vector.
        self.ray.state.pack()

        # Set state vector on integrator.
        self.integrator.set_state(
            self.ray.state.time_ns, self.ray.state.state_vector
        )

    def create_ray(self, initial_conditions: InitialConditions):
        """
        Create ray from initial conditions.

        Parameters
        ----------
        initial_conditions : InitialConditions
            Ray initial conditions.
        """
        # Create ray.
        self.ray = Ray(self.system_data, initial_conditions)

        # Output from ray tracing.
        self.output = RayTracingOutput(
            self.system_data.coordinate_coordinator.coordinates.keys(),
            max_ray_nodes=self._options_ray_tracing.max_ray_nodes,
        )

    def trace(
        self,
        initial_conditions: InitialConditions,
        /,
        *,
        create_ray: bool = True,
    ):
        """
        Trace ray.

        Parameters
        ----------
        initial_conditions : InitialConditions
            Ray initial conditions.
        create_ray : bool, optional
            If True, create a new ray.

        Raises
        ------
        ValueError
            create_ray is False but no ray has been created.
        """
        start_time = time.time()

        # Create ray.
        logger.info("Start %s", initial_conditions.name)

        if create_ray:
            self.create_ray(initial_conditions)
        elif self.ray is None:
            raise ValueError("No ray and create_ray = False")

        # Set total reflections on parent rays.
        self._limiter_cache.reflection_count = initial_conditions.reflections

        # Initialise ray into integrator.
        self.integrator.set_dy_dt_func(self.ray.calculate_state_vector_dt)
        self.synchronise_integrator()

        # Set fixed values in trajectory.
        self.output.set_initial(
            self.ray.wave_cache.frequency_ghz,
            self.ray.wave_cache.vacuum_wavenumber_per_m,
            self.ray.state.power_w,
        )

        # Write initial conditions to trajectory.
        self.accept_step(override=True, synchronise_integrator=True)

        # Set initial timestep of integrator.
        self.integrator.set_initial_timestep(
            initial_timestep=self._options_integrator.initial_timestep
        )

        for _ in range(1, self._options_ray_tracing.max_ray_nodes):
            # Take a step using the integrator adaptive timestep.
            self.step()

            # Check for stop condition from integrator.
            if self.ray.stop_condition:
                break

            # Check if the ray crossed a plasma-vacuum boundary.
            self.check_vacuum_plasma_boundary_crossing()

            # Check for caustics.
            self.check_caustics()

            # Check for tunnelling.
            self.check_tunnelling()

            # Check for mode conversion.
            self.check_mode_conversion()

            # Check limiter intersections.
            self.check_limiter_intersections()

            # Accept step. Might have happened already if there was any
            # reflections, tunneling or mode conversion.
            if not self.ray.step_accepted:
                self.accept_step()

            # Check stop conditions.
            self.check_stop_conditions()

            if self.ray.stop_condition:
                # Final timestep is written at the end of this routine.
                break

        # Write number of mode conversions.
        self.ray.conversions = self._mode_conversion_cache.conversions

        # Debug output about ray.
        time_elapsed = time.time() - start_time

        logger.info("Total steps: %s", self.ray.index)
        logger.info("dy/dt evaluations: %s", self.ray.dy_dt_evaluations)
        logger.info("Calculation time [s]: %s", np.round(time_elapsed, 2))
        logger.info("Children = %s", len(self.children))

    def step(self, /, *, timestep_ns: float | None = None):
        """
        Take a single integration step. If timestep is not provided, the
        adaptive timestep from the integrator is used.

        Parameters
        ----------
        timestep_ns : float, optional
            If provided, take a step of this size. Otherwise take an adaptive
            step.

        Notes
        -----
        This step will not accepted until accept_step is called.
        """
        self.ray.step_accepted = False

        if timestep_ns:
            self.integrator.step_manual(timestep_ns)
        else:
            self.integrator.step_adaptive()

        # Check for stop condition from integrator.
        if self.integrator.stop_condition:
            self.ray.set_stop_condition(self.integrator.stop_condition)

    def accept_step(
        self,
        /,
        *,
        synchronise_integrator: bool = False,
        override: bool = False,
    ):
        """
        Accept the current proposed integration step. This will also evaluate
        the hamiltonian model at the accepted point.

        Parameters
        ----------
        synchronise_integrator : bool, optional
            If True, recalculate first stage of integrator to match current
            state. Use if the state vector has been updated outside the
            integration loop.
        override : bool, optional
            Override current accepted step instead of saving new step.
        """
        # Accept step on integrator. This will also ensure the state vector
        # derivative is calculated at the accepted point i.e.
        # calculate_state_vector_dt is called (if it wasn't as the last stage
        # of the integration step).
        if synchronise_integrator:
            self.ray.state.pack()
            self.integrator.set_state(self.ray.time_ns, self.ray.state_vector)

            # Force dy/dt recalculation at modified position.
            self.integrator.accept_step(force_dy_dt_calculation=True)
        else:
            self.integrator.accept_step(force_dy_dt_calculation=False)

        # Accept current ray state.
        self.ray.accept_step()

        # Avoid ray index exceeding maximum value. Possible if we accept
        # multiple steps before checking stop conditions.
        self.ray.index = min(
            self.ray.index, self._options_ray_tracing.max_ray_nodes - 1
        )

        # Save ray state to output.
        if override:
            self.ray.index -= 1

        self.output.save_ray_step(
            self.ray.index,
            self.ray.coordinate_cache,
            self.ray.hamiltonian_cache,
            self.ray.plasma_cache,
            self.ray.wave_cache,
            self.ray.state,
            self.ray.state_dt,
            self._mode_conversion_cache,
            self._caustic_cache,
        )

        # Update flag if in vacuum..
        self._plasma_vacuum_boundary_cache.previous_step_vacuum = (
            self.ray.plasma_cache.vacuum
        )

        # Save last (x, k) position.
        self._limiter_cache.last_position[:] = (
            self.ray.state.position_cartesian
        )
        self._limiter_cache.last_wavevector[:] = (
            self.ray.state.wavevector_cartesian
        )

        logger.info(
            "i, t, [x, y, z], X, Y, n_perp, n_parallel, power = "
            "%s, %s, %s, %s, %s, %s, %s, %s",
            self.ray.index,
            self.ray.time_ns,
            self.ray.state.position_cartesian,
            self.ray.plasma_cache.normalised_electron_density.value.item(),
            self.ray.plasma_cache.normalised_magnetic_field_strength.value.item(),
            self.ray.wave_cache.n_perp,
            self.ray.wave_cache.n_parallel,
            self.ray.state.power_w,
        )

    def add_child(
        self,
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
    ):
        """
        Add child ray.

        Parameters
        ----------
        time_ns : float
            Initial time [ns].
        frequency_ghz : float
            Wave frequency [GHz].
        position_cartesian : np.array[float]
            Initial position (Cartesian) [m].
        refractive_index_cartesian : np.array[float]
            Initial refractive index (Cartesian).
        eikonal_phase_rad : float
            Initial eikonal phase [radians].
        adiabatic_phase_rad : float
            Initial adiabatic phase [radians].
        polarisation_stix : np.array[complex]
            Stix frame electric field polarisation.
        wave_mode : WaveMode
            Wave mode.
        power_w : float
            Initial power [W].
        intensity_w_per_m2 : float
            Initial intensity [W.m^-2].
        """
        # Check if ray has too many children already.
        if len(self.children) >= self._options_ray_tracing.max_ray_children:
            logger.warning(
                "[%s] Discarding child as reached max number of children "
                "per ray.",
                self.ray.name,
            )
            return

        # Check if power in child ray too small.
        min_power_w = (
            self._options_ray_tracing.min_power_fraction_new_ray
            * self.ray.state.initial_power_w
        )

        if power_w <= min_power_w:
            logger.warning(
                "[%s] Discarding child as power fraction too small.",
                self.ray.name,
            )
            return

        initial_conditions = InitialConditions(
            f"{self.ray.name}-{len(self.children)}",
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
            self.ray.beam_waist_radius_m,
            bundle=self.ray.bundle,
        )
        initial_conditions.reflections = self._limiter_cache.reflection_count

        self.children.append(initial_conditions)

    def check_stop_conditions(self):
        """
        Check ray stop conditions.
        """
        # Reached max ray nodes.
        if self.ray.index >= self._options_ray_tracing.max_ray_nodes - 1:
            self.ray.set_stop_condition(
                "Reached max ray nodes "
                f"({self._options_ray_tracing.max_ray_nodes})"
            )
            return

        # Optical depth too high.
        optical_depth = self.ray.state.optical_depth
        optical_depth_max = self._options_ray_tracing.max_optical_depth

        if optical_depth > optical_depth_max:
            msg = (
                "Optical depth too large: "
                f"{optical_depth:.3f} > "
                f"{optical_depth_max:.3f}"
            )
            self.ray.set_stop_condition(msg, error=False)
            return

        # Warn about the weak damping approximation breaking down.
        alpha_ghz = self.ray.state_dt.damping_rate
        frequency_ghz = self.ray.wave_cache.frequency_ghz

        if alpha_ghz > 0.01 * frequency_ghz:
            logger.warning(
                "[%s] Weak damping approximation violated: "
                "alpha / f = %s > 0.01",
                self.ray.name,
                alpha_ghz,
            )

    def check_vacuum_plasma_boundary_crossing(self):
        """
        Check if the last integration step crossed over a vacuum plasma
        boundary.
        """
        # XOR only true if values are different.
        transition = (
            self.ray.plasma_cache.vacuum
            ^ self._plasma_vacuum_boundary_cache.previous_step_vacuum
        )

        if not transition:
            return

        # Flag to show which direction we are going.
        vacuum_plasma_transition = (
            self._plasma_vacuum_boundary_cache.previous_step_vacuum
        )

        # Step directly onto vacuum / plasma boundary transition.
        original_timestep_ns = copy.copy(self.integrator.timestep)

        max_attempts = 16
        timestep_ns = 0.0
        threshold = 0.9 * MIN_NORM_ELECTRON_DENSITY
        target = MIN_NORM_ELECTRON_DENSITY

        _bracket = np.empty(2)
        _bracket[0] = 0.0
        _bracket[1] = original_timestep_ns

        # Binary search for transition.
        _success = False

        for _ in range(max_attempts):
            # Step to middle of range.
            timestep_ns = 0.5 * sum(_bracket)
            self.step(timestep_ns=timestep_ns)

            # Shrink bracket based on normalised density.
            x = self.ray.plasma_cache.normalised_electron_density.value.item()
            dx = x - target

            if vacuum_plasma_transition:
                # Want to finish on plasma side.
                if dx > 0 and abs(dx) < threshold:
                    _success = True
                    break

                # Vacuum - plasma transition.
                if dx > 0.0:
                    # Too far into plasma. Search for smaller timestep.
                    _bracket[1] = timestep_ns
                else:
                    # Not in plasma. Search for larger timestep.
                    _bracket[0] = timestep_ns
            else:
                # Want to finish on vacuum side.
                if dx < 0 and abs(dx) < threshold:
                    _success = True
                    break

                # Plasma - vacuum transition.
                if dx > 0.0:
                    # Not far enough out of plasma. Search for larger timestep.
                    _bracket[0] = timestep_ns
                else:
                    # Too far out of plasma. Search for smaller timestep.
                    _bracket[1] = timestep_ns

        if _success:
            logger.info(
                "Found vacuum - plasma boundary: %s",
                self.integrator.y[Dimensions.slice_x],
            )
        else:
            # Failed to find transition point.
            if vacuum_plasma_transition:
                self.ray.set_stop_condition(
                    "Unable to resolve vacuum -> plasma transition "
                    f"within max attempts ({max_attempts})",
                )
            else:
                self.ray.set_stop_condition(
                    "Unable to resolve plasma -> vacuum transition "
                    f"within max attempts ({max_attempts})",
                )

            return

        if vacuum_plasma_transition:
            # Shrink timestep substantially. When first entering the plasma is
            # the primary place error is accumulated.
            self.integrator.set_timestep(0.2 * original_timestep_ns)

            if self.ray.wave_mode == WaveMode.ANY:
                # Fudge this as we are effectively in vacuum.
                self.ray.plasma_cache.vacuum = True

                # Split ray if following hybrid mode.
                o_mode, x_mode = self.ray.vacuum_power_fraction_in_ox()

                logger.info(
                    "[%s] Split power O mode = %s, X mode = %s",
                    self.ray.name,
                    np.round(o_mode, 3),
                    np.round(x_mode, 3),
                )

                # Stick with highest power mode.
                if o_mode >= x_mode:
                    self_polarisation = WaveMode.O
                    other_polarisation = WaveMode.X
                    transmission = 1 - o_mode
                else:
                    self_polarisation = WaveMode.X
                    other_polarisation = WaveMode.O
                    transmission = 1 - x_mode

                # Calculate new wave polarisation and power.
                # Polarisation is ~ vacuum polarisation as density very low.
                self.ray.wave_mode = self_polarisation
                self.ray.hamiltonian_cache.calculate_vacuum_stix_polarisation(
                    self.ray.wave_mode,
                    self.ray.plasma_cache,
                    self.ray.wave_cache,
                )

                # Remove transmitted power from ray.
                # Clip to avoid log(0). Minus as log(x) < 0 as x < 1.
                self.ray.state.increment_optical_depth_external(
                    -np.log(max(1.0e-100, 1 - transmission))
                )

                # Create child ray for other polarisation.
                intensity_w_per_m2 = (
                    self.ray.wave_cache.frequency_ghz
                    * self.ray.state.wave_action_density
                )

                self.add_child(
                    self.ray.state.time_ns,
                    self.ray.wave_cache.frequency_ghz,
                    np.copy(self.ray.state.position_cartesian),
                    np.copy(self.ray.wave_cache.refractive_index),
                    self.ray.state.eikonal_phase_rad,
                    self.ray.state.adiabatic_phase_rad,
                    np.zeros(3, dtype=ComplexType),
                    other_polarisation,
                    transmission * self.ray.state.initial_power_w,
                    transmission * intensity_w_per_m2,
                )
            else:
                # Back fill optimal polarisation for this mode.
                self.output.backfill_polarisation(self.ray.index)
        else:
            # Reset timestep.
            self.integrator.set_timestep(original_timestep_ns)

            # Wave mode becomes ambiguous once in vacuum.
            self.ray.wave_mode = WaveMode.ANY

        # Root find to satisfy dispersion relation.
        self.ray.find_root_n(kinetic=False)

        # Synchronise integrator with corrected wavevector position.
        self.accept_step(synchronise_integrator=True)

    def check_limiter_intersections(self):
        """
        Check for intersections with the limiter.

        Raises
        ------
        ValueError
            Wavevector has zero norm.
        """
        # Get previous and proposed Cartesian position.
        previous_position = self._limiter_cache.last_position
        ray_direction = self.ray.state.position_cartesian - previous_position

        # No need to check limiter interactions if ray in same place.
        # Catches edge cases when stepping outside integration loop.
        if np.allclose(ray_direction, 0.0):
            return

        # Get any intersected limiter element.
        (
            intersects,
            limiter_name,
            element_index,
            s,
            normal,
            effect,
            extinction_coefficient_nepers,
        ) = self.system_data.limiters.intersects(
            previous_position,
            ray_direction,
            ignore=self._limiter_cache.last_intersected_element,
        )

        # If ignoring a limiter element then start noticing it again.
        self._limiter_cache.reset_last_intersected_element()

        if not intersects:
            return

        # Track the id of the element we just intersected. This is to avoid
        # numerical precision issues causing a double intersection with the
        # same element next ray step.
        self._limiter_cache.set_last_intersected_element(
            limiter_name, element_index
        )

        # Jump to intersection point. All steps are in straight line and in
        # vacuum the rays travel at constant speed so we can bypass the solve
        # ivp step (which is convenient).
        self.ray.set_time(
            self.ray.state.time_ns + s * self.integrator.timestep
        )

        previous_wavevector = self._limiter_cache.last_wavevector
        proposed_wavevector = self.ray.state.wavevector_cartesian

        self.ray.set_xk_position(
            previous_position + s * ray_direction,
            previous_wavevector
            + s * (proposed_wavevector - previous_wavevector),
        )

        logger.info(
            "Found limiter intersection: %s",
            np.array2string(
                self.ray.state.position_cartesian,
                precision=3,
                floatmode="fixed",
            ),
        )

        # Accept step.
        self.accept_step(synchronise_integrator=True)

        # Apply effect from intersection.
        if effect == LimiterEffect.STOP:
            self.ray.set_stop_condition(
                "Hit limiter element with STOP effect", error=False
            )
            return

        if effect in {
            LimiterEffect.REFLECT_SPECULAR,
            LimiterEffect.REFLECT_POLARISER,
        }:
            # Check if exceeded max reflections.
            if (
                self._limiter_cache.reflection_count
                > self._options_ray_tracing.max_reflections
            ):
                self.ray.set_stop_condition(
                    "Reached limit on reflections "
                    f"({self._options_ray_tracing.max_reflections})"
                )

            # Apply reflection to wavevector.
            new_wavevector = reflect_specular(
                self.ray.state.wavevector_cartesian, normal
            )

            # Apply extinction due to reflection.
            self.ray.state.increment_optical_depth_external(
                extinction_coefficient_nepers
            )

            # Reflect electric field polarisation.
            if effect == LimiterEffect.REFLECT_POLARISER:
                # Simple model for now. Assume polariser purely transforms
                # O <-> X regardless of how its struck.

                # Fraction of power in each mode, power ~ amplitude**2.
                o_mode_power_fraction, x_mode_power_fraction = (
                    self.ray.vacuum_power_fraction_in_ox()
                )

                # Calculate fundamental polarisations.
                arguments = self.ray.hamiltonian_cache.arguments

                stix_polarisation_o_mode = vacuum_stix_polarisation(
                    arguments.value[Dimensions.IDX_Y],
                    arguments.value[Dimensions.IDX_N_PERP],
                    arguments.value[Dimensions.IDX_N_PARALLEL],
                    WaveMode.O,
                )
                stix_polarisation_x_mode = vacuum_stix_polarisation(
                    arguments.value[Dimensions.IDX_Y],
                    arguments.value[Dimensions.IDX_N_PERP],
                    arguments.value[Dimensions.IDX_N_PARALLEL],
                    WaveMode.X,
                )

                self.ray.set_polarisation_stix(
                    np.sqrt(o_mode_power_fraction) * stix_polarisation_o_mode
                    + np.sqrt(x_mode_power_fraction) * stix_polarisation_x_mode
                )
            else:
                # Find wavevector unit vector.
                _norm = np.linalg.norm(self.ray.state.wavevector_cartesian)

                if np.isclose(_norm, 0.0):
                    raise ValueError("Wavevector has norm zero")

                k_hat = self.ray.state.wavevector_cartesian / _norm

                # Calculate unit vector normal to plane from k_hat and normal.
                parallel_hat = np.cross(k_hat, normal)

                # Electric field component in that plane is unchanged.
                # Otherwise have an 180 degree phase shift (E -> -E).
                e_parallel = np.dot(
                    self.ray.hamiltonian_cache.polarisation.cartesian,
                    parallel_hat,
                )

                self.ray.set_polarisation_cartesian(
                    2.0 * e_parallel * parallel_hat
                    - self.ray.hamiltonian_cache.polarisation.cartesian
                )

            # Advance time to next floating point number so we can save a step
            # containing the new wavevector / optical depth.
            self.ray.set_time(
                np.nextafter(self.integrator.t, self.integrator.t + 1.0)
            )
            self.ray.set_xk_position(
                self.ray.state.position_cartesian, new_wavevector
            )

            # Save step.
            self.accept_step(synchronise_integrator=True)
        else:
            raise NotImplementedError(effect.effect_name)

    def check_caustics(self):
        """
        Check for caustics.
        """
        if self._caustic_cache.caustic_detected(
            self.ray.state.wavevector_cartesian,
            self.ray.state.focusing_tensor_x,
        ):
            # Approaching a caustic.
            logger.info(
                "[%s] Detected approaching caustic: %s",
                self.ray.name,
                np.array2string(
                    self.ray.state.position_cartesian,
                    precision=3,
                    floatmode="fixed",
                ),
            )

            # x space caustic when dk/dx -> infinity => dH/dk = 0.
            # Want to switch before this point.
            # Maslov transformation to switch amplitude evolution to k space.
            return

            # Switch to other representation.
            if self.state.x_representation:
                self.logger.info("Switch to k representation")
                self.state.use_k_representation()
            else:
                self.logger.info("Switch to x representation")
                self.state.use_x_representation()

    def check_tunnelling(self):
        """
        Check for ray tunnelling.
        """
        if (
            self._enable_tunnelling
            or not self._tunnelling_cache.tunnelling_detected()
        ):
            return

    def _find_saddle_point(
        self,
        xk_closest: FloatArray,
        xk_osculating_plane_basis: FloatArray,
        /,
        *,
        max_attempts: int = 32,
    ) -> tuple[float, FloatArray, FloatArray, FloatArray]:
        """
        Find saddle point of Hamiltonian (determinant of dispersion tensor) in
        mode conversion osculating plane.

        Parameters
        ----------
        xk_closest: np.array[float]
            Phase space position (x, k) of point of closest approach to mode
            conversion on incoming branch.
        xk_osculating_plane_basis: np.array[float]
            Phase space coordinates (x, k) of basis vectors of osculating
            plane.
        max_attempts : int
            Maximum number of Netwon iterations.

        Returns
        -------
        h_saddle : float
            Determinant of dispersion tensor
        xk_saddle : np.array[float]
            Phase space position (x, k) of saddle point.
        pq_saddle : np.array[float]
            Osculating plane coordinates (p, q) of saddle point.
        pq_conversion_normal : np.array[float]
            Osculating plane coordinates (p, q) of hyperbola axis pointing
            from closest point towards saddle point
        """
        # Find saddle point via Netwon iteration.
        xk_saddle = self._mode_conversion_cache.xk_saddle
        xk_saddle[:] = xk_closest

        # Calculate derivatives of determinant. The determinant contains
        # information about all modes.
        self.ray.calculate_hamiltonian(derivatives=2, determinant=True)

        # Jacobian and Hessian of Hamiltonian on phase space projected onto
        # the osculating plane i.e. z = z0 + p * e_p + q * e_q
        reduced_jacobian = np.empty(2, dtype=FloatType)
        reduced_hessian = np.empty((2, 2), dtype=FloatType)

        e_p = xk_osculating_plane_basis[0]
        e_q = xk_osculating_plane_basis[1]

        determinant = self.ray.hamiltonian_cache.determinant

        max_attempts = 32
        alpha = 0.2
        pq_saddle = np.zeros(2)

        _success = False
        for attempt in range(1, max_attempts + 1):
            xk_saddle[:] = xk_closest + np.matmul(
                xk_osculating_plane_basis.T, pq_saddle
            )

            # Evaluate reduced Jacobian.
            h_dxk = determinant.first_derivative.z[_xk]

            reduced_jacobian[0] = np.dot(h_dxk, e_p)
            reduced_jacobian[1] = np.dot(h_dxk, e_q)

            # Evaluate reduced Hessian.
            h_dxk2 = determinant.second_derivative.z[_xk, _xk]

            reduced_hessian[0, 0] = np.einsum("i, ij, j", e_p, h_dxk2, e_p)
            reduced_hessian[0, 1] = np.einsum("i, ij, j", e_p, h_dxk2, e_q)
            reduced_hessian[1, 0] = np.einsum("i, ij, j", e_q, h_dxk2, e_p)
            reduced_hessian[1, 1] = np.einsum("i, ij, j", e_q, h_dxk2, e_q)

            # If jacobian vanishes we have found saddle point.
            if max(abs(reduced_jacobian)) < 1e-4:  # noqa: PLR2004
                _success = True
                break

            # Otherwise take a Newton step towards saddle point.
            try:
                dpq = -np.linalg.solve(reduced_hessian, reduced_jacobian)
            except np.linalg.LinAlgError as e:
                self.ray.set_stop_condition(getattr(e, "message", repr(e)))
                break

            # Calculate hamiltonian at new position.
            # Use an acceleration where we take larger steps after more
            # attempt so we don't jump too far when we are far from the saddle
            pq_saddle[:] += (
                alpha + (1 - alpha) * attempt / max_attempts
            ) * dpq

            self.ray.set_xk_position(
                xk_saddle[Dimensions.slice_x],
                xk_saddle[Dimensions.slice_k],
            )
            self.ray.calculate_hamiltonian(derivatives=2, determinant=True)

        if not _success:
            self.ray.set_stop_condition(
                "Unable to find Hamiltonian saddle point after "
                f"{max_attempts} attempts"
            )
            return None

        # Check for saddle structure. Expect eigenvalues of reduced
        # hessian to have opposite signs => determinant should be < 0.
        _trace = reduced_hessian[0, 0] + reduced_hessian[1, 1]
        _det = (
            reduced_hessian[0, 0] * reduced_hessian[1, 1]
            - reduced_hessian[0, 1] * reduced_hessian[1, 0]
        )

        if _det >= 0.0:
            self.ray.set_stop_condition(
                "Reduced Hessian does not have saddle structure. "
                "Abort mode conversion."
            )
            return None

        logger.info(
            "Found saddle point at (x, k) = %s",
            np.array2string(xk_saddle, precision=3, floatmode="fixed"),
        )

        # Principle directions of curvature are eigenvectors of the Hessian.
        # This helps to define a direction normal to the conversion i.e.
        # direction between two branches at point of closest approach.
        _gap = np.sqrt(_trace * _trace - 4 * _det)

        # plus has positive curvature, minus has negative curvature.
        lambda_plus = 0.5 * (_trace + _gap)
        lambda_minus = 0.5 * (_trace - _gap)

        # If hamiltonian at saddle point > 0, direction with negative curvature
        # will include both roots. Otherwise, choose positive curvature.
        h_saddle = self.ray.hamiltonian_cache.determinant.real

        # The normal is the direction between the two branches at the point
        # of closest approach.
        pq_conversion_normal = np.empty(2)

        if h_saddle > 0:
            # _p, _q is eigenvector for given eigenvalue. Follows from
            # det(M - lambda v) = 0, solve for v for a 2x2 matrix.
            pq_conversion_normal[0] = reduced_hessian[0, 1]
            pq_conversion_normal[1] = lambda_minus - reduced_hessian[0, 0]
        else:
            pq_conversion_normal[0] = reduced_hessian[0, 1]
            pq_conversion_normal[1] = lambda_plus - reduced_hessian[0, 0]

        # Ensure conversion normal points from current branch to other branch
        # i.e. they have positive projection on each other.
        pq_conversion_normal *= np.sign(
            np.dot(pq_saddle, pq_conversion_normal)
        ) / np.linalg.norm(pq_conversion_normal)

        return (h_saddle, xk_saddle, pq_saddle, pq_conversion_normal)

    def _osculating_plane_root_search_1d(
        self,
        xk_origin: FloatArray,
        xk_search: FloatArray,
        xk_root: FloatArray,
        initial_step_size: float,
        /,
        *,
        max_attempts: int = 32,
        target: float = 0.0,
        threshold: float = 1.0e-8,
    ) -> bool:
        """
        Search for root of dispersion relation in osculating plane along
        1D trajectory using Newton iteration.

        Parameters
        ----------
        xk_origin : np.array[float]
            Phase space position (x, k) of search origin.
        xk_search : np.array[float]
            Phase space position (x, k) of search direction.
        xk_root : np.array[float]
            Phase space position (x, k) of root result is stored in.
        initial_step_size : float
            Initial step size.
        max_attempts : int, optional
            Maximum number of attempts to find root.
        target : float, optional
            Target value for Hamiltonian. Default = 0.0.
        threshold : float, optional
            Threshold on allowed deviation of Hamiltonian from target.
            Default = 1.0e-8.

        Returns
        -------
        success : bool
            If root succesfully found within iterations limit.
        """
        s = initial_step_size
        success = False

        da = 0.5 / max(1, max_attempts - 1)

        for i in range(max_attempts):
            xk_root[:] = xk_origin + s * xk_search

            # Calculate hamiltonian at new position.
            self.ray.set_xk_position(xk_root[_x], xk_root[_k])
            self.ray.calculate_hamiltonian(derivatives=1, determinant=True)

            # Check if Hamiltonian is acceptable.
            h = self.ray.hamiltonian_cache.determinant.real
            if abs(h - target) < threshold:
                success = True
                break

            # Newton step towards root.
            dh_da = np.dot(
                self.ray.hamiltonian_cache.determinant.first_derivative.z[_xk],
                xk_search,
            )
            s -= (0.5 + i * da) * ((h - target) / dh_da)

        return success

    def check_mode_conversion(self):
        """
        Check for mode conversion.
        """
        # Detect mode conversion using second tensor invariant of dispersion
        # tensor. If this passes through a local minimum, two eigenvalues of
        # the dispersion relation make a closest approach.
        alarm_value = abs(
            second_tensor_invariant_3x3(
                self.ray.hamiltonian_cache.dispersion_tensor.hermitian.value
            )
        )
        self._mode_conversion_cache.update_alarm_value(alarm_value)

        if (
            not self._enable_mode_conversion
            or not self._mode_conversion_cache.mode_conversion_detected()
        ):
            return

        # In vacuum modes are degenerate so will trigger a mode conversion
        # calculation as the alarm value is always zero.
        # Similarly for unmagnetised plasmas.
        # Also set minimum on alarm value to stop it triggering on
        # oscillations due to floating point error.
        x = self.ray.hamiltonian_cache.arguments.value[Dimensions.IDX_X]

        if any((
            not self._enable_mode_conversion,
            self._plasma_vacuum_boundary_cache.previous_step_vacuum,
            self.ray.plasma_cache.unmagnetised,
            x < 0.1,  # noqa: PLR2004
            np.any(self._mode_conversion_cache.alarm_history < 1e-4),  # noqa: PLR2004
            not np.any(self._mode_conversion_cache.alarm_history > 1e-3),  # noqa: PLR2004
            self._mode_conversion_cache.alarm_history[1] > 10.0,  # noqa: PLR2004
        )):
            return

        # Save ray parameters at closest approach.
        xk_closest = self._mode_conversion_cache.xk_closest
        xk_closest[_x] = self._limiter_cache.last_position
        xk_closest[_k] = self._limiter_cache.last_wavevector

        self._mode_conversion_cache.save_n_components(
            self.ray.plasma_cache, self.ray.wave_cache
        )

        logger.info(
            "[%s] Detected mode conversion at (x, k) = %s",
            self.ray.name,
            np.array2string(xk_closest, precision=3, floatmode="fixed"),
        )

        # Save original timestep.
        original_timestep_ns = copy.copy(self.integrator.timestep)

        logger.info(
            "Mode conversion alarm minimum at (x, k) = %s",
            np.array2string(xk_closest, precision=3, floatmode="fixed"),
        )

        # Find matching point on first branch back up the ray where WKB
        # approximation was still valid.
        xk_matching = np.empty(Dimensions.xk.size)

        # Choose last ray step to match.
        idx_matching = self.ray.index - 1

        xk_matching[_x] = self.output.position[CoordinateSystem.CARTESIAN][
            idx_matching, :
        ]
        xk_matching[_k] = self.output.wavevector_per_m[idx_matching, :]

        logger.info(
            "Found matching point at (x, k) = %s",
            np.array2string(xk_matching, precision=3, floatmode="fixed"),
        )

        # Compute basis of osculating plane, the ray velocity and acceleration
        # on phase space (x, k) at the matching point.
        e_p = self._mode_conversion_cache.xk_osculating_plane[0, :]
        e_q = self._mode_conversion_cache.xk_osculating_plane[1, :]

        e_p[:] = np.einsum(
            "ij, j -> i",
            SYMPLECTIC_MATRIX_J6,
            self.ray.hamiltonian_cache.eigenvalue.first_derivative.z[_xk],
        )

        # This is naturally orthogonal in phase space to velocity.
        e_q[:] = self.ray.hamiltonian_cache.eigenvalue.first_derivative.z[_xk]

        # Normalise basis vectors such that spatial size is 1 wavelength.
        e_p *= min(
            1.0,
            self.ray.wave_cache.vacuum_wavelength_m / np.linalg.norm(e_p[_x]),
        )
        e_q *= min(
            1.0,
            self.ray.wave_cache.vacuum_wavelength_m / np.linalg.norm(e_q[_x]),
        )

        logger.info(
            "Osculating plane basis p: %s",
            np.array2string(e_p, precision=3, floatmode="fixed"),
        )
        logger.info(
            "Osculating plane basis q: %s",
            np.array2string(e_q, precision=3, floatmode="fixed"),
        )

        # (p, q) coordinates defined with closest point as origin.
        pq_matching = np.linalg.lstsq(
            self._mode_conversion_cache.xk_osculating_plane.T,
            xk_matching - xk_closest,
            rcond=None,
        )[0]

        # Find saddle point of Hamiltonian in osculating plane.
        (h_saddle, xk_saddle, pq_saddle, pq_conversion_normal) = (
            self._find_saddle_point(
                xk_closest,
                self._mode_conversion_cache.xk_osculating_plane,
                max_attempts=MAX_ATTEMPTS,
            )
        )

        # Estimate time taken to travel from matching point to saddle point.
        # Used to estimate start time of outgoing rays.
        dt_matching_to_saddle = (
            np.linalg.norm(
                xk_matching[Dimensions.slice_x] - xk_saddle[Dimensions.slice_x]
            )
            / self.ray.state_dt.velocity
        )

        # Construct normalised vector in phase space.
        xk_conversion_normal = np.empty_like(xk_saddle)
        xk_conversion_normal[:] = np.matmul(
            self._mode_conversion_cache.xk_osculating_plane.T,
            pq_conversion_normal,
        )

        # Conversion axis along which we will search for converted mode.
        pq_conversion_axis = np.empty(2)
        pq_conversion_axis[:] = np.linalg.lstsq(
            self._mode_conversion_cache.xk_osculating_plane.T,
            xk_saddle - xk_matching,
            rcond=None,
        )[0]
        pq_conversion_axis /= np.linalg.norm(pq_conversion_axis)

        xk_conversion_axis = np.matmul(
            self._mode_conversion_cache.xk_osculating_plane.T,
            pq_conversion_axis,
        )

        # Save some values at the saddle point we will use later.
        dispersion_tensor_saddle = np.copy(
            self.ray.hamiltonian_cache.dispersion_tensor.hermitian.value
        )
        dispersion_tensor_saddle_dxk = np.copy(
            self.ray.hamiltonian_cache.dispersion_tensor.hermitian.first_derivative.z[
                :, :, _xk
            ]
        )
        h_saddle_dxk2 = np.copy(
            self.ray.hamiltonian_cache.determinant.second_derivative.z[
                _xk, _xk
            ]
        )

        self._mode_conversion_cache.save_saddle_plasma_parameters(
            self.ray.plasma_cache, self.ray.wave_cache
        )

        # Search for second root of determinant along line connecting matching
        # point to the saddle point using Newton iteration.
        xk_converted = np.empty(Dimensions.xk.size)

        # Guess converted ray symmetric to matching point across saddle point.
        # Start ray on other side of saddle to hopefully it 'slides' down.
        initial_step_size = 2.0 * np.linalg.norm(pq_saddle - pq_matching)

        # If saddle point is too close to the branches of the dispersion
        # its likely we will end up on the wrong branch. Instead look
        # for a branch which is well separated from the saddle point,
        # step along that and then project back onto H = 0.
        threshold = 1.0e-8

        if abs(h_saddle) <= 1.0e4 * threshold:
            target = 1.0e-4 * np.sign(
                np.einsum(
                    "i, ij, j",
                    xk_conversion_axis,
                    h_saddle_dxk2,
                    xk_conversion_axis,
                )
            )
        else:
            # Saddle point well separated from dispersion branches. Look
            # for root H = 0.
            target = None

        _success = self._osculating_plane_root_search_1d(
            xk_matching,
            xk_conversion_axis,
            xk_converted,
            initial_step_size,
            max_attempts=MAX_ATTEMPTS,
            target=target if target is not None else 0.0,
            threshold=threshold,
        )

        if not _success:
            logger.warning(
                "Unable to find converted ray initial conditions "
                "after %s attempts",
                MAX_ATTEMPTS,
            )
            return

        logger.info(
            "Found converted ray (x, k) = %s",
            np.array2string(xk_converted, precision=3, floatmode="fixed"),
        )

        # Start time of converted ray.
        # Assume it takes same time to travel from matching point to saddle
        # point as saddle point to outgoing point.
        t_converted = (
            self.output.time_ns[idx_matching] + 2 * dt_matching_to_saddle
        )

        # If we are following a branch of nonzero H, project back onto H = 0.
        if target is not None:
            _success = self._osculating_plane_root_search_1d(
                xk_converted.copy(),
                xk_conversion_normal,
                xk_converted,
                0.01 * initial_step_size,
                max_attempts=MAX_ATTEMPTS,
                target=0.0,
                threshold=threshold,
            )

            if not _success:
                self.ray.set_stop_condition(
                    "Unable to project converted ray onto dispersion curve "
                    f"after {MAX_ATTEMPTS} attempts"
                )
                return

        # Also search for the starting point of the unconverted ray.
        xk_unconverted = np.empty(Dimensions.xk.size)

        if target is not None:
            # If we are very close to the saddle point root find the branch
            # at the matching point and follow that to avoid hopping branches.
            # Find new branch away from saddle point.
            _success = self._osculating_plane_root_search_1d(
                xk_closest,
                -xk_conversion_axis,
                xk_unconverted,
                0.01 * initial_step_size,
                max_attempts=MAX_ATTEMPTS,
                target=target,
                threshold=threshold,
            )

            if not _success:
                self.ray.set_stop_condition(
                    "Unable to find displaced branch from closest point "
                    f"after {MAX_ATTEMPTS} attempts"
                )
                return

            # Take step past the conversion point.
            self.ray.set_xk_position(xk_unconverted[_x], xk_unconverted[_k])
            self.synchronise_integrator()

            self.integrator.set_timestep(original_timestep_ns)
            self.step()

            t_converted = self.integrator.t
            xk_unconverted[_x] = self.integrator.y[_x]
            xk_unconverted[_k] = self.integrator.y[_k]

            # Project back onto H=0 branch away from saddle point.
            _success = self._osculating_plane_root_search_1d(
                xk_unconverted.copy(),
                xk_conversion_normal,
                xk_unconverted,
                0.01 * initial_step_size,
                max_attempts=MAX_ATTEMPTS,
                target=target,
                threshold=threshold,
            )

            if not _success:
                self.ray.set_stop_condition(
                    "Unable to project unconverted ray back onto H=0 "
                    f"after {MAX_ATTEMPTS} attempts"
                )
                return
        else:
            # Otherwise just take a step away from mode conversion region.
            self.ray.set_xk_position(xk_closest[_x], xk_closest[_k])
            self.synchronise_integrator()

            self.integrator.set_timestep(original_timestep_ns)
            self.step()

            t_converted = self.integrator.t
            xk_unconverted[_x] = self.integrator.y[_x]
            xk_unconverted[_k] = self.integrator.y[_k]

        logger.info(
            "Found unconverted ray (x, k): %s",
            np.array2string(xk_unconverted, precision=3, floatmode="fixed"),
        )

        # Find coupling constant.
        use_mjolhus = True

        if use_mjolhus:
            # If single ray use transmission for a beam.
            # Otherwise if a ray bundle use transmission for a plane wave.
            if self.ray.bundle:
                tau_transmission = mjolhus_plane_wave(
                    self.ray.wave_cache.vacuum_wavenumber_per_m,
                    self._mode_conversion_cache.k0ln_saddle,
                    self._mode_conversion_cache.y_saddle,
                    self._mode_conversion_cache.n_parallel,
                    self._mode_conversion_cache.n_y,
                )
            else:
                tau_transmission = mjolhus_gaussian_beam(
                    self._mode_conversion_cache.k0ln_saddle,
                    self._mode_conversion_cache.y_saddle,
                    self._mode_conversion_cache.n_parallel,
                    self._mode_conversion_cache.n_y,
                    (
                        self.ray.wave_cache.vacuum_wavenumber_per_m
                        * self.ray.beam_waist_radius_m
                    ),
                )

            phase_shift = 0.0
        else:
            # Find asymptotes of outgoing rays.
            h_star = np.einsum(
                "ij, jk -> ik", SYMPLECTIC_MATRIX_J6, h_saddle_dxk2
            )

            # Expect two dominant eigenvalues of opposite sign.
            _eigenvalues, _eigenvectors = np.linalg.eigh(h_star)
            v_alpha, v_beta = _eigenvectors[:, 0], _eigenvectors[:, 5]

            # Find asymptotic polarisations of outgoing rays from null space
            # of dispersion matrix linearised about saddle point.
            e_alpha = polarisation(
                np.einsum(
                    "ijk, k -> ij", dispersion_tensor_saddle_dxk, v_alpha
                )
            )
            e_beta = polarisation(
                np.einsum("ijk, k -> ij", dispersion_tensor_saddle_dxk, v_beta)
            )

            # Find derivative of Hamiltonian of outgoing asymptotic rays.
            h_alpha_dz = np.einsum(
                "i, ijk, j -> k",
                np.conj(e_alpha),
                dispersion_tensor_saddle_dxk,
                e_alpha,
            )

            h_beta_dz = np.einsum(
                "i, ijk, j -> k",
                np.conj(e_beta),
                dispersion_tensor_saddle_dxk,
                e_beta,
            )

            # Coupling coefficient.
            eta = np.einsum(
                "i, ij, j", np.conj(e_alpha), dispersion_tensor_saddle, e_beta
            )

            # Normalise to Poisson bracket of outgoing Hamiltonians.
            # Can be re-written in terms of velocities.
            bracket = abs(
                np.einsum(
                    "i, ij, j", h_alpha_dz, SYMPLECTIC_MATRIX_J6, h_beta_dz
                )
            )
            eta_norm = eta / np.sqrt(bracket)

            # Power transmission coefficient hence factor 2.
            tau_transmission = 2 * np.pi * abs(eta_norm * np.conj(eta_norm))
            phase_shift = 0.0

        # Find transmission coefficient of new wave.
        transmission = np.exp(-tau_transmission)

        # Discard converted ray if predicted power transfer is too low.
        logger.info("Transmission = %s", np.round(transmission, 3))

        other_mode = (
            WaveMode.X if self.ray.wave_mode == WaveMode.O else WaveMode.O
        )

        intensity_w_per_m2 = (
            self.ray.wave_cache.frequency_ghz
            * self.ray.state.wave_action_density
        )

        self.add_child(
            t_converted,
            self.ray.wave_cache.frequency_ghz,
            xk_converted[_x],
            xk_converted[_k] / self.ray.wave_cache.vacuum_wavenumber_per_m,
            self.ray.state.eikonal_phase_rad + phase_shift,
            self.ray.state.adiabatic_phase_rad,
            self.ray.hamiltonian_cache.polarisation.stix.value,
            other_mode,
            transmission * self.ray.state.initial_power_w,
            transmission * intensity_w_per_m2,
        )

        # Remove transmitted power from ray.
        # Clip to avoid log(0). Minus as log(x) < 0 as x < 1.
        self.ray.state.increment_optical_depth_external(
            -np.log(max(1.0e-100, 1 - transmission))
        )

        # Re-initialise integrator at new position.
        # Assume same time to travel to new point as it took from matching
        # point to the point of closest approach.
        self.ray.set_time(t_converted)
        self.ray.set_xk_position(
            xk_unconverted[Dimensions.slice_x],
            xk_unconverted[Dimensions.slice_k],
        )
        self.ray.calculate_hamiltonian(derivatives=0, determinant=False)

        alarm_value = abs(
            second_tensor_invariant_3x3(
                self.ray.hamiltonian_cache.dispersion_tensor.hermitian.value
            )
        )

        self._mode_conversion_cache.update_alarm_value(
            alarm_value, override=True
        )

        # Set timeout to prevent constant mode conversion calculations due to
        # numerical noise near a minimum of the alarm. Possible if timestep
        # shrinks.
        self._mode_conversion_cache.complete()

        # Reset integrator but take a smaller step just in case.
        self.integrator.set_timestep(original_timestep_ns)

        # Accept step at unconverted position.
        self.accept_step(override=True, synchronise_integrator=True)
