"""
Unit tests for ray_tracing.ray and ray_tracing.ray_tracer
"""

# Standard imports
import logging
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Third party imports
# Local imports
from crayon.coordinates import CoordinateCoordinator, CoordinateSystem
from crayon.ray_tracing.initial_conditions import InitialConditions
from crayon.ray_tracing.ray_tracer import (
    OptionsIntegrator,
    OptionsRayTracing,
    Ray,
    RayTracer,
    RayTracingOutput,
)
from crayon.shared.constants import C_M_PER_NS, WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.system_data import Kinetic, Limiters, Magnetic, SystemData
from crayon.value_model.models import ValueModel

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def system_data() -> SystemData:
    """
    System data object.

    Returns
    -------
    system_data : SystemData
        System data.
    """
    coordinate_coordinator = CoordinateCoordinator()
    coordinate_coordinator.calculate_conversion_paths()

    electron_density_per_m3 = ValueModel.electron_density_per_m3().constant(
        CoordinateSystem.CARTESIAN, 0.0
    )
    electron_temperature_ev = ValueModel.electron_density_per_m3().constant(
        CoordinateSystem.CARTESIAN, 0.0
    )
    effective_charge = ValueModel.effective_charge().constant(
        CoordinateSystem.CARTESIAN, 0.0
    )
    magnetic_field_t = ValueModel.magnetic_field_t().constant(
        CoordinateSystem.CARTESIAN, np.array([0.0, 0.0, 1.0])
    )

    kinetic = Kinetic(
        electron_density_per_m3, electron_temperature_ev, effective_charge
    )
    magnetic = Magnetic(magnetic_field_t)

    return SystemData(coordinate_coordinator, kinetic, magnetic, Limiters({}))


@pytest.fixture(scope="class")
def initial_conditions() -> InitialConditions:
    """
    Ray initial conditions.

    Returns
    -------
    initial_conditions : InitialConditions
        Initial conditions.
    """
    return InitialConditions(
        "test",
        0.0,
        1.0,
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.0,
        0.0,
        [0.0, 0.0, 1.0],
        WaveMode.O,
        1.0,
        1.0,
        1.0,
        bundle=False,
    )


class TestRay:
    """
    Unit tests for Ray.
    """

    @pytest.fixture
    @staticmethod
    def ray(
        system_data: SystemData, initial_conditions: InitialConditions
    ) -> Ray:
        """
        Ray

        Parameters
        ----------
        system_data : SystemData
            System data.
        initial_conditions : InitialConditions
            Ray initial conditions.

        Returns
        -------
        ray : Ray
            Ray.
        """
        return Ray(system_data, initial_conditions)

    @staticmethod
    def test_set(ray: Ray):
        """
        Test setting ray parameters.

        Parameters
        ----------
        ray : Ray
            Ray.
        """
        # Set time.
        ray.set_time(1.0)
        nptest.assert_allclose(ray.time_ns, 1.0)

        # Set frequency.
        ray.set_frequency(2.0)
        nptest.assert_allclose(ray.wave_cache.frequency_ghz, 2.0)
        nptest.assert_allclose(
            ray.plasma_cache._inv_critical_damping_frequency_ghz, 0.5
        )

        # Set xk.
        x = np.array([1.0, 2.0, 3.0])
        k = np.array([4.0, 5.0, 6.0])
        ray.set_xk_position(x, k)

        nptest.assert_allclose(ray.coordinate_cache.position_cartesian, x)
        nptest.assert_allclose(ray.wave_cache.wavevector_per_m, k)

        # Set xn.
        x = np.array([1.5, 2.0, 3.0])
        n = np.array([4.0, 5.0, 6.0])
        ray.set_xn_position(x, n)

        nptest.assert_allclose(ray.coordinate_cache.position_cartesian, x)
        nptest.assert_allclose(
            ray.wave_cache.wavevector_per_m,
            ray.wave_cache.vacuum_wavenumber_per_m * k,
        )

    @staticmethod
    def test_synchronise_xk(ray: Ray):
        """
        Test synchronising ray (x, k) position.

        Parameters
        ----------
        ray : Ray
            Ray.
        """
        x = [1.0, 2.0, 3.0]
        k = [4.0, 5.0, 6.0]

        # Test to state.
        ray.coordinate_cache.set_position(CoordinateSystem.CARTESIAN, x)
        ray.wave_cache.set_wavevector(k)
        ray._synchronise_xk(to_state=True)

        nptest.assert_allclose(ray.state.position_cartesian, x)
        nptest.assert_allclose(ray.state.wavevector_cartesian, k)

        # To from state.
        x = [7.0, 8.0, 9.0]
        k = [10.0, 11.0, 12.0]

        ray.state.position_cartesian[:] = x
        ray.state.wavevector_cartesian[:] = k
        ray._synchronise_xk(to_state=False)

        nptest.assert_allclose(ray.coordinate_cache.position_cartesian, x)
        nptest.assert_allclose(ray.wave_cache.wavevector_per_m, k)

    @staticmethod
    def test_calculate_hamiltonian(ray: Ray):
        """
        Test calculating Hamiltonian.

        Parameters
        ----------
        ray : Ray
            Ray.
        """
        n = 1.1

        ray.set_frequency(1.0)
        ray.set_xn_position([1.0, 0.0, 0.0], [n, 0.0, 0.0])
        ray.calculate_hamiltonian(derivatives=2, determinant=True)

        # Should be vacuum dispersion relation D = (1 - N**2)**2.
        nptest.assert_allclose(
            ray.hamiltonian_cache.eigenvalue.real, 1 - n * n
        )
        nptest.assert_allclose(
            ray.hamiltonian_cache.determinant.real, np.square(1 - n * n)
        )
        nptest.assert_allclose(
            ray.hamiltonian_cache.polarisation.stix.value,
            np.array([0.0, 0.0, 1.0]),
        )

    @staticmethod
    def test_calculate_state_vector_dt(ray: Ray):
        """
        Test calculating time derivative of state vector.

        Parameters
        ----------
        ray : Ray
            Ray.
        """
        ray.set_frequency(1.0)
        ray.set_xn_position([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        ray.calculate_hamiltonian(derivatives=2, determinant=True)
        ray.state_dt.calculate(
            ray.state, ray.plasma_cache, ray.hamiltonian_cache
        )

        nptest.assert_allclose(ray.state_dt.velocity, C_M_PER_NS)
        nptest.assert_allclose(ray.state_dt.velocity_x, [C_M_PER_NS, 0.0, 0.0])

    @staticmethod
    def test_set_stop_condition(ray: Ray):
        """
        Test setting ray tracing stop condition.

        Parameters
        ----------
        ray : Ray
            Ray.
        """
        message = "test message"
        ray.set_stop_condition(message)

        assert ray.stop_condition == message

    @staticmethod
    def test_save_step(ray: Ray):
        """
        Test saving ray step.

        Parameters
        ----------
        ray : Ray
            Ray.
        """
        ray.set_frequency(1.0)
        ray.set_xn_position([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        ray.calculate_hamiltonian(derivatives=2, determinant=False)

        ray.accept_step()
        assert ray.index == 1


class TestRayTracingOutput:
    """
    Unit tests for RayTracingOutput.
    """

    n = 50
    n_c = 3

    @pytest.fixture
    def cache(self) -> RayTracingOutput:
        """
        Ray tracing output cache.

        Returns
        -------
        cache : RayTracingOutput
            Ray tracing output.
        """
        return RayTracingOutput(
            (CoordinateSystem.CARTESIAN, CoordinateSystem.CYLINDRICAL),
            max_ray_nodes=self.n,
        )

    @staticmethod
    def test_set_initial(cache: RayTracingOutput):
        """
        Test setting initial parameters.

        Parameters
        ----------
        cache : RayTracingOutput
            Ray tracing output.
        """
        cache.set_initial(1.0, 2.0, 3.0)

        nptest.assert_allclose(cache.frequency_ghz, 1.0)
        nptest.assert_allclose(cache.vacuum_wavenumber_per_m, 2.0)
        nptest.assert_allclose(cache.initial_power_w, 3.0)

    def test_round_trip_netcdf(self, cache: RayTracingOutput):
        """
        Test writing and reading object through netCDF4 file gives same object.

        Parameters
        ----------
        cache : RayTracingOutput
            Ray tracing output.
        """
        # Fill all arrays with -1. On initialisation all array values are
        # set to zero so we will catch anything that is missed.
        for name in cache.__slots__:
            var = getattr(cache, name)

            if isinstance(var, np.ndarray):
                var.fill(-1.0)

        # Fill cache.
        cache.frequency_ghz = 1.0
        cache.vacuum_wavenumber_per_m = 2.0
        cache.initial_power_w = 3.0

        cache.position[CoordinateSystem.CARTESIAN].fill(4.0)
        cache.position[CoordinateSystem.CYLINDRICAL].fill(5.0)
        cache.wavevector_per_m.fill(6.0)
        cache.refractive_index.fill(7.0)
        cache.k_perp.fill(8.0)
        cache.k_parallel.fill(9.0)
        cache.n_perp.fill(10.0)
        cache.n_parallel.fill(11.0)

        cache.velocity.fill(46.0)
        cache.velocity_x.fill(47.0)
        cache.velocity_k.fill(48.0)

        cache.electron_density_per_m3.fill(12.0)
        cache.electron_temperature_ev.fill(13.0)
        cache.effective_charge.fill(14.0)
        cache.magnetic_field_t.fill(15.0)
        cache.magnetic_field_strength_t.fill(16.0)

        cache.normalised_electron_density.fill(17.0)
        cache.normalised_electron_temperature.fill(18.0)
        cache.normalised_collision_rate.fill(19.0)
        cache.normalised_magnetic_field_strength.fill(20.0)

        cache.eigenvalue.fill(49.0 + 49.0j)
        cache.determinant.fill(50.0 + 50.0j)
        cache.eigenvalue_error_frequency.fill(51.0)
        cache.determinant_error_frequency.fill(52.0)

        cache.time_ns.fill(22.0)
        cache.arc_length_m.fill(23.0)
        cache.eikonal_phase_rad.fill(24.0)
        cache.adiabatic_phase_rad.fill(25.0)
        cache.phase_rad.fill(26.0)

        cache.optical_depth.fill(28.0)
        cache.optical_depth_internal.fill(29.0)
        cache.optical_depth_external.fill(30.0)

        cache.power_w.fill(31.0)
        cache.damping_fraction_resonance.fill(32.0)
        cache.damping_fraction_collisional.fill(33.0)
        cache.cumulative_damped_power_w.fill(34.0)
        cache.cumulative_damped_power_resonance_w.fill(35.0)
        cache.cumulative_damped_power_collisional_w.fill(36.0)
        cache.cumulative_damped_power_external_w.fill(37.0)
        cache.polarisation_cartesian.fill(38.0)
        cache.polarisation_stix.fill(39.0)
        cache.polarisation_right_handed.fill(40.0)
        cache.polarisation_left_handed.fill(41.0)
        cache.polarisation_parallel.fill(42.0)

        cache.magnification_x.fill(43.0)
        cache.focusing_tensor_x.fill(44.0)
        cache.intensity_amplification.fill(45.0)

        cache.wkb_validity.fill(46.0)
        cache.k0ln_saddle.fill(47.0)
        cache.y_saddle.fill(48.0)
        cache.n_parallel_at_conversion.fill(49.0)
        cache.n_y_at_conversion.fill(50.0)
        cache.closest_to_conversion.fill(51.0)
        cache.saddle_at_conversion.fill(52.0)
        cache.osculating_plane_basis.fill(53.0)

        with (
            tempfile.TemporaryFile("r+") as f,
            nc4.Dataset(f, "w", auto_complex=True) as dset,
        ):
            Dimensions.write_netcdf(dset)
            cache.write_netcdf(dset, self.n, self.n_c)
            cache2 = RayTracingOutput.read_netcdf(dset)

        c, c2 = cache, cache2

        nptest.assert_allclose(c.frequency_ghz, c2.frequency_ghz)
        nptest.assert_allclose(
            c.vacuum_wavenumber_per_m, c2.vacuum_wavenumber_per_m
        )
        nptest.assert_allclose(c.initial_power_w, c2.initial_power_w)

        nptest.assert_allclose(
            c.position[CoordinateSystem.CARTESIAN],
            c2.position[CoordinateSystem.CARTESIAN],
        )
        nptest.assert_allclose(
            c.position[CoordinateSystem.CYLINDRICAL],
            c2.position[CoordinateSystem.CYLINDRICAL],
        )
        nptest.assert_allclose(c.wavevector_per_m, c2.wavevector_per_m)
        nptest.assert_allclose(c.refractive_index, c2.refractive_index)
        nptest.assert_allclose(c.k_perp, c2.k_perp)
        nptest.assert_allclose(c.k_parallel, c2.k_parallel)
        nptest.assert_allclose(c.n_perp, c2.n_perp)
        nptest.assert_allclose(c.n_parallel, c2.n_parallel)

        nptest.assert_allclose(
            c.electron_density_per_m3, c2.electron_density_per_m3
        )
        nptest.assert_allclose(
            c.electron_temperature_ev, c2.electron_temperature_ev
        )
        nptest.assert_allclose(c.effective_charge, c2.effective_charge)
        nptest.assert_allclose(c.magnetic_field_t, c2.magnetic_field_t)
        nptest.assert_allclose(
            c.magnetic_field_strength_t, c2.magnetic_field_strength_t
        )

        nptest.assert_allclose(
            c.normalised_electron_density, c2.normalised_electron_density
        )
        nptest.assert_allclose(
            c.normalised_electron_temperature,
            c2.normalised_electron_temperature,
        )
        nptest.assert_allclose(
            c.normalised_collision_rate, c2.normalised_collision_rate
        )
        nptest.assert_allclose(
            c.normalised_magnetic_field_strength,
            c2.normalised_magnetic_field_strength,
        )

        nptest.assert_allclose(c.eigenvalue, c2.eigenvalue)
        nptest.assert_allclose(c.determinant, c2.determinant)
        nptest.assert_allclose(
            c.eigenvalue_error_frequency, c2.eigenvalue_error_frequency
        )
        nptest.assert_allclose(
            c.determinant_error_frequency, c2.determinant_error_frequency
        )

        nptest.assert_allclose(c.time_ns, c2.time_ns)
        nptest.assert_allclose(c.arc_length_m, c2.arc_length_m)
        nptest.assert_allclose(c.eikonal_phase_rad, c2.eikonal_phase_rad)
        nptest.assert_allclose(c.adiabatic_phase_rad, c2.adiabatic_phase_rad)
        nptest.assert_allclose(c.phase_rad, c2.phase_rad)
        nptest.assert_allclose(c.initial_power_w, c2.initial_power_w)
        nptest.assert_allclose(c.optical_depth, c2.optical_depth)
        nptest.assert_allclose(
            c.optical_depth_internal, c2.optical_depth_internal
        )
        nptest.assert_allclose(
            c.optical_depth_external, c2.optical_depth_external
        )

        nptest.assert_allclose(c.damping_rate, c2.damping_rate)
        nptest.assert_allclose(c.power_w, c2.power_w)

        nptest.assert_allclose(
            c.damping_fraction_resonance, c2.damping_fraction_resonance
        )
        nptest.assert_allclose(
            c.damping_fraction_collisional, c2.damping_fraction_collisional
        )
        nptest.assert_allclose(
            c.cumulative_damped_power_w, c2.cumulative_damped_power_w
        )
        nptest.assert_allclose(
            c.cumulative_damped_power_resonance_w,
            c2.cumulative_damped_power_resonance_w,
        )
        nptest.assert_allclose(
            c.cumulative_damped_power_collisional_w,
            c2.cumulative_damped_power_collisional_w,
        )
        nptest.assert_allclose(
            c.cumulative_damped_power_external_w,
            c2.cumulative_damped_power_external_w,
        )
        nptest.assert_allclose(
            c.polarisation_cartesian, c2.polarisation_cartesian
        )
        nptest.assert_allclose(c.polarisation_stix, c2.polarisation_stix)
        nptest.assert_allclose(
            c.polarisation_right_handed, c2.polarisation_right_handed
        )
        nptest.assert_allclose(
            c.polarisation_left_handed, c2.polarisation_left_handed
        )
        nptest.assert_allclose(
            c.polarisation_parallel, c2.polarisation_parallel
        )
        nptest.assert_allclose(c.magnification_x, c2.magnification_x)
        nptest.assert_allclose(c.focusing_tensor_x, c2.focusing_tensor_x)
        nptest.assert_allclose(
            c.intensity_amplification, c2.intensity_amplification
        )


class TestRayTracer:
    """
    Unit tests for RayTracer.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def options_ray_tracing() -> OptionsRayTracing:
        """
        Ray tracing options.

        Returns
        -------
        options_ray_tracing : OptionsRayTracing
            Ray tracing options.
        """
        return OptionsRayTracing()

    @pytest.fixture(scope="class")
    @staticmethod
    def options_integrator() -> OptionsIntegrator:
        """
        Options for integrator.

        Returns
        -------
        options_integrator : OptionsIntegrator
            Options for integrator.
        """
        return OptionsIntegrator()

    @pytest.fixture
    @staticmethod
    def ray_tracer(
        system_data: SystemData,
        initial_conditions: InitialConditions,
        options_ray_tracing: OptionsRayTracing,
        options_integrator: OptionsIntegrator,
    ) -> RayTracer:
        """
        Ray tracer.

        Parameters
        ----------
        system_data : SystemData
            System data.
        initial_conditions : InitialConditions
            Ray initial conditions.
        options_ray_tracing : OptionsRayTracing
            Ray tracing options.
        options_integrator : OptionsIntegrator
            Options for integrator.

        Returns
        -------
        ray_tracer : RayTracer
            Ray tracer.
        """
        rt = RayTracer(system_data, options_ray_tracing, options_integrator)
        rt.ray = Ray(system_data, initial_conditions)

        return rt

    @staticmethod
    def test_check_stop_conditions(ray_tracer: RayTracer):
        """
        Test setting ray tracing stop condition.

        Parameters
        ----------
        ray_tracer : RayTracer
            Ray tracer.
        """
        # Check stop on max ray nodes.
        ray_tracer.ray.stop_condition = ""
        ray_tracer._options_ray_tracing.max_ray_nodes = 5
        ray_tracer.ray.index = 6
        ray_tracer.check_stop_conditions()

        assert len(ray_tracer.ray.stop_condition) > 0
        assert ray_tracer.ray.stop_condition.startswith(
            "Reached max ray nodes"
        )

        ray_tracer.ray.index = 0

        # Check stop on optical depth.
        ray_tracer.ray.stop_condition = ""
        ray_tracer._options_ray_tracing.max_optical_depth = 1.0
        ray_tracer.ray.state.optical_depth_internal = 2.0
        ray_tracer.check_stop_conditions()

        assert len(ray_tracer.ray.stop_condition) > 0
        assert ray_tracer.ray.stop_condition.startswith(
            "Optical depth too large"
        )

        ray_tracer.ray.state.optical_depth_internal = 0.0
