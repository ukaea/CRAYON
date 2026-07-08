"""
Unit tests for ray_tracing.caches.hamiltonian.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import CoordinateCoordinator, CoordinateSystem
from crayon.ray_tracing.caches.coordinates import CoordinateCache
from crayon.ray_tracing.caches.hamiltonian import (
    DispersionType,
    HamiltonianCache,
)
from crayon.ray_tracing.caches.plasma import PlasmaCache
from crayon.ray_tracing.caches.wave import WaveCache
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.system_data import Kinetic, Limiters, Magnetic, SystemData
from crayon.value_model.models import ValueModel

logger = logging.getLogger(__name__)

_x = Dimensions.slice_x
_k = Dimensions.slice_k
_xk = Dimensions.slice_xk
_f = Dimensions.slice_f
_t = Dimensions.slice_t

_X = Dimensions.IDX_X
_Y = Dimensions.IDX_Y
_Z = Dimensions.IDX_Z
_THETA = Dimensions.IDX_THETA
_N_PERP = Dimensions.IDX_N_PERP
_N_PAR = Dimensions.IDX_N_PARALLEL


class TestHamiltonianCache:
    """
    Unit tests for HamiltonianCache.
    """

    f_ghz = 2.0

    @pytest.fixture(scope="class")
    @staticmethod
    def system_data() -> SystemData:
        """
        System data object.

        Returns
        -------
        system_data : SystemData
            System data.
        """
        # Coordinates.
        coordinate_coordinator = CoordinateCoordinator()
        coordinate_coordinator.calculate_conversion_paths()

        # Kinetic models.
        electron_density_per_m3 = (
            ValueModel.electron_density_per_m3().quadratic_well(
                [0.0, 0.0, 0.0], 1.0e18, 1.0e19, 0.1
            )
        )

        electron_temperature_ev = (
            ValueModel.electron_temperature_ev().quadratic_well(
                [0.0, 0.0, 0.0], 1.0e2, 1.0e3, 0.12
            )
        )

        effective_charge = ValueModel.effective_charge().quadratic_well(
            [0.0, 0.0, 0.0], 1.0, 1.3, 0.15
        )

        kinetic = Kinetic(
            electron_density_per_m3, electron_temperature_ev, effective_charge
        )

        # Magnetic models.
        magnetic_field_t = ValueModel.magnetic_field_t().quadratic_well(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 2.0, 1.0], 0.2
        )

        magnetic = Magnetic(magnetic_field_t)

        # Limiters.
        limiters = Limiters({})

        return SystemData(coordinate_coordinator, kinetic, magnetic, limiters)

    @pytest.fixture(scope="class")
    def plasma_cache(self, system_data: SystemData) -> PlasmaCache:
        """
        Plasma parameter cache.

        Parameters
        ----------
        system_data : SystemData
            System data.

        Returns
        -------
        plasma_cache : PlasmaCache
            Plasma cache.
        """
        coordinate_cache = CoordinateCache(system_data.coordinate_coordinator)
        coordinate_cache.set_position(CoordinateSystem.CARTESIAN, np.ones(3))
        coordinate_cache.calculate_transforms()

        cache = PlasmaCache(system_data.kinetic, system_data.magnetic)
        cache.set_frequency(self.f_ghz)
        cache.calculate(coordinate_cache, derivatives=2)

        return cache

    @pytest.fixture(scope="class")
    def wave_cache(self, plasma_cache: PlasmaCache) -> WaveCache:
        """
        Wave parameter cache.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Plasma cache.

        Returns
        -------
        wave_cache : WaveCache
            Wave cache.
        """
        wc = WaveCache()
        wc.set_frequency(self.f_ghz)
        wc.set_refractive_index(np.linspace(0, 1, 3))
        wc.calculate_k_components(
            plasma_cache.magnetic_field_unit, derivatives=2
        )

        return wc

    def test_set_arguments(
        self, plasma_cache: PlasmaCache, wave_cache: WaveCache
    ):
        """
        Test setting Hamiltonian arguments.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Plasma cache.
        wave_cache : WaveCache
            Wave cache.
        """
        cache = HamiltonianCache()
        cache.set_hamiltonian_arguments(
            plasma_cache, wave_cache, derivatives=2
        )

        # Check values set correctly.
        _q = cache.arguments.value
        _q_dz = cache.arguments.first_derivative
        _q_dz2 = cache.arguments.second_derivative

        # Check x, k derivatives.
        # Density.
        nptest.assert_allclose(
            _q[_X], plasma_cache.normalised_electron_density.value
        )

        nptest.assert_allclose(
            _q_dz[_X, _x],
            plasma_cache.normalised_electron_density.first_derivative,
        )
        nptest.assert_allclose(_q_dz[_X, _k], 0.0)

        nptest.assert_allclose(
            _q_dz2[_X, _x, _x],
            plasma_cache.normalised_electron_density.second_derivative,
        )
        nptest.assert_allclose(_q_dz2[_X, _x, _k], 0.0)
        nptest.assert_allclose(_q_dz2[_X, _k, _x], 0.0)
        nptest.assert_allclose(_q_dz2[_X, _k, _k], 0.0)

        # Magnetic field strength.
        nptest.assert_allclose(
            _q[_Y], plasma_cache.normalised_magnetic_field_strength.value
        )

        nptest.assert_allclose(
            _q_dz[_Y, _x],
            plasma_cache.normalised_magnetic_field_strength.first_derivative,
        )
        nptest.assert_allclose(_q_dz[_Y, _k], 0.0)

        nptest.assert_allclose(
            _q_dz2[_Y, _x, _x],
            plasma_cache.normalised_magnetic_field_strength.second_derivative,
        )
        nptest.assert_allclose(_q_dz2[_Y, _x, _k], 0.0)
        nptest.assert_allclose(_q_dz2[_Y, _k, _x], 0.0)
        nptest.assert_allclose(_q_dz2[_Y, _k, _k], 0.0)

        # Temperature.
        nptest.assert_allclose(
            _q[_THETA], plasma_cache.normalised_electron_temperature.value
        )

        nptest.assert_allclose(
            _q_dz[_THETA, _x],
            plasma_cache.normalised_electron_temperature.first_derivative,
        )
        nptest.assert_allclose(_q_dz[_THETA, _k], 0.0)

        nptest.assert_allclose(
            _q_dz2[_THETA, _x, _x],
            plasma_cache.normalised_electron_temperature.second_derivative,
        )
        nptest.assert_allclose(_q_dz2[_THETA, _x, _k], 0.0)
        nptest.assert_allclose(_q_dz2[_THETA, _k, _x], 0.0)
        nptest.assert_allclose(_q_dz2[_THETA, _k, _k], 0.0)

        # Collision rate.
        nptest.assert_allclose(
            _q[_Z], plasma_cache.normalised_collision_rate.value
        )

        nptest.assert_allclose(
            _q_dz[_Z, _x],
            plasma_cache.normalised_collision_rate.first_derivative,
        )
        nptest.assert_allclose(_q_dz[_Z, _k], 0.0)

        nptest.assert_allclose(
            _q_dz2[_Z, _x, _x],
            plasma_cache.normalised_collision_rate.second_derivative,
        )
        nptest.assert_allclose(_q_dz2[_Z, _x, _k], 0.0)
        nptest.assert_allclose(_q_dz2[_Z, _k, _x], 0.0)
        nptest.assert_allclose(_q_dz2[_Z, _k, _k], 0.0)

        # N perp.
        k0 = wave_cache.vacuum_wavenumber_per_m

        nptest.assert_allclose(_q[_N_PERP], wave_cache.k_perp.value / k0)

        nptest.assert_allclose(
            _q_dz[_N_PERP, _xk], wave_cache.k_perp.first_derivative / k0
        )

        nptest.assert_allclose(
            _q_dz2[_N_PERP, _xk, _xk], wave_cache.k_perp.second_derivative / k0
        )

        # N parallel.
        nptest.assert_allclose(_q[_N_PAR], wave_cache.k_parallel.value / k0)

        nptest.assert_allclose(
            _q_dz[_N_PAR, _xk], wave_cache.k_parallel.first_derivative / k0
        )

        nptest.assert_allclose(
            _q_dz2[_N_PAR, _xk, _xk],
            wave_cache.k_parallel.second_derivative / k0,
        )

        # All time derivatives should be zero.
        nptest.assert_allclose(_q_dz[:, _t], 0.0)
        nptest.assert_allclose(_q_dz2[:, _t, _t], 0.0)

        # Test frequency derivatives.
        def q(f):
            nonlocal plasma_cache, wave_cache

            h = self.f_ghz / f
            h2 = h * h

            normalised_density = (
                plasma_cache.normalised_electron_density.value * h2
            )
            normalised_magnetic_field_strength = (
                plasma_cache.normalised_magnetic_field_strength.value * h
            )
            normalised_collision_rate = (
                plasma_cache.normalised_collision_rate.value * h
            )
            theta = plasma_cache.normalised_electron_temperature.value
            n_perp = (wave_cache.k_perp.value / k0) * h
            n_parallel = (wave_cache.k_parallel.value / k0) * h

            return np.array([
                normalised_density.item(),
                normalised_magnetic_field_strength.item(),
                normalised_collision_rate.item(),
                theta.item(),
                n_perp.item(),
                n_parallel.item(),
            ])

        expected_value = _q_dz[:, _f]
        actual_value = first_derivative_finite_difference(self.f_ghz, q, (6,))[
            :, 0
        ]

        nptest.assert_allclose(expected_value, actual_value)

        expected_value = _q_dz2[:, _f, _f]
        actual_value = second_derivative_finite_difference(
            self.f_ghz, q, (6,)
        )[:, 0, 0]

        nptest.assert_allclose(expected_value, actual_value)

        def dq_dxk(f):
            nonlocal plasma_cache, wave_cache

            h = self.f_ghz / f
            h2 = h * h

            normalised_density_dx = (
                plasma_cache.normalised_electron_density.first_derivative * h2
            )
            normalised_magnetic_field_strength_dx = (
                plasma_cache.normalised_magnetic_field_strength.first_derivative
                * h
            )
            normalised_collsion_rate_dx = (
                plasma_cache.normalised_collision_rate.first_derivative * h
            )
            dtheta_dx = (
                plasma_cache.normalised_electron_temperature.first_derivative
            )

            dn_perp_dxk = (wave_cache.k_perp.first_derivative / k0) * h
            dn_parallel_dxk = (wave_cache.k_parallel.first_derivative / k0) * h

            return_array = np.zeros((6, 6))

            return_array[_X, _x] = normalised_density_dx
            return_array[_Y, _x] = normalised_magnetic_field_strength_dx
            return_array[_Z, _x] = normalised_collsion_rate_dx
            return_array[_THETA, _x] = dtheta_dx
            return_array[_N_PERP, :] = dn_perp_dxk
            return_array[_N_PAR, :] = dn_parallel_dxk

            return return_array

        actual_value = first_derivative_finite_difference(
            self.f_ghz, dq_dxk, (6, 6)
        )[:, :, 0]

        logger.warning(abs(_q_dz2[:, _xk, _f] - actual_value))
        nptest.assert_allclose(_q_dz2[:, _xk, _f], actual_value)
        nptest.assert_allclose(_q_dz2[:, _f, _xk], actual_value)

    def test_calculate(self, plasma_cache: PlasmaCache, wave_cache: WaveCache):
        """
        Test calculating quantities on Hamiltonian cache.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Plasma cache.
        wave_cache : WaveCache
            Wave cache.
        """
        cache = HamiltonianCache()
        cache.set_hamiltonian_arguments(
            plasma_cache, wave_cache, derivatives=2
        )

        cache.calculate_recommended_models()
        cache.calculate_dispersion_tensor(
            DispersionType.COLD, DispersionType.COLD, derivatives=2
        )

        cache.calculate_stix_polarisation(
            plasma_cache, wave_cache, derivatives=1
        )
        cache.calculate_eigenvalue(derivatives=2)
        cache.calculate_determinant(derivatives=2)
        cache.calculate_normalised_em_flux(self.f_ghz)
