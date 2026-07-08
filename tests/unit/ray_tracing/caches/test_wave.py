"""
Unit tests for ray_tracing.caches.wave.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus import (
    component_parallel,
    component_perp,
    v_parallel_first_derivative_v,
    v_parallel_first_derivative_x,
    v_parallel_second_derivative_v,
    v_parallel_second_derivative_x,
    v_parallel_second_derivative_xv,
    v_perp_first_derivative_v,
    v_perp_first_derivative_x,
    v_perp_second_derivative_v,
    v_perp_second_derivative_x,
    v_perp_second_derivative_xv,
)
from crayon.ray_tracing.caches.base import DerivativeCacheX
from crayon.ray_tracing.caches.wave import WaveCache
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


class TestWaveCache:
    """
    Unit tests for WaveCache.
    """

    @staticmethod
    def magnetic_field_unit(position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Magnetic field unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        magnetic_field_unit : np.ndarray[float]
            Magnetic field unit vector.
        """
        x, _, _ = position

        b = np.zeros(3)
        b[0] = np.cos(x)
        b[1] = np.sin(x)

        return b

    @staticmethod
    def magnetic_field_unit_dx(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        First derivative of magnetic field unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        magnetic_field_unit_dx : np.ndarray[float]
            Magnetic field unit vector first derivative.
        """
        x, _, _ = position

        b = np.zeros((3, 3))

        b[0, 0] = -np.sin(x)
        b[1, 0] = np.cos(x)

        return b

    @staticmethod
    def magnetic_field_unit_dx2(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Second derivative of magnetic field unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        magnetic_field_unit_dx2 : np.ndarray[float]
            Magnetic field unit vector second derivative.
        """
        x, _, _ = position

        b = np.zeros((3, 3, 3))

        b[0, 0, 0] = -np.cos(x)
        b[1, 0, 0] = -np.sin(x)

        return b

    positions = (
        np.array([1.0, -1.0, 0.5]),
        np.array([0.2, -0.5, 0.4]),
        np.array([-0.4, 0.2, 0.1]),
    )

    @pytest.mark.parametrize("position", positions)
    def test_derivatives(self, position: np.ndarray[float]):
        """
        Test derivatives of magnetic field unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        """
        # First derivative.
        expected_value = self.magnetic_field_unit_dx(position)
        actual_value = first_derivative_finite_difference(
            position, self.magnetic_field_unit, (3,)
        )

        nptest.assert_allclose(expected_value, actual_value)

        # First derivative.
        expected_value = self.magnetic_field_unit_dx2(position)
        actual_value = second_derivative_finite_difference(
            position, self.magnetic_field_unit, (3,)
        )

        nptest.assert_allclose(expected_value, actual_value)

    def magnetic_field_unit_cache(
        self, position: np.ndarray[float]
    ) -> DerivativeCacheX:
        """
        Cache for magnetic field unit values and derivatives.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        cache : DerivativeCacheX
            Cache for magnetic field unit vector.
        """
        cache = DerivativeCacheX((Dimensions.x.size,))

        cache.value[:] = self.magnetic_field_unit(position)
        cache.first_derivative[:, :] = self.magnetic_field_unit_dx(position)
        cache.second_derivative[:, :] = self.magnetic_field_unit_dx2(position)

        return cache

    @pytest.fixture(scope="class")
    @staticmethod
    def cache() -> WaveCache:
        """
        Wave parameter cache.

        Returns
        -------
        wave_cache : WaveCache
            Wave cache.
        """
        return WaveCache()

    test_positions = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5 * np.pi, 0.0, 0.0]),
        np.array([np.pi, 0.0, 0.0]),
    )

    @pytest.mark.parametrize("position", test_positions)
    def test_calculate(self, cache: WaveCache, position: np.ndarray[float]):
        """
        Test calculation of wave cache parameters.

        Parameters
        ----------
        cache : WaveCache
            Wave cache.
        position : np.ndarray[float]
            Position.
        """
        _frequency_ghz = 1.0
        _refractive_index = np.array([1.0, 0.0, 0.0])
        _magnetic_field_unit_cache = self.magnetic_field_unit_cache(position)

        cache.set_frequency(_frequency_ghz)
        cache.set_refractive_index(_refractive_index)
        cache.calculate_k_components(_magnetic_field_unit_cache, derivatives=2)

        # Test refractive index.
        nptest.assert_allclose(_refractive_index, cache.refractive_index)

        _xk = Dimensions.xk.size
        _x, _k = Dimensions.slice_x, Dimensions.slice_k

        k = cache.wavevector_per_m
        b = _magnetic_field_unit_cache.value
        db_dx = _magnetic_field_unit_cache.first_derivative
        db_dx2 = _magnetic_field_unit_cache.second_derivative

        # Test parallel wavevector components.
        expected_value = component_parallel(k, b)
        actual_value = cache.k_parallel.value

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = np.empty(_xk)
        expected_value[_x] = v_parallel_first_derivative_x(k, db_dx)
        expected_value[_k] = v_parallel_first_derivative_v(k, b)

        actual_value = cache.k_parallel.first_derivative

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = np.empty((_xk, _xk))
        expected_value[_x, _x] = v_parallel_second_derivative_x(k, db_dx2)
        expected_value[_k, _k] = v_parallel_second_derivative_v(k)
        expected_value[_x, _k] = v_parallel_second_derivative_xv(k, db_dx)
        expected_value[_k, _x] = expected_value[_x, _k].T

        actual_value = cache.k_parallel.second_derivative

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test perpendicular wavevector components.
        expected_value = component_perp(
            cache.wavevector_per_m, _magnetic_field_unit_cache.value
        )

        actual_value = cache.k_perp.value

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = np.empty(_xk)
        expected_value[_x] = v_perp_first_derivative_x(k, b, db_dx)
        expected_value[_k] = v_perp_first_derivative_v(k, b)

        actual_value = cache.k_perp.first_derivative

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = np.empty((_xk, _xk))
        expected_value[_x, _x] = v_perp_second_derivative_x(
            k, b, db_dx, db_dx2
        )
        expected_value[_k, _k] = v_perp_second_derivative_v(k, b)
        expected_value[_x, _k] = v_perp_second_derivative_xv(k, b, db_dx)
        expected_value[_k, _x] = expected_value[_x, _k].T

        actual_value = cache.k_perp.second_derivative

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test Stix frame vectors are orthonormal.
        ex = cache._stix_frame_rotation[:, 0]
        ey = cache._stix_frame_rotation[:, 1]
        ez = cache._stix_frame_rotation[:, 2]

        nptest.assert_allclose(np.linalg.norm(ex), 1.0, atol=1e-8)
        nptest.assert_allclose(np.linalg.norm(ey), 1.0, atol=1e-8)
        nptest.assert_allclose(np.linalg.norm(ez), 1.0, atol=1e-8)

        nptest.assert_allclose(np.dot(ex, ey), 0.0, atol=1e-8)
        nptest.assert_allclose(np.dot(ex, ez), 0.0, atol=1e-8)
        nptest.assert_allclose(np.dot(ey, ez), 0.0, atol=1e-8)

        # Test Stix frame vectors are right handed set.
        nptest.assert_allclose(np.dot(ex, np.cross(ey, ez)), 1.0, atol=1e-8)
