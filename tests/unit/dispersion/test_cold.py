"""
Unit tests for dispersion.cold
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.dispersion import ColdDispersion, SusceptibilityCache
from crayon.shared.constants import WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)

dimensions = Dimensions()


class TestColdDispersion:
    """
    Unit tests for ColdDispersion.
    """

    q_values = (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.2, 0.4, 0.3, 0.5, 0.2, 0.7]),
        np.array([0.3, 0.6, 0.9, 0.6, 0.2, -0.3]),
    )

    @staticmethod
    @pytest.mark.parametrize("q", q_values)
    def test_chi_derivatives(q: np.ndarray[float]):
        """
        Test first derivative of susceptibility tensor with respect to q.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        # Test first derivative.
        expected_value = ColdDispersion.susceptibility_hermitian_dq(q)
        actual_value = first_derivative_finite_difference(
            q, ColdDispersion.susceptibility_hermitian, (3, 3), is_complex=True
        )

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value)

        # Second derivative.
        expected_value = ColdDispersion.susceptibility_hermitian_dq2(q)
        actual_value = second_derivative_finite_difference(
            q, ColdDispersion.susceptibility_hermitian, (3, 3), is_complex=True
        )

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("q", q_values)
    def test_calculate_susceptibility(q: np.ndarray[float]):
        """
        Test calcultion of multiple values in calculate_susceptibility.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        cache = SusceptibilityCache()

        ColdDispersion.calculate_susceptibility(
            q, cache, 2, hermitian=True, antihermitian=True
        )

        expected_value = ColdDispersion.susceptibility_hermitian(q)
        actual_value = cache.hermitian
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = ColdDispersion.susceptibility_hermitian_dq(q)
        actual_value = cache.hermitian_dq
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = ColdDispersion.susceptibility_hermitian_dq2(q)
        actual_value = cache.hermitian_dq2
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = ColdDispersion.susceptibility_antihermitian_resonance(
            q
        )
        actual_value = cache.antihermitian_resonance
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = (
            ColdDispersion.susceptibility_antihermitian_collisional(q)
        )
        actual_value = cache.antihermitian_collisional
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def test_root_finding():
        """
        Test root finding of dispersion relation.
        """
        # X, Y, Z, theta, N_perp, N_parallel.
        q = np.zeros(6)

        # Vacuum mode.
        q[5] = 1.0

        result = ColdDispersion.calculate_n(q, WaveMode.O)
        assert not result.message
        nptest.assert_allclose(result.value, 1.0)

        result = ColdDispersion.calculate_n(q, WaveMode.X)
        assert not result.message
        nptest.assert_allclose(result.value, 1.0)

        result = ColdDispersion.calculate_n_perp(q, WaveMode.O)
        assert not result.message
        nptest.assert_allclose(result.value, 0.0)

        result = ColdDispersion.calculate_n_perp(q, WaveMode.X)
        assert not result.message
        nptest.assert_allclose(result.value, 0.0)

        q[5] = 0.6

        result = ColdDispersion.calculate_n_perp(q, WaveMode.O)
        assert not result.message
        nptest.assert_allclose(result.value, 0.8)

        result = ColdDispersion.calculate_n_perp(q, WaveMode.X)
        assert not result.message
        nptest.assert_allclose(result.value, 0.8)

        # O mode at cutoff.
        q[0] = 1.0  # X
        q[1] = 0.6  # Y
        q[4] = 1.0  # N_perp
        q[5] = 0.0  # N_parallel

        result = ColdDispersion.calculate_n(q, WaveMode.O)
        assert not result.message
        nptest.assert_allclose(result.value, 0.0)

        result = ColdDispersion.calculate_n_perp(q, WaveMode.O)
        assert not result.message
        nptest.assert_allclose(result.value, 0.0)

        # X mode at right hand cutoff.
        q[0] = 0.4  # X
        q[1] = 0.6  # Y

        result = ColdDispersion.calculate_n(q, WaveMode.X)
        assert not result.message
        nptest.assert_allclose(result.value, 0.0)

        result = ColdDispersion.calculate_n_perp(q, WaveMode.X)
        assert not result.message
        nptest.assert_allclose(result.value, 0.0)

        # Arbitrary value.
        theta = np.pi / 3
        s, c = np.sin(theta), np.cos(theta)

        q[0] = 0.4
        q[1] = 0.3
        q[4] = s
        q[5] = c

        # O mode root find |N|.
        result_n = ColdDispersion.calculate_n(q, WaveMode.O)
        assert not result_n.message
        q[4] = s * result_n.value
        q[5] = c * result_n.value

        nptest.assert_allclose(
            np.linalg.det(ColdDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        # O mode root find N_perp.
        result_n_perp = ColdDispersion.calculate_n_perp(q, WaveMode.O)
        assert not result_n_perp.message
        q[4] = result_n_perp.value

        nptest.assert_allclose(
            np.linalg.det(ColdDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        # X mode root find |N|.
        result_n = ColdDispersion.calculate_n(q, WaveMode.X)
        assert not result.message
        q[4] = s * result_n.value
        q[5] = c * result_n.value

        nptest.assert_allclose(
            np.linalg.det(ColdDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        # X mode root find N_perp.
        result_n_perp = ColdDispersion.calculate_n_perp(q, WaveMode.X)
        assert not result_n_perp.message
        q[4] = result_n_perp.value

        nptest.assert_allclose(
            np.linalg.det(ColdDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )
