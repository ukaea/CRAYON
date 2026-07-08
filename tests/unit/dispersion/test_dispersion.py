"""
Unit tests for dispersion.base.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.dispersion.base import (
    calculate_harmonic_range,
    determinant,
    determinant_dx,
    determinant_dx2,
    eigenvalue,
    eigenvalue_dx,
    eigenvalue_dx2,
    eigenvector_dx,
    polarisation,
    polarisation_phase_convention_factor,
    vacuum_dispersion_tensor,
    vacuum_dispersion_tensor_dq,
    vacuum_dispersion_tensor_dq2,
    vacuum_stix_polarisation,
)
from crayon.shared.constants import WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


class TestDispersion:
    """
    Unit tests for various functions from dispersion.base.
    """

    q_values = (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.2, 0.4, 0.3, 0.5, 0.2, 0.7]),
        np.array([0.3, 0.6, 0.9, 0.6, 0.2, -0.3]),
    )

    @staticmethod
    @pytest.mark.parametrize("q", q_values)
    def test_vacuum_dispersion_tensor_derivatives(q):
        """
        Test derivatives of vacuum dispersion tensor with respect to q.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        # First derivative.
        expected_value = vacuum_dispersion_tensor_dq(q)
        actual_value = first_derivative_finite_difference(
            q, vacuum_dispersion_tensor, (3, 3), is_complex=True
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Second derivative.
        expected_value = vacuum_dispersion_tensor_dq2()
        actual_value = second_derivative_finite_difference(
            q, vacuum_dispersion_tensor, (3, 3), is_complex=True
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    @staticmethod
    def test_polarisation():
        """
        Test calculation of polarisation.
        """
        # Test polarisation satisfies m . E = 0.
        m = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        expected_value = np.array([1.0, 0.0, 0.0])
        actual_value = polarisation(m)

        nptest.assert_allclose(expected_value, actual_value)

        # Test polarisation phase convention.
        actual_value = np.array([1.0j, 0.0, 0.0])
        expected_value = np.array([1.0, 0.0, 0.0])
        actual_value *= polarisation_phase_convention_factor(actual_value)

        nptest.assert_allclose(expected_value, actual_value)

        actual_value = np.array([0.0, -1.0j, 0.0])
        expected_value = np.array([0.0, 1.0, 0.0])
        actual_value *= polarisation_phase_convention_factor(actual_value)

        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("y", [0.0, 0.5, 1.2])
    @pytest.mark.parametrize("n_parallel", [0.0, 0.5, 1.0, -0.5, -1.0])
    def test_vacuum_polarisation(y: float, n_parallel: float):
        """
        Test polarisations are orthogonal and null eigenvectors of the vacuum
        dispersion tensor.

        Parameters
        ----------
        y : float
            Normalised magnetic field strength.
        n_parallel : float
            Parallel refractive index.
        """
        n_perp = np.sqrt(1 - n_parallel * n_parallel)

        q = np.zeros(Dimensions.q.size)
        q[Dimensions.IDX_Y] = y
        q[Dimensions.IDX_N_PERP] = n_perp
        q[Dimensions.IDX_N_PARALLEL] = n_parallel

        d = vacuum_dispersion_tensor(q)
        e_o = vacuum_stix_polarisation(y, n_perp, n_parallel, WaveMode.O)
        e_x = vacuum_stix_polarisation(y, n_perp, n_parallel, WaveMode.X)

        nptest.assert_allclose(d @ e_o, 0.0, atol=1e-8)
        nptest.assert_allclose(e_o @ d, 0.0, atol=1e-8)
        nptest.assert_allclose(d @ e_x, 0.0, atol=1e-8)
        nptest.assert_allclose(e_x @ d, 0.0, atol=1e-8)
        nptest.assert_allclose(np.vdot(e_o, e_o), 1.0, atol=1e-8)
        nptest.assert_allclose(np.vdot(e_x, e_x), 1.0, atol=1e-8)
        nptest.assert_allclose(np.vdot(e_o, e_x), 0.0, atol=1e-8)

    @staticmethod
    @pytest.mark.parametrize("x", [0.0, 1.0, 2.0, 3.0])
    def test_eigenvalue_derivatives(x: float):
        """
        Test derivatives of eigenvalue of dispersion tensor.

        Parameters
        ----------
        x : float
            Element [0, 0] value.
        """
        m = np.zeros((3, 3))
        m[0, 0] = x * x
        m[1, 1] = 1.0
        m[2, 2] = 1.0

        e = np.array([1.0, 0.0, 0.0])

        # Test value.
        expected_value = x**2
        actual_value = eigenvalue(m, e)

        nptest.assert_allclose(expected_value, actual_value)

        # Test first derivative.
        m_dx = np.zeros((3, 3, 3))
        m_dx[0, 0, 0] = 2.0 * x

        expected_value = np.zeros(3)
        expected_value[0] = 2.0 * x

        actual_value = eigenvalue_dx(m_dx, e)

        nptest.assert_allclose(expected_value, actual_value)

        # Test second derivative.
        m_dx2 = np.zeros((3, 3, 3, 3))
        m_dx2[0, 0, 0, 0] = 2.0
        e_dx = np.zeros((3, 3))

        expected_value = np.zeros((3, 3))
        expected_value[0, 0] = 2.0

        actual_value = eigenvalue_dx2(m_dx, m_dx2, e, e_dx)

        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("x", [0.0, 1.0, 2.0, 3.0])
    def test_determinant_derivatives(x: float):
        """
        Test derivatives of determinant of dispersion tensor.

        Parameters
        ----------
        x : float
            Element [0, 0] value.
        """
        m = np.zeros((3, 3))
        m[0, 0] = x * x
        m[1, 1] = 1.0
        m[2, 2] = 1.0

        # Test value.
        expected_value = x**2
        actual_value = determinant(m)

        nptest.assert_allclose(expected_value, actual_value)

        # Test first derivative.
        m_dx = np.zeros((3, 3, 1))
        m_dx[0, 0, 0] = 2.0 * x

        expected_value = np.zeros(1)
        expected_value[0] = 2.0 * x

        actual_value = determinant_dx(m, m_dx)

        nptest.assert_allclose(expected_value, actual_value)

        # Test second derivative.
        m_dx2 = np.zeros((3, 3, 1, 1))
        m_dx2[0, 0, 0, 0] = 2.0

        expected_value = np.zeros((1, 1))
        expected_value[0, 0] = 2.0

        actual_value = determinant_dx2(m, m_dx, m_dx2)

        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("x", [0.0, 1.0, 2.0, 3.0])
    def test_eigenvector_derivative(x: float):
        """
        Test derivatives of eigenvector of dispersion tensor.

        Parameters
        ----------
        x : float
            Element [0, 0] value.
        """
        c, s = np.cos(x), np.sin(x)
        m = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 0.0]])

        m_dx = np.zeros((3, 3, 1))
        m_dx[0, 0, 0] = -s
        m_dx[0, 1, 0] = c
        m_dx[1, 0, 0] = c
        m_dx[1, 1, 0] = s

        e = np.array([1 + c, s, 0])
        norm = np.sqrt(2 * (1 + c))
        e /= norm

        eigenvalues = np.empty(3, dtype=complex)
        eigenvectors = np.empty((3, 3), dtype=complex)

        eigenvalues[:], eigenvectors[:, :] = np.linalg.eig(m)
        idx = np.argmin(abs(eigenvalues - 1.0))

        # Eig returns them transposed to how we want them.
        eigenvectors[:, :] = eigenvectors[:, :].T

        for i in range(3):
            eigenvectors[i, :] *= polarisation_phase_convention_factor(
                eigenvectors[i, :]
            ) / np.linalg.norm(eigenvectors[i])

        nptest.assert_allclose(eigenvectors[idx], e)

        # Test first derivative.
        expected_value = np.zeros((3, 1))
        expected_value[0, 0] = -s * (1 + c) / norm**3
        expected_value[1, 0] = norm / 4.0

        actual_value = eigenvector_dx(eigenvalues, eigenvectors, m_dx, idx)

        nptest.assert_allclose(actual_value, expected_value)

    @staticmethod
    def test_calculate_harmonic_range():
        """
        Test calculation of cyclotron harmonic range.
        """
        nptest.assert_allclose(calculate_harmonic_range(0.7), (1, 2))
        nptest.assert_allclose(calculate_harmonic_range(0.4), (2, 3))
        nptest.assert_allclose(calculate_harmonic_range(0.32), (3, 4))
        nptest.assert_allclose(calculate_harmonic_range(1.1), (0, 1))
