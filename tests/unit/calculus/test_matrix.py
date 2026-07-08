"""
Unit tests for calculus.matrix
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus.matrix import (
    adjugate_3x3_cayley_hamilton,
    adjugate_3x3_cofactors,
    antihermitian,
    hermitian,
    matrix_3x3_adjugate_first_derivative,
    matrix_3x3_determinant_first_derivative,
    matrix_3x3_determinant_second_derivative,
    mirror_upper_triangular_to_lower_triangular,
)
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


def test_mirror_upper_triangular_to_lower_triangular():
    """
    Test function correctly mirrors upper triangular to lower triangular.
    """
    actual_value = np.zeros((3, 3))

    actual_value[0, 0] = 1.0
    actual_value[0, 1] = 2.0
    actual_value[0, 2] = 3.0
    actual_value[1, 1] = 4.0
    actual_value[1, 2] = 5.0
    actual_value[2, 2] = 6.0

    mirror_upper_triangular_to_lower_triangular(actual_value)

    expected_value = np.zeros((3, 3))

    expected_value[0, 0] = 1.0
    expected_value[0, 1] = 2.0
    expected_value[0, 2] = 3.0
    expected_value[1, 0] = 2.0
    expected_value[1, 1] = 4.0
    expected_value[1, 2] = 5.0
    expected_value[2, 0] = 3.0
    expected_value[2, 1] = 5.0
    expected_value[2, 2] = 6.0

    nptest.assert_allclose(expected_value, expected_value)


def test_hermitian_antihermitian():
    """
    Test function correctly returns hermitian and anti-hermitian part of
    a matrix.
    """
    m = np.array(
        [
            [0.099 + 0.213j, 0.536 - 0.882j, 0.288 - 0.683j],
            [0.009 + 0.274j, -0.874 + 0.441j, -0.527 - 0.661j],
            [0.848 - 0.048j, -0.357 - 0.270j, 0.388 - 0.022j],
        ],
        dtype=complex,
    )

    m_h = hermitian(m)
    m_ah = antihermitian(m)

    nptest.assert_allclose(m, m_h + m_ah)
    nptest.assert_allclose(np.conj(m_h).T, m_h)
    nptest.assert_allclose(np.conj(m_ah).T, -m_ah)


class TestMatrixReal:
    """
    Test matrix formulas for real valued matrix.
    """

    @staticmethod
    def m(
        position: np.ndarray[float], a: float, b: float, c: float
    ) -> np.ndarray[float]:
        """
        Calculate matrix function.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.

        Returns
        -------
        m : np.ndarray[float]
            Test matrix.
        """
        x, y, z = position

        m = np.zeros((3, 3))
        m[0, 0] = np.cos(x) - a
        m[1, 1] = np.cos(y) - b
        m[2, 2] = np.cos(z) - c

        return m

    @staticmethod
    def det_m(
        position: np.ndarray[float], a: float, b: float, c: float
    ) -> float:
        """
        Calculate determinant of test matrix function.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.

        Returns
        -------
        det_m : float
            Matrix determinant.
        """
        x, y, z = position
        return (np.cos(x) - a) * (np.cos(y) - b) * (np.cos(z) - c)

    @staticmethod
    def m_first_derivative(position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative test matrix function with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        m_dx : np.ndarray[float]
            First derivative of m with respect to x.
        """
        x, y, z = position

        m_dx = np.zeros((3, 3, 3))
        m_dx[0, 0, 0] = -np.sin(x)
        m_dx[1, 1, 1] = -np.sin(y)
        m_dx[2, 2, 2] = -np.sin(z)

        return m_dx

    @staticmethod
    def det_m_first_derivative(
        position: np.ndarray[float], a: float, b: float, c: float
    ) -> np.ndarray[float]:
        """
        Calculate first derivative of determinant of test matrix function
        with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.

        Returns
        -------
        det_m_dx : np.ndarray[float]
            First derivative of determinant with respect to x.
        """
        x, y, z = position

        det_m_dx = np.zeros(3)
        det_m_dx[0] = -np.sin(x) * (np.cos(y) - b) * (np.cos(z) - c)
        det_m_dx[1] = -np.sin(y) * (np.cos(x) - a) * (np.cos(z) - c)
        det_m_dx[2] = -np.sin(z) * (np.cos(x) - a) * (np.cos(y) - b)

        return det_m_dx

    @staticmethod
    def m_second_derivative(position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative test matrix function with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        m_dx2 : np.ndarray[float]
            Second derivative of m with respect to x.
        """
        x, y, z = position

        m_dx2 = np.zeros((3, 3, 3, 3))
        m_dx2[0, 0, 0, 0] = -np.cos(x)
        m_dx2[1, 1, 1, 1] = -np.cos(y)
        m_dx2[2, 2, 2, 2] = -np.cos(z)

        return m_dx2

    @staticmethod
    def det_m_second_derivative(
        position: np.ndarray[float], a: float, b: float, c: float
    ) -> np.ndarray[float]:
        """
        Calculate second derivative of determinant of test matrix function
        with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.

        Returns
        -------
        det_m_dx2 : np.ndarray[float]
            Second derivative of determinant with respect to x.
        """
        x, y, z = position

        det_m_dx2 = np.zeros((3, 3))
        det_m_dx2[0, 0] = -np.cos(x) * (np.cos(y) - b) * (np.cos(z) - c)
        det_m_dx2[0, 1] = np.sin(x) * np.sin(y) * (np.cos(z) - c)
        det_m_dx2[0, 2] = np.sin(x) * (np.cos(y) - b) * np.sin(z)
        det_m_dx2[1, 0] = det_m_dx2[0, 1]
        det_m_dx2[1, 1] = -np.cos(y) * (np.cos(x) - a) * (np.cos(z) - c)
        det_m_dx2[1, 2] = np.sin(z) * (np.cos(x) - a) * np.sin(y)
        det_m_dx2[2, 0] = det_m_dx2[0, 2]
        det_m_dx2[2, 1] = det_m_dx2[1, 2]
        det_m_dx2[2, 2] = -np.cos(z) * (np.cos(x) - a) * (np.cos(y) - b)

        return det_m_dx2

    @staticmethod
    def m_adjugate(
        position: np.ndarray[float], a: float, b: float, c: float
    ) -> np.ndarray[float]:
        """
        Calculate adjugate of matrix function.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.

        Returns
        -------
        adj_m : np.ndarray[float]
            Adjugate of test matrix.
        """
        x, y, z = position

        adj_m = np.zeros((3, 3))
        adj_m[0, 0] = (np.cos(y) - b) * (np.cos(z) - c)
        adj_m[1, 1] = (np.cos(x) - a) * (np.cos(z) - c)
        adj_m[2, 2] = (np.cos(x) - a) * (np.cos(y) - b)

        return adj_m

    @staticmethod
    def m_adjugate_first_derivative(position, a, b, c):
        """
        Calculate first derivative of adjugate of matrix function with
        respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.

        Returns
        -------
        adj_m_dx : np.ndarray[float]
            First derivative of adjugate of test matrix with respect to x.
        """
        x, y, z = position

        adj_m_dx = np.zeros((3, 3, 3))
        adj_m_dx[0, 0, 1] = -np.sin(y) * (np.cos(z) - c)
        adj_m_dx[0, 0, 2] = -np.sin(z) * (np.cos(y) - b)
        adj_m_dx[1, 1, 0] = -np.sin(x) * (np.cos(z) - c)
        adj_m_dx[1, 1, 2] = -np.sin(z) * (np.cos(x) - a)
        adj_m_dx[2, 2, 0] = -np.sin(x) * (np.cos(y) - b)
        adj_m_dx[2, 2, 1] = -np.sin(y) * (np.cos(x) - a)

        return adj_m_dx

    test_values_1 = (
        (np.array([-0.61, -0.34, 0.76]), -0.61, 0.03, -0.6),
        (np.array([0.11, 0.44, -0.82]), 0.05, -0.42, 0.43),
        (np.array([-0.79, 0.06, 0.67]), 0.94, 0.44, -0.23),
    )

    # This set should have det(m) = 0.
    test_values_2 = (
        (np.array([-0.39, -0.22, 0.87]), np.cos(-0.39), 1.0, 0.18),
        (np.array([0.74, -0.15, -0.62]), 0.54, np.cos(-0.15), -0.18),
        (np.array([-0.98, -0.22, 0.72]), 0.03, 0.64, np.cos(0.72)),
    )

    parametrize_1 = ("position, a, b, c", test_values_2)

    @pytest.mark.parametrize(*parametrize_1)
    def test_matrix_determinant_eq_0(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix determinant is zero for special cases.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """
        actual_value = self.det_m(position, a, b, c)

        logger.warning(abs(actual_value))
        nptest.assert_allclose(0.0, actual_value)

    parametrize_2 = ("position, a, b, c", (*test_values_1, *test_values_2))

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_derivative(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix first and second derivative agrees with finite difference.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """

        def _func(x):
            return self.m(x, a, b, c)

        value_analytic = self.m_first_derivative(position)
        value_fd = first_derivative_finite_difference(
            position, _func, (3, 3), order=4, is_complex=True
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-8)

        value_analytic = self.m_second_derivative(position)
        value_fd = second_derivative_finite_difference(
            position, _func, (3, 3), order=4, is_complex=True
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_determinant(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix determinant formula correct.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """
        expected_value = self.det_m(position, a, b, c)

        m = self.m(position, a, b, c)
        actual_value = np.linalg.det(m)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_determinant_derivative(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix determinant first and second derivative agrees with finite
        difference.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """

        def _func(x):
            return self.det_m(x, a, b, c)

        m = self.m(position, a, b, c)
        m_dx = self.m_first_derivative(position)
        m_dx2 = self.m_second_derivative(position)

        # First derivative.
        value_analytic = self.det_m_first_derivative(position, a, b, c)
        value_fd = first_derivative_finite_difference(
            position, _func, (), order=4, is_complex=True
        )

        value_analytic2 = matrix_3x3_determinant_first_derivative(m, m_dx)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-8)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=1e-8)

        # Second derivative.
        value_analytic = self.det_m_second_derivative(position, a, b, c)
        value_fd = second_derivative_finite_difference(
            position, _func, (), order=4, is_complex=True
        )

        value_analytic2 = matrix_3x3_determinant_second_derivative(
            m, m_dx, m_dx2
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=2e-7)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=2e-7)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_adjugate(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix adjugate satisfies m @ adj_m = det(m) I.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """
        m = self.m(position, a, b, c)
        adj_m = self.m_adjugate(position, a, b, c)
        det_m = self.det_m(position, a, b, c)

        expected_value = det_m * np.identity(3)

        actual_value = np.matmul(m, adj_m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        actual_value = np.matmul(adj_m, m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_adjugate_formulas(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix adjugate agrees with formulas.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """
        m = self.m(position, a, b, c)
        expected_value = self.m_adjugate(position, a, b, c)

        actual_value = adjugate_3x3_cofactors(m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        actual_value = adjugate_3x3_cayley_hamilton(m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_adjugate_derivative(
        self, position: np.ndarray[float], a: float, b: float, c: float
    ):
        """
        Test matrix first derivative agrees with finite differences.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        a, b, c : float
            Additional parameters.
        """

        def _func(x):
            return self.m_adjugate(x, a, b, c)

        value_analytic = self.m_adjugate_first_derivative(position, a, b, c)
        value_fd = first_derivative_finite_difference(
            position, _func, (3, 3), h=1e-6, order=4, is_complex=True
        )

        m = self.m(position, a, b, c)
        m_dx = self.m_first_derivative(position)
        value_analytic2 = matrix_3x3_adjugate_first_derivative(m, m_dx)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-8)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=1e-8)


class TestMatrixComplex:
    """
    Test matrix formulas for complex valued matrix.
    """

    @staticmethod
    def m(z: np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculate matrix function.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        m : np.ndarray[complex]
            Test matrix.
        """
        z = z[0]
        matrix = np.zeros((3, 3), dtype=complex)
        matrix[0, 0] = z**2
        matrix[0, 1] = np.cos(z)
        matrix[1, 0] = np.cos(z)
        matrix[1, 1] = z**3
        matrix[2, 2] = z**4

        return matrix

    @staticmethod
    def det_m(z) -> complex:
        """
        Calculate determinant of matrix function.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        det_m : complex
            Determinant of test matrix.
        """
        z = z[0]
        return z**9 - z**4 * np.cos(z) ** 2

    @staticmethod
    def m_first_derivative(z: np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculate first derivative test matrix function with respect to z.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        m_dz : np.ndarray[complex]
            First derivative of m with respect to z.
        """
        z = z[0]
        matrix = np.zeros((3, 3, 1), dtype=complex)
        matrix[0, 0, 0] = 2 * z
        matrix[0, 1, 0] = -np.sin(z)
        matrix[1, 0, 0] = -np.sin(z)
        matrix[1, 1, 0] = 3 * z**2
        matrix[2, 2, 0] = 4 * z**3

        return matrix

    @staticmethod
    def det_m_first_derivative(z: np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculate first derivative of determinant of test matrix function
        with respect to z.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        det_m_dz : np.ndarray[float]
            First derivative of determinant with respect to z.
        """
        z = z[0]
        c, s = np.cos(z), np.sin(z)
        return 9 * z**8 - 4 * z**3 * c**2 + 2 * z**4 * c * s

    @staticmethod
    def m_second_derivative(z: np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculate second derivative test matrix function with respect to z.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        m_dz2 : np.ndarray[complex]
            Second derivative of m with respect to z.
        """
        z = z[0]
        matrix = np.zeros((3, 3, 1, 1), dtype=complex)
        matrix[0, 0, 0, 0] = 2
        matrix[0, 1, 0, 0] = -np.cos(z)
        matrix[1, 0, 0, 0] = -np.cos(z)
        matrix[1, 1, 0, 0] = 6 * z
        matrix[2, 2, 0, 0] = 12 * z**2
        return matrix

    @staticmethod
    def det_m_second_derivative(z: np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculate second derivative of determinant of test matrix function
        with respect to z.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        det_m_dz2 : np.ndarray[float]
            Second derivative of determinant with respect to z.
        """
        z = z[0]
        c, s = np.cos(z), np.sin(z)
        return (
            72 * z**7
            - 12 * z**2 * c**2
            + 8 * z**3 * c * s
            + 8 * z**3 * c * s
            + 2 * z**4 * (c**2 - s**2)
        )

    @staticmethod
    def m_adjugate(z: np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculate adjugate of matrix function.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        adj_m : np.ndarray[complex]
            Adjugate of test matrix.
        """
        z = z[0]
        adj_m = np.zeros((3, 3))

        adj_m[0, 0] = z**7
        adj_m[0, 1] = -(z**4) * np.cos(z)
        adj_m[1, 0] = -(z**4) * np.cos(z)
        adj_m[1, 1] = z**6
        adj_m[2, 2] = z**5 - np.cos(z) ** 2

        return adj_m

    @staticmethod
    def m_adjugate_first_derivative(
        z: np.ndarray[float],
    ) -> np.ndarray[complex]:
        """
        Calculate first derivative of adjugate of matrix function with
        respect to z.

        Parameters
        ----------
        z : np.ndarray[float]
            Position.

        Returns
        -------
        adj_m_dz : np.ndarray[float]
            First derivative of adjugate of test matrix with respect to z.
        """
        z = z[0]
        dadj_m_dz = np.zeros((3, 3, 1))

        dadj_m_dz[0, 0, 0] = 7 * z**6
        dadj_m_dz[0, 1, 0] = z**4 * np.sin(z) - 4 * z**3 * np.cos(z)
        dadj_m_dz[1, 0, 0] = z**4 * np.sin(z) - 4 * z**3 * np.cos(z)
        dadj_m_dz[1, 1, 0] = 6 * z**5
        dadj_m_dz[2, 2, 0] = 5 * z**4 + 2 * np.cos(z) * np.sin(z)

        return dadj_m_dz

    test_values_1 = ([0.13], [-0.18], [0.73], [0.12], [-0.14])

    test_values_2 = ([0.0], [0.84773486373281])
    # 2nd: Solution of z**5 = cos(z)**2

    parametrize_1 = ("z", test_values_2)

    @pytest.mark.parametrize(*parametrize_1)
    def test_matrix_determinant_eq_0(self, z: np.ndarray[float]):
        """
        Test matrix determinant is zero for special cases.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        actual_value = self.det_m(z)

        logger.warning(abs(actual_value))
        nptest.assert_allclose(0.0, actual_value, atol=1e-8)

    parametrize_2 = ("z", (*test_values_1, *test_values_2))

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_derivative(self, z: np.ndarray[float]):
        """
        Test matrix first and second derivative agrees with finite difference.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        value_analytic = self.m_first_derivative(z)
        value_fd = first_derivative_finite_difference(
            z, self.m, (3, 3), order=4, is_complex=True
        )

        logger.warning((value_analytic.shape, value_fd.shape))
        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-8)

        value_analytic = self.m_second_derivative(z)
        value_fd = second_derivative_finite_difference(
            z, self.m, (3, 3), order=4, is_complex=True
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_determinant(self, z: np.ndarray[float]):
        """
        Test matrix determinant correct.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        expected_value = self.det_m(z)

        m = self.m(z)
        actual_value = np.linalg.det(m)

        logger.warning(abs(expected_value - actual_value))
        assert np.isclose(expected_value, actual_value)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_determinant_derivative(self, z: np.ndarray[float]):
        """
        Test matrix determinant first derivative agrees with finite
        differences.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        m = self.m(z)
        m_dx = self.m_first_derivative(z)
        m_dx2 = self.m_second_derivative(z)

        # First derivative.
        value_analytic = self.det_m_first_derivative(z)
        value_fd = first_derivative_finite_difference(
            z, self.det_m, (), order=4, is_complex=True
        )
        value_analytic2 = matrix_3x3_determinant_first_derivative(m, m_dx)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2)

        # Second derivative.
        value_analytic = self.det_m_second_derivative(z)
        value_fd = second_derivative_finite_difference(
            z, self.det_m, (), order=4, is_complex=True
        )
        value_analytic2 = matrix_3x3_determinant_second_derivative(
            m, m_dx, m_dx2
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-8)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=1e-8)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_adjugate(self, z: np.ndarray[float]):
        """
        Test matrix adjugate satisfies m @ adj_m = det(m) I.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        m = self.m(z)
        adj_m = self.m_adjugate(z)
        det_m = self.det_m(z)

        expected_value = det_m * np.identity(3)

        actual_value = np.matmul(m, adj_m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        actual_value = np.matmul(adj_m, m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_adjugate_formulas(self, z: np.ndarray[float]):
        """
        Test matrix adjugate agrees with formulas.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        m = self.m(z)
        expected_value = self.m_adjugate(z)

        actual_value = adjugate_3x3_cofactors(m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        actual_value = adjugate_3x3_cayley_hamilton(m)
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @pytest.mark.parametrize(*parametrize_2)
    def test_matrix_adjugate_first_derivative(self, z: np.ndarray[float]):
        """
        Test matrix first derivative agrees with finite differences.

        Parameters
        ----------
        z : np.ndarray[float]
            Test position.
        """
        value_analytic = self.m_adjugate_first_derivative(z)
        value_fd = first_derivative_finite_difference(
            z, self.m_adjugate, (3, 3), h=1e-6, order=4, is_complex=True
        )

        m = self.m(z)
        m_dx = self.m_first_derivative(z)
        value_analytic2 = matrix_3x3_adjugate_first_derivative(m, m_dx)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-8)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=1e-8)
