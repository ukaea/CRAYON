"""
Unit tests for shared.numerics.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.shared import numerics

logger = logging.getLogger(__name__)


class TestGaussianQuadrature:
    """
    Test gaussian quadrature routines.
    """

    @staticmethod
    @pytest.mark.parametrize(("x0", "x1"), [(-1, 1), (0, 1), (0.3, 1.4)])
    def test_integral(x0: float, x1: float):
        """
        Test integral of x dx is x**2 / 2 as expected.

        Parameters
        ----------
        x0, x1 : float
            Integration interval.
        """
        samples, weights = numerics.get_leggauss_samples_weights(x0, x1)

        expected_value = 0.5 * (x1**2 - x0**2)
        actual_value = sum(weights * samples)

        assert np.isclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("n", [7, 10, 15, 20, 25, 30])
    def test_gauss_kronrod(n: int):
        """
        Test Gauss-Kronrod integration.

        Parameters
        ----------
        n : int
            Number of integration nodes.
        """
        nodes, weights, gauss_mask, gauss_weights = (
            numerics.gauss_kronrod_nodes_weights(0.0, np.pi, n)
        )

        samples = np.sin(nodes)
        value_gauss = np.sum(samples[gauss_mask] * gauss_weights)
        value_kronrod = np.sum(samples * weights)

        nptest.assert_allclose(value_gauss, 2.0)
        assert (
            # Too small to compare properly.
            np.isclose(abs(2.0 - value_gauss), 1e-14)
            # or higher order method more accurate.
            or abs(2.0 - value_kronrod) <= abs(2.0 - value_gauss)
        )


class TestFiniteDifference:
    """
    Test finite difference derivatives.
    """

    @staticmethod
    def f(p: np.ndarray[float]) -> float:
        """
        Test function.

        Parameters
        ----------
        p : np.ndarray[float]
            Arguments.

        Returns
        -------
        f : float
            Function value.
        """
        x, y = p
        return np.cos(x) * np.sin(y)

    @staticmethod
    def f_dp(p: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate function first derivative with respect to p.

        Parameters
        ----------
        p : np.ndarray[float]
            Arguments.

        Returns
        -------
        f_dp : float
            Function first derivative.
        """
        x, y = p

        return_array = np.empty(2)
        return_array[0] = -np.sin(x) * np.sin(y)
        return_array[1] = np.cos(x) * np.cos(y)

        return return_array

    @staticmethod
    def f_dp2(p: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate function second derivative with respect to p.

        Parameters
        ----------
        p : np.ndarray[float]
            Arguments.

        Returns
        -------
        f_dp2 : float
            Function second derivative.
        """
        x, y = p

        return_array = np.empty((2, 2))
        return_array[0, 0] = -np.cos(x) * np.sin(y)
        return_array[0, 1] = -np.sin(x) * np.cos(y)
        return_array[1, 0] = return_array[0, 1]
        return_array[1, 1] = -np.cos(x) * np.sin(y)

        return return_array

    test_values = (
        (0.0, 0.0),
        (1.0, 0.5),
        (-0.5, 0.7),
        (0.3, -0.1),
        (-0.3, 0.8),
    )

    @pytest.mark.parametrize("p", test_values)
    def test_derivatives(self, p: np.ndarray[float]):
        """
        Test derivatives of function against finite difference.

        Parameters
        ----------
        p : np.ndarray[float]
            Arguments.
        """
        # Test first derivative
        expected_value = self.f_dp(p)
        actual_value = numerics.first_derivative_finite_difference(
            p, self.f, ()
        )

        nptest.assert_allclose(expected_value, actual_value)

        # Test second derivative
        expected_value = self.f_dp2(p)
        actual_value = numerics.second_derivative_finite_difference(
            p, self.f, ()
        )

        nptest.assert_allclose(expected_value, actual_value)


class TestSolveQuadratic:
    """
    Unit tests for solve_quadratic.

    Notes
    -----
    Examples from Forsythe, George E. "Pitfalls in computation,
    or why a math book isn't enough." The American Mathematical
    Monthly 77.9 (1970): 931-956.
    """

    @staticmethod
    @pytest.mark.parametrize(
        ("a", "b", "c", "x_plus", "x_minus"),
        [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 2.0, 2.0, 2.0),
            (1.0, -2.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, -1.0, 1.0, -1.0),
            (1.0, -(10**5), 1, 99999.999990, 0.00001),
            (6.0, 5.0, -4.0, 0.5, -1.33333333333),
            (6.0e30, 5.0e30, -4.0e30, 0.5, -1.33333333333),
            (1.0, -4.0, 3.9999999, 2.000316228, 1.999683772),
        ],
    )
    def test_solve_quadratic(
        a: float, b: float, c: float, x_plus: float, x_minus: float
    ):
        """
        Test roots of quadratic equations.

        Parameters
        ----------
        a, b, c : float
            Quadratic coefficients for a*x**2 + b*x + c.
        x_plus, x_minus : float
            Roots of quadratic.
        """
        expected_value = (x_plus, x_minus)
        actual_value = numerics.solve_quadratic(a, b, c)
        assert np.allclose(actual_value, expected_value)
