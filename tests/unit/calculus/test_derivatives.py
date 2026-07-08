"""
Unit tests for calculus.derivatives
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus.derivatives import (
    first_derivative,
    fourth_derivative,
    second_derivative,
    third_derivative,
)
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


class TestDerivatives:
    """
    Test functions for calculating derivatives via chain rule.
    """

    @staticmethod
    def f(g: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates outer test function f.

        Parameters
        ----------
        g : np.ndarray[float]
            Inner test function.

        Returns
        -------
        f: np.ndarray[float]
        """
        a = g[0] * g[1]

        return_value = np.zeros(2)

        return_value[0] = np.cos(a)
        return_value[1] = (g[0] * g[1]) ** 3

        return return_value

    @staticmethod
    def f_dg(g: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates first derivative of f with respect to g.

        Parameters
        ----------
        g : np.ndarray[float]
            Inner test function.

        Returns
        -------
        f_dg: np.ndarray[float]
            First derivative of f with respect to g.
        """
        a = g[0] * g[1]

        return_value = np.zeros((2, 2))

        return_value[0, 0] = -g[1] * np.sin(a)
        return_value[0, 1] = -g[0] * np.sin(a)
        return_value[1, 0] = 3 * g[0] ** 2 * g[1] ** 3
        return_value[1, 1] = 3 * g[0] ** 3 * g[1] ** 2

        return return_value

    @staticmethod
    def f_dg2(g):
        """
        Calculates second derivative of f with respect to g.

        Parameters
        ----------
        g : np.ndarray[float]
            Inner test function.

        Returns
        -------
        f_dg2: np.ndarray[float]
            Second derivative of f with respect to g.
        """
        a = g[0] * g[1]

        return_value = np.zeros((2, 2, 2))

        return_value[0, 0, 0] = -(g[1] ** 2) * np.cos(a)
        return_value[0, 0, 1] = -np.sin(a) - g[0] * g[1] * np.cos(a)
        return_value[0, 1, 0] = return_value[0, 0, 1]
        return_value[0, 1, 1] = -(g[0] ** 2) * np.cos(a)

        return_value[1, 0, 0] = 6 * g[0] * g[1] ** 3
        return_value[1, 0, 1] = 9 * g[0] ** 2 * g[1] ** 2
        return_value[1, 1, 0] = return_value[1, 0, 1]
        return_value[1, 1, 1] = 6 * g[0] ** 3 * g[1]

        return return_value

    @staticmethod
    def f_dg3(g):
        """
        Calculates third derivative of f with respect to g.

        Parameters
        ----------
        g : np.ndarray[float]
            Inner test function.

        Returns
        -------
        f_dg3: np.ndarray[float]
            Third derivative of f with respect to g.
        """
        a = g[0] * g[1]

        return_value = np.zeros((2, 2, 2, 2))

        return_value[0, 0, 0, 0] = g[1] ** 3 * np.sin(a)

        return_value[0, 0, 0, 1] = g[0] * g[1] ** 2 * np.sin(a) - 2 * g[
            1
        ] * np.cos(a)
        return_value[0, 0, 1, 0] = return_value[0, 0, 0, 1]
        return_value[0, 1, 0, 0] = return_value[0, 0, 0, 1]

        return_value[0, 0, 1, 1] = g[0] ** 2 * g[1] * np.sin(a) - 2 * g[
            0
        ] * np.cos(a)
        return_value[0, 1, 0, 1] = return_value[0, 0, 1, 1]
        return_value[0, 1, 1, 0] = return_value[0, 0, 1, 1]

        return_value[0, 1, 1, 1] = g[0] ** 3 * np.sin(a)

        return_value[1, 0, 0, 0] = 6 * g[1] ** 3

        return_value[1, 0, 0, 1] = 18 * g[0] * g[1] ** 2
        return_value[1, 0, 1, 0] = return_value[1, 0, 0, 1]
        return_value[1, 1, 0, 0] = return_value[1, 0, 0, 1]

        return_value[1, 0, 1, 1] = 18 * g[0] ** 2 * g[1]
        return_value[1, 1, 0, 1] = return_value[1, 0, 1, 1]
        return_value[1, 1, 1, 0] = return_value[1, 0, 1, 1]

        return_value[1, 1, 1, 1] = 6 * g[0] ** 3

        return return_value

    @staticmethod
    def f_dg4(g):
        """
        Calculates fourth derivative of f with respect to g.

        Parameters
        ----------
        g : np.ndarray[float]
            Inner test function.

        Returns
        -------
        f_dg4: np.ndarray[float]
            Fourth derivative of f with respect to g.
        """
        a = g[0] * g[1]

        return_value = np.zeros((2, 2, 2, 2, 2))

        return_value[0, 0, 0, 0, 0] = g[1] ** 4 * np.cos(a)

        return_value[0, 0, 0, 0, 1] = 3 * g[1] ** 2 * np.sin(a) + g[0] * g[
            1
        ] ** 3 * np.cos(a)
        return_value[0, 0, 0, 1, 0] = return_value[0, 0, 0, 0, 1]
        return_value[0, 0, 1, 0, 0] = return_value[0, 0, 0, 0, 1]
        return_value[0, 1, 0, 0, 0] = return_value[0, 0, 0, 0, 1]

        return_value[0, 0, 0, 1, 1] = 4 * g[0] * g[1] * np.sin(a) + (
            g[0] ** 2 * g[1] ** 2 - 2
        ) * np.cos(a)
        return_value[0, 0, 1, 0, 1] = return_value[0, 0, 0, 1, 1]
        return_value[0, 1, 0, 0, 1] = return_value[0, 0, 0, 1, 1]
        return_value[0, 0, 1, 1, 0] = return_value[0, 0, 0, 1, 1]
        return_value[0, 1, 0, 1, 0] = return_value[0, 0, 0, 1, 1]
        return_value[0, 1, 1, 0, 0] = return_value[0, 0, 0, 1, 1]

        return_value[0, 0, 1, 1, 1] = 3 * g[0] ** 2 * np.sin(a) + g[
            0
        ] ** 3 * g[1] * np.cos(a)
        return_value[0, 1, 0, 1, 1] = return_value[0, 0, 1, 1, 1]
        return_value[0, 1, 1, 0, 1] = return_value[0, 0, 1, 1, 1]
        return_value[0, 1, 1, 1, 0] = return_value[0, 0, 1, 1, 1]

        return_value[0, 1, 1, 1, 1] = g[0] ** 4 * np.cos(a)

        return_value[1, 0, 0, 0, 1] = 18 * g[1] ** 2
        return_value[1, 0, 0, 1, 0] = return_value[1, 0, 0, 0, 1]
        return_value[1, 0, 1, 0, 0] = return_value[1, 0, 0, 0, 1]
        return_value[1, 1, 0, 0, 0] = return_value[1, 0, 0, 0, 1]

        return_value[1, 0, 0, 1, 1] = 36 * g[0] * g[1]
        return_value[1, 0, 1, 0, 1] = return_value[1, 0, 0, 1, 1]
        return_value[1, 1, 0, 0, 1] = return_value[1, 0, 0, 1, 1]
        return_value[1, 0, 1, 1, 0] = return_value[1, 0, 0, 1, 1]
        return_value[1, 1, 0, 1, 0] = return_value[1, 0, 0, 1, 1]
        return_value[1, 1, 1, 0, 0] = return_value[1, 0, 0, 1, 1]

        return_value[1, 0, 1, 1, 1] = 18 * g[0] ** 2
        return_value[1, 1, 0, 1, 1] = return_value[1, 0, 1, 1, 1]
        return_value[1, 1, 1, 0, 1] = return_value[1, 0, 1, 1, 1]
        return_value[1, 1, 1, 1, 0] = return_value[1, 0, 1, 1, 1]

        return return_value

    @staticmethod
    def g(x):
        """
        Calculates inner function g.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        g: np.ndarray[float]
            Inner function g.
        """
        return_value = np.zeros(2)
        return_value[0] = np.cos(x[0])
        return_value[1] = np.sin(x[0])

        return return_value

    @staticmethod
    def g_dx(x):
        """
        Calculates first derivative of g with respect to x.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        g_dx: np.ndarray[float]
            First derivative of g with respect to x.
        """
        return_value = np.zeros((2, 1))
        return_value[0, 0] = -np.sin(x[0])
        return_value[1, 0] = np.cos(x[0])

        return return_value

    @staticmethod
    def g_dx2(x):
        """
        Calculates second derivative of g with respect to x.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        g_dx2: np.ndarray[float]
            Second derivative of g with respect to x.
        """
        return_value = np.zeros((2, 1, 1))
        return_value[0, 0, 0] = -np.cos(x[0])
        return_value[1, 0, 0] = -np.sin(x[0])

        return return_value

    @staticmethod
    def g_dx3(x):
        """
        Calculates third derivative of g with respect to x.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        g_dx3: np.ndarray[float]
            Third derivative of g with respect to x.
        """
        return_value = np.zeros((2, 1, 1, 1))
        return_value[0, 0, 0, 0] = np.sin(x[0])
        return_value[1, 0, 0, 0] = -np.cos(x[0])

        return return_value

    @staticmethod
    def g_dx4(x):
        """
        Calculates fourth derivative of g with respect to x.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        g_dx4: np.ndarray[float]
            Fourth derivative of g with respect to x.
        """
        return_value = np.zeros((2, 1, 1, 1, 1))
        return_value[0, 0, 0, 0, 0] = np.cos(x[0])
        return_value[1, 0, 0, 0, 0] = np.sin(x[0])

        return return_value

    def h(self, x):
        """
        Calculates composed function f(g(x))

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        h : np.ndarray[float]
            Composed function f(g(x))
        """
        return self.f(self.g(x))

    test_values_g = (np.array([0.6, -0.2]), np.array([1.6, 2.2]))
    test_values_x = (1.0, -1.0, 0.6, 11.4)

    @pytest.mark.parametrize("g", test_values_g)
    def test_f_dg(self, g: np.ndarray[float]):
        """
        Test derivatives of f with respect to g.

        Parameters
        ----------
        g : np.ndarray[float]
            Test values.
        """
        value_analytic = self.f_dg(g)
        value_fd = first_derivative_finite_difference(g, self.f, (2,))

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        value_analytic = self.f_dg2(g)
        value_fd = second_derivative_finite_difference(g, self.f, (2,))

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        value_analytic = self.f_dg3(g)
        value_fd = second_derivative_finite_difference(g, self.f_dg, (2, 2))

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        value_analytic = self.f_dg4(g)
        value_fd = second_derivative_finite_difference(
            g, self.f_dg2, (2, 2, 2)
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

    @pytest.mark.parametrize("x", test_values_x)
    def test_g_dx(self, x: np.ndarray[float]):
        """
        Test derivatives of g with respect to x.

        Parameters
        ----------
        x : np.ndarray[float]
            Test values.
        """
        x = np.asarray(x).reshape((1,))
        value_analytic = self.g_dx(x)
        value_fd = first_derivative_finite_difference(x, self.g, (2,))

        nptest.assert_allclose(value_analytic, value_fd)

        value_analytic = self.g_dx2(x)
        value_fd = second_derivative_finite_difference(x, self.g, (2,))

        assert np.allclose(value_analytic, value_fd)

        value_analytic = self.g_dx3(x)
        value_fd = second_derivative_finite_difference(x, self.g_dx, (2, 2))

        assert np.allclose(value_analytic, value_fd)

        value_analytic = self.g_dx4(x)
        value_fd = second_derivative_finite_difference(
            x, self.g_dx2, (2, 2, 2)
        )

        assert np.allclose(value_analytic, value_fd)

    @pytest.mark.parametrize("x", test_values_x)
    def test_h_dx(self, x: np.ndarray[float]):
        """
        Test derivatives of composed function h(x) = f(g(x)) with respect to x.

        Parameters
        ----------
        x : np.ndarray[float]
            Test values.
        """
        x = np.asarray(x).reshape((1,))
        g = self.g(x)
        f_dg = self.f_dg(g)
        f_dg2 = self.f_dg2(g)
        f_dg3 = self.f_dg3(g)
        f_dg4 = self.f_dg4(g)

        g_dx = self.g_dx(x)
        g_dx2 = self.g_dx2(x)
        g_dx3 = self.g_dx3(x)
        g_dx4 = self.g_dx4(x)

        # First derivative.
        expected_value = first_derivative_finite_difference(
            x, self.h, (2,)
        ).reshape(2)
        actual_value = first_derivative(
            f_dg,
            g_dx,
            (2,),
            2,
            1,
        ).reshape(2)

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value)

        # Second derivative.
        expected_value = second_derivative_finite_difference(
            x, self.h, (2,)
        ).reshape(2)

        actual_value = second_derivative(
            f_dg,
            f_dg2,
            g_dx,
            g_dx2,
            (2,),
            2,
            1,
        ).reshape(2)

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value)

        # Third derivative.
        def _d2h_dx2(x):
            return second_derivative_finite_difference(x, self.h, (2,))

        # Need to use weird step sizes as third order finite difference.
        expected_value = first_derivative_finite_difference(
            x, _d2h_dx2, (2, 1, 1), order=4, h=5e-3
        ).reshape(2)

        actual_value = third_derivative(
            f_dg,
            f_dg2,
            f_dg3,
            g_dx,
            g_dx2,
            g_dx3,
            (2,),
            2,
            1,
        ).reshape(2)

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value, atol=5e-6)

        # Fourth derivative.
        # Need to use weird step sizes as fourth order finite difference.
        expected_value = second_derivative_finite_difference(
            x, _d2h_dx2, (2, 1, 1), order=4, h=5e-2
        ).reshape(2)

        actual_value = fourth_derivative(
            f_dg,
            f_dg2,
            f_dg3,
            f_dg4,
            g_dx,
            g_dx2,
            g_dx3,
            g_dx4,
            (2,),
            2,
            1,
        ).reshape(2)

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value, atol=5e-3)
