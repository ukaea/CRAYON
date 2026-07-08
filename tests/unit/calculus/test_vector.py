"""
Unit tests for calculus.vector
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus.vector import (
    component_parallel,
    component_parallel2,
    component_perp,
    component_perp2,
    rotation_a_onto_b,
    unit_vector,
    unit_vector_first_derivative_x,
    unit_vector_second_derivative_x,
    v_parallel2_first_derivative_v,
    v_parallel2_first_derivative_x,
    v_parallel2_second_derivative_v,
    v_parallel2_second_derivative_x,
    v_parallel2_second_derivative_xv,
    v_parallel_first_derivative_v,
    v_parallel_first_derivative_x,
    v_parallel_second_derivative_v,
    v_parallel_second_derivative_x,
    v_parallel_second_derivative_xv,
    v_perp2_first_derivative_v,
    v_perp2_first_derivative_x,
    v_perp2_second_derivative_v,
    v_perp2_second_derivative_x,
    v_perp2_second_derivative_xv,
    v_perp_first_derivative_v,
    v_perp_first_derivative_x,
    v_perp_second_derivative_v,
    v_perp_second_derivative_x,
    v_perp_second_derivative_xv,
    vector_magnitude,
    vector_magnitude_first_derivative_x,
    vector_magnitude_second_derivative_x,
)
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
    second_mixed_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


class TestVectorDerivative:
    """
    Test formulas for vector components, transforms and derivatives.
    """

    @staticmethod
    def vector(position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Test vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        vector : np.ndarray[float]
            Vector.
        """
        x, y, z = position

        v = np.zeros(3)
        v[0] = z**2
        v[1] = x**2
        v[2] = y**2

        return v

    @staticmethod
    def magnitude(position: np.ndarray[float]) -> float:
        """
        Vector magnitude.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        magnitude : float
            Vector magnitude.
        """
        x, y, z = position
        return (x**4 + y**4 + z**4) ** 0.5

    @staticmethod
    def unit(position: np.ndarray[float]) -> np.ndarray[float]:
        """
        Unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        unit_vector : np.ndarray[float]
            Vector normalised to unit length.
        """
        x, y, z = position

        denom = (x**4 + y**4 + z**4) ** 0.5
        unit = np.zeros(3)
        unit[0] = z**2 / denom
        unit[1] = x**2 / denom
        unit[2] = y**2 / denom

        return unit

    @staticmethod
    def vector_first_derivative_x(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Calculate first derivative of vector with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        vector_dx : np.ndarray[float]
            First derivative of vector with respect to x.

        Notes
        -----
        vector_dx[i, j] = d{v[i]} / dx^j
        """
        x, y, z = position

        dv_dx = np.zeros((3, 3))
        dv_dx[0, 2] = 2 * z
        dv_dx[1, 0] = 2 * x
        dv_dx[2, 1] = 2 * y

        return dv_dx

    @staticmethod
    def vector_second_derivative_x() -> np.ndarray[float]:
        """
        Calculate second derivative of vector with respect to x.

        Returns
        -------
        vector_dx2 : np.ndarray[float]
            Second derivative of vector with respect to x.

        Notes
        -----
        vector_dx2[i, j, k] = d{v[i]} / dx^j dx^k
        """
        d2v_dx2 = np.zeros((3, 3, 3))
        d2v_dx2[0, 2, 2] = 2
        d2v_dx2[1, 0, 0] = 2
        d2v_dx2[2, 1, 1] = 2

        return d2v_dx2

    @staticmethod
    def magnitude_first_derivative_x(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Calculate first derivative of vector magnitude with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        magnitude_dx : np.ndarray[float]
            First derivative of vector magnitude with respect to x.

        Notes
        -----
        magnitude_dx[i] = d|v| / dx^i
        """
        x, y, z = position

        denom = (x**4 + y**4 + z**4) ** 0.5
        dmagnitude_dx = np.zeros(3)
        dmagnitude_dx[0] = 2 * x**3 / denom
        dmagnitude_dx[1] = 2 * y**3 / denom
        dmagnitude_dx[2] = 2 * z**3 / denom

        return dmagnitude_dx

    @staticmethod
    def magnitude_second_derivative_x(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Calculate second derivative of vector magnitude with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        magnitude_dx2 : np.ndarray[float]
            Second derivative of vector magnitude with respect to x.

        Notes
        -----
        magnitude_dx2[i, j] = d^2 |v| / dx^i dx^j
        """
        x, y, z = position

        denom = (x**4 + y**4 + z**4) ** 1.5
        d2magnitude_dx2 = np.zeros((3, 3))
        d2magnitude_dx2[0, 0] = 2 * x**2 * (x**4 + 3 * (y**4 + z**4)) / denom
        d2magnitude_dx2[0, 1] = -4 * x**3 * y**3 / denom
        d2magnitude_dx2[0, 2] = -4 * x**3 * z**3 / denom
        d2magnitude_dx2[1, 0] = d2magnitude_dx2[0, 1]
        d2magnitude_dx2[1, 1] = 2 * y**2 * (y**4 + 3 * (x**4 + z**4)) / denom
        d2magnitude_dx2[1, 2] = -4 * y**3 * z**3 / denom
        d2magnitude_dx2[2, 0] = d2magnitude_dx2[0, 2]
        d2magnitude_dx2[2, 1] = d2magnitude_dx2[1, 2]
        d2magnitude_dx2[2, 2] = 2 * z**2 * (z**4 + 3 * (x**4 + y**4)) / denom

        return d2magnitude_dx2

    @staticmethod
    def unit_first_derivative_x(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Calculate first derivative of unit vector with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        unit_dx : np.ndarray[float]
            First derivative of unit vector with respect to x.

        Notes
        -----
        unit_dx[i, j] = dv_hat[i] / dx^j.
        """
        x, y, z = position

        denom = (x**4 + y**4 + z**4) ** 1.5
        dunit_dx = np.zeros((3, 3))
        dunit_dx[0, 0] = -2 * x**3 * z**2 / denom
        dunit_dx[0, 1] = -2 * y**3 * z**2 / denom
        dunit_dx[0, 2] = 2 * z * (x**4 + y**4) / denom
        dunit_dx[1, 0] = 2 * x * (z**4 + y**4) / denom
        dunit_dx[1, 1] = -2 * y**3 * x**2 / denom
        dunit_dx[1, 2] = -2 * z**3 * x**2 / denom
        dunit_dx[2, 0] = -2 * x**3 * y**2 / denom
        dunit_dx[2, 1] = 2 * y * (x**4 + z**4) / denom
        dunit_dx[2, 2] = -2 * z**3 * y**2 / denom

        return dunit_dx

    @staticmethod
    def unit_second_derivative_x(
        position: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Calculate second derivative of unit vector with respect to x.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.

        Returns
        -------
        unit_dx2 : np.ndarray[float]
            Second derivative of unit vector with respect to x.

        Notes
        -----
        unit_dx2[i, j, k] = d2v_hat[i] / dx^j dx^k.
        """
        x, y, z = position

        denom = (x**4 + y**4 + z**4) ** 2.5

        d2unit_dx2 = np.zeros((3, 3, 3))
        d2unit_dx2[0, 0, 0] = 6 * x**2 * z**2 * (x**4 - y**4 - z**4) / denom
        d2unit_dx2[0, 0, 1] = 12 * x**3 * y**3 * z**2 / denom
        d2unit_dx2[0, 0, 2] = 4 * x**3 * z * (2 * z**4 - x**4 - y**4) / denom
        d2unit_dx2[0, 1, 0] = d2unit_dx2[0, 0, 1]
        d2unit_dx2[0, 1, 1] = 6 * y**2 * z**2 * (y**4 - x**4 - z**4) / denom
        d2unit_dx2[0, 1, 2] = 4 * y**3 * z * (2 * z**4 - x**4 - y**4) / denom
        d2unit_dx2[0, 2, 0] = d2unit_dx2[0, 0, 2]
        d2unit_dx2[0, 2, 1] = d2unit_dx2[0, 1, 2]
        d2unit_dx2[0, 2, 2] = (
            -2 * (x**4 + y**4) * (5 * z**4 - x**4 - y**4) / denom
        )

        d2unit_dx2[1, 0, 0] = (
            -2 * (z**4 + y**4) * (5 * x**4 - z**4 - y**4) / denom
        )
        d2unit_dx2[1, 0, 1] = 4 * y**3 * x * (2 * x**4 - z**4 - y**4) / denom
        d2unit_dx2[1, 0, 2] = 4 * z**3 * x * (2 * x**4 - z**4 - y**4) / denom
        d2unit_dx2[1, 1, 0] = d2unit_dx2[1, 0, 1]
        d2unit_dx2[1, 1, 1] = 6 * y**2 * x**2 * (y**4 - x**4 - z**4) / denom
        d2unit_dx2[1, 1, 2] = 12 * z**3 * y**3 * x**2 / denom
        d2unit_dx2[1, 2, 0] = d2unit_dx2[1, 0, 2]
        d2unit_dx2[1, 2, 1] = d2unit_dx2[1, 1, 2]
        d2unit_dx2[1, 2, 2] = 6 * x**2 * z**2 * (z**4 - y**4 - x**4) / denom

        d2unit_dx2[2, 0, 0] = 6 * x**2 * y**2 * (x**4 - y**4 - z**4) / denom
        d2unit_dx2[2, 0, 1] = 4 * x**3 * y * (2 * y**4 - x**4 - z**4) / denom
        d2unit_dx2[2, 0, 2] = 12 * x**3 * z**3 * y**2 / denom
        d2unit_dx2[2, 1, 0] = d2unit_dx2[2, 0, 1]
        d2unit_dx2[2, 1, 1] = (
            -2 * (x**4 + z**4) * (5 * y**4 - x**4 - z**4) / denom
        )
        d2unit_dx2[2, 1, 2] = 4 * z**3 * y * (2 * y**4 - x**4 - z**4) / denom
        d2unit_dx2[2, 2, 0] = d2unit_dx2[2, 0, 2]
        d2unit_dx2[2, 2, 1] = d2unit_dx2[2, 1, 2]
        d2unit_dx2[2, 2, 2] = 6 * y**2 * z**2 * (z**4 - x**4 - y**4) / denom

        return d2unit_dx2

    test_values_1 = (
        "position",
        (
            np.array([0.29, 0.19, 0.16]),
            np.array([0.30, 0.64, 0.72]),
            np.array([0.30, 0.46, 0.72]),
            np.array([0.67, 0.61, 0.99]),
            np.array([0.91, 0.93, 0.60]),
            np.array([0.38, 0.61, 0.89]),
        ),
    )

    @pytest.mark.parametrize(*test_values_1)
    def test_vector_derivative(self, position: np.ndarray[float]):
        """
        Test analytic function for first derivative of vector with respect
        to position compared to finite differences.

        Parameters
        ----------
        position : np.ndarray[float]
            Test position.
        """
        # First derivative.
        value_analytic = self.vector_first_derivative_x(position)
        value_fd = first_derivative_finite_difference(
            position, self.vector, (3,)
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # Second derivative.
        value_analytic = self.vector_second_derivative_x()
        value_fd = second_derivative_finite_difference(
            position, self.vector, (3,)
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

    @pytest.mark.parametrize(*test_values_1)
    def test_magnitude(self, position: np.ndarray[float]):
        """
        Test analytic function for first derivative of magnitude with respect
        to position compared to finite differences.

        Parameters
        ----------
        position : np.ndarray[float]
            Test position.
        """
        # Value.
        expected_value = self.magnitude(position)
        v = self.vector(position)
        actual_value = vector_magnitude(v)

        logger.warning(abs(expected_value - actual_value))
        assert np.isclose(expected_value, actual_value)

        # First derivative.
        dv_dx = self.vector_first_derivative_x(position)

        value_analytic = self.magnitude_first_derivative_x(position)
        value_fd = first_derivative_finite_difference(
            position, self.magnitude, ()
        )
        value_analytic2 = vector_magnitude_first_derivative_x(v, dv_dx)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2)

        # Second derivative.
        d2v_dx2 = self.vector_second_derivative_x()

        value_analytic = self.magnitude_second_derivative_x(position)
        value_fd = second_derivative_finite_difference(
            position, self.magnitude, (), order=4
        )
        value_analytic2 = vector_magnitude_second_derivative_x(
            v, dv_dx, d2v_dx2
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=1e-7)

    @pytest.mark.parametrize(*test_values_1)
    def test_unit(self, position: np.ndarray[float]):
        """
        Test unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Test position.
        """
        # Value
        expected_value = self.unit(position)
        v = self.vector(position)
        actual_value = unit_vector(v)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # First derivative.
        dv_dx = self.vector_first_derivative_x(position)

        value_analytic = self.unit_first_derivative_x(position)
        value_fd = first_derivative_finite_difference(
            position, self.unit, (3,)
        )
        value_analytic2 = unit_vector_first_derivative_x(v, dv_dx)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2)

        # Second derivative.
        d2v_dx2 = self.vector_second_derivative_x()

        value_analytic = self.unit_second_derivative_x(position)
        value_fd = second_derivative_finite_difference(
            position, self.unit, (3,), order=4
        )
        value_analytic2 = unit_vector_second_derivative_x(v, dv_dx, d2v_dx2)

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        logger.warning(abs(value_analytic - value_analytic2))
        nptest.assert_allclose(value_analytic, value_analytic2, atol=1e-7)

    def v_perp(
        self, position: np.ndarray[float], vector: np.ndarray[float]
    ) -> float:
        """
        Calculate magnitude of component of vector perpendicular to unit
        vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.

        Returns
        -------
        v_perp : float
            Perpendicular component.
        """
        unit = self.unit(position)
        return component_perp(vector, unit)

    def v_perp2(self, position, vector) -> float:
        """
        Calculate squared magnitude of component of vector perpendicular to
        unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.

        Returns
        -------
        v_perp2 : float
            Perpendicular component squared.
        """
        unit = self.unit(position)
        return component_perp2(vector, unit)

    def v_parallel(self, position, vector) -> float:
        """
        Calculate magnitude of component of vector parallel to unit
        vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.

        Returns
        -------
        v_parallel : float
            Parallel component.
        """
        unit = self.unit(position)
        return component_parallel(vector, unit)

    def v_parallel2(self, position, vector) -> float:
        """
        Calculate squared magnitude of component of vector parallel to
        unit vector.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.

        Returns
        -------
        v_parallel2 : float
            Perpendicular component squared.
        """
        unit = self.unit(position)
        return component_parallel2(vector, unit)

    test_values2 = (
        "position, vector",
        (
            (np.array([0.29, 0.19, 0.16]), np.array([0.06, 0.05, 0.79])),
            (np.array([0.30, 0.64, 0.72]), np.array([0.30, 0.64, 0.72])),
            (np.array([0.30, 0.46, 0.72]), np.array([0.76, 0.83, 0.93])),
            (np.array([0.67, 0.61, 0.99]), np.array([0.05, 0.61, 0.51])),
            (np.array([0.91, 0.93, 0.60]), np.array([0.37, 0.92, 0.02])),
            (np.array([0.38, 0.61, 0.89]), np.array([0.41, 0.86, 0.22])),
        ),
    )

    @pytest.mark.parametrize(*test_values2)
    def test_v_perp_derivative(
        self, position: np.ndarray[float], vector: np.ndarray[float]
    ):
        """
        Test derivatives of v_perp.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.
        """
        # First derivative x.
        n = self.unit(position)
        dn_dx = self.unit_first_derivative_x(position)

        value_analytic = v_perp_first_derivative_x(vector, n, dn_dx)
        value_fd = first_derivative_finite_difference(
            position, lambda x: self.v_perp(x, vector), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # First derivative v.
        value_analytic = v_perp_first_derivative_v(vector, n)
        value_fd = first_derivative_finite_difference(
            vector, lambda v: self.v_perp(position, v), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # Second derivative x.
        d2n_dx2 = self.unit_second_derivative_x(position)

        value_analytic = v_perp_second_derivative_x(vector, n, dn_dx, d2n_dx2)
        value_fd = second_derivative_finite_difference(
            position, lambda x: self.v_perp(x, vector), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative v.
        value_analytic = v_perp_second_derivative_v(vector, n)
        value_fd = second_derivative_finite_difference(
            vector, lambda v: self.v_perp(position, v), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative xv.
        value_analytic = v_perp_second_derivative_xv(vector, n, dn_dx)
        value_fd = second_mixed_derivative_finite_difference(
            position, vector, self.v_perp, (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

    @pytest.mark.parametrize(*test_values2)
    def test_v_perp2_derivative(
        self, position: np.ndarray[float], vector: np.ndarray[float]
    ):
        """
        Test derivatives of v_perp squared.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.
        """
        # First derivative x.
        n = self.unit(position)
        dn_dx = self.unit_first_derivative_x(position)

        value_analytic = v_perp2_first_derivative_x(vector, n, dn_dx)
        value_fd = first_derivative_finite_difference(
            position, lambda x: self.v_perp2(x, vector), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # First derivative v.
        value_analytic = v_perp2_first_derivative_v(vector, n)
        value_fd = first_derivative_finite_difference(
            vector, lambda v: self.v_perp2(position, v), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # Second derivative x.
        d2n_dx2 = self.unit_second_derivative_x(position)

        value_analytic = v_perp2_second_derivative_x(vector, n, dn_dx, d2n_dx2)
        value_fd = second_derivative_finite_difference(
            position, lambda x: self.v_perp2(x, vector), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative v.
        value_analytic = v_perp2_second_derivative_v(vector, n)
        value_fd = second_derivative_finite_difference(
            vector, lambda v: self.v_perp2(position, v), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative xv.
        value_analytic = v_perp2_second_derivative_xv(vector, n, dn_dx)
        value_fd = second_mixed_derivative_finite_difference(
            position, vector, self.v_perp2, (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

    @pytest.mark.parametrize(*test_values2)
    def test_v_parallel_derivative(
        self, position: np.ndarray[float], vector: np.ndarray[float]
    ):
        """
        Test derivatives of v_parallel.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.
        """
        n = self.unit(position)
        dn_dx = self.unit_first_derivative_x(position)
        d2n_dx2 = self.unit_second_derivative_x(position)

        # First derivative x.
        value_analytic = v_parallel_first_derivative_x(vector, dn_dx)
        value_fd = first_derivative_finite_difference(
            position, lambda x: self.v_parallel(x, vector), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # First derivative v.
        value_analytic = v_parallel_first_derivative_v(vector, n)
        value_fd = first_derivative_finite_difference(
            vector, lambda v: self.v_parallel(position, v), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # Second derivative x.
        value_analytic = v_parallel_second_derivative_x(vector, d2n_dx2)
        value_fd = second_derivative_finite_difference(
            position, lambda x: self.v_parallel(x, vector), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative v.
        value_analytic = v_parallel_second_derivative_v(vector)
        value_fd = second_derivative_finite_difference(
            vector, lambda v: self.v_parallel(position, v), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative xv.
        value_analytic = v_parallel_second_derivative_xv(vector, dn_dx)
        value_fd = second_mixed_derivative_finite_difference(
            position, vector, self.v_parallel, (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

    @pytest.mark.parametrize(*test_values2)
    def test_v_parallel2_derivative(
        self, position: np.ndarray[float], vector: np.ndarray[float]
    ):
        """
        Test derivatives of v_parallel squared.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        vector : np.ndarray[float]
            Vector to get component of.
        """
        n = self.unit(position)
        dn_dx = self.unit_first_derivative_x(position)
        d2n_dx2 = self.unit_second_derivative_x(position)

        # First derivative x.
        value_analytic = v_parallel2_first_derivative_x(vector, n, dn_dx)
        value_fd = first_derivative_finite_difference(
            position, lambda x: self.v_parallel2(x, vector), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # First derivative v.
        value_analytic = v_parallel2_first_derivative_v(vector, n)
        value_fd = first_derivative_finite_difference(
            vector, lambda v: self.v_parallel2(position, v), ()
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd)

        # Second derivative x.
        value_analytic = v_parallel2_second_derivative_x(
            vector, n, dn_dx, d2n_dx2
        )
        value_fd = second_derivative_finite_difference(
            position, lambda x: self.v_parallel2(x, vector), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)

        # Second derivative v.
        value_analytic = v_parallel2_second_derivative_v(vector, n)
        value_fd = second_derivative_finite_difference(
            vector, lambda v: self.v_parallel2(position, v), (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=2e-7)

        # Second derivative xv.
        value_analytic = v_parallel2_second_derivative_xv(vector, n, dn_dx)
        value_fd = second_mixed_derivative_finite_difference(
            position, vector, self.v_parallel2, (), order=4
        )

        logger.warning(abs(value_analytic - value_fd))
        nptest.assert_allclose(value_analytic, value_fd, atol=1e-7)


def test_rotation_a_onto_b():
    """
    Test rotation matrix for rotating a onto b.
    """
    # Test null rotation.
    a = np.array([1.0, 0.0, 0.0])
    rot = rotation_a_onto_b(a, a)

    nptest.assert_allclose(rot, np.identity(3))

    # Test simple rotation.
    b = np.array([0.0, 1.0, 0.0])
    rot = rotation_a_onto_b(a, b)

    nptest.assert_allclose(rot @ a, b)

    # Test general rotation.
    a = np.array([0.97, 0.83, 0.77])
    b = np.array([0.62, 0.98, 0.82])

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    rot = rotation_a_onto_b(a, b)

    nptest.assert_allclose(np.linalg.det(rot), 1.0)
    nptest.assert_allclose(rot @ a, b)
    nptest.assert_allclose(rot.T @ b, a)

    # Test general rotation.
    a = np.array([-0.97, 0.83, -0.77])
    b = np.array([-0.62, -0.98, -0.82])

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    rot = rotation_a_onto_b(a, b)

    nptest.assert_allclose(np.linalg.det(rot), 1.0)
    nptest.assert_allclose(rot @ a, b)
    nptest.assert_allclose(rot.T @ b, a)
