"""
Unit tests for coordinates.coordinate_system.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus import (
    first_derivative,
    second_derivative,
    third_derivative,
)
from crayon.coordinates import (
    CoordinateSystem,
    Cylindrical,
    Toroidal,
    connection_coefficients,
    connection_coefficients_2,
    connection_coefficients_2_dx,
    connection_coefficients_dx,
    forward_transform_dx2,
    forward_transform_dx3,
    metric_tensor,
)
from crayon.shared.numerics import first_derivative_finite_difference

logger = logging.getLogger(__name__)


class TestMetric:
    """
    Unit tests for metric calculation.
    """

    positions_cylindrical = (
        np.array([1.0, 0.0, 1.0]),
        np.array([np.sqrt(2), np.pi / 4, 2.0]),
        np.array([5.0, -0.92729522, 0.0]),
    )

    @staticmethod
    @pytest.mark.parametrize("x_cylindrical", positions_cylindrical)
    def test_cylindrical_metric(x_cylindrical: np.ndarray[float]):
        """
        Test metric tensor calculation for cylindrical coordinate system.

        Parameters
        ----------
        x_cylindrical : np.ndarray[float]
            Test position.
        """
        r, _, _ = x_cylindrical
        expected_value = np.identity(3)
        expected_value[1, 1] = r * r

        cylindrical = Cylindrical()
        backward_transform_dx = cylindrical.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        covariant_transform = cylindrical.covariant_transform(
            backward_transform_dx
        )

        actual_value = metric_tensor(covariant_transform)

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

    positions_toroidal = (
        np.array([0.2, 0.0, 0.0]),
        np.array([0.2, np.pi / 2, 0.0]),
        np.array([1.5620499, 0.0, 0.69473828]),
        np.array([0.61421356, 0.7853981, 0.0]),
        np.array([1.17356649, 0.7853981, 1.01999118]),
        np.array([0.43720203, 0.46364761, -0.75622683]),
    )

    @staticmethod
    @pytest.mark.parametrize("x_toroidal", positions_toroidal)
    def test_toroidal_metric(x_toroidal: np.ndarray[float]):
        """
        Test metric tensor calculation for toroidal coordinate system.

        Parameters
        ----------
        x_toroidal : np.ndarray[float]
            Test position.
        """
        r0, z0 = 0.8, 0.0

        r, _, theta = x_toroidal
        expected_value = np.zeros((3, 3))
        a = r0 + r * np.cos(theta)
        expected_value[0, 0] = 1.0
        expected_value[1, 1] = a * a
        expected_value[2, 2] = r * r

        cylindrical = Cylindrical()
        toroidal = Toroidal((r0, z0))

        # Need to compose transforms to get transform from Cartesian.
        x_cylindrical = toroidal.backward_transform(x_toroidal)

        toroidal_backward_transform_dx = toroidal.backward_transform_dx(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        cylindrical_backard_transform_dx = cylindrical.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        backward_transform_dx = first_derivative(
            cylindrical_backard_transform_dx,
            toroidal_backward_transform_dx,
            (3,),
            3,
            3,
        )

        covariant_transform = toroidal.covariant_transform(
            backward_transform_dx
        )

        actual_value = metric_tensor(covariant_transform)

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)


class TestForwardTransform:
    """
    Unit tests for calculation of forward transform.
    """

    positions_cylindrical = TestMetric.positions_cylindrical

    @staticmethod
    @pytest.mark.parametrize("x_cylindrical", positions_cylindrical)
    def test_cylindrical(x_cylindrical: np.ndarray[float]):
        """
        Test forward transform calculation for cylindrical coordinate system.

        Parameters
        ----------
        x_cylindrical : np.ndarray[float]
            Test position.
        """
        cylindrical = Cylindrical()

        expected_value = cylindrical.forward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        _forward_transform_dx = cylindrical.forward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        _backward_transform_dx2 = cylindrical.backward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = forward_transform_dx2(
            _forward_transform_dx, _backward_transform_dx2
        )

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

        expected_value = cylindrical.forward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        _forward_transform_dx2 = cylindrical.forward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        _backward_transform_dx3 = cylindrical.backward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = forward_transform_dx3(
            _forward_transform_dx,
            _forward_transform_dx2,
            _backward_transform_dx2,
            _backward_transform_dx3,
        )

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)


class TestConnectionCoefficients:
    """
    Unit tests for calculation of connection coefficients aka Christoffel
    symbols of second kind.
    """

    positions_cylindrical = TestMetric.positions_cylindrical

    @staticmethod
    def gamma_cylindrical(
        x_cylindrical: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Calculate connection coefficients for cylindrical coordinate
        system.

        Parameters
        ----------
        x_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        gamma_cylindrical : np.ndarray[float]
            Connection coefficients for cylindrical coordinate system.
        """
        r, _, _ = x_cylindrical

        gamma = np.zeros((3, 3, 3))
        gamma[0, 1, 1] = -r
        gamma[1, 0, 1] = 1 / r
        gamma[1, 1, 0] = 1 / r

        return gamma

    @pytest.mark.parametrize("x_cylindrical", positions_cylindrical)
    def test_cylindrical(self, x_cylindrical: np.ndarray[float]):
        """
        Test connection coefficient calculation for cylindrical coordinate
        system.

        Parameters
        ----------
        x_cylindrical : np.ndarray[float]
            Test position.
        """
        # Values.
        expected_value = self.gamma_cylindrical(x_cylindrical)

        cylindrical = Cylindrical()
        x_cartesian = cylindrical.backward_transform(x_cylindrical)
        forward_transform_dx = cylindrical.forward_transform_dx(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        backward_transform_dx = cylindrical.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        forward_transform_dx2 = cylindrical.forward_transform_dx2(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        backward_transform_dx2 = cylindrical.backward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        gamma = connection_coefficients(
            backward_transform_dx, forward_transform_dx2
        )
        gamma_2 = connection_coefficients_2(
            forward_transform_dx, backward_transform_dx2
        )

        logger.info(abs(gamma - expected_value))
        nptest.assert_allclose(gamma, expected_value, atol=1e-8)

        logger.info(abs(gamma_2 - expected_value))
        nptest.assert_allclose(gamma_2, expected_value, atol=1e-8)

        # First derivative.
        expected_value = first_derivative_finite_difference(
            x_cylindrical, self.gamma_cylindrical, (3, 3, 3)
        )

        forward_transform_dx3 = cylindrical.forward_transform_dx3(
            x_cartesian, CoordinateSystem.CARTESIAN
        )

        actual_value = connection_coefficients_dx(
            gamma, backward_transform_dx, forward_transform_dx3
        )

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

        backward_transform_dx3 = cylindrical.backward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = connection_coefficients_2_dx(
            gamma_2,
            forward_transform_dx,
            backward_transform_dx3,
        )

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

    positions_toroidal = TestMetric.positions_toroidal

    r0, z0 = 0.8, 0.0

    def gamma_toroidal(
        self, x_toroidal: np.ndarray[float]
    ) -> np.ndarray[float]:
        """
        Calculate connection coefficients for toroidal coordinate
        system.

        Parameters
        ----------
        x_toroidal : np.ndarray[float]
            Toroidal position.

        Returns
        -------
        gamma_toroidal : np.ndarray[float]
            Connection coefficients for toroidal coordinate system.
        """
        r, _, theta = x_toroidal
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        a = self.r0 + r * cos_theta

        gamma = np.zeros((3, 3, 3))

        gamma[0, 1, 1] = -a * cos_theta
        gamma[0, 2, 2] = -r

        gamma[1, 0, 1] = cos_theta / a
        gamma[1, 1, 0] = gamma[1, 0, 1]
        gamma[1, 1, 2] = -r * sin_theta / a
        gamma[1, 2, 1] = gamma[1, 1, 2]

        gamma[2, 0, 2] = 1 / r
        gamma[2, 2, 0] = gamma[2, 0, 2]
        gamma[2, 1, 1] = a * sin_theta / r

        return gamma

    @pytest.mark.parametrize("x_toroidal", positions_toroidal)
    def test_toroidal(self, x_toroidal: np.ndarray[float]):
        """
        Test connection coefficient calculation for toroidal coordinate
        system.

        Parameters
        ----------
        x_toroidal : np.ndarray[float]
            Test position.
        """
        expected_value = self.gamma_toroidal(x_toroidal)

        cylindrical = Cylindrical()
        toroidal = Toroidal((self.r0, self.z0))

        # Need to compose transforms to get transform from Cartesian.
        x_cylindrical = toroidal.backward_transform(x_toroidal)

        toroidal_forward_transform_dx = toroidal.forward_transform_dx(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        toroidal_backward_transform_dx = toroidal.backward_transform_dx(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        toroidal_backward_transform_dx2 = toroidal.backward_transform_dx2(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        toroidal_backward_transform_dx3 = toroidal.backward_transform_dx3(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        cylindrical_foward_transform_dx = cylindrical.forward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        cylindrical_backard_transform_dx = cylindrical.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        cylindrical_backard_transform_dx2 = cylindrical.backward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        cylindrical_backard_transform_dx3 = cylindrical.backward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        forward_transform_dx = first_derivative(
            toroidal_forward_transform_dx,
            cylindrical_foward_transform_dx,
            (3,),
            3,
            3,
        )

        backward_transform_dx2 = second_derivative(
            cylindrical_backard_transform_dx,
            cylindrical_backard_transform_dx2,
            toroidal_backward_transform_dx,
            toroidal_backward_transform_dx2,
            (3,),
            3,
            3,
        )

        backward_transform_dx3 = third_derivative(
            cylindrical_backard_transform_dx,
            cylindrical_backard_transform_dx2,
            cylindrical_backard_transform_dx3,
            toroidal_backward_transform_dx,
            toroidal_backward_transform_dx2,
            toroidal_backward_transform_dx3,
            (3,),
            3,
            3,
        )

        gamma_2 = connection_coefficients_2(
            forward_transform_dx, backward_transform_dx2
        )

        logger.info(abs(gamma_2 - expected_value))
        nptest.assert_allclose(gamma_2, expected_value, atol=1e-8)

        # First derivative.
        expected_value = first_derivative_finite_difference(
            x_toroidal, self.gamma_toroidal, (3, 3, 3)
        )

        actual_value = connection_coefficients_2_dx(
            gamma_2,
            forward_transform_dx,
            backward_transform_dx3,
        )

        logger.info(abs(actual_value - expected_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)
