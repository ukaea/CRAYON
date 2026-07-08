"""
Unit tests for calculus.tensor
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus.tensor import (
    TensorType,
    first_covariant_derivative,
    get_transform,
    second_covariant_derivative,
    transform_tensor_field,
)
from crayon.coordinates import (
    CoordinateSystem,
    Cylindrical,
    connection_coefficients,
    connection_coefficients_dx,
)
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


class TestTensor:
    """
    Unit tests for tensor formulas.
    """

    @staticmethod
    def scalar_cartesian(position_cartesian: np.ndarray[float]) -> float:
        """
        Scalar valued function of Cartesian position.

        Parameters
        ----------
        position_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        value : float
            Scalar value.
        """
        x, _, z = position_cartesian

        return x**2 + z**2

    @staticmethod
    def scalar_cartesian_dx(
        position_cartesian: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        First derivative of scalar valued function with respect to position.

        Parameters
        ----------
        position_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        value_dx : float
            First derivative of scalar value.
        """
        x, _, z = position_cartesian

        return_array = np.zeros(3)
        return_array[0] = 2 * x
        return_array[2] = 2 * z

        return return_array

    @staticmethod
    def scalar_cartesian_dx2():
        """
        Second derivative of scalar valued function with respect to position.

        Returns
        -------
        value_dx2 : float
            Second derivative of scalar value.
        """
        return_array = np.zeros((3, 3))
        return_array[0, 0] = 2
        return_array[2, 2] = 2

        return return_array

    @staticmethod
    def scalar_cylindrical(position_cylindrical: np.ndarray[float]) -> float:
        """
        Scalar valued function of cylindrical position.

        Parameters
        ----------
        position_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        value : float
            Scalar value.
        """
        r, phi, z = position_cylindrical
        c, _ = np.cos(phi), np.sin(phi)

        return r**2 * c**2 + z**2

    @staticmethod
    def scalar_cylindrical_dx(
        position_cylindrical: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        First derivative of scalar valued function with respect to position.

        Parameters
        ----------
        position_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        value_dx : float
            First derivative of scalar value.
        """
        r, phi, z = position_cylindrical
        c, s = np.cos(phi), np.sin(phi)

        return_array = np.zeros(3)
        return_array[0] = 2 * r * c**2
        return_array[1] = -2 * r**2 * s * c
        return_array[2] = 2 * z

        return return_array

    @staticmethod
    def scalar_cylindrical_dx2(position_cylindrical):
        """
        Second derivative of scalar valued function with respect to position.

        Parameters
        ----------
        position_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        value_dx2 : float
            Second derivative of scalar value.
        """
        r, phi, _ = position_cylindrical
        c, s = np.cos(phi), np.sin(phi)

        return_array = np.zeros((3, 3))
        return_array[0, 0] = 2 * c**2
        return_array[0, 1] = -4 * r * c * s
        return_array[1, 0] = return_array[0, 1]
        return_array[1, 1] = -2 * r**2 * (c**2 - s**2)
        return_array[2, 2] = 2

        return return_array

    @staticmethod
    def vector_cartesian(
        position_cartesian: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Vector valued function of Cartesian position.

        Parameters
        ----------
        position_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        value : float
            Vector value.
        """
        x, y, z = position_cartesian

        return_array = np.zeros(3)
        return_array[0] = y * z**2
        return_array[1] = x
        return_array[2] = y**2

        return return_array

    @staticmethod
    def vector_cartesian_dx(
        position_cartesian: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        First derivative of vector valued function with respect to position.

        Parameters
        ----------
        position_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        value_dx : float
            First derivative of vector value.
        """
        _, y, z = position_cartesian

        return_array = np.zeros((3, 3))

        return_array[0, 1] = z**2
        return_array[0, 2] = 2 * y * z
        return_array[1, 0] = 1
        return_array[2, 1] = 2 * y

        return return_array

    @staticmethod
    def vector_cartesian_dx2(
        position_cartesian: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Second derivative of vector valued function with respect to position.

        Parameters
        ----------
        position_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        value_dx2 : float
            Second derivative of vector value.
        """
        _, y, z = position_cartesian

        return_array = np.zeros((3, 3, 3))
        return_array[0, 1, 2] = 2 * z
        return_array[0, 2, 1] = return_array[0, 1, 2]
        return_array[0, 2, 2] = 2 * y
        return_array[2, 1, 1] = 2

        return return_array

    @staticmethod
    def vector_cylindrical(
        position_cylindrical: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Vector valued function of cylindrical position.

        Parameters
        ----------
        position_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        value : float
            Vector value.
        """
        r, phi, z = position_cylindrical
        c, s = np.cos(phi), np.sin(phi)
        r2, z2, c2, s2 = r * r, z * z, c * c, s * s

        return_array = np.zeros(3)

        return_array[0] = r * c * s * (1 + z2)
        return_array[1] = c2 - z2 * s2
        return_array[2] = r2 * s2

        return return_array

    @staticmethod
    def vector_cylindrical_dx(
        position_cylindrical: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        First derivative of vector valued function with respect to position.

        Parameters
        ----------
        position_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        value_dx : float
            First derivative of vector value.
        """
        r, phi, z = position_cylindrical
        c, s = np.cos(phi), np.sin(phi)
        r2, z2, c2, s2 = r * r, z * z, c * c, s * s

        return_array = np.zeros((3, 3))

        return_array[0, 0] = s * c * (1 + z2)
        return_array[0, 1] = r * (1 + z2) * (c2 - s2)
        return_array[0, 2] = 2 * z * r * s * c
        return_array[1, 1] = -2 * c * s * (1 + z2)
        return_array[1, 2] = -2 * z * s2
        return_array[2, 0] = 2 * r * s2
        return_array[2, 1] = 2 * r2 * s * c

        return return_array

    @staticmethod
    def vector_cylindrical_dx2(
        position_cylindrical: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Second derivative of vector valued function with respect to position.

        Parameters
        ----------
        position_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        value_dx2 : float
            Second derivative of vector value.
        """
        r, phi, z = position_cylindrical
        c, s = np.cos(phi), np.sin(phi)
        r2, z2, c2, s2 = r * r, z * z, c * c, s * s

        return_array = np.zeros((3, 3, 3))

        return_array[0, 0, 1] = (c2 - s2) * (1 + z2)
        return_array[0, 0, 2] = 2 * z * s * c
        return_array[0, 1, 0] = return_array[0, 0, 1]
        return_array[0, 1, 1] = -4 * s * c * r * (1 + z2)
        return_array[0, 1, 2] = 2 * z * r * (c2 - s2)
        return_array[0, 2, 0] = return_array[0, 0, 2]
        return_array[0, 2, 1] = return_array[0, 1, 2]
        return_array[0, 2, 2] = 2 * r * s * c

        return_array[1, 1, 1] = -2 * (1 + z2) * (c2 - s2)
        return_array[1, 1, 2] = -4 * z * s * c
        return_array[1, 2, 1] = return_array[1, 1, 2]
        return_array[1, 2, 2] = -2 * s2

        return_array[2, 0, 0] = 2 * s2
        return_array[2, 0, 1] = 4 * r * s * c
        return_array[2, 1, 0] = return_array[2, 0, 1]
        return_array[2, 1, 1] = 2 * r2 * (c2 - s2)

        return return_array

    positions_cartesian = (
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([3.0, -4.0, 0.0]),
    )

    positions_cylindrical = (
        np.array([1.0, 0.0, 1.0]),
        np.array([np.sqrt(2), np.pi / 4, 2.0]),
        np.array([5.0, -0.92729522, 0.0]),
    )

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_derivatives(
        self, x_cartesian: np.ndarray[float], x_cylindrical: np.ndarray[float]
    ):
        """
        Test derivative functions against finite difference.

        Parameters
        ----------
        x_cartesian : np.ndarray[float]
            Test position Cartesian components.
        x_cylindrical : np.ndarray[float]
            Test position cylindrical components.
        """
        # Rank 0 Cartesian.
        expected_value = self.scalar_cartesian_dx(x_cartesian)
        actual_value = first_derivative_finite_difference(
            x_cartesian, self.scalar_cartesian, ()
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = self.scalar_cartesian_dx2()
        actual_value = second_derivative_finite_difference(
            x_cartesian, self.scalar_cartesian, (), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        # Rank 0 cylindrical.
        expected_value = self.scalar_cylindrical_dx(x_cylindrical)
        actual_value = first_derivative_finite_difference(
            x_cylindrical, self.scalar_cylindrical, ()
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = self.scalar_cylindrical_dx2(x_cylindrical)
        actual_value = second_derivative_finite_difference(
            x_cylindrical, self.scalar_cylindrical, (), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        # Rank 1 Cartesian.
        expected_value = self.vector_cartesian_dx(x_cartesian)
        actual_value = first_derivative_finite_difference(
            x_cartesian, self.vector_cartesian, (3,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = self.vector_cartesian_dx2(x_cartesian)
        actual_value = second_derivative_finite_difference(
            x_cartesian, self.vector_cartesian, (3,), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        # Rank 1 cylindrical.
        expected_value = self.vector_cylindrical_dx(x_cylindrical)
        actual_value = first_derivative_finite_difference(
            x_cylindrical, self.vector_cylindrical, (3,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = self.vector_cylindrical_dx2(x_cylindrical)
        actual_value = second_derivative_finite_difference(
            x_cylindrical, self.vector_cylindrical, (3,), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

    @staticmethod
    def test_get_transform():
        """
        Get getting covariant and contravariant transform.
        """
        covariant_transform = 1.0
        contravariant_transform = 2.0

        value = get_transform(
            covariant_transform, contravariant_transform, index_covariant=True
        )

        nptest.assert_allclose(value, covariant_transform)

        value = get_transform(
            covariant_transform, contravariant_transform, index_covariant=False
        )

        nptest.assert_allclose(value, contravariant_transform)

    @staticmethod
    @pytest.mark.parametrize("x_cylindrical", positions_cylindrical)
    def test_basis_vectors_transform(x_cylindrical: np.ndarray[float]):
        """
        Test basis vectors for tangent and cotangent basis transform as
        expected.

        Parameters
        ----------
        x_cylindrical : np.ndarray[float]
            Test position cylindrical components.
        """
        cylindrical = Cylindrical()
        covariant_transform = cylindrical.covariant_transform(
            cylindrical.backward_transform_dx(
                x_cylindrical, CoordinateSystem.CYLINDRICAL
            )
        )
        contravariant_transform = cylindrical.contravariant_transform(
            cylindrical.forward_transform_dx(
                x_cylindrical, CoordinateSystem.CYLINDRICAL
            )
        )

        r, phi, _ = x_cylindrical
        c, s = np.cos(phi), np.sin(phi)

        inputs = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        # Tangent basis Cartesian -> cylindrical.
        outputs = (
            np.array([c, -r * s, 0.0]),
            np.array([s, r * c, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        for _input, expected_value in zip(inputs, outputs, strict=True):
            actual_value = transform_tensor_field(
                TensorType.COVECTOR,
                _input,
                covariant_transform,
                contravariant_transform,
                reverse=False,
            )

            nptest.assert_allclose(expected_value, actual_value)

        # Tangent basis Cylindrical -> Cartesian.
        outputs = (
            np.array([c, s, 0.0]),
            np.array([-s / r, c / r, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        for _input, expected_value in zip(inputs, outputs, strict=True):
            actual_value = transform_tensor_field(
                TensorType.COVECTOR,
                _input,
                covariant_transform,
                contravariant_transform,
                reverse=True,
            )

            nptest.assert_allclose(expected_value, actual_value)

        # Co-tangent basis Cartesian -> cylindrical.
        outputs = (
            np.array([c, -s / r, 0.0]),
            np.array([s, c / r, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        for _input, expected_value in zip(inputs, outputs, strict=False):
            actual_value = transform_tensor_field(
                TensorType.VECTOR,
                _input,
                covariant_transform,
                contravariant_transform,
                reverse=False,
            )

            nptest.assert_allclose(expected_value, actual_value)

        # Co-tangent basis Cylindrical -> Cartesian.
        outputs = (
            np.array([c, s, 0.0]),
            np.array([-r * s, r * c, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        for _input, expected_value in zip(inputs, outputs, strict=False):
            actual_value = transform_tensor_field(
                TensorType.VECTOR,
                _input,
                covariant_transform,
                contravariant_transform,
                reverse=True,
            )

            nptest.assert_allclose(expected_value, actual_value)

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_transform_tensor(
        self, x_cartesian: np.ndarray[float], x_cylindrical: np.ndarray[float]
    ):
        """
        Test transforming tensor fields between Cartesian and cylindrical.

        Parameters
        ----------
        x_cartesian : np.ndarray[float]
            Test position Cartesian components.
        x_cylindrical : np.ndarray[float]
            Test position cylindrical components.
        """
        cylindrical = Cylindrical()
        contravariant_transform = cylindrical.contravariant_transform(
            cylindrical.forward_transform_dx(
                x_cylindrical, CoordinateSystem.CYLINDRICAL
            )
        )
        covariant_transform = cylindrical.covariant_transform(
            cylindrical.backward_transform_dx(
                x_cylindrical, CoordinateSystem.CYLINDRICAL
            )
        )

        # Rank 0 transform.
        tensor_field = np.array(1.0)
        actual_value = transform_tensor_field(
            TensorType.SCALAR,
            tensor_field,
            covariant_transform,
            contravariant_transform,
            reverse=False,
        )

        nptest.assert_allclose(tensor_field, actual_value)

        # Rank 1 transform.
        tensor_cartesian = self.vector_cartesian(x_cartesian)
        tensor_cylindrical = self.vector_cylindrical(x_cylindrical)

        tensor_cylindrical_2 = transform_tensor_field(
            TensorType.VECTOR,
            tensor_cartesian,
            covariant_transform,
            contravariant_transform,
            reverse=False,
        )

        nptest.assert_allclose(tensor_cylindrical, tensor_cylindrical_2)

        tensor_cartesian_2 = transform_tensor_field(
            TensorType.VECTOR,
            tensor_cylindrical,
            covariant_transform,
            contravariant_transform,
            reverse=True,
        )

        nptest.assert_allclose(tensor_cartesian, tensor_cartesian_2, atol=5e-8)

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_covariant_derivative(
        self, x_cartesian: np.ndarray[float], x_cylindrical: np.ndarray[float]
    ):
        """
        Test covariant derivatives of tensor fields.

        Parameters
        ----------
        x_cartesian : np.ndarray[float]
            Test position Cartesian components.
        x_cylindrical : np.ndarray[float]
            Test position cylindrical components.
        """
        cylindrical = Cylindrical()
        _forward_transform_dx = cylindrical.forward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        _backward_transform_dx = cylindrical.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        _contravariant_transform = cylindrical.contravariant_transform(
            _forward_transform_dx
        )
        _covariant_transform = cylindrical.covariant_transform(
            _backward_transform_dx
        )
        _forward_transform_dx2 = cylindrical.forward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        _forward_transform_dx3 = cylindrical.forward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        _connection_coefficients = connection_coefficients(
            _backward_transform_dx, _forward_transform_dx2
        )
        _connection_coefficients_dx = connection_coefficients_dx(
            _connection_coefficients,
            _backward_transform_dx,
            _forward_transform_dx3,
        )

        # First derivative of scalar field.
        expected_value = self.scalar_cartesian_dx(x_cartesian)

        _scalar_cylindrical = np.array(self.scalar_cylindrical(x_cylindrical))
        _scalar_cylindrical_dx = self.scalar_cylindrical_dx(x_cylindrical)

        actual_value_cylindrical = first_covariant_derivative(
            TensorType.SCALAR,
            _scalar_cylindrical,
            _scalar_cylindrical_dx,
            _connection_coefficients,
        )

        actual_value = transform_tensor_field(
            TensorType.SCALAR_FIRST_DERIVATIVE,
            actual_value_cylindrical,
            _covariant_transform,
            _contravariant_transform,
            reverse=True,
        )

        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

        # Second derivative of scalar field.
        expected_value = self.scalar_cartesian_dx2()

        _scalar_cylindrical_dx2 = self.scalar_cylindrical_dx2(x_cylindrical)

        actual_value_cylindrical = second_covariant_derivative(
            TensorType.SCALAR,
            _scalar_cylindrical,
            _scalar_cylindrical_dx,
            _scalar_cylindrical_dx2,
            actual_value_cylindrical,
            _connection_coefficients,
            _connection_coefficients_dx,
        )

        actual_value = transform_tensor_field(
            TensorType.SCALAR_SECOND_DERIVATIVE,
            actual_value_cylindrical,
            _covariant_transform,
            _contravariant_transform,
            reverse=True,
        )

        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

        # First derivative of vector field.
        expected_value = self.vector_cartesian_dx(x_cartesian)

        _vector_cylindrical = self.vector_cylindrical(x_cylindrical)
        _vector_cylindrical_dx = self.vector_cylindrical_dx(x_cylindrical)

        actual_value_cylindrical = first_covariant_derivative(
            TensorType.VECTOR,
            _vector_cylindrical,
            _vector_cylindrical_dx,
            _connection_coefficients,
        )

        actual_value = transform_tensor_field(
            TensorType.VECTOR_FIRST_DERIVATIVE,
            actual_value_cylindrical,
            _covariant_transform,
            _contravariant_transform,
            reverse=True,
        )

        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)

        # Second derivative of vector field.
        expected_value = self.vector_cartesian_dx2(x_cartesian)

        _vector_cylindrical_dx2 = self.vector_cylindrical_dx2(x_cylindrical)

        actual_value_cylindrical = second_covariant_derivative(
            TensorType.VECTOR,
            _vector_cylindrical,
            _vector_cylindrical_dx,
            _vector_cylindrical_dx2,
            actual_value_cylindrical,
            _connection_coefficients,
            _connection_coefficients_dx,
        )

        actual_value = transform_tensor_field(
            TensorType.VECTOR_SECOND_DERIVATIVE,
            actual_value_cylindrical,
            _covariant_transform,
            _contravariant_transform,
            reverse=True,
        )

        nptest.assert_allclose(actual_value, expected_value, atol=1e-8)
