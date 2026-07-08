"""
Unit tests for ray_tracing.caches.coordinates.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.calculus import TensorType
from crayon.coordinates import (
    CoordinateCoordinator,
    CoordinateSystem,
    Toroidal,
)
from crayon.ray_tracing.caches.coordinates import (
    CoordinateCache,
    ForwardTransformCache,
)
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)

r0, z0 = 0.8, 0.0


@pytest.fixture(scope="module")
def coordinate_coordinator() -> CoordinateCoordinator:
    """
    Coordinate coordinator holding coordinate system information.

    Returns
    -------
    coordinate_coordinator : CoordinateCoordinator
        Coordinate coordinator.
    """
    cc = CoordinateCoordinator()
    cc.register_coordinate(Toroidal((r0, z0)))
    cc.calculate_conversion_paths()

    return cc


positions_cartesian = (
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.3]),
    np.array([0.4, 0.4, -0.4]),
)

positions_cylindrical = (
    np.array([1.0, 0.0, 0.0]),
    np.array([1.41421356, 0.78539816, 0.3]),
    np.array([0.56568542, 0.78539816, -0.4]),
)

positions_toroidal = (
    np.array([0.2, 0.0, 0.0]),
    np.array([0.68356295, 0.78539816, 0.45434841]),
    np.array([0.46357666, 0.78539816, -2.10069912]),
)


class TestForwardTransformCache:
    """
    Unit tests for ForwardTransformCache.
    """

    @staticmethod
    @pytest.fixture
    def cache() -> ForwardTransformCache:
        """
        Forward transform cache.

        Returns
        -------
        cache : ForwardTransformCache
            Cache.
        """
        return ForwardTransformCache()

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical", "x_toroidal"),
        zip(
            positions_cartesian,
            positions_cylindrical,
            positions_toroidal,
            strict=True,
        ),
    )
    def test_calculate_transform_derivatives(
        cache: ForwardTransformCache,
        coordinate_coordinator: CoordinateCoordinator,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
        x_toroidal: np.ndarray[float],
    ):
        """
        Test derivatives of transforms.

        Parameters
        ----------
        cache: ForwardTransformCache
            Forward transform cache.
        coordinate_coordinator: CoordinateCoordinator
            Coordinate coordinator.
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        x_toroidal: np.ndarray[float]
            Test position in toroidal.
        """
        # Test cylindrical.
        _coordinate = coordinate_coordinator.coordinates[
            CoordinateSystem.CYLINDRICAL
        ]

        cache.calculate_transform_derivatives(
            _coordinate,
            x_cylindrical,
            CoordinateSystem.CYLINDRICAL,
            x_cartesian,
            CoordinateSystem.CARTESIAN,
        )

        nptest.assert_allclose(
            cache.forward_transform_dx,
            _coordinate.forward_transform_dx(
                x_cylindrical, CoordinateSystem.CYLINDRICAL
            ),
        )

        nptest.assert_allclose(
            cache.backward_transform_dx,
            _coordinate.backward_transform_dx(
                x_cylindrical, CoordinateSystem.CYLINDRICAL
            ),
        )

        # Test toroidal.
        _coordinate = coordinate_coordinator.coordinates[
            CoordinateSystem.TOROIDAL
        ]

        cache.calculate_transform_derivatives(
            _coordinate,
            x_toroidal,
            CoordinateSystem.TOROIDAL,
            x_cylindrical,
            CoordinateSystem.CYLINDRICAL,
        )

        nptest.assert_allclose(
            cache.forward_transform_dx,
            _coordinate.forward_transform_dx(
                x_toroidal, CoordinateSystem.TOROIDAL
            ),
        )

        nptest.assert_allclose(
            cache.backward_transform_dx,
            _coordinate.backward_transform_dx(
                x_toroidal, CoordinateSystem.TOROIDAL
            ),
        )


class TestCoordinateCache:
    """
    Unit tests for CoordinateCache.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def cache(
        coordinate_coordinator: CoordinateCoordinator,
    ) -> CoordinateCache:
        """
        Cache holding coordinate system data.

        Returns
        -------
        cache : CoordinateCache
            Coordinate cache.
        """
        return CoordinateCache(coordinate_coordinator)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical", "x_toroidal"),
        zip(
            positions_cartesian,
            positions_cylindrical,
            positions_toroidal,
            strict=True,
        ),
    )
    def test_positions(
        cache: CoordinateCache,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
        x_toroidal: np.ndarray[float],
    ):
        """
        Test set position.

        Parameters
        ----------
        cache: CoordinateCache
            Coordinate cache.
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        x_toroidal: np.ndarray[float]
            Test position in toroidal.
        """
        cache.set_position(CoordinateSystem.CARTESIAN, x_cartesian)

        nptest.assert_allclose(
            cache.position[CoordinateSystem.CARTESIAN], x_cartesian
        )

        nptest.assert_allclose(
            cache.position[CoordinateSystem.CYLINDRICAL], x_cylindrical
        )

        nptest.assert_allclose(
            cache.position[CoordinateSystem.TOROIDAL], x_toroidal
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical", "x_toroidal"),
        zip(
            positions_cartesian,
            positions_cylindrical,
            positions_toroidal,
            strict=True,
        ),
    )
    def test_metric(
        cache: CoordinateCache,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
        x_toroidal: np.ndarray[float],
    ):
        """
        Test calculation of metric tensor.

        Parameters
        ----------
        cache: CoordinateCache
            Coordinate cache.
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        x_toroidal: np.ndarray[float]
            Test position in toroidal.
        """
        cache.set_position(CoordinateSystem.CARTESIAN, x_cartesian)

        # Test cylindrical.
        expected_value = np.identity(3)
        expected_value[1, 1] = x_cylindrical[0] ** 2
        actual_value = cache.transforms[CoordinateSystem.CYLINDRICAL].metric

        nptest.assert_allclose(expected_value, actual_value)
        nptest.assert_allclose(
            np.matmul(
                cache.transforms[CoordinateSystem.CYLINDRICAL].metric,
                cache.transforms[CoordinateSystem.CYLINDRICAL].inverse_metric,
            ),
            np.identity(3),
            atol=1e-8,
        )

        # Test toroidal.
        expected_value = np.identity(3)
        expected_value[1, 1] = x_cylindrical[0] ** 2
        expected_value[2, 2] = x_toroidal[0] ** 2
        actual_value = cache.transforms[CoordinateSystem.TOROIDAL].metric

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)
        nptest.assert_allclose(
            np.matmul(
                cache.transforms[CoordinateSystem.TOROIDAL].metric,
                cache.transforms[CoordinateSystem.TOROIDAL].inverse_metric,
            ),
            np.identity(3),
            atol=1e-8,
        )

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

    @staticmethod
    def gamma_toroidal(x_toroidal: np.ndarray[float]) -> np.ndarray[float]:
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
        a = r0 + r * cos_theta

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

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical", "x_toroidal"),
        zip(
            positions_cartesian,
            positions_cylindrical,
            positions_toroidal,
            strict=True,
        ),
    )
    def test_connection_coefficients(
        self,
        cache: CoordinateCache,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
        x_toroidal: np.ndarray[float],
    ):
        """
        Test calculation of connection coefficients.

        Parameters
        ----------
        cache: CoordinateCache
            Coordinate cache.
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        x_toroidal: np.ndarray[float]
            Test position in toroidal.
        """
        cache.set_position(CoordinateSystem.CARTESIAN, x_cartesian)

        # Test cylindrical.
        expected_value = self.gamma_cylindrical(x_cylindrical)
        actual_value = cache.transforms[
            CoordinateSystem.CYLINDRICAL
        ].connection_coefficients

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = first_derivative_finite_difference(
            x_cylindrical, self.gamma_cylindrical, (3, 3, 3)
        )
        actual_value = cache.transforms[
            CoordinateSystem.CYLINDRICAL
        ].connection_coefficients_dx

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test toroidal.
        expected_value = self.gamma_toroidal(x_toroidal)
        actual_value = cache.transforms[
            CoordinateSystem.TOROIDAL
        ].connection_coefficients

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = first_derivative_finite_difference(
            x_toroidal, self.gamma_toroidal, (3, 3, 3)
        )
        actual_value = cache.transforms[
            CoordinateSystem.TOROIDAL
        ].connection_coefficients_dx

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    @staticmethod
    def test_transform_basis(cache: CoordinateCache):
        """
        Test transforms between basis.

        Parameters
        ----------
        cache : CoordinateCache
            Coordinate cache.
        """
        position_cylindrical = np.array([1.2, 0.3, 0.4])
        vector_holonomic = np.array([1.0, 1.0, 1.0])
        vector_physical = np.array([1.0, 1.2, 1.0])

        # Test transform to holonomic.
        cache.set_position(CoordinateSystem.CYLINDRICAL, position_cylindrical)

        expected_value = vector_holonomic
        actual_value = cache.transform_basis(
            CoordinateSystem.CYLINDRICAL,
            vector_physical,
            TensorType.VECTOR,
            to_holonomic=True,
        )

        nptest.assert_allclose(expected_value, actual_value)

        # Test transform to physical.
        expected_value = vector_physical
        actual_value = cache.transform_basis(
            CoordinateSystem.CYLINDRICAL,
            vector_holonomic,
            TensorType.VECTOR,
            to_holonomic=False,
        )

        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_transform_tensor_field(
        cache: CoordinateCache,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
    ):
        """
        Test vector and covector field transform correctly.

        Parameters
        ----------
        cache: CoordinateCache
            Coordinate cache.
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        """
        cache.set_position(CoordinateSystem.CARTESIAN, x_cartesian)

        vector_cartesian = np.array([1.0, 1.0, 1.0])
        covector_cartesian = vector_cartesian

        r, phi, _ = x_cylindrical
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        vx, vy, vz = vector_cartesian

        # Test vector transform.
        vector_cylindrical = np.array([
            cos_phi * vx + sin_phi * vy,
            (-sin_phi * vx + cos_phi * vy) / r,
            vz,
        ])

        vector_cartesian_2 = cache.transform_tensor_field(
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.CARTESIAN,
            vector_cylindrical,
            TensorType.VECTOR,
        )

        nptest.assert_allclose(vector_cartesian, vector_cartesian_2)

        covector_cylindrical = np.array([
            cos_phi * vx + sin_phi * vy,
            (-sin_phi * vx + cos_phi * vy) * r,
            vz,
        ])

        covector_cartesian_2 = cache.transform_tensor_field(
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.CARTESIAN,
            covector_cylindrical,
            TensorType.COVECTOR,
        )

        nptest.assert_allclose(covector_cartesian, covector_cartesian_2)

    @staticmethod
    def vector_cartesian(
        position_cartesian: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Cartesian vector field.

        Parameters
        ----------
        position_cartesian: np.ndarray[float]
            Test position in Cartesian.

        Returns
        -------
        vector_cartesian : np.ndarray[float]
            Cartesian vector field.
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
        First derivative of Cartesian vector field.

        Parameters
        ----------
        position_cartesian: np.ndarray[float]
            Test position in Cartesian.

        Returns
        -------
        vector_cartesian_dx : np.ndarray[float]
            First derivative of Cartesian vector field.
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
        Second derivative of Cartesian vector field.

        Parameters
        ----------
        position_cartesian: np.ndarray[float]
            Test position in Cartesian.

        Returns
        -------
        vector_cartesian_dx2 : np.ndarray[float]
            Second derivative of Cartesian vector field.
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
        Cylindrical vector field.

        Parameters
        ----------
        position_cylindrical: np.ndarray[float]
            Test position in cylindrical.

        Returns
        -------
        vector_cylindrical : np.ndarray[float]
            Cylindrical vector field.
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
        First derivative of cylindrical vector field.

        Parameters
        ----------
        position_cylindrical: np.ndarray[float]
            Test position in cylindrical.

        Returns
        -------
        vector_cylindrical_dx : np.ndarray[float]
            First derivative of cylindrical vector field.
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
        Second derivative of cylindrical vector field.

        Parameters
        ----------
        position_cylindrical: np.ndarray[float]
            Test position in cylindrical.

        Returns
        -------
        vector_cylindrical_dx2 : np.ndarray[float]
            Second derivative of cylindrical vector field.
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

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_vector(
        self, x_cartesian: np.ndarray[float], x_cylindrical: np.ndarray[float]
    ):
        """
        Test vector field derivatives.

        Parameters
        ----------
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        """
        # Test Cartesian first derivative.
        expected_value = self.vector_cartesian_dx(x_cartesian)
        actual_value = first_derivative_finite_difference(
            x_cartesian, self.vector_cartesian, (3,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test Cartesian second derivative.
        expected_value = self.vector_cartesian_dx2(x_cartesian)
        actual_value = second_derivative_finite_difference(
            x_cartesian, self.vector_cartesian, (3,), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        # Test cylindrical first derivative.
        expected_value = self.vector_cylindrical_dx(x_cylindrical)
        actual_value = first_derivative_finite_difference(
            x_cylindrical, self.vector_cylindrical, (3,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test cylindrical second derivative.
        expected_value = self.vector_cylindrical_dx2(x_cylindrical)
        actual_value = second_derivative_finite_difference(
            x_cylindrical, self.vector_cylindrical, (3,), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_covariant_derivative(
        self,
        cache: CoordinateCache,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
    ):
        """
        Test calculation of covariant derivatives.

        Parameters
        ----------
        cache : CoordinateCache
            Coordinate cache.
        x_cartesian: np.ndarray[float]
            Test position in Cartesian.
        x_cylindrical: np.ndarray[float]
            Test position in cylindrical.
        """
        cache.set_position(CoordinateSystem.CARTESIAN, x_cartesian)

        _vector_cylindrical = self.vector_cylindrical(x_cylindrical)
        _vector_cylindrical_dx = self.vector_cylindrical_dx(x_cylindrical)
        _vector_cylindrical_dx2 = self.vector_cylindrical_dx2(x_cylindrical)

        # Test first derivative.
        expected_value = self.vector_cartesian_dx(x_cartesian)
        vector_cylindrical_first_derivative = cache.first_covariant_derivative(
            CoordinateSystem.CYLINDRICAL,
            _vector_cylindrical,
            _vector_cylindrical_dx,
            TensorType.VECTOR,
        )

        actual_value = cache.transform_tensor_field(
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.CARTESIAN,
            vector_cylindrical_first_derivative,
            TensorType.VECTOR_FIRST_DERIVATIVE,
        )

        nptest.assert_allclose(expected_value, actual_value, atol=5e-8)

        # Test second derivative.
        expected_value = self.vector_cartesian_dx2(x_cartesian)
        vector_cylindrical_second_derivative = (
            cache.second_covariant_derivative(
                CoordinateSystem.CYLINDRICAL,
                _vector_cylindrical,
                _vector_cylindrical_dx,
                _vector_cylindrical_dx2,
                vector_cylindrical_first_derivative,
                TensorType.VECTOR,
            )
        )

        actual_value = cache.transform_tensor_field(
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.CARTESIAN,
            vector_cylindrical_second_derivative,
            TensorType.VECTOR_SECOND_DERIVATIVE,
        )

        nptest.assert_allclose(expected_value, actual_value, atol=5e-8)
