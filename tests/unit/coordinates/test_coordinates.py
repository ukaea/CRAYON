"""
Unit tests for coordinates.coordinates.
"""

# Standard imports
import logging
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import (
    Cartesian,
    CoordinateSystem,
    Cylindrical,
    Spherical,
    Toroidal,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)


class TestCartesian:
    """
    Test Cartesian coordinate system.
    """

    @staticmethod
    @pytest.fixture
    def coordinate() -> Cartesian:
        """
        Cartesian coordinate system.

        Returns
        -------
        coordinate : Cartesian
            Coordinate system.
        """
        return Cartesian()

    positions_cartesian = (
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([3.0, -4.0, 0.0]),
    )

    @staticmethod
    @pytest.mark.parametrize("position", positions_cartesian)
    def test_bound_components(
        coordinate: Cartesian, position: np.ndarray[float]
    ):
        """
        Test bound components.

        Parameters
        ----------
        coordinate : Cartesian
            Test coordinate.
        position : np.ndarray[float]
            Test position.
        """
        expected_value = position
        actual_value = coordinate.bound_components(position)

        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def test_round_trip_netcdf(coordinate: Cartesian):
        """
        Test writing and reading from netCDF4 file gives same object.

        Parameters
        ----------
        coordinate : Cartesian
            Test coordinate.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            coordinate.write_netcdf(dset)
            coordinate_2 = Cartesian.read_netcdf()

        assert coordinate.coordinate_system == coordinate_2.coordinate_system


class TestCylindrical:
    """
    Test cylindrical coordinate system.
    """

    @staticmethod
    @pytest.fixture
    def coordinate() -> Cylindrical:
        """
        Cylindrical coordinate system.

        Returns
        -------
        coordinate : Cylindrical
            Coordinate system.
        """
        return Cylindrical()

    @staticmethod
    @pytest.mark.parametrize(
        ("position", "expected_value"),
        [
            (np.array([1.0, np.pi / 2, 2.0]), np.array([1.0, np.pi / 2, 2.0])),
            (
                np.array([-1.0, -np.pi / 2, -2.0]),
                np.array([1.0, np.pi / 2, -2.0]),
            ),
            (
                np.array([1.0, 1.5 * np.pi, 0.0]),
                np.array([1.0, -np.pi / 2, 0.0]),
            ),
            (
                np.array([1.0, -1.5 * np.pi, 0.0]),
                np.array([1.0, np.pi / 2, 0.0]),
            ),
        ],
    )
    def test_bound_components(
        coordinate: Cylindrical,
        position: np.ndarray[float],
        expected_value: np.ndarray[float],
    ):
        """
        Test formula to bound coordinate components i.e. r mapped positive,
        theta mapped into [-pi, pi] and z not changed.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        position : np.ndarray[float]
            Test position.
        expected_value : np.ndarray[float]
            Expected value.
        """
        actual_value = coordinate.bound_components(position)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def test_round_trip_netcdf(coordinate: Cylindrical):
        """
        Test writing and reading from netCDF4 file gives same object.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            coordinate.write_netcdf(dset)
            coordinate_2 = Cylindrical.read_netcdf()

        assert coordinate.coordinate_system == coordinate_2.coordinate_system

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

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_transform(
        coordinate: Cylindrical,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
    ):
        """
        Test forward and backward transform.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        x_cartesian : np.ndarray[float]
            Test Cartesian position.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        """
        x_cartesian_2 = coordinate.backward_transform(x_cylindrical)
        x_cylindrical_2 = coordinate.forward_transform(x_cartesian)

        nptest.assert_allclose(x_cartesian, x_cartesian_2)
        nptest.assert_allclose(x_cylindrical, x_cylindrical_2)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_forward_transform_derivatives(
        coordinate: Cylindrical,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
    ):
        """
        Test derivatives of forward transform.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        x_cartesian : np.ndarray[float]
            Test Cartesian position.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        """
        # First derivative.
        expected_value = coordinate.forward_transform_dx(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.forward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = first_derivative_finite_difference(
            x_cartesian, coordinate.forward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value)

        # Second derivative.
        expected_value = coordinate.forward_transform_dx2(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.forward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = second_derivative_finite_difference(
            x_cartesian, coordinate.forward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)

        # Third derivative.
        expected_value = coordinate.forward_transform_dx3(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.forward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = second_derivative_finite_difference(
            x_cartesian,
            lambda x: coordinate.forward_transform_dx(
                x, CoordinateSystem.CARTESIAN
            ),
            (Dimensions.x.size, Dimensions.x.size),
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_backward_transform_derivatives(
        coordinate: Cylindrical,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
    ):
        """
        Test derivatives of backward transform.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        x_cartesian : np.ndarray[float]
            Test Cartesian position.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        """
        # First derivative.
        expected_value = coordinate.backward_transform_dx(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = first_derivative_finite_difference(
            x_cylindrical, coordinate.backward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value)

        # Second derivative.
        expected_value = coordinate.backward_transform_dx2(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.backward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = second_derivative_finite_difference(
            x_cylindrical, coordinate.backward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)

        # Third derivative.
        expected_value = coordinate.backward_transform_dx3(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.backward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )

        actual_value = second_derivative_finite_difference(
            x_cylindrical,
            lambda x: coordinate.backward_transform_dx(
                x, CoordinateSystem.CYLINDRICAL
            ),
            (Dimensions.x.size, Dimensions.x.size),
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)


class TestSpherical:
    """
    Test spherical coordinate system.
    """

    @staticmethod
    @pytest.fixture
    def coordinate() -> Spherical:
        """
        Spherical coordinate system.

        Returns
        -------
        coordinate : Spherical
            Coordinate system.
        """
        return Spherical()

    @staticmethod
    @pytest.mark.parametrize(
        ("position", "expected_value"),
        [
            (np.array([1.0, np.pi / 2, 0.0]), np.array([1.0, np.pi / 2, 0.0])),
            (
                np.array([-1.0, np.pi / 2, 0.0]),
                np.array([1.0, np.pi / 2, np.pi]),
            ),
            (
                np.array([1.0, 1.5 * np.pi, 0.0]),
                np.array([1.0, np.pi / 2, np.pi]),
            ),
            (
                np.array([1.0, -1.5 * np.pi, 0.0]),
                np.array([1.0, np.pi / 2, 0.0]),
            ),
        ],
    )
    def test_bound_components(
        coordinate: Spherical,
        position: np.ndarray[float],
        expected_value: np.ndarray[float],
    ):
        """
        Test formula to bound coordinate components i.e. r mapped positive,
        theta mapped into [-pi, pi] and z not changed.

        Parameters
        ----------
        coordinate : Spherical
            Test coordinate.
        position : np.ndarray[float]
            Test position.
        expected_value : np.ndarray[float]
            Expected value.
        """
        actual_value = coordinate.bound_components(position)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def test_round_trip_netcdf(coordinate: Spherical):
        """
        Test writing and reading from netCDF4 file gives same object.

        Parameters
        ----------
        coordinate : Spherical
            Test coordinate.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            coordinate.write_netcdf(dset)
            coordinate_2 = Spherical.read_netcdf()

        assert coordinate.coordinate_system == coordinate_2.coordinate_system

    positions_cartesian = (
        np.array([0.05394023, 0.08400692, 0.99500417]),
        np.array([0.56535421, -0.30885441, 0.76484219]),
        np.array([-0.63717123, 1.86506233, 0.33993429]),
    )

    positions_spherical = (
        np.array([1.0, 0.1, 1.0]),
        np.array([1.0, 0.7, -0.5]),
        np.array([2.0, 1.4, 1.9]),
    )

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_spherical"),
        zip(positions_cartesian, positions_spherical, strict=False),
    )
    def test_transform(
        coordinate: Cylindrical,
        x_cartesian: np.ndarray[float],
        x_spherical: np.ndarray[float],
    ):
        """
        Test forward and backward transform.

        Parameters
        ----------
        coordinate : Spherical
            Test coordinate.
        x_cartesian : np.ndarray[float]
            Test Cartesian position.
        x_spherical : np.ndarray[float]
            Test spherical position.
        """
        x_cartesian_2 = coordinate.backward_transform(x_spherical)
        x_spherical_2 = coordinate.forward_transform(x_cartesian)

        nptest.assert_allclose(x_cartesian, x_cartesian_2)
        nptest.assert_allclose(x_spherical, x_spherical_2)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_spherical"),
        zip(positions_cartesian, positions_spherical, strict=False),
    )
    def test_forward_transform_derivatives(
        coordinate: Cylindrical,
        x_cartesian: np.ndarray[float],
        x_spherical: np.ndarray[float],
    ):
        """
        Test derivatives of forward transform.

        Parameters
        ----------
        coordinate : Spherical
            Test coordinate.
        x_cartesian : np.ndarray[float]
            Test Cartesian position.
        x_spherical : np.ndarray[float]
            Test spherical position.
        """
        # First derivative.
        expected_value = coordinate.forward_transform_dx(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.forward_transform_dx(
            x_spherical, CoordinateSystem.SPHERICAL
        )

        actual_value = first_derivative_finite_difference(
            x_cartesian, coordinate.forward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cartesian", "x_spherical"),
        zip(positions_cartesian, positions_spherical, strict=False),
    )
    def test_backward_transform_derivatives(
        coordinate: Cylindrical,
        x_cartesian: np.ndarray[float],
        x_spherical: np.ndarray[float],
    ):
        """
        Test derivatives of backward transform.

        Parameters
        ----------
        coordinate : Spherical
            Test coordinate.
        x_cartesian : np.ndarray[float]
            Test Cartesian position.
        x_spherical : np.ndarray[float]
            Test spherical position.
        """
        # First derivative.
        expected_value = coordinate.backward_transform_dx(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.backward_transform_dx(
            x_spherical, CoordinateSystem.SPHERICAL
        )

        actual_value = first_derivative_finite_difference(
            x_spherical, coordinate.backward_transform, (Dimensions.x.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        logger.warning(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value)

        # Second derivative.
        expected_value = coordinate.backward_transform_dx2(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.backward_transform_dx2(
            x_spherical, CoordinateSystem.SPHERICAL
        )

        actual_value = second_derivative_finite_difference(
            x_spherical, coordinate.backward_transform, (Dimensions.x.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.warning(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)

        # Third derivative.
        expected_value = coordinate.backward_transform_dx3(
            x_cartesian, CoordinateSystem.CARTESIAN
        )
        expected_value_2 = coordinate.backward_transform_dx3(
            x_spherical, CoordinateSystem.SPHERICAL
        )

        actual_value = second_derivative_finite_difference(
            x_spherical,
            lambda x: coordinate.backward_transform_dx(
                x, CoordinateSystem.SPHERICAL
            ),
            (Dimensions.x.size, Dimensions.x.size),
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.warning(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)


class TestToroidal:
    """
    Test toroidal coordinate system.
    """

    @staticmethod
    @pytest.fixture
    def coordinate() -> Toroidal:
        """
        Toroidal coordinate system.

        Returns
        -------
        coordinate : Toroidal
            Coordinate system.
        """
        return Toroidal((0.8, 0.0))

    @staticmethod
    @pytest.mark.parametrize(
        ("position", "expected_value"),
        [
            (
                np.array([1.0, np.pi / 2, np.pi / 2]),
                np.array([1.0, np.pi / 2, np.pi / 2]),
            ),
            (
                np.array([-1.0, -np.pi / 2, -np.pi / 2]),
                np.array([1.0, -np.pi / 2, -np.pi / 2]),
            ),
            (
                np.array([1.0, 1.5 * np.pi, 1.5 * np.pi]),
                np.array([1.0, -np.pi / 2, -np.pi / 2]),
            ),
            (
                np.array([1.0, -1.5 * np.pi, -1.5 * np.pi]),
                np.array([1.0, np.pi / 2, np.pi / 2]),
            ),
        ],
    )
    def test_bound_components(
        coordinate: Toroidal,
        position: np.ndarray[float],
        expected_value: np.ndarray[float],
    ):
        """
        Test formula to bound coordinate components i.e. r mapped positive,
        theta mapped into [-pi, pi] and z not changed.

        Parameters
        ----------
        coordinate : Toroidal
            Test coordinate.
        position : np.ndarray[float]
            Test position.
        expected_value : np.ndarray[float]
            Expected value.
        """
        actual_value = coordinate.bound_components(position)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def test_round_trip_netcdf(coordinate: Toroidal):
        """
        Test writing and reading from netCDF4 file gives same object.

        Parameters
        ----------
        coordinate : Toroidal
            Test coordinate.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            coordinate.write_netcdf(dset)
            coordinate_2 = Toroidal.read_netcdf(
                dset[CoordinateSystem.TOROIDAL.name]
            )

        assert coordinate.coordinate_system == coordinate_2.coordinate_system
        nptest.assert_allclose(coordinate.axis_m, coordinate_2.axis_m)

    positions_cartesian = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.5, -0.3]),
    )

    positions_cylindrical = (
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.5 * np.pi, 0.0]),
        np.array([2.0, 0.0, 1.0]),
        np.array([1.41421356, 0.25 * np.pi, 0.0]),
        np.array([1.41421356, 0.25 * np.pi, 1.0]),
        np.array([1.11803399, 0.46364761, -0.3]),
    )

    positions_toroidal = (
        np.array([0.2, 0.0, 0.0]),
        np.array([0.2, np.pi / 2, 0.0]),
        np.array([1.5620499, 0.0, 0.69473828]),
        np.array([0.61421356, 0.25 * np.pi, 0.0]),
        np.array([1.17356649, 0.25 * np.pi, 1.01999118]),
        np.array([0.43720203, 0.46364761, -0.75622683]),
    )

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cylindrical", "x_toroidal"),
        zip(positions_cylindrical, positions_toroidal, strict=True),
    )
    def test_transform(
        coordinate: Toroidal,
        x_cylindrical: np.ndarray[float],
        x_toroidal: np.ndarray[float],
    ):
        """
        Test forward and backward transform.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        x_toroidal : np.ndarray[float]
            Test toroidal position.
        """
        x_cylindrical_2 = coordinate.backward_transform(x_toroidal)
        x_toroidal_2 = coordinate.forward_transform(x_cylindrical)

        nptest.assert_allclose(x_cylindrical, x_cylindrical_2, atol=1e-8)
        nptest.assert_allclose(x_toroidal, x_toroidal_2, atol=1e-8)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cylindrical", "x_toroidal"),
        zip(positions_cylindrical, positions_toroidal, strict=False),
    )
    def test_backward_transform_derivatives(
        coordinate: Toroidal,
        x_cylindrical: np.ndarray[float],
        x_toroidal: np.ndarray[float],
    ):
        """
        Test backward transform derivatives.

        Parameters
        ----------
        coordinate : Cylindrical
            Test coordinate.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        x_toroidal : np.ndarray[float]
            Test toroidal position.
        """
        # First derivative.
        expected_value = coordinate.backward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        expected_value_2 = coordinate.backward_transform_dx(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        actual_value = first_derivative_finite_difference(
            x_toroidal, coordinate.backward_transform, (Dimensions.x.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        logger.warning(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-8)

        # Second derivative.
        expected_value = coordinate.backward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        expected_value_2 = coordinate.backward_transform_dx2(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        actual_value = second_derivative_finite_difference(
            x_toroidal, coordinate.backward_transform, (Dimensions.x.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.warning(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)

        # Third derivative.
        expected_value = coordinate.backward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        expected_value_2 = coordinate.backward_transform_dx3(
            x_toroidal, CoordinateSystem.TOROIDAL
        )

        actual_value = second_derivative_finite_difference(
            x_toroidal,
            lambda x: coordinate.backward_transform_dx(
                x, CoordinateSystem.TOROIDAL
            ),
            (Dimensions.x.size, Dimensions.x.size),
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        logger.warning(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-7)
