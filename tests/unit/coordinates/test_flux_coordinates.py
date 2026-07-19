"""
Unit tests for coordinates.coordinates.flux_coordinates.
"""

# Standard imports
import logging
import pathlib
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.coordinates.coordinates.flux_coordinate import (
    AxisymmetricFluxCoordinate,
    AxisymmetricFluxCoordinateRebase,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.value_model.models import ValueModel

logger = logging.getLogger(__name__)


def test_contour_winds():
    """
    Test contour_winds correctly identifies contours which wind about an axis.
    """
    theta = np.linspace(0, 2 * np.pi, 51)
    x, y = np.cos(theta), np.sin(theta)

    assert AxisymmetricFluxCoordinate._contour_winds(x, y, (0.0, 0.0))
    assert not AxisymmetricFluxCoordinate._contour_winds(x, y, (2.0, 0.0))


@pytest.mark.parametrize("r", [0.03, 0.5, 0.8, 12.0])
def test_interpolate_closed_contour(r):
    """
    Test interpolate closed contour interpolates as expected using a circle.

    Parameters
    ----------
    r : float
        Radius of circle
    """
    theta = np.linspace(-np.pi, np.pi, 51)
    theta_target = np.linspace(-np.pi, np.pi, 123)

    x, y = r * np.cos(theta), r * np.sin(theta)

    x_expected = r * np.cos(theta_target)
    y_expected = r * np.sin(theta_target)

    x_actual = np.empty_like(theta_target)
    y_actual = np.empty_like(theta_target)

    AxisymmetricFluxCoordinate._interpolate_closed_contour(
        theta, x, y, theta_target, x_actual, y_actual
    )

    nptest.assert_allclose(x_actual, x_expected, atol=1e-5)
    nptest.assert_allclose(y_actual, y_expected, atol=1e-5)


class TestFindContours:
    """
    Unit tests for finding contours.
    """

    @staticmethod
    def well(
        r_maj: float, r_min: float
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        A quadratic well function to find contours for.

        Parameters
        ----------
        r_maj : float
            Radius of bottom of well.
        r_min : float
            Radius of sepratrix contour.

        Returns
        -------
        x, y : np.ndarray[float]
            x and y 1d arrays.
        z : float
            2d value function.
        """
        x0, x1 = r_maj - 1.1 * r_min, r_maj + 1.1 * r_min
        y0, y1 = -1.1 * r_min, 1.1 * r_min

        x = np.linspace(x0, x1, 101)
        y = np.linspace(y0, y1, 151)
        _x, _y = np.meshgrid(x, y, indexing="ij")
        z = np.sqrt((_x - r_maj) ** 2 + _y**2)

        return x, y, z

    @pytest.mark.parametrize(
        ("r_maj", "r_min"), [(1.0, 0.2), (1.5, 0.5), (6.05, 0.1), (5.0, 4.9)]
    )
    def test_well(self, r_maj: float, r_min: float):
        """
        Test find_contours can find circular contours.
        Also test fails to find impossible contour.

        Parameters
        ----------
        r_maj : float
            Radius of bottom of well.
        r_min : float
            Radius of sepratrix contour.
        """
        levels = r_min * np.linspace(0.2, 1.0, 5)

        n_theta = 51
        theta_target = np.linspace(-np.pi, np.pi, n_theta)
        c, s = np.cos(theta_target), np.sin(theta_target)

        expected_contours = np.empty((len(levels), n_theta, 2))
        for i, r in enumerate(levels):
            expected_contours[i, :, 0] = r_maj + r * c
            expected_contours[i, :, 1] = r * s

        # Test contour search for well.
        x, y, z = self.well(r_maj, r_min)
        contour_found, _, actual_contours = (
            AxisymmetricFluxCoordinate._find_contours(
                x, y, z, (r_maj, 0.0), levels, theta_target
            )
        )

        assert np.all(contour_found)
        nptest.assert_allclose(expected_contours, actual_contours, atol=5e-3)

        # Test contour search for peak.
        contour_found, _, actual_contours = (
            AxisymmetricFluxCoordinate._find_contours(
                x, y, -z, (r_maj, 0.0), -levels, theta_target
            )
        )

        assert np.all(contour_found)
        nptest.assert_allclose(expected_contours, actual_contours, atol=5e-3)

        # Test cannot find impossible contours.
        contour_found, _, _ = AxisymmetricFluxCoordinate._find_contours(
            x, y, z, (r_maj, 0.0), (-1.0, 10000.0), theta_target
        )

        assert not np.any(contour_found)

    @staticmethod
    def double_well(
        r_maj: float, r_min: float
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        Quadratic well containing two wells.

        Parameters
        ----------
        r_maj : float
            Radius of bottom of well.
        r_min : float
            Radius of sepratrix contour.

        Returns
        -------
        x, y : np.ndarray[float]
            x and y 1d arrays.
        z : float
            2d value function.
        """
        x0, x1 = r_maj - 1.1 * r_min, r_maj + 1.1 * r_min
        y0, y1 = -1.1 * r_min, 3.3 * r_min

        x = np.linspace(x0, x1, 101)
        y = np.linspace(y0, y1, 301)
        _x, _y = np.meshgrid(x, y, indexing="ij")

        z = np.empty((len(x), len(y)))

        z[:, :151] = np.sqrt((_x[:, :151] - r_maj) ** 2 + _y[:, :151] ** 2)
        z[:, 151:] = np.sqrt(
            (_x[:, 151:] - r_maj) ** 2 + (_y[:, 151:] - 2.2 * r_min) ** 2
        )

        return x, y, z

    @pytest.mark.parametrize(
        ("r_maj", "r_min"), [(1.0, 0.2), (1.5, 0.5), (6.05, 0.1), (5.0, 4.9)]
    )
    def test_double_well(self, r_maj: float, r_min: float):
        """
        Test find_contours can distinguish contours based on axis.

        Parameters
        ----------
        r_maj : float
            Radius of bottom of well.
        r_min : float
            Radius of sepratrix contour.
        """
        levels = r_min * np.linspace(0.2, 1.0, 5)

        n_theta = 51
        theta_target = np.linspace(-np.pi, np.pi, n_theta)
        c, s = np.cos(theta_target), np.sin(theta_target)

        expected_contours = np.empty((len(levels), n_theta, 2))
        for i, r in enumerate(levels):
            expected_contours[i, :, 0] = r_maj + r * c
            expected_contours[i, :, 1] = r * s

        # Test contour search for well.
        x, y, z = self.double_well(r_maj, r_min)
        contour_found, _, actual_contours = (
            AxisymmetricFluxCoordinate._find_contours(
                x, y, z, (r_maj, 0.0), levels, theta_target
            )
        )

        assert np.all(contour_found)
        nptest.assert_allclose(expected_contours, actual_contours, atol=5e-3)

        # Test contour search for peak.
        contour_found, _, actual_contours = (
            AxisymmetricFluxCoordinate._find_contours(
                x, y, -z, (r_maj, 0.0), -levels, theta_target
            )
        )

        assert np.all(contour_found)
        nptest.assert_allclose(expected_contours, actual_contours, atol=5e-3)

        # Test contour search for other well.
        expected_contours[:, :, 1] += 2.2 * r_min
        contour_found, _, actual_contours = (
            AxisymmetricFluxCoordinate._find_contours(
                x, y, z, (r_maj, 2.2 * r_min), levels, theta_target
            )
        )

        assert np.all(contour_found)
        nptest.assert_allclose(expected_contours, actual_contours, atol=5e-3)


class TestAxisymmetricFluxCoordinate:
    """
    Test axisymmetric flux coordinate system.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def coordinate() -> AxisymmetricFluxCoordinate:
        """
        Axisymmetric flux coordinate system.

        Returns
        -------
        coordinate : AxisymmetricFluxCoordinate
            Coordinate system.
        """
        r_maj, r_min = 1.2, 0.8
        r = np.linspace(r_maj - 1.1 * r_min, r_maj + 1.1 * r_min, 81)
        z = np.linspace(-1.1 * r_min, 1.1 * r_min, 80)

        _dr, _dz = np.meshgrid(r - r_maj, z, indexing="ij")
        rho = np.sqrt(_dr**2 + _dz**2) / r_min

        rho_spline = ValueModel.root_normalised_poloidal_flux().spline_2d(
            CoordinateSystem.CYLINDRICAL, r, z, rho, (True, False, True)
        )

        return AxisymmetricFluxCoordinate.find_contours(
            CoordinateSystem.RHO_POLOIDAL,
            rho_spline,
            (r_maj, 0.0),
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("position", "expected_value"),
        [
            (np.array([0.4, 0.3, np.pi / 2]), np.array([0.4, 0.3, np.pi / 2])),
            (
                np.array([-1.0, -np.pi / 2, -np.pi / 2]),
                np.array([1.0, -np.pi / 2, np.pi / 2]),
            ),
            (
                np.array([1.0, -0.7, 1.5 * np.pi]),
                np.array([1.0, -0.7, -np.pi / 2]),
            ),
            (
                np.array([1.0, -0.7, -1.5 * np.pi]),
                np.array([1.0, -0.7, np.pi / 2]),
            ),
        ],
    )
    def test_bound_components(
        coordinate: AxisymmetricFluxCoordinate,
        position: np.ndarray[float],
        expected_value: np.ndarray[float],
    ):
        """
        Test formula to bound coordinate components i.e. r mapped positive,
        theta mapped into [-pi, pi] and z not changed.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinate
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
    def test_round_trip_netcdf(coordinate: AxisymmetricFluxCoordinate):
        """
        Test writing and reading from netCDF4 file gives same object.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinate
            Test coordinate.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            nc4.Dataset(pathlib.Path(tmpdir).joinpath("tmp.nc"), "w") as dset,
        ):
            Dimensions.write_netcdf(dset)
            coordinate.write_netcdf(dset)
            coordinate_2 = AxisymmetricFluxCoordinate.read_netcdf(
                dset[CoordinateSystem.RHO_POLOIDAL.name]
            )

        assert coordinate.coordinate_system == coordinate_2.coordinate_system

        nptest.assert_allclose(
            coordinate.rho_spline._abscissas[0],
            coordinate_2.rho_spline._abscissas[0],
        )
        nptest.assert_allclose(
            coordinate.rho_spline._abscissas[1],
            coordinate_2.rho_spline._abscissas[1],
        )
        nptest.assert_allclose(
            coordinate.rho_spline._data, coordinate_2.rho_spline._data
        )
        nptest.assert_allclose(
            coordinate.magnetic_axis_m, coordinate_2.magnetic_axis_m
        )

        nptest.assert_allclose(coordinate.rho_1d, coordinate_2.rho_1d)

        nptest.assert_allclose(coordinate.theta_1d, coordinate_2.theta_1d)

        nptest.assert_allclose(
            coordinate.isocontours_rz, coordinate_2.isocontours_rz
        )

    positions_cylindrical = (
        np.array([1.7, 0.0, 0.0]),
        np.array([1.4, np.pi / 3, 0.3]),
        np.array([1.1, 0.0, -0.3]),
    )

    positions_flux_coordinate = (
        np.array([0.625, 0.0, 0.0]),
        np.array([0.450693909433, np.pi / 3, 0.98279372324733]),
        np.array([0.39528470752105, 0.0, -1.89254688119154]),
    )

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cylindrical", "x_flux_coordinate"),
        zip(positions_cylindrical, positions_flux_coordinate, strict=False),
    )
    def test_transform(
        coordinate: AxisymmetricFluxCoordinate,
        x_cylindrical: np.ndarray[float],
        x_flux_coordinate: np.ndarray[float],
    ):
        """
        Test forward and backward transform.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinate
            Test coordinate.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        x_flux_coordinate : np.ndarray[float]
            Test flux coordinate position.
        """
        x_cylindrical_2 = coordinate.backward_transform(x_flux_coordinate)
        x_flux_coordinate_2 = coordinate.forward_transform(x_cylindrical)

        nptest.assert_allclose(x_cylindrical, x_cylindrical_2, atol=1e-7)
        nptest.assert_allclose(
            x_flux_coordinate, x_flux_coordinate_2, atol=1e-7
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("x_cylindrical", "x_flux_coordinate"),
        zip(positions_cylindrical, positions_flux_coordinate, strict=False),
    )
    def test_forward_transform_derivatives(
        coordinate: AxisymmetricFluxCoordinate,
        x_cylindrical: np.ndarray[float],
        x_flux_coordinate: np.ndarray[float],
    ):
        """
        Test derivatives of forward transform.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinate
            Test coordinate.
        x_cylindrical : np.ndarray[float]
            Test cylindrical position.
        x_flux_coordinate : np.ndarray[float]
            Test flux coordinate position.
        """
        # First derivative.
        expected_value = coordinate.forward_transform_dx(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        expected_value_2 = coordinate.forward_transform_dx(
            x_flux_coordinate, CoordinateSystem.RHO_POLOIDAL
        )

        actual_value = first_derivative_finite_difference(
            x_cylindrical, coordinate.forward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-6)

        # Second derivative.
        expected_value = coordinate.forward_transform_dx2(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        expected_value_2 = coordinate.forward_transform_dx2(
            x_flux_coordinate, CoordinateSystem.RHO_POLOIDAL
        )

        actual_value = second_derivative_finite_difference(
            x_cylindrical,
            coordinate.forward_transform,
            (Dimensions.x.size,),
            order=4,
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-6)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=5e-6)

        # Third derivative.
        expected_value = coordinate.forward_transform_dx3(
            x_cylindrical, CoordinateSystem.CYLINDRICAL
        )
        expected_value_2 = coordinate.forward_transform_dx3(
            x_flux_coordinate, CoordinateSystem.RHO_POLOIDAL
        )

        actual_value = second_derivative_finite_difference(
            x_cylindrical,
            lambda x: coordinate.forward_transform_dx(
                x, CoordinateSystem.CYLINDRICAL
            ),
            (Dimensions.x.size, Dimensions.x.size),
            order=4,
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-5)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=5e-5)


class TestAxisymmetricFluxCoordinateRebase:
    """
    Test axisymmetric flux coordinate rebase.
    """

    parent_coordinate = TestAxisymmetricFluxCoordinate.coordinate

    @staticmethod
    @pytest.fixture
    def coordinate(
        parent_coordinate: AxisymmetricFluxCoordinate,
    ) -> AxisymmetricFluxCoordinateRebase:
        """
        Axisymmetric rebased flux coordinate system.

        Parameters
        ----------
        parent_coordinate : AxisymmetricFluxCoordinate
            Parent flux coordinate system.

        Returns
        -------
        coordinate : AxisymmetricFluxCoordinateRebase
            Coordinate system.
        """
        rho_1 = np.linspace(0, 1, 51)
        rho_2 = np.sin(0.5 * np.pi * rho_1)

        rho_spline_1_to_2 = (
            ValueModel.root_normalised_toroidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_POLOIDAL, rho_1, rho_2, (True,)
            )
        )

        rho_spline_2_to_1 = (
            ValueModel.root_normalised_poloidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_TOROIDAL, rho_2, rho_1, (True,)
            )
        )

        return AxisymmetricFluxCoordinateRebase(
            parent_coordinate,
            CoordinateSystem.RHO_TOROIDAL,
            rho_spline_1_to_2,
            rho_spline_2_to_1,
        )

    @staticmethod
    def test_round_trip_netcdf(coordinate, parent_coordinate):
        """
        Test writing and reading from netCDF4 file gives same object.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinateRebase
            Test coordinate.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            nc4.Dataset(pathlib.Path(tmpdir).joinpath("tmp.nc"), "w") as dset,
        ):
            Dimensions.write_netcdf(dset)
            coordinate.write_netcdf(dset)
            coordinate_2 = AxisymmetricFluxCoordinateRebase.read_netcdf(
                dset[CoordinateSystem.RHO_TOROIDAL.name], parent_coordinate
            )

        assert coordinate.coordinate_system == coordinate_2.coordinate_system
        nptest.assert_allclose(
            coordinate.rho_spline_1_to_2._abscissas,
            coordinate_2.rho_spline_1_to_2._abscissas,
        )
        nptest.assert_allclose(
            coordinate.rho_spline_1_to_2._data,
            coordinate_2.rho_spline_1_to_2._data,
        )

        nptest.assert_allclose(
            coordinate.rho_spline_2_to_1._abscissas,
            coordinate_2.rho_spline_2_to_1._abscissas,
        )
        nptest.assert_allclose(
            coordinate.rho_spline_2_to_1._data,
            coordinate_2.rho_spline_2_to_1._data,
        )

    positions_rho_1 = (
        np.array([0.7, 0.0, 0.0]),
        np.array([0.4, np.pi / 3, 0.3]),
        np.array([0.1, 0.0, -0.3]),
    )

    positions_rho_2 = (
        np.array([0.89100652, 0.0, 0.0]),
        np.array([0.58778525, np.pi / 3, 0.3]),
        np.array([0.15643447, 0.0, -0.3]),
    )

    @staticmethod
    @pytest.mark.parametrize(
        ("x_rho_1", "x_rho_2"),
        zip(positions_rho_1, positions_rho_2, strict=False),
    )
    def test_transform(
        coordinate: AxisymmetricFluxCoordinateRebase,
        x_rho_1: np.ndarray[float],
        x_rho_2: np.ndarray[float],
    ):
        """
        Test forward and backward transform.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinateRebase
            Test coordinate.
        x_rho_1 : np.ndarray[float]
            Test flux coordinate 1 position.
        x_rho_2 : np.ndarray[float]
            Test flux coordinate 2 position.
        """
        x_rho_2_2 = coordinate.forward_transform(x_rho_1)
        x_rho_1_2 = coordinate.backward_transform(x_rho_2)

        nptest.assert_allclose(x_rho_1, x_rho_1_2, atol=1e-7)
        nptest.assert_allclose(x_rho_2, x_rho_2_2, atol=1e-7)

    @staticmethod
    @pytest.mark.parametrize(
        ("x_rho_1", "x_rho_2"),
        zip(positions_rho_1, positions_rho_2, strict=False),
    )
    def test_forward_transform_derivatives(
        coordinate: AxisymmetricFluxCoordinateRebase,
        x_rho_1: np.ndarray[float],
        x_rho_2: np.ndarray[float],
    ):
        """
        Test derivatives of forward transform.

        Parameters
        ----------
        coordinate : AxisymmetricFluxCoordinateRebase
            Test coordinate.
        x_rho_1 : np.ndarray[float]
            Test flux coordinate 1 position.
        x_rho_2 : np.ndarray[float]
            Test flux coordinate 2 position.
        """
        # First derivative.
        expected_value = coordinate.forward_transform_dx(
            x_rho_1, CoordinateSystem.RHO_POLOIDAL
        )
        expected_value_2 = coordinate.forward_transform_dx(
            x_rho_2, CoordinateSystem.RHO_TOROIDAL
        )

        actual_value = first_derivative_finite_difference(
            x_rho_1, coordinate.forward_transform, (Dimensions.x.size,)
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=1e-6)

        # Second derivative.
        expected_value = coordinate.forward_transform_dx2(
            x_rho_1, CoordinateSystem.RHO_POLOIDAL
        )
        expected_value_2 = coordinate.forward_transform_dx2(
            x_rho_2, CoordinateSystem.RHO_TOROIDAL
        )

        actual_value = second_derivative_finite_difference(
            x_rho_1,
            coordinate.forward_transform,
            (Dimensions.x.size,),
            order=4,
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-6)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=5e-6)

        # Third derivative.
        expected_value = coordinate.forward_transform_dx3(
            x_rho_1, CoordinateSystem.RHO_POLOIDAL
        )
        expected_value_2 = coordinate.forward_transform_dx3(
            x_rho_2, CoordinateSystem.RHO_TOROIDAL
        )

        actual_value = second_derivative_finite_difference(
            x_rho_1,
            lambda x: coordinate.forward_transform_dx(
                x, CoordinateSystem.RHO_POLOIDAL
            ),
            (Dimensions.x.size, Dimensions.x.size),
            order=4,
        )

        logger.info(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-5)

        logger.info(abs(expected_value_2 - actual_value))
        nptest.assert_allclose(expected_value_2, actual_value, atol=5e-5)
