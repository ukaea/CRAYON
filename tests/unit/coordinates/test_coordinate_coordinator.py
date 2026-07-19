"""
Unit tests for coordinates.coordinate_coordinator.
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
from crayon.coordinates import (
    AxisymmetricFluxCoordinate,
    AxisymmetricFluxCoordinateRebase,
    CoordinateCoordinator,
    CoordinateSystem,
    Toroidal,
)
from crayon.shared.dimensions import Dimensions
from crayon.value_model.models import ValueModel

logger = logging.getLogger(__name__)


class TestCoordinateCoordinator:
    """
    Unit tests for CoordinateCoordinator.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def coordinate_coordinator() -> CoordinateCoordinator:
        """
        CoordinateCoordinator with toroidal and root normalised poloidal flux
        coordinates defined.

        Returns
        -------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.
        """
        cc = CoordinateCoordinator()

        r_maj, r_min = 0.8, 0.6
        r = np.linspace(r_maj - 1.1 * r_min, r_maj + 1.1 * r_min, 81)
        z = np.linspace(-1.1 * r_min, 1.1 * r_min, 80)

        _dr, _dz = np.meshgrid(r - r_maj, z, indexing="ij")
        rho = np.sqrt(_dr**2 + _dz**2) / r_min

        rho_spline = ValueModel.root_normalised_poloidal_flux().spline_2d(
            CoordinateSystem.CYLINDRICAL, r, z, rho, (True, False, True)
        )

        toroidal = Toroidal((r_maj, 0.0))
        rho_poloidal = AxisymmetricFluxCoordinate.find_contours(
            CoordinateSystem.RHO_POLOIDAL,
            rho_spline,
            (r_maj, 0.0),
        )

        rho_poloidal_1d = np.linspace(0, 1, 21)
        rho_toroidal_1d = np.square(rho_poloidal_1d)

        rho_poloidal_to_toroidal = (
            ValueModel.root_normalised_toroidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal_1d,
                rho_toroidal_1d,
                (True,),
            )
        )

        rho_toroidal_to_poloidal = (
            ValueModel.root_normalised_poloidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_toroidal_1d,
                rho_poloidal_1d,
                (True,),
            )
        )

        rho_toroidal = AxisymmetricFluxCoordinateRebase(
            rho_poloidal,
            CoordinateSystem.RHO_TOROIDAL,
            rho_poloidal_to_toroidal,
            rho_toroidal_to_poloidal,
        )

        cc.register_coordinate(toroidal)
        cc.register_coordinate(rho_poloidal)
        cc.register_coordinate(rho_toroidal)

        cc.calculate_conversion_paths()

        return cc

    @staticmethod
    def test_conversion_paths(coordinate_coordinator: CoordinateCoordinator):
        """
        Test conversion paths generated correctly.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.
        """
        # Test cylindrical.
        expected_path = [
            CoordinateSystem.CARTESIAN,
            CoordinateSystem.CYLINDRICAL,
        ]
        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.CYLINDRICAL, to_target=True
        )

        assert list(path) == expected_path

        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.CYLINDRICAL, to_target=False
        )

        assert list(path) == list(reversed(expected_path))

        # Test toroidal.
        expected_path = [
            CoordinateSystem.CARTESIAN,
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.TOROIDAL,
        ]
        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.TOROIDAL, to_target=True
        )

        assert list(path) == expected_path

        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.TOROIDAL, to_target=False
        )

        assert list(path) == list(reversed(expected_path))

        # Test rho poloidal.
        expected_path = [
            CoordinateSystem.CARTESIAN,
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.RHO_POLOIDAL,
        ]
        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.RHO_POLOIDAL, to_target=True
        )

        assert list(path) == expected_path

        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.RHO_POLOIDAL, to_target=False
        )

        assert list(path) == list(reversed(expected_path))

        # Test rho toroidal.
        expected_path = [
            CoordinateSystem.CARTESIAN,
            CoordinateSystem.CYLINDRICAL,
            CoordinateSystem.RHO_POLOIDAL,
            CoordinateSystem.RHO_TOROIDAL,
        ]
        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.RHO_TOROIDAL, to_target=True
        )

        assert list(path) == expected_path

        path = coordinate_coordinator.get_conversion_path(
            CoordinateSystem.RHO_TOROIDAL, to_target=False
        )

        assert list(path) == list(reversed(expected_path))

    @staticmethod
    def test_convert_coordinate(coordinate_coordinator: CoordinateCoordinator):
        """
        Test coordinate conversion.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.
        """
        x_cartesian = np.array([1.0, 1.0, -0.8])
        x_cylindrical = np.array([np.sqrt(2), 0.25 * np.pi, -0.8])

        x_cartesian_2 = coordinate_coordinator.convert_coordinate(
            x_cylindrical, CoordinateSystem.CYLINDRICAL, forward=False
        )

        x_cylindrical_2 = coordinate_coordinator.convert_coordinate(
            x_cartesian, CoordinateSystem.CYLINDRICAL, forward=True
        )

        nptest.assert_allclose(x_cartesian, x_cartesian_2)
        nptest.assert_allclose(x_cylindrical, x_cylindrical_2)

    @staticmethod
    def test_metric_tensor(coordinate_coordinator: CoordinateCoordinator):
        """
        Test calculatetion of metric tensor.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.
        """
        # Test cylindrical.
        x_cylindrical = np.array([0.56568542, 0.78539816, -0.4])

        expected_value = np.identity(3)
        expected_value[1, 1] = x_cylindrical[0] ** 2

        actual_value = coordinate_coordinator.metric_tensor(
            CoordinateSystem.CYLINDRICAL, x_cylindrical
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test toroidal.
        x_toroidal = np.array([0.46357666, 0.78539816, -2.10069912])

        expected_value = np.identity(3)
        expected_value[1, 1] = x_cylindrical[0] ** 2
        expected_value[2, 2] = x_toroidal[0] ** 2

        actual_value = coordinate_coordinator.metric_tensor(
            CoordinateSystem.TOROIDAL, x_toroidal
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    @staticmethod
    def test_round_trip_netcdf(coordinate_coordinator: CoordinateCoordinator):
        """
        Test writing and reading from netCDF gives the same object.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.
        """
        cc = coordinate_coordinator

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            nc4.Dataset(pathlib.Path(tmpdir).joinpath("tmp.nc"), "w") as dset,
        ):
            Dimensions.write_netcdf(dset)
            cc.write_netcdf(dset)
            cc2 = CoordinateCoordinator.read_netcdf(dset)

        assert cc.coordinates.keys() == cc2.coordinates.keys()
