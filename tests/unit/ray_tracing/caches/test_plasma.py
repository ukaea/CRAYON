"""
Unit tests for ray_tracing.caches.plasma.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import CoordinateCoordinator
from crayon.ray_tracing.caches.coordinates import CoordinateCache
from crayon.ray_tracing.caches.plasma import PlasmaCache
from crayon.shared.constants import CoordinateSystem
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.shared.physics import (
    ELECTRON_REST_MASS_ENERGY_EV,
    critical_density_per_m3,
    critical_magnetic_field_strength_t,
)
from crayon.system_data import Kinetic, Magnetic
from crayon.value_model.models import ValueModel

logger = logging.getLogger(__name__)


class TestPlasmaCache:
    """
    Unit tests for PlasmaCache.
    """

    @staticmethod
    def scalar_cartesian(x_cartesian: np.ndarray[float]) -> float:
        """
        Scalar function as a function of Cartesian.

        Parameters
        ----------
        x_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        scalar : float
            Scalar function value.
        """
        x, y, z = x_cartesian
        return x**2 + y**2 + z**2

    @staticmethod
    def scalar_cylindrical(x_cylindrical: np.ndarray[float]) -> float:
        """
        Scalar function as a function of cylindrical.

        Parameters
        ----------
        x_cylindrical : np.ndarray[float]
            Cylindrical position.

        Returns
        -------
        scalar : float
            Scalar function value.
        """
        r, _, z = x_cylindrical
        return r**2 + z**2

    @staticmethod
    def scalar_dx_cartesian(
        x_cartesian: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Scalar function first derivative with respect to Cartesian position.

        Parameters
        ----------
        x_cartesian : np.ndarray[float]
            Cartesian position.

        Returns
        -------
        scalar_dx_cartesian : float
            Scalar function first derivative with respect to Cartesian
            position.
        """
        x, y, z = x_cartesian

        return_array = np.empty(3)
        return_array[0] = 2 * x
        return_array[1] = 2 * y
        return_array[2] = 2 * z

        return return_array

    @staticmethod
    def scalar_dx2_cartesian() -> np.ndarray[float]:
        """
        Scalar function second derivative with respect to Cartesian position.

        Returns
        -------
        scalar_dx2_cartesian : float
            Scalar function second derivative with respect to Cartesian
            position.
        """
        return_array = np.zeros((3, 3))

        return_array[0, 0] = 2
        return_array[1, 1] = 2
        return_array[2, 2] = 2

        return return_array

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

    @pytest.mark.parametrize("x_cartesian", positions_cartesian)
    def test_derivatives(self, x_cartesian: np.ndarray[float]):
        """
        Test derivatives of scalar function.

        Parameters
        ----------
        x_cartesian : np.ndarray[float]
            Cartesian position.
        """
        # Test first derivative.
        expected_value = self.scalar_dx_cartesian(x_cartesian)
        actual_value = first_derivative_finite_difference(
            x_cartesian, self.scalar_cartesian, ()
        )

        nptest.assert_allclose(expected_value, actual_value)

        # Test second derivative.
        expected_value = self.scalar_dx2_cartesian()
        actual_value = second_derivative_finite_difference(
            x_cartesian,
            self.scalar_cartesian,
            (),
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

    # Add multiplier as we are reusing same model.
    ne_factor = 1.0
    te_factor = 2.0
    zeff_factor = 3.0

    @pytest.fixture(scope="class")
    @staticmethod
    def coordinate_coordinator() -> CoordinateCoordinator:
        """
        Coordinate coordinator holding coordinate system information.

        Returns
        -------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.
        """
        cc = CoordinateCoordinator()
        cc.calculate_conversion_paths()

        return cc

    @pytest.fixture(scope="class")
    @staticmethod
    def coordinate_cache(
        coordinate_coordinator: CoordinateCoordinator,
    ) -> CoordinateCache:
        """
        Cache holding coordinate system data.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator.

        Returns
        -------
        cache : CoordinateCache
            Coordinate cache.
        """
        return CoordinateCache(coordinate_coordinator)

    @pytest.fixture(scope="class")
    def kinetic(self) -> Kinetic:
        """
        Kinetic plasma parameter models.

        Returns
        -------
        kinetic : Kinetic
            Kinetic plasma parameter models.
        """
        r, z = np.linspace(0.5, 2, 51), np.linspace(-1, 1, 41)
        data = np.empty((r.size, z.size))

        _x = np.zeros(3)
        for i, rr in enumerate(r):
            _x[0] = rr
            for j, zz in enumerate(z):
                _x[2] = zz
                data[i, j] = self.scalar_cylindrical(_x)

        electron_density_per_m3 = (
            ValueModel.electron_density_per_m3().spline_2d(
                CoordinateSystem.CYLINDRICAL,
                r,
                z,
                data,
                (True, False, True),
                scale_factor=self.ne_factor,
            )
        )

        electron_temperature_ev = (
            ValueModel.electron_density_per_m3().spline_2d(
                CoordinateSystem.CYLINDRICAL,
                r,
                z,
                data,
                (True, False, True),
                scale_factor=self.te_factor,
            )
        )

        effective_charge = ValueModel.effective_charge().spline_2d(
            CoordinateSystem.CYLINDRICAL,
            r,
            z,
            data,
            (True, False, True),
            scale_factor=self.zeff_factor,
        )

        return Kinetic(
            electron_density_per_m3, electron_temperature_ev, effective_charge
        )

    @pytest.fixture(scope="class")
    @staticmethod
    def magnetic() -> Magnetic:
        """
        Magnetic plasma parameter models.

        Returns
        -------
        magnetic : Magnetic
            Magnetic plasma parameter models.
        """
        magnetic_field_t = ValueModel.magnetic_field_t().constant(
            CoordinateSystem.CYLINDRICAL, np.array([0.0, 1.0, 0.0])
        )

        return Magnetic(magnetic_field_t)

    @pytest.fixture(scope="class")
    @staticmethod
    def cache(kinetic: Kinetic, magnetic: Magnetic) -> PlasmaCache:
        """
        Plasma parameter cache.

        Parameters
        ----------
        kinetic : Kinetic
            Kinetic plasma parameter models.
        magnetic : Magnetic
            Magnetic plasma parameter models.

        Returns
        -------
        cache : PlasmaCache
            Plasma parameter cache.
        """
        return PlasmaCache(kinetic, magnetic)

    @pytest.mark.parametrize(
        ("x_cartesian", "x_cylindrical"),
        zip(positions_cartesian, positions_cylindrical, strict=False),
    )
    def test_calculate(
        self,
        cache: PlasmaCache,
        coordinate_cache: CoordinateCache,
        x_cartesian: np.ndarray[float],
        x_cylindrical: np.ndarray[float],
    ):
        """
        Test calculate plasma cache.

        Parameters
        ----------
        cache : PlasmaCache
            Plasma parameter cache.
        coordinate_cache : CoordinateCache
            Coordinate cache.
        x_cartesian: np.ndarray[float]
            Cartesian position.
        x_cylindrical: np.ndarray[float]
            Cylindrical position.
        """
        frequency_ghz = 1.0
        _critical_density_per_m3 = critical_density_per_m3(frequency_ghz)
        _critical_temperature_ev = ELECTRON_REST_MASS_ENERGY_EV
        _critical_magnetic_field_strength_t = (
            critical_magnetic_field_strength_t(frequency_ghz)
        )

        coordinate_cache.set_position(CoordinateSystem.CARTESIAN, x_cartesian)

        cache.set_frequency(frequency_ghz)
        cache.calculate(coordinate_cache, derivatives=2)

        # Scalar model derivatives.
        scalar = self.scalar_cartesian(x_cartesian)
        scalar_dx = self.scalar_dx_cartesian(x_cartesian)
        scalar_dx2 = self.scalar_dx2_cartesian()

        x, y, _ = x_cartesian
        vector = np.zeros(3)
        vector[0] = -y
        vector[1] = x

        vector_dx = np.zeros((3, 3))
        vector_dx[0, 1] = -1
        vector_dx[1, 0] = 1

        vector_dx2 = np.zeros((3, 3, 3))

        # Test electron density
        _density = cache.electron_density_per_m3
        _density_norm = cache.normalised_electron_density

        nptest.assert_allclose(
            _density.value.cartesian, scalar * self.ne_factor, atol=1e-8
        )
        nptest.assert_allclose(
            _density.first_derivative.cartesian,
            scalar_dx * self.ne_factor,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _density.second_derivative.cartesian,
            scalar_dx2 * self.ne_factor,
            atol=1e-8,
        )

        nptest.assert_allclose(
            _density_norm.value,
            scalar * self.ne_factor / _critical_density_per_m3,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _density_norm.first_derivative,
            scalar_dx * self.ne_factor / _critical_density_per_m3,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _density_norm.second_derivative,
            scalar_dx2 * self.ne_factor / _critical_density_per_m3,
            atol=1e-8,
        )

        # Test temperature
        _temperature = cache.electron_temperature_ev
        _temperature_norm = cache.normalised_electron_temperature

        nptest.assert_allclose(
            _temperature.value.cartesian, scalar * self.te_factor, atol=1e-8
        )
        nptest.assert_allclose(
            _temperature.first_derivative.cartesian,
            scalar_dx * self.te_factor,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _temperature.second_derivative.cartesian,
            scalar_dx2 * self.te_factor,
            atol=1e-8,
        )

        nptest.assert_allclose(
            _temperature_norm.value,
            scalar * self.te_factor / _critical_temperature_ev,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _temperature_norm.first_derivative,
            scalar_dx * self.te_factor / _critical_temperature_ev,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _temperature_norm.second_derivative,
            scalar_dx2 * self.te_factor / _critical_temperature_ev,
            atol=1e-8,
        )

        # Test effective charge.
        _zeff = cache.effective_charge

        nptest.assert_allclose(
            _zeff.value.cartesian, scalar * self.zeff_factor, atol=1e-8
        )
        nptest.assert_allclose(
            _zeff.first_derivative.cartesian,
            scalar_dx * self.zeff_factor,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _zeff.second_derivative.cartesian,
            scalar_dx2 * self.zeff_factor,
            atol=1e-8,
        )

        # Test magnetic field.
        _magnetic_field = cache.magnetic_field_t

        nptest.assert_allclose(
            _magnetic_field.value.cartesian, vector, atol=1e-8
        )
        nptest.assert_allclose(
            _magnetic_field.first_derivative.cartesian, vector_dx, atol=1e-8
        )
        nptest.assert_allclose(
            _magnetic_field.second_derivative.cartesian, vector_dx2, atol=1e-8
        )

        # Test magnetic field strength.
        _b_magnitude = cache.magnetic_field_strength_t
        _b_magnitude_norm = cache.normalised_magnetic_field_strength

        _b = x_cylindrical[0]
        _b3 = _b * _b * _b
        x2, y2 = x * x, y * y

        _b_dx = np.zeros(3)
        _b_dx[0] = x / _b
        _b_dx[1] = y / _b

        _b_dx2 = np.zeros((3, 3))
        _b_dx2[0, 0] = y2 / _b3
        _b_dx2[0, 1] = -x * y / _b3
        _b_dx2[1, 0] = _b_dx2[0, 1]
        _b_dx2[1, 1] = x2 / _b3

        nptest.assert_allclose(_b_magnitude.value, _b, atol=1e-8)
        nptest.assert_allclose(_b_magnitude.first_derivative, _b_dx, atol=1e-8)
        nptest.assert_allclose(
            _b_magnitude.second_derivative, _b_dx2, atol=1e-8
        )

        # Minus sign as electrons Y < 0.
        nptest.assert_allclose(
            _b_magnitude_norm.value,
            -_b / _critical_magnetic_field_strength_t,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _b_magnitude_norm.first_derivative,
            -_b_dx / _critical_magnetic_field_strength_t,
            atol=1e-8,
        )
        nptest.assert_allclose(
            _b_magnitude_norm.second_derivative,
            -_b_dx2 / _critical_magnetic_field_strength_t,
            atol=1e-8,
        )

        # Test magnetic field unit vector.
        _b_unit = cache.magnetic_field_unit

        _n = np.array([-y, x, 0]) / _b
        _b5 = _b3 * _b * _b
        x3, y3 = x2 * x, y2 * y

        _n_dx = np.zeros((3, 3))
        _n_dx[0, 0] = x * y / _b3
        _n_dx[0, 1] = -x2 / _b3
        _n_dx[1, 0] = y2 / _b3
        _n_dx[1, 1] = -x * y / _b3

        _n_dx2 = np.zeros((3, 3, 3))
        _n_dx2[0, 0, 0] = (y3 - 2 * x2 * y) / _b5
        _n_dx2[0, 0, 1] = (x3 - 2 * x * y2) / _b5
        _n_dx2[0, 1, 0] = _n_dx2[0, 0, 1]
        _n_dx2[0, 1, 1] = 3 * x2 * y / _b5

        _n_dx2[1, 0, 0] = -3 * x * y2 / _b5
        _n_dx2[1, 0, 1] = (2 * x2 * y - y3) / _b5
        _n_dx2[1, 1, 0] = _n_dx2[1, 0, 1]
        _n_dx2[1, 1, 1] = (2 * x * y2 - x3) / _b5

        nptest.assert_allclose(_b_unit.value, _n, atol=1e-8)
        nptest.assert_allclose(_b_unit.first_derivative, _n_dx, atol=1e-8)
        nptest.assert_allclose(_b_unit.second_derivative, _n_dx2, atol=1e-8)
