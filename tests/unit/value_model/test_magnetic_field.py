"""
Unit tests for value_models.magnetic_field.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.value_model import ValueCache
from crayon.value_model.magnetic_field import AxisymmetricMagneticField
from crayon.value_model.splines import Spline1D, Spline2D

logger = logging.getLogger(__name__)


class TestAxisymmetricMagneticField:
    """
    Unit tests for AxisymmetricMagneticField.
    """

    __slots__ = ()

    @staticmethod
    @pytest.fixture
    def magnetic_field() -> AxisymmetricMagneticField:
        """
        Axisymmetric magnetic field model.

        Returns
        -------
        magnetic_field : AxisymmetricMagneticField
            Magnetic field.
        """
        radius_1d = np.linspace(1, 3, 101)
        height_1d = np.linspace(-1, 1, 91)

        _r, _z = np.meshgrid(radius_1d, height_1d, indexing="ij")
        rho_poloidal_2d = (
            1 - np.cos(0.5 * np.pi * (_r - 2.0)) * np.cos(0.5 * np.pi * _z)
        ) / 0.8

        total_flux = 2.0
        rho_poloidal_1d = np.linspace(0, 1, 51)
        f_toroidal_1d = np.sin(0.5 * np.pi * rho_poloidal_1d) ** 2

        rho_poloidal = Spline2D(
            CoordinateSystem.CYLINDRICAL,
            Dimensions.x,
            (),
            "",
            radius_1d,
            height_1d,
            rho_poloidal_2d,
            (True, False, True),
        )

        f_toroidal_1d = Spline1D(
            CoordinateSystem.RHO_POLOIDAL,
            Dimensions.one,
            (),
            "T.m",
            rho_poloidal_1d,
            f_toroidal_1d,
            (True,),
        )

        return AxisymmetricMagneticField(
            rho_poloidal,
            f_toroidal_1d,
            total_flux,
        )

    test_positions = (
        np.array([2.5, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.5]),
        np.array([2.5, 0.0, 0.5]),
    )

    @staticmethod
    @pytest.mark.parametrize("position", test_positions)
    def test_psi_derivatives(
        magnetic_field: AxisymmetricMagneticField, position: np.ndarray[float]
    ):
        """
        Test derivatives of normalised poloidal flux function.

        Parameters
        ----------
        magnetic_field: AxisymmetricMagneticField,
            Magnetic field model.
        position: np.ndarray[float]
            Test position.
        """
        position_reshaped = position.reshape((1, 3))

        def psi_func(x):
            return magnetic_field._psi_norm(x.reshape((1, 3))).item()

        expected_value = magnetic_field._psi_norm_dx(
            position_reshaped
        ).reshape((3,))
        actual_value = first_derivative_finite_difference(
            position, psi_func, ()
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test hessian
        expected_value = magnetic_field._psi_norm_dx2(
            position_reshaped
        ).reshape((3, 3))
        actual_value = second_derivative_finite_difference(
            position, psi_func, ()
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        # Test third derivative (jerk).
        def psi_dx_func(x):
            return magnetic_field._psi_norm_dx(x.reshape((1, 3))).reshape((3,))

        expected_value = magnetic_field._psi_norm_dx3(
            position_reshaped
        ).reshape((3, 3, 3))
        actual_value = second_derivative_finite_difference(
            position, psi_dx_func, (3,)
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

    @staticmethod
    @pytest.mark.parametrize("position", test_positions)
    def test_f_derivatives(
        magnetic_field: AxisymmetricMagneticField, position: np.ndarray[float]
    ):
        """
        Test derivatives of diamagnetic function F.

        Parameters
        ----------
        magnetic_field: AxisymmetricMagneticField,
            Magnetic field model.
        position: np.ndarray[float]
            Test position.
        """
        position_reshaped = position.reshape((1, 3))

        def f_func(x):
            return magnetic_field._f_toroidal(x.reshape((1, 3))).item()

        # Test jacobian.
        expected_value = magnetic_field._f_toroidal_dx(
            position_reshaped
        ).reshape((3,))

        actual_value = first_derivative_finite_difference(position, f_func, ())

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-6)

        # Test hessian.
        expected_value = magnetic_field._f_toroidal_dx2(
            position_reshaped
        ).reshape((3, 3))
        actual_value = second_derivative_finite_difference(
            position, f_func, ()
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-6)

        # Test jerk.
        def f_dx_func(x):
            return magnetic_field._f_toroidal_dx(x.reshape((1, 3))).reshape((
                3,
            ))

        expected_value = magnetic_field._f_toroidal_dx3(
            position_reshaped
        ).reshape((3, 3, 3))
        actual_value = second_derivative_finite_difference(
            position, f_dx_func, (3,)
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-6)

    @staticmethod
    @pytest.mark.parametrize("position", test_positions)
    def test_derivatives(
        magnetic_field: AxisymmetricMagneticField, position: np.ndarray[float]
    ):
        """
        Test derivatives of magnetic field vector..

        Parameters
        ----------
        magnetic_field: AxisymmetricMagneticField,
            Magnetic field model.
        position: np.ndarray[float]
            Test position.

        Notes
        -----
        Generous tolerances as spline fits.
        """
        # Test jacobian.
        expected_value = magnetic_field.jacobian(position)
        actual_value = first_derivative_finite_difference(
            position, magnetic_field.value, (3,), order=4
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test hessian.
        expected_value = magnetic_field.hessian(position)
        actual_value = second_derivative_finite_difference(
            position, magnetic_field.value, (3,), order=4
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-8)

        # Test jerk.
        expected_value = magnetic_field.jerk(position)
        actual_value = second_derivative_finite_difference(
            position, magnetic_field.jacobian, (3, 3), order=4
        )

        assert expected_value.shape == actual_value.shape
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

    @staticmethod
    @pytest.mark.parametrize("position", test_positions)
    def test_fill_cache(
        magnetic_field: AxisymmetricMagneticField, position: np.ndarray[float]
    ):
        """
        Test filling of cache.

        Parameters
        ----------
        magnetic_field: AxisymmetricMagneticField,
            Magnetic field model.
        position: np.ndarray[float]
            Test position.
        """
        cache = ValueCache.for_model(magnetic_field)
        magnetic_field.fill_cache(cache, position, derivatives=3)

        # Test value.
        expected_value = magnetic_field(position, nu=0)
        actual_value = cache.value

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test jacobian.
        expected_value = magnetic_field(position, nu=1)
        actual_value = cache.jacobian

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test hessian.
        expected_value = magnetic_field(position, nu=2)
        actual_value = cache.hessian

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test jerk.
        expected_value = magnetic_field(position, nu=3)
        actual_value = cache.jerk

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)
