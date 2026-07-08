"""
Unit tets for shared.physics
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.shared.physics import (
    critical_density_per_m3,
    critical_magnetic_field_strength_t,
    cyclotron_frequency_ghz,
    electron_ion_collision_frequency_first_derivative,
    electron_ion_collision_frequency_ghz,
    electron_ion_collision_frequency_second_derivative,
    plasma_frequency_ghz,
)

logger = logging.getLogger(__name__)


class TestRoundTrip:
    """
    Unit tests for formulas of plasma parameters and inverses.
    """

    params_fpe = (
        (20.076901, 5.0e18),
        (2.839303, 1.0e17),
        (126.977468, 2.0e20),
    )

    @staticmethod
    @pytest.mark.parametrize(("f_pe", "ne"), params_fpe)
    def test_plasma_frequency(f_pe: float, ne: float):
        """
        Test electron plasma frequency formula.

        Parameters
        ----------
        f_pe : float
            Electron plasma frequency.
        ne : float
            Plasma density correponding to the frequency.
        """
        f_pe2 = plasma_frequency_ghz(ne)
        ne2 = critical_density_per_m3(f_pe)

        nptest.assert_allclose(ne, ne2, rtol=5e-7)
        nptest.assert_allclose(f_pe, f_pe2, rtol=5e-7)

    params_fce = ((27.992490, 1.0), (11.196996, 0.4), (103.572214, 3.7))

    @staticmethod
    @pytest.mark.parametrize(("f_ce", "b"), params_fce)
    def test_cyclotron_frequency(f_ce: float, b: float):
        """
        Test electron cyclotron frequency formula.

        Parameters
        ----------
        f_ce : float
            Electron cyclotron frequency.
        b : float
            Magnetic field strength correponding to the frequency.
        """
        f_ce2 = cyclotron_frequency_ghz(b)
        b2 = critical_magnetic_field_strength_t(f_ce)

        nptest.assert_allclose(b, b2)
        nptest.assert_allclose(f_ce, f_ce2, rtol=5e-7)


class TestDerivatives:
    """
    Unit tests for derivative formulas.
    """

    params_nu = (
        (2.0e17, 500.0, 1.0),
        (1.0e19, 1500.0, 1.5),
        (2.0e21, 1200.0, 1.0),
        (1.0e15, 20.0, 2.0),
    )

    @staticmethod
    @pytest.mark.parametrize(("ne", "te", "zeff"), params_nu)
    def test_electron_ion_collision_frequency(
        ne: float, te: float, zeff: float
    ):
        """
        Test electron-ion collision frequency formulas against finite
        difference.

        Parameters
        ----------
        ne : float
            Electron density.
        te : float
            Electron temperature.
        zeff : float
            Effective charge.
        """
        _x = np.array([ne, te, zeff])

        # Test first derivative.
        expected_value = electron_ion_collision_frequency_first_derivative(
            ne, te, zeff
        )

        actual_value = np.empty(3)

        actual_value = first_derivative_finite_difference(
            _x, lambda x: electron_ion_collision_frequency_ghz(*x), ()
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test second derivative.
        expected_value = electron_ion_collision_frequency_second_derivative(
            ne, te, zeff
        )

        actual_value = np.empty(3)

        actual_value = second_derivative_finite_difference(
            _x, lambda x: electron_ion_collision_frequency_ghz(*x), ()
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)
