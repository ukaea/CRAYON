"""
Unit tests for dispersion.nonrelativistic.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest
from scipy import special

# Local imports
from crayon.dispersion.non_relativistic import (
    BesselIExp,
    NonRelativisticDispersion,
    PlasmaZ,
    SusceptibilityCache,
)
from crayon.shared.constants import WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)

logger = logging.getLogger(__name__)

test_values_q = (
    np.array([0.1, 0.4, 0.0, 0.1, 0.4, 0.2]),
    np.array([0.8, 0.7, 1.0e-4, 0.3, 0.6, 0.2]),
)


class TestBesselIExp:
    """
    Unit tests for exponentially scaled modified Bessel function of second
    kind in BesselIExp.
    """

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_flr_derivatives(q: np.ndarray[float]):
        """
        Test derivatives of finite Larmor radius parameter.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        expected_value = BesselIExp.flr_dq(q)
        actual_value = first_derivative_finite_difference(
            q, BesselIExp.flr, (), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = BesselIExp.flr_dq2(q)
        actual_value = second_derivative_finite_difference(
            q, BesselIExp.flr, (), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

    test_values = ((0.0, 1), (0.0, 7), (0.5, 1), (1.0, 2), (1.0, 9))

    @staticmethod
    @pytest.mark.parametrize(("x", "nmax"), test_values)
    def test_bessel_exp_derivatives(x: float, nmax: int):
        """
        Test derivatives of exponentially scaled modified bessel function
        of second kind with respect to argument.

        Parameters
        ----------
        x : float
            Argument of Bessel function.
        nmax : int
            Maximum harmonic number.
        """

        def _func(x):
            return BesselIExp.bessel_exp(x, nmax, derivative=0)

        # Test first derivative.
        expected_value = BesselIExp.bessel_exp(x, nmax, derivative=1)
        actual_value = first_derivative_finite_difference(x, _func, (nmax,))[
            :, 0
        ]

        # Finite differencing fails at x = 0.0 for n = 0 as there is a cusp.
        # Take the 1 sided limit x -> 0+ = -1.0 .
        if np.isclose(x, 0.0):
            actual_value[0] = -1.0

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value)

        # Test second derivative.
        expected_value = BesselIExp.bessel_exp(x, nmax, derivative=2)
        actual_value = second_derivative_finite_difference(x, _func, (nmax,))[
            :, 0, 0
        ]

        # Finite differencing fails at x = 0.0 for n = 0, 1 as there is a cusp.
        # Take the 1 sided limit x -> 0+ = 1.5, -1.0.
        if np.isclose(x, 0.0):
            actual_value[0] = 1.5
            if nmax > 1:
                actual_value[1] = -1.0

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value, atol=5e-5)

        # Test third derivative.
        def _func(x):
            return BesselIExp.bessel_exp(x, nmax, derivative=1)

        expected_value = BesselIExp.bessel_exp(x, nmax, derivative=3)
        actual_value = second_derivative_finite_difference(x, _func, (nmax,))[
            :, 0, 0
        ]

        # Finite differencing fails at x = 0.0 for n = 0, 1, 2 as there is a
        # cusp. Take the 1 sided limit x -> 0+ = -2.5, 1.875, -0.75
        if np.isclose(x, 0.0):
            actual_value[0] = -2.5
            if nmax > 1:
                actual_value[1] = 1.875
            if nmax > 2:  # noqa: PLR2004
                actual_value[2] = -0.75

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value, atol=5e-5)

        # Test fourth derivative.
        def _func(x):
            return BesselIExp.bessel_exp(x, nmax, derivative=2)

        expected_value = BesselIExp.bessel_exp(x, nmax, derivative=4)
        actual_value = second_derivative_finite_difference(x, _func, (nmax,))[
            :, 0, 0
        ]

        # Finite differencing fails at x = 0.0 for n = 0, 1, 2, 3 as there is a
        # cusp. Take the 1 sided limit x -> 0+ = -2.5, 1.875, -0.75
        if np.isclose(x, 0.0):
            actual_value[0] = 4.375
            if nmax > 1:
                actual_value[1] = -3.5
            if nmax > 2:  # noqa: PLR2004
                actual_value[2] = 1.75
            if nmax > 3:  # noqa: PLR2004
                actual_value[3] = -0.5

        logger.warning(abs(expected_value - actual_value))
        assert np.allclose(expected_value, actual_value, atol=5e-5)


class TestPlasmaZ:
    """
    Unit tests for plasma dispersion function in PlasmaZ.
    """

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_doppler_derivatives(q: np.ndarray[float]):
        """
        Test derivatives of doppler parameter.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        # Test first derivative.
        expected_value = PlasmaZ.doppler_dq(q)
        actual_value = first_derivative_finite_difference(
            q, PlasmaZ.doppler, (), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test second derivative.
        expected_value = PlasmaZ.doppler_dq2(q)
        actual_value = second_derivative_finite_difference(
            q, PlasmaZ.doppler, (), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=3e-8)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_p_derivatives(q: np.ndarray[float]):
        """
        Test derivatives of resonance term parameters p.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        # Test first derivative.
        expected_value = PlasmaZ.p_dq(q)
        actual_value = first_derivative_finite_difference(
            q, PlasmaZ.p, (2,), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test second derivative.
        expected_value = PlasmaZ.p_dq2(q)
        actual_value = second_derivative_finite_difference(
            q, PlasmaZ.p, (2,), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=3e-8)

    @staticmethod
    @pytest.mark.parametrize(
        ("harmonic_gap", "doppler"),
        [(0.5, 1.0), (0.5, -1.0), (3.0, 1.0), (2.4, -1.0)],
    )
    def test_z_norm_real(harmonic_gap: float, doppler: float):
        """
        Test derivatives of real part of plasma dispersion function.

        Parameters
        ----------
        harmonic_gap : float
            Normalised frequency gap between wave and cyclotron harmonic.
        doppler : float
            Doppler parameter.

        Notes
        -----
        Avoid zeta values > 3 as otherwise sixth derivative numerically
        unstable when calculated using recursion formula.
        """
        _harmonic_gap = np.asarray(harmonic_gap).reshape((1,))
        zeta = harmonic_gap / doppler

        z_alt = -2.0 * special.dawsn(zeta)
        z1_alt = -2.0 * (1 + zeta * z_alt)
        z2_alt = -2.0 * (1.0 * z_alt + zeta * z1_alt)
        z3_alt = -2.0 * (2.0 * z1_alt + zeta * z2_alt)
        z4_alt = -2.0 * (3.0 * z2_alt + zeta * z3_alt)
        z5_alt = -2.0 * (4.0 * z3_alt + zeta * z4_alt)
        z6_alt = -2.0 * (5.0 * z4_alt + zeta * z5_alt)

        # Test first few derivatives (numerics gets bad afterwards).
        expected_value = z1_alt
        actual_value = -2.0 * first_derivative_finite_difference(
            zeta, special.dawsn, (1,), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        expected_value = z2_alt
        actual_value = -2.0 * second_derivative_finite_difference(
            zeta, special.dawsn, (1,), order=4
        )

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        # Value.
        expected_value = z_alt / doppler
        actual_value = PlasmaZ.z_norm_real(_harmonic_gap, doppler)[0]

        nptest.assert_allclose(expected_value, actual_value)

        # First derivative
        expected_value = z1_alt / doppler**2
        actual_value = PlasmaZ.z_norm_real(
            _harmonic_gap, doppler, derivative=1
        )[0]

        nptest.assert_allclose(expected_value, actual_value)

        # Second derivative
        expected_value = z2_alt / doppler**3
        actual_value = PlasmaZ.z_norm_real(
            _harmonic_gap, doppler, derivative=2
        )[0]

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        # Third derivative
        expected_value = z3_alt / doppler**4
        actual_value = PlasmaZ.z_norm_real(
            _harmonic_gap, doppler, derivative=3
        )[0]

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        # Fourth derivative
        expected_value = z4_alt / doppler**5
        actual_value = PlasmaZ.z_norm_real(
            _harmonic_gap, doppler, derivative=4
        )[0]

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        # Fifth derivative
        expected_value = z5_alt / doppler**6
        actual_value = PlasmaZ.z_norm_real(
            _harmonic_gap, doppler, derivative=5
        )[0]

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        # Sixth derivative
        expected_value = z6_alt / doppler**7
        actual_value = PlasmaZ.z_norm_real(
            _harmonic_gap, doppler, derivative=6
        )[0]

        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

    @staticmethod
    @pytest.mark.parametrize(
        ("harmonic_gap", "doppler", "normalised_collision_rate"),
        [
            (0.7, 0.9, 0.0),
            (0.7, 0.9, 0.0001),
            (0.7, 0.9, 0.001),
        ],
    )
    def test_z_norm_imag(
        harmonic_gap: float, doppler: float, normalised_collision_rate: float
    ):
        """
        Test derivatives of imaginary part of plasma dispersion function.

        Parameters
        ----------
        harmonic_gap : float
            Normalised frequency gap between wave and cyclotron harmonic.
        doppler : float
            Doppler parameter.
        normalised_collision_rate : float
            Normalised collision rate.
        """
        _harmonic_gap = np.asarray(harmonic_gap).reshape((1,))
        zeta = (harmonic_gap + 1.0j * normalised_collision_rate) / doppler

        z_norm_imag = PlasmaZ.z_norm_imag(
            _harmonic_gap, doppler, normalised_collision_rate, derivative=0
        )

        # Test total imaginary part close to calculating with complex arg.
        expected_value = (
            1.0j * np.sqrt(np.pi) * special.wofz(zeta)
        ).imag / doppler
        actual_value = z_norm_imag[0, 0] + z_norm_imag[1, 0]

        nptest.assert_allclose(expected_value, actual_value, atol=1e-6)

        # Test first derivative.
        def z_func(z):
            return -2.0 * special.dawsn(z)

        def z1_func(z):
            return -2.0 * (1 + z * z_func(z))

        def func(z):
            return np.sqrt(np.pi) * np.exp(
                -z * z
            ) + normalised_collision_rate * z1_func(z)

        expected_value = (
            first_derivative_finite_difference(
                harmonic_gap / doppler, func, (1,), order=4
            )
            / doppler**2
        )
        actual_value = np.sum(
            PlasmaZ.z_norm_imag(
                _harmonic_gap, doppler, normalised_collision_rate, derivative=1
            )[:, 0]
        )

        nptest.assert_allclose(expected_value, actual_value, atol=5e-4)

        # Test second derivative.
        expected_value = (
            second_derivative_finite_difference(
                harmonic_gap / doppler, func, (1,), order=4
            )
            / doppler**3
        )
        actual_value = np.sum(
            PlasmaZ.z_norm_imag(
                _harmonic_gap, doppler, normalised_collision_rate, derivative=2
            )[:, 0]
        )

        nptest.assert_allclose(expected_value, actual_value, atol=5e-4)

    @staticmethod
    def test_sum_diff():
        """
        Test plasma dispersion function sum and difference over opposite sign
        harmonics.
        """
        plus = np.array([1.0, 2.0, 3.0])
        minus = np.array([1.0, 4.0, 6.0])

        expected_value = np.array([[1.0, 6.0, 9.0], [0.0, -2.0, -3.0]])
        actual_value = PlasmaZ.sum_diff(plus, minus)

        nptest.assert_allclose(expected_value, actual_value)


class TestNonRelativisticDispersion:
    """
    Unit tests for NonRelativisticDispersion.
    """

    @staticmethod
    @pytest.mark.parametrize("flr", [0.001, 0.01, 0.1, 1.0])
    def test_flr_term(flr: float):
        """
        Test flr term agrees with Fitzpatrick.

        Parameters
        ----------
        flr : float
            Finite Larmor radius term.
        """
        nmax = 20

        expected_value = np.zeros(6)

        for n in range(nmax):
            ie = special.ive(n, flr)
            i1e = np.exp(-flr) * special.ivp(n, flr)

            expected_value[0] += n * n * ie / flr
            expected_value[1] += n * (i1e - ie)
            expected_value[2] += n * ie / np.sqrt(2 * flr)
            expected_value[3] += n * n * ie / flr - 2 * flr * (i1e - ie)
            expected_value[4] += np.sqrt(0.5 * flr) * (i1e - ie)
            expected_value[5] += ie

        actual_value = np.sum(
            NonRelativisticDispersion.flr_term(flr, nmax), axis=1
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("flr", [0.001, 0.01, 0.1, 1.0])
    def test_flr_term_derivatives(flr: float):
        """
        Test derivatives of finite Larmor radius dependent term with respect
        to flr.

        Parameters
        ----------
        flr : float
            Finite Larmor radius term.
        """
        n = 5

        def func(x):
            return NonRelativisticDispersion.flr_term(x, n)

        # Test first derivative.
        expected_value = NonRelativisticDispersion._flr_term_dflr(flr, n)
        actual_value = first_derivative_finite_difference(
            flr, func, (6, n), order=4
        )[:, :, 0]

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, rtol=2e-6)

        # Test second derivative.
        expected_value = NonRelativisticDispersion._flr_term_dflr2(flr, n)
        actual_value = second_derivative_finite_difference(
            flr, func, (6, n), order=4, h=5e-5
        )[:, :, 0, 0]

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, rtol=1e-5)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_flr_term_derivatives_q(q: np.ndarray[float]):
        """
        Test derivatives of finite Larmor radius dependent term with respect
        to q.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        n = 5

        def func(x):
            return NonRelativisticDispersion.flr_term(BesselIExp.flr(x), n)

        flr = BesselIExp.flr(q)
        flr_dq = BesselIExp.flr_dq(q)
        flr_dq2 = BesselIExp.flr_dq2(q)

        flr_term_dflr = NonRelativisticDispersion._flr_term_dflr(flr, n)
        flr_term_dflr2 = NonRelativisticDispersion._flr_term_dflr2(flr, n)

        # Test first derivative.
        expected_value = NonRelativisticDispersion.flr_term_dq(
            flr_term_dflr, flr_dq
        )
        actual_value = first_derivative_finite_difference(
            q, func, (6, n), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test second derivative.
        expected_value = NonRelativisticDispersion.flr_term_dq2(
            flr_term_dflr, flr_term_dflr2, flr_dq, flr_dq2
        )
        actual_value = second_derivative_finite_difference(
            q, func, (6, n), order=4, h=5e-5
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

    # Avoid making zeta > 0 as the numerical derivatives get bad.
    test_values_p = (
        np.array([0.7, 0.8]),
        np.array([0.4, -0.3]),
        np.array([1.3, 0.5]),
    )

    @staticmethod
    @pytest.mark.parametrize("p", test_values_p)
    def test_resonance_term(p: np.ndarray[float]):
        """
        Test resonance term against Fitzpatrick.

        Parameters
        ----------
        p : np.ndarray[float]
            Resonance parameters p.
        """

        def z_func(z):
            return -2 * special.dawsn(z)

        def z1_func(z):
            return -2 * (1 + z * z_func(z))

        nmax = 20
        y, d = p[0], p[1]

        expected_value = np.zeros(6)

        for n in range(-nmax + 1, nmax):
            zeta = (1 - n * y) / d
            z = z_func(zeta)
            z1 = z1_func(zeta)

            expected_value[0] += z
            expected_value[1] += np.sign(n) * z
            expected_value[2] += np.sign(n) * z1
            expected_value[3] += z
            expected_value[4] += z1
            expected_value[5] += -zeta * z1

        expected_value /= d

        actual_value = np.sum(
            NonRelativisticDispersion.resonance_term(p, nmax), axis=1
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("p", test_values_p)
    def test_resonance_term_derivatives(p: np.ndarray[float]):
        """
        Test derivatives of resonance term with respect to p.

        Parameters
        ----------
        p : np.ndarray[float]
            Resonance parameters p.
        """
        n = 5

        def func(x):
            return NonRelativisticDispersion.resonance_term(x, n)

        # Test first derivative.
        expected_value = NonRelativisticDispersion._resonance_term_dp(p, n)
        actual_value = first_derivative_finite_difference(
            p, func, (6, n), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Test second derivative.
        expected_value = NonRelativisticDispersion._resonance_term_dp2(p, n)
        actual_value = second_derivative_finite_difference(
            p,
            func,
            (6, n),
            order=4,
        )

        # Error is mostly good, doppler doppler numerical derivatives get
        # sensitive as zeta gets > ~10 hence the generous tolerance.
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=3e-6)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_resonance_term_derivatives_q(q: np.ndarray[float]):
        """
        Test derivatives of resonance term with respect to q.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        n = 5

        def func(q):
            return NonRelativisticDispersion.resonance_term(PlasmaZ.p(q), n)

        p = PlasmaZ.p(q)
        p_dq = PlasmaZ.p_dq(q)
        p_dq2 = PlasmaZ.p_dq2(q)
        resonance_term_dp = NonRelativisticDispersion._resonance_term_dp(p, n)
        resonance_term_dp2 = NonRelativisticDispersion._resonance_term_dp2(
            p, n
        )

        # Test first derivative.
        expected_value = NonRelativisticDispersion.resonance_term_dq(
            resonance_term_dp, p_dq
        )
        actual_value = first_derivative_finite_difference(
            q, func, (6, n), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-7)

        # Test second derivative.
        expected_value = NonRelativisticDispersion.resonance_term_dq2(
            resonance_term_dp, resonance_term_dp2, p_dq, p_dq2
        )
        actual_value = second_derivative_finite_difference(
            q,
            func,
            (6, n),
            order=4,
        )

        # Error is mostly good, doppler doppler numerical derivatives get
        # sensitive as zeta gets > ~10 hence the generous tolerance.
        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-4)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_chi(q: np.ndarray[float]):
        """
        Test chi value against Fitzpatrick.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        nmax = 20

        def z_func(z):
            return -2.0 * special.dawsn(z)

        def z1_func(z):
            return -2 * (1 + z * z_func(z))

        flr = BesselIExp.flr(q)
        y = q[Dimensions.IDX_Y]
        doppler = PlasmaZ.doppler(q)

        expected_value = np.zeros((3, 3), dtype=complex)

        for n in range(1 - nmax, nmax):
            ie = special.ive(n, flr)
            i1e = np.exp(-flr) * special.ivp(n, flr)

            zeta = (1 - n * y) / doppler
            z = z_func(zeta)
            z1 = z1_func(zeta)

            expected_value[0, 0] += n * n * ie / flr * z
            expected_value[0, 1] += 1.0j * n * (i1e - ie) * z
            expected_value[0, 2] += -n * ie / np.sqrt(2 * flr) * z1
            expected_value[1, 1] += (
                n * n * ie / flr - 2 * flr * (i1e - ie)
            ) * z
            expected_value[1, 2] += 1.0j * np.sqrt(0.5 * flr) * (i1e - ie) * z1
            expected_value[2, 2] += -ie * zeta * z1

        expected_value *= q[Dimensions.IDX_X] / doppler

        expected_value[1, 0] = -expected_value[0, 1]
        expected_value[2, 0] = expected_value[0, 2]
        expected_value[2, 1] = -expected_value[1, 2]

        actual_value = (
            NonRelativisticDispersion.susceptibility_hermitian(q)
            # + NonRelativisticDispersion
            #     .susceptibility_antihermitian_resonance(q)
        )

        logger.warning(actual_value)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_chi_precursor_derivatives(q: np.ndarray[float]):
        """
        Test derivatives of chi precursor with respect to q.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        nmax = 20

        def func(q):
            return NonRelativisticDispersion.chi_precursor(q, nmax)

        # First derivative.
        expected_value = NonRelativisticDispersion.chi_precursor_dq(q, nmax)
        actual_value = first_derivative_finite_difference(
            q, func, (6,), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Second derivative.
        expected_value = NonRelativisticDispersion.chi_precursor_dq2(q, nmax)
        actual_value = second_derivative_finite_difference(
            q, func, (6,), order=4
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-4)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_chi_derivatives(q: np.ndarray[float]):
        """
        Test derivatives of chi with respect to q.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        # First derivative.
        expected_value = NonRelativisticDispersion.susceptibility_hermitian_dq(
            q
        )
        actual_value = first_derivative_finite_difference(
            q,
            NonRelativisticDispersion.susceptibility_hermitian,
            (3, 3),
            is_complex=True,
            order=4,
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-8)

        # Second derivative.
        expected_value = (
            NonRelativisticDispersion.susceptibility_hermitian_dq2(q)
        )
        actual_value = second_derivative_finite_difference(
            q,
            NonRelativisticDispersion.susceptibility_hermitian,
            (3, 3),
            is_complex=True,
            order=4,
        )

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value, atol=1e-4)

    @staticmethod
    @pytest.mark.parametrize("q", test_values_q)
    def test_calculate_susceptibility(q: np.ndarray[float]):
        """
        Test calcultion of multiple values in calculate_susceptibility.

        Parameters
        ----------
        q : np.ndarray[float]
            Hamiltonian arguments q.
        """
        cache = SusceptibilityCache()

        NonRelativisticDispersion.calculate_susceptibility(
            q, cache, 2, hermitian=True, antihermitian=True
        )

        expected_value = NonRelativisticDispersion.susceptibility_hermitian(q)
        actual_value = cache.hermitian

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = NonRelativisticDispersion.susceptibility_hermitian_dq(
            q
        )
        actual_value = cache.hermitian_dq

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = (
            NonRelativisticDispersion.susceptibility_hermitian_dq2(q)
        )
        actual_value = cache.hermitian_dq2

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = (
            NonRelativisticDispersion.susceptibility_antihermitian_resonance(q)
        )
        actual_value = cache.antihermitian_resonance

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = (
            NonRelativisticDispersion.susceptibility_antihermitian_collisional(
                q
            )
        )
        actual_value = cache.antihermitian_collisional

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def test_root_finding():
        """
        Test root finding of dispersion relation.
        """
        q = np.zeros(6)
        q[0] = 0.3  # X
        q[1] = 0.7  # Y
        q[3] = 0.002  # theta

        theta = np.pi / 3.0
        s, c = np.sin(theta), np.cos(theta)
        q[4] = s
        q[5] = c

        # Generic mode.
        result_n = NonRelativisticDispersion.calculate_n(
            q, WaveMode.ANY, kinetic=False
        )
        assert not result_n.message

        q[4] = s * result_n.value
        q[5] = c * result_n.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        result_n_perp = NonRelativisticDispersion.calculate_n_perp(
            q, WaveMode.ANY, kinetic=False
        )
        assert not result_n_perp.message

        q[4] = result_n_perp.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        # Generic O mode.
        result_n = NonRelativisticDispersion.calculate_n(
            q, WaveMode.O, kinetic=False
        )
        assert not result_n.message

        q[4] = s * result_n.value
        q[5] = c * result_n.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        result_n_perp = NonRelativisticDispersion.calculate_n_perp(
            q, WaveMode.O, kinetic=False
        )
        assert not result_n_perp.message

        q[4] = result_n_perp.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        # Generic X mode.
        result_n = NonRelativisticDispersion.calculate_n(
            q, WaveMode.X, kinetic=False
        )
        assert not result_n.message

        q[4] = s * result_n.value
        q[5] = c * result_n.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        result_n_perp = NonRelativisticDispersion.calculate_n_perp(
            q, WaveMode.X, kinetic=False
        )
        assert not result_n_perp.message

        q[4] = result_n_perp.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        # Generic EBW.
        result_n = NonRelativisticDispersion.calculate_n(
            q, WaveMode.O, kinetic=True
        )
        assert not result_n.message

        q[4] = s * result_n.value
        q[5] = c * result_n.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )

        result_n_perp = NonRelativisticDispersion.calculate_n_perp(
            q, WaveMode.O, kinetic=True
        )
        assert not result_n_perp.message

        q[4] = result_n_perp.value

        nptest.assert_allclose(
            np.linalg.det(NonRelativisticDispersion.dispersion_tensor(q)).real,
            0.0,
            atol=1e-8,
        )
