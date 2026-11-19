"""
Unit tests for ray_tracing.caches.mode_conversion (Mjolhus O-X transmission).
"""

# Third party imports
import numpy as np
import numpy.testing as nptest
import pytest
from scipy import integrate

# Local imports
from crayon.ray_tracing.caches.mode_conversion import (
    mjolhus_gaussian_beam,
    mjolhus_plane_wave,
)

Y = 0.75
NPAR_OPT = np.sqrt(Y / (1 + Y))


def plane_wave_exponent(k0ln, y, n_parallel, n_y):
    """
    Mjolhus exponent A, written out independently of the code.
    """
    npar_opt = np.sqrt(y / (1 + y))
    return (
        np.pi
        * k0ln
        * np.sqrt(0.5 * y)
        * (2 * (1 + y) * (abs(n_parallel) - npar_opt) ** 2 + n_y**2)
    )


def beam_quadrature(k0ln, y, n_parallel, n_y, k0w):
    """
    exp(-A) averaged over the power spectrum (w / pi) exp(-w |dN|^2) of a
    Gaussian beam with 1/e field waist W, w = (k0 W)^2 / 2, by quadrature.
    """
    w = 0.5 * k0w**2
    half_width = 8 / np.sqrt(w)

    def integrand(ny, npar):
        spectrum = (w / np.pi) * np.exp(
            -w * ((npar - n_parallel) ** 2 + (ny - n_y) ** 2)
        )
        return spectrum * np.exp(-plane_wave_exponent(k0ln, y, npar, ny))

    return integrate.dblquad(
        integrand,
        n_parallel - half_width,
        n_parallel + half_width,
        n_y - half_width,
        n_y + half_width,
        epsabs=1e-12,
    )[0]


class TestMjolhusPlaneWave:
    """
    Unit tests for mjolhus_plane_wave.
    """

    @staticmethod
    def test_full_conversion_at_optimum():
        """
        At the optimal N_parallel with N_y = 0 all power converts.
        """
        nptest.assert_allclose(
            mjolhus_plane_wave(20.0, Y, NPAR_OPT, 0.0), 1.0, rtol=1e-12
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("n_parallel", "n_y"), [(0.5, 0.0), (0.6, 0.1), (-0.6, 0.1)]
    )
    def test_is_transmission(n_parallel, n_y):
        """
        Returns T = exp(-A), between 0 and 1, even in N_parallel.
        """
        k0ln = 16.755
        transmission = mjolhus_plane_wave(k0ln, Y, n_parallel, n_y)

        assert 0.0 <= transmission <= 1.0
        nptest.assert_allclose(
            transmission,
            np.exp(-plane_wave_exponent(k0ln, Y, n_parallel, n_y)),
            rtol=1e-12,
        )


class TestMjolhusGaussianBeam:
    """
    Unit tests for mjolhus_gaussian_beam.
    """

    @staticmethod
    @pytest.mark.parametrize(
        ("k0ln", "n_parallel", "n_y"),
        [(16.755, 0.65524, 0.0), (19.924, 0.49976, 0.0), (16.755, 0.6, 0.05)],
    )
    def test_matches_quadrature(k0ln, n_parallel, n_y):
        """
        Equals the plane-wave transmission averaged over the beam spectrum.
        """
        k0w = 209.5845 * 0.05  # 10 GHz, 5 cm waist

        nptest.assert_allclose(
            mjolhus_gaussian_beam(k0ln, Y, n_parallel, n_y, k0w),
            beam_quadrature(k0ln, Y, n_parallel, n_y, k0w),
            rtol=1e-6,
        )

    @staticmethod
    def test_tends_to_plane_wave():
        """
        As the waist grows the beam result tends to the plane-wave result.
        """
        nptest.assert_allclose(
            mjolhus_gaussian_beam(16.755, Y, 0.6, 0.05, 1.0e4),
            mjolhus_plane_wave(16.755, Y, 0.6, 0.05),
            rtol=1e-6,
        )

    @staticmethod
    def test_no_barrier():
        """
        With no evanescent layer (k0 Ln -> 0) all power converts.
        """
        nptest.assert_allclose(
            mjolhus_gaussian_beam(1e-9, Y, NPAR_OPT, 0.0, 10.0), 1.0, rtol=1e-6
        )

    @staticmethod
    def test_even_in_n_parallel():
        """
        Depends on |N_parallel| only, like the plane-wave result.
        """
        nptest.assert_allclose(
            mjolhus_gaussian_beam(16.755, Y, -0.655, 0.0, 10.48),
            mjolhus_gaussian_beam(16.755, Y, 0.655, 0.0, 10.48),
            rtol=1e-12,
        )
