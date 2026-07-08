"""
Fully relativistic kinetic dispersion tensor model.
"""

# Standard imports
import itertools
import logging

# Third party imports
import numpy as np
from scipy import special

# Local imports
from crayon.dispersion.base import DispersionModel, DispersionType
from crayon.shared.constants import WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array, pairwise
from crayon.shared.numerics import gauss_kronrod_nodes_weights
from crayon.shared.types import (
    ComplexType,
    FloatArray,
    FloatType,
)

logger = logging.getLogger(__name__)


def get_q_sqrt_mesh(q0: float, q: float, n_points: int):
    """
    Generate normalised momentum mesh. Points are spaced such that q**0.5 is
    evenly spaced.
    """
    x_norm = np.linspace(0, 1, n_points)
    return q0 + (q - q0) * x_norm * x_norm


# This returns zero elements of an array.
EMPTY_SLICE = slice(0)


def q_perp_resonance_curve(
    normalised_magnetic_field_strength: float,
    n_parallel: float,
    harmonic_number: int,
    theta_thermal: float,
    /,
    *,
    n_points: int = 100,
    q_thermal_max: float = 5.0,
    q_perp: FloatArray = None,
    q_parallel: FloatArray = None,
) -> tuple[bool, FloatArray, FloatArray]:
    """
    Calculate normalised perpendicular momentum q_perp along the cyclotron
    resonance curve as a function of normalised parallel momentum q_parallel.

    The momentum is normalised with respect to the rest mass momentum i.e.
    q = p / p0 where p0 = m_0 * c.

    Parameters
    ----------
    normalised_magnetic_field_strength : float
        Normalised magnetic field strength Y = f_ce / f [].

    n_parallel : float
        Parallel refractive index [].

    harmonic_number : int
        Cyclotron harmonic number.

    theta_thermal : float
        Temperature normalised to rest mass energy []. Used to calculate the
        maximum momentum q_max = q_thermal_max * q_thermal where
        q_thermal = sqrt(theta * (2 + theta)).

    n_points : int, optional
        Number of q_parallel values to evaluate resonance curve at.

    q_thermal_max : float, optional
        Maximum absolute value of normalised momentum used in calculations.
        Default = 5.0

    Returns
    -------
    is_resonance : bool
        Flag if there is a resonance or not.

    q_parallel : np.ndarray(float)
        Normalised parallel momentum of the resonance curve.

    q_perp : np.ndarray(float)
        Normalised parallel momentum of the resonance curve.
    """
    ny = harmonic_number * normalised_magnetic_field_strength
    ny2 = ny * ny
    n_parallel2 = n_parallel * n_parallel
    delta = 1 - n_parallel2

    q_parallel = get_return_array(q_parallel, (n_points,), FloatType)
    q_perp = get_return_array(q_perp, (n_points,), FloatType)

    if ny == 0.0 or ny2 < delta:
        return False, None, None

    # Calculate maximum momentum.
    q_thermal = np.sqrt(theta_thermal * (2 + theta_thermal))
    q_max = q_thermal_max * q_thermal

    sigma = np.sign(n_parallel)

    if delta > 0.0:
        # Resonance curve is an ellipse.

        # No resonance if n <= 0.
        if harmonic_number <= 0:
            return False, None, None

        # Calculate minimum and maximum q_parallel value on resonance ellipse.
        q_par_mid = ny * n_parallel / delta
        q_parallel_radius2 = (ny2 - delta) / (delta * delta)
        q_parallel_radius = np.sqrt(q_parallel_radius2)
        q_par_min = q_par_mid - q_parallel_radius
        q_par_max = q_par_mid + q_parallel_radius

        # Check if resonance curve in bounds.
        # Depending on n_parallel sign, plus or minus term could be closest.
        if q_par_min >= q_max or q_par_max <= -q_max:
            return False, None, None

        # Check if vertices of ellipse are in range.
        q_parallel[:] = get_q_sqrt_mesh(
            max(q_par_min, -q_max), min(q_par_max, q_max), n_points
        )

        # Check if extremal points lie inside q_parallel mesh.
        in_range_left = q_par_min >= -q_max
        in_range_right = q_par_max <= q_max

        # Don't calculate q_perp at the vertices as we know its zero and
        # numerical errors might cause the squared radius to be < 0.
        if in_range_left and in_range_right:
            # q_par_min > -q_max and q_par_max < q_max.
            q_perp[0] = 0.0
            q_perp[-1] = 0.0
            s = slice(1, n_points - 1)
        elif in_range_right:
            # q_par_min < -q_max and q_par_max < q_max.
            q_perp[-1] = 0.0
            s = slice(0, n_points - 1)
        elif in_range_left:
            # q_par_min > -q_max and q_par_max > q_max.
            q_perp[0] = 0.0
            s = slice(1, n_points)
        else:
            # q_par_min < -q_max and q_par_max > q_max.
            # However resonance curve might still pass through grid.
            s = slice(0, n_points)

        # Calculate q_perp values.
        ellipse_radius2 = ny2 / delta - 1
        q_perp[s] = np.sqrt(
            ellipse_radius2 - delta * np.square(q_parallel[s] - q_par_mid)
        )

    elif delta == 0.0:
        # Resonance curve is a parabola.

        # No resonance if n <= 0.
        if harmonic_number <= 0:
            return False, None, None

        # Calculate minimum |q_parallel| value on resonance curve.
        q_par0 = 0.5 * sigma * (1 - ny2) / ny

        # Check if resonance curve in bounds.
        if abs(q_par0) >= q_thermal:
            return False, None, None

        # Generate q_parallel values.
        # Don't calculate q_perp at the vertices as we know its zero and
        # numerical errors might cause the squared radius to be < 0.
        if sigma > 0:
            # N_parallel = 1 so no solutions for q_parallel < q_par0.
            q_parallel[:] = get_q_sqrt_mesh(q_par0, q_max, n_points)
            q_perp[0] = 0.0
            s = slice(1, None)
        else:
            # N_parallel = -1 so no solutions for q_parallel > q_par0.
            q_parallel[:] = get_q_sqrt_mesh(-q_max, q_par0, n_points)
            q_perp[-1] = 0.0
            s = slice(0, n_points - 1)

        # Calculate q_perp values.
        q_perp[s] = np.sqrt(2 * ny * sigma * q_parallel[s] + ny2 - 1)

    else:
        # Resonance curve is a hyperbola.

        # Calculate minimum |q_parallel| value on resonance curve.
        abs_delta = abs(delta)
        q_par0 = (
            -ny * n_parallel + sigma * np.sqrt(ny2 + abs_delta)
        ) / abs_delta

        # Check if resonance curve in bounds.
        if (n_parallel > 0 and q_par0 > q_max) or (
            n_parallel < 0 and q_par0 < -q_max
        ):
            return False, None, None

        # Generate q_parallel values.
        # Don't calculate q_perp at the vertices as we know its zero and
        # numerical errors might cause the squared radius to be < 0.
        if sigma > 0:
            # N_parallel = 1 so no solutions for q_parallel < q_par0.
            q_parallel[:] = get_q_sqrt_mesh(q_par0, q_max, n_points)
            q_perp[0] = 0.0
            s = slice(1, None)
        else:
            # N_parallel = -1 so no solutions for q_parallel > q_par0.
            q_parallel[:] = get_q_sqrt_mesh(-q_max, q_par0, n_points)
            q_perp[-1] = 0.0
            s = slice(0, n_points - 1)

        # Calculate q_perp values.
        q_perp[s] = np.sqrt(
            abs_delta * np.square(q_parallel[s] + ny * n_parallel / abs_delta)
            - ny2 / abs_delta
            - 1
        )

    return True, q_parallel, q_perp


class BesselJ:
    @classmethod
    def b(cls, q):
        y = q[Dimensions.IDX_Y]
        n_perp = q[Dimensions.IDX_N_PERP]

        return n_perp / y

    @classmethod
    def bessel(cls, b: float, n: int, /, *, return_array: FloatArray = None):
        """
        Return Bessel function of first kind J_k(x) for k = 0 to n.
        """
        return_array = get_return_array(return_array, (n,), FloatType)

        special.jv(np.arange(n), b.real, out=return_array)

        return return_array

    @classmethod
    def bessel_dx(
        cls, b: float, n: int, /, *, return_array: FloatArray = None
    ):
        """
        Return first derivative with respect to argument of Bessel function
        of first kind J_k'(x) for k = 0 to n.
        """
        return_array = get_return_array(return_array, (n,), FloatType)

        special.jvp(np.arange(n), b.real, n=1, out=return_array)

        return return_array

    @classmethod
    def estimate_number_zeros(cls, b: float, n: int):
        """
        Estimate the number of zeros of Jn and Jn' in interval [0, b].
        """
        m = int(max(0, np.floor(b / np.pi + 0.25 - 0.5 * abs(n))))

        m_prime = int(max(0, np.floor(b / np.pi + 0.75 - 0.5 * abs(n))))

        return m, m_prime

    @classmethod
    def estimate_argument_zeros_bessel(cls, n: int, m: int):
        """
        Estimate the argument of m-th zero of Jn in interval [0, b].

        Notes
        -----
        Use DLMF 10.21.19 keeping first 2 terms.
        For integer n, Jn = (-1)^n J{-n} so use absolute value of harmonic.
        """
        a = np.pi * (m + 0.5 * abs(n) - 0.25)
        return a - 0.125 * (4.0 * n * n - 1.0) / a

    @classmethod
    def estimate_argument_zeros_bessel_dx(cls, n: int, m: int):
        """
        Estimate the argument of m-th zero of Jn' in interval [0, b].

        Notes
        -----
        Use DLMF 10.21.20 keeping first 2 terms.
        For integer n, J'n = (-1)^n J{-n}' so use absolute value of harmonic.
        """
        b = np.pi * (m + 0.5 * abs(n) - 0.75)
        return b - 0.125 * (n * n + 3.0) / b


class ResonanceCurve:
    __slots__ = (
        "abs_n_parallel",
        "delta",
        "ellipse",
        "n_parallel",
        "n_parallel2",
        "normalised_magnetic_field_strength",
        "ny",
        "ny2",
    )

    def __init__(
        self,
        normalised_magnetic_field_strength: float,
        n_parallel: float,
    ):
        """ """
        self.normalised_magnetic_field_strength = float(
            normalised_magnetic_field_strength
        )
        self.n_parallel = float(n_parallel)

        self.abs_n_parallel = abs(self.n_parallel)
        self.n_parallel2 = self.n_parallel * self.n_parallel
        self.delta = 1.0 - self.n_parallel2
        self.ellipse = self.abs_n_parallel < 1

        self.ny = 0.0
        self.ny2 = 0.0

    def set_harmonic(self, harmonic_number: int):
        self.ny = harmonic_number * self.normalised_magnetic_field_strength
        self.ny2 = self.ny * self.ny

    def exists(self):
        return self.ny2 > self.delta

    def min_gamma(self):
        """
        Calculate minimum value of gamma there is a cyclotron resonance.
        """
        if np.isclose(self.n_parallel, 1.0):
            min_gamma = np.sqrt(1 + 0.25 * np.square(1 - self.ny2 / self.ny))
        else:
            min_gamma = (
                self.ny - self.abs_n_parallel * np.sqrt(self.ny2 - self.delta)
            ) / self.delta

        return min_gamma

    def max_gamma(self):
        """
        Calculate maximum value of gamma on resonance curve.

        Parameters
        ----------
        theta
            Electron thermal energy normalised to rest mass energy.
        """
        if self.ellipse:
            return (
                self.ny + self.abs_n_parallel * np.sqrt(self.ny2 - self.delta)
            ) / self.delta
        return np.inf

    def max_q_perp(self, max_gamma):
        """
        Calculate maximum value of perpendicular normalised momentum on the
        resonance curve.

        Parameters
        ----------
        max_gamma
            Maximum value of Lorentz factor on resonance curve.
        """
        if self.ellipse:
            return np.sqrt(self.ny2 / self.delta - 1.0)
        return np.sqrt(
            max_gamma * max_gamma
            - 1.0
            - np.square((max_gamma - self.ny) / self.n_parallel)
        )

    def q_perp_to_gamma(self, q_perp: float, /, *, minus: bool = True):
        """
        Calculate Lorentz factor corresponding to a given perpendicular
        normalised momentum on the resonance curve.
        """
        if np.isclose(self.n_parallel, 1.0):
            return 0.5 * (1.0 + self.ny2 + q_perp * q_perp) / self.ny
        if not minus and self.ellipse:
            return (
                self.ny
                + self.abs_n_parallel
                * np.sqrt(self.ny2 - self.delta * (1 + q_perp * q_perp))
            ) / self.delta
        return (
            self.ny
            - self.abs_n_parallel
            * np.sqrt(self.ny2 - self.delta * (1 + q_perp * q_perp))
        ) / self.delta

    def gamma_to_q_perp(self, gamma: float):
        """
        Calculate perpendicular normalised momentum corresponding to a given
        Lorentz factor on the resonance curve.
        """
        return np.sqrt(
            np.clip(
                gamma * gamma
                - 1.0
                - np.square((gamma - self.ny) / self.n_parallel),
                0.0,
                None,
            )
        )


class FullyRelativisticDispersion(DispersionModel):
    __slots__ = ()

    dispersion_type = DispersionType.FULLY_RELATIVISTIC

    nodes_leggauss, weights_leggauss = np.polynomial.legendre.leggauss(40)
    nodes_laggauss, weights_laggauss = np.polynomial.laguerre.laggauss(24)

    MAX_ZEROS = 20

    @classmethod
    def _fill_leggauss_nodes_weights(
        cls, a: float, b: float, nodes: FloatArray, weights: FloatArray
    ):
        nodes[:] = a + 0.5 * (b - a) * cls.nodes_leggauss
        weights[:] = 0.5 * (b - a) * cls.weights_laggauss

    @classmethod
    def _chi_precursor_antihermitian(
        cls,
        q: FloatArray,
        n: int,
        /,
        *,
        return_array: FloatArray = None,
        h: float = 1.0,
        atol: float = 1.0e-8,
    ):
        """
        Anti-hermitian part of conductivity elements.

        Calculated using Gauss Laguerre quadrature over a scaled gamma =
        mu * (gamma - 1)
        """
        # Generate return array.
        return_array = get_return_array(return_array, 6, ComplexType)
        return_array.fill(0.0)

        # Unpack plasma variables.
        x = q[Dimensions.IDX_X]
        y = q[Dimensions.IDX_Y]
        theta = q[Dimensions.IDX_THETA]
        n_perp = q[Dimensions.IDX_N_PERP]
        n_parallel = q[Dimensions.IDX_N_PARALLEL]

        mu = 1.0 / theta
        nu_perp = n_perp / y
        inv_nu_perp = 1 / nu_perp
        inv_nu_perp2 = inv_nu_perp * inv_nu_perp

        # Nodes and weights for quadrature.
        # NOTE: nodes are scaled Lorentz factor gamma' = mu * (gamma - 1).
        nodes = np.empty_like(cls.nodes_leggauss)
        weights = np.empty_like(cls.weights_leggauss)
        samples = np.empty(nodes.size, dtype=FloatType)

        # Consider gamma up to 23x thermal energy.
        # Integrands decay ~ e^-x where x = (gamma - 1) / theta.
        # chi_yy, chi_yz, chi_zz decay slowest as they are ~ x^2 e^-x
        # At x=23 x^2 e^-x ~ 10^-8.
        gamma_limit = 1.0 + 23.0 * theta

        # Useful intermediate arrays.
        normalised_momentum = np.empty_like(nodes)
        q_perp, q_parallel = np.empty_like(nodes), np.empty_like(nodes)
        bessel_term = np.empty_like(nodes)

        # Bounds of gamma integration.
        gamma_0, gamma_1 = 0.0, 0.0

        # Error estimate of susceptibility elements.
        contribution_coarse = np.empty_like(return_array)
        contribution_fine = np.empty_like(return_array)

        gamma_interval_jn2 = np.zeros(2 + 2 * cls.MAX_ZEROS)
        gamma_interval_jn_dx2 = np.zeros(2 + 2 * cls.MAX_ZEROS)
        gamma_interval_jn_jn_dx = np.zeros(2 + 2 * 2 * cls.MAX_ZEROS)

        # Only get contribution if a resonance exists.
        # No resonances are possible for n <= 0 if |n_parallel| < 1.
        harmonic_range = (
            range(1, n + 1) if abs(n_parallel) < 1 else range(-n, n + 1)
        )

        bessel_j = np.zeros(nodes.size, dtype=FloatType)
        bessel_j_dx = np.zeros(nodes.size, dtype=FloatType)

        resonance = ResonanceCurve(y, n_parallel)

        for _n in harmonic_range:
            resonance.set_harmonic(_n)

            if not resonance.exists():
                continue

            # Calculate bounds of gamma integral.
            gamma_0 = resonance.min_gamma()

            if gamma_0 >= gamma_limit:
                # Resonance curve only for a neglibible population of
                # hyper-energetic electrons.
                continue

            gamma_1 = min(gamma_limit, resonance.max_gamma())

            # Estimate out how many Bessel function zeros are in the integrand.
            # First calculate maximum q_perp on resonance curve.
            q_perp_max = resonance.max_q_perp(gamma_1)

            # Use asymptotic formula to estimate number of zeros.
            # Zeros of jn^2 are zeros of jn.
            # Zeros of (jn')^2 are zeros of jn'.
            # Zeros of jn * jn' are combination of both jn and jn' zeros.
            m, m_prime = BesselJ.estimate_number_zeros(
                nu_perp * q_perp_max, _n
            )

            m = min(m, cls.MAX_ZEROS)
            m_prime = min(m_prime, cls.MAX_ZEROS)

            gamma_interval_jn2[0] = gamma_0
            gamma_interval_jn_dx2[0] = gamma_0

            n_jn2, n_jn_dx2 = 2, 2

            q_perp_zeros_jn = (
                inv_nu_perp
                * BesselJ.estimate_argument_zeros_bessel(
                    _n, np.arange(1, m + 1)
                )
            )

            if resonance.ellipse:
                # Encounter each zero twice.
                n_zeros = 2 * m
                q_perp_iter = itertools.chain(
                    q_perp_zeros_jn, reversed(q_perp_zeros_jn)
                )
            else:
                n_zeros = m
                q_perp_iter = q_perp_zeros_jn

            for i, _q_perp in enumerate(q_perp_iter, start=1):
                gamma_interval_jn2[i] = resonance.q_perp_to_gamma(
                    _q_perp, minus=i <= m
                )

                if i == n_zeros or gamma_interval_jn2[i] >= gamma_1:
                    n_jn2 += i - 1
                    break

            gamma_interval_jn2[n_jn2 - 1] = gamma_1

            q_perp_zeros_jn_dx = (
                inv_nu_perp
                * BesselJ.estimate_argument_zeros_bessel_dx(
                    _n, np.arange(1, m_prime + 1)
                )
            )

            if resonance.ellipse:
                # Encounter each zero twice.
                n_zeros = 2 * m_prime
                q_perp_iter = itertools.chain(
                    q_perp_zeros_jn_dx, reversed(q_perp_zeros_jn_dx)
                )
            else:
                n_zeros = m_prime
                q_perp_iter = q_perp_zeros_jn_dx

            for i, _q_perp in enumerate(q_perp_iter, start=1):
                gamma_interval_jn_dx2[i] = resonance.q_perp_to_gamma(
                    _q_perp, minus=i <= m_prime
                )

                if i == n_zeros or gamma_interval_jn_dx2[i] >= gamma_1:
                    n_jn_dx2 += i - 1
                    break

            gamma_interval_jn_dx2[n_jn_dx2 - 1] = gamma_1

            # Calculate zeros of jn * jn'.
            n_jn_jn_dx = n_jn2 + n_jn_dx2 - 2

            gamma_interval_jn_jn_dx[0] = gamma_0
            gamma_interval_jn_jn_dx[1 : n_jn2 - 1] = gamma_interval_jn2[
                1 : n_jn2 - 1
            ]
            gamma_interval_jn_jn_dx[n_jn2 - 1 : n_jn_jn_dx - 1] = (
                gamma_interval_jn_dx2[1 : n_jn_dx2 - 1]
            )

            gamma_interval_jn_jn_dx[1 : n_jn_jn_dx - 1] = sorted(
                gamma_interval_jn_jn_dx[1 : n_jn_jn_dx - 1]
            )

            gamma_interval_jn_jn_dx[n_jn_jn_dx - 1] = gamma_1

            # logger.info((gamma_0, gamma_1))
            # logger.info(gamma_interval_jn2[:n_jn2])
            # logger.info(gamma_interval_jn_dx2[:n_jn_dx2])
            # logger.info(gamma_interval_jn_jn_dx[:n_jn_jn_dx])

            # import matplotlib.pyplot as plt

            # _g = np.linspace(gamma_0, gamma_1, 101)
            # _q_perp = resonance.gamma_to_q_perp(_g)

            # bessel = special.jv(_n, nu_perp * _q_perp)
            # bessel_dx = special.jvp(_n, nu_perp * _q_perp, n=1)

            # plt.plot(_q_perp, bessel, color="black")
            # plt.scatter(
            #     q_perp_zeros_jn,
            #     np.zeros_like(q_perp_zeros_jn),
            #     color="red"
            # )
            # plt.show()

            # plt.title(r"$J_n^2$")
            # plt.plot(_g, bessel * bessel, color="black")
            # plt.scatter(
            #     gamma_interval_jn2[:n_jn2],
            #     np.zeros_like(gamma_interval_jn2[:n_jn2]),
            #     color="red"
            # )
            # plt.show()

            # plt.title(r"$(J_n')^2$")
            # plt.plot(_g, bessel_dx * bessel_dx, color="black")
            # plt.scatter(
            #     gamma_interval_jn_dx2[:n_jn_dx2],
            #     np.zeros_like(gamma_interval_jn_dx2[:n_jn_dx2]),
            #     color="red"
            # )
            # plt.show()

            # plt.title(r"$J_n J_n'$")
            # plt.plot(_g, bessel * bessel_dx, color="black")
            # plt.scatter(
            #     gamma_interval_jn_jn_dx[:n_jn_jn_dx],
            #     np.zeros_like(gamma_interval_jn_jn_dx)[:n_jn_jn_dx],
            #     color="red"
            # )
            # plt.show()

            # continue

            # Integrate along each integrand.
            reached_tolerance = False

            # for order in (7, 10,):
            for order in (7, 10, 15, 20, 25, 30):
                # Reset arrays.
                contribution_coarse.fill(0.0)
                contribution_fine.fill(0.0)

                # for g0, g1 in [
                #     gamma_interval_jn2[0],
                #     gamma_interval_jn2[n_jn2 - 1]
                # ]:

                # Jn^2 terms.
                for g0, g1 in pairwise(gamma_interval_jn2[:n_jn2]):
                    nodes, weights, gauss_mask, gauss_weights = (
                        gauss_kronrod_nodes_weights(g0, g1, order)
                    )

                    # Critical pitch angle.
                    q = np.sqrt(nodes * nodes - 1.0)
                    tau_c = (nodes - _n * y) / (n_parallel * q)

                    q_parallel = q * tau_c
                    q_perp = q * np.sqrt(np.clip(1 - tau_c * tau_c, 0.0, None))
                    bessel_arg = nu_perp * q_perp
                    bessel_term = np.square(special.jv(_n, bessel_arg))
                    decay = np.exp(-mu * (nodes - 1.0))

                    if _n != 0:
                        # Chi_xx.
                        samples = bessel_term * decay
                        contribution_coarse[0] += np.sum(
                            samples[gauss_mask] * gauss_weights
                        )
                        contribution_fine[0] += np.sum(samples * weights)

                        # Chi_xz.
                        samples = q_parallel * bessel_term * decay
                        contribution_coarse[2] += np.sum(
                            samples[gauss_mask] * gauss_weights
                        )
                        contribution_fine[2] += np.sum(samples * weights)

                    # Chi_zz.
                    samples = q_parallel * q_parallel * bessel_term * decay
                    contribution_coarse[5] += np.sum(
                        samples[gauss_mask] * gauss_weights
                    )
                    contribution_fine[5] += np.sum(samples * weights)

                # for g0, g1 in [
                #     gamma_interval_jn_dx2[0],
                #     gamma_interval_jn_dx2[n_jn_dx2 - 1]
                # ]:

                # (Jn')^2 terms.
                for g0, g1 in pairwise(gamma_interval_jn_dx2[:n_jn_dx2]):
                    nodes, weights, gauss_mask, gauss_weights = (
                        gauss_kronrod_nodes_weights(g0, g1, order)
                    )

                    # Critical pitch angle.
                    q = np.sqrt(nodes * nodes - 1.0)
                    tau_c = (nodes - _n * y) / (n_parallel * q)

                    q_perp = q * np.sqrt(np.clip(1 - tau_c * tau_c, 0.0, None))
                    bessel_arg = nu_perp * q_perp
                    bessel_term = np.square(special.jvp(_n, bessel_arg, n=1))
                    decay = np.exp(-mu * (nodes - 1.0))

                    # Chi_yy.
                    samples = q_perp * q_perp * bessel_term * decay
                    contribution_coarse[3] += np.sum(
                        samples[gauss_mask] * gauss_weights
                    )
                    contribution_fine[3] += np.sum(samples * weights)

                    # import matplotlib.pyplot as plt
                    # plt.plot(nodes, samples)
                    # plt.show()

                # for g0, g1 in [
                #     gamma_interval_jn_jn_dx[0],
                #     gamma_interval_jn_jn_dx[n_jn_jn_dx - 1]
                # ]:

                # Jn * Jn' terms.
                for g0, g1 in pairwise(gamma_interval_jn_jn_dx[:n_jn_jn_dx]):
                    nodes, weights, gauss_mask, gauss_weights = (
                        gauss_kronrod_nodes_weights(g0, g1, order)
                    )

                    # Critical pitch angle.
                    q = np.sqrt(nodes * nodes - 1.0)
                    tau_c = (nodes - _n * y) / (n_parallel * q)

                    q_parallel = q * tau_c
                    q_perp = q * np.sqrt(np.clip(1 - tau_c * tau_c, 0.0, None))
                    bessel_arg = nu_perp * q_perp
                    bessel_term = special.jv(_n, bessel_arg) * special.jvp(
                        _n, bessel_arg, n=1
                    )
                    decay = np.exp(-mu * (nodes - 1.0))

                    if _n != 0:
                        # Chi_xy.
                        samples = bessel_term * decay
                        contribution_coarse[1] += np.sum(
                            samples[gauss_mask] * gauss_weights
                        )
                        contribution_fine[1] += np.sum(samples * weights)

                    # Chi_yz.
                    samples = q_parallel * q_perp * bessel_term * decay
                    contribution_coarse[4] += np.sum(
                        samples[gauss_mask] * gauss_weights
                    )
                    contribution_fine[4] += np.sum(samples * weights)

                # Check if we can improve precision with a finer
                # integration scheme.
                max_error = np.max(
                    abs(contribution_coarse - contribution_fine)
                )

                if max_error < atol:
                    reached_tolerance = True
                    break

            if not reached_tolerance:
                logger.warning(
                    "Gauss quadrature did not converge for harmonic %s: "
                    "error %s > atol %s",
                    _n,
                    max_error,
                    atol,
                )

            return_array[0] += _n * _n * contribution_fine[0]
            return_array[1] += _n * contribution_fine[1]
            return_array[2] += _n * contribution_fine[2]
            return_array[3] += contribution_fine[3]
            return_array[4] += contribution_fine[4]
            return_array[5] += contribution_fine[5]

        # Multiply by local prefactor.
        return_array[0] *= inv_nu_perp2
        return_array[1] *= -1.0j * inv_nu_perp
        return_array[2] *= -inv_nu_perp
        return_array[4] *= 1.0j

        # Multiply by global prefactor.
        return_array *= (
            0.5 * np.pi * x * mu * mu / (n_parallel * special.kve(2, mu))
        )

        return return_array

        # Calculate critical pitch for cyclotron resonance.
        # NOTE: h is a frequency scaling parameter used to calculate
        # hermitian part. It will be 1.0 unless we are in that routine.
        tau_c = (gamma * h - _n * Y) / (n_parallel * normalised_momentum)

        # Integrals over terms ~ Jn^2 (chi_xx, chi_xz, chi_zz).
        out_of_range = False
        zero_1 = gamma_0

        for idx_zero in range(number_zeros_jn):
            # Calculate approximate q_perp value of next zero of Jn.
            zero_q_perp = np.pi * inv_nu_perp * (idx_zero + 0.5 * n - 0.25)

            # Calculate equivalent gamma factor on resonance curve.
            # gamma is solution of _a * g**2 + _b * g - _c = 0
            _a = delta
            _b = 2.0 * ny
            _c = n_parallel2 * (1.0 + zero_q_perp * zero_q_perp) + ny2

            # Zero only on resonance curve if descriminant is positive.
            _f = _b * _b + 4.0 * _a * _c

            if _f > 0.0:
                pass
            else:
                zero_2 = gamma_max
                out_of_range = True

            # If |n_parallel| > 1 q_perp increases monotonically along
            # resonance curve. Otherwise, q_perp has a maximum and each
            # zero is encountered twice.
            # Tricky tricky!

            zero_2 = np.sqrt(max(0.0, 1.0 + bessel_arg_to_gamma * bessel_arg))

            # If zero is less than integration lower bound then continue.
            if zero_2 <= zero_1:
                continue

            # If zero is outside integration bound then truncate.
            if zero_2 >= gamma_max:
                zero_2 = gamma_max
                out_of_range = True

            # Calculate Gausss-Legendre nodes for this interval.
            samples.fill(0.0)

            cls._fill_leggauss_nodes_weights(zero_1, zero_2, nodes, weights)

            # Calculate critical pitch for cyclotron resonance.
            gamma = nodes
            normalised_momentum[:] = np.sqrt(nodes * nodes - 1.0)

            # Calculate bessel function.
            q_parallel[:] = normalised_momentum * tau_c
            q_perp[:] = normalised_momentum * np.sqrt(1 - tau_c * tau_c)
            bessel_term[:] = np.square(special.jv(n, nu_perp * q_perp))
            # bessel_j_dx[:] = special.jvp(n, bessel_arg, n=1)

            if n != 0.0:
                # Calculate chi_xx.
                samples[:] = bessel_term
                return_array[0] += n * n * np.sum(samples * weights)

                # Calculate chi_xz.
                samples[:] = q_parallel * bessel_term
                return_array[2] += n * np.sum(samples * weights)

            # Calculate chi_zz.
            samples[:] = q_parallel * q_parallel * bessel_term
            return_array[5] += np.sum(samples * weights)

            # Set next lower integration bound to be current upper bound.
            zero_1 = zero_2

            # If next zero outside of integration bounds then break.
            if out_of_range:
                break

        # Integrals over terms ~ Jn'^2 (chi_yy).

        # Integrals over terms ~ Jn * Jn' (chi_xy, chi_yz).

        q_perp = q[in_resonance] * np.sqrt(1 - np.square(tau_c[in_resonance]))
        q_par = q[in_resonance] * tau_c[in_resonance]

        # Calculate numerator at critical pitch
        samples.fill(0.0)

        # BesselJ.bessel(bessel_arg, n, return_array=bessel_j)
        # BesselJ.bessel_dx(bessel_arg, n, return_array=bessel_j_dx)

        bessel_arg = nu_perp * q_perp
        bessel_j[in_resonance] = special.jv(_n, bessel_arg)
        bessel_j_dx[in_resonance] = special.jvp(_n, bessel_arg, n=1)

        if _n != 0:
            # Multiplied by n so no need to calculate if n = 0.
            samples[0, in_resonance] = (
                bessel_j[in_resonance] * bessel_j[in_resonance]
            )
            samples[1, in_resonance] = (
                bessel_j[in_resonance] * bessel_j_dx[in_resonance]
            )
            samples[2, in_resonance] = q_par * samples[0, in_resonance]

        samples[3, in_resonance] = np.square(
            q_perp * bessel_j_dx[in_resonance]
        )
        samples[4, in_resonance] = q_perp * q_par * samples[1, in_resonance]
        samples[5, in_resonance] = np.square(q_par) * samples[0, in_resonance]

        # Multiply by exponential decay.
        for k in range(6):
            samples[k, :] *= np.exp(-nodes)

        # Perform integration over normalised gamma.
        if _n != 0:
            return_array[0] += _n * _n * np.sum(samples[0] * weights)
            return_array[1] += _n * np.sum(samples[1] * weights)
            return_array[2] += _n * np.sum(samples[2] * weights)

        return_array[3] += np.sum(samples[3] * weights)
        return_array[4] += np.sum(samples[4] * weights)
        return_array[5] += np.sum(samples[5] * weights)

        # import matplotlib.pyplot as plt

        # _g = np.linspace(1.00001, gamma.max(), 101)
        # _q = np.sqrt(_g * _g - 1)
        # _tau_c = (_g - _n * Y) / (n_parallel * _q)
        # _yy = (
        #     np.exp(-mu * (_g - 1))
        #     * np.square(_q * _tau_c)
        #     * np.square(
        #         special.jv(
        #             _n, nu_perp * _q
        #             * np.sqrt(1 - np.clip(_tau_c * _tau_c, 0.0, 1.0))
        #         )
        #     )
        # )

        # fig, ax = plt.subplots()
        # ax.plot(_g, _yy, color="black")
        # ax.scatter(gamma, samples[5], color="red", marker="x")
        # ax.set_yscale('log')
        # plt.show()

        # Multiply by local prefactor.
        return_array[0] *= inv_nu_perp2
        return_array[1] *= -1.0j * inv_nu_perp
        return_array[2] *= -inv_nu_perp
        return_array[4] *= 1.0j

        # Multiply by global prefactor.
        return_array *= (
            h * 0.5 * np.pi * X * mu / (n_parallel * special.kve(2, mu))
        )

        return return_array

    @classmethod
    def _chi_precursor_hermitian(cls, q: FloatArray):
        """
        Hermitian part of conductivity elements.

        Calculated using Kramers-Kronig relation and Gauss Legendre quadrature.
        """
        # Generate return array.
        return_array = get_return_array(return_array, 6, ComplexType)
        return_array.fill(0.0)

    @classmethod
    def _chi_ah_precursor(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Anti-Hermitian part of linear electric susceptibility
        without prefactor X mu**2 / 2 k_2(mu).
        """
        # Generate return array.
        return_array = get_return_array(return_array, 6, ComplexType)
        return_array.fill(0.0)

        # Unpack arguments.
        x = q[Dimensions.IDX_X]
        y = q[Dimensions.IDX_Y]
        theta = q[Dimensions.IDX_THETA]
        n_perp = q[Dimensions.IDX_N_PERP]
        n_parallel = q[Dimensions.IDX_N_PARALLEL]

        if y == 0.0:
            # Unmagnetised.
            raise NotImplementedError

        nu_perp = n_perp / y
        prefactor = 0.0

        if n_parallel == 0.0:
            # Special case.
            # chi_xz and chi_yz are zero.
            raise NotImplementedError

        # General case.
        raise NotImplementedError

    @classmethod
    def susceptibility_tensor(cls, q: FloatArray):
        """ """
        x = q[Dimensions.IDX_X]
        mu = 1.0 / q[Dimensions.IDX_THETA]

        prefactor = -0.5 * x * mu * mu / special.kve(2, mu)

        # Calculate anti-hermitian components.
        chi_precursor_ah = cls._chi_precursor_antihermitian

        raise NotImplementedError

    @classmethod
    def susceptibility_tensor_dq(cls, q: FloatArray):
        """ """
        raise NotImplementedError

    @classmethod
    def susceptibility_tensor_dq_2(cls, q: FloatArray):
        """ """
        raise NotImplementedError

    @classmethod
    def find_root_n(cls, q: FloatArray, wave_mode: WaveMode) -> FloatType:
        """
        Find magnitude of refractive index N that sets the determinant of the
        dispersion tensor to zero, preserving the relative sizes of N_perp
        and N_parallel.
        """
        raise NotImplementedError

    @classmethod
    def find_root_n_perp(cls, q: FloatArray, wave_mode: WaveMode) -> FloatType:
        """
        Find perpendicular refractive index N that sets the determinant of the
        dispersion tensor to zero with the initial value of N_parallel fixed.
        """
        raise NotImplementedError
