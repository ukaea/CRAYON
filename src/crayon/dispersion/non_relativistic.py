"""
Non-relativistic kinetic dispersion tensor model.
"""

# Standard imports
import logging

# Third party imports
import numpy as np
from scipy import optimize, special

# Local imports
from crayon.calculus import (
    first_derivative,
    mirror_upper_triangular_to_lower_triangular,
    second_derivative,
)
from crayon.dispersion.base import (
    DispersionModel,
    DispersionType,
    SusceptibilityCache,
)
from crayon.dispersion.cold import ColdDispersion
from crayon.shared.constants import SQRT_PI, WaveMode
from crayon.shared.data_structures import Result
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import (
    BooleanArray,
    ComplexArray,
    ComplexType,
    FloatArray,
    FloatType,
)

logger = logging.getLogger(__name__)

_Y = Dimensions.IDX_Y
_THETA = Dimensions.IDX_THETA
_N_PERP = Dimensions.IDX_N_PERP
_N_PARALLEL = Dimensions.IDX_N_PARALLEL


class BesselIExp:
    """
    Functions for calculation of modified Bessel functions of 1st kind and
    derivatives with respect to argument.

    Methods
    -------
    flr
        Calculate finite Larmor radius parameter lambda.
    flr_dq
        Calculate first derivative of finite Larmor radius parameter lambda
        with respect to q.
    flr_dq2
        Calculate second derivative of finite Larmor radius parameter lambda
        with respect to q.
    bessel_exp
        Calculate value and derivatives of exponentially scaled modified
        Bessel functions of the first kind.
    """

    __slots__ = ()

    @classmethod
    def flr(cls, q: FloatArray) -> float:
        """
        Calculate finite Larmor radius parameter lambda.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
            Y must be non-zero.

        Returns
        -------
        lambda : float
            Finite Larmor radius parameter.

        Raises
        ------
        ValueError
            Normalised magnetic field strength is zero.
        """
        y = q[_Y]
        theta = q[_THETA]
        n_perp = q[_N_PERP]

        if abs(y) > 0.0:
            return theta * np.square(n_perp / y)

        raise ValueError("y == 0.0")

    @classmethod
    def flr_dq(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        First derivative of finite Larmor radius parameter with respect to
        Hamiltonian arguments q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
            Y must be non-zero.

        Returns
        -------
        lambda_dq : np.array[float]
            First derivative of finite Larmor radius parameter.

        Raises
        ------
        ValueError
            Normalised magnetic field strength is zero.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.q.size,),
            FloatType,
        )

        # Unpack q2.
        y = q[_Y]
        theta = q[_THETA]
        n_perp = q[_N_PERP]

        if abs(y) > 0.0:
            y2 = y * y
            y3 = y * y2
            n_perp2 = n_perp * n_perp

            return_array.fill(0.0)

            return_array[_Y] = -2.0 * theta * n_perp2 / y3
            return_array[_THETA] = n_perp2 / y2
            return_array[_N_PERP] = 2.0 * theta * n_perp / y2

            return return_array

        raise ValueError("y == 0.0")

    @classmethod
    def flr_dq2(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        Second derivative of finite Larmor radius parameter with respect to
        Hamiltonian arguments q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
            Y must be non-zero.

        Returns
        -------
        lambda_dq2 : np.array[float]
            Second derivative of finite Larmor radius parameter.

        Raises
        ------
        ValueError
            Normalised magnetic field strength is zero.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.q.size, Dimensions.q.size),
            FloatType,
        )

        # Unpack q2.
        y = q[_Y]
        theta = q[_THETA]
        n_perp = q[_N_PERP]

        if abs(y) > 0.0:
            y2 = y * y
            y3, y4 = y * y2, y2 * y2
            n_perp2 = n_perp * n_perp

            return_array.fill(0.0)

            return_array[_Y, _Y] = 6.0 * theta * n_perp2 / y4
            return_array[_Y, _THETA] = -2 * n_perp2 / y3
            return_array[_Y, _N_PERP] = -4 * n_perp * theta / y3
            return_array[_THETA, _N_PERP] = 2 * n_perp / y2
            return_array[_N_PERP, _N_PERP] = 2 * theta / y2

            # Second partial derivatives commute.
            mirror_upper_triangular_to_lower_triangular(return_array)

            return return_array

        raise ValueError("y == 0.0")

    @classmethod
    def bessel_exp(
        cls,
        x: float,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        derivative: int = 0,
        bessel_exp_m1: FloatArray = None,
    ) -> FloatArray:
        """
        Return exponentially scaled modified Bessel function of the first kind
        e^-x I_k(x) for k = 0 to nmax (nmax total harmonics including 0 i.e.
        nmax is excluded) for the requested derivative order wrt argument.

        Parameters
        ----------
        x : float
            Argument of Bessel function
        nmax : int
            Number of harmonics to calculate (including zero so highest
            harmonic is n = nmax - 1).
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        derivative : int, optional
            Order of derivative to evaluate. Must be > 0.
        bessel_exp_m1 : np.array[float], optional
            Function evaluated for 1 order lower derivative. If not provided
            it will be calculated.

        Returns
        -------
        bessel_exp
            Requested derivative of exponentially scaled modified Bessel
            function of the first kind.

        Raises
        ------
        ValueError
            Requested derivative is negative.
        """
        return_array = get_return_array(return_array, (nmax,), FloatType)

        if derivative == 0:
            # ive complains x is complex but return_array is real.
            special.ive(np.arange(nmax), x.real, out=return_array)
            return return_array

        if derivative > 0:
            # Derivative. Only depends on 1 order lower derivative so all
            # derivatives use the same formula.
            if bessel_exp_m1 is None:
                ie_m1 = cls.bessel_exp(x, nmax + 1, derivative=derivative - 1)
            elif bessel_exp_m1.size < nmax + 1:
                raise ValueError(
                    "bessel_exp_m1 must be evaluated to >= nmax + 1."
                )
            else:
                ie_m1 = bessel_exp_m1

            return_array[0] = ie_m1[1] - ie_m1[0]
            return_array[1:nmax] = (
                0.5 * (ie_m1[: nmax - 1] + ie_m1[2 : nmax + 1]) - ie_m1[1:nmax]
            )

            return return_array

        raise ValueError(derivative)


SUM: int = 0
DIFF: int = 1
PLUS: int = 0
MINUS: int = 1
RESONANCE: int = 0
COLLISIONAL: int = 1


class PlasmaZ:
    """
    Functions for calculation of plasma dispersion function and derivatives
    with respect to argument.

    Methods
    -------
    doppler
        Calculate doppler parameter d.
    doppler_dq
        Caclculate first derivative of doppler parameter with respect to q.
    doppler_dq2
        Caclculate second derivative of doppler parameter with respect to q.
    p
        Calculate dependent variables for the plasma dispersion function
        p = [Y, doppler].
    p_dq
        Calculate first derivative of p with respect to Hamiltonian arguments
        q.
    p_dq2
        Calculate second derivative of p with respect to Hamiltonian arguments
        q.
    zeta_expansion_masks
        Calculate boolean masks for different asymptotic expansion cases of
        the plasma dispersion function.
    harmonic_gap
        Difference between unity and harmonic of resonant magnetic field
        1 - n * Y.
    z_norm_real
        Calculate real part of plasma dispersion function k-th derivative
        normalised to doppler**(k+1).
    z_norm_imag
        Calculate imaginary part of plasma dispersion function normalised to
        doppler parameter for both resonant and collisional damping.
    sum_diff
        Construct value summed and differenced over opposite sign harmonics.
        For the zero-eth harmonic the value at zero is returned.
    z_dxk_norm_real_sum
        Convenience method for calculating z_norm_real and its derivatives
        summed and differenced over opposite sign harmonics. Not efficient.
    z_dxk_norm_imag_sum
        Convenience method for calculating z_norm_imag and its derivatives
        summed and differenced over opposite sign harmonics. Not efficient.
    """

    __slots__ = ()

    IDX_Y = 0
    IDX_DOPPLER = 1

    @classmethod
    def doppler(cls, q: FloatArray):
        """
        Calculate Doppler parameter d.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].

        Returns
        -------
        doppler : float
            Doppler parameter.
        """
        theta = q[_THETA]
        n_parallel = q[_N_PARALLEL]

        # Non-relativistic thermal velocity.
        beta = np.sqrt(2 * theta)

        return beta * n_parallel

    @classmethod
    def doppler_dq(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        Calculate first derivative of Doppler parameter d with respect to q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        doppler_dq : float
            First derivative of Doppler parameter.
        """
        return_array = get_return_array(
            return_array, (Dimensions.q.size,), FloatType
        )

        theta = q[_THETA]
        n_parallel = q[_N_PARALLEL]

        # Non-relativistic thermal velocity.
        beta = np.sqrt(2 * theta)

        return_array.fill(0.0)
        return_array[_THETA] = n_parallel / beta
        return_array[_N_PARALLEL] = beta

        return return_array

    @classmethod
    def doppler_dq2(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        Calculate second derivative of Doppler parameter d with respect to q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        doppler_dq2 : float
            Second derivative Doppler parameter.
        """
        return_array = get_return_array(
            return_array, (Dimensions.q.size, Dimensions.q.size), FloatType
        )

        theta = q[Dimensions.IDX_THETA]
        n_parallel = q[_N_PARALLEL]

        # Non-relativistic thermal velocity.
        beta = np.sqrt(2 * theta)

        return_array.fill(0.0)
        return_array[_THETA, _THETA] = -n_parallel / (beta * beta * beta)
        return_array[_THETA, _N_PARALLEL] = 1.0 / beta

        # Second partial derivatives commute.
        mirror_upper_triangular_to_lower_triangular(return_array)

        return return_array

    @classmethod
    def p(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        Calculate dependent variables for the plasma dispersion function
        p = [Y, doppler].

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        p : float
            Dependent variables for the plasma dispersion function.
        """
        return_array = get_return_array(return_array, (2,), FloatType)

        return_array[cls.IDX_Y] = q[_Y]
        return_array[cls.IDX_DOPPLER] = cls.doppler(q)

        return return_array

    @classmethod
    def p_dq(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        Calculate first derivative of p with respect to Hamiltonian arguments
        q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        p_dq : float
            First derivative of p with respect to Hamiltonian arguments q.
        """
        return_array = get_return_array(
            return_array, (2, Dimensions.q.size), FloatType
        )

        return_array.fill(0.0)
        return_array[cls.IDX_Y, _Y] = 1.0
        return_array[cls.IDX_DOPPLER, :] = cls.doppler_dq(q)

        return return_array

    @classmethod
    def p_dq2(cls, q: FloatArray, /, *, return_array: FloatArray = None):
        """
        Calculate second derivative of p with respect to Hamiltonian arguments
        q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        p_dq2 : float
            Second derivative of p with respect to Hamiltonian arguments q.
        """
        return_array = get_return_array(
            return_array, (2, Dimensions.q.size, Dimensions.q.size), FloatType
        )

        return_array.fill(0.0)
        return_array[cls.IDX_DOPPLER, :, :] = cls.doppler_dq2(q)

        return return_array

    @classmethod
    def zeta_expansion_masks(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        min_ratio: float,
    ) -> tuple[BooleanArray, BooleanArray, BooleanArray]:
        """
        Calculate boolean masks for different asymptotic expansion cases of
        the plasma dispersion function.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        min_ratio : float
            Minimum ratio of harmonic gap and doppler beyond which a large
            argument expansion is used.

        Returns
        -------
        zeta_is_zero : np.array[bool]
            Mask for the harmonic gap being identically zero.
        zeta_is_big : np.array[bool]
            Mask for the argument to be sufficiently large to use
            asymptotically large value.
        other : np.array[bool]
            Any value which didn't fit into either of the previous cases.

        Notes
        -----
        As the order of derivative of the plasma dispersion function increases
        the large argument expansion will have to be used for smaller arguments
        due to numerical error.
        """
        zeta_is_zero = np.isclose(harmonic_gap, 0.0, atol=1e-8, rtol=0.0)
        zeta_is_big = abs(harmonic_gap) > min_ratio * abs(doppler)
        other = np.logical_and(
            np.logical_not(zeta_is_zero), np.logical_not(zeta_is_big)
        )

        return zeta_is_zero, zeta_is_big, other

    @classmethod
    def harmonic_gap(
        cls,
        normalised_magnetic_field_strength: float,
        nmax: int,
        /,
        *,
        plus: bool,
    ):
        """
        Difference between unity and harmonic of resonant magnetic field
        1 - n * Y. Strong damping when is small compared to doppler.

        Parameters
        ----------
        normalised_magnetic_field_strength : float
            Normalised magnetic field strength aka Stix Y.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        plus : bool
            If True, calculate for positive harmonic number. Otherwise
            calculate for negative harmonic number.

        Returns
        -------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y.
        """
        ny = np.arange(nmax) * normalised_magnetic_field_strength

        if plus:
            return 1 - ny

        return 1 + ny

    @classmethod
    def _z_norm_real_0(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
    ):
        """
        Calculate real part of plasma dispersion function normalised to
        doppler.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 1.0e7
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = 0.0

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = -1 / harmonic_gap[zeta_is_big]

        if any(other):
            # Real part is -2 * dawson function
            doppler_inv = 1.0 / doppler
            zeta = harmonic_gap[other] * doppler_inv
            return_array[other] = -2.0 * doppler_inv * special.dawsn(zeta)

    @classmethod
    def _z_norm_real_1(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
        /,
        *,
        z_norm_real_0: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function 1st derivative
        normalised to doppler**2.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        z_norm_real_0 : FloatArray, optional
            Function value.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 1.0e7
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = -2.0 / doppler

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = 1.0 / np.square(
                harmonic_gap[zeta_is_big]
            )

        if any(other):
            if z_norm_real_0 is None:
                _z_norm_real_m1 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=0
                )
            else:
                _z_norm_real_m1 = z_norm_real_0[other]

            doppler_inv2 = 1.0 / (doppler * doppler)
            return_array[other] = -(2.0 * doppler_inv2) * (
                1.0 + harmonic_gap[other] * _z_norm_real_m1
            )

    @classmethod
    def _z_norm_real_2(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
        /,
        *,
        z_norm_real_0: FloatArray = None,
        z_norm_real_1: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function 2nd derivative
        normalised to doppler**3.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        z_norm_real_0 : FloatArray, optional
            Function value normalised to doppler.
        z_norm_real_1 : FloatArray, optional
            First derivative normalised to doppler**2.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 500.0
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = 0.0

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = -2.0 / np.power(
                harmonic_gap[zeta_is_big], 3
            )

        if any(other):
            if z_norm_real_0 is None:
                _z_norm_real_m2 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=0
                )
            else:
                _z_norm_real_m2 = z_norm_real_0[other]

            if z_norm_real_1 is None:
                _z_norm_real_m1 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=1
                )
            else:
                _z_norm_real_m1 = z_norm_real_1[other]

            doppler_inv2 = 1.0 / (doppler * doppler)
            return_array[other] = -(2.0 * doppler_inv2) * (
                1.0 * _z_norm_real_m2 + harmonic_gap[other] * _z_norm_real_m1
            )

    @classmethod
    def _z_norm_real_3(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
        /,
        *,
        z_norm_real_1: FloatArray = None,
        z_norm_real_2: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function 3rd derivative
        normalised to doppler**4.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        z_norm_real_1 : FloatArray, optional
            First derivative normalised to doppler**2.
        z_norm_real_2 : FloatArray, optional
            Second derivative normalised to doppler**3.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 200.0
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = 0.0

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = 6.0 / np.power(
                harmonic_gap[zeta_is_big], 4
            )

        if any(other):
            if z_norm_real_1 is None:
                _z_norm_real_m2 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=1
                )
            else:
                _z_norm_real_m2 = z_norm_real_1[other]

            if z_norm_real_2 is None:
                _z_norm_real_m1 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=2
                )
            else:
                _z_norm_real_m1 = z_norm_real_2[other]

            doppler_inv2 = 1.0 / (doppler * doppler)
            return_array[other] = -(2.0 * doppler_inv2) * (
                2.0 * _z_norm_real_m2 + harmonic_gap[other] * _z_norm_real_m1
            )

    @classmethod
    def _z_norm_real_4(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
        /,
        *,
        z_norm_real_2: FloatArray = None,
        z_norm_real_3: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function 4th derivative
        normalised to doppler**5.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        z_norm_real_2 : FloatArray, optional
            Second derivative normalised to doppler**3.
        z_norm_real_3 : FloatArray, optional
            Third derivative normalised to doppler**4.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 100.0
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = 0.0

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = -24.0 / np.power(
                harmonic_gap[zeta_is_big], 5
            )

        if any(other):
            if z_norm_real_2 is None:
                _z_norm_real_m2 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=2
                )
            else:
                _z_norm_real_m2 = z_norm_real_2[other]

            if z_norm_real_3 is None:
                _z_norm_real_m1 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=3
                )
            else:
                _z_norm_real_m1 = z_norm_real_3[other]

            doppler_inv2 = 1.0 / (doppler * doppler)
            return_array[other] = -(2.0 * doppler_inv2) * (
                3.0 * _z_norm_real_m2 + harmonic_gap[other] * _z_norm_real_m1
            )

    @classmethod
    def _z_norm_real_5(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
        /,
        *,
        z_norm_real_3: FloatArray = None,
        z_norm_real_4: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function 5th derivative
        normalised to doppler**6.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        z_norm_real_3 : FloatArray, optional
            Third derivative normalised to doppler**4.
        z_norm_real_4 : FloatArray, optional
            Fourth derivative normalised to doppler**5.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 50.0
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = 0.0

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = 120.0 / np.power(
                harmonic_gap[zeta_is_big], 6
            )

        if any(other):
            if z_norm_real_3 is None:
                _z_norm_real_m2 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=3
                )
            else:
                _z_norm_real_m2 = z_norm_real_3[other]

            if z_norm_real_4 is None:
                _z_norm_real_m1 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=4
                )
            else:
                _z_norm_real_m1 = z_norm_real_4[other]

            doppler_inv2 = 1.0 / (doppler * doppler)
            return_array[other] = -(2.0 * doppler_inv2) * (
                4.0 * _z_norm_real_m2 + harmonic_gap[other] * _z_norm_real_m1
            )

    @classmethod
    def _z_norm_real_6(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        return_array: FloatArray,
        /,
        *,
        z_norm_real_4: FloatArray = None,
        z_norm_real_5: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function 5th derivative
        normalised to doppler**6.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        return_array : np.array[float]
            Array to store result. Must have same size as harmonic gap.
        z_norm_real_4 : FloatArray, optional
            Third derivative normalised to doppler**5.
        z_norm_real_5 : FloatArray, optional
            Fourth derivative normalised to doppler**6.
        """
        # Masks for expansion cases.
        zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
            harmonic_gap, doppler, 30.0
        )

        if any(zeta_is_zero):
            # Use small argument expansion.
            return_array[zeta_is_zero] = 0.0

        if any(zeta_is_big):
            # Use large argument expansion.
            return_array[zeta_is_big] = -720.0 / np.power(
                harmonic_gap[zeta_is_big], 7
            )

        if any(other):
            if z_norm_real_4 is None:
                _z_norm_real_m2 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=4
                )
            else:
                _z_norm_real_m2 = z_norm_real_4[other]

            if z_norm_real_5 is None:
                _z_norm_real_m1 = cls.z_norm_real(
                    harmonic_gap[other], doppler, derivative=5
                )
            else:
                _z_norm_real_m1 = z_norm_real_5[other]

            doppler_inv2 = 1.0 / (doppler * doppler)
            return_array[other] = -(2.0 * doppler_inv2) * (
                5.0 * _z_norm_real_m2 + harmonic_gap[other] * _z_norm_real_m1
            )

    @classmethod
    def z_norm_real(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        /,
        *,
        derivative: int = 0,
        z_norm_real_m1: FloatArray = None,
        z_norm_real_m2: FloatArray = None,
    ):
        """
        Calculate real part of plasma dispersion function k-th derivative
        normalised to doppler**(k+1).

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        derivative : int, optional
            Derivative order to evaluate.
        z_norm_real_m1 : FloatArray, optional
            Function value evaluated at derivative order 1 smaller than
            requested (only used if derivative >= 1).
        z_norm_real_m2 : FloatArray, optional
            Function value evaluated at derivative order 2 smaller than
            requested (only used if derivative >= 2).

        Returns
        -------
        z_norm_real : np.array[float]
            Real part of plasma dispersion function k-th derivative normalised
            to doppler**(k+1).

        Raises
        ------
        ValueError
            Requested derivative < 0 or > 6.
        """
        return_array = np.empty_like(harmonic_gap)

        if derivative == 0:
            cls._z_norm_real_0(harmonic_gap, doppler, return_array)

            return return_array

        if derivative == 1:
            cls._z_norm_real_1(
                harmonic_gap,
                doppler,
                return_array,
                z_norm_real_0=z_norm_real_m1,
            )

            return return_array

        if derivative == 2:  # noqa: PLR2004
            cls._z_norm_real_2(
                harmonic_gap,
                doppler,
                return_array,
                z_norm_real_0=z_norm_real_m2,
                z_norm_real_1=z_norm_real_m1,
            )

            return return_array

        if derivative == 3:  # noqa: PLR2004
            cls._z_norm_real_3(
                harmonic_gap,
                doppler,
                return_array,
                z_norm_real_1=z_norm_real_m2,
                z_norm_real_2=z_norm_real_m1,
            )

            return return_array

        if derivative == 4:  # noqa: PLR2004
            cls._z_norm_real_4(
                harmonic_gap,
                doppler,
                return_array,
                z_norm_real_2=z_norm_real_m2,
                z_norm_real_3=z_norm_real_m1,
            )

            return return_array

        if derivative == 5:  # noqa: PLR2004
            cls._z_norm_real_5(
                harmonic_gap,
                doppler,
                return_array,
                z_norm_real_3=z_norm_real_m2,
                z_norm_real_4=z_norm_real_m1,
            )

            return return_array

        if derivative == 6:  # noqa: PLR2004
            cls._z_norm_real_6(
                harmonic_gap,
                doppler,
                return_array,
                z_norm_real_4=z_norm_real_m2,
                z_norm_real_5=z_norm_real_m1,
            )

            return return_array

        raise ValueError(derivative)

    @classmethod
    def z_norm_imag(
        cls,
        harmonic_gap: FloatArray,
        doppler: float,
        normalised_collision_rate: float,
        /,
        *,
        derivative: int = 0,
        z_norm_real_p1: FloatArray = None,
    ):
        """
        Calculate imaginary part of plasma dispersion function normalised to
        doppler parameter for both resonant and collisional damping.

        Parameters
        ----------
        harmonic_gap : np.array[float]
            Difference between unity and harmonic of resonant magnetic field
            1 - n * Y. Strong damping when is small compared to doppler.
        doppler : float
            Doppler parameter.
        normalised_collision_rate : float
            Normalised electron-ion collision frequency aka Stix Z.
        derivative : int, optional
            Derivative order to evaluate.
        z_norm_real_p1 : FloatArray, optional
            z_norm_real evaluated at derivative order 1 larger than requested.

        Returns
        -------
        z_norm_imag : np.array[float]
            Imaginary part of plasma dispersion function normalised to doppler
            parameter for both resonant and collisional damping.

        Raises
        ------
        ValueError
            Requested derivative < 0 or > 2.
        """
        return_array = np.empty((2, harmonic_gap.size))

        doppler_sign = np.sign(doppler)

        if derivative == 0:
            zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
                harmonic_gap, doppler, 1.0e7
            )

            if any(zeta_is_zero):
                return_array[RESONANCE, zeta_is_zero] = (
                    doppler_sign * SQRT_PI / doppler
                )
                return_array[COLLISIONAL, zeta_is_zero] = 0.0

            if any(zeta_is_big):
                return_array[RESONANCE, zeta_is_big] = 0.0
                return_array[COLLISIONAL, zeta_is_big] = (
                    normalised_collision_rate
                    / np.square(harmonic_gap[zeta_is_big])
                )

            if any(other):
                doppler_inv = 1.0 / doppler
                zeta = harmonic_gap[other] * doppler_inv
                return_array[RESONANCE, other] = (
                    SQRT_PI
                    * doppler_inv
                    * doppler_sign
                    * np.exp(-np.square(zeta))
                )

                if z_norm_real_p1 is None:
                    _z_norm_real_p1 = cls.z_norm_real(
                        harmonic_gap[other], doppler, derivative=1
                    )
                else:
                    _z_norm_real_p1 = z_norm_real_p1[other]

                return_array[COLLISIONAL, other] = (
                    normalised_collision_rate * _z_norm_real_p1
                )

            return return_array

        if derivative == 1:
            zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
                harmonic_gap, doppler, 1.0e7
            )

            if any(zeta_is_zero):
                doppler_inv2 = 1.0 / (doppler * doppler)
                return_array[RESONANCE, zeta_is_zero] = 0.0
                return_array[COLLISIONAL, zeta_is_zero] = (
                    -2.0 * normalised_collision_rate * doppler_inv2
                )

            if any(zeta_is_big):
                return_array[RESONANCE, zeta_is_big] = 0.0
                return_array[COLLISIONAL, zeta_is_big] = (
                    -2.0
                    * normalised_collision_rate
                    / np.power(harmonic_gap[zeta_is_big], 3)
                )

            if any(other):
                doppler_inv = 1.0 / doppler
                doppler_inv2 = doppler_inv * doppler_inv

                zeta = harmonic_gap[other] * doppler_inv
                return_array[RESONANCE, other] = (
                    SQRT_PI
                    * doppler_sign
                    * doppler_inv2
                    * -2.0
                    * zeta
                    * np.exp(-zeta * zeta)
                )

                if z_norm_real_p1 is None:
                    _z_norm_real_p1 = cls.z_norm_real(
                        harmonic_gap[other], doppler, derivative=2
                    )
                else:
                    _z_norm_real_p1 = z_norm_real_p1[other]

                return_array[COLLISIONAL, other] = (
                    normalised_collision_rate * _z_norm_real_p1
                )

            return return_array

        if derivative == 2:  # noqa: PLR2004
            zeta_is_zero, zeta_is_big, other = cls.zeta_expansion_masks(
                harmonic_gap, doppler, 1.0e7
            )

            if any(zeta_is_zero):
                doppler_inv3 = 1 / (doppler * doppler * doppler)

                return_array[RESONANCE, zeta_is_zero] = (
                    -2.0 * SQRT_PI * doppler_sign * doppler_inv3
                )
                return_array[COLLISIONAL, zeta_is_zero] = 0.0

            if any(zeta_is_big):
                return_array[RESONANCE, zeta_is_big] = 0.0
                return_array[COLLISIONAL, zeta_is_big] = (
                    normalised_collision_rate
                    * 6.0
                    / np.power(harmonic_gap[zeta_is_big], 4)
                )

            if any(other):
                doppler_inv = 1 / doppler
                doppler_inv3 = doppler_inv * doppler_inv * doppler_inv

                zeta = harmonic_gap[other] * doppler_inv
                return_array[RESONANCE, other] = (
                    SQRT_PI
                    * doppler_sign
                    * doppler_inv3
                    * (4.0 * zeta * zeta - 2.0)
                    * np.exp(-zeta * zeta)
                )

                if z_norm_real_p1 is None:
                    _z_norm_real_p1 = cls.z_norm_real(
                        harmonic_gap[other], doppler, derivative=3
                    )
                else:
                    _z_norm_real_p1 = z_norm_real_p1[other]

                return_array[COLLISIONAL, other] = (
                    normalised_collision_rate * _z_norm_real_p1
                )

            return return_array

        raise ValueError(derivative)

    @classmethod
    def sum_diff(
        cls,
        plus: FloatArray,
        minus: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Construct value summed and differenced over opposite sign harmonics.
        For the zero-eth harmonic the value at zero is returned.

        Parameters
        ----------
        plus : np.array[float]
            Value for positive harmonics.
        minus : np.array[float]
            Value for negative harmonics.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        sum_diff : np.array[float]
            Value summed and differenced over opposite sign harmonics.
        """
        return_array = get_return_array(
            return_array, (2, plus.size), FloatType
        )

        return_array[SUM, 0] = plus[0]
        return_array[SUM, 1:] = plus[1:] + minus[1:]
        return_array[DIFF, 0] = 0.0
        return_array[DIFF, 1:] = plus[1:] - minus[1:]

        return return_array

    @classmethod
    def z_dxk_norm_real_sum(
        cls,
        p: FloatArray,
        nmax: int,
        derivative: int,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convenience method for calculating z_norm_real and its derivatives
        summed and differenced over opposite sign harmonics. Not efficient.

        Parameters
        ----------
        p : np.array[float]
            Dependent variables for the plasma dispersion function.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        derivative : int
            Derivative order to evaluate.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        z_dxk_norm_real_sum : np.array[float]
            Derivative of z real normalised to doppler**(k + 1) summed and
            differenced over opposite sign harmonics.
        """
        return_array = get_return_array(return_array, (2, nmax), FloatType)

        y = p[PlasmaZ.IDX_Y]
        doppler = p[PlasmaZ.IDX_DOPPLER]

        harmonic_gap_plus = PlasmaZ.harmonic_gap(y, nmax, plus=True)
        z_dxk_norm_real_plus = PlasmaZ.z_norm_real(
            harmonic_gap_plus, doppler, derivative=derivative
        )

        harmonic_gap_minus = PlasmaZ.harmonic_gap(y, nmax, plus=False)
        z_dxk_norm_real_minus = PlasmaZ.z_norm_real(
            harmonic_gap_minus, doppler, derivative=derivative
        )

        cls.sum_diff(
            z_dxk_norm_real_plus,
            z_dxk_norm_real_minus,
            return_array=return_array,
        )

        return return_array

    @classmethod
    def z_dxk_norm_imag_sum(
        cls,
        p: FloatArray,
        normalised_collision_rate: float,
        nmax: int,
        derivative: int,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convenience method for calculating z_norm_imag and its derivatives
        summed and differenced over opposite sign harmonics. Not efficient.

        Parameters
        ----------
        p : np.array[float]
            Dependent variables for the plasma dispersion function.
        normalised_collision_rate : float
            Normalised electron-ion collision frequency aka Stix Z.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        derivative : int
            Derivative order to evaluate.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        z_dxk_norm_imag_sum : np.array[float]
            Derivative of z imag normalised to doppler**(k + 1) summed and
            differenced over opposite sign harmonics.
        """
        return_array = get_return_array(return_array, (2, 2, nmax), FloatType)

        y = p[PlasmaZ.IDX_Y]
        doppler = p[PlasmaZ.IDX_DOPPLER]

        harmonic_gap_plus = PlasmaZ.harmonic_gap(y, nmax, plus=True)
        z_dxk_norm_imag_plus = PlasmaZ.z_norm_imag(
            harmonic_gap_plus,
            doppler,
            normalised_collision_rate,
            derivative=derivative,
        )

        harmonic_gap_minus = PlasmaZ.harmonic_gap(y, nmax, plus=False)
        z_dxk_norm_imag_minus = PlasmaZ.z_norm_imag(
            harmonic_gap_minus,
            doppler,
            normalised_collision_rate,
            derivative=derivative,
        )

        cls.sum_diff(
            z_dxk_norm_imag_plus[RESONANCE, :],
            z_dxk_norm_imag_minus[RESONANCE, :],
            return_array=return_array[RESONANCE, :, :],
        )

        cls.sum_diff(
            z_dxk_norm_imag_plus[COLLISIONAL, :],
            z_dxk_norm_imag_minus[COLLISIONAL, :],
            return_array=return_array[COLLISIONAL, :, :],
        )

        return return_array


class NonRelativisticDispersion(DispersionModel):
    """
    Dispersion model accounting for thermal but non-relativistic thermal
    effects.

    Methods
    -------
    susceptibility_hermitian
        Calculate hermitian part of electric susceptibility.
    susceptibility_antihermitian_resonance
        Calculate anti-hermitian part of electric susceptibility associated
        with resonant absorption.
    susceptibility_antihermitian_collisional
        Calculate anti-hermitian part of electric susceptibility associated
        with collisional absorption.
    susceptibility_hermitian_dq
        Calculate first derivative of hermitian part of electric
        susceptibility with respect to q.
    susceptibility_hermitian_dq2
        Calculate second derivative of hermitian part of electric
        susceptibility with respect to q.
    calculate_susceptibility
        Calculate hermitian / anti-hermitian part of electric susceptibility
        maximally re-using shared elements.
    dispersion_tensor
        Calculate dispersion tensor.
    calculate_n
        Calculate magnitude of refractive index for possible wave mode
        preserving angle to magnetic field.
    calculate_n_perp
        Calculate perpendicular refractive index for possible wave for fix
        parallel refractive index.
    """

    __slots__ = ()

    dispersion_type = DispersionType.NON_RELATIVISTIC

    @classmethod
    def flr_term(
        cls,
        flr: FloatType,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        bessel_exp: FloatArray = None,
        bessel_exp_dx: FloatArray = None,
    ):
        """
        Calculate finite Larmor radius dependent terms for each 6 unique
        elements of the electric susceptibility.

        Parameters
        ----------
        flr : float
            Finite Larmor radius parameter.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        bessel_exp : np.array[float], optional
            Exponentially scaled modified Bessel function. If provided must
            have size >= nmax + 1.
        bessel_exp_dx : np.array[float], optional
            First derivative of exponentially scaled modified Bessel function
            If provided must have size >= nmax.

        Returns
        -------
        flr_term : np.array[float]
            Finite Larmor radius dependent terms of electric susceptibility.

        Raises
        ------
        ValueError
            bessel_exp provided and has size < nmax + 1.
            bessel_exp_dx provided and has size < nmax.
        """
        return_array = get_return_array(return_array, (6, nmax), FloatType)

        if bessel_exp is None:
            bessel_exp = BesselIExp.bessel_exp(flr, nmax + 1)
        elif bessel_exp.size < nmax + 1:
            raise ValueError(
                f"bessel_exp must have size >= {nmax + 1}: {bessel_exp.size}"
            )

        if bessel_exp_dx is None:
            bessel_exp_dx = BesselIExp.bessel_exp(
                flr, nmax, derivative=1, bessel_exp_m1=bessel_exp
            )
        elif bessel_exp_dx.size < nmax:
            raise ValueError(
                f"bessel_exp_dx must have size >= {nmax}: {bessel_exp_dx.size}"
            )

        _n = np.arange(nmax)

        ie = bessel_exp
        ie1 = bessel_exp_dx
        sqrt_2_flr = np.sqrt(2.0 * flr)

        return_array.fill(0.0)

        return_array[0, 1:] = (
            0.5 * _n[1:] * (ie[: nmax - 1] - ie[2 : nmax + 1])
        )
        return_array[1, 1:] = _n[1:] * ie1[1:nmax]
        return_array[2, 1:] = (
            0.25 * sqrt_2_flr * (ie[: nmax - 1] - ie[2 : nmax + 1])
        )
        return_array[3, :] = return_array[0, :] - 2.0 * flr * ie1[:nmax]
        return_array[4, :] = 0.5 * sqrt_2_flr * ie1[:nmax]
        return_array[5, :] = ie[:nmax]

        return return_array

    @classmethod
    def _flr_term_dflr(
        cls,
        flr: FloatType,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        bessel_exp_dx: FloatArray = None,
        bessel_exp_dx2: FloatArray = None,
        flr_term: FloatArray = None,
    ):
        """
        Calculate first derivative of finite Larmor radius dependent term with
        respect to flr for each 6 unique elements of the electric
        susceptibility.

        Parameters
        ----------
        flr : float
            Finite Larmor radius parameter.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        bessel_exp_dx : np.array[float], optional
            First derivative of exponentially scaled modified Bessel function.
            If provided must have size >= nmax + 1.
        bessel_exp_dx2 : np.array[float], optional
            Second derivative of exponentially scaled modified Bessel function.
            If provided must have size >= nmax.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.

        Returns
        -------
        flr_term_dflr : np.array[float]
            First derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to flr.

        Raises
        ------
        ValueError
            bessel_exp_dx provided and has size < nmax + 1.
            bessel_exp_dx2 provided and has size < nmax.
        """
        return_array = get_return_array(return_array, (6, nmax), FloatType)

        if bessel_exp_dx is None:
            bessel_exp_dx = BesselIExp.bessel_exp(flr, nmax + 1, derivative=1)
        elif bessel_exp_dx.size < nmax + 1:
            raise ValueError(
                f"bessel_exp_dx must have size >= {nmax + 1}: "
                f"{bessel_exp_dx.size}"
            )

        if bessel_exp_dx2 is None:
            bessel_exp_dx2 = BesselIExp.bessel_exp(
                flr, nmax, derivative=2, bessel_exp_m1=bessel_exp_dx
            )
        elif bessel_exp_dx2.size < nmax:
            raise ValueError(
                f"bessel_exp_dx2 must have size >= {nmax}: "
                f"{bessel_exp_dx2.size}"
            )

        if flr_term is None:
            flr_term = cls.flr_term(flr, nmax)

        return_array.fill(0.0)

        sqrt_2_flr = np.sqrt(2.0 * flr)
        inv_2_flr = 0.5 / flr

        _n = np.arange(nmax)
        ie1 = bessel_exp_dx
        ie2 = bessel_exp_dx2

        return_array[0, 1:] = (
            0.5 * _n[1:] * (ie1[: nmax - 1] - ie1[2 : nmax + 1])
        )
        return_array[1, 1:] = _n[1:] * ie2[1:nmax]
        return_array[2, 1:] = inv_2_flr * flr_term[
            2, 1:
        ] + 0.25 * sqrt_2_flr * (ie1[: nmax - 1] - ie1[2 : nmax + 1])
        return_array[3, :] = return_array[0, :] - 2.0 * (
            ie1[:nmax] + flr * ie2[:nmax]
        )
        return_array[4, :] = (
            inv_2_flr * flr_term[4, :] + 0.5 * sqrt_2_flr * ie2[:nmax]
        )
        return_array[5, :] = ie1[:nmax]

        return return_array

    @classmethod
    def _flr_term_dflr2(
        cls,
        flr: FloatType,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        bessel_exp_dx2: FloatArray = None,
        bessel_exp_dx3: FloatArray = None,
        flr_term: FloatArray = None,
        flr_term_dflr: FloatArray = None,
    ):
        """
        Calculate second derivative of finite Larmor radius dependent term with
        respect to flr for each 6 unique elements of the electric
        susceptibility.

        Parameters
        ----------
        flr : float
            Finite Larmor radius parameter.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        bessel_exp_dx2 : np.array[float], optional
            Second derivative of exponentially scaled modified Bessel function.
            If provided must have size >= nmax + 1.
        bessel_exp_dx3 : np.array[float], optional
            Third derivative of exponentially scaled modified Bessel function.
            If provided must have size >= nmax.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.
        flr_term_dflr : np.array[float], optional
            First derivative of finite Larmor radius dependent terms of
            electric susceptibilit with respect to flr.

        Returns
        -------
        flr_term_dflr2 : np.array[float]
            Second derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to flr.

        Raises
        ------
        ValueError
            bessel_exp_dx2 provided and has size < nmax + 1.
            bessel_exp_dx3 provided and has size < nmax.
        """
        return_array = get_return_array(return_array, (6, nmax), FloatType)

        if bessel_exp_dx2 is None:
            bessel_exp_dx2 = BesselIExp.bessel_exp(flr, nmax + 1, derivative=2)
        elif bessel_exp_dx2.size < nmax + 1:
            raise ValueError(
                f"bessel_exp_dx2 must have size >= {nmax + 1}: "
                f"{bessel_exp_dx2.size}"
            )

        if bessel_exp_dx3 is None:
            bessel_exp_dx3 = BesselIExp.bessel_exp(
                flr, nmax, derivative=3, bessel_exp_m1=bessel_exp_dx2
            )
        elif bessel_exp_dx3.size < nmax:
            raise ValueError(
                f"bessel_exp_dx3 must have size >= {nmax}: "
                f"{bessel_exp_dx3.size}"
            )

        if flr_term is None:
            flr_term = cls.flr_term(flr, nmax)

        if flr_term_dflr is None:
            flr_term_dflr = cls._flr_term_dflr(
                flr, nmax, bessel_exp_dx2=bessel_exp_dx2
            )

        sqrt_2_flr = np.sqrt(2.0 * flr)
        inv_flr = 1 / flr
        inv_flr2 = inv_flr * inv_flr

        _n = np.arange(nmax)
        ie2 = bessel_exp_dx2
        ie3 = bessel_exp_dx3

        return_array.fill(0.0)

        return_array[0, 1:] = (
            0.5 * _n[1:] * (ie2[: nmax - 1] - ie2[2 : nmax + 1])
        )
        return_array[1, 1:] = _n[1:] * ie3[1:nmax]
        return_array[2, 1:] = (
            -0.75 * inv_flr2 * flr_term[2, 1:]
            + inv_flr * flr_term_dflr[2, 1:]
            + 0.25 * sqrt_2_flr * (ie2[: nmax - 1] - ie2[2 : nmax + 1])
        )
        return_array[3, :] = return_array[0, :] - 2.0 * (
            2.0 * ie2[:nmax] + flr * ie3[:nmax]
        )
        return_array[4, :] = (
            -0.75 * inv_flr2 * flr_term[4, :]
            + inv_flr * flr_term_dflr[4, :]
            + 0.5 * sqrt_2_flr * ie3[:nmax]
        )
        return_array[5, :] = ie2[:nmax]

        return return_array

    @classmethod
    def flr_term_dq(
        cls,
        flr_term_dflr: FloatArray,
        flr_dq: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Calculate first derivative of finite Larmor radius dependent term for
        each 6 unique elements of the electric susceptibility with respect to
        Hamiltonian arguments q.

        Parameters
        ----------
        flr_term_dflr : np.array[float]
            First derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to flr.
        flr_dq : np.array[float]
            First derivative of finite Larmor radius parameter with respect to
            q.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        flr_term_dq
            First derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to q.
        """
        n = flr_term_dflr.shape[1]
        return_array = get_return_array(
            return_array, (6, n, Dimensions.q.size), FloatType
        )

        first_derivative(
            flr_term_dflr,
            flr_dq,
            (6, n),
            1,
            Dimensions.q.size,
            return_array=return_array,
        )

        return return_array

    @classmethod
    def flr_term_dq2(
        cls,
        flr_term_dflr: FloatArray,
        flr_term_dflr2: FloatArray,
        flr_dq: FloatArray,
        flr_dq2: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Calculate second derivative of finite Larmor radius dependent term for
        each 6 unique elements of the electric susceptibility with respect to
        Hamiltonian arguments q.

        Parameters
        ----------
        flr_term_dflr : np.array[float]
            First derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to flr.
        flr_term_dflr2 : np.array[float]
            Second derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to flr.
        flr_dq : np.array[float]
            First derivative of finite Larmor radius parameter with respect to
            q.
        flr_dq2 : np.array[float]
            Second derivative of finite Larmor radius parameter with respect
            to q.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        flr_term_dq2
            Second derivative of finite Larmor radius dependent terms of
            electric susceptibility with respect to q.
        """
        n = flr_term_dflr.shape[1]
        return_array = get_return_array(
            return_array,
            (6, n, Dimensions.q.size, Dimensions.q.size),
            FloatType,
        )

        second_derivative(
            flr_term_dflr,
            flr_term_dflr2,
            flr_dq,
            flr_dq2,
            (6, n),
            1,
            Dimensions.q.size,
            return_array=return_array,
        )

        return return_array

    @classmethod
    def resonance_term(
        cls,
        p: FloatArray,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        z_norm_real_sum: ComplexArray = None,
        z_dx_norm_real_sum: ComplexArray = None,
        z_dx2_norm_real_sum: ComplexArray = None,
    ):
        """
        Calculate resonance dispersion terms for each 6 unique elements of the
        electric susceptibility.

        Parameters
        ----------
        p : np.array[float]
            Dependent variables for the plasma dispersion function.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        z_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function real part sum / difference
            over opposite sign harmonics. If provided must have size >= nmax.
        z_dx_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function first derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx2_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function second derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.

        Returns
        -------
        resonance_term
            Resonance dispersion terms of electric susceptibility.

        Raises
        ------
        ValueError
            z_norm_real_sum provided and has size < nmax.
            z_dx_norm_real_sum provided and has size < nmax.
            z_dx2_norm_real_sum provided and has size < nmax.

        Notes
        -----
        This contributes only to the Hermitian part of the susceptibility.
        """
        return_array = get_return_array(return_array, (6, nmax), FloatType)

        if z_norm_real_sum is None:
            z_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 0)
        elif z_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_norm_real_sum must have size >= {nmax}: "
                f"{z_norm_real_sum.size}"
            )

        if z_dx_norm_real_sum is None:
            z_dx_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 1)
        elif z_dx_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx_norm_real_sum must have size >= {nmax}: "
                f"{z_dx_norm_real_sum.size}"
            )

        if z_dx2_norm_real_sum is None:
            z_dx2_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 2)
        elif z_dx2_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx2_norm_real_sum must have size >= {nmax}: "
                f"{z_dx2_norm_real_sum.size}"
            )

        doppler = p[PlasmaZ.IDX_DOPPLER]
        z = z_norm_real_sum
        z1 = z_dx_norm_real_sum
        z2 = z_dx2_norm_real_sum

        return_array.fill(0.0)

        return_array[0, :] = z[SUM, :nmax]
        return_array[1, :] = z[DIFF, :nmax]
        return_array[2, :] = doppler * z1[DIFF, :nmax]
        return_array[3, :] = return_array[0, :]
        return_array[4, :] = doppler * z1[SUM, :nmax]
        return_array[5, :] = (
            z[SUM, :nmax] + (0.5 * doppler * doppler) * z2[SUM, :nmax]
        )

        return return_array

    @classmethod
    def _resonance_term_dp(
        cls,
        p: FloatArray,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        z_dx_norm_real_sum: ComplexArray = None,
        z_dx2_norm_real_sum: ComplexArray = None,
        z_dx3_norm_real_sum: ComplexArray = None,
        z_dx4_norm_real_sum: ComplexArray = None,
    ):
        """
        Calculate first derivative of resonance dispersion terms for each 6
        unique elements of the electric susceptibility with respect to p.

        Parameters
        ----------
        p : np.array[float]
            Dependent variables for the plasma dispersion function.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        z_dx_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function first derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx2_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function second derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx3_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function third derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx4_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function fourth derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.

        Returns
        -------
        resonance_term_dp
            First derivative of resonance dispersion terms of electric
            susceptibility with respect to p.

        Raises
        ------
        ValueError
            z_dx_norm_real_sum provided and has size < nmax.
            z_dx2_norm_real_sum provided and has size < nmax.
            z_dx3_norm_real_sum provided and has size < nmax.
            z_dx4_norm_real_sum provided and has size < nmax.
        """
        return_array = get_return_array(return_array, (6, nmax, 2), FloatType)

        if z_dx_norm_real_sum is None:
            z_dx_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 1)
        elif z_dx_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx_norm_real_sum must have size >= {nmax}: "
                f"{z_dx_norm_real_sum.size}"
            )

        if z_dx2_norm_real_sum is None:
            z_dx2_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 2)
        elif z_dx2_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx2_norm_real_sum must have size >= {nmax}: "
                f"{z_dx2_norm_real_sum.size}"
            )

        if z_dx3_norm_real_sum is None:
            z_dx3_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 3)
        elif z_dx3_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx3_norm_real_sum must have size >= {nmax}: "
                f"{z_dx3_norm_real_sum.size}"
            )

        if z_dx4_norm_real_sum is None:
            z_dx4_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 4)
        elif z_dx4_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx4_norm_real_sum must have size >= {nmax}: "
                f"{z_dx4_norm_real_sum.size}"
            )

        n = np.arange(nmax)
        doppler = p[PlasmaZ.IDX_DOPPLER]
        doppler2 = doppler * doppler
        doppler3 = doppler * doppler2

        z1 = z_dx_norm_real_sum
        z2 = z_dx2_norm_real_sum
        z3 = z_dx3_norm_real_sum
        z4 = z_dx4_norm_real_sum

        return_array.fill(0.0)

        # Derivatives with respect to Y.
        return_array[0, :, PlasmaZ.IDX_Y] = -n * z1[DIFF, :]
        return_array[1, :, PlasmaZ.IDX_Y] = -n * z1[SUM, :]
        return_array[2, :, PlasmaZ.IDX_Y] = -doppler * n * z2[SUM, :]
        return_array[4, :, PlasmaZ.IDX_Y] = -doppler * n * z2[DIFF, :]
        return_array[5, :, PlasmaZ.IDX_Y] = -n * (
            z1[DIFF, :] + 0.5 * doppler2 * z3[DIFF, :]
        )

        # Derivatives with respect to dopper.
        return_array[0, :, PlasmaZ.IDX_DOPPLER] = 0.5 * doppler * z2[SUM, :]
        return_array[1, :, PlasmaZ.IDX_DOPPLER] = 0.5 * doppler * z2[DIFF, :]
        return_array[2, :, PlasmaZ.IDX_DOPPLER] = (
            z1[DIFF, :] + 0.5 * doppler2 * z3[DIFF, :]
        )
        return_array[4, :, PlasmaZ.IDX_DOPPLER] = (
            z1[SUM, :] + 0.5 * doppler2 * z3[SUM, :]
        )
        return_array[5, :, PlasmaZ.IDX_DOPPLER] = (
            1.5 * doppler * z2[SUM, :] + 0.25 * doppler3 * z4[SUM, :]
        )

        # xx and yy terms are equal.
        return_array[3, :, :] = return_array[0, :, :]

        return return_array

    @classmethod
    def _resonance_term_dp2(
        cls,
        p: FloatArray,
        nmax: int,
        /,
        *,
        return_array: FloatArray = None,
        z_dx2_norm_real_sum: ComplexArray = None,
        z_dx3_norm_real_sum: ComplexArray = None,
        z_dx4_norm_real_sum: ComplexArray = None,
        z_dx5_norm_real_sum: ComplexArray = None,
        z_dx6_norm_real_sum: ComplexArray = None,
    ):
        """
        Calculate second derivative of resonance dispersion terms for each 6
        unique elements of the electric susceptibility with respect to p.

        Parameters
        ----------
        p : np.array[float]
            Dependent variables for the plasma dispersion function.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        z_dx2_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function second derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx3_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function third derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx4_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function fourth derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx5_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function fifth derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.
        z_dx6_norm_real_sum : np.array[float], optional
            Normalised plasma dispersion function sixth derivative real part
            sum / difference over opposite sign harmonics. If provided must
            have size >= nmax.

        Returns
        -------
        resonance_term_dp2
            Second derivative of resonance dispersion terms of electric
            susceptibility with respect to p.

        Raises
        ------
        ValueError
            z_dx2_norm_real_sum provided and has size < nmax.
            z_dx3_norm_real_sum provided and has size < nmax.
            z_dx4_norm_real_sum provided and has size < nmax.
            z_dx5_norm_real_sum provided and has size < nmax.
            z_dx6_norm_real_sum provided and has size < nmax.
        """
        return_array = get_return_array(
            return_array, (6, nmax, 2, 2), FloatType
        )

        if z_dx2_norm_real_sum is None:
            z_dx2_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 2)
        elif z_dx2_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx2_norm_real_sum must have size >= {nmax}: "
                f"{z_dx2_norm_real_sum.size}"
            )

        if z_dx3_norm_real_sum is None:
            z_dx3_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 3)
        elif z_dx3_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx3_norm_real_sum must have size >= {nmax}: "
                f"{z_dx3_norm_real_sum.size}"
            )

        if z_dx4_norm_real_sum is None:
            z_dx4_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 4)
        elif z_dx4_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx4_norm_real_sum must have size >= {nmax}: "
                f"{z_dx4_norm_real_sum.size}"
            )

        if z_dx5_norm_real_sum is None:
            z_dx5_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 5)
        elif z_dx5_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx5_norm_real_sum must have size >= {nmax}: "
                f"{z_dx5_norm_real_sum.size}"
            )

        if z_dx6_norm_real_sum is None:
            z_dx6_norm_real_sum = PlasmaZ.z_dxk_norm_real_sum(p, nmax, 6)
        elif z_dx6_norm_real_sum.size < nmax:
            raise ValueError(
                f"z_dx6_norm_real_sum must have size >= {nmax}: "
                f"{z_dx6_norm_real_sum.size}"
            )

        n = np.arange(nmax)
        n2 = n * n
        doppler = p[PlasmaZ.IDX_DOPPLER]
        doppler2 = doppler * doppler
        doppler3 = doppler * doppler2
        doppler4 = doppler2 * doppler2

        z2 = z_dx2_norm_real_sum
        z3 = z_dx3_norm_real_sum
        z4 = z_dx4_norm_real_sum
        z5 = z_dx5_norm_real_sum
        z6 = z_dx6_norm_real_sum

        return_array.fill(0.0)

        # Derivatives with respect to Y.
        return_array[0, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_Y] = n2 * z2[SUM, :]
        return_array[1, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_Y] = n2 * z2[DIFF, :]
        return_array[2, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_Y] = (
            doppler * n2 * z3[DIFF, :]
        )
        return_array[4, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_Y] = (
            doppler * n2 * z3[SUM, :]
        )
        return_array[5, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_Y] = n2 * (
            z2[SUM, :] + 0.5 * doppler2 * z4[SUM, :]
        )

        # Derivatives with respect to Y and doppler.
        return_array[0, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_DOPPLER] = (
            -0.5 * doppler * n * z3[DIFF, :]
        )
        return_array[1, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_DOPPLER] = (
            -0.5 * doppler * n * z3[SUM, :]
        )
        return_array[2, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_DOPPLER] = -n * (
            z2[SUM, :] + 0.5 * doppler2 * z4[SUM, :]
        )
        return_array[4, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_DOPPLER] = -n * (
            z2[DIFF, :] + 0.5 * doppler2 * z4[DIFF, :]
        )
        return_array[5, :, PlasmaZ.IDX_Y, PlasmaZ.IDX_DOPPLER] = (
            -1.5 * n * doppler * z3[DIFF, :]
            - 0.25 * n * doppler3 * z5[DIFF, :]
        )

        # Derivatives with respect to doppler.
        return_array[0, :, PlasmaZ.IDX_DOPPLER, PlasmaZ.IDX_DOPPLER] = (
            0.5 * z2[SUM, :] + 0.25 * doppler2 * z4[SUM, :]
        )
        return_array[1, :, PlasmaZ.IDX_DOPPLER, PlasmaZ.IDX_DOPPLER] = (
            0.5 * z2[DIFF, :] + 0.25 * doppler2 * z4[DIFF, :]
        )
        return_array[2, :, PlasmaZ.IDX_DOPPLER, PlasmaZ.IDX_DOPPLER] = (
            1.5 * doppler * z3[DIFF, :] + 0.25 * doppler3 * z5[DIFF, :]
        )
        return_array[4, :, PlasmaZ.IDX_DOPPLER, PlasmaZ.IDX_DOPPLER] = (
            1.5 * doppler * z3[SUM, :] + 0.25 * doppler3 * z5[SUM, :]
        )
        return_array[5, :, PlasmaZ.IDX_DOPPLER, PlasmaZ.IDX_DOPPLER] = (
            1.5 * z2[SUM, :]
            + 1.5 * doppler2 * z4[SUM, :]
            + 0.125 * doppler4 * z6[SUM, :]
        )

        # xx and yy terms are equal.
        return_array[3, :, :, :] = return_array[0, :, :, :]

        # Partial second derivatives commute.
        mirror_upper_triangular_to_lower_triangular(return_array)

        return return_array

    @classmethod
    def resonance_term_dq(
        cls,
        resonance_term_dp: ComplexArray,
        p_dq: ComplexArray,
        /,
        *,
        return_array: ComplexArray = None,
    ):
        """
        Calculate first derivative of resonance dispersion terms for each 6
        unique elements of the electric susceptibility with respect to
        Hamiltonian arguments q.

        Parameters
        ----------
        resonance_term_dp
            First derivative of resonance dispersion terms of electric
            susceptibility with respect to p.
        p_dq
            First derivative of p with respect to q.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        resonance_term_dq
            First derivative of resonance dispersion terms of electric
            susceptibility with respect to q.
        """
        n = resonance_term_dp.shape[1]
        return_array = get_return_array(
            return_array, (6, n, Dimensions.q.size), FloatType
        )

        first_derivative(
            resonance_term_dp[:, :, :],
            p_dq,
            (6, n),
            2,
            Dimensions.q.size,
            return_array=return_array,
        )

        return return_array

    @classmethod
    def resonance_term_dq2(
        cls,
        resonance_term_dp: ComplexArray,
        resonance_term_dp2: ComplexArray,
        p_dq: ComplexArray,
        p_dq2: ComplexArray,
        /,
        *,
        return_array: ComplexArray = None,
    ):
        """
        Calculate second derivative of resonance dispersion terms for each 6
        unique elements of the electric susceptibility with respect to
        Hamiltonian arguments q.

        Parameters
        ----------
        resonance_term_dp
            First derivative of resonance dispersion terms of electric
            susceptibility with respect to p.
        resonance_term_dp2
            Second derivative of resonance dispersion terms of electric
            susceptibility with respect to p.
        p_dq
            First derivative of p with respect to q.
        p_dq2
            Second derivative of p with respect to q.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        resonance_term_dq2
            Second derivative of resonance dispersion terms of electric
            susceptibility with respect to q.
        """
        n = resonance_term_dp.shape[1]
        return_array = get_return_array(
            return_array,
            (6, n, Dimensions.q.size, Dimensions.q.size),
            FloatType,
        )

        second_derivative(
            resonance_term_dp[:, :, :],
            resonance_term_dp2[:, :, :, :],
            p_dq,
            p_dq2,
            (6, n),
            2,
            Dimensions.q.size,
            return_array=return_array,
        )

        return return_array

    @classmethod
    def damping_term(
        cls,
        q: FloatArray,
        nmax: int,
        /,
        *,
        resonance: bool,
        return_array: FloatArray = None,
        p: FloatArray = None,
        z_norm_imag_sum: FloatArray = None,
        z_dx_norm_imag_sum: FloatArray = None,
        z_dx2_norm_imag_sum: FloatArray = None,
    ):
        """
        Calculate resonance damping terms for each 6 unique elements of the
        electric susceptibility.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        resonance : bool
            If True calculate for cyclotron damping. Otherwise calculate
            for collisional damping.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        z_norm_imag_sum : np.array[float], optional
            Normalised plasma dispersion function imaginary part sum /
            difference over opposite sign harmonics. If provided must have
            size >= nmax.
        z_dx_norm_imag_sum : np.array[float], optional
            Normalised plasma dispersion function first derivative imaginary
            part sum / difference over opposite sign harmonics. If provided
            must have size >= nmax.
        z_dx2_norm_imag_sum : np.array[float], optional
            Normalised plasma dispersion function second derivative imaginary
            part sum / difference over opposite sign harmonics. If provided
            must have size >= nmax.

        Returns
        -------
        damping_term
            Resonance damping terms of electric susceptibility.

        Raises
        ------
        ValueError
            z_norm_imag_sum provided and has size < nmax.
            z_dx_norm_imag_sum provided and has size < nmax.
            z_dx2_norm_imag_sum provided and has size < nmax.

        Notes
        -----
        This contributes only to the anti-Hermitian part of the susceptibility.
        """
        return_array = get_return_array(return_array, (6, nmax), FloatType)

        if p is None:
            p = PlasmaZ.p(q)

        z = q[Dimensions.IDX_Z]

        if z_norm_imag_sum is None:
            z_norm_imag_sum = PlasmaZ.z_dxk_norm_imag_sum(p, z, nmax, 0)
        elif z_norm_imag_sum.size < nmax:
            raise ValueError(
                f"z_norm_imag_sum must have size >= {nmax}: "
                f"{z_norm_imag_sum.size}"
            )

        if z_dx_norm_imag_sum is None:
            z_dx_norm_imag_sum = PlasmaZ.z_dxk_norm_imag_sum(p, z, nmax, 1)
        elif z_dx_norm_imag_sum.size < nmax:
            raise ValueError(
                f"z_dx_norm_imag_sum must have size >= {nmax}: "
                f"{z_dx_norm_imag_sum.size}"
            )

        if z_dx2_norm_imag_sum is None:
            z_dx2_norm_imag_sum = PlasmaZ.z_dxk_norm_imag_sum(p, z, nmax, 2)
        elif z_dx2_norm_imag_sum.size < nmax:
            raise ValueError(
                f"z_dx2_norm_imag_sum must have size >= {nmax}: "
                f"{z_dx2_norm_imag_sum.size}"
            )

        doppler = p[PlasmaZ.IDX_DOPPLER]

        return_array.fill(0.0)

        i = RESONANCE if resonance else COLLISIONAL

        return_array[0, :] = z_norm_imag_sum[i, SUM, :nmax]
        return_array[1, :] = z_norm_imag_sum[i, DIFF, :nmax]
        return_array[2, :] = doppler * z_dx_norm_imag_sum[i, DIFF, :nmax]
        return_array[3, :] = return_array[0, :]
        return_array[4, :] = doppler * z_dx_norm_imag_sum[i, SUM, :nmax]
        return_array[5, :] = (
            z_norm_imag_sum[i, SUM, :nmax]
            + 0.5 * doppler * doppler * z_dx2_norm_imag_sum[i, SUM, :nmax]
        )

        return return_array

    @classmethod
    def chi_precursor(
        cls,
        q: FloatArray,
        nmax: int,
        /,
        *,
        flr_term: FloatArray = None,
        resonance_term: FloatArray = None,
    ):
        """
        Precursor factors to susceptibility elements of Hermitian part.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.
        resonance_term : np.array[float], optional
            Resonance dispersion terms of electric susceptibility.

        Returns
        -------
        chi_precursor : np.array[float]
            Precursor factors to susceptibility elements of Hermitian part.
        """
        if flr_term is None:
            flr = BesselIExp.flr(q)
            flr_term = cls.flr_term(flr, nmax)

        if resonance_term is None:
            # This contains the hermitian contribution from cyclotron
            # resonances. The anti-hermitian part is called damping_term.
            p = PlasmaZ.p(q)
            resonance_term = cls.resonance_term(p, nmax)

        return q[Dimensions.IDX_X] * np.einsum(
            "ia, ia -> i", flr_term, resonance_term
        )

    @classmethod
    def chi_precursor_dq(
        cls,
        q: FloatArray,
        nmax: int,
        /,
        *,
        flr_term: FloatArray = None,
        flr_term_dq: FloatArray = None,
        resonance_term: FloatArray = None,
        resonance_term_dq: FloatArray = None,
    ):
        """
        Precursor factors to susceptibility elements of Hermitian part first
        derivative with respect to q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.
        flr_term_dq : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility
            first derivative with respect to q.
        resonance_term : np.array[float], optional
            Resonance dispersion terms of electric susceptibility.
        resonance_term_dq : np.array[float], optional
            Resonance dispersion terms of electric susceptibility
            first derivative with respect to q.

        Returns
        -------
        chi_precursor_dq : np.array[float]
            Precursor factors to susceptibility elements of Hermitian part
            first derivative with respect to q.
        """
        if flr_term is None:
            flr = BesselIExp.flr(q)
            flr_term = cls.flr_term(flr, nmax)

        if flr_term_dq is None:
            flr = BesselIExp.flr(q)
            flr_dq = BesselIExp.flr_dq(q)
            flr_term = cls.flr_term(flr, nmax)
            flr_term_dflr = cls._flr_term_dflr(flr, nmax)

            flr_term_dq = cls.flr_term_dq(flr_term_dflr, flr_dq)

        if resonance_term is None:
            p = PlasmaZ.p(q)
            resonance_term = cls.resonance_term(p, nmax)

        if resonance_term_dq is None:
            p = PlasmaZ.p(q)
            p_dq = PlasmaZ.p_dq(q)
            resonance_term = cls.resonance_term(p, nmax)
            resonance_term_dp = cls._resonance_term_dp(p, nmax)
            resonance_term_dq = cls.resonance_term_dq(resonance_term_dp, p_dq)

        _chi_dq = q[Dimensions.IDX_X] * (
            np.einsum("iak, ia -> ik", flr_term_dq, resonance_term)
            + np.einsum("ia, iak -> ik", flr_term, resonance_term_dq)
        )

        # Density dependent derivatives.
        _chi_dq[:, Dimensions.IDX_X] = np.einsum(
            "ia, ia -> i", flr_term, resonance_term
        )

        return _chi_dq

    @classmethod
    def chi_precursor_dq2(
        cls,
        q: FloatArray,
        nmax: int,
        /,
        *,
        flr_term: FloatArray = None,
        flr_term_dq: FloatArray = None,
        flr_term_dq2: FloatArray = None,
        resonance_term: FloatArray = None,
        resonance_term_dq: FloatArray = None,
        resonance_term_dq2: FloatArray = None,
    ):
        """
        Precursor factors to susceptibility elements of Hermitian part second
        derivative with respect to q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.
        flr_term_dq : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility
            first derivative with respect to q.
        flr_term_dq2 : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility
            second derivative with respect to q.
        resonance_term : np.array[float], optional
            Resonance dispersion terms of electric susceptibility.
        resonance_term_dq : np.array[float], optional
            Resonance dispersion terms of electric susceptibility
            first derivative with respect to q.
        resonance_term_dq2 : np.array[float], optional
            Resonance dispersion terms of electric susceptibility
            second derivative with respect to q.

        Returns
        -------
        chi_precursor_dq2 : np.array[float]
            Precursor factors to susceptibility elements of Hermitian part
            second derivative with respect to q.
        """
        if flr_term is None:
            flr = BesselIExp.flr(q)
            flr_term = cls.flr_term(flr, nmax)

        if flr_term_dq is None:
            flr = BesselIExp.flr(q)
            flr_dq = BesselIExp.flr_dq(q)
            flr_term = cls.flr_term(flr, nmax)
            flr_term_dflr = cls._flr_term_dflr(flr, nmax)

            flr_term_dq = cls.flr_term_dq(flr_term_dflr, flr_dq)

        if flr_term_dq2 is None:
            flr = BesselIExp.flr(q)
            flr_dq = BesselIExp.flr_dq(q)
            flr_dq2 = BesselIExp.flr_dq2(q)
            flr_term = cls.flr_term(flr, nmax)
            flr_term_dflr = cls._flr_term_dflr(flr, nmax)
            flr_term_dflr2 = cls._flr_term_dflr2(flr, nmax)

            flr_term_dq2 = cls.flr_term_dq2(
                flr_term_dflr, flr_term_dflr2, flr_dq, flr_dq2
            )

        if resonance_term is None:
            p = PlasmaZ.p(q)
            resonance_term = cls.resonance_term(p, nmax)

        if resonance_term_dq is None:
            p = PlasmaZ.p(q)
            p_dq = PlasmaZ.p_dq(q)
            resonance_term = cls.resonance_term(p, nmax)
            resonance_term_dp = cls._resonance_term_dp(p, nmax)
            resonance_term_dq = cls.resonance_term_dq(resonance_term_dp, p_dq)

        if resonance_term_dq2 is None:
            p = PlasmaZ.p(q)
            p_dq = PlasmaZ.p_dq(q)
            p_dq2 = PlasmaZ.p_dq2(q)
            resonance_term = cls.resonance_term(p, nmax)
            resonance_term_dp = cls._resonance_term_dp(p, nmax)
            resonance_term_dp2 = cls._resonance_term_dp2(p, nmax)

            resonance_term_dq2 = cls.resonance_term_dq2(
                resonance_term_dp, resonance_term_dp2, p_dq, p_dq2
            )

        _chi_dq2 = q[Dimensions.IDX_X] * (
            np.einsum("iajk, ia -> ijk", flr_term_dq2, resonance_term)
            + np.einsum("iaj, iak -> ijk", flr_term_dq, resonance_term_dq)
            + np.einsum("iak, iaj -> ijk", flr_term_dq, resonance_term_dq)
            + np.einsum("ia, iajk -> ijk", flr_term, resonance_term_dq2)
        )

        # Density dependent derivatives.
        _chi_dq2[:, :, Dimensions.IDX_X] = np.einsum(
            "iak, ia -> ik", flr_term_dq, resonance_term
        ) + np.einsum("ia, iak -> ik", flr_term, resonance_term_dq)
        _chi_dq2[:, Dimensions.IDX_X, :] = _chi_dq2[:, :, Dimensions.IDX_X]

        return _chi_dq2

    @classmethod
    def susceptibility_hermitian(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
        nmax: int | None = None,
        chi_precursor: FloatArray = None,
    ) -> ComplexArray:
        """
        Hermitian part of electric susceptibility tensor.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        return_array : np.array[complex], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        nmax : int, optional
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        chi_precursor : np.array[float]
            Precursor factors to susceptibility elements of Hermitian part.

        Returns
        -------
        susceptibility_hermitian : np.array[complex]
            Hermitian part of electric susceptibility tensor.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        if nmax is None:
            nmax = Dimensions.max_harmonic.size

        _chi = (
            cls.chi_precursor(q, nmax)
            if chi_precursor is None
            else chi_precursor
        )

        # Construct tensor.
        return_array.fill(0.0)

        return_array[0, 0] = _chi[0]
        return_array[0, 1] = 1.0j * _chi[1]
        return_array[0, 2] = -_chi[2]
        return_array[1, 1] = _chi[3]
        return_array[1, 2] = 1.0j * _chi[4]
        return_array[2, 2] = _chi[5]

        return_array[1, 0] = -return_array[0, 1]
        return_array[2, 0] = return_array[0, 2]
        return_array[2, 1] = -return_array[1, 2]

        return return_array

    @classmethod
    def susceptibility_antihermitian_resonance(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
        nmax: int | None = None,
        flr_term: ComplexArray = None,
        damping_term_resonance: ComplexArray = None,
    ) -> ComplexArray:
        """
        Anti-Hermitian part of electric susceptibility tensor due to cyclotron
        damping.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        return_array : np.array[complex], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.
        damping_term_resonance : np.array[float], optional
            Resonance damping terms of electric susceptibility due to
            cyclotron resonance.

        Returns
        -------
        susceptibility_antihermitian_resonance : np.array[float]
            Anti-Hermitian part of electric susceptibility tensor due to
            cyclotron damping.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        if nmax is None:
            nmax = Dimensions.max_harmonic.size

        if flr_term is None:
            flr = BesselIExp.flr(q)
            flr_term = cls.flr_term(flr, nmax)

        # This contains the hermitian contribution from cyclotron resonances.
        # The anti-hermitian part is called damping_term.
        if damping_term_resonance is None:
            damping_term_resonance = cls.damping_term(q, nmax, resonance=True)

        # Sum over harmonics.
        precursor = q[Dimensions.IDX_X] * np.einsum(
            "ia, ia -> i", flr_term, damping_term_resonance
        )

        # Construct tensor.
        return_array.fill(0.0)

        return_array[0, 0] = 1.0j * precursor[0]
        return_array[0, 1] = -precursor[1]
        return_array[0, 2] = -1.0j * precursor[2]
        return_array[1, 1] = 1.0j * precursor[3]
        return_array[1, 2] = -precursor[4]
        return_array[2, 2] = 1.0j * precursor[5]

        return_array[1, 0] = -return_array[0, 1]
        return_array[2, 0] = return_array[0, 2]
        return_array[2, 1] = -return_array[1, 2]

        return return_array

    @classmethod
    def susceptibility_antihermitian_collisional(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
        nmax: int | None = None,
        flr_term: ComplexArray = None,
        damping_term_collisional: ComplexArray = None,
    ) -> ComplexArray:
        """
        Anti-Hermitian part of electric susceptibility tensor due to
        collisional damping.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        return_array : np.array[complex], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        nmax : int
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        flr_term : np.array[float], optional
            Finite Larmor radius dependent terms of electric susceptibility.
        damping_term_collisional : np.array[float], optional
            Resonance damping terms of electric susceptibility due to
            collisional resonance.

        Returns
        -------
        susceptibility_antihermitian_collisional : np.array[float]
            Anti-Hermitian part of electric susceptibility tensor due to
            collisional damping.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        if nmax is None:
            nmax = Dimensions.max_harmonic.size

        if flr_term is None:
            flr = BesselIExp.flr(q)
            flr_term = cls.flr_term(flr, nmax)

        # This contains the hermitian contribution from cyclotron resonances.
        # The anti-hermitian part is called damping_term.
        if damping_term_collisional is None:
            damping_term_collisional = cls.damping_term(
                q, nmax, resonance=False
            )

        # Sum over harmonics.
        precursor = q[Dimensions.IDX_X] * np.einsum(
            "ia, ia -> i", flr_term, damping_term_collisional
        )

        # Construct tensor.
        return_array.fill(0.0)

        return_array[0, 0] = 1.0j * precursor[0]
        return_array[0, 1] = -precursor[1]
        return_array[0, 2] = -1.0j * precursor[2]
        return_array[1, 1] = 1.0j * precursor[3]
        return_array[1, 2] = -precursor[4]
        return_array[2, 2] = 1.0j * precursor[5]

        return_array[1, 0] = -return_array[0, 1]
        return_array[2, 0] = return_array[0, 2]
        return_array[2, 1] = -return_array[1, 2]

        return return_array

    @classmethod
    def susceptibility_hermitian_dq(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
        nmax: int | None = None,
        chi_precursor_dq: FloatArray = None,
    ) -> ComplexArray:
        """
        Hermitian part of electric susceptibility tensor first derivative with
        respect to q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        return_array : np.array[complex], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        nmax : int, optional
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        chi_precursor_dq : np.array[float]
            Precursor factors to susceptibility elements of Hermitian part
            first derivative with respect to q.

        Returns
        -------
        susceptibility_hermitian_dq : np.array[complex]
            Hermitian part of electric susceptibility tensor first derivative
            with respect to q.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.x.size, Dimensions.x.size, Dimensions.q.size),
            ComplexType,
        )

        if nmax is None:
            nmax = Dimensions.max_harmonic.size

        if chi_precursor_dq is None:
            _chi_dq = cls.chi_precursor_dq(q, nmax)
        else:
            _chi_dq = chi_precursor_dq

        # Construct tensor.
        return_array.fill(0.0)

        return_array[0, 0, :] = _chi_dq[0]
        return_array[0, 1, :] = 1.0j * _chi_dq[1]
        return_array[0, 2, :] = -_chi_dq[2]
        return_array[1, 1, :] = _chi_dq[3]
        return_array[1, 2, :] = 1.0j * _chi_dq[4]
        return_array[2, 2, :] = _chi_dq[5]

        # Apply symmetries in susceptibility.
        return_array[1, 0, :] = -return_array[0, 1, :]
        return_array[2, 0, :] = return_array[0, 2, :]
        return_array[2, 1, :] = -return_array[1, 2, :]

        return return_array

    @classmethod
    def susceptibility_hermitian_dq2(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
        nmax: int | None = None,
        chi_precursor_dq2: FloatArray = None,
    ) -> ComplexArray:
        """
        Hermitian part of electric susceptibility tensor second derivative with
        respect to q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments.
        return_array : np.array[complex], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        nmax : int, optional
            Maximum harmonics to calculate. Includes zero so largest harmonic
            will be nmax - 1.
        chi_precursor_dq2 : np.array[float]
            Precursor factors to susceptibility elements of Hermitian part
            second derivative with respect to q.

        Returns
        -------
        susceptibility_hermitian_dq2 : np.array[complex]
            Hermitian part of electric susceptibility tensor second derivative
            with respect to q.
        """
        return_array = get_return_array(
            return_array,
            (
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.q.size,
                Dimensions.q.size,
            ),
            ComplexType,
        )

        if nmax is None:
            nmax = Dimensions.max_harmonic.size

        if chi_precursor_dq2 is None:
            _chi_dq2 = cls.chi_precursor_dq2(q, nmax)
        else:
            _chi_dq2 = chi_precursor_dq2

        # Construct tensor.
        return_array.fill(0.0)

        return_array[0, 0, :, :] = _chi_dq2[0]
        return_array[0, 1, :, :] = 1.0j * _chi_dq2[1]
        return_array[0, 2, :, :] = -_chi_dq2[2]
        return_array[1, 1, :, :] = _chi_dq2[3]
        return_array[1, 2, :, :] = 1.0j * _chi_dq2[4]
        return_array[2, 2, :, :] = _chi_dq2[5]

        # Apply symmetries in susceptibility.
        return_array[1, 0, :, :] = -return_array[0, 1, :, :]
        return_array[2, 0, :, :] = return_array[0, 2, :, :]
        return_array[2, 1, :, :] = -return_array[1, 2, :, :]

        return return_array

    @classmethod
    def calculate_susceptibility(
        cls,
        q: FloatArray,
        cache: SusceptibilityCache,
        derivatives: int,
        /,
        *,
        nmax: int | None = None,
        hermitian: bool = True,
        antihermitian: bool = True,
    ):
        """
        Evaluate susceptibility up to desired derivative and store results in
        provided cache. This allows maximum re-use of shared elements.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        cache : SusceptibilityCache
            Cache into which the results are stored.
        derivatives : int
            Number of derivatives of the hermitian part of the susceptibility
            to evaluate.
        hermitian : bool
            If True, evaluate the hermitian part of the susceptibility.
        antihermitian : bool
            If True, evaluate all anti-hermitian parts of the susceptibility.
        """
        if nmax is None:
            nmax = Dimensions.max_harmonic.size

        # Finite Larmor radius terms.
        flr = BesselIExp.flr(q)
        n_flr = nmax + 1 + derivatives

        bessel_exp = BesselIExp.bessel_exp(flr, n_flr)
        bessel_exp_dx = BesselIExp.bessel_exp(
            flr, n_flr - 1, derivative=1, bessel_exp_m1=bessel_exp
        )

        flr_term = cls.flr_term(
            flr, nmax, bessel_exp=bessel_exp, bessel_exp_dx=bessel_exp_dx
        )

        # Plasma dispersion terms.
        p = PlasmaZ.p(q)
        y = p[PlasmaZ.IDX_Y]
        doppler = p[PlasmaZ.IDX_DOPPLER]

        harmonic_gap = np.empty((2, nmax))
        harmonic_gap[PLUS, :] = PlasmaZ.harmonic_gap(y, nmax, plus=True)
        harmonic_gap[MINUS, :] = PlasmaZ.harmonic_gap(y, nmax, plus=False)

        z_norm_real = np.zeros((7, 2, nmax))

        for i in (PLUS, MINUS):
            z_norm_real[0, i, :] = PlasmaZ.z_norm_real(
                harmonic_gap[i, :], doppler, derivative=0
            )

            z_norm_real[1, i, :] = PlasmaZ.z_norm_real(
                harmonic_gap[i, :],
                doppler,
                derivative=1,
                z_norm_real_m1=z_norm_real[0, i, :],
            )

            z_norm_real[2, i, :] = PlasmaZ.z_norm_real(
                harmonic_gap[i, :],
                doppler,
                derivative=2,
                z_norm_real_m1=z_norm_real[1, i, :],
                z_norm_real_m2=z_norm_real[0, i, :],
            )

            z_norm_real[3, i, :] = PlasmaZ.z_norm_real(
                harmonic_gap[i, :],
                doppler,
                derivative=3,
                z_norm_real_m1=z_norm_real[2, i, :],
                z_norm_real_m2=z_norm_real[1, i, :],
            )

        if antihermitian:
            z = q[Dimensions.IDX_Z]

            # derivative, res/coll, plus/minus, harmonic.
            z_norm_imag = np.zeros((3, 2, 2, nmax))
            z_norm_imag_sum = np.zeros((3, 2, 2, nmax))

            for i in (PLUS, MINUS):
                for k in range(3):
                    z_norm_imag[k, :, i, :] = PlasmaZ.z_norm_imag(
                        harmonic_gap[i, :],
                        doppler,
                        z,
                        derivative=k,
                        z_norm_real_p1=z_norm_real[k + 1, i, :],
                    )

            for i in (RESONANCE, COLLISIONAL):
                for k in range(3):
                    PlasmaZ.sum_diff(
                        z_norm_imag[k, i, PLUS, :],
                        z_norm_imag[k, i, MINUS, :],
                        return_array=z_norm_imag_sum[k, i, :, :],
                    )

            damping_term_resonance = cls.damping_term(
                q,
                nmax,
                resonance=True,
                p=p,
                z_norm_imag_sum=z_norm_imag_sum[0, :, :, :],
                z_dx_norm_imag_sum=z_norm_imag_sum[1, :, :, :],
                z_dx2_norm_imag_sum=z_norm_imag_sum[2, :, :, :],
            )

            damping_term_collisional = cls.damping_term(
                q,
                nmax,
                resonance=False,
                p=p,
                z_norm_imag_sum=z_norm_imag_sum[0, :, :, :],
                z_dx_norm_imag_sum=z_norm_imag_sum[1, :, :, :],
                z_dx2_norm_imag_sum=z_norm_imag_sum[2, :, :, :],
            )

            cls.susceptibility_antihermitian_resonance(
                q,
                return_array=cache.antihermitian_resonance,
                flr_term=flr_term,
                damping_term_resonance=damping_term_resonance,
            )

            cls.susceptibility_antihermitian_collisional(
                q,
                return_array=cache.antihermitian_collisional,
                flr_term=flr_term,
                damping_term_collisional=damping_term_collisional,
            )

        if hermitian:
            # Value
            z_norm_real_sum = np.zeros((7, 2, nmax))

            for i in range(4):
                PlasmaZ.sum_diff(
                    z_norm_real[i, PLUS, :],
                    z_norm_real[i, MINUS, :],
                    return_array=z_norm_real_sum[i, :, :],
                )

            p = PlasmaZ.p(q)

            resonance_term = cls.resonance_term(
                p,
                nmax,
                z_norm_real_sum=z_norm_real_sum[0, :, :],
                z_dx_norm_real_sum=z_norm_real_sum[1, :, :],
                z_dx2_norm_real_sum=z_norm_real_sum[2, :, :],
            )

            chi_precursor = cls.chi_precursor(
                q, nmax, flr_term=flr_term, resonance_term=resonance_term
            )

            cls.susceptibility_hermitian(
                q,
                return_array=cache.hermitian,
                nmax=nmax,
                chi_precursor=chi_precursor,
            )

            if derivatives == 0:
                return

            # First derivative.
            # Finite Larmor radius terms.
            bessel_exp_dx2 = BesselIExp.bessel_exp(
                flr, n_flr - 2, derivative=2, bessel_exp_m1=bessel_exp_dx
            )
            flr_term_dflr = cls._flr_term_dflr(
                flr,
                nmax,
                bessel_exp_dx=bessel_exp_dx,
                bessel_exp_dx2=bessel_exp_dx2,
                flr_term=flr_term,
            )
            flr_dq = BesselIExp.flr_dq(q)

            flr_term_dq = cls.flr_term_dq(flr_term_dflr, flr_dq)

            # Plasma dispersion terms.
            for i in (PLUS, MINUS):
                z_norm_real[4, i, :] = PlasmaZ.z_norm_real(
                    harmonic_gap[i, :],
                    doppler,
                    derivative=4,
                    z_norm_real_m1=z_norm_real[3, i, :],
                    z_norm_real_m2=z_norm_real[2, i, :],
                )

            PlasmaZ.sum_diff(
                z_norm_real[4, PLUS, :],
                z_norm_real[4, MINUS, :],
                return_array=z_norm_real_sum[4, :, :],
            )

            resonance_term_dp = cls._resonance_term_dp(
                p,
                nmax,
                z_dx_norm_real_sum=z_norm_real_sum[1, :, :],
                z_dx2_norm_real_sum=z_norm_real_sum[2, :, :],
                z_dx3_norm_real_sum=z_norm_real_sum[3, :, :],
                z_dx4_norm_real_sum=z_norm_real_sum[4, :, :],
            )
            p_dq = PlasmaZ.p_dq(q)

            resonance_term_dq = cls.resonance_term_dq(resonance_term_dp, p_dq)

            chi_precursor_dq = cls.chi_precursor_dq(
                q,
                nmax,
                flr_term=flr_term,
                flr_term_dq=flr_term_dq,
                resonance_term=resonance_term,
                resonance_term_dq=resonance_term_dq,
            )

            cls.susceptibility_hermitian_dq(
                q,
                return_array=cache.hermitian_dq,
                nmax=nmax,
                chi_precursor_dq=chi_precursor_dq,
            )

            if derivatives == 1:
                return

            # Second derivative.
            # Finite Larmor radius terms.
            bessel_exp_dx3 = BesselIExp.bessel_exp(
                flr, n_flr - 3, derivative=3, bessel_exp_m1=bessel_exp_dx2
            )
            flr_term_dflr2 = cls._flr_term_dflr2(
                flr,
                nmax,
                bessel_exp_dx2=bessel_exp_dx2,
                bessel_exp_dx3=bessel_exp_dx3,
                flr_term=flr_term,
                flr_term_dflr=flr_term_dflr,
            )
            flr_dq2 = BesselIExp.flr_dq2(q)

            flr_term_dq2 = cls.flr_term_dq2(
                flr_term_dflr, flr_term_dflr2, flr_dq, flr_dq2
            )

            # Plasma dispersion terms.
            for i in (PLUS, MINUS):
                z_norm_real[5, i, :] = PlasmaZ.z_norm_real(
                    harmonic_gap[i, :],
                    doppler,
                    derivative=5,
                    z_norm_real_m1=z_norm_real[4, i, :],
                    z_norm_real_m2=z_norm_real[3, i, :],
                )
                z_norm_real[6, i, :] = PlasmaZ.z_norm_real(
                    harmonic_gap[i, :],
                    doppler,
                    derivative=6,
                    z_norm_real_m1=z_norm_real[5, i, :],
                    z_norm_real_m2=z_norm_real[4, i, :],
                )

            PlasmaZ.sum_diff(
                z_norm_real[5, PLUS, :],
                z_norm_real[5, MINUS, :],
                return_array=z_norm_real_sum[5, :, :],
            )
            PlasmaZ.sum_diff(
                z_norm_real[6, PLUS, :],
                z_norm_real[6, MINUS, :],
                return_array=z_norm_real_sum[6, :, :],
            )

            resonance_term_dp2 = cls._resonance_term_dp2(
                p,
                nmax,
                z_dx2_norm_real_sum=z_norm_real_sum[2, :, :],
                z_dx3_norm_real_sum=z_norm_real_sum[3, :, :],
                z_dx4_norm_real_sum=z_norm_real_sum[4, :, :],
                z_dx5_norm_real_sum=z_norm_real_sum[5, :, :],
                z_dx6_norm_real_sum=z_norm_real_sum[6, :, :],
            )
            p_dq2 = PlasmaZ.p_dq2(q)

            resonance_term_dq2 = cls.resonance_term_dq2(
                resonance_term_dp, resonance_term_dp2, p_dq, p_dq2
            )

            chi_precursor_dq2 = cls.chi_precursor_dq2(
                q,
                nmax,
                flr_term=flr_term,
                flr_term_dq=flr_term_dq,
                flr_term_dq2=flr_term_dq2,
                resonance_term=resonance_term,
                resonance_term_dq=resonance_term_dq,
                resonance_term_dq2=resonance_term_dq2,
            )

            cls.susceptibility_hermitian_dq2(
                q,
                return_array=cache.hermitian_dq2,
                nmax=nmax,
                chi_precursor_dq2=chi_precursor_dq2,
            )

    @classmethod
    def _func_root_find_n(cls, q: FloatArray):
        """
        Objective function for root finding refractive index.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].

        Returns
        -------
        func
            Function which returns determinant of dispersion tensor as a
            function of refractive index.
        """
        _q = np.copy(q)
        _theta = np.arctan2(_q[_N_PERP], _q[_N_PARALLEL])
        s, c = np.sin(_theta), np.cos(_theta)

        def func(n):
            _q[_N_PERP] = s * n
            _q[_N_PARALLEL] = c * n

            return abs(np.linalg.det(cls.dispersion_tensor(_q)).real)

        return func

    @classmethod
    def calculate_n(
        cls, q: FloatArray, wave_mode: WaveMode, /, *, kinetic: bool
    ) -> Result:
        """
        Find magnitude of refractive index n that sets the determinant of the
        dispersion tensor to zero, preserving the relative sizes of n_perp
        and n_parallel.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        wave_mode : WaveMode
            Wave mode to calculate for.
        kinetic : bool
            If True, look for a kinetic solution (large n_perp).

        Returns
        -------
        n : float
            Magnitude of refractive index.
        """
        # No electromagnetic modes for N**2 > 1 + Y.
        y = q[_Y]
        max_n_em = np.sqrt(1.0 + y)

        if kinetic:
            # Bernstein waves have N_perp ~ 1 / beta_thermal away from mode
            # conversion regions. So assume no modes with double this (???).
            _min = max_n_em

            theta = q[_THETA]
            _max = 2.0 / np.sqrt(2 * theta)
        # Look for any electromagnetic mode.
        elif wave_mode == WaveMode.ANY:
            _min = 0.0
            _max = max_n_em
        else:
            # Look for specific electromagnetic mode.
            # Use cold dispersion as guess.
            guess = ColdDispersion.calculate_n(q, wave_mode)

            # If cannot find cold root likely parameters are bad.
            if guess.message:
                return Result.Failure(
                    "Unable to find cold guess for n_perp: " + guess.message
                )

            # Find root near guess.
            _delta = 0.05 + 0.2 * guess.value
            _min = guess.value - _delta
            _max = guess.value + _delta

        # Find root.
        func = cls._func_root_find_n(q)
        result = optimize.minimize_scalar(
            func, bracket=(_min, _max), method="golden", tol=1e-8
        )

        if result.success:
            return Result.success(result.x)
        return Result.failure(f"Unable to find root: {result.message}")

    @classmethod
    def _func_root_find_n_perp(cls, q: FloatArray):
        """
        Objective function for root finding perpendicular refractive index.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].

        Returns
        -------
        func
            Function which returns determinant of dispersion tensor as a
            function of perpendicular refractive index.
        """
        _q = np.copy(q)

        def func(n_perp):
            _q[_N_PERP] = n_perp

            return abs(np.linalg.det(cls.dispersion_tensor(_q)).real)

        return func

    @classmethod
    def calculate_n_perp(
        cls, q: FloatArray, wave_mode: WaveMode, /, *, kinetic: bool
    ) -> Result:
        """
        Find perpendicular refractive index n that sets the determinant of the
        dispersion tensor to zero with the initial value of n_parallel fixed.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        wave_mode : WaveMode
            Wave mode to calculate for.
        kinetic : bool
            If True, look for a kinetic solution (large n_perp).

        Returns
        -------
        n_perp : float
            Magnitude of perpendicular refractive index.
        """
        # No electromagnetic modes for N**2 > 1 + Y.
        y = q[_Y]
        max_n_perp_em = np.sqrt(1.0 + y)

        if kinetic:
            # Bernstein waves have N_perp ~ 1 / beta_thermal away from mode
            # conversion regions. So assume no modes with double this (???).
            _min = max_n_perp_em

            theta = q[_THETA]
            _max = 2.0 / np.sqrt(2 * theta)
        # Look for any electromagnetic mode.
        elif wave_mode == WaveMode.ANY:
            _min = 0.0
            _max = max_n_perp_em
        else:
            # Look for specific electromagnetic mode.
            # Use cold dispersion as guess.
            guess = ColdDispersion.calculate_n_perp(q, wave_mode)

            # If cannot find cold root likely parameters are bad.
            if guess.message:
                return Result.Failure(
                    "Unable to find cold guess for n_perp: " + guess.message
                )

            # Find root near guess.
            _delta = 0.05 + 0.2 * guess.value
            _min = guess.value - _delta
            _max = guess.value + _delta

        # Find root.
        func = cls._func_root_find_n_perp(q)
        result = optimize.minimize_scalar(
            func, bracket=(_min, _max), method="golden", tol=1e-8
        )

        if result.success:
            return Result.success(result.x)
        return Result.failure(f"Unable to find root: {result.message}")
