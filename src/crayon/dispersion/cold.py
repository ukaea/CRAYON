"""
Cold plasma dispersion tensor model.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.dispersion.base import (
    DispersionModel,
    DispersionType,
    SusceptibilityCache,
)
from crayon.shared.constants import WaveMode
from crayon.shared.data_structures import Result
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.numerics import solve_quadratic
from crayon.shared.types import ComplexArray, ComplexType, FloatArray

logger = logging.getLogger(__name__)

_X = Dimensions.IDX_X
_Y = Dimensions.IDX_Y
_Z = Dimensions.IDX_Z
_N_PERP = Dimensions.IDX_N_PERP
_N_PARALLEL = Dimensions.IDX_N_PARALLEL


class ColdDispersion(DispersionModel):
    """
    Cold dispersion model treating plasma as a cold fluid.

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

    dispersion_type = DispersionType.COLD

    @classmethod
    def susceptibility_hermitian(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
    ) -> ComplexArray:
        """
        Hermitian part of electric susceptibility tensor chi_h.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        chi_h : np.array[complex]
            Hermitian part of electric susceptibility.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        x, y = q[_X], q[_Y]

        return_array.fill(0.0)

        plus = 1.0 / (1.0 + y)
        minus = 1.0 / (1.0 - y)

        return_array[0, 0] = -0.5 * x * (plus + minus)
        return_array[0, 1] = 0.5j * x * (plus - minus)
        return_array[2, 2] = -x

        # Apply symmetries in chi.
        # chi_yx = -chi_xy.
        return_array[1, 0] = -return_array[0, 1]

        # chi_yy = chi_xx.
        return_array[1, 1] = return_array[0, 0]

        return return_array

    @classmethod
    def susceptibility_antihermitian_resonance(
        cls,
        _q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
    ) -> ComplexArray:
        """
        Anti-hermitian part of electric susceptibility tensor chi_ah due to
        resonant absorption.

        Parameters
        ----------
        _q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        chi_ah_resonance : np.array[complex]
            Anti-hermitian part of electric susceptibility due to resonant
            absorption.

        Notes
        -----
        Require a thermal dispersion model to correctly describe resonant
        damping as it is a kinetic effect.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        return_array.fill(0.0)

        return return_array

    @classmethod
    def susceptibility_antihermitian_collisional(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
    ) -> ComplexArray:
        """
        Anti-hermitian part of electric susceptibility tensor chi_ah due to
        collisional absorption.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        chi_ah_collisional : np.array[complex]
            Anti-hermitian part of electric susceptibility due to collisional
            absorption.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        x, y, z = q[_X], q[_Y], q[_Z]

        return_array.fill(0.0)

        return_array[0, 0] = (
            0.5j * x * z * (1 / np.square(1 + y) + 1 / np.square(1 - y))
        )
        return_array[0, 1] = (
            0.5 * x * z * (1 / np.square(1 + y) - 1 / np.square(1 - y))
        )
        return_array[2, 2] = -1.0j * x * z

        # Apply symmetries in chi.
        # chi_yx = -chi_xy.
        return_array[1, 0] = -return_array[0, 1]

        # chi_yy = chi_xx.
        return_array[1, 1] = return_array[0, 0]

        return return_array

    @classmethod
    def susceptibility_hermitian_dq(
        cls, q: FloatArray, /, *, return_array: ComplexArray = None
    ) -> ComplexArray:
        """
        First derivative of chi_h with respect to hamiltonian arguments q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        chi_h_dq : np.array[complex]
            First derivative of chi_h with respect to q.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.x.size, Dimensions.x.size, Dimensions.q.size),
            ComplexType,
        )

        x, y = q[_X], q[_Y]

        # Chi only has 3 independent components chi_xx, chi_xy and chi_zz.
        # Chi doesn't explicitly depend on N_perp or N_parallel but D does.
        return_array.fill(0.0)

        plus = 1 / (1 + y)
        minus = 1 / (1 - y)

        # Derivatives with respect to X.
        return_array[0, 0, _X] = -0.5 * (plus + minus)
        return_array[0, 1, _X] = 0.5j * (plus - minus)
        return_array[2, 2, _X] = -1

        # Derivatives with respect to Y.
        plus_dy = -plus * plus
        minus_dy = minus * minus

        return_array[0, 0, _Y] = -0.5 * x * (plus_dy + minus_dy)
        return_array[0, 1, _Y] = 0.5j * x * (plus_dy - minus_dy)

        # Apply symmetries in chi.
        # chi_yx = -chi_xy.
        return_array[1, 0, :] = -return_array[0, 1, :]

        # chi_yy = chi_xx.
        return_array[1, 1, :] = return_array[0, 0, :]

        return return_array

    @classmethod
    def susceptibility_hermitian_dq2(
        cls, q: FloatArray, /, *, return_array: ComplexArray = None
    ) -> ComplexArray:
        """
        Second derivative of chi_h with respect to hamiltonian arguments q.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        chi_h_dq2 : np.array[complex]
            Second derivative of chi_h with respect to q.
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

        x, y = q[_X], q[_Y]

        return_array.fill(0.0)

        plus = 1 / (1 + y)
        minus = 1 / (1 - y)

        # Derivatives with respect to x and Y.
        plus_dy = -plus * plus
        minus_dy = minus * minus
        return_array[0, 0, _X, _Y] = -0.5 * (plus_dy + minus_dy)
        return_array[0, 1, _X, _Y] = 0.5j * (plus_dy - minus_dy)

        # Second mixed derivatives commute.
        return_array[:, :, _Y, _X] = return_array[:, :, _X, _Y]

        # Derivatives with respect to Y.
        plus_dy2 = -2.0 * plus * plus_dy
        minus_dy2 = 2.0 * minus * minus_dy
        return_array[0, 0, _Y, _Y] = -0.5 * x * (plus_dy2 + minus_dy2)
        return_array[0, 1, _Y, _Y] = 0.5j * x * (plus_dy2 - minus_dy2)

        # Apply symmetries in chi.
        # chi_yx = -chi_xy.
        return_array[1, 0, :, :] = -return_array[0, 1, :, :]

        # chi_yy = chi_xx.
        return_array[1, 1, :, :] = return_array[0, 0, :, :]

        return return_array

    @classmethod
    def calculate_susceptibility(
        cls,
        q: FloatArray,
        cache: SusceptibilityCache,
        derivatives: int,
        /,
        *,
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
        # Calculate value.
        if antihermitian:
            cls.susceptibility_antihermitian_resonance(
                q, return_array=cache.antihermitian_resonance
            )
            cls.susceptibility_antihermitian_collisional(
                q, return_array=cache.antihermitian_collisional
            )

        if not hermitian:
            return

        cls.susceptibility_hermitian(q, return_array=cache.hermitian)

        if derivatives == 0:
            return

        # Calculate first derivative.
        cls.susceptibility_hermitian_dq(q, return_array=cache.hermitian_dq)

        if derivatives == 1:
            return

        # Calculate second derivative.
        cls.susceptibility_hermitian_dq2(q, return_array=cache.hermitian_dq2)

        if derivatives == 2:  # noqa: PLR2004
            return

        # Calculate third derivative.
        raise NotImplementedError

    @classmethod
    def calculate_n(
        cls, q: FloatArray, wave_mode: WaveMode, /, **_kwargs
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
        # Solve Booker quartic for root.
        x, y, n_perp, n_parallel = (
            q[_X],
            abs(q[_Y]),
            q[_N_PERP],
            q[_N_PARALLEL],
        )

        # Calculate coefficients in Booker Quartic.
        n_parallel2 = n_parallel * n_parallel
        n2 = n_perp * n_perp + n_parallel2

        if n2 == 0.0:  # noqa: RUF069
            return Result.failure(
                "Initial refractive index has magnitude zero"
            )

        y2 = y * y

        stix_s = 1 - x / (1 - y2)
        stix_p = 1 - x
        stix_r = 1 - x / (1 + y)
        stix_l = 1 - x / (1 - y)

        c2 = n_parallel2 / n2
        s2 = 1 - c2

        _a = stix_s * s2 + stix_p * c2
        _b = -(stix_r * stix_l * s2 + stix_p * stix_s * (1 + c2))
        _c = stix_p * stix_r * stix_l

        descriminant = _b * _b - 4 * _a * _c

        if descriminant.real < 0.0:
            if np.isclose(descriminant, 0.0):
                # Negative value likely a rounding error.
                n2_plus = -_b / (2 * _a)
                n2_minus = n2_plus
            else:
                return Result.failure("No real root (Re(descriminant) < 0)")
        else:
            n2_plus, n2_minus = solve_quadratic(_a, _b, _c)

        if wave_mode == WaveMode.O:
            if n2_plus < 0:
                return Result.failure("N**2 < 0")

            root = np.sqrt(n2_plus)
        elif wave_mode == WaveMode.X:
            if n2_minus < 0:
                return Result.failure("N**2 < 0")

            root = np.sqrt(n2_minus)
        else:
            raise NotImplementedError(wave_mode)

        return Result.success(root.real)

    @classmethod
    def calculate_n_perp(
        cls, q: FloatArray, wave_mode: WaveMode, /, **_kwargs
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
        # Solve Booker quartic for root.
        x, y, n_parallel = q[_X], abs(q[_Y]), q[_N_PARALLEL]

        y2 = y * y
        npar2 = n_parallel * n_parallel

        _a = 1 - x - y2
        _b = (1 + npar2) * x * y2 - 2 * _a * ((1 - npar2) - x)
        _c = (1 - x) * (
            npar2 * npar2 * (1 - y2) - 2 * npar2 * _a + (1 - x) * (1 - x) - y2
        )

        descriminant = _b * _b - 4 * _a * _c

        if descriminant < 0.0:
            if np.isclose(descriminant, 0.0):
                # Negative value likely a rounding error.
                nperp2_minus = -_b / (2 * _a)
                nperp2_plus = nperp2_minus
            else:
                return Result.failure("No real root (Re(descriminant) < 0)")
        else:
            nperp2_plus, nperp2_minus = solve_quadratic(_a, _b, _c)

        if wave_mode == WaveMode.O:
            if x <= 1:
                root = np.sqrt(max(0.0, nperp2_plus))
            else:
                root = np.sqrt(max(0.0, nperp2_minus))
        elif wave_mode == WaveMode.X:
            if x <= 1:
                root = np.sqrt(max(0.0, nperp2_minus))
            else:
                root = np.sqrt(max(0.0, nperp2_plus))
        else:
            raise NotImplementedError(wave_mode)

        return Result.success(root.real)
