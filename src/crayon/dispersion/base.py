"""
Base classes and helper methods for dispersion tensor models.
"""

# Standard imports
import abc
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import (
    adjugate_3x3_cofactors,
    matrix_3x3_adjugate_first_derivative,
    matrix_3x3_determinant_first_derivative,
    matrix_3x3_determinant_second_derivative,
)
from crayon.shared.constants import (
    MIN_NORM_MAGNETIC_FIELD_STRENGTH,
    DispersionType,
    WaveMode,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import (
    ComplexArray,
    ComplexType,
    FloatArray,
    FloatType,
)

logger = logging.getLogger(__name__)

_N_PERP = Dimensions.IDX_N_PERP
_N_PARALLEL = Dimensions.IDX_N_PARALLEL


def vacuum_dispersion_tensor(
    q: FloatArray, /, *, return_array: FloatArray = None
) -> FloatArray:
    """
    Vacuum dispersion tensor NN + (1 - N**2) I
        n[i, j] = N_i N_j + (1 - N**2) delta_ij
    where N is in the Stix frame i.e. N = [N_perp, 0, N_parallel].

    Parameters
    ----------
    q : np.array[float]
        Hamiltonian arguments vector [X, Y, Z, theta, n_perp, n_parallel].
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    vacuum_dispersion_tensor : np.array[float]
        Vacuum dispersion tensor.
    """
    return_array = get_return_array(
        return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
    )

    n_perp = q[_N_PERP]
    n_parallel = q[_N_PARALLEL]
    n_perp2 = n_perp * n_perp
    n_parallel2 = n_parallel * n_parallel

    return_array.fill(0.0)

    return_array[0, 0] = 1 - n_parallel**2
    return_array[1, 1] = 1 - n_perp2 - n_parallel2
    return_array[2, 2] = 1 - n_perp**2
    return_array[0, 2] = n_perp * n_parallel
    return_array[2, 0] = return_array[0, 2]

    return return_array


def vacuum_dispersion_tensor_dq(
    q: FloatArray, /, *, return_array: FloatArray = None
) -> FloatArray:
    """
    First derivative of vacuum dispersion tensor with respect to q.

    Parameters
    ----------
    q : np.array[float]
        Hamiltonian arguments vector [X, Y, Z, theta, n_perp, n_parallel].
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    vacuum_dispersion_tensor_dq : np.array[float]
        First derivative of vacuum dispersion tensor with respect to q.
    """
    return_array = get_return_array(
        return_array,
        (Dimensions.x.size, Dimensions.x.size, Dimensions.q.size),
        ComplexType,
    )

    n_perp = q[_N_PERP]
    n_parallel = q[_N_PARALLEL]

    return_array.fill(0.0)

    # N_perp derivatives.
    return_array[0, 2, _N_PERP] = n_parallel
    return_array[1, 1, _N_PERP] = -2 * n_perp
    return_array[2, 0, _N_PERP] = return_array[0, 2, _N_PERP]
    return_array[2, 2, _N_PERP] = return_array[1, 1, _N_PERP]

    # N_parallel derivatives.
    return_array[0, 0, _N_PARALLEL] = -2 * n_parallel
    return_array[0, 2, _N_PARALLEL] = n_perp
    return_array[2, 0, _N_PARALLEL] = return_array[0, 2, _N_PARALLEL]
    return_array[1, 1, _N_PARALLEL] = return_array[0, 0, _N_PARALLEL]

    return return_array


def vacuum_dispersion_tensor_dq2(
    *, return_array: FloatArray = None
) -> FloatArray:
    """
    Second derivative of vacuum dispersion tensor with respect to q.

    Parameters
    ----------
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    vacuum_dispersion_tensor_dq2 : np.array[float]
        Second derivative of vacuum dispersion tensor with respect to q
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

    return_array.fill(0.0)

    # N_perp N_perp derivatives.
    return_array[1, 1, _N_PERP, _N_PERP] = -2
    return_array[2, 2, _N_PERP, _N_PERP] = return_array[1, 1, _N_PERP, _N_PERP]

    # N_perp N_parallel derivatives.
    return_array[0, 2, _N_PERP, _N_PARALLEL] = 1
    return_array[2, 0, _N_PERP, _N_PARALLEL] = return_array[
        0, 2, _N_PERP, _N_PARALLEL
    ]

    # N_parallel N_perp derivatives are symmetric.
    return_array[0, 2, _N_PARALLEL, _N_PERP] = return_array[
        0, 2, _N_PERP, _N_PARALLEL
    ]
    return_array[2, 0, _N_PARALLEL, _N_PERP] = return_array[
        2, 0, _N_PERP, _N_PARALLEL
    ]

    # N_parallel N_parallel derivatives.
    return_array[0, 0, _N_PARALLEL, _N_PARALLEL] = -2
    return_array[1, 1, _N_PARALLEL, _N_PARALLEL] = return_array[
        0, 0, _N_PARALLEL, _N_PARALLEL
    ]

    return return_array


def vacuum_dispersion_tensor_dq3(
    *, return_array: FloatArray = None
) -> FloatArray:
    """
    Third derivative of vacuum dispersion tensor with respect to q.

    Parameters
    ----------
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    vacuum_dispersion_tensor_dq3 : np.array[float]
        Third derivative of vacuum dispersion tensor with respect to q
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

    return_array.fill(0.0)

    return return_array


def vacuum_stix_polarisation(
    normalised_magnetic_field_strength: float,
    n_perp: float,
    n_parallel: float,
    wave_mode: WaveMode,
    /,
    *,
    return_array: ComplexArray = None,
):
    """
    Calculate stix polarisation of O and X mode in vacuum. This has a phase
    convention such that the first nonzero component is positive and real.

    Parameters
    ----------
    normalised_magnetic_field_strength : float
        Magnetic field strength normalised to resonant magnetic field strength
        aka Stix Y.
    n_perp : float
        Perpendicular refractive index.
    n_parallel : float
        Parallel refractive index.
    wave_mode : WaveMode
        Wave mode to calculate for.
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    vacuum_stix_polarisation : np.array[complex]
        Vacuum Stix frame polarisation.

    Notes
    -----
    Formula is from [1] but note there is an error in i Ex / Ey.
    The correct version is

    i Ex / Ey = (-Y * n_perp**2
        -/+ sqrt{Y**2 * n_perp**4 + 4 * n_parallel**2}) / 2

    Ex / Ez = - n_parallel / n_perp

    We can parameterise the polarisation as
        Ex = sin(theta) * sin(phi)
        Ey = i * sin(theta) * cos(phi)
        Ez = cos(theta)

    NOTE: This formula uses Y > 0!

    References
    ----------
    [1] Hansen, F.R., et al. Full-wave calculations of the O-X mode
        conversion process. Journal of Plasma Physics. 1988;39(2):319-337
    """
    return_array = get_return_array(
        return_array, (Dimensions.x.size,), ComplexType
    )

    # Limits n_perp = 0 or n_parallel = 0 usually involve 0 / 0 behaviour.
    # Take |Y| as this formula is only valid for electrons anyway.
    y = abs(normalised_magnetic_field_strength)

    if y < MIN_NORM_MAGNETIC_FIELD_STRENGTH:
        # Unmagnetised plasma.
        if wave_mode == WaveMode.O:
            return_array[0] = -n_parallel
            return_array[1] = 0.0
            return_array[2] = n_perp
        elif wave_mode == WaveMode.X:
            return_array[0] = 0.0
            return_array[1] = 1.0
            return_array[2] = 0.0
        else:
            raise NotImplementedError(wave_mode)
    elif np.isclose(n_parallel, 0.0):
        # This case is too annoying to deal with below.
        if wave_mode == WaveMode.O:
            return_array[0] = 0.0
            return_array[1] = 0.0
            return_array[2] = 1.0
        elif wave_mode == WaveMode.X:
            return_array[0] = 0.0
            return_array[1] = 1.0
            return_array[2] = 0.0
        else:
            raise NotImplementedError(wave_mode)
    else:
        y2 = y * y
        n_perp2, n_parallel2 = n_perp * n_perp, n_parallel * n_parallel
        n_perp4 = n_perp2 * n_perp2
        sign = -1.0 if wave_mode == WaveMode.O else 1.0

        # Magnetised plasma.
        _phi = np.arctan(
            0.5
            * (-y * n_perp2 + sign * np.sqrt(y2 * n_perp4 + 4 * n_parallel2))
        )

        sin_phi, cos_phi = np.sin(_phi), np.cos(_phi)

        if np.isclose(n_perp, 0.0):
            _theta = sign * 0.5 * np.pi * np.sign(n_parallel)
        else:
            _theta = np.arctan2(-n_parallel, (n_perp * sin_phi))

        sin_theta, cos_theta = np.sin(_theta), np.cos(_theta)

        return_array[0] = sin_theta * sin_phi
        return_array[1] = 1j * sin_theta * cos_phi
        return_array[2] = cos_theta

    # Apply phase convention.
    return_array *= polarisation_phase_convention_factor(return_array)

    return return_array


def calculate_harmonic_range(
    normalised_magnetic_field_strength: float,
) -> tuple[int, int]:
    """
    Calculate which cyclotron harmonics sit above and below based on the
    normalised magnetic field strength.

    Parameters
    ----------
    normalised_magnetic_field_strength : float
        Magnetic field strength normalised to resonant magnetic field strength
        aka Stix Y.

    Returns
    -------
    n_below, n_above : int
        Cyclotron harmonic numbers for resonance below and above.
    """
    y = abs(normalised_magnetic_field_strength)
    n_above = np.ceil(1 / y) if y > 0.0 else 1
    return n_above - 1, n_above


def polarisation_phase_convention_factor(
    polarisation: ComplexArray,
) -> ComplexType:
    """
    Calculate factor to enforce phase convention that first non-zero component
    of the polarisation should be positive and real.

    Parameters
    ----------
    polarisation : np.array[complex]
        Components of polarisation.

    Returns
    -------
    factor : complex
        Factor to multiply polarisation by to enforce phase convention.
    """
    factor = 1.0
    for x in polarisation:
        if abs(x) > 0.0:
            factor = np.exp(-1j * np.angle(x))
            break

    return factor


def polarisation(
    dispersion_tensor: ComplexArray,
    /,
    *,
    return_array: ComplexArray = None,
    nearby_polarisation: ComplexArray = None,
):
    """
    Find polarisation vectors (null vectors of dispersion tensor) using SVD.
    A phase convention is chosen such that the first nonzero element is both
    real and positive.

    Parameters
    ----------
    dispersion_tensor : np.array[complex]
        Dispersion tensor.
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    nearby_polarisation : np.array[complex], optional
        Polarisation close to the desired value. Used to select the correct
        polarisation in near degenerate cases where both modes are close.

    Returns
    -------
    polarisation : np.array[complex]
        Polarisation vector.

    Raises
    ------
    ValueError
        Eigenvalues are nearly degenerate and no nearby_polarisation provided.
    """
    return_array = get_return_array(
        return_array, (Dimensions.x.size,), ComplexType
    )

    # Find eigenvector with smallest eigenvalue using SVD.
    # Everything is sorted from largest to smallest abs(S).
    _, s, vh = np.linalg.svd(dispersion_tensor, full_matrices=True)

    abs_s = abs(s)
    order = np.argsort(abs_s)
    i, j = order[0], order[1]

    if 10.0 * abs_s[i] < abs_s[j]:
        return_array[:] = vh[i].T.conj().astype(ComplexType)
        return_array *= polarisation_phase_convention_factor(return_array)
        return_array /= np.linalg.norm(return_array)
    else:
        if nearby_polarisation is None:
            raise ValueError(
                "Polarisations are near degenerate. "
                "Provide nearby_polarisation to distinguish correct mode."
            )

        # Too degenerate, we might end up with wrong value.
        # Use whichever polarisation is closest to provided nearby value.
        e1 = vh[i].T.conj().astype(ComplexType)
        e1 *= polarisation_phase_convention_factor(e1)
        e1 /= np.linalg.norm(e1)
        overlap_1 = abs(np.vdot(nearby_polarisation, e1))

        e2 = vh[j].T.conj().astype(ComplexType)
        e2 *= polarisation_phase_convention_factor(e2)
        e2 /= np.linalg.norm(e2)
        overlap_2 = abs(np.vdot(nearby_polarisation, e2))

        if overlap_1 > overlap_2:
            return_array[:] = e1
        elif overlap_1 < overlap_2:
            return_array[:] = e2
        else:
            raise ValueError("Polarisations are degenerate.")

    return return_array


def eigenvalue(
    dispersion_tensor: ComplexArray, polarisation: ComplexArray, /
) -> ComplexType:
    """
    Calculate eigenvalue for given polarisation.

    Parameters
    ----------
    dispersion_tensor : np.array[complex]
        Dispersion tensor.
    polarisation : np.array[complex]
        Polarisation of given mode (an eigenvector of the dispersion tensor).

    Returns
    -------
    eigenvalue : complex
        Eigenvalue of dispersion tensor.
    """
    return np.einsum(
        "i, ij, j", np.conj(polarisation), dispersion_tensor, polarisation
    )


def eigenvalue_dx(
    dispersion_tensor_dx: ComplexArray,
    polarisation: ComplexArray,
    /,
    *,
    return_array: ComplexArray = None,
) -> ComplexArray:
    """
    Calculate first derivative of the eigenvalue for given polarisation.

    Parameters
    ----------
    dispersion_tensor_dx : np.array[complex]
        Element-wise first derivative of dispersion tensor.
    polarisation : np.array[complex]
        Polarisation of given mode (an eigenvector of the dispersion tensor).
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    eigenvalue_dx : complex
        First derivative of eigenvalue.
    """
    return_array = get_return_array(
        return_array, (dispersion_tensor_dx.shape[2],), ComplexType
    )

    np.einsum(
        "a, abi, b",
        np.conj(polarisation),
        dispersion_tensor_dx,
        polarisation,
        out=return_array,
    )

    return return_array


def eigenvector_dx(
    eigenvalues: ComplexArray,
    eigenvectors: ComplexArray,
    dispersion_tensor_dx: ComplexArray,
    index: int,
    /,
    *,
    return_array: ComplexArray = None,
):
    """
    First derivative of eigenvector of dispersion tensor.

    Parameters
    ----------
    eigenvalues : np.array[complex]
        All eigenvalues of dispersion tensor.
    eigenvectors : np.array[complex]
        Corresponding eigenvectors of dispersion tensor.
    dispersion_tensor_dx : np.array[complex]
        Element-wise first derivative of dispersion tensor.
    index : int
        Index of desired eigenvector in list of eigenvalues and eigenvectors.
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    eigenvector_dx : np.array[complex]
        First derivative of eigenvector.

    Notes
    -----
    This formula requires a hermitian dispersion tensor so all the eigenvectors
    are orthogonal.
    """
    return_array = get_return_array(
        return_array,
        (eigenvectors.shape[0], dispersion_tensor_dx.shape[2]),
        ComplexType,
    )

    # NOTE: Requires eigenvalues to be orthogonal => Hermitian matrix?
    return_array.fill(0.0)

    l0, e0 = eigenvalues[index], eigenvectors[index, :]
    for i, (_l, _e) in enumerate(zip(eigenvalues, eigenvectors, strict=False)):
        if i == index:
            continue

        return_array[:, :] += np.einsum(
            "a, abj, b, i -> ij", _e, dispersion_tensor_dx, e0, _e
        ) / (l0 - _l)

    return return_array


def eigenvalue_dx2(
    dispersion_tensor_dx: ComplexArray,
    dispersion_tensor_dx2: ComplexArray,
    polarisation: ComplexArray,
    polarisation_dx: ComplexArray,
    /,
    *,
    return_array: ComplexArray = None,
) -> ComplexArray:
    """
    Calculate second derivative of the eigenvalue for given polarisation.

    Parameters
    ----------
    dispersion_tensor_dx2 : np.array[complex]
        Element-wise second derivative of dispersion tensor.
    polarisation : np.array[complex]
        Polarisation of given mode (an eigenvector of the dispersion tensor).
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    eigenvalue_dx2 : complex
        Second derivative of eigenvalue.
    """
    _x = dispersion_tensor_dx.shape[2]
    return_array = get_return_array(return_array, (_x, _x), ComplexType)

    return_array[:, :] = (
        np.einsum(
            "a, abij, b",
            np.conj(polarisation),
            dispersion_tensor_dx2,
            polarisation,
        )
        + np.einsum(
            "aj, b, abi -> ij",
            np.conj(polarisation_dx),
            polarisation,
            dispersion_tensor_dx,
        )
        + np.einsum(
            "a, bj, abi -> ij",
            polarisation,
            np.conj(polarisation_dx),
            dispersion_tensor_dx,
        )
    )

    return return_array


def determinant(dispersion_tensor: ComplexArray) -> ComplexType:
    """
    Determinant of dispersion tensor.

    Parameters
    ----------
    dispersion_tensor : np.array[complex]
        Dispersion tensor.

    Returns
    -------
    determinant : complex
        Determinant of dispersion tensor.
    """
    return np.linalg.det(dispersion_tensor)


def determinant_dx(
    dispersion_tensor: ComplexArray,
    dispersion_tensor_dx: ComplexArray,
    /,
    *,
    dispersion_tensor_adjugate: ComplexArray = None,
    return_array: ComplexArray = None,
):
    """
    First derivative of determinant of dispersion tensor.

    Parameters
    ----------
    dispersion_tensor : np.array[complex]
        Dispersion tensor.
    dispersion_tensor_dx : np.array[complex]
        Element-wise first derivative of dispersion tensor.
    dispersion_tensor_adjugate : np.array[complex], optional
        Adjugate of dispersion tensor. If not provided, it will be calculated.
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    determinant_dx : complex
        First derivative of determinant of dispersion tensor.
    """
    if dispersion_tensor_adjugate is None:
        dispersion_tensor_adjugate = adjugate_3x3_cofactors(dispersion_tensor)

    _x = dispersion_tensor_dx.shape[2]
    return_array = get_return_array(return_array, (_x,), ComplexType)

    return_array[:] = matrix_3x3_determinant_first_derivative(
        dispersion_tensor,
        dispersion_tensor_dx,
        adjugate=dispersion_tensor_adjugate,
    )

    return return_array


def determinant_dx2(
    dispersion_tensor: ComplexArray,
    dispersion_tensor_dx: ComplexArray,
    dispersion_tensor_dx2: ComplexArray,
    /,
    *,
    dispersion_tensor_adjugate: ComplexArray = None,
    return_array: ComplexArray = None,
):
    """
    Second derivative of determinant of dispersion tensor.

    Parameters
    ----------
    dispersion_tensor : np.array[complex]
        Dispersion tensor.
    dispersion_tensor_dx : np.array[complex]
        Element-wise first derivative of dispersion tensor.
    dispersion_tensor_dx2 : np.array[complex]
        Element-wise second derivative of dispersion tensor.
    dispersion_tensor_adjugate : np.array[complex], optional
        Adjugate of dispersion tensor. If not provided, it will be calculated.
    return_array : np.array[complex], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    determinant_dx : complex
        First derivative of determinant of dispersion tensor.
    """
    if dispersion_tensor_adjugate is None:
        dispersion_tensor_adjugate = adjugate_3x3_cofactors(dispersion_tensor)

    dispersion_tensor_adjugate_dx = matrix_3x3_adjugate_first_derivative(
        dispersion_tensor, dispersion_tensor_dx
    )

    _x = dispersion_tensor_dx.shape[2]
    return_array = get_return_array(return_array, (_x, _x), ComplexType)

    return_array[:, :] = matrix_3x3_determinant_second_derivative(
        dispersion_tensor,
        dispersion_tensor_dx,
        dispersion_tensor_dx2,
        adjugate=dispersion_tensor_adjugate,
        adjugate_first_derivative=dispersion_tensor_adjugate_dx,
    )

    return return_array


class SusceptibilityCache:
    """
    Cache holding hermitian and anti-hermitian components of the
    susceptibility with derivatives of the hermitian part with respect to
    Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].

    Attributes
    ----------
    hermitian : np.array[float]
        Hermitian part of susceptibility tensor.
    hermitian_dq : np.array[float]
        First derivative of hermitian part with respect to q.
    hermitian_dq2 : np.array[float]
        Second derivative of hermitian part with respect to q.
    antihermitian_resonance : np.array[float]
        Anti-hermitian part of susceptibility tensor associated with
        cyclotron damping.
    antihermitian_collisional : np.array[float]
        Anti-hermitian part of susceptibility tensor associated with
        collisional damping.
    """

    __slots__ = (
        "antihermitian_collisional",
        "antihermitian_resonance",
        "hermitian",
        "hermitian_dq",
        "hermitian_dq2",
    )

    def __init__(self):
        """
        Inits SusceptibilityCache.
        """
        self.hermitian = np.empty(
            (Dimensions.x.size, Dimensions.x.size), dtype=ComplexType
        )
        self.hermitian_dq = np.empty(
            (Dimensions.x.size, Dimensions.x.size, Dimensions.q.size),
            dtype=ComplexType,
        )
        self.hermitian_dq2 = np.empty(
            (
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.q.size,
                Dimensions.q.size,
            ),
            dtype=ComplexType,
        )
        self.antihermitian_resonance = np.empty(
            (Dimensions.x.size, Dimensions.x.size), dtype=ComplexType
        )
        self.antihermitian_collisional = np.empty(
            (Dimensions.x.size, Dimensions.x.size), dtype=ComplexType
        )


class DispersionModel(abc.ABC):
    """
    Base class for dispersion model which provides models for the plasma
    electric susceptibility.

    Attributes
    ----------
    dispersion_type : DispersionType
        Dispersion model type this object implements.

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

    dispersion_type: DispersionType = NotImplemented

    @classmethod
    @abc.abstractmethod
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

    @classmethod
    @abc.abstractmethod
    def susceptibility_antihermitian_resonance(
        cls,
        q: FloatArray,
        /,
        *,
        return_array: ComplexArray = None,
    ) -> ComplexArray:
        """
        Anti-hermitian part of electric susceptibility tensor chi_ah due to
        resonant absorption.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        chi_ah_resonance : np.array[complex]
            Anti-hermitian part of electric susceptibility due to resonant
            absorption.
        """

    @classmethod
    @abc.abstractmethod
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

    @classmethod
    @abc.abstractmethod
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

    @classmethod
    @abc.abstractmethod
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

    @classmethod
    @abc.abstractmethod
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

    @classmethod
    def dispersion_tensor(
        cls, q: FloatArray, /, return_array: ComplexArray = None, **kwargs
    ):
        """
        Helper method to evaluate the full dispersion tensor.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        return_array : np.array[complex]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        dispersion_tensor : np.array[complex]
            Dispersion tensor.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), ComplexType
        )

        vacuum_dispersion_tensor(q, return_array=return_array)
        return_array[:, :] += cls.susceptibility_hermitian(q, **kwargs)

        return return_array

    @classmethod
    @abc.abstractmethod
    def calculate_n(
        cls, q: FloatArray, wave_mode: WaveMode, /, *, kinetic: bool
    ) -> FloatType:
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

    @classmethod
    @abc.abstractmethod
    def calculate_n_perp(
        cls, q: FloatArray, wave_mode: WaveMode, /, *, kinetic: bool
    ) -> FloatType:
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
