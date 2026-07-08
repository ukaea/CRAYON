"""
Methods for calculating vector perpendicular and parallel components, magnitude
and unit vectors including their derivatives.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


# Vector magnitude and first and second derivatives.
def vector_magnitude(v: FloatArray) -> float:
    """
    Calculate magnitude of a vector of size n in Cartesian coordinates.

    Parameters
    ----------
    v : np.array[float]
        Components of vector with shape (n,).

    Returns
    -------
    magnitude : float
        Magnitude of vector.
    """
    return np.linalg.norm(v)


def vector_magnitude_first_derivative_x(
    v: FloatArray,
    v_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_magnitude: float | None = None,
) -> FloatArray:
    """
    Calculate first derivative of the magnitude of a vector of size n with
    respect to an parameter x of size m in Cartesian coordinates.

    Parameters
    ----------
    v : np.array[float]
        Vector field components with shape (n,).
    v_dx : np.array[float]
        First derivative of vector components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_magnitude : float, optional
        Magnitude of vector. If not provided, it will be calculated.

    Returns
    -------
    magnitude_first_derivative : np.array[float]
        First derivative of magnitude of vector with respect to x with shape
        (m,).

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_magnitude is None:
        v_magnitude = vector_magnitude(v)

    _m = v_dx.shape[-1]
    return_array = get_return_array(return_array, (_m,), FloatType)

    if np.isclose(v_magnitude, 0.0):
        # Magnitude of vector is zero. Assume first derivative also vanishes
        return_array[:] = 0.0
    else:
        return_array[:] = np.einsum("j, ji -> i", v, v_dx) / v_magnitude

    return return_array


def vector_magnitude_second_derivative_x(
    v: FloatArray,
    v_dx: FloatArray,
    v_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_magnitude: float | None = None,
    v_magnitude_dx: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of the magnitude of a vector of size n with
    respect to an parameter x of size m in Cartesian coordinates.

    Parameters
    ----------
    v : np.array[float]
        Vector field components with shape (n,).
    v_dx : np.array[float]
        First derivative of vector components with shape (n, m).
    v_dx2 : np.array[float]
        Second derivative of vector components with shape (n, m, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_magnitude : float
        Magnitude of vector. If not provided, it will be calculated.
    v_magnitude_dx : np.array[float]
        First derivative of vector magnitude with shape (m,). If not provided,
        it will be calculated.

    Returns
    -------
    magnitude_second_derivative : np.array[float]
        Second derivative of magnitude of vector with respect to x with shape
        (m,) m.

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_magnitude is None:
        v_magnitude = vector_magnitude(v)

    if v_magnitude_dx is None:
        v_magnitude_dx = vector_magnitude_first_derivative_x(
            v, v_dx, v_magnitude=v_magnitude
        )

    _m = v_dx.shape[-1]
    return_array = get_return_array(return_array, (_m, _m), FloatType)

    if np.isclose(v_magnitude, 0.0):
        # Magnitude of vector is zero. Assume second derivative also vanishes.
        return_array[:] = 0.0
    else:
        return_array[:, :] = (
            np.einsum("k, kij -> ij", v, v_dx2)
            + np.einsum("ki, kj -> ij", v_dx, v_dx)
            - np.einsum("i, j -> ij", v_magnitude_dx, v_magnitude_dx)
        ) / v_magnitude

    return return_array


# Unit vector and its derivatives with respect to space.
def unit_vector(
    v: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_magnitude: float | None = None,
):
    """
    Calculate vector of size n normalised to unit length.

    Parameters
    ----------
    v : np.array[float]
        Vector field components of shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_magnitude : float, optional
        Magnitude of vector. If not provided, it will be calculated.

    Returns
    -------
    unit_vector : np.array[float]
        v normalised to unit length of shape (n,).

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_magnitude is None:
        v_magnitude = vector_magnitude(v)

    return_array = get_return_array(return_array, (v.size,), v.dtype)

    if np.isclose(v_magnitude, 0.0):
        # Magnitude of vector is zero. Set unit to zero.
        return_array[:] = 0.0
    else:
        return_array[:] = v / v_magnitude

    return return_array


def unit_vector_first_derivative_x(
    v: FloatArray,
    v_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_magnitude: float | None = None,
    v_magnitude_dx: FloatArray = None,
    v_unit: FloatArray = None,
) -> FloatArray:
    """
    Calculate first derivative of vector of size n normalised to unit length.
    with respect to a parameter x of size m in Cartesian coordinates.

    Parameters
    ----------
    v : np.array[float]
        Vector field components with shape (n,).
    v_dx : np.array[float]
        First derivative of vector components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_magnitude : float, optional
        Magnitude of vector. If not provided, it will be calculated.
    v_magnitude_dx : np.array[float], optional
        First derivative of vector magnitude with shape (m,). If not provided,
        it will be calculated.
    v_unit : np.array[float], optional
        Vector normalised to unit length with shape (n,). If not provided, it
        will be calculated.

    Returns
    -------
    unit_vector_first_derivative : np.array[float]
        First derivative of unit vector with respect to x with shape (n, m).

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_magnitude is None:
        v_magnitude = vector_magnitude(v)

    if v_magnitude_dx is None:
        v_magnitude_dx = vector_magnitude_first_derivative_x(
            v, v_dx, v_magnitude=v_magnitude
        )

    if v_unit is None:
        v_unit = unit_vector(v, v_magnitude=v_magnitude)

    n = v_dx.shape[-1]
    return_array = get_return_array(return_array, (v.size, n), v.dtype)

    if np.isclose(v_magnitude, 0.0):
        # Magnitude of vector is zero. Set unit first derivative to zero.
        return_array[:, :] = 0.0
    else:
        return_array[:, :] = (
            v_dx - np.outer(v_unit, v_magnitude_dx)
        ) / v_magnitude

    return return_array


def unit_vector_second_derivative_x(
    v: FloatArray,
    v_dx: FloatArray,
    v_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_magnitude: float | None = None,
    v_magnitude_dx: FloatArray = None,
    v_magnitude_dx2: FloatArray = None,
    v_unit: FloatArray = None,
    v_unit_dx: FloatArray = None,
):
    """
    Calculate first derivative of vector of size n normalised to unit length.
    with respect to a parameter x of size m in Cartesian coordinates.

    Parameters
    ----------
    v : np.array[float]
        Vector field components with shape (n,).
    v_dx : np.array[float]
        First derivative of vector components with shape (n, m).
    v_dx2 : np.array[float]
        Second derivative of vector components with shape (n, m, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_magnitude : float, optional
        Magnitude of vector. If not provided, it will be calculated.
    v_magnitude_dx : np.array[float], optional
        First derivative of vector magnitude with shape (m,). If not provided,
        it will be calculated.
    v_magnitude_dx2 : np.array[float], optional
        First derivative of vector magnitude with shape (m, m). If not
        provided, it will be calculated.
    v_unit : np.array[float], optional
        Vector normalised to unit length with shape (n,). If not provided, it
        will be calculated.
    v_unit_dx : np.array[float], optional
        First derivaive of unit vector with shape (n, m). If not provided, it
        will be calculated.

    Returns
    -------
    unit_vector_first_derivative : np.array[float]
        Second derivative of unit vector with respect to x with shape
        (n, m, m).

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_magnitude is None:
        v_magnitude = vector_magnitude(v)

    if v_magnitude_dx is None:
        v_magnitude_dx = vector_magnitude_first_derivative_x(
            v, v_dx, v_magnitude=v_magnitude
        )

    if v_magnitude_dx2 is None:
        v_magnitude_dx2 = vector_magnitude_second_derivative_x(
            v,
            v_dx,
            v_dx2,
            v_magnitude=v_magnitude,
            v_magnitude_dx=v_magnitude_dx,
        )

    if v_unit is None:
        v_unit = unit_vector(v, v_magnitude=v_magnitude)

    if v_unit_dx is None:
        v_unit_dx = unit_vector_first_derivative_x(
            v,
            v_dx,
            v_magnitude=v_magnitude,
            v_magnitude_dx=v_magnitude_dx,
            v_unit=v_unit,
        )

    _m = v_dx.shape[-1]
    return_array = get_return_array(return_array, (v.size, _m, _m), v.dtype)

    if np.isclose(v_magnitude, 0.0):
        # Magnitude of vector is zero. Set unit second derivative to zero.
        return_array[:, :, :] = 0.0
    else:
        return_array[:, :, :] = (
            v_dx2
            - np.einsum("i, jk -> ijk", v_unit, v_magnitude_dx2)
            - np.einsum("ik, j -> ijk", v_unit_dx, v_magnitude_dx)
            - np.einsum("ij, k -> ijk", v_unit_dx, v_magnitude_dx)
        ) / v_magnitude

    return return_array


# Components of vector with respect to a unit vector.
def component_perp(
    v: FloatArray, n: FloatArray, /, *, v_parallel: float | None = None
) -> float:
    """
    Calculate magnitude of component of a vector v of size n perpendicular to
    a unit vector n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    component_perp : float
        Magnitude of component of vector perpendicular to normal.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    return np.sqrt(component_perp2(v, n, v_parallel=v_parallel))


def component_perp2(
    v: FloatArray, n: FloatArray, /, *, v_parallel: float | None = None
) -> float:
    """
    Calculate squared magnitude of component of a vector v of size n
    perpendicular to a unit vector n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    component_perp2 : float
        Squared magnitude of component of vector perpendicular to normal.

    Notes
    -----
    This function will never return a negative number. This is to avoid issues
    where due to numerical error v_parallel**2 > |v|**2.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    return max(0, np.dot(v, v) - v_parallel * v_parallel)


def component_parallel(v: FloatArray, n: FloatArray) -> float:
    """
    Calculate magnitude of component of a vector v of size n parallel to a
    unit vector n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).

    Returns
    -------
    component_parallel : float
        Magnitude of component of vector parallel to normal.
    """
    return np.dot(v, n)


def component_parallel2(v: FloatArray, n: FloatArray) -> float:
    """
    Calculate squared magnitude of component of a vector v of size n parallel
    to a unit vector n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).

    Returns
    -------
    component_parallel2 : float
        Squared magnitude of component of vector parallel to normal.
    """
    return np.square(component_parallel(v, n))


# First derivatives with respect to independent coordinate.
def v_perp_first_derivative_x(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_perp: float | None = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray | None = None,
) -> FloatArray:
    """
    Calculate first derivative of v_perp with respect to a parameter x of
    size m. v_perp is the magnitude of the component of v perpendicular to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_perp : float, optional
        Magnitude of component of v perpendicular to normal. If not provided,
        it will be calculated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).

    Returns
    -------
    v_perp_dx : float
        First derivative of v_perp with respect to x.

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    if v_perp is None:
        v_perp = component_perp(v, n, v_parallel=v_parallel)

    if v_parallel_dx is None:
        v_parallel_dx = v_parallel_first_derivative_x(v, n_dx)

    m = n_dx.shape[1]
    return_array = get_return_array(return_array, (m,), v.dtype)

    if np.isclose(v_perp, 0.0):
        # v_perp is zero. Assume first derivative wrt x is also zero.
        return_array[:] = 0.0
    else:
        return_array[:] = -(v_parallel / v_perp) * v_parallel_dx

    return return_array


def v_perp2_first_derivative_x(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray = None,
    v_parallel2_dx: FloatArray = None,
) -> FloatArray:
    """
    Calculate first derivative of v_perp squared with respect to a parameter x
    of size m. v_perp is the magnitude of the component of v perpendicular to
    n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).
    v_parallel2_dx : np.array[float], optional
        First derivative of v_parallel squared with respect to x with shape
        (m,).

    Returns
    -------
    v_perp2_dx : float
        First derivative of v_perp squared with respect to x.
    """
    if v_parallel2_dx is None:
        v_parallel2_dx = v_parallel2_first_derivative_x(
            v, n, n_dx, v_parallel=v_parallel, v_parallel_dx=v_parallel_dx
        )

    _m = n_dx.shape[1]
    return_array = get_return_array(return_array, (_m,), v.dtype)
    return_array[:] = -v_parallel2_dx

    return return_array


def v_parallel_first_derivative_x(
    v: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate first derivative of v_parallel with respect to a parameter x of
    size m. v_parallel is the magnitude of the component of v parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_parallel_dx : float
        First derivative of v_parallel with respect to x.
    """
    _m = n_dx.shape[1]
    return_array = get_return_array(return_array, (_m,), v.dtype)
    return_array[:] = np.einsum("i, ij", v, n_dx)

    return return_array


def v_parallel2_first_derivative_x(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray = None,
) -> FloatArray:
    """
    Calculate first derivative of v_parallel squared with respect to a
    parameter x of size m. v_parallel is the magnitude of the component of v
    parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).

    Returns
    -------
    v_parallel2_dx : float
        First derivative of v_parallel squared with respect to x.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    if v_parallel_dx is None:
        v_parallel_dx = v_parallel_first_derivative_x(v, n_dx)

    _m = n_dx.shape[1]
    return_array = get_return_array(return_array, (_m,), v.dtype)

    return_array[:] = 2 * v_parallel * v_parallel_dx

    return return_array


# First derivatives with respect to vector components.
def v_perp_first_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_perp: float | None = None,
    v_parallel: float | None = None,
) -> FloatArray:
    """
    Calculate first derivative of v_perp with respect to vector components v.
    v_perp is the magnitude of the component of v perpendicular to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_perp : float, optional
        Magnitude of component of v perpendicular to normal. If not provided,
        it will be calculated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    v_perp_dv : float
        First derivative of v_perp with respect to v.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)
    if v_perp is None:
        v_perp = component_perp(v, n, v_parallel=v_parallel)

    return_array = get_return_array(return_array, v.shape, v.dtype)

    if np.isclose(v_perp, 0.0):
        # v_perp is zero. Assume first derivative wrt v is also zero.
        return_array[:] = 0.0
    else:
        return_array[:] = (v - v_parallel * n) / v_perp

    return return_array


def v_perp2_first_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
) -> FloatArray:
    """
    Calculate first derivative of v_perp squared with respect to vector
    components v. v_perp is the magnitude of the component of v perpendicular
    to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    v_perp2_dv : float
        First derivative of v_perp squared with respect to v.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    return_array = get_return_array(return_array, v.shape, v.dtype)
    return_array[:] = 2 * (v - v_parallel * n)

    return return_array


def v_parallel2_first_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
) -> FloatArray:
    """
    Calculate first derivative of v_parallel squared with respect to vector
    components v. v_parallel is the magnitude of the component of v parallel
    to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    v_parallel2_dv : float
        First derivative of v_parallel squared with respect to v.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    return_array = get_return_array(return_array, v.shape, v.dtype)
    return_array[:] = 2 * v_parallel * n

    return return_array


def v_parallel_first_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate first derivative of v_parallel with respect to vector components
    v. v_parallel is the magnitude of the component of v parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_parallel_dv : float
        First derivative of v_parallel with respect to v.
    """
    return_array = get_return_array(return_array, v.shape, v.dtype)
    return_array[:] = n

    return n


# Second derivatives with respect to independent coordinate.
def v_perp_second_derivative_x(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    n_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_perp: float | None = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray = None,
    v_parallel_dx2: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of v_perp with respect to a parameter x of
    size m. v_perp is the magnitude of the component of v perpendicular to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    n_dx2 : np.array[float]
        Second derivative of normal components with shape (n, m, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_perp : float, optional
        Magnitude of component of v perpendicular to normal. If not provided,
        it will be calculated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).
    v_parallel_dx2 : np.array[float], optional
        Second derivative of v_parallel with respect to x with shape (m, m).

    Returns
    -------
    v_perp_dx2 : float
        Second derivative of v_perp with respect to x.

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    if v_perp is None:
        v_perp = component_perp(v, n, v_parallel=v_parallel)

    if v_parallel_dx is None:
        v_parallel_dx = v_parallel_first_derivative_x(v, n_dx)

    if v_parallel_dx2 is None:
        v_parallel_dx2 = v_parallel_second_derivative_x(v, n_dx2)

    _m = n_dx.shape[1]
    return_array = get_return_array(return_array, (_m, _m), v.dtype)

    if np.isclose(v_perp, 0.0):
        # v_perp is zero. Assume second derivative wrt x is also zero.
        return_array[:, :] = 0.0
    else:
        a = v_parallel / v_perp
        return_array[:, :] = -(
            (1 + a**2) * np.outer(v_parallel_dx, v_parallel_dx) / v_perp
            + a * v_parallel_dx2
        )

    return return_array


def v_perp2_second_derivative_x(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    n_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray = None,
    v_parallel_dx2: FloatArray = None,
    v_parallel2_dx2: FloatArray = None,
):
    """
    Calculate second derivative of v_perp squared with respect to a parameter
    x of size m. v_perp is the magnitude of the component of v perpendicular
    to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    n_dx2 : np.array[float]
        Second derivative of normal components with shape (n, m, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).
    v_parallel_dx2 : np.array[float], optional
        Second derivative of v_parallel with respect to x with shape (m, m).
    v_parallel2_dx2 : np.array[float], optional
        Second derivative of v_parallel squared with respect to x with shape
        (m, m).

    Returns
    -------
    v_perp_dx2 : float
        Second derivative of v_perp squared with respect to x.

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_parallel2_dx2 is None:
        v_parallel2_dx2 = v_parallel2_second_derivative_x(
            v,
            n,
            n_dx,
            n_dx2,
            v_parallel=v_parallel,
            v_parallel_dx=v_parallel_dx,
            v_parallel_dx2=v_parallel_dx2,
        )

    _m = n_dx.shape[1]
    return_array = get_return_array(return_array, (_m, _m), v.dtype)
    return_array[:] = -v_parallel2_dx2

    return return_array


def v_parallel2_second_derivative_x(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    n_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray = None,
    v_parallel_dx2: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of v_parallel squared with respect to a
    parameter x of size m. v_parallel is the magnitude of the component of
    v parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    n_dx2 : np.array[float]
        Second derivative of normal components with shape (n, m, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).
    v_parallel_dx2 : np.array[float], optional
        Second derivative of v_parallel with respect to x with shape (m, m).

    Returns
    -------
    v_parallel2_dx : float
        Second derivative of v_parallel squared with respect to x.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    if v_parallel_dx is None:
        v_parallel_dx = v_parallel_first_derivative_x(v, n_dx)

    if v_parallel_dx2 is None:
        v_parallel_dx2 = v_parallel_second_derivative_x(v, n_dx2)

    _m = n_dx.shape[1]
    return_array = get_return_array(return_array, (_m, _m), v.dtype)

    return_array[:, :] = 2 * (
        v_parallel * v_parallel_dx2 + np.outer(v_parallel_dx, v_parallel_dx)
    )

    return return_array


def v_parallel_second_derivative_x(
    v: FloatArray,
    n_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of v_parallel with respect to a parameter x of
    size m. v_parallel is the magnitude of the component of v parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n_dx2 : np.array[float]
        Second derivative of normal components with shape (n, m, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_parallel_dx2 : float
        Second derivative of v_parallel with respect to x.
    """
    _m = n_dx2.shape[1]
    return_array = get_return_array(return_array, (_m, _m), v.dtype)
    return_array[:, :] = np.einsum("k, kij -> ij", v, n_dx2)

    return return_array


# Second derivatives with respect to vector components.
def v_perp_second_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_perp: float | None = None,
    v_parallel: float | None = None,
) -> FloatArray:
    """
    Calculate second derivative of v_perp with respect to vector components v.
    v_perp is the magnitude of the component of v perpendicular to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_perp : float, optional
        Magnitude of component of v perpendicular to normal. If not provided,
        it will be calculated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    v_perp_dv2 : float
        Second derivative of v_perp with respect to v.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    if v_perp is None:
        v_perp = component_perp(v, n, v_parallel=v_parallel)

    _m = v.size
    return_array = get_return_array(return_array, (_m, _m), v.dtype)

    if np.isclose(v_perp, 0.0):
        # v_perp is zero. Assume first derivative wrt v is also zero.
        return_array[:, :] = 0.0
    else:
        v_perp_vector = v - v_parallel * n
        return_array[:, :] = (
            np.identity(_m) - np.outer(n, n)
        ) / v_perp - np.outer(v_perp_vector, v_perp_vector) / v_perp**3

    return return_array


def v_perp2_second_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate second derivative of v_perp squared with respect to vector
    components v. v_perp is the magnitude of the component of v perpendicular
    to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_perp2_dv2 : float
        Second derivative of v_perp squared with respect to v.
    """
    _m = v.size
    return_array = get_return_array(return_array, (_m, _m), v.dtype)
    return_array[:, :] = 2 * (np.identity(_m) - np.outer(n, n))

    return return_array


def v_parallel2_second_derivative_v(
    v: FloatArray,
    n: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of v_parallel squared with respect to vector
    components v. v_parallel is the magnitude of the component of v parallel
    to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_parallel2_dv2 : float
        First derivative of v_parallel squared with respect to v.
    """
    _m = v.size
    return_array = get_return_array(return_array, (_m, _m), v.dtype)
    return_array[:, :] = 2 * np.outer(n, n)

    return return_array


def v_parallel_second_derivative_v(
    v: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of v_parallel with respect to vector
    components v. v_parallel is the magnitude of the component of v parallel
    to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_parallel_dv2 : float
        Second derivative of v_parallel with respect to v.
    """
    _m = v.size
    return_array = get_return_array(return_array, (_m, _m), v.dtype)
    return_array[:, :] = 0.0

    return return_array


# Second derivatives with respect to mixed independent coordinate and
# vector component.
def v_perp_second_derivative_xv(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_perp: float | None = None,
    v_parallel: float | None = None,
    v_parallel_dx: FloatArray = None,
) -> FloatArray:
    """
    Calculate second mixed derivative of v_perp with respect to a parameter x
    of size m and vector components v of size n. v_perp is the magnitude of
    the component of v perpendicular to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_perp : float, optional
        Magnitude of component of v perpendicular to normal. If not provided,
        it will be calculated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel_dx : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).

    Returns
    -------
    v_perp_dxdv : float
        Second derivative of v_perp with respect to x and v with shape (m, n).

    Notes
    -----
    If the magnitude of the vector is zero it will return zero.
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    if v_perp is None:
        v_perp = component_perp(v, n, v_parallel=v_parallel)

    if v_parallel_dx is None:
        v_parallel_dx = v_parallel_first_derivative_x(v, n_dx)

    _m1, _m2 = n_dx.shape[1], v.size
    return_array = get_return_array(return_array, (_m1, _m2), v.dtype)

    if np.isclose(v_perp, 0.0):
        # v_perp is zero. Assume second mixed derivative wrt x, v is also zero.
        return_array[:, :] = 0.0
    else:
        long_term = (1 + v_parallel**2 / v_perp**2) * n - (
            v_parallel / v_perp**2
        ) * v

        return_array[:, :] = (
            -(v_parallel * n_dx.T + np.outer(v_parallel_dx, long_term))
            / v_perp
        )

    return return_array


def v_perp2_second_derivative_xv(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
    v_parallel2_dxdv: FloatArray = None,
) -> FloatArray:
    """
    Calculate second mixed derivative of v_perp squared with respect to a
    parameter x of size m and vector components v of size n. v_perp is the
    magnitude of the component of v perpendicular to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n : np.array[float]
        Normal components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.
    v_parallel2_dxdv : np.array[float], optional
        First derivative of v_parallel with respect to x with shape (m,).

    Returns
    -------
    v_perp2_dxdv : float
        Second derivative of v_perp squared with respect to x and v with shape
        (m, n).
    """
    if v_parallel2_dxdv is None:
        v_parallel2_dxdv = v_parallel2_second_derivative_xv(
            v, n, n_dx, v_parallel=v_parallel
        )

    _m1, _m2 = n_dx.shape[1], v.size
    return_array = get_return_array(return_array, (_m1, _m2), v.dtype)
    return_array[:, :] = -v_parallel2_dxdv

    return return_array


def v_parallel_second_derivative_xv(
    v: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate second mixed derivative of v_parallel with respect to a
    parameter x of size m and vector components v of size n. v_parallel is the
    magnitude of the component of v parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    v_parallel_dxdv : float
        Second derivative of v_parallel with respect to x and v with shape
        (m, n).
    """
    m, n = n_dx.shape[1], v.size
    return_array = get_return_array(return_array, (m, n), v.dtype)
    return_array[:, :] = n_dx.T

    return return_array


def v_parallel2_second_derivative_xv(
    v: FloatArray,
    n: FloatArray,
    n_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    v_parallel: float | None = None,
) -> FloatArray:
    """
    Calculate second mixed derivative of v_parallel squared with respect to a
    parameter x of size m and vector components v of size n. v_parallel is the
    magnitude of the component of v parallel to n.

    Parameters
    ----------
    v : np.array[float]
        Vector components with shape (n,).
    n_dx : np.array[float]
        First derivative of normal components with shape (n, m).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    v_parallel : float, optional
        Magnitude of component of v parallel to normal. If not provided, it
        will be calculated.

    Returns
    -------
    v_parallel2_dxdv : float
        Second derivative of v_parallel squared with respect to x and v with
        shape (m, n).
    """
    if v_parallel is None:
        v_parallel = component_parallel(v, n)

    _m1, _m2 = n_dx.shape[1], v.size
    return_array = get_return_array(return_array, (_m1, _m2), float)

    return_array[:, :] = 2 * (
        v_parallel * n_dx.T + np.einsum("k, ki, j -> ij", v, n_dx, n)
    )

    return return_array


def rotation_a_onto_b(
    a: FloatArray, b: FloatArray, /, *, normalised: bool = False
):
    """
    Calculate rotation matrix R that maps direction of a onto b i.e.
    b_hat = R @ a_hat where a_hat and b_hat are a and b normalised to unit
    length.

    Parameters
    ----------
    a : np.array[float]
        Components of start vector.
    b : np.array[float]
        Components of end vector.
    normalised : bool, optional
        If True, indiciates a and b are already normalised.

    Returns
    -------
    rotation_matrix : np.array[float]
        Rotation matrix which maps a onto b.
    """
    if not normalised:
        a = np.copy(a) / np.linalg.norm(a)
        b = np.copy(b) / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(1.0 + c, 0.0, atol=1e-4):
        # Formula has singularity when vectors anti-parallel.
        return -np.identity(3)

    cross = np.zeros((3, 3))
    cross[0, 1] = -v[2]
    cross[0, 2] = v[1]
    cross[1, 0] = v[2]
    cross[1, 2] = -v[0]
    cross[2, 0] = -v[1]
    cross[2, 1] = v[0]

    return c * np.identity(3) + cross + np.outer(v, v) / (1 + c)
