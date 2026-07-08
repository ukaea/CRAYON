"""
Methods for calculating derivatives of composite functions.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)


def first_derivative(
    f_dg: FloatArray,
    g_dx: FloatArray,
    f_shape: tuple[int],
    g_size: int,
    x_size: int,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate first derivative of a function f(g(x)). This is vectorised and
    can accept an array of values of f where additional dimensions must be
    given in the first axes.

    f is a tensor valued function with shape (..., a1, ..., ak) where the
    first ... are the additional dimensions.
    g is a vector of shape (m,).
    x is a vector of shape (n,).

    Parameters
    ----------
    f_dg : np.array[float]
        First derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m).
    g_dx : np.array[float]
        First derivative of g with respect to x.
        Must have shape (..., m, n).
    f_shape : tuple[int]
        Shape of f (a1, ..., ak).
    g_size: int
        Number of arguments in g.
    x_size : int
        Number of arguments in x.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    first_derivative : np.array[float]
        First derivative of f with respect to x.
        Has shape (..., a1, ..., ak, n).
    """
    _f_dg = np.asarray(f_dg)
    _g_dx = np.asarray(g_dx)

    # Check shapes.
    _f = tuple(int(d) for d in f_shape)
    _g = max(1, int(g_size))
    _x = max(1, int(x_size))

    core_dim = len(_f) + _g
    extra_dims = _f_dg.shape[:-core_dim]
    _n = np.prod(extra_dims).astype(int)

    # Flatten over extra dimensions.
    _f_dg = np.reshape(_f_dg, (_n, *_f, _g))
    _g_dx = np.reshape(_g_dx, (_n, _g, _x))

    # Calculate value.
    return_array = get_return_array(
        return_array,
        (*extra_dims, *_f, _x),
        _f_dg.dtype,
    )

    # einsum value is flattened over extra dimensions so need to reshape to
    # match input.
    return_array[..., :] = np.reshape(
        np.einsum("...a, ...ai -> ...i", _f_dg, _g_dx), (*extra_dims, *_f, _x)
    )

    return return_array


def second_derivative(
    f_dg: FloatArray,
    f_dg2: FloatArray,
    g_dx: FloatArray,
    g_dx2: FloatArray,
    f_shape: tuple[int],
    g_size: int,
    x_size: int,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate second derivative of a function f(g(x)). This is vectorised and
    can accept an array of values of f where additional dimensions must be
    given in the first axes.

    f is a tensor valued function with shape (..., a1, ..., ak) where the
    first ... are the additional dimensions.
    g is a vector of shape (m,).
    x is a vector of shape (n,).

    Parameters
    ----------
    f_dg : np.array[float]
        First derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m).
    f_dg2 : np.array[float]
        Second derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m, m).
    g_dx : np.array[float]
        First derivative of g with respect to x.
        Must have shape (..., m, n).
    g_dx2 : np.array[float]
        Second derivative of g with respect to x.
        Must have shape (..., m, n, n).
    f_shape : tuple[int]
        Shape of f (a1, ..., ak).
    g_size: int
        Number of arguments in g.
    x_size : int
        Number of arguments in x.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    second_derivative : np.array[float]
        Second derivative of f with respect to x.
        Has shape (..., a1, ..., ak, n, n).
    """
    _f_dg = np.asarray(f_dg)
    _f_dg2 = np.asarray(f_dg2)
    _g_dx = np.asarray(g_dx)
    _g_dx2 = np.asarray(g_dx2)

    # Check shapes.
    _f = tuple(int(d) for d in f_shape)
    _g = max(1, int(g_size))
    _x = max(1, int(x_size))

    core_dim = len(_f) + 1
    extra_dims = _f_dg.shape[:-core_dim]
    _n = np.prod(extra_dims).astype(int)

    # Flatten over extra dimensions.
    _f_dg = np.reshape(_f_dg, (_n, *_f, _g))
    _g_dx = np.reshape(_g_dx, (_n, _g, _x))
    _f_dg2 = np.reshape(_f_dg2, (_n, *_f, _g, _g))
    _g_dx2 = np.reshape(_g_dx2, (_n, _g, _x, _x))

    # Calculate value.
    return_array = get_return_array(
        return_array,
        (*extra_dims, *_f, _x, _x),
        _f_dg.dtype,
    )

    return_array[..., :, :] = np.reshape(
        (
            np.einsum("...ab, ...ai, ...bj -> ...ij", _f_dg2, _g_dx, _g_dx)
            + np.einsum("...a, ...aij -> ...ij", _f_dg, _g_dx2)
        ),
        (*extra_dims, *_f, _x, _x),
    )

    # Reshape extra dimensions of return value to match input
    return return_array


def third_derivative(
    f_dg: FloatArray,
    f_dg2: FloatArray,
    f_dg3: FloatArray,
    g_dx: FloatArray,
    g_dx2: FloatArray,
    g_dx3: FloatArray,
    f_shape: tuple[int],
    g_size: int,
    x_size: int,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate third derivative of a function f(g(x)). This is vectorised and
    can accept an array of values of f where additional dimensions must be
    given in the first axes.

    f is a tensor valued function with shape (..., a1, ..., ak) where the
    first ... are the additional dimensions.
    g is a vector of shape (m,).
    x is a vector of shape (n,).

    Parameters
    ----------
    f_dg : np.array[float]
        First derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m).
    f_dg2 : np.array[float]
        Second derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m, m).
    f_dg3 : np.array[float]
        Third derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m, m, m).
    g_dx : np.array[float]
        First derivative of g with respect to x.
        Must have shape (..., m, n).
    g_dx2 : np.array[float]
        Second derivative of g with respect to x.
        Must have shape (..., m, n, n).
    g_dx3 : np.array[float]
        Third derivative of g with respect to x.
        Must have shape (..., m, n, n, n).
    f_shape : tuple[int]
        Shape of f (a1, ..., ak).
    g_size: int
        Number of arguments in g.
    x_size : int
        Number of arguments in x.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    third_derivative : np.array[float]
        Third derivative of f with respect to x.
        Has shape (..., a1, ..., ak, n, n, n).
    """
    _f_dg = np.asarray(f_dg)
    _f_dg2 = np.asarray(f_dg2)
    _f_dg3 = np.asarray(f_dg3)
    _g_dx = np.asarray(g_dx)
    _g_dx2 = np.asarray(g_dx2)
    _g_dx3 = np.asarray(g_dx3)

    # Check shapes.
    _f = tuple(int(d) for d in f_shape)
    _g = max(1, int(g_size))
    _x = max(1, int(x_size))

    core_dim = len(_f) + 1
    extra_dims = _f_dg.shape[:-core_dim]
    _n = np.prod(extra_dims).astype(int)

    # Flatten over extra dimensions.
    _f_dg = np.reshape(_f_dg, (_n, *_f, _g))
    _g_dx = np.reshape(_g_dx, (_n, _g, _x))
    _f_dg2 = np.reshape(_f_dg2, (_n, *_f, _g, _g))
    _g_dx2 = np.reshape(_g_dx2, (_n, _g, _x, _x))
    _f_dg3 = np.reshape(_f_dg3, (_n, *_f, _g, _g, _g))
    _g_dx3 = np.reshape(_g_dx3, (_n, _g, _x, _x, _x))

    # Calculate value.
    return_array = get_return_array(
        return_array,
        (*extra_dims, *_f, _x, _x, _x),
        _f_dg.dtype,
    )

    return_array[..., :, :, :] = np.reshape(
        (
            np.einsum(
                "...abc, ...ai, ...bj, ...ck -> ...ijk",
                _f_dg3,
                _g_dx,
                _g_dx,
                _g_dx,
            )
            + np.einsum(
                "...ab, ...aik, ...bj -> ...ijk", _f_dg2, _g_dx2, _g_dx
            )
            + np.einsum(
                "...ab, ...ai, ...bjk -> ...ijk", _f_dg2, _g_dx, _g_dx2
            )
            + np.einsum(
                "...ab, ...aij, ...bk -> ...ijk", _f_dg2, _g_dx2, _g_dx
            )
            + np.einsum("...a, ...aijk -> ...ijk", _f_dg, _g_dx3)
        ),
        (*extra_dims, *_f, _x, _x, _x),
    )

    # Reshape extra dimensions of return value to match input
    return return_array


def fourth_derivative(
    f_dg: FloatArray,
    f_dg2: FloatArray,
    f_dg3: FloatArray,
    f_dg4: FloatArray,
    g_dx: FloatArray,
    g_dx2: FloatArray,
    g_dx3: FloatArray,
    g_dx4: FloatArray,
    f_shape: tuple[int],
    g_size: int,
    x_size: int,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate fourth derivative of a function f(g(x)). This is vectorised and
    can accept an array of values of f where additional dimensions must be
    given in the first axes.

    f is a tensor valued function with shape (..., a1, ..., ak) where the
    first ... are the additional dimensions.
    g is a vector of shape (m,).
    x is a vector of shape (n,).

    Parameters
    ----------
    f_dg : np.array[float]
        First derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m).
    f_dg2 : np.array[float]
        Second derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m, m).
    f_dg3 : np.array[float]
        Third derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m, m, m).
    f_dg4 : np.array[float]
        Fourth derivative of f with respect to g.
        Must have shape (..., a1, ..., ak, m, m, m, m).
    g_dx : np.array[float]
        First derivative of g with respect to x.
        Must have shape (..., m, n).
    g_dx2 : np.array[float]
        Second derivative of g with respect to x.
        Must have shape (..., m, n, n).
    g_dx3 : np.array[float]
        Third derivative of g with respect to x.
        Must have shape (..., m, n, n, n).
    g_dx4 : np.array[float]
        Fourth derivative of g with respect to x.
        Must have shape (..., m, n, n, n, n).
    f_shape : tuple[int]
        Shape of f (a1, ..., ak).
    g_size: int
        Number of arguments in g.
    x_size : int
        Number of arguments in x.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    fourth_derivative : np.array[float]
        Fourth derivative of f with respect to x.
        Has shape (..., a1, ..., ak, n, n, n, n).
    """
    _f_dg = np.asarray(f_dg)
    _f_dg2 = np.asarray(f_dg2)
    _f_dg3 = np.asarray(f_dg3)
    _f_dg4 = np.asarray(f_dg4)
    _g_dx = np.asarray(g_dx)
    _g_dx2 = np.asarray(g_dx2)
    _g_dx3 = np.asarray(g_dx3)
    _g_dx4 = np.asarray(g_dx4)

    # Check shapes.
    _f = tuple(int(d) for d in f_shape)
    _g = max(1, int(g_size))
    _x = max(1, int(x_size))

    core_dim = len(_f) + 1
    extra_dims = _f_dg.shape[:-core_dim]
    _n = np.prod(extra_dims).astype(int)

    # Flatten over extra dimensions.
    _f_dg = np.reshape(_f_dg, (_n, *_f, _g))
    _g_dx = np.reshape(_g_dx, (_n, _g, _x))
    _f_dg2 = np.reshape(_f_dg2, (_n, *_f, _g, _g))
    _g_dx2 = np.reshape(_g_dx2, (_n, _g, _x, _x))
    _f_dg3 = np.reshape(_f_dg3, (_n, *_f, _g, _g, _g))
    _g_dx3 = np.reshape(_g_dx3, (_n, _g, _x, _x, _x))
    _f_dg4 = np.reshape(_f_dg4, (_n, *_f, _g, _g, _g, _g))
    _g_dx4 = np.reshape(_g_dx4, (_n, _g, _x, _x, _x, _x))

    # Calculate value.
    return_array = get_return_array(
        return_array,
        (*extra_dims, *_f, _x, _x, _x, _x),
        _f_dg.dtype,
    )

    return_array[..., :, :, :, :] = np.reshape(
        (
            np.einsum(
                "...abcd, ...ai, ...bj, ...ck, ...dl -> ...ijkl",
                _f_dg4,
                _g_dx,
                _g_dx,
                _g_dx,
                _g_dx,
            )
            + np.einsum(
                "...abc, ...ail, ...bj, ...ck -> ...ijkl",
                _f_dg3,
                _g_dx2,
                _g_dx,
                _g_dx,
            )
            + np.einsum(
                "...abc, ...ai, ...bjl, ...ck -> ...ijkl",
                _f_dg3,
                _g_dx,
                _g_dx2,
                _g_dx,
            )
            + np.einsum(
                "...abc, ...ai, ...bj, ...ckl -> ...ijkl",
                _f_dg3,
                _g_dx,
                _g_dx,
                _g_dx2,
            )
            + np.einsum(
                "...abc, ...aik, ...bj, ...cl -> ...ijkl",
                _f_dg3,
                _g_dx2,
                _g_dx,
                _g_dx,
            )
            + np.einsum(
                "...ab, ...aikl, ...bj -> ...ijkl", _f_dg2, _g_dx3, _g_dx
            )
            + np.einsum(
                "...ab, ...aik, ...bjl -> ...ijkl", _f_dg2, _g_dx2, _g_dx2
            )
            + np.einsum(
                "...abc, ...ai, ...bjk, ...cl -> ...ijkl",
                _f_dg3,
                _g_dx,
                _g_dx2,
                _g_dx,
            )
            + np.einsum(
                "...ab, ...ail, ...bjk -> ...ijkl", _f_dg2, _g_dx2, _g_dx2
            )
            + np.einsum(
                "...ab, ...ai, ...bjkl -> ...ijkl", _f_dg2, _g_dx, _g_dx3
            )
            + np.einsum(
                "...abc, ...aij, ...bk, ...cl -> ...ijkl",
                _f_dg3,
                _g_dx2,
                _g_dx,
                _g_dx,
            )
            + np.einsum(
                "...ab, ...aijl, ...bk -> ...ijkl", _f_dg2, _g_dx3, _g_dx
            )
            + np.einsum(
                "...ab, ...aij, ...bkl -> ...ijkl", _f_dg2, _g_dx2, _g_dx2
            )
            + np.einsum(
                "...ab, ...aijk, ...bl -> ...ijkl", _f_dg2, _g_dx3, _g_dx
            )
            + np.einsum("...a, ...aijkl -> ...ijkl", _f_dg, _g_dx4)
        ),
        (*extra_dims, *_f, _x, _x, _x, _x),
    )

    # Reshape extra dimensions of return value to match input
    return return_array
