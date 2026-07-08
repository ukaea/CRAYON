"""
Methods for construcing matricies and matrix calculus.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.helpers import get_return_array
from crayon.shared.types import Array, FloatArray

logger = logging.getLogger(__name__)


def mirror_upper_triangular_to_lower_triangular(array: Array):
    """
    Mirror values in place from upper triangular section over last 2 dimensions
    of a square matrix to the lower triangular section.

    Parameters
    ----------
    array : np.array
        Array to be symmetrised. Must have at least 2 dimensions and be square
        in last 2 dimensions.

    Raises
    ------
    ValueError
        matrix has invalid shape.

    Notes
    -----
    Useful for calculating symmetric matricies e.g. hessians.
    """
    if not (array.ndim >= 2 and array.shape[-2] == array.shape[-1]):  # noqa: PLR2004
        raise ValueError(
            "array must have dimension >= 2 and be square in last 2 "
            f"dimensions: {array.shape}"
        )

    _n = array.shape[-1]
    for i in range(_n):
        for j in range(i + 1, _n):
            np.copyto(array[..., j, i], array[..., i, j])


def hermitian(matrix: Array) -> Array:
    """
    Return hermitian part of matrix M_H where M = M_H + M_AH.

    Parameters
    ----------
    matrix : np.array
        Input matrix. Must be 2 dimensional and square.

    Returns
    -------
    hermitian_part
        Hermitian part of input matrix.

    Raises
    ------
    ValueError
        matrix has invalid shape.
    """
    if not (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]):  # noqa: PLR2004
        raise ValueError(
            f"matrix must have dimension 2 and be square: {matrix.shape}"
        )

    axes = np.arange(matrix.ndim, dtype=int)
    axes[-1] = matrix.ndim - 2
    axes[-2] = matrix.ndim - 1

    return 0.5 * (matrix + np.conj(np.transpose(matrix, axes=axes)))


def antihermitian(matrix: Array) -> Array:
    """
    Return anti-hermitian part of matrix M_AH where M = M_H + M_AH.

    Parameters
    ----------
    matrix : np.array
        Input matrix. Must be 2 dimensional and square.

    Returns
    -------
    antihermitian_part
        Anti-hermitian part of input matrix.

    Raises
    ------
    ValueError
        matrix has invalid shape.
    """
    if not (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]):  # noqa: PLR2004
        raise ValueError(
            f"matrix must have dimension 2 and be square: {matrix.shape}"
        )

    axes = np.arange(matrix.ndim, dtype=int)
    axes[-1] = matrix.ndim - 2
    axes[-2] = matrix.ndim - 1

    return 0.5 * (matrix - np.conj(np.transpose(matrix, axes=axes)))


def second_tensor_invariant_3x3(matrix: FloatArray) -> float:
    """
    Calculate second tensor invariant of a 3x3 matrix.

    Parameters
    ----------
    matrix : np.array[float]
        Input matrix. Must have shape (3, 3).

    Returns
    -------
    second_tensor_invariant : float
        Second tensor invariant of input matrix.

    Raises
    ------
    ValueError
        matrix has invalid shape.

    Notes
    -----
    Second tensor invariant is the sum of pairwise products of eigenvalues.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3): {matrix.shape}")

    m = matrix

    return (
        np.real(m[0, 0] * m[1, 1] + m[1, 1] * m[2, 2] + m[0, 0] * m[2, 2])
        - np.abs(m[0, 1] ** 2)
        - np.abs(m[0, 2] ** 2)
        - np.abs(m[1, 2] ** 2)
    )


def adjugate_3x3_cofactors(
    matrix: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate adjugate of a 3x3 matrix using cofactors.

    Parameters
    ----------
    matrix : np.array[float]
        Input matrix. Must have shape (3, 3).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    adjugate : np.array[float]
        Adjugate of input matrix.

    Raises
    ------
    ValueError
        Provided matrix does not have shape (3, 3).

    Notes
    -----
    The adjugate A of a matrix M satisfies A @ M = M @ A = det(M) * I where
    det(M) is the determinant of M and I is the identity matrix.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3): {matrix.shape}")

    return_array = get_return_array(return_array, (3, 3), matrix.dtype)

    # Adjugate is transpose of cofactor matrix.
    m = matrix

    return_array[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    return_array[1, 0] = -(m[1, 0] * m[2, 2] - m[2, 0] * m[1, 2])
    return_array[2, 0] = m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]

    return_array[0, 1] = -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1])
    return_array[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
    return_array[2, 1] = -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0])

    return_array[0, 2] = m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2]
    return_array[1, 2] = -(m[0, 0] * m[1, 2] - m[1, 0] * m[0, 2])
    return_array[2, 2] = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]

    return return_array


def adjugate_3x3_cayley_hamilton(
    matrix: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    matrix_squared: FloatArray = None,
    matrix_trace: complex | None = None,
    matrix_squared_trace: complex | None = None,
) -> FloatArray:
    """
    Calculate adjugate of a 3x3 matrix using the Cayley-Hamilton formula.

    Parameters
    ----------
    matrix : np.array[float]
        Input matrix. Must have shape (3, 3).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    adjugate : np.array[float]
        Adjugate of input matrix.

    Raises
    ------
    ValueError
        Provided matrix does not have shape (3, 3).

    Notes
    -----
    The adjugate A of a matrix M satisfies A @ M = M @ A = det(M) * I where
    det(M) is the determinant of M and I is the identity matrix.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3): {matrix.shape}")

    if matrix_trace is None:
        matrix_trace = np.trace(matrix)

    if matrix_squared is None:
        matrix_squared = np.matmul(matrix, matrix)

    if matrix_squared_trace is None:
        matrix_squared_trace = np.trace(matrix_squared)

    return_array = get_return_array(return_array, (3, 3), matrix.dtype)

    return_array[:, :] = (
        matrix_squared
        - matrix_trace * matrix
        + (
            0.5
            * (matrix_trace**2 - matrix_squared_trace)
            * np.identity(3, dtype=matrix.dtype)
        )
    )

    return return_array


def matrix_3x3_determinant_first_derivative(
    matrix: FloatArray,
    matrix_first_derivative: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    adjugate: FloatArray = None,
):
    """
    Calculate first derivative of the determinant of a 3x3 matrix using
    Jacobi's formula.

    Parameters
    ----------
    matrix : np.array[float]
        Input matrix. Must have shape (3, 3).
    matrix_first_derivative : np.array[float]
        Element-wise first derivative of input matrix. Must have shape
        (3, 3, n).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    adjugate : np.array[float], optional
        Adjugate of input matrix. If not provided it is calculated.

    Returns
    -------
    determinant_first_derivative : float
        First derivative of determinant of input matrix.

    Raises
    ------
    ValueError
        Provided matrix does not have shape (3, 3).

    Notes
    -----
    Jacobi's formula gives d/dx det(M) = Tr[Adj(M) @ (dM / dx)] where det
    is the determinant, Tr is the trace and Adj is the adjugate. Using the
    adjugate allows this to be applied to singular matrices.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3): {matrix.shape}")

    if adjugate is None:
        adjugate = adjugate_3x3_cofactors(matrix)

    n = matrix_first_derivative.shape[2]
    return_array = get_return_array(return_array, (n,), matrix.dtype)
    return_array[:] = np.einsum(
        "ij, jik -> k", adjugate, matrix_first_derivative
    )

    return return_array


def matrix_3x3_adjugate_first_derivative(
    matrix: FloatArray,
    matrix_first_derivative: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate element-wise derivative of the adjugate of a 3x3 matrix.

    Parameters
    ----------
    matrix : np.array[float]
        Input matrix. Must have shape (3, 3).
    matrix_first_derivative : np.array[float]
        Element-wise first derivative of input matrix. Must have shape
        (3, 3, n).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    adjugate_first_derivative : np.array[float]
        Element-wise first derivative of adjugate of input matrix.

    Raises
    ------
    ValueError
        matrix has invalid shape.
        matrix_first_derivative has invalid shape.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3): {matrix.shape}")

    if matrix_first_derivative.shape[:2] != (3, 3):
        raise ValueError(
            "matrix_first_derivative must have shape (3, 3, n): "
            f"{matrix_first_derivative.shape}"
        )

    identity = np.identity(3)
    matrix_trace = np.trace(matrix)
    matrix_first_derivative_trace = np.trace(matrix_first_derivative)

    n = matrix_first_derivative.shape[2]
    return_array = get_return_array(return_array, (3, 3, n), matrix.dtype)

    return_array[:, :, :] = (
        np.einsum("ij, jkl -> ikl", matrix, matrix_first_derivative)
        + np.einsum("ijl, jk -> ikl", matrix_first_derivative, matrix)
        - matrix_trace * matrix_first_derivative
        - np.einsum("ij, k", matrix, matrix_first_derivative_trace)
        + np.einsum(
            "ij, k",
            identity,
            matrix_trace * matrix_first_derivative_trace
            - np.einsum("ij, jil -> l", matrix, matrix_first_derivative),
        )
    )

    return return_array


def matrix_3x3_determinant_second_derivative(
    matrix: FloatArray,
    matrix_first_derivative: FloatArray,
    matrix_second_derivative: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
    adjugate: FloatArray = None,
    adjugate_first_derivative: FloatArray = None,
):
    """
    Calculate second derivative of the determinant of a 3x3 matrix.

    Parameters
    ----------
    matrix : np.array[float]
        Input matrix. Must have shape (3, 3).
    matrix_first_derivative : np.array[float]
        Element-wise first derivative of input matrix. Must have shape
        (3, 3, n).
    matrix_second_derivative : np.array[float]
        Element-wise second derivative of input matrix. Must have shape
        (3, 3, n, n).
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.
    adjugate : np.array[float], optional
        Adjugate of input matrix. If not provided it is calculated.
    adjugate_first_derivative : np.array[float], optional
        Element-wise first derivative of adjugate of input matrix. If not
        provided it is calculated.

    Returns
    -------
    determinant_second_derivative : float
        Second derivative of determinant of input matrix.

    Raises
    ------
    ValueError
        matrix has invalid shape.
        matrix_first_derivative has invalid shape.
        matrix_second_derivative has invalid shape.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3): {matrix.shape}")

    if matrix_first_derivative.shape[:2] != (3, 3):
        raise ValueError(
            "matrix_first_derivative must have shape (3, 3, n): "
            f"{matrix_first_derivative.shape}"
        )

    if matrix_second_derivative.shape[:2] != (3, 3):
        raise ValueError(
            "matrix_second_derivative must have shape (3, 3, n, n): "
            f"{matrix_second_derivative.shape}"
        )

    if adjugate is None:
        adjugate = adjugate_3x3_cofactors(matrix)

    if adjugate_first_derivative is None:
        adjugate_first_derivative = matrix_3x3_adjugate_first_derivative(
            matrix, matrix_first_derivative
        )

    n = matrix_first_derivative.shape[-1]
    return_array = get_return_array(return_array, (n, n), matrix.dtype)

    return_array[:, :] = np.einsum(
        "ij, jikl -> kl", adjugate, matrix_second_derivative
    ) + np.einsum(
        "ijk, jil -> kl", adjugate_first_derivative, matrix_first_derivative
    )

    return return_array
