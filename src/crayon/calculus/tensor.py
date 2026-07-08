"""
Methods for constructing and transforming tensor fields.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)


# Transform tensor fields.
class TensorType(tuple, CrayonEnum):
    """
    Rank and index transformation type of tensor.
    """

    __slots__ = ()

    # Rank 0.
    SCALAR = ()

    # Rank 1.
    VECTOR = (False,)
    COVECTOR = (True,)
    SCALAR_FIRST_DERIVATIVE = COVECTOR

    # Rank 2.
    VECTOR_FIRST_DERIVATIVE = (False, True)
    COVECTOR_FIRST_DERIVATIVE = (True, True)
    SCALAR_SECOND_DERIVATIVE = COVECTOR_FIRST_DERIVATIVE

    # Rank 3.
    VECTOR_SECOND_DERIVATIVE = (False, True, True)
    COVECTOR_SECOND_DERIVATIVE = (True, True, True)
    SCALAR_THIRD_DERIVATIVE = COVECTOR_SECOND_DERIVATIVE

    # Rank 4.
    VECTOR_THIRD_DERIVATIVE = (False, True, True, True)
    COVECTOR_THIRD_DERIVATIVE = (True, True, True, True)

    @property
    def rank(self) -> int:
        """Rank of tensor."""
        return len(self.value)

    @property
    def index_covariance(self) -> tuple[bool]:
        """Flag if tensor index transforms covariantly."""
        return self.value

    @property
    def first_derivative(self) -> "TensorType":
        """TensorType of first derivative of self."""
        if self == TensorType.SCALAR:
            return TensorType.SCALAR_FIRST_DERIVATIVE
        if self == TensorType.VECTOR:
            return TensorType.VECTOR_FIRST_DERIVATIVE
        if self == TensorType.COVECTOR:
            return TensorType.COVECTOR_FIRST_DERIVATIVE
        raise NotImplementedError(self.name)

    @property
    def second_derivative(self) -> "TensorType":
        """TensorType of second derivative of self."""
        if self == TensorType.SCALAR:
            return TensorType.SCALAR_SECOND_DERIVATIVE
        if self == TensorType.VECTOR:
            return TensorType.VECTOR_SECOND_DERIVATIVE
        if self == TensorType.COVECTOR:
            return TensorType.COVECTOR_SECOND_DERIVATIVE
        raise NotImplementedError(self.name)

    @property
    def third_derivative(self) -> "TensorType":
        """TensorType of third derivative of self."""
        if self == TensorType.SCALAR:
            return TensorType.SCALAR_THIRD_DERIVATIVE
        if self == TensorType.VECTOR:
            return TensorType.VECTOR_THIRD_DERIVATIVE
        if self == TensorType.COVECTOR:
            return TensorType.COVECTOR_THIRD_DERIVATIVE
        raise NotImplementedError(self.name)


def transform_tensor_field(
    tensor_type: TensorType,
    tensor_field: FloatArray,
    covariant_transform: FloatArray,
    contravariant_transform: FloatArray,
    /,
    *,
    reverse: bool,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Transform a tensor field to a different representation using the provided
    covariant and contravariant transforms.

    Tensor field components must be provided in the holonomic basis.

    Parameters
    ----------
    tensor_type : TensorType
        Type of tensor field input.
    tensor_field : np.array[float]
        Components of tensor field in holonomic basis.
    covariant_transform : np.array[float]
        Covariant transformation matrix.
    contravariant_transform : np.array[float]
        Contravariant transformation matrix.
    reverse : bool
        If True, transform in opposite direction to provided covariant and
        contravariant transforms. Otherwise transform in same direction.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    tensor_field_new : np.array[float]
        Transformed tensor field components.
    """
    return_array = get_return_array(
        return_array, tensor_field.shape, tensor_field.dtype
    )

    if reverse:
        # Contravariant backwards transform is transpose of covariant forward
        # transform and vice versa for covariant backwards transform.
        _contravariant_transform = covariant_transform.T
        _covariant_transform = contravariant_transform.T
    else:
        _contravariant_transform = contravariant_transform
        _covariant_transform = covariant_transform

    if tensor_type.rank == 0:
        # Scalar fields are the same in all coordinates.
        np.copyto(return_array, tensor_field)
        return return_array
    if tensor_type.rank == 1:
        _transform_rank_1(
            tensor_field,
            tensor_type.index_covariance,
            _covariant_transform,
            _contravariant_transform,
            return_array,
        )
    elif tensor_type.rank == 2:  # noqa: PLR2004
        _transform_rank_2(
            tensor_field,
            tensor_type.index_covariance,
            _covariant_transform,
            _contravariant_transform,
            return_array,
        )
    elif tensor_type.rank == 3:  # noqa: PLR2004
        _transform_rank_3(
            tensor_field,
            tensor_type.index_covariance,
            _covariant_transform,
            _contravariant_transform,
            return_array,
        )
    else:
        raise NotImplementedError(tensor_type.rank)

    return return_array


def get_transform(
    covariant_transform: FloatArray,
    contravariant_transform: FloatArray,
    /,
    *,
    index_covariant: bool,
):
    """
    Get correct transform if an index is covariant or contravariant.

    Parameters
    ----------
    covariant_transform : np.array[float]
        Transform for a covariant index.
    contravariant_transform : np.array[float]
        Transform for a contravariant index.
    index_covariant : bool
        Flag if the index is covariant or contravariant.

    Returns
    -------
    transform : np.array[float]
        Correct transform for index covariance.
    """
    return covariant_transform if index_covariant else contravariant_transform


def _transform_rank_1(
    rank_1_tensor: FloatArray,
    index_covariance: tuple[bool],
    covariant_transform: FloatArray,
    contravariant_transform: FloatArray,
    return_array: FloatArray,
):
    """
    Transform a rank 1 tensor field.

    Parameters
    ----------
    rank_1_tensor : np.array[float]
        Components of tensor field.
    index_covariance : tuple[bool]
        Flag if tensor index transforms covariantly (True) or contravariantly
        (False).
    covariant_transform : np.array[float]
        Covariant transformation matrix.
    contravariant_transform : np.array[float]
        Contravariant transformation matrix.
    return_array : np.array[float]
        Array into which the result is stored.
    """
    transform_1 = get_transform(
        covariant_transform,
        contravariant_transform,
        index_covariant=index_covariance[0],
    )

    np.einsum("ia, a -> i", transform_1, rank_1_tensor, out=return_array)


def _transform_rank_2(
    rank_2_tensor: FloatArray,
    index_covariance: tuple[bool, bool],
    covariant_transform: FloatArray,
    contravariant_transform: FloatArray,
    return_array: FloatArray,
):
    """
    Transform a rank 2 tensor field.

    Parameters
    ----------
    rank_2_tensor : np.array[float]
        Components of tensor field.
    index_covariance : tuple[bool]
        Flag if tensor index transforms covariantly (True) or contravariantly
        (False).
    covariant_transform : np.array[float]
        Covariant transformation matrix.
    contravariant_transform : np.array[float]
        Contravariant transformation matrix.
    return_array : np.array[float]
        Array into which the result is stored.
    """
    transform_1 = get_transform(
        covariant_transform,
        contravariant_transform,
        index_covariant=index_covariance[0],
    )
    transform_2 = get_transform(
        covariant_transform,
        contravariant_transform,
        index_covariant=index_covariance[1],
    )

    np.einsum(
        "ia, jb, ab -> ij",
        transform_1,
        transform_2,
        rank_2_tensor,
        out=return_array,
    )


def _transform_rank_3(
    rank_3_tensor: FloatArray,
    index_covariance: tuple[bool, bool, bool],
    covariant_transform: FloatArray,
    contravariant_transform: FloatArray,
    return_array: FloatArray,
):
    """
    Transform a rank 3 tensor field.

    Parameters
    ----------
    rank_3_tensor : np.array[float]
        Components of tensor field.
    index_covariance : tuple[bool]
        Flag if tensor index transforms covariantly (True) or contravariantly
        (False).
    covariant_transform : np.array[float]
        Covariant transformation matrix.
    contravariant_transform : np.array[float]
        Contravariant transformation matrix.
    return_array : np.array[float]
        Array into which the result is stored.
    """
    transform_1 = get_transform(
        covariant_transform,
        contravariant_transform,
        index_covariant=index_covariance[0],
    )
    transform_2 = get_transform(
        covariant_transform,
        contravariant_transform,
        index_covariant=index_covariance[1],
    )
    transform_3 = get_transform(
        covariant_transform,
        contravariant_transform,
        index_covariant=index_covariance[2],
    )

    np.einsum(
        "ia, jb, kc, abc -> ijk",
        transform_1,
        transform_2,
        transform_3,
        rank_3_tensor,
        out=return_array,
    )


def first_covariant_derivative(
    tensor_type: TensorType,
    tensor_field: FloatArray,
    tensor_field_dx: FloatArray,
    connection_coefficients: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate first covariant derivative of a tensor field.

    Tensor field components must be provided in the holonomic basis.

    Parameters
    ----------
    tensor_type : TensorType
        Type of tensor field input.
    tensor_field : np.array[float]
        Components of tensor field in holonomic basis.
    tensor_field_dx : np.array[float]
        First derivative (jacobian) of tensor field in holonomic basis.
    connection_coefficients : np.array[float]
        Connection coefficients aka Christoffel symbols.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    first_covariant_derivative : np.array[float]
        First covariant derivative of tensor field.
    """
    return_array = get_return_array(
        return_array,
        tensor_field_dx.shape,
        tensor_field.dtype,
    )

    if tensor_type.rank == 0:
        np.copyto(return_array, tensor_field_dx)
    elif tensor_type.rank == 1:
        _first_covariant_derivative_rank_1(
            tensor_type.index_covariance,
            tensor_field,
            tensor_field_dx,
            connection_coefficients,
            return_array,
        )
    else:
        raise NotImplementedError(tensor_type)

    return return_array


def _first_covariant_derivative_rank_1(
    index_covariance: tuple[bool],
    tensor_field: FloatArray,
    tensor_field_dx: FloatArray,
    connection_coefficients: FloatArray,
    return_array: FloatArray,
):
    """
    Calculate first covariant derivative for a rank 1 tensor.

    Parameters
    ----------
    index_covariance : tuple[bool]
        Flag if tensor index transforms covariantly (True) or contravariantly
        (False).
    tensor_field : np.array[float]
        Components of tensor field in holonomic basis.
    tensor_field_dx : np.array[float]
        First derivative (jacobian) of tensor field in holonomic basis.
    connection_coefficients : np.array[float]
        Connection coefficients aka Christoffel symbols.
    return_array : np.array[float]
        Array into which the result is stored.
    """
    if index_covariance[0]:
        # Index is covariant i.e. a covector.
        return_array[:, :] = tensor_field_dx - np.einsum(
            "kij, k -> ij", connection_coefficients, tensor_field
        )
    else:
        # Index is contravariant i.e. a vector.
        return_array[:, :] = tensor_field_dx + np.einsum(
            "ijk, k -> ij", connection_coefficients, tensor_field
        )


def second_covariant_derivative(
    tensor_type: TensorType,
    tensor_field: FloatArray,
    tensor_field_dx: FloatArray,
    tensor_field_dx2: FloatArray,
    tensor_field_first_covariant_derivative: FloatArray,
    connection_coefficients: FloatArray,
    connection_coefficients_dx: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate second covariant derivative of a tensor field.

    Tensor field components must be provided in the holonomic basis.

    Parameters
    ----------
    tensor_type : TensorType
        Type of tensor field input.
    tensor_field : np.array[float]
        Components of tensor field in holonomic basis.
    tensor_field_dx : np.array[float]
        First derivative (jacobian) of tensor field in holonomic basis.
    tensor_field_dx2 : np.array[float]
        Second derivative (hessian) of tensor field in holonomic basis.
    tensor_field_first_covariant_derivative : np.array[float]
        First covariant derivative of tensor field in holonomic basis.
    connection_coefficients : np.array[float]
        Connection coefficients aka Christoffel symbols.
    connection_coefficients_dx : np.array[float]
        First derivative (jacobian) of connection coefficients aka
        Christoffel symbols.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    first_covariant_derivative : np.array[float]
        First covariant derivative of tensor field.
    """
    return_array = get_return_array(
        return_array,
        tensor_field_dx2.shape,
        tensor_field.dtype,
    )

    if tensor_type.rank == 0:
        # The derivative of a rank 0 tensor is a covector so
        # we can treat the second derivative as the first derivative
        # of that covector field.
        _first_covariant_derivative_rank_1(
            TensorType.COVECTOR.index_covariance,
            tensor_field_dx,
            tensor_field_dx2,
            connection_coefficients,
            return_array,
        )
    elif tensor_type.rank == 1:
        _second_covariant_derivative_rank_1(
            tensor_type.index_covariance,
            tensor_field,
            tensor_field_dx,
            tensor_field_dx2,
            tensor_field_first_covariant_derivative,
            connection_coefficients,
            connection_coefficients_dx,
            return_array,
        )
    else:
        raise NotImplementedError(tensor_type)

    return return_array


def _second_covariant_derivative_rank_1(
    index_covariance: tuple[bool],
    tensor_field: FloatArray,
    tensor_field_dx: FloatArray,
    tensor_field_dx2: FloatArray,
    tensor_field_first_covariant_derivative: FloatArray,
    connection_coefficients: FloatArray,
    connection_coefficients_dx: FloatArray,
    return_array: FloatArray,
):
    """
    Calculate second covariant derivative for a rank 1 tensor.

    Parameters
    ----------
    index_covariance : tuple[bool]
        Flag if tensor index transforms covariantly (True) or contravariantly
        (False).
    tensor_field : np.array[float]
        Components of tensor field in holonomic basis.
    tensor_field_dx : np.array[float]
        First derivative (jacobian) of tensor field in holonomic basis.
    tensor_field_dx2 : np.array[float]
        Second derivative (hessian) of tensor field in holonomic basis.
    tensor_field_first_covariant_derivative : np.array[float]
        First covariant derivative of tensor field in holonomic basis.
    connection_coefficients : np.array[float]
        Connection coefficients aka Christoffel symbols.
    connection_coefficients_dx : np.array[float]
        First derivative (jacobian) of connection coefficients aka
        Christoffel symbols.
    return_array : np.array[float]
        Array into which the result is stored.
    """
    # Copy Hessian term.
    np.copyto(return_array, tensor_field_dx2)

    # Add corrections index by index.
    if index_covariance[0]:
        # Index is covariant.
        return_array[:, :, :] += (
            -np.einsum(
                "aij, ak -> ijk",
                connection_coefficients,
                tensor_field_dx,
            )
            - np.einsum(
                "aijk, a -> ijk",
                connection_coefficients_dx,
                tensor_field,
            )
            - np.einsum(
                "aik, aj -> ijk",
                connection_coefficients,
                tensor_field_first_covariant_derivative,
            )
            - np.einsum(
                "ajk, ia -> ijk",
                connection_coefficients,
                tensor_field_first_covariant_derivative,
            )
        )
    else:
        # Index is contravariant
        return_array[:, :, :] += (
            np.einsum(
                "ija, ak -> ijk",
                connection_coefficients,
                tensor_field_dx,
            )
            + np.einsum(
                "ijak, a -> ijk",
                connection_coefficients_dx,
                tensor_field,
            )
            + np.einsum(
                "iak, aj -> ijk",
                connection_coefficients,
                tensor_field_first_covariant_derivative,
            )
            - np.einsum(
                "ajk, ia -> ijk",
                connection_coefficients,
                tensor_field_first_covariant_derivative,
            )
        )
