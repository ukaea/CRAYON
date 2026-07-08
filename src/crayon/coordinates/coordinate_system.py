"""
Methods for calculating components required for transforming tensor fields
between coordinate systems and calculating covariant derivatives.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


METRIC_CARTESIAN = np.identity(Dimensions.x.size)


def metric_tensor(
    covariant_transform_from_cartesian: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate components of the metric tensor from covariant transform.

    Parameters
    ----------
    covariant_transform_from_cartesian : np.array[float]
        Covariant transform matrix from target coordinate system to Cartesian.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    metric_tensor : np.array[float]
        Metric tensor in provided coordinate system.
    """
    return_array = get_return_array(
        return_array,
        (Dimensions.x.size, Dimensions.x.size),
        FloatType,
    )

    return_array[:, :] = np.einsum(
        "ia, jb, ab -> ij",
        covariant_transform_from_cartesian,
        covariant_transform_from_cartesian,
        METRIC_CARTESIAN,
    )

    return return_array


def forward_transform_dx2(
    forward_transform_dx: FloatArray,
    backward_transform_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate second derivative of forward transform using second derivative
    of backward transform. The forward transform maps new coordinate
    components to the old whereas the backward transform does the reverse.

    Parameters
    ----------
    forward_transform_dx : np.array[float]
        First derivative of forward transform.
    backward_transform_dx2 : np.array[float]
        Second derivative of backward transform.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    forward_transform_dx2 : np.array[float]
        Second derivative of forward transform.
    """
    return_array = get_return_array(
        return_array,
        (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
        FloatType,
    )

    return_array[:, :, :] = -np.einsum(
        "ka, bi, cj, acb -> kij",
        forward_transform_dx,
        forward_transform_dx,
        forward_transform_dx,
        backward_transform_dx2,
    )

    return return_array


def forward_transform_dx3(
    forward_transform_dx: FloatArray,
    forward_transform_dx2: FloatArray,
    backward_transform_dx2: FloatArray,
    backward_transform_dx3: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Calculate third derivative of forward transform using third derivative
    of backward transform. The forward transform maps new coordinate
    components to the old whereas the backward transform does the reverse.

    Parameters
    ----------
    forward_transform_dx : np.array[float]
        First derivative of forward transform.
    forward_transform_dx2 : np.array[float]
        Second derivative of forward transform.
    backward_transform_dx2 : np.array[float]
        Second derivative of backward transform.
    backward_transform_dx3 : np.array[float]
        Third derivative of backward transform.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    forward_transform_dx2 : np.array[float]
        Third derivative of forward transform .
    """
    return_array = get_return_array(
        return_array,
        (
            Dimensions.x.size,
            Dimensions.x.size,
            Dimensions.x.size,
            Dimensions.x.size,
        ),
        FloatType,
    )

    return_array[:, :, :, :] = -(
        np.einsum(
            "abc, ial, bj, ck -> ijkl",
            backward_transform_dx2,
            forward_transform_dx2,
            forward_transform_dx,
            forward_transform_dx,
        )
        + np.einsum(
            "abc, ia, bjl, ck -> ijkl",
            backward_transform_dx2,
            forward_transform_dx,
            forward_transform_dx2,
            forward_transform_dx,
        )
        + np.einsum(
            "abc, ia, bj, ckl -> ijkl",
            backward_transform_dx2,
            forward_transform_dx,
            forward_transform_dx,
            forward_transform_dx2,
        )
        + np.einsum(
            "ia, bj, ck, dl, abcd -> ijkl",
            forward_transform_dx,
            forward_transform_dx,
            forward_transform_dx,
            forward_transform_dx,
            backward_transform_dx3,
        )
    )

    return return_array


def connection_coefficients(
    backward_transform_to_cartesian_dx: FloatArray,
    forward_transform_from_cartesian_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate connection coefficients aka Christoffel symbols of second kind.

    Parameters
    ----------
    backward_transform_to_cartesian_dx : np.array[float]
        First derivative of backward transform from new coordinate system
        components to Cartesian.
    forward_transform_from_cartesian_dx2 : np.array[float]
        Second derivative of forward transform from Cartesian components to
        new coordinate system.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    connection_coefficients : np.array[float]
        Connection coefficients.
    """
    return_array = get_return_array(
        return_array,
        (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
        FloatType,
    )

    return_array[:, :, :] = -np.einsum(
        "ai, bj, kab -> kij",
        backward_transform_to_cartesian_dx,
        backward_transform_to_cartesian_dx,
        forward_transform_from_cartesian_dx2,
    )

    return return_array


def connection_coefficients_2(
    forward_transform_to_cartesian_dx: FloatArray,
    backward_transform_from_cartesian_dx2: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate connection coefficients aka Christoffel symbols of second kind.

    Parameters
    ----------
    forward_transform_to_cartesian_dx : np.array[float]
        First derivative of forward transform from Cartesian components to
        new coordinate system.
    backward_transform_from_cartesian_dx2 : np.array[float]
        Second derivative of backward transform from new coordinate system
        components to Cartesian.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    connection_coefficients : np.array[float]
        Connection coefficients.
    """
    return_array = get_return_array(
        return_array,
        (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
        FloatType,
    )

    return_array[:, :, :] = np.einsum(
        "aij, ka -> kij",
        backward_transform_from_cartesian_dx2,
        forward_transform_to_cartesian_dx,
    )

    return return_array


def connection_coefficients_dx(
    connection_coefficients: FloatArray,
    backward_transform_to_cartesian_dx: FloatArray,
    forward_transform_from_cartesian_dx3: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate first derivative of connection coefficients.

    Parameters
    ----------
    connection_coefficients : np.array[float]
        Connection coefficients.
    backward_transform_to_cartesian_dx : np.array[float]
        First derivative of backward transform from new coordinate system
        components to Cartesian.
    forward_transform_from_cartesian_dx3 : np.array[float]
        Third derivative of forward transform from Cartesian components to
        new coordinate system.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    connection_coefficients_dx : np.array[float]
        First derivative (jacobian) of connection coefficients.
    """
    return_array = get_return_array(
        return_array,
        (
            Dimensions.x.size,
            Dimensions.x.size,
            Dimensions.x.size,
            Dimensions.x.size,
        ),
        FloatType,
    )

    return_array[:, :, :, :] = (
        -np.einsum(
            "ai, bj, cl, kabc -> kijl",
            backward_transform_to_cartesian_dx,
            backward_transform_to_cartesian_dx,
            backward_transform_to_cartesian_dx,
            forward_transform_from_cartesian_dx3,
        )
        + np.einsum(
            "ail, kaj -> kijl",
            connection_coefficients,
            connection_coefficients,
        )
        + np.einsum(
            "ajl, kai -> kijl",
            connection_coefficients,
            connection_coefficients,
        )
    )

    return return_array


def connection_coefficients_2_dx(
    connection_coefficients: FloatArray,
    forward_transform_to_cartesian_dx: FloatArray,
    backward_transform_to_cartesian_dx3: FloatArray,
    /,
    *,
    return_array: FloatArray = None,
):
    """
    Calculate first derivative of connection coefficients.

    Parameters
    ----------
    forward_transform_to_cartesian_dx : np.array[float]
        First derivative of forward transform from Cartesian components to
        new coordinate system.
    backward_transform_to_cartesian_dx3 : np.array[float]
        Third derivative of backward transform from new coordinate system
        components to Cartesian.
    return_array : np.array[float]
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    connection_coefficients_dx : np.array[float]
        First derivative (jacobian) of connection coefficients.
    """
    return_array = get_return_array(
        return_array,
        (
            Dimensions.x.size,
            Dimensions.x.size,
            Dimensions.x.size,
            Dimensions.x.size,
        ),
        FloatType,
    )

    return_array[:, :, :, :] = np.einsum(
        "aijl, ka -> kijl",
        backward_transform_to_cartesian_dx3,
        forward_transform_to_cartesian_dx,
    ) - np.einsum(
        "aij, kal -> kijl", connection_coefficients, connection_coefficients
    )

    return return_array
