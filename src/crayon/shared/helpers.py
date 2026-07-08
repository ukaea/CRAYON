"""
Helper functions.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.types import Array

logger = logging.getLogger(__name__)


def to_bool(value) -> bool:
    """
    Coerce value to boolean.

    Parameters
    ----------
    value : any
        Value to coerce.

    Returns
    -------
    value_as_bool : bool
        Value coerced to boolean.
    """
    return str(value).lower() in {"true", "1"}


def pairwise(iterable):
    """
    Gather elements from an iterator pairwise i.e. if iterator returns
    (a, b, c, d, ...) this will return (a, b), (b, c), (c, d), ...

    Yields
    ------
    a, b
        Pairs of items from the iterable.
    """
    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b


def get_return_array(
    array: Array | None, shape: tuple[int], dtype: type
) -> Array:
    """
    Get return array of given shape or check array has correct shape.

    Parameters
    ----------
    array : np.array | None
        Array if it exists. If None, array will be created.
    shape : tuple[int]
        Array shape.
    dtype : type
        Datatype.

    Returns
    -------
    array : Array
        Return array.

    Raises
    ------
    ValueError
        Array provided and has incorrect shape.
    """
    if array is None:
        array = np.empty(shape, dtype=dtype)
    elif array.shape != shape:
        raise ValueError(
            f"Array has incorrect shape. Expected {array.shape}, got {shape}"
        )

    return array
