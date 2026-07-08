"""
Base classes for caches for ray tracing.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.dimensions import Dimension, Dimensions

logger = logging.getLogger(__name__)


class DerivativeCache:
    """
    Cache for a value and its derivatives with respect to a dimension. All
    coordinate dependent objects are in given in Cartesian.

    Attributes
    ----------
    value : np.array
        Parameter value.
    dz : np.array
        First derivative of parameter value with respect to extended phase
        space position z = [x, k, f].
    dz2 : np.array
        Second derivative of parameter value with respect to extended phase
        space position z = [x, k, f].
    """

    __slots__ = ("first_derivative", "second_derivative", "value")

    derivative_dimension: Dimension = NotImplemented

    def __init__(self, shape: tuple[int], /, *, is_complex: bool = False):
        """
        Inits DerivativeCache.

        Parameters
        ----------
        *shape : list[int]
            Shape of value.
        is_complex : bool
            If True, datatype is complex. Otherwise datatype is float.
        """
        n = self.derivative_dimension.size
        dtype = complex if is_complex else float

        self.value = np.empty(shape, dtype=dtype)
        self.first_derivative = np.empty((*shape, n), dtype=dtype)
        self.second_derivative = np.empty((*shape, n, n), dtype=dtype)


class DerivativeCacheX(DerivativeCache):
    """
    Cache for a value and its derivatives with respect to position x.
    """

    derivative_dimension = Dimensions.x


class DerivativeCacheXK(DerivativeCache):
    """
    Cache for a value and its derivatives with respect to phase space position
    x and k.
    """

    derivative_dimension = Dimensions.xk


class DerivativeCacheZ(DerivativeCache):
    """
    Cache for a value and its derivatives with respect to extended phase space
    position x, k and f.
    """

    derivative_dimension = Dimensions.z
