"""
Caches for ray caustics and mode tunnelling.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.dimensions import Dimensions
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)


class CausticCache:
    """
    Cache holding information about caustics.

    Attributes
    ----------
    wkb_validity : np.array[float]
        Number indicating relative size of neglected terms in WKB
        approximation in each coordinate component. WKB approximation in x
        space is invalid if << 1. WKB approximation in k space is invalid if
        >> 1.
    """

    __slots__ = ("wkb_validity",)

    def __init__(self):
        """
        Inits CausticCache.
        """
        self.wkb_validity = np.zeros(Dimensions.x.size)

    def caustic_detected(
        self,
        wavevector_cartesian: FloatArray,
        focusing_tensor_x: FloatArray,
    ) -> bool:
        """
        Calculate if approaching a caustic.

        Parameters
        ----------
        wavevector_cartesian : np.array[float]
            Cartesian wavevector [m^-1].
        focusing_tensor_x : np.array[float]
            Cartesian focusing tensor in x representation [m^-2].

        Returns
        -------
        caustic_detected : bool
            True if approaching a caustic.

        Notes
        -----
        WKB approximation in x space requires |dk_i/dx| << k_i**2.
        """
        # If dk/dx = 0 and k_i = 0 this is also fine hence add a small
        # constant so 1.0 / k**2 is never zero.
        k2 = wavevector_cartesian * wavevector_cartesian + 1.0e-8
        self.wkb_validity[:] = np.diagonal(focusing_tensor_x) / k2

        return any(self.wkb_validity > 1.0)


class TunnellingCache:
    """
    Cache holding information about wave tunnelling.
    """

    __slots__ = ()

    def __init__(self):
        """
        Inits TunnellingCache.
        """

    @staticmethod
    def tunnelling_detected() -> bool:
        """
        Calculate if at a tunnelling point.

        Returns
        -------
        tunnelling_detected : bool
            If True, made closest approach to a tunnelling point.
        """
        return False
