"""
Caches for ray intersections with limiter elements and the plasma boundary.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.dimensions import Dimensions

logger = logging.getLogger(__name__)


class LimiterCache:
    """
    Cache holding information about limiter intersections.

    Attributes
    ----------
    last_intersected_element_index : int
        Element index of the limiter element last intersected. This is ignored
        in the next step to avoid double intersections due to numerical error.
    last_intersected_limiter_name : str
        Name of last intersected limiter element collection.
    last_position : np.array[float]
        Cartesian position at last ray step.
    last_wavevector : np.array[float]
        Cartesian wavevector at last ray step.
    reflection_count : int
        Total number of accumulated reflections.
    """

    __slots__ = (
        "last_intersected_element_index",
        "last_intersected_limiter_name",
        "last_position",
        "last_wavevector",
        "reflection_count",
    )

    def __init__(self):
        """
        Inits LimiterCache.
        """
        self.last_position = np.zeros(Dimensions.x.size)
        self.last_wavevector = np.zeros(Dimensions.x.size)
        self.reflection_count = 0

        self.reset_last_intersected_element()

    def set_last_intersected_element(self, name: str, idx: int):
        """
        Register element index of last intersected limiter element. Also
        increments counter on reflections.

        Parameters
        ----------
        idx : int
            Element index of last intersected limiter element.
        """
        self.last_intersected_limiter_name = name
        self.last_intersected_element_index = idx
        self.reflection_count += 1

    def reset_last_intersected_element(self):
        """
        Reset element index of last intersected limiter element.
        """
        self.last_intersected_limiter_name = ""
        self.last_intersected_element_index = -1

    @property
    def last_intersected_element(self) -> tuple[str, int]:
        """
        Name and index of last intersected limiter element.

        Returns
        -------
        name : str
            Name of limiter element collection.
        index : int
            Index of element on limiter.
        """
        return (
            self.last_intersected_limiter_name,
            self.last_intersected_element_index,
        )


class PlasmaVacuumBoundaryCache:
    """
    Cache holding information about plasma vacuum transitions.

    Attributes
    ----------
    previous_step_vacuum : bool
        If previous ray position was in vacuum.
    """

    __slots__ = ("previous_step_vacuum",)

    def __init__(self):
        """
        Inits PlasmaVacuumBoundaryCache.
        """
        self.previous_step_vacuum = False
