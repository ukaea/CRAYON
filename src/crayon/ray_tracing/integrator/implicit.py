"""
Implicit integration methods.
"""

# Standard imports
import logging

# Local imports
from crayon.ray_tracing.integrator.base import SolverBase
from crayon.shared.data_structures import CrayonEnum

logger = logging.getLogger(__name__)


class ImplicitIntegratorType(CrayonEnum):
    """
    Implicit integration methods.

    Attributes
    ----------
    BACKWARDS_EULER
        2nd order backwards euler method.
    """

    BACKWARDS_EULER = 1


class ImplicitSolverBase(SolverBase):
    """
    Base class for explicit integrator.
    """


class BackwardsEuler(ImplicitSolverBase):
    """
    Backwards Euler method.
    """


implicit_solvers = {ImplicitIntegratorType.BACKWARDS_EULER: BackwardsEuler}
