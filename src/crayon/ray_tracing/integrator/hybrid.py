"""
Hybrid methods switching between explicit and implicit methods.
"""

# Standard imports
import logging

# Local imports
from crayon.ray_tracing.integrator.base import (
    SolverBase,
)
from crayon.ray_tracing.integrator.explicit import ExplicitSolverBase
from crayon.ray_tracing.integrator.implicit import ImplicitSolverBase

logger = logging.getLogger(__name__)


class HybridSolverBase(SolverBase):
    """
    Base class for hybrid ivp solver.

    Attributes
    ----------
    explicit : ExplicitSolverBase
        Explicit integrator.
    implicit : ImplicitSolverBase
        Implicit integrator.
    """

    __slots__ = ("explicit", "implicit")

    def __init__(
        self, explicit: ExplicitSolverBase, implicit: ImplicitSolverBase
    ):
        """
        Inits HybridSolverBase.

        Parameters
        ----------
        explicit : ExplicitSolverBase
            Explicit integrator.
        implicit : ImplicitSolverBase
            Implicit integrator.
        """
        self.explicit = explicit
        self.implicit = implicit
