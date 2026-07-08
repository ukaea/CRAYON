"""
Helper methods for constructing integrators.
"""

# Standard imports
import logging

# Local imports
from crayon.ray_tracing.integrator.base import AdaptiveSolverBase, Integrator
from crayon.ray_tracing.integrator.explicit import explicit_solvers
from crayon.ray_tracing.integrator.hybrid import HybridSolverBase
from crayon.ray_tracing.integrator.implicit import implicit_solvers
from crayon.ray_tracing.integrator.options import (
    IntegratorType,
    OptionsIntegrator,
)
from crayon.ray_tracing.integrator.timestep_controller import (
    TimestepController,
    TimestepControllerType,
)

logger = logging.getLogger(__name__)


def get_integrator_from_options(
    size: int,
    options: OptionsIntegrator,
    /,
    *,
    is_complex: bool,
    primary_size: int | None = None,
) -> Integrator:
    """
    Construct integrator from options.

    Parameters
    ----------
    size : int
        Size of state vector.
    options : OptionsIntegrator
        Options for integrator.
    is_complex : bool
        If True, state vector is complex. Otherwise float.
    primary_size : int, optional
        Number of elements at start of state vector used for error estimate.

    Returns
    -------
    integrator : Integrator
        Integrator.
    """
    explicit_solver = explicit_solvers[options.explicit.solver]
    implicit_solver = implicit_solvers[options.implicit.solver]

    if issubclass(explicit_solver, AdaptiveSolverBase):
        explicit_timestep_controller = TimestepController.preset(
            options.timestep_controller,
            options.norm,
            explicit_solver.error_estimate_order,
        )
    else:
        explicit_timestep_controller = TimestepController.preset(
            TimestepControllerType.NONE, options.norm, 1
        )

    if issubclass(implicit_solver, AdaptiveSolverBase):
        implicit_timestep_controller = TimestepController.preset(
            options.timestep_controller,
            options.norm,
            implicit_solver.error_estimate_order,
        )
    else:
        implicit_timestep_controller = TimestepController.preset(
            TimestepControllerType.NONE, options.norm, 1
        )

    if options.solver_type == IntegratorType.EXPLICIT:
        solver = explicit_solver
        timestep_controller = explicit_timestep_controller
        max_iterations = options.explicit.max_iterations
    elif options.solver_type == IntegratorType.IMPLICIT:
        solver = implicit_solver
        timestep_controller = implicit_timestep_controller
        max_iterations = options.implicit.max_iterations
    elif options.solver_type == IntegratorType.HYBRID:
        solver = HybridSolverBase(explicit_solver, implicit_solver)
        timestep_controller = HybridSolverBase(
            explicit_timestep_controller, implicit_timestep_controller
        )
        max_iterations = HybridSolverBase(
            options.explicit.max_iterations, options.implicit.max_iterations
        )
    else:
        raise NotImplementedError(options.solver_type)

    return Integrator(
        size,
        solver,
        options.absolute_tolerance,
        options.relative_tolerance,
        options.min_timestep,
        options.max_timestep,
        max_iterations,
        timestep_controller,
        is_complex=is_complex,
        primary_size=primary_size,
    )
