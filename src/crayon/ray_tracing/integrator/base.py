"""
Base classes for integrator.
"""

# Standard imports
import abc
import logging
import typing

# Third party imports
import numpy as np

# Local imports
from crayon.ray_tracing.integrator.timestep_controller import (
    TimestepController,
)
from crayon.shared.types import Array

logger = logging.getLogger(__name__)


class SolverCache:
    """
    Generic initial value problem solver cache.

    Attributes
    ----------
    dy_dt_func : callable[[float, np.array], np.array]
        Function which calculates time derivative of the state vector.
    dy_dt_stages : np.array
        State vector derivative for each stage.
    step_attempts : int
        Number of attempts required for last step.
    step_attempts_cache : np.array[int]
        Number of attempts required for each step in the last 5 steps.
    stop_condition : str
        Error message.
    t_cache : np.array
        Cache of time values from previous steps.
    t_proposed : float
        Proposed time for next step.
    t_stages : np.array
        Time for each stage.
    timestep_proposed : float
        Proposed timestep at next step.
    y_cache : np.array
        Cache of state vector values from previous steps.
    y_proposed : np.array
        Proposed state vector at next step.
    y_stages : np.array
        State vector for each stage.

    Methods
    -------
    set_dy_dt_func
        Set function which calculates the time derivative of the state vector.
    initialise
        Set initial conditions in cache.
    prepare_first_stage
        Prepare first integration stage.
    calculate_stage_dy_dt
        Calculate dy/dt for the given stage index.
    accept_step
        Accept proposed step in cache.
    """

    __slots__ = (
        "dy_dt_func",
        "dy_dt_stages",
        "step_attempts",
        "step_attempts_cache",
        "stop_condition",
        "t_cache",
        "t_proposed",
        "t_stages",
        "timestep_proposed",
        "y_cache",
        "y_proposed",
        "y_stages",
    )

    def __init__(
        self,
        size: int,
        stages: int,
        steps: int,
        /,
        *,
        is_complex: bool,
    ):
        """
        Inits SolverCache.

        Parameters
        ----------
        size : int
            Size of the state vector.
        stages : int
            Number of stages used in each step calculation.
        steps : int
            Number of previous steps used in each step calculation.
        is_complex : bool
            Flag if data is complex
        """
        # Need at least 2 stages for initial timestep calculation.
        stages = max(2, stages)

        # Need at least 1 slot to hold accepted state vector.
        cache_size = max(1, steps)

        # Cache of t, y, dy/dt for each stages.
        dtype = complex if is_complex else float
        self.t_stages = np.zeros(stages, dtype=float)
        self.y_stages = np.zeros((stages, size), dtype=dtype)
        self.dy_dt_stages = np.zeros((stages, size), dtype=dtype)

        # Cache of t and y from previous steps.
        self.t_cache = np.zeros(cache_size, dtype=float)
        self.y_cache = np.zeros((cache_size, size), dtype=dtype)

        # Proposed timestep for next integration.
        self.timestep_proposed = 0.0
        self.t_proposed = 0.0
        self.y_proposed = np.zeros(size, dtype=dtype)

        # Counter of attempted integration steps.
        self.step_attempts = 0
        self.step_attempts_cache = np.zeros(5)

        # Function while provided time derivative of state vector.
        self.dy_dt_func = None

        # Place for any error messages from the integrator.
        self.stop_condition: str = ""

    def set_dy_dt_func(
        self, dy_dt_func: typing.Callable[[float, Array], Array]
    ):
        """
        Set function which calculates the time derivative of the state vector.

        Parameters
        ----------
        dy_dt_func : callable[[float, np.array], np.array]
            Function which calculates dy/dt = f(t, y).
        """
        self.dy_dt_func = dy_dt_func

    def initialise(self, t0: float, y0: Array):
        """
        Set initial conditions in cache.

        Parameters
        ----------
        t0 : float
            Initial time.
        y0 : np.array
            Initial state vector.

        Raises
        ------
        ValueError
            dy_dt_func is not set.

        Notes
        -----
        Must call set_dy_dt_func before this function.
        """
        if self.dy_dt_func is None:
            raise ValueError(
                "dy_dt_func is not set. Did you call set_dy_dt_func?"
            )

        self.t_proposed = t0
        self.y_proposed[:] = y0

        self.accept_step()
        self.prepare_first_stage(reuse_last_dy_dt=False)

    def prepare_first_stage(self, /, *, reuse_last_dy_dt: bool):
        """
        Prepare first integration stage.

        Parameters
        ----------
        reuse_last_dy_dt : bool
            Reuse the dy_dt stage from the previous integration step.
        """
        self.t_stages[0] = self.t_cache[0]
        self.y_stages[0, :] = self.y_cache[0]

        if reuse_last_dy_dt:
            self.dy_dt_stages[0, :] = self.dy_dt_stages[-1]
        else:
            self.calculate_stage_dy_dt(0)

    def calculate_stage_dy_dt(self, index: int):
        """
        Calculate dy/dt for the given stage index.

        Parameters
        ----------
        index : int
            Stage index to calculate.
        """
        self.dy_dt_stages[index, :] = self.dy_dt_func(
            self.t_stages[index],
            self.y_stages[index],
        )

    def accept_step(self):
        """
        Accept proposed step in cache.
        """
        # Roll caches to move oldest value to front.
        self.t_cache[:] = np.roll(self.t_cache, 1)
        self.y_cache[:, :] = np.roll(self.y_cache, 1, axis=0)

        # Cache current value.
        self.t_cache[0] = self.t_proposed
        self.y_cache[0, :] = self.y_proposed

        # Update step attempts cache.
        self.step_attempts_cache[:] = np.roll(self.step_attempts_cache, 1)
        self.step_attempts_cache[0] = self.step_attempts

        # Reset counter on step attempts.
        self.step_attempts = 0


class SolverBase(abc.ABC):
    """
    Base class for IVP solver.

    Attributes
    ----------
    order : int
        Order of integration.
    stages : int
        Number of stages used to calculate each step.
    steps : int
        Number of steps used to calculate each step (including current).
    fsal : bool
        First stage of next step is same as last stage of previous step.

    Methods
    -------
    get_cache
        Get solver cache.
    step
        Integrate a single timestep.
    method_stable
        Return estimate for if the solver is currently numerically stable.
    calculate_initial_timestep
        Calculate initial timestep.
    """

    __slots__ = ()

    order: int = NotImplemented
    stages: int = NotImplemented
    steps: int = NotImplemented
    fsal: bool = NotImplemented

    @classmethod
    @abc.abstractmethod
    def get_cache(
        cls, size: int, /, *, is_complex: bool, primary_size: int | None = None
    ) -> SolverCache:
        """
        Get solver cache.

        Parameters
        ----------
        size : int
            Size of state vector.
        is_complex : bool
            If state vector is complex.
        primary_size : int, optional
            Number of elements from start of state vector used for error
            estimate.

        Returns
        -------
        solver_cache : SolverCache
            Solver cache.
        """

    @classmethod
    @abc.abstractmethod
    def step(cls, cache: SolverCache):
        """
        Integrate a single timestep.

        Parameters
        ----------
        cache : SolverCache
            Solver cache.
        """

    @classmethod
    def method_stable(cls, cache: SolverCache) -> bool:
        """
        Return estimate for if the solver is currently numerically stable.

        Parameters
        ----------
        cache : SolverCache
            Solver cache.

        Returns
        -------
        stable : bool
            Estimate for if solver is numerically stable.
        """
        raise NotImplementedError

    @classmethod
    def calculate_initial_timestep(
        cls,
        cache: SolverCache,
        atol: float,
        rtol: float,
    ):
        """
        Estimate initial timestep for IVP using algorithm from [1].

        Parameters
        ----------
        cache : SolverCache
            Solver cache.
        atol : float
            Absolute tolerance in solution.
        rtol : float
            Relative tolerance in solution.

        References
        ----------
        [1] E. Hairer, S. P. Norsett, G. Wanner, 'Solving Ordinary
            Differential Equations I: Nonstiff Problems'
        """
        t0 = cache.t_stages[0]
        y0 = cache.y_stages[0]
        f0 = cache.dy_dt_stages[0, :]

        d0 = np.linalg.norm(atol + rtol * y0)
        d1 = np.linalg.norm(atol + rtol * f0)

        dt0 = 1e-05 if d0 < 1e-05 or d1 < 1e-05 else 0.01 * (d0 / d1)  # noqa: PLR2004

        cache.t_stages[1] = t0 + dt0
        cache.y_stages[1] = y0 + dt0 * f0
        cache.calculate_stage_dy_dt(1)

        f1 = cache.dy_dt_stages[1, :]
        d2 = np.linalg.norm(atol + rtol * (f1 - f0) / dt0)
        dmax = max(d1, d2)

        if dmax <= 1e-15:  # noqa: PLR2004
            dt1 = max(1e-6, 1e-3 * dt0)
        else:
            dt1 = (0.01 / dmax) ** (1 / (cls.order + 1))

        cache.timestep_proposed = min(100 * dt0, dt1)

        # Return to original position.
        cache.calculate_stage_dy_dt(0)


class AdaptiveSolverCache(SolverCache):
    """
    Base class for cache for solver with adaptive timestep.

    Attributes
    ----------
    dy_dt_func : callable[[float, np.array], np.array]
        Function which calculates time derivative of the state vector.
    dy_dt_stages : np.array
        State vector derivative for each stage.
    step_attempts : int
        Number of attempts required for last step.
    step_attempts_cache : np.array[int]
        Number of attempts required for each step in the last 5 steps.
    stop_condition : str
        Error message.
    t_cache : np.array
        Cache of time values from previous steps.
    t_proposed : float
        Proposed time for next step.
    t_stages : np.array
        Time for each stage.
    timestep_proposed : float
        Proposed timestep at next step.
    y_cache : np.array
        Cache of state vector values from previous steps.
    y_proposed : np.array
        Proposed state vector at next step.
    y_stages : np.array
        State vector for each stage.
    error_cache : np.array[float]
        Cache of error values from previous timesteps.
    error_proposed : float
        Proposed error in next step.
    first_step : bool
        Flag if this the first integration step.
    primary_size : int
        Number of elements from the start of the state vector used to estimate
        the error.
    second_step : bool
        Flag if this the second integration step.
    timestep_next : float
        Estimate for next timestep based on error estimate.

    Properties
    ----------
    timestep_proposed : float
        Timestep proposed for next step.

    Methods
    -------
    set_dy_dt_func
        Set function which calculates the time derivative of the state vector.
    initialise
        Set initial conditions in cache.
    prepare_first_stage
        Prepare first integration stage.
    calculate_stage_dy_dt
        Calculate dy/dt for the given stage index.
    accept_step
        Accept proposed step in cache.
    """

    __slots__ = (
        "_timestep_proposed",
        "error_cache",
        "error_proposed",
        "first_step",
        "primary_size",
        "second_step",
        "timestep_next",
    )

    def __init__(
        self,
        size: int,
        stages: int,
        cache_size: int,
        /,
        *,
        is_complex: bool,
        primary_size: int | None = None,
    ):
        """
        Inits AdaptiveSolverCache.

        Parameters
        ----------
        size : int
            Size of state vector.
        stages : int
            Number of stages in integration step.
        cache_size : int
            Number of cached values from previous steps.
        is_complex : bool
            If True, state vector has complex type. Otherwise float.
        primary_size : int
            Number of values at start of state vector used for error estimate.
        """
        super().__init__(size, stages, cache_size, is_complex=is_complex)

        self.error_proposed = 0.0
        self.error_cache = np.zeros(2, dtype=float)

        # Primary variables are used to calculate error estimate. Take first
        # X values at start of state vector.
        self.primary_size = (
            int(primary_size) if primary_size is not None else size
        )

        # Suggested next timestep.
        self.timestep_next = 0.0

        # Flags for which step we are on.
        self.first_step = True
        self.second_step = False

    @property
    def timestep_proposed(self):
        """Timestep proposed for next step."""
        return self._timestep_proposed

    @timestep_proposed.setter
    def timestep_proposed(self, value: float):
        """Timestep proposed for next step."""
        self._timestep_proposed = value
        self.timestep_next = value

    def accept_step(self):
        """
        Accept proposed step in cache.
        """
        super().accept_step()

        # Cache error estimate for step.
        self.error_cache[:] = np.roll(self.error_cache, 1)
        self.error_cache[0] = self.error_proposed

        # Update flags.
        if self.first_step:
            self.first_step = False
            self.second_step = True
        elif self.second_step:
            self.second_step = False

    def initialise(self, t0, y0):
        """
        Set initial conditions in cache.

        Parameters
        ----------
        t0 : float
            Initial time.
        y0 : np.array
            Initial state vector.

        Notes
        -----
        Must call set_dy_dt_func before this function.
        """
        # Synchronise suggested next timestep with proposed.
        self.timestep_next = self.timestep_proposed

        super().initialise(t0, y0)

        # Reset step flags.
        self.first_step = True
        self.second_step = False


class AdaptiveSolverBase(SolverBase):
    """
    Base class for IVP solver with adaptive timestep.

    Attributes
    ----------
    order : int
        Order of integration.
    stages : int
        Number of stages used to calculate each step.
    steps : int
        Number of steps used to calculate each step (including current).
    fsal : bool
        First stage of next step is same as last stage of previous step.
    error_estimate_order : int
        Order of error estimate.

    Methods
    -------
    get_cache
        Get solver cache.
    step
        Integrate a single timestep.
    method_stable
        Return estimate for if the solver is currently numerically stable.
    calculate_initial_timestep
        Calculate initial timestep.
    """

    error_estimate_order: int = NotImplemented

    @classmethod
    @abc.abstractmethod
    def estimate_error(cls, cache: AdaptiveSolverCache) -> float:
        """
        Calculate error estimate for integration step.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Solver cache.

        Returns
        -------
        error_estimate : float
            Error estimate.
        """

    @classmethod
    def estimate_normalised_error(
        cls,
        cache: AdaptiveSolverCache,
        timestep_controller: TimestepController,
        atol: float,
        rtol: float,
    ):
        """
        Estimate error in ODE solution normalised to the desired error.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Solver cache.
        timestep_controller : TimestepController
            Algorithm used to control timestep based on error.
        atol : float
            Absolute tolerance in solution.
        rtol : float
            Relative tolerance in solution.
        """
        # Calculate error limit on each state vector component.
        error_limit = atol + rtol * np.maximum(
            cache.y_cache[0, :], cache.y_proposed
        )

        # Calculate ratio of estimated error to the limit.
        error_estimate = cls.estimate_error(cache)
        cache.error_proposed = timestep_controller.normalised_error(
            error_estimate[: cache.primary_size],
            error_limit[: cache.primary_size],
        )

    @classmethod
    def error_timestep_too_small(
        cls, timestep: float, min_timestep: float
    ) -> str:
        """
        Error message for timestep below minimum.

        Parameters
        ----------
        timestep : float
            Timestep.
        min_timestep : float
            Minimum timestep.

        Returns
        -------
        message : str
            Error message.
        """
        return f"Timestep too small: {timestep:5.3e} < {min_timestep:5.3e}"

    @classmethod
    def error_too_many_iterations(cls, interations: int) -> str:
        """
        Error message for too many iterations to complete step.

        Parameters
        ----------
        interations : int
            Iterations.

        Returns
        -------
        message : str
            Error message.
        """
        return (
            f"Reached max iterations ({interations}) without finding "
            "acceptable step"
        )

    @classmethod
    def step_adaptive(
        cls,
        cache: AdaptiveSolverCache,
        timestep_controller: TimestepController,
        atol: float,
        rtol: float,
        min_timestep: float,
        max_timestep: float,
        max_iterations: int,
    ):
        """
        Advance ODE system using a timestep adaptively chosen to meet an
        estimated error tolerance.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Solver cache.
        timestep_controller : TimestepController
            Algorithm used to control timestep based on error.
        atol : float
            Absolute tolerance in solution.
        rtol : float
            Relative tolerance in solution.
        min_timestep: float
            Minimum acceptable timestep.
        max_timestep: float
            Maximum acceptable timestep.
        max_iterations: int
            Maximum number of iterations to find an acceptable timestep.
        """
        # Set suggested next timestep.
        cache.timestep_proposed = cache.timestep_next

        for _ in range(max_iterations):
            # Clip timestep to maximum.
            cache.timestep_proposed = np.clip(
                cache.timestep_proposed, 0.0, max_timestep
            )

            # Check if proposed timestep is too small.
            if cache.timestep_proposed < min_timestep:
                cache.stop_condition = cls.error_timestep_too_small(
                    cache.timestep_proposed, min_timestep
                )

                return

            # Attempt step with given timestep.
            cls.step(cache)
            cache.step_attempts += 1

            # Estimate the error.
            cls.estimate_normalised_error(
                cache, timestep_controller, atol, rtol
            )

            # Check if step acceptable and calculate timestep adjustment.
            reject_step, factor = timestep_controller.check_timestep(
                cache.error_proposed,
                cache.error_cache,
                first_step=cache.first_step,
                second_step=cache.second_step,
            )

            if reject_step:
                # Error too large, attempt the step again with an adjusted
                # timestep based on the timestep controller.
                cache.timestep_proposed *= factor
            else:
                # Error acceptable.
                # Attempt to increase timestep for next step only if the
                # first attempted timestep worked. Otherwise it is likely
                # any increased timestep will fail on the next step.
                if cache.step_attempts == 1:
                    cache.timestep_next = factor * cache.timestep_proposed

                return

        # Too many iterations.
        cache.stop_condition = cls.error_too_many_iterations(
            cache.step_attempts
        )


class Integrator:
    """
    Initial value problem integrator main interface.

    Attributes
    ----------
    atol : float
        Absolute tolerance in solution.
    max_iterations : int
        Maximum number of iterations allowed to find a solution of the
        integration step.
    max_timestep : float
        Minimum allowed timestep in IVP solution. Timesteps are clipped
        to be <= maximum.
    min_timestep : float
        Minimum allowed timestep in IVP solution. If timestep < minimum
        integration is stopped and an error is raised.
    rtol : float
        Relative tolerance in solution.
    solver : SolverBase
        Initial value problem solver.
    solver_cache : SolverCache
        Solver cache.
    timestep_controller : TimestepController
        Algorithm for selecting timestep based on estimated error.

    Methods
    -------
    set_dy_dt_func
    set_state
    set_initial_timestep
    set_timestep
    step_manual
    step_adaptive
    accept_step
    """

    __slots__ = (
        "atol",
        "max_iterations",
        "max_timestep",
        "min_timestep",
        "rtol",
        "solver",
        "solver_cache",
        "timestep_controller",
    )

    def __init__(
        self,
        size: int,
        solver: SolverBase,
        atol: float,
        rtol: float,
        min_timestep: float,
        max_timestep: float,
        max_iterations: int,
        timestep_controller: TimestepController,
        /,
        *,
        is_complex: bool,
        primary_size: int | None = None,
    ):
        """
        Inits Integrator.

        Parameters
        ----------
        size : int
            Size of state vector.
        solver : Solver
            IVP solver.
        atol : float
            Absolute tolerance in solution.
        rtol : float
            Relative tolerance in solution.
        min_timestep : float
            Minimum allowed timestep in IVP solution. If timestep < minimum
            integration is stopped and an error is raised.
        max_timestep : float
            Minimum allowed timestep in IVP solution. Timesteps are clipped
            to be <= maximum.
        max_iterations : int
            Maximum number of iterations allowed to find a solution of the
            integration step.
        is_complex: bool
            Flag if state vector is complex valued.
        primary_size : int
            If provided, only this many values at the start of the state vector
            are used for error control.
        """
        # Set solver and create solver cache.
        self.solver = solver
        self.solver_cache = solver.get_cache(
            size, is_complex=is_complex, primary_size=primary_size
        )

        # Set controls on integration.
        self.atol = atol
        self.rtol = rtol
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.max_iterations = max_iterations

        # Timestep controller for adaptive integration.
        self.timestep_controller = timestep_controller

    @property
    def t(self):
        """Time proposed for next step."""
        return self.solver_cache.t_proposed

    @property
    def y(self):
        """State vector proposed for next step."""
        return self.solver_cache.y_proposed

    @property
    def y_last(self):
        """State vector for last accepted step."""
        return self.solver_cache.y_cache[0, :]

    @property
    def timestep(self):
        """Timestep proposed for next step."""
        return self.solver_cache.timestep_proposed

    @property
    def stop_condition(self) -> str:
        """Stop condtion message."""
        return self.solver_cache.stop_condition

    def set_dy_dt_func(
        self, dy_dt_func: typing.Callable[[float, Array], Array]
    ):
        """
        Set function which calculates the time derivative of the state vector.

        Parameters
        ----------
        dy_dt_func : callable[[float, np.array], np.array]
            Function which calculates the rate of change of the state vector.
        """
        self.solver_cache.set_dy_dt_func(dy_dt_func)

    def set_state(self, t0: float, y0: Array):
        """
        Set initial state vector.

        Parameters
        ----------
        t0 : float
            Initial time.
        y0 : np.array
            Initial state vector.
        """
        self.solver_cache.initialise(t0, y0)

    def set_initial_timestep(self, /, *, initial_timestep: float = 0.0):
        """
        Calculate initial timestep of integrator.

        Parameters
        ----------
        initial_timestep : float, optional
            Initial timestep for integration. If not provided, a value is
            automatically calculated.
        """
        # Calculate initial timestep if not provided.
        if initial_timestep > 0:
            self.solver_cache.timestep_proposed = initial_timestep
        else:
            self.solver.calculate_initial_timestep(
                self.solver_cache, self.atol, self.rtol
            )

    def set_timestep(self, timestep: float):
        """
        Set timestep for next step.

        Parameters
        ----------
        timestep : float
            Timestep for next step.
        """
        self.solver_cache.timestep_proposed = timestep

    def step_manual(self, timestep: float):
        """
        Integrate a single step using a provided timestep.

        Parameters
        ----------
        timestep : float
            Timestep for next step.
        """
        self.solver_cache.timestep_proposed = timestep
        self.solver.step(self.solver_cache)

    def step_adaptive(self):
        """
        Integrate a single step using an adaptively chosen timestep. If the
        solver does not support adaptive timestepping a fixed timestep is taken
        using the current proposed timestep.
        """
        if issubclass(self.solver, AdaptiveSolverBase):
            self.solver.step_adaptive(
                self.solver_cache,
                self.timestep_controller,
                self.atol,
                self.rtol,
                self.min_timestep,
                self.max_timestep,
                self.max_iterations,
            )
        else:
            self.step_manual(self.solver_cache.timestep_proposed)

    def accept_step(self, /, *, force_dy_dt_calculation: bool = False):
        """
        Accept proposed time and state vector and calculate first state vector
        derivative at the new position.

        Parameters
        ----------
        force_dy_dt_calculation : bool, optional
            If True, calculate dy/dt at new position ignoring any cached
            values. Default = False.
        """
        # Accept step in solver cache.
        self.solver_cache.accept_step()

        # Prepare for next step from accepted position.
        self.solver_cache.prepare_first_stage(
            reuse_last_dy_dt=self.solver.fsal and not force_dy_dt_calculation
        )
