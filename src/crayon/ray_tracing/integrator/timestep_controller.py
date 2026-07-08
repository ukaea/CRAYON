"""
Algorithms for controlling integration timestep based on error estimates.
"""

# Standard imports
import abc
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class TimestepControllerType(CrayonEnum):
    """
    Methods for timestep control.

    Attributes
    ----------
    NONE
        No timestep control.
    I_NUMERICAL_RECIPES
        Integral controller from numerical recipes in Fortran / C / etc.
    PI_42
        Proportional-integral controller.
    PI_33
        Proportional-integral controller.
    PI_34
        Proportional-integral controller.
    PI_H211
        Proportional-integral controller.
    PID_H312
        Proportional-integral-derivative controller.
    """

    NONE = 1
    I_NUMERICAL_RECIPES = 2
    PI_42 = 3
    PI_33 = 4
    PI_34 = 5
    PI_H211 = 6
    PID_H312 = 7


class ErrorEstimateNorm(CrayonEnum):
    """
    Norm used to calculate total error estimate from error estimates for
    entire state vector.

    Attributes
    ----------
    HARRIER
        Use L2 norm divided by size of array n.
    INFINITY
        Take the infinity norm (largest element) of err.
    """

    HARRIER = 1
    INFINITY = 2


class LimiterBase(abc.ABC):
    """
    Base class for timestep limiter.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __call__(self, value: float) -> float:
        """
        Limit input value.

        Parameters
        ----------
        value : float
            Value to limit.

        Returns
        -------
        value_limited : float
            Limited value.
        """


class TrivialLimiter(LimiterBase):
    """
    Limiter which does nothing.
    """

    __slots__ = ()

    def __call__(self, value: float) -> float:
        """
        Limit input value.

        Parameters
        ----------
        value : float
            Value to limit.

        Returns
        -------
        value_limited : float
            Limited value.
        """
        return value


class LimiterMinMax(LimiterBase):
    """
    Limiter which clips between min and max value.
    """

    __slots__ = ("max_value", "min_value")

    def __init__(self, min_value: float, max_value: float):
        """
        Inits LimiterMinMax.

        Parameters
        ----------
        min_value : float
            Minimum value.
        max_value : float
            Maximum value.
        """
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, value: float) -> float:
        """
        Limit input value.

        Parameters
        ----------
        value : float
            Value to limit.

        Returns
        -------
        value_limited : float
            Limited value.
        """
        return np.clip(value, self.min_value, self.max_value)


class LimiterArctan(LimiterBase):
    """
    Limiter using arctan function.

    Notes
    -----
    The minimum value a timestep can decrease is 1 - pi / 4.
    The maximum value a timestep can increase is 1 + pi / 2.
    """

    __slots__ = ()

    def __call__(self, value: float) -> float:
        """
        Limit input value.

        Parameters
        ----------
        value : float
            Value to limit.

        Returns
        -------
        value_limited : float
            Limited value.
        """
        return 1 + np.arctan(value - 1)


class TimestepController:
    """
    Algorithm to adaptively adjust integrator timestep to control the
    estimated error to a given target.
    """

    __slots__ = (
        "derivative_gain",
        "integral_gain",
        "limiter",
        "norm",
        "proportional_gain",
        "safety_factor",
    )

    def __init__(
        self,
        error_estimate_order: int,
        proportional_gain: float,
        integral_gain: float,
        derivative_gain: float,
        safety_factor: float,
        limiter: LimiterBase,
        norm: ErrorEstimateNorm,
    ):
        """
        Inits TimestepController.

        Attributes
        ----------
        error_estimate_order : int
            Order of the error estimate.
        proportional_gain : float
            Controller proportional gain.
        integral_gain : float
            Controller integral gain.
        derivative_gain : float
            Controller derivative gain.
        safety_factor : float
            Error estimates are multiplied by this safety factor. Set < 1.0.
        limiter: LimiterBase
            The limits the output from the
        norm : ErrorEstimateNorm
            Norm used to construct error estimate from state vector.
        """
        # Correct gains by the order of the error estimate
        self.proportional_gain = (
            float(proportional_gain) / error_estimate_order
        )
        self.integral_gain = float(integral_gain) / error_estimate_order
        self.derivative_gain = float(derivative_gain) / error_estimate_order

        self.safety_factor = float(safety_factor)
        self.limiter = limiter
        self.norm = norm

    @classmethod
    def preset(
        cls,
        controller_type: TimestepControllerType,
        norm: ErrorEstimateNorm,
        error_estimate_order: int,
    ) -> "TimestepController":
        """
        Construct a preset timestep controller.

        Attributes
        ----------
        controller_type : TimestepControllerType
            Name of controller preset.
        norm : ErrorEstimateNorm
            Norm used to construct error estimate from state vector.
        error_estimate_order : int
            Order of error estimate.

        Returns
        -------
        timestep_controller : TimestepController
            Timestep controller.

        Raises
        ------
        ValueError
            Unknown controller_type.
        """
        if controller_type == TimestepControllerType.NONE:
            return NoControl()

        if controller_type == TimestepControllerType.I_NUMERICAL_RECIPES:
            return cls.integral(
                error_estimate_order, 1.0, 0.9, LimiterMinMax(0.2, 10.0), norm
            )

        if controller_type == TimestepControllerType.PI_33:
            return cls.proportional_integral(
                error_estimate_order,
                2.0 / 3.0,
                -1.0 / 3.0,
                0.9,
                LimiterMinMax(0.2, 10.0),
                norm,
            )

        if controller_type == TimestepControllerType.PI_34:
            return cls.proportional_integral(
                error_estimate_order,
                0.7,
                -0.4,
                0.9,
                LimiterMinMax(0.2, 10.0),
                norm,
            )

        if controller_type == TimestepControllerType.PI_42:
            return cls.proportional_integral(
                error_estimate_order,
                0.6,
                -0.2,
                0.9,
                LimiterMinMax(0.2, 10.0),
                norm,
            )

        if controller_type == TimestepControllerType.PI_H211:
            return cls.proportional_integral(
                error_estimate_order,
                1.0 / 6.0,
                1.0 / 6.0,
                0.9,
                LimiterMinMax(0.2, 10.0),
                norm,
            )

        if controller_type == TimestepControllerType.PID_H312:
            return cls.proportional_integral_derivative(
                error_estimate_order,
                1.0 / 18.0,
                1.0 / 9.0,
                1.0 / 18.0,
                0.9,
                LimiterArctan,
                norm,
            )

        raise ValueError(controller_type)

    def normalised_error(
        self,
        error_estimate: FloatArray,
        error_limit: FloatArray,
    ) -> float:
        """
        Calculate normalised error from element-wise error estimate and error
        limit.

        Attributes
        ----------
        error_estimate : np.array[float]
            Error estimate for each state vector element.
        error_limit : np.array[float]
            Limit on error for each state vector element.

        Returns
        -------
        normalised_error : float
            Estimated error normalised to desired error.

        Raises
        ------
        ValueError
            Unknown norm.
        """
        if self.norm == ErrorEstimateNorm.HARRIER:
            return (
                np.linalg.norm(abs(error_estimate / error_limit))
                / error_limit.size
            )

        if self.norm == ErrorEstimateNorm.INFINITY:
            return np.max(abs(error_estimate / error_limit))

        raise NotImplementedError(self.norm)

    def check_timestep(
        self,
        normalised_error_estimate: FloatType,
        normalised_error_history: FloatArray,
        /,
        *,
        first_step: bool,
        second_step: bool,
    ) -> tuple[bool, float]:
        """
        Check if step has acceptable error and recommend factor to modify
        timestep by based on the error estimate.

        Attributes
        ----------
        normalised_error_estimate : float
            Error estimate normalised to desired error.
        normalised_error_history : np.array[float]
            Normalised error estimate from previous steps.
        first_step : bool
            Flag if this is the first integration step.
        second_step : bool
            Flag if this is the second integration step.

        Returns
        -------
        reject_step : bool
            Whether to reject the step as the error is too large.
        factor : float
            Factor to adjust the timestep by.
        """
        eps_norm = normalised_error_estimate
        eps_norm_m1 = normalised_error_history[0]
        eps_norm_m2 = normalised_error_history[1]

        # Timestep adjustment factor.
        skip = False
        factor = 1.0

        # Reject step if normalised error is above 1.
        reject_step = eps_norm > 1

        # If integral gain is 0 there is no integral control.
        if not np.isclose(self.integral_gain, 0.0):
            if np.isclose(eps_norm, 0.0):
                skip = True
                factor = np.inf if self.integral_gain > 0 else 0.0
            else:
                factor *= eps_norm**-self.integral_gain

        # If proportional gain is zero there is no proportional control.
        # Also on the first step we do not have the normalised error for the
        # last timestep so only apply integral control.
        if (
            not skip
            and not first_step
            and not np.isclose(self.proportional_gain, 0.0)
        ):
            if np.isclose(eps_norm_m1, 0.0):
                skip = True
                factor = np.inf if self.proportional_gain > 0 else 0.0
            else:
                factor *= eps_norm_m1**-self.proportional_gain

        # If derivative gain is zero there is no derivative control.
        # Also on the second step we do not have the normalised error for the
        # second last timestep so only apply proportional and integral control.
        if (
            not skip
            and not (first_step or second_step)
            and not np.isclose(self.derivative_gain, 0.0)
        ):
            if np.isclose(eps_norm_m2, 0.0):
                skip = True
                factor = np.inf if self.derivative_gain > 0 else 0.0
            else:
                factor *= eps_norm_m2**-self.derivative_gain

        # Limit timestep adjustment factor to acceptable range
        factor = self.limiter(self.safety_factor * factor)

        return reject_step, factor

    @classmethod
    def integral(
        cls,
        error_estimate_order: int,
        integral_gain: float,
        safety_factor: float,
        limiter: LimiterBase,
        norm: ErrorEstimateNorm,
    ):
        """
        Construct integral controller.

        Parameters
        ----------
        error_estimate_order : int
            Order of the error estimate.
        integral_gain : float
            Controller integral gain.
        safety_factor : float
            Error estimates are multiplied by this safety factor. Set < 1.0.
        limiter: LimiterBase
            The limits the output from the
        norm : ErrorEstimateNorm
            Norm used to construct error estimate from state vector.

        Returns
        -------
        timestep_controller : TimestepController
            Timestep controller.
        """
        return cls(
            error_estimate_order,
            0.0,
            integral_gain,
            0.0,
            safety_factor,
            limiter,
            norm,
        )

    @classmethod
    def proportional_integral(
        cls,
        error_estimate_order: int,
        proportional_gain: float,
        integral_gain: float,
        safety_factor: float,
        limiter: LimiterBase,
        norm: ErrorEstimateNorm,
    ):
        """
        Construct proportional integral controller.

        Parameters
        ----------
        error_estimate_order : int
            Order of the error estimate.
        proportional_gain : float
            Controller proportional gain.
        integral_gain : float
            Controller integral gain.
        safety_factor : float
            Error estimates are multiplied by this safety factor. Set < 1.0.
        limiter: LimiterBase
            The limits the output from the
        norm : ErrorEstimateNorm
            Norm used to construct error estimate from state vector.

        Returns
        -------
        timestep_controller : TimestepController
            Timestep controller.
        """
        return cls(
            error_estimate_order,
            integral_gain,
            proportional_gain,
            0.0,
            safety_factor,
            limiter,
            norm,
        )

    @classmethod
    def proportional_integral_derivative(
        cls,
        error_estimate_order: int,
        proportional_gain: float,
        integral_gain: float,
        derivative_gain: float,
        safety_factor: float,
        limiter: LimiterBase,
        norm: ErrorEstimateNorm,
    ):
        """
        Construct proportional integral derivative controller.

        Parameters
        ----------
        error_estimate_order : int
            Order of the error estimate.
        proportional_gain : float
            Controller proportional gain.
        integral_gain : float
            Controller integral gain.
        derivative_gain : float
            Controller derivative gain.
        safety_factor : float
            Error estimates are multiplied by this safety factor. Set < 1.0.
        limiter: LimiterBase
            The limits the output from the
        norm : ErrorEstimateNorm
            Norm used to construct error estimate from state vector.

        Returns
        -------
        timestep_controller : TimestepController
            Timestep controller.
        """
        return cls(
            error_estimate_order,
            integral_gain,
            proportional_gain,
            derivative_gain,
            safety_factor,
            limiter,
            norm,
        )


class NoControl(TimestepController):
    """
    Timestep controller which does nothing. Used for integrators which do not
    support adaptive timestep.
    """

    __slots__ = ()

    def __init__(*_args, **_kwargs):
        """
        Inits NoControl.
        """

    @staticmethod
    def normalised_error(*_args, **_kwargs) -> float:
        """
        Calculate normalised error from element-wise error estimate and error
        limit.

        Returns
        -------
        normalised_error : float
            Estimated error normalised to desired error.
        """
        return 0.0

    @staticmethod
    def check_timestep(*_args, **_kwargs) -> tuple[bool, float]:
        """
        Check if step has acceptable error and recommend factor to modify
        timestep by based on the error estimate.

        Returns
        -------
        reject_step : bool
            Whether to reject the step as the error is too large.
        factor : float
            Factor to adjust the timestep by.
        """
        return False, 1.0
