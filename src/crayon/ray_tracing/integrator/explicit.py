"""
Explicit integration methods.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.ray_tracing.integrator.base import (
    AdaptiveSolverBase,
    AdaptiveSolverCache,
    SolverBase,
)
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class ExplicitIntegratorType(CrayonEnum):
    """
    Explicit integration methods.

    Attributes
    ----------
    RK4_RUNGE
        Runge-Kutta 4th order with 'classic' coefficients found in any
        introductory textbooks. Only use for testing!
    RK45_DORMAND_PRINCE
        Runge-Kutta 4th order with 5th order error estimate and 4th order
        interpolating polynomial. Coefficients from Dormand and Prince.
    RK45_CASH_KARP
        Runge-Kutta 4th order with 5th order error estimate. Coefficients from
        Cash and Karp.
    RK45_TSITOURAS
        Runge-Kutta 4th order with 5th order error estimate and 4th order
        interpolating polynomial. Coefficients from Tsitouras.
    """

    RK4_RUNGE = 1
    RK45_DORMAND_PRINCE = 2
    RK45_CASH_KARP = 3
    RK45_TSITOURAS = 4


class ExplicitSolverBase(SolverBase):
    """
    Base class for explicit integrator.
    """


class ExplicitAdaptiveSolverBase(AdaptiveSolverBase):
    """
    Base class for adaptive explicit integrator.
    """


# Explicit Runge-Kutta methods.
class ExplicitRungeKutta(ExplicitAdaptiveSolverBase):
    """
    An s stage Runge-Kutta integrator. Adapted from [1].

    Attributes
    ----------
    stages : int
        Number of stages in integration step.
    steps : int
        Numbers of steps per integration step.
    step_stages : int
        Numbers of stages used to calculate integration step. Equal to
        stages unless there is an embedded error estimate.
    weights_a : np.array[float]
        Weights for Runge-Kutta tableau.
    weights_b : np.array[float]
        Weights for Runge-Kutta tableau.
    weights_c : np.array[float]
        Weights for Runge-Kutta tableau.
    fsal : bool
        Flag if 'first same as last' i.e. first stage of next step is same as
        last stage of previous step.
    interpolant_order : int
        Order of polynomial interpolant.
    weights_interpolant : np.array[float]
        Weight for calculating polynomial interpolant.

    References
    ----------
    [1] Eric Jones, Travis Oliphant, Pearu Peterson and others, "SciPy: Open
        Source Scientific Tools for Python", 2001, "http://www.scipy.org/"
    """

    stages: int = NotImplemented
    steps: int = 1
    step_stages: int = NotImplemented

    # Weights in Runge-Kutta tableau.
    weights_a: FloatArray = NotImplemented
    weights_b: FloatArray = NotImplemented
    weights_c: FloatArray = NotImplemented
    fsal: bool = False

    # Weights for constructing dense interpolant. If not provided a cubic
    # Lagrange polynomial is constructed.
    interpolant_order: int = 3
    weights_interpolant: FloatArray = NotImplemented

    @classmethod
    def get_cache(
        cls, size: int, /, *, is_complex: bool, primary_size: int | None = None
    ) -> AdaptiveSolverCache:
        """
        Get solver cache for explicit Runge-Kutta method.

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
        solver_cache : AdaptiveSolverCache
            Solver cache.
        """
        return AdaptiveSolverCache(
            size,
            cls.stages,
            cls.steps,
            is_complex=is_complex,
            primary_size=primary_size,
        )

    @classmethod
    def step(
        cls,
        cache: AdaptiveSolverCache,
    ):
        """
        Integrate a single timestep.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Solver cache.
        """
        a, c = cls.weights_a, cls.weights_c
        dt = cache.timestep_proposed

        for stage, (_a, _c) in enumerate(zip(a, c, strict=True), start=1):
            # Calculate t, y for next stage.
            cache.t_stages[stage] = cache.t_stages[0] + _c * dt
            cache.y_stages[stage, :] = cache.y_stages[0] + dt * np.dot(
                cache.dy_dt_stages[:stage, :].T, _a[:stage]
            )

            # Calculate dy/dt at stage position.
            cache.calculate_stage_dy_dt(stage)

        # Calculate new proposed solution.
        cache.t_proposed = cache.t_stages[0] + dt
        cache.y_proposed[:] = cache.y_stages[0] + dt * np.dot(
            cache.dy_dt_stages[: cls.step_stages, :].T, cls.weights_b
        )

        if cls.stages > cls.step_stages:
            # Calculate additional stage at integration position required for
            # embedded error estimate.
            cache.t_stages[cls.stages - 1] = cache.t_proposed
            cache.y_stages[cls.stages - 1] = cache.y_proposed

            # Calculate dy/dt at stage position.
            cache.calculate_stage_dy_dt(cls.stages - 1)

    @classmethod
    def estimate_error(cls, cache: AdaptiveSolverCache):
        """
        Calculate error estimate for integration step using step doubling.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Solver cache.

        Returns
        -------
        error_estimate : float
            Error estimate.
        """
        # Copy solution using full timestep.
        _timestep_proposed = cache.timestep_proposed
        _t_proposed = cache.t_proposed
        _y_proposed = cache.y_proposed.copy()

        # Take two steps with half the timestep.
        cache.timestep_proposed *= 0.5

        cls.step(cache)

        cache.t_stages[0] = cache.t_proposed
        cache.y_stages[0, :] = cache.y_proposed
        cls.step(cache)

        # Calculate error estimate.
        err = (cache.y_proposed - _y_proposed) / (2**cls.order - 1)

        # Restore solution using full timestep
        cache.timestep_proposed = _timestep_proposed
        cache.t_proposed = _t_proposed
        cache.y_proposed[:] = _y_proposed

        return err


class RK4Runge(ExplicitRungeKutta):
    """
    'Classic' 4th order Runge-Kutta. Coefficients are those found in
    introductory textbooks on Runge-Kutta analysis and are from [1].

    Notes
    -----
    This method is only included for benchmarking. Do not use for real
    problems as it is extremely inefficient.

    References
    ----------
    [1] M. Kutta, "Beitrag zur näherungsweisen Integration totaler
        Differentialgleichungen", Zeitschrift für Mathematik und Physik,
        Vol. 46, pp. 435-453, 1901.
    """

    __slots__ = ()

    order = 5
    stages = 4
    step_stages = 4
    error_estimate_order = 4
    stability_size = 0.0

    weights_a: FloatArray = np.array(
        [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1.0, 0]], dtype=FloatType
    )

    weights_b: FloatArray = np.array(
        [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=FloatType
    )

    weights_c: FloatArray = np.array([0.5, 0.5, 1.0], dtype=FloatType)


class EmbeddedExplicitRungeKutta(ExplicitRungeKutta):
    """
    An s stage Runge-Kutta integrator with an embedded error estimate. Adapted
    from [1].

    Attributes
    ----------
    weights_error : np.float[array]
        Weights for embedded error estimate.

    References
    ----------
    [1] Eric Jones, Travis Oliphant, Pearu Peterson and others, "SciPy: Open
        Source Scientific Tools for Python", 2001, "http://www.scipy.org/"
    """

    # Accuracy of error estimate
    weights_error: FloatArray = NotImplemented

    @classmethod
    def estimate_error(cls, cache: AdaptiveSolverCache):
        """
        Calculate error estimate for integration step using higher order
        embedded method.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Solver cache.

        Returns
        -------
        error_estimate : float
            Error estimate.
        """
        return cache.timestep_proposed * np.dot(
            cache.dy_dt_stages.T, cls.weights_error
        )


class RK45DormandPrince(EmbeddedExplicitRungeKutta):
    """
    4th order Runge-Kutta with 5th order error estimate. Adapted from [1].
    Coefficients are taken from [2] while the interpolant coefficients are
    constructed using the procedure from [3].

    References
    ----------
    [1] E. Jones, T. Oliphant, P. Peterson and others, "SciPy: Open Source
        Scientific Tools for Python", 2001, "http://www.scipy.org/"
    [2] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
        formulae", Journal of Computational and Applied Mathematics, Vol. 6,
        No. 1, pp. 19-26, 1980.
    [3] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
        of Computation, Vol. 46, No. 173, pp. 135-150, 1986.
    """

    __slots__ = ()

    order = 5
    stages = 7
    step_stages = 6
    fsal = True
    error_estimate_order = 4
    interpolant_order = 4
    stability_size = 0.0

    weights_a: FloatArray = np.array(
        [
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ],
        dtype=FloatType,
    )

    # \hat{b}_i from [1].
    weights_b: FloatArray = np.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        dtype=FloatType,
    )

    weights_c: FloatArray = np.array(
        [1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], dtype=FloatType
    )

    # b_i - \hat{b}_i from [1]
    weights_error: FloatArray = np.array(
        [
            -71 / 57600,
            0,
            71 / 16695,
            -71 / 1920,
            17253 / 339200,
            -22 / 525,
            1 / 40,
        ],
        dtype=FloatType,
    )

    weights_interpolant: FloatArray = np.array(
        [
            [
                1,
                -8048581381 / 2820520608,
                8663915743 / 2820520608,
                -12715105075 / 1128208243,
            ],
            [0, 0, 0, 0],
            [
                0,
                131558114200 / 32700410799,
                -68118460800 / 10900136933,
                87487479700 / 32700410799,
            ],
            [
                0,
                -1754552775 / 470086768,
                14199869525 / 1410260304,
                -10690763975 / 1880347072,
            ],
            [
                0,
                127303824393 / 49829197408,
                -318862633887 / 49829197408,
                701980252875 / 199316789632,
            ],
            [
                0,
                -282668133 / 205662961,
                2019193451 / 616988883,
                -1453857185 / 822651844,
            ],
            [
                0,
                40617522 / 29380423,
                -110615467 / 29380423,
                69997945 / 29380423,
            ],
        ],
        dtype=FloatType,
    )


class RK45CashKarp(EmbeddedExplicitRungeKutta):
    """
    4th order Runge-Kutta with 5th order error estimate. Coefficients are taken
    from [1] while the interpolant coefficients are constructed using the
    procedure from [2].

    References
    ----------
    [1] J. R. Cash, A. H. Karp, "A Variable Order Runge-Kutta Method for
        Initial Value Problems with Rapidly Varying Right-Hand Sides",
        ACM Transactions on Mathematical Software, Vol. 16, No. 3, pp. 201-222,
        1990.
    [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
        of Computation, Vol. 46, No. 173, pp. 135-150, 1986.
    """

    __slots__ = ()

    order = 5
    stages = 7
    step_stages = 6
    fsal = True
    error_estimate_order = 4
    interpolant_order = 4
    stability_size = 0.0

    weights_a: FloatArray = np.array(
        [
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [3 / 10, -9 / 10, 6 / 5, 0, 0],
            [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0],
            [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
        ],
        dtype=FloatType,
    )

    # Order 4 in [1].
    weights_b: FloatArray = np.array(
        [2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4],
        dtype=FloatType,
    )

    weights_c: FloatArray = np.array(
        [1 / 5, 3 / 10, 3 / 5, 1, 7 / 8], dtype=FloatType
    )

    # Order 5 - Order 4 in [1].
    weights_error: FloatArray = np.array(
        [
            -277 / 64512,
            0,
            6925 / 370944,
            -6925 / 202752,
            -277 / 14336,
            277 / 7084,
            0,
        ],
        dtype=FloatType,
    )

    weights_interpolant: FloatArray = np.array(
        [
            [1, -10405 / 3843, 32357 / 11529, -855 / 854],
            [0, 0, 0, 0],
            [0, 308500 / 88389, -1424000 / 265167, 67250 / 29463],
            [0, 5875 / 24156, 12875 / 36234, -3125 / 8052],
            [0, 235 / 1708, -235 / 854, 235 / 1708],
            [0, -287744 / 108031, 700416 / 108031, -381440 / 108031],
            [0, 3 / 2, -4, 5 / 2],
        ],
        dtype=FloatType,
    )


class RK45Tsitouras(EmbeddedExplicitRungeKutta):
    """
    4th order Runge-Kutta with 5th order error estimate. Coefficients and
    interpolant coefficients are from [1].

    References
    ----------
    [1] Ch. Tsitouras "Runge-Kutta pairs of order 5(4) satisfying only the
        first column simplifying assumption", Computers and Mathematics with
        Applications, Vol. 62, No. 2, pp. 770-775, 2011.
    """

    __slots__ = ()

    order = 5
    stages = 7
    step_stages = 6
    fsal = True
    error_estimate_order = 4
    interpolant_order = 4
    stability_size = 3.5068

    weights_a: FloatArray = np.array(
        [
            [0.161, 0, 0, 0, 0, 0],
            [-0.008480655492356989, 0.335480655492357, 0, 0, 0, 0],
            [
                2.8971530571054935,
                -6.359448489975075,
                4.3622954328695815,
                0,
                0,
                0,
            ],
            [
                5.325864828439257,
                -11.748883564062828,
                7.4955393428898365,
                -0.09249506636175525,
                0,
                0,
            ],
            [
                5.86145544294642,
                -12.92096931784711,
                8.159367898576159,
                -0.071584973281401,
                -0.028269050394068383,
                0,
            ],
        ],
        dtype=FloatType,
    )

    weights_b: FloatArray = np.array(
        [
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
        ],
        dtype=FloatType,
    )

    weights_c: FloatArray = np.array(
        [0.161, 0.327, 0.9, 0.9800255409045097, 1], dtype=FloatType
    )

    # Corrected last term to include a minus sign missing in the paper.
    weights_error = np.array(
        [
            0.00178001105222577714,
            0.0008164344596567469,
            -0.007880878010261995,
            0.1447110071732629,
            -0.5823571654525552,
            0.45808210592918697,
            -0.015151515151515152,
        ],
        dtype=FloatType,
    )

    # Expand interpolant in [1].
    weights_interpolant: FloatArray = np.array(
        [
            [1.0, -2.763706197274826, 2.9132554618219126, -1.0530884977290216],
            [0.0, 0.1317, -0.2234, 0.1017],
            [0.0, 3.9302962368947516, -5.941033872131505, 2.490627285651253],
            [0.0, -12.411077166933676, 30.33818863028232, -16.548102889244902],
            [0.0, 37.50931341651104, -88.1789048947664, 47.37952196281928],
            [0.0, -27.896526289197286, 65.09189467479366, -34.87065786149661],
            [0.0, 1.5, -4.0, 2.5],
        ],
        dtype=FloatType,
    )


# Linear multistep methods (Adams Bashford)


# Semi-Implicit methods (predictor-corrector)


explicit_solvers = {
    ExplicitIntegratorType.RK4_RUNGE: RK4Runge,
    ExplicitIntegratorType.RK45_DORMAND_PRINCE: RK45DormandPrince,
    ExplicitIntegratorType.RK45_CASH_KARP: RK45CashKarp,
    ExplicitIntegratorType.RK45_TSITOURAS: RK45Tsitouras,
}
