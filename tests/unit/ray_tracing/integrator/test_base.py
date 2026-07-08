"""
Unit tests for ray_tracing.integrator.base.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.ray_tracing.integrator.base import AdaptiveSolverCache, SolverCache

logger = logging.getLogger(__name__)


class TestSolverCache:
    """
    Unit tests for SolverCache.
    """

    @pytest.fixture
    @staticmethod
    def cache():
        """
        IVP solver cache.

        Returns
        -------
        cache : SolverCache
            Solver cache.
        """
        return SolverCache(2, 5, 2, is_complex=False)

    @staticmethod
    def dy_dt_func(_t: float, y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Function for time derivative of state vector.

        Parameters
        ----------
        _t : float
            Time.
        y : np.ndarray[float]
            State vector.

        Returns
        -------
        dy_dt : np.ndarray[float]
            Time derivative of state vector.
        """
        dy_dt = np.empty(2)
        dy_dt[0] = -y[0]
        dy_dt[1] = -y[1]

        return dy_dt

    @staticmethod
    def test_accept_step(cache: SolverCache):
        """
        Test proposed position is set correctly.

        Parameters
        ----------
        cache : SolverCache
            Solver cache.
        """
        cache.t_proposed = 0.5
        cache.y_proposed[:] = 1.0, 2.0
        cache.step_attempts = 3
        cache.accept_step()

        nptest.assert_allclose(cache.t_cache[0], 0.5)
        nptest.assert_allclose(cache.y_cache[0, :], [1.0, 2.0])
        nptest.assert_allclose(cache.step_attempts_cache[0], 3)

        # Test step attempts resets correctly.
        nptest.assert_allclose(cache.step_attempts, 0)

        # Test cache rolls correct way.
        cache.t_proposed = 1.0
        cache.y_proposed[:] = 3.0, 4.0
        cache.step_attempts = 5
        cache.accept_step()

        nptest.assert_allclose(cache.t_cache[0], 1.0)
        nptest.assert_allclose(cache.t_cache[1], 0.5)
        nptest.assert_allclose(cache.y_cache[0, :], [3.0, 4.0])
        nptest.assert_allclose(cache.y_cache[1, :], [1.0, 2.0])
        nptest.assert_allclose(cache.step_attempts_cache[0], 5)
        nptest.assert_allclose(cache.step_attempts_cache[1], 3)

    def test_prepare_first_stage(self, cache: SolverCache):
        """
        Test preparation of first stage.

        Parameters
        ----------
        cache : SolverCache
            Solver cache.
        """
        cache.t_proposed = 0.5
        cache.y_proposed[:] = 1.0, 2.0
        cache.dy_dt_func = self.dy_dt_func
        cache.accept_step()

        # Test fsal = False.
        cache.prepare_first_stage(reuse_last_dy_dt=False)

        nptest.assert_allclose(cache.t_stages[0], 0.5)
        nptest.assert_allclose(cache.y_stages[0, :], [1.0, 2.0])
        nptest.assert_allclose(cache.dy_dt_stages[0], [-1.0, -2.0])

        # Test fsal = True.
        cache.dy_dt_stages[-1, :] = 8.0, 9.0
        cache.prepare_first_stage(reuse_last_dy_dt=True)

        nptest.assert_allclose(cache.dy_dt_stages[0], 8.0, 9.0)

    def test_initialise(self, cache: SolverCache):
        """
        Test cache initialise.

        Parameters
        ----------
        cache : SolverCache
            Solver cache.
        """
        t0, y0 = 1.0, [2.0, 3.0]
        dy_dt0 = self.dy_dt_func(t0, y0)

        cache.set_dy_dt_func(self.dy_dt_func)
        cache.initialise(t0, y0)

        nptest.assert_allclose(cache.t_cache[0], t0)
        nptest.assert_allclose(cache.y_cache[0, :], y0)

        nptest.assert_allclose(cache.t_stages[0], t0)
        nptest.assert_allclose(cache.y_stages[0, :], y0)
        nptest.assert_allclose(cache.dy_dt_stages[0, :], dy_dt0)

    def test_calculate_stage_dy_dt(self, cache: SolverCache):
        """
        Test calculation of dy/dt for stages.

        Parameters
        ----------
        cache : SolverCache
            Solver cache.
        """
        cache.set_dy_dt_func(self.dy_dt_func)

        expected_value = np.empty(2)
        for stage in range(1, 5):
            cache.t_stages[stage] = 0.0
            cache.y_stages[stage, :] = stage, 2 * stage
            cache.calculate_stage_dy_dt(stage)

            expected_value = self.dy_dt_func(
                cache.t_stages[stage], cache.y_stages[stage, :]
            )

            nptest.assert_allclose(cache.dy_dt_stages[stage], expected_value)


class TestAdaptiveSolverCache:
    """
    Unit tests for AdaptiveSolverCache.
    """

    @pytest.fixture
    @staticmethod
    def cache() -> AdaptiveSolverCache:
        """
        Adaptive IVP solver cache.

        Returns
        -------
        cache : AdaptiveSolverCache
            Adaptive solver cache.
        """
        return AdaptiveSolverCache(2, 5, 2, is_complex=False)

    @staticmethod
    def test_accept_step(cache: AdaptiveSolverCache):
        """
        Test accept step.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Adaptive solver cache.
        """
        # Test value is cached.
        cache.error_proposed = 1.0
        cache.accept_step()

        nptest.assert_allclose(cache.error_cache[0], 1.0)

        # Test cache rolls correct way.
        cache.error_proposed = 2.0
        cache.accept_step()

        nptest.assert_allclose(cache.error_cache[0], 2.0)
        nptest.assert_allclose(cache.error_cache[1], 1.0)

    @staticmethod
    def test_step_flags(cache: AdaptiveSolverCache):
        """
        Test step flags.

        Parameters
        ----------
        cache : AdaptiveSolverCache
            Adaptive solver cache.
        """
        assert cache.first_step
        assert not cache.second_step

        cache.accept_step()
        assert not cache.first_step
        assert cache.second_step

        cache.accept_step()
        assert not cache.first_step
        assert not cache.second_step
