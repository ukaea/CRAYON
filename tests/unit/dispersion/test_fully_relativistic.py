"""
Unit tests for dispersion.fully_relativistic.
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.dispersion.fully_relativistic import q_perp_resonance_curve

logger = logging.getLogger(__name__)


class TestResonanceCurve:
    """
    Unit tests for ResonanceCurve.
    """

    @staticmethod
    @pytest.mark.parametrize(
        ("y", "n_parallel", "n"),
        [(0.85, 0.3, 1), (0.85, -0.3, 1), (0.4, 0.3, 2)],
    )
    def test_no_curve(y: float, n_parallel: float, n: int):
        """
        Test no resonance curve.

        Parameters
        ----------
        y : float
            Normalised magnetic field strength.
        n_parallel : float
            Parallel refractive index.
        n : int
            Harmonic number.
        """
        exists, _, _ = q_perp_resonance_curve(y, n_parallel, n, 0.01)
        assert not exists

    @staticmethod
    def resonance(
        q_perp: float, q_parallel: float, n: int, y: float, n_parallel: float
    ) -> float:
        """
        Resonance condition. Equals zero on resonance curve.

        Parameters
        ----------
        q_perp : float
            Normalised perpendicular momentum.
        q_parallel : float
            Normalised parallel momentum.
        y : float
            Normalised magnetic field strength.
        n_parallel : float
            Parallel refractive index.
        n : int
            Harmonic number.

        Returns
        -------
        resonance : float
            Resonance condition.
        """
        gamma = np.sqrt(1 + q_perp * q_perp + q_parallel * q_parallel)
        return gamma - n * y - n_parallel * q_parallel

    def test_ellipse(self):
        """
        Test ellipsoidal resonance curve (|n_parallel| < 1).
        """
        # This sets q_thermal = 1.
        theta = np.sqrt(2) - 1

        # Test identifies out of range.
        y, n_parallel, n = 0.2, 0.99, 1

        exists, _, _ = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )
        assert not exists

        exists, _, _ = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )
        assert not exists

        # Test satisfies resonance condition.
        # Entire ellipse in range.
        y, n_parallel, n = 0.48, 0.4, 2
        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        assert q_parallel.min() > -1.0
        assert q_parallel.max() < 1.0
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, n_parallel),
            0.0,
            atol=1e-8,
        )

        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        assert q_parallel.min() >= -1.0
        assert q_parallel.max() <= 1.0
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, -n_parallel),
            0.0,
            atol=1e-8,
        )

        # Right hand vertex out of range.
        y, n_parallel, n = 0.65, 0.9, 1
        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        assert q_parallel.min() >= -1.0
        nptest.assert_allclose(q_parallel.max(), 1.0)
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, n_parallel),
            0.0,
            atol=1e-8,
        )

        # Left hand vertex out of range.
        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        nptest.assert_allclose(q_parallel.min(), -1.0)
        assert q_parallel.min() <= 1.0
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, -n_parallel),
            0.0,
            atol=1e-8,
        )

        # Both verticies out of range.
        y, n_parallel, n = 0.8, 0.0, 2
        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        nptest.assert_allclose(q_parallel.min(), -1.0)
        nptest.assert_allclose(q_parallel.max(), 1.0)
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, n_parallel),
            0.0,
            atol=1e-8,
        )

    def test_parabola(self):
        """
        Test parabolic resonance curve (n_parallel = 1).
        """
        # This sets q_thermal = 1.
        theta = np.sqrt(2) - 1

        # Test identifies out of range.
        y, n_parallel, n = 0.3, 1.0, 1

        exists, _, _ = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )
        assert not exists

        exists, _, _ = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )
        assert not exists

        # Test satisfies resonance condition.
        y, n_parallel, n = 0.7, 1.0, 1
        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        assert q_parallel.min() > -1.0
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, n_parallel),
            0.0,
            atol=1e-8,
        )

        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        assert q_parallel.max() <= 1.0
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, -n_parallel),
            0.0,
            atol=1e-8,
        )

    def test_hyperbola(self):
        """
        Test hyperbolic resonance curve (|n_parallel| > 1).
        """
        # This sets q_thermal = 1.
        theta = np.sqrt(2) - 1

        # Test identifies out of range.
        y, n_parallel, n = 0.3, 1.05, 1

        exists, _, _ = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )
        assert not exists

        exists, _, _ = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )
        assert not exists

        # Test satisfies resonance condition.
        y, n_parallel, n = 0.95, 1.1, 3
        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, n_parallel),
            0.0,
            atol=1e-8,
        )

        exists, q_parallel, q_perp = q_perp_resonance_curve(
            y, -n_parallel, n, theta, q_thermal_max=1.0
        )

        assert exists
        nptest.assert_allclose(
            self.resonance(q_perp, q_parallel, n, y, -n_parallel),
            0.0,
            atol=1e-8,
        )
