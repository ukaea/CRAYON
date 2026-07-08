"""
Unit tests for ray_tracing.integrator.options.
"""

# Standard imports
import logging
import tempfile

import netCDF4 as nc4  # noqa: N813
import pytest

# Third party imports
# Local imports
from crayon.ray_tracing.integrator.options import (
    ErrorEstimateNorm,
    ExplicitIntegratorType,
    IntegratorFidelity,
    IntegratorType,
    OptionsIntegrator,
    OptionsIntegratorExplicit,
    TimestepControllerType,
)

logger = logging.getLogger(__name__)


class TestOptionsIntegrator:
    """
    Test OptionsIntegrator.
    """

    @pytest.fixture
    @staticmethod
    def options() -> OptionsIntegrator:
        """
        Options for integrator.

        Returns
        -------
        options : OptionsIntegrator
            Options for integrator.
        """
        return OptionsIntegrator(
            solver_type=IntegratorType.EXPLICIT,
            fidelity=IntegratorFidelity.HIGH,
            max_timestep=100.0,
            initial_timestep=0.1,
            timestep_controller=TimestepControllerType.NONE,
            norm=ErrorEstimateNorm.INFINITY,
            explicit=OptionsIntegratorExplicit(
                solver=ExplicitIntegratorType.RK45_CASH_KARP, max_iterations=42
            ),
        )

    @staticmethod
    def assert_equal(obj, other):
        """
        Assert two options are equal.
        """
        assert obj.solver_type == other.solver_type
        assert obj.fidelity == other.fidelity
        assert obj.max_timestep == other.max_timestep
        assert obj.initial_timestep == other.initial_timestep
        assert obj.timestep_controller == other.timestep_controller
        assert obj.norm == other.norm

        ex1, ex2 = obj.explicit, other.explicit
        assert ex1.solver == ex2.solver
        assert ex1.max_iterations == ex2.max_iterations

    def test_round_trip_netcdf(self, options: OptionsIntegrator):
        """
        Test write and reading options from netCDF4 file gives same object.

        Parameters
        ----------
        options : OptionsIntegrator
            Options for integrator.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            options.write_netcdf(dset)
            options2 = OptionsIntegrator.read_netcdf(dset)

        self.assert_equal(options, options2)
