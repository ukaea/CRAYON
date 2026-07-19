"""
Unit tests for ray_tracing.options.
"""

# Standard imports
import logging
import pathlib
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
from crayon.ray_tracing.options import (
    OptionsRayTracing,
    read_options_ray_tracing_toml,
    write_options_ray_tracing_toml,
)

logger = logging.getLogger(__name__)


class TestOptionsRayTracing:
    """
    Unit tests for OptionsRayTracing.
    """

    @pytest.fixture
    @staticmethod
    def options() -> OptionsRayTracing:
        """
        Ray tracing options.

        Returns
        -------
        options : OptionsRayTracing
            Ray tracing options.
        """
        return OptionsRayTracing(
            max_ray_nodes=1,
            max_ray_children=2,
            max_generations=3,
            max_optical_depth=4.0,
            max_reflections=5.0,
            min_power_fraction_new_ray=6.0,
        )

    @staticmethod
    def assert_equal(obj: OptionsRayTracing, other: OptionsRayTracing):
        """
        Assert two ray tracing options are equal.

        Parameters
        ----------
        obj, other : OptionsRayTracing
            Objects to check.
        """
        assert obj.max_ray_nodes == other.max_ray_nodes
        assert obj.max_ray_children == other.max_ray_children
        assert obj.max_generations == other.max_generations
        assert obj.max_optical_depth == other.max_optical_depth
        assert obj.max_reflections == other.max_reflections
        assert (
            obj.min_power_fraction_new_ray == other.min_power_fraction_new_ray
        )

    def test_round_trip_netcdf(self, options):
        """
        Test writing and reading object through netCDF4 file gives same object.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            nc4.Dataset(pathlib.Path(tmpdir).joinpath("tmp.nc"), "w") as dset,
        ):
            options.write_netcdf(dset)
            options2 = OptionsRayTracing.read_netcdf(dset)

        self.assert_equal(options, options2)


def assert_options_integrator_equal(
    obj: OptionsIntegrator, other: OptionsIntegrator
):
    """
    Assert integrator options are equal.

    Parameters
    ----------
    obj, other : OptionsIntegrator
        Objects to check.
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


def test_options_ray_tracing_round_trip_toml():
    """
    Test functions for writing options to / from netCDF4.
    """
    options_ray_tracing = OptionsRayTracing(
        max_ray_nodes=1,
        max_ray_children=2,
        max_generations=3,
        max_optical_depth=4.0,
        max_reflections=5.0,
        min_power_fraction_new_ray=6.0,
    )

    options_integrator = OptionsIntegrator(
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

    with tempfile.TemporaryFile("r+") as fh:
        write_options_ray_tracing_toml(
            fh, options_ray_tracing, options_integrator
        )

        fh.seek(0)
        options_ray_tracing_2, options_integrator_2 = (
            read_options_ray_tracing_toml(fh)
        )

    TestOptionsRayTracing.assert_equal(
        options_ray_tracing, options_ray_tracing_2
    )
    assert_options_integrator_equal(options_integrator, options_integrator_2)
