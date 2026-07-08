"""
Wrapper scripts for ray tracing.
"""

# Standard imports
import logging
import pathlib
from concurrent import futures

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np
import toml

# Local imports
from crayon.ray_tracing.caches import CoordinateCache, PlasmaCache
from crayon.ray_tracing.initial_conditions import (
    InitialConditions,
    read_initial_conditions_toml,
)
from crayon.ray_tracing.options import read_options_ray_tracing_toml
from crayon.ray_tracing.ray_tracer import (
    OptionsIntegrator,
    OptionsRayTracing,
    RayTracer,
)
from crayon.shared.constants import (
    INPUT,
    INPUT_DATA_NETCDF,
    OPTIMAL_OX_TOML,
    OPTIONS_TOML,
    OUTPUT,
    RAYS_NETCDF,
    RAYS_TOML,
    SYSTEM_DATA_TOML,
)
from crayon.shared.dimensions import Dimensions
from crayon.system_data import SystemData, SystemDataProvider

logger = logging.getLogger(__name__)


def trace_single(
    system_data: SystemData,
    options_ray_tracing: OptionsRayTracing,
    options_integrator: OptionsIntegrator,
    initial_conditions: InitialConditions,
    dset: nc4.Dataset,
):
    """
    Trace rays on a single process.

    Parameters
    ----------
    system_data : SystemData
        Object containing plasma system data.
    options_ray_tracing : OptionsRayTracing
        Options for ray tracing.
    options_integrator : OptionsIntegrator
        Options for integration.
    initial_conditions : InitialConditions
        Ray initial conditions.
    dset : netCDF4.Dataset
        netCDF4 dataset to write output data into.
    """
    max_generations = options_ray_tracing.max_generations
    new_initial_conditions = []

    for i in range(1, max_generations + 1):
        for initial_condition in initial_conditions:
            ray_tracer = RayTracer(
                system_data, options_ray_tracing, options_integrator
            )

            ray_tracer.trace(initial_condition)

            if ray_tracer.children:
                # If we reached the max generation then warn.
                if i == max_generations:
                    logger.warning(
                        "[%s] Reached max ray generations (%s)",
                        ray_tracer.ray.name,
                        max_generations,
                    )
                else:
                    new_initial_conditions.extend(ray_tracer.children)

            try:
                ray_tracer.output.write_netcdf(
                    dset.createGroup(initial_condition.name),
                    ray_tracer.ray.index + 1,
                    ray_tracer.ray.conversions,
                )
            except Exception:
                logger.exception(
                    "[%s] Writing ray tracing output failed.",
                    ray_tracer.ray.name,
                )

        # If no new rays were generated then break.
        if new_initial_conditions:
            initial_conditions = new_initial_conditions
            new_initial_conditions = []
        else:
            break


def _trace_worker(
    ray_tracer: RayTracer, initial_conditions: InitialConditions
) -> RayTracer:
    """
    Worker for ray tracing for parallel processes.

    Parameters
    ----------
    ray_tracer: RayTracer
        Ray tracer.
    initial_conditions: InitialConditions
        Ray initial conditions.

    Returns
    -------
    ray_tracer : RayTracer
        Ray tracer.
    """
    # Trace ray.
    try:
        ray_tracer.trace(initial_conditions)
    except Exception:
        logger.exception("[%s]: Ray tracing failed.", ray_tracer.ray.name)

    return ray_tracer


def trace_parallel(
    system_data,
    options_ray_tracing,
    options_integrator,
    initial_conditions,
    dset: nc4.Dataset,
    max_workers: int,
):
    """
    Trace rays on using multiple process.

    Parameters
    ----------
    system_data : SystemData
        Object containing plasma system data.
    options_ray_tracing : OptionsRayTracing
        Options for ray tracing.
    options_integrator : OptionsIntegrator
        Options for integration.
    initial_conditions : InitialConditions
        Ray initial conditions.
    dset : netCDF4.Dataset
        netCDF4 dataset to write output data into.
    max_workers : int
        Number of parallel processes to trace rays with.
    """
    max_generations = options_ray_tracing.max_generations
    _futures = []

    with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for i in range(1, max_generations + 1):
            for initial_condition in initial_conditions:
                ray_tracer = RayTracer(
                    system_data, options_ray_tracing, options_integrator
                )

                future = pool.submit(
                    _trace_worker, ray_tracer, initial_condition
                )
                _futures.append(future)

            # Clear initial conditions.
            initial_conditions = []

            for future in futures.as_completed(_futures, timeout=300.0):
                ray_tracer = future.result()

                if len(ray_tracer.children) > 0:
                    # If we reached the max generation then warn.
                    if i == max_generations:
                        logger.warning(
                            "[%s] Reached max ray generations (%s)",
                            ray_tracer.ray.name,
                            max_generations,
                        )
                    else:
                        initial_conditions.extend(ray_tracer.children)

                try:
                    ray_tracer.output.write_netcdf(
                        dset.createGroup(ray_tracer.ray.name),
                        ray_tracer.ray.index + 1,
                        ray_tracer.ray.conversions,
                    )
                except Exception:
                    logger.exception(
                        "[%s] Writing ray tracing output failed.",
                        ray_tracer.ray.name,
                    )

            # If no new rays were generated then break.
            if len(initial_conditions) == 0:
                break


def trace(
    run_directory: pathlib.Path,
    times_s: list[float],
    /,
    *,
    overwrite: bool = False,
    max_workers: int = 1,
):
    """
    Wrapper script for ray tracing.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    times_s : list[float]
        Times to run.
    overwrite : bool, optional
        If True, overwrite existing output file. Default = False.
    max_workers : int
        Number of parallel processes to trace rays with.

    Raises
    ------
    ValueError
        Output data file exists and overwrite = False.
    """
    # Load input files.
    with open(run_directory.joinpath(INPUT, OPTIONS_TOML)) as fh:
        options_ray_tracing, options_integrator = (
            read_options_ray_tracing_toml(fh)
        )

    with open(run_directory.joinpath(INPUT, RAYS_TOML)) as fh:
        ray_initial_conditions = read_initial_conditions_toml(fh)

    with open(run_directory.joinpath(INPUT, SYSTEM_DATA_TOML)) as fh:
        system_data_provider = SystemDataProvider.read_toml(fh)

    for time_s in times_s:
        # Build system data object at given time.
        system_data = system_data_provider.build(time_s)

        # Create directory containing output file if it doesn't exist.
        output_dir = run_directory.joinpath(OUTPUT, f"{time_s}s")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Unpack ray initial conditions.
        initial_conditions = []

        coordinate_cache = CoordinateCache(system_data.coordinate_coordinator)
        plasma_cache = PlasmaCache(system_data.kinetic, system_data.magnetic)

        # Load optimal OX launch conditions if available.
        optimal_ox_file = output_dir.joinpath(OPTIMAL_OX_TOML)

        if optimal_ox_file.exists():
            with open(optimal_ox_file) as fh:
                document = toml.load(fh)

            optimal_refractive_index = {
                name: [
                    np.asarray(x) for x in data["refractive_index_cartesian"]
                ]
                for name, data in document.items()
            }
        else:
            optimal_refractive_index = {}

        for initial_condition in ray_initial_conditions:
            initial_conditions.extend(
                initial_condition.unpack(
                    coordinate_cache,
                    plasma_cache,
                    optimal_refractive_index.get(
                        f"{initial_condition.name}-0", []
                    ),
                )
            )

        # Write input data.
        input_data_file = output_dir.joinpath(INPUT_DATA_NETCDF)

        if not overwrite and input_data_file.exists():
            raise ValueError(
                f"Input data file already exists: {input_data_file}"
            )

        with nc4.Dataset(input_data_file, "w", auto_complex=True) as dset:
            # Write global dimensions.
            Dimensions.write_netcdf(dset)

            # Write input data.
            _options = dset.createGroup("options")
            _system_data = dset.createGroup("system_data")
            _initial_conditions = dset.createGroup("initial_conditions")

            options_ray_tracing.write_netcdf(
                _options.createGroup("ray_tracing")
            )
            options_integrator.write_netcdf(_options.createGroup("integrator"))

            system_data.write_netcdf(_system_data)

            for initial_condition in initial_conditions:
                initial_condition.write_netcdf(
                    _initial_conditions.createGroup(initial_condition.name)
                )

        # Open ray data file.
        output_file = output_dir.joinpath(RAYS_NETCDF)

        if not overwrite and output_file.exists():
            raise ValueError(f"Output file already exists: {output_file}")

        with nc4.Dataset(output_file, "w", auto_complex=True) as dset:
            Dimensions.write_netcdf(dset)

            if max_workers == 1:
                trace_single(
                    system_data,
                    options_ray_tracing,
                    options_integrator,
                    initial_conditions,
                    dset,
                )
            else:
                trace_parallel(
                    system_data,
                    options_ray_tracing,
                    options_integrator,
                    initial_conditions,
                    dset,
                    max_workers,
                )
