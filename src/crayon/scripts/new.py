"""
Wrapper script for creating new Crayon run directory.
"""

# Standard imports
import logging
import pathlib

# Local imports
from crayon.ray_tracing.initial_conditions import (
    InitialConditionsSchema,
    PolarisationWaveMode,
    RefractiveIndexNparallel,
    write_initial_conditions_toml,
)
from crayon.ray_tracing.options import (
    OptionsIntegrator,
    OptionsRayTracing,
    write_options_ray_tracing_toml,
)
from crayon.shared.constants import (
    INPUT,
    OPTIONS_TOML,
    RAYS_TOML,
    SYSTEM_DATA_TOML,
    CoordinateSystem,
    WaveMode,
)
from crayon.shared.physics import (
    critical_density_per_m3,
    critical_magnetic_field_strength_t,
)
from crayon.system_data import SystemDataProvider
from crayon.system_data.limiter import LimiterEffect
from crayon.system_data.schemas import (
    LimiterAnalyticBoundingBox2D,
    ModelAnalyticConstant,
    ModelAnalyticRamp,
)

logger = logging.getLogger(__name__)


def new(run_directory: pathlib.Path, /, *, overwrite: bool = False):
    """
    Create Crayon run template.

    Attributes
    ----------
    run_directory : pathlib.Path
        Directory to create new run directory within.
    overwrite : bool, optional
        If True, overwrite any existing files. Default = False.

    Raises
    ------
    ValueError
        Run directory not empty and overwrite = False.
    """
    # Create input directory. Create parents if they don't exist.
    input_directory = run_directory.joinpath(INPUT)

    logger.info("Creating new Crayon run directory: %s", run_directory)
    input_directory.mkdir(parents=True, exist_ok=True)

    # Check if files already exist.
    if not overwrite and (
        input_directory.joinpath(OPTIONS_TOML).exists()
        or input_directory.joinpath(SYSTEM_DATA_TOML).exists()
        or input_directory.joinpath(RAYS_TOML).exists()
    ):
        raise ValueError("Run directory already in use.")

    # Write template input files.
    logger.info("Writing template input files")

    options_ray_tracing = OptionsRayTracing()
    options_integrator = OptionsIntegrator()

    with open(input_directory.joinpath(OPTIONS_TOML), "w") as fh:
        write_options_ray_tracing_toml(
            fh, options_ray_tracing, options_integrator
        )

    # Write plasma slab OXB case.
    frequency_ghz = 10.0

    x0, x1 = 0.0, 2.0
    ne_crit = critical_density_per_m3(frequency_ghz)

    y0, y1 = 0.75, 1.0
    b_crit = critical_magnetic_field_strength_t(frequency_ghz)

    # Write system data.
    data_sources = {}
    coordinates = {}
    electron_density_per_m3 = ModelAnalyticRamp(
        CoordinateSystem.CARTESIAN,
        [0.2, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        x0,
        x1 * ne_crit,
        0.3,
        2,
        1.0,
    )
    electron_temperature_ev = ModelAnalyticConstant(
        CoordinateSystem.CARTESIAN, 100.0, 1.0
    )
    effective_charge = ModelAnalyticConstant(
        CoordinateSystem.CARTESIAN, 1.0, 1.0
    )
    magnetic_field_t = ModelAnalyticRamp(
        CoordinateSystem.CARTESIAN,
        [0.8, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, y0 * b_crit, 0.0],
        [0.0, y1 * b_crit, 0.0],
        0.2,
        2,
        1.0,
    )
    limiters = {
        "bounding_box": LimiterAnalyticBoundingBox2D(
            LimiterEffect.STOP,
            LimiterAnalyticBoundingBox2D.XY,
            (0.0, 1.0),
            (-0.1, 1.0),
            0.0,
        )
    }

    system_data_provider = SystemDataProvider(
        data_sources,
        coordinates,
        electron_density_per_m3,
        electron_temperature_ev,
        effective_charge,
        magnetic_field_t,
        limiters,
    )

    with open(input_directory.joinpath(SYSTEM_DATA_TOML), "w") as fh:
        system_data_provider.write_toml(fh)

    # Write rays initial conditions.
    initial_conditions = [
        InitialConditionsSchema(
            "ray_1",
            0.0,
            frequency_ghz,
            [0.1, 0.0, 0.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexNparallel(0.655, 0.0, radians=True),
            PolarisationWaveMode(WaveMode.O),
            1.0,
            0.05,
            1,
        )
    ]

    with open(input_directory.joinpath(RAYS_TOML), "w") as fh:
        write_initial_conditions_toml(fh, initial_conditions)
