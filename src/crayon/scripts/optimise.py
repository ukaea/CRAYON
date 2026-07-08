"""
Wrapper scripts for OX angle optimisation.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import numpy as np
import toml
from scipy import optimize

# Local imports
from crayon.calculus import TensorType, rotation_a_onto_b
from crayon.ray_tracing.caches import CoordinateCache, PlasmaCache
from crayon.ray_tracing.initial_conditions import (
    InitialConditions,
    RefractiveIndexComponents,
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
    OPTIMAL_OX_TOML,
    OPTIONS_TOML,
    OUTPUT,
    RAD_TO_DEG,
    RAYS_TOML,
    SYSTEM_DATA_TOML,
    CoordinateSystem,
    DispersionType,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.io import IOToml
from crayon.shared.types import FloatArray
from crayon.system_data import SystemData, SystemDataProvider

logger = logging.getLogger(__name__)


def spherical_unit_vector(theta: float, phi: float, unit_vector: FloatArray):
    """
    Construct Cartesian components of a unit vector defined using spherical
    angles.

    Attributes
    ----------
    theta : float
        Poloidal angle.
    phi : float
        Azimuthal angle.
    unit_vector : np.array[float]
        Array to store result in.
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    unit_vector[0] = sin_theta * cos_phi
    unit_vector[1] = sin_theta * sin_phi
    unit_vector[2] = cos_theta


def find_cutoff_along_los(
    coordinate_cache: CoordinateCache,
    plasma_cache: PlasmaCache,
    initial_conditions: InitialConditions,
    theta_range: tuple[float, float],
    n_theta: int,
    phi_range: tuple[float, float],
    n_phi: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Find 2d angular range over which the line of sight intersects the O mode
    cutoff using brute force.

    Parameters
    ----------
    coordinate_cache : CoordinateCache
        Cache containing coordinate system data.
    plasma_cache : PlasmaCache
        Cache containing plasma parameter data.
    initial_conditions : InitialConditions
        Initial conditions of ray.
    theta_range: tuple[float, float]
        Poloidal angle range to search.
    n_theta: int
        Number of poloidal angles to consider.
    phi_range: tuple[float, float],
        Azimuthal angle range to search.
    n_phi: int
        Number of azimuthal angles to consider.

    Returns
    -------
    theta_range : tuple[float, float]
        Min and max poloidal angle.
    phi_range : tuple[float, float]
        Min and max azimuthal angle.
    """
    direction = np.zeros(Dimensions.x.size)
    position_last = np.empty(Dimensions.x.size)
    position = np.empty(Dimensions.x.size)
    max_density = np.empty((n_theta, n_phi))
    max_density[:, :] = -1

    theta_samples = np.linspace(*theta_range, n_theta)
    phi_samples = np.linspace(*phi_range, n_phi)

    # Up to 100 x 10 cm steps = 10m max search.
    ds = 0.1
    max_steps = 100

    for i, theta in enumerate(theta_samples):
        for j, phi in enumerate(phi_samples):
            # Evaluate search direction.
            spherical_unit_vector(theta, phi, direction)

            # Reset last position to ray start.
            position_last[:] = initial_conditions.position_cartesian

            if np.isclose(phi, phi_samples[0] + 2.0 * np.pi):
                max_density[i, j] = max_density[i, 0]
                break

            for _ in range(1, max_steps):
                position[:] = position_last + ds * direction
                position_last[:] = position

                # Evaluate plasma density at new position.
                coordinate_cache.set_position(
                    CoordinateSystem.CARTESIAN,
                    position,
                    calculate_transforms=False,
                )
                plasma_cache.calculate_electron_density(
                    coordinate_cache, derivatives=0
                )

                max_density[i, j] = max(
                    max_density[i, j],
                    plasma_cache.normalised_electron_density.value.item(),
                )

            # If theta = 0 or pi direction is independent of phi.
            if np.isclose(theta, 0.0) or np.isclose(theta, np.pi):
                max_density[i, :] = max_density[i, j]
                break

    # Calculate location of peak density.
    i_max, j_max = np.unravel_index(np.argmax(max_density), max_density.shape)
    theta_max, phi_max = theta_samples[i_max], phi_samples[j_max]

    # Find box enclosing all points with density > cutoff relative to peak.
    dtheta_range = np.zeros(2)
    dphi_range = np.zeros(2)

    for i, theta in enumerate(theta_samples):
        for j, phi in enumerate(phi_samples):
            dtheta = theta - theta_max
            dphi = phi - phi_max

            if dtheta > 0.5 * np.pi:
                dphi -= np.pi

            if dtheta < -0.5 * np.pi:
                dphi += np.pi

            if dphi > np.pi:
                dphi -= 2.0 * np.pi

            if dphi < -np.pi:
                dphi += 2.0 * np.pi

            if max_density[i, j] > 1.0:
                dtheta_range[0] = min(dtheta, dtheta_range[0])
                dtheta_range[1] = max(dtheta, dtheta_range[1])
                dphi_range[0] = min(dphi, dphi_range[0])
                dphi_range[1] = max(dphi, dphi_range[1])

    # Extend box slightly to include points in between samples.
    dtheta = 0.5 * (theta_samples[1] - theta_samples[0])
    dtheta_range[0] -= dtheta
    dtheta_range[1] += dtheta

    dphi = 0.5 * (phi_samples[1] - phi_samples[0])
    dphi_range[0] -= dphi
    dphi_range[1] += dphi

    return theta_max + dtheta_range, phi_max + dphi_range


def ox_conversion_proximity(
    angles: tuple[float, float],
    initial_conditions: InitialConditions,
    ray_tracer: RayTracer,
    n_parallel_sign: float,
) -> float:
    """
    Objective function for proximity to the OX mode conversion in phase space.

    Parameters
    ----------
    angles : tuple[float, float]
        Poloidal and toroidla launch angle.
    initial_conditions : InitialConditions
        Initial conditions of ray.
    ray_tracer : RayTracer
        Ray tracer.
    n_parallel_sign : float
        Sign of parallel refractive index (1.0 or -1.0) at targeted mode
        conversion window.

    Returns
    -------
    ox_conversion_proximity : float
        Proximity to the OX mode conversion in phase space.
    """
    # Construct initial conditions with new refractive index.
    _initial_conditions = initial_conditions.clone()
    spherical_unit_vector(
        angles[0], angles[1], _initial_conditions.refractive_index_cartesian
    )

    # Create ray.
    ray_tracer.create_ray(_initial_conditions)

    # Run with minimal features.
    ray_tracer.ray.force_propagation_model = DispersionType.COLD
    ray_tracer.ray.force_damping_model = DispersionType.COLD

    # Trace ray.
    ray_tracer.trace(_initial_conditions, create_ray=False)

    idx = np.argmax(ray_tracer.output.normalised_electron_density)
    ray = ray_tracer.ray
    ray.set_xk_position(
        ray_tracer.output.position[CoordinateSystem.CARTESIAN][idx, :],
        ray_tracer.output.wavevector_per_m[idx, :],
    )
    ray.plasma_cache.calculate_electron_density(
        ray_tracer.ray.coordinate_cache, derivatives=1
    )
    ray.plasma_cache.calculate_magnetic_field(
        ray_tracer.ray.coordinate_cache, derivatives=0
    )

    b_hat = ray.plasma_cache.magnetic_field_unit.value
    dn_dx = ray.plasma_cache.normalised_electron_density.first_derivative[
        Dimensions.slice_x
    ]

    y_direction = np.cross(b_hat, dn_dx)
    _norm = np.linalg.norm(y_direction)

    if _norm == 0:
        ny = ray_tracer.output.n_perp[idx]
    else:
        y_direction /= _norm
        ny = np.dot(ray_tracer.output.refractive_index[idx, :], y_direction)

    n_parallel = ray_tracer.output.n_parallel[idx]
    y = abs(ray_tracer.output.normalised_magnetic_field_strength[idx])
    n_parallel_opt = np.sqrt(y / (1 + y))

    return (
        np.square(n_parallel - n_parallel_sign * n_parallel_opt)
        + np.square(ny)
        + np.square(1 - ray_tracer.output.normalised_electron_density[idx])
    )


def optimise_ox_conversion(
    system_data: SystemData,
    options_ray_tracing: OptionsRayTracing,
    options_integrator: OptionsIntegrator,
    initial_conditions: InitialConditions,
) -> tuple[
    FloatArray, tuple[float, float], tuple[float, float], tuple[float, float]
]:
    """
    Calculate optimum refractive index and polarisation for OX mode conversion
    for both windows.

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

    Returns
    -------
    optimum_refractive_index : np.array[float]
        Optimum Cartesian refractive index vector.
        optimum_refractive_index[0] is for positive n_parallel window,
        optimum_refractive_index[1] is for negative n_parallel window.
    optimum_launch_angles_geometric : np.array[float]
        Optimum launch angles (geometric definition).
        optimum_launch_angles_geometric[0] is for positive n_parallel window,
        optimum_launch_angles_geometric[1] is for negative n_parallel window.
    optimum_launch_angles_imas : np.array[float]
        Optimum launch angles (IMAS definition).
        optimum_launch_angles_imas[0] is for positive n_parallel window,
        optimum_launch_angles_imas[1] is for negative n_parallel window.
    optimum_polarisation_angles : np.array[float]
        Optimum polarisation ellipse angles.
        optimum_polarisation_angles[0] is for positive n_parallel window,
        optimum_polarisation_angles[1] is for negative n_parallel window.

    Raises
    ------
    ValueError
        Fails to find OX conversion.
    """
    coordinate_cache = CoordinateCache(system_data.coordinate_coordinator)
    plasma_cache = PlasmaCache(system_data.kinetic, system_data.magnetic)
    plasma_cache.set_frequency(initial_conditions.frequency_ghz)

    # Find line of sight directions which intersect with plasma cutoff.
    theta_range, phi_range = find_cutoff_along_los(
        coordinate_cache,
        plasma_cache,
        initial_conditions,
        (0.0, np.pi),
        6,
        (-np.pi, np.pi),
        12,
    )

    logger.info(
        "Search theta in [%s, %s]",
        np.round(theta_range[0], 3),
        np.round(theta_range[1], 3),
    )
    logger.info(
        "Search phi in [%s, %s]",
        np.round(phi_range[0], 3),
        np.round(phi_range[1], 3),
    )

    # Trace rays to find optimum conversion.
    optimum_refractive_index = np.zeros((2, 3))
    optimum_launch_angles_geometric = np.zeros((2, 2))
    optimum_launch_angles_imas = np.zeros((2, 2))
    optimum_polarisation_angles = np.zeros((2, 2))

    ray_tracer = RayTracer(
        system_data, options_ray_tracing, options_integrator
    )

    # Run with minimal features.
    ray_tracer.enable_mode_conversion = False
    ray_tracer.enable_tunnelling = False

    # Estimate initial angles.
    coordinate_cache.set_position(
        CoordinateSystem.CARTESIAN,
        initial_conditions.position_cartesian,
        calculate_transforms=False,
    )
    plasma_cache.calculate_magnetic_field(coordinate_cache, derivatives=0)

    for i, sign in enumerate((1.0, -1.0)):
        result = optimize.direct(
            ox_conversion_proximity,
            (theta_range, phi_range),
            args=(initial_conditions, ray_tracer, sign),
            eps=0.01,
            maxiter=100,
            f_min=0.0,
            f_min_rtol=0.002,
            len_tol=0.001,
        )

        if not result.success:
            raise ValueError(result.message)

        # Run ray at optimum angle.
        ox_conversion_proximity(result.x, initial_conditions, ray_tracer, sign)

        # Set optimum Cartesian refractive index components.
        optimum_refractive_index[i, :] = ray_tracer.output.refractive_index[
            0, :
        ]

        # Get refractive index in Cylindrical in physical basis.
        coordinate_cache.set_position(
            CoordinateSystem.CARTESIAN,
            ray_tracer.output.position[CoordinateSystem.CARTESIAN][0, :],
        )

        refractive_index_cylindrical = coordinate_cache.transform_tensor_field(
            CoordinateSystem.CARTESIAN,
            CoordinateSystem.CYLINDRICAL,
            ray_tracer.output.refractive_index[0, :],
            TensorType.COVECTOR,
        )

        refractive_index_anholonomic = coordinate_cache.transform_basis(
            CoordinateSystem.CYLINDRICAL,
            refractive_index_cylindrical,
            TensorType.COVECTOR,
            to_holonomic=False,
        )

        # Calculate launch angles.
        optimum_launch_angles_geometric[i, 0] = np.arctan2(
            refractive_index_anholonomic[1], -refractive_index_anholonomic[0]
        )
        optimum_launch_angles_geometric[i, 1] = np.arctan2(
            refractive_index_anholonomic[2],
            np.sqrt(
                np.square(refractive_index_anholonomic[0])
                + np.square(refractive_index_anholonomic[1])
            ),
        )

        optimum_launch_angles_imas[i, 0] = np.arctan2(
            refractive_index_anholonomic[2], refractive_index_anholonomic[0]
        )
        optimum_launch_angles_imas[i, 1] = np.arctan2(
            refractive_index_anholonomic[1],
            np.sqrt(
                np.square(refractive_index_anholonomic[0])
                + np.square(refractive_index_anholonomic[2])
            ),
        )

        # Get optimum polarisation. Rotate Cartesian polarisation into frame so
        # refractive index // z axis.
        rot = rotation_a_onto_b(
            ray_tracer.output.refractive_index[0, :],
            np.array([0.0, 0.0, 1.0]),
        )
        polarisation = rot @ ray_tracer.output.polarisation_cartesian[0]
        optimum_polarisation_angles[i, 0] = np.arctan2(
            polarisation[1].real, polarisation[0].real
        )
        optimum_polarisation_angles[i, 1] = np.arctan2(
            polarisation[0].imag, polarisation[1].real
        )

    return (
        optimum_refractive_index,
        optimum_launch_angles_geometric,
        optimum_launch_angles_imas,
        optimum_polarisation_angles,
    )


def optimise(
    run_directory: pathlib.Path,
    times_s: list[float],
    /,
    *,
    overwrite: bool = False,
):
    """
    Wrapper script for calculating optimum launch conditions for OX conversion.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    times_s : list[float]
        Times to run.
    overwrite : bool, optional
        If True, overwrite existing output file. Default = False.

    Raises
    ------
    ValueError
        Output file already exists and overwrite = False.
    """
    # Load input files.
    with open(run_directory.joinpath(INPUT, OPTIONS_TOML)) as fh:
        options_ray_tracing, options_integrator = (
            read_options_ray_tracing_toml(fh)
        )

    # Ensure rays stop when intersecting limiter.
    options_ray_tracing.max_reflections = 0

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
        output_file = output_dir.joinpath(OPTIMAL_OX_TOML)

        if not overwrite and output_file.exists():
            raise ValueError(f"Output file already exists: {output_file}")

        # Unpack ray initial conditions.
        initial_conditions = []

        coordinate_cache = CoordinateCache(system_data.coordinate_coordinator)
        plasma_cache = PlasmaCache(system_data.kinetic, system_data.magnetic)

        for initial_condition in ray_initial_conditions:
            # Only launch central ray.
            initial_condition.n_radial_zones = 1

            # Replace refractive index condition with a dummy value.
            initial_condition.refractive_index = RefractiveIndexComponents(
                np.array([1.0, 0.0, 0.0]),
                CoordinateSystem.CARTESIAN,
                holonomic=True,
            )

            initial_conditions.extend(
                initial_condition.unpack(coordinate_cache, plasma_cache, [])
            )

        output = {}

        for initial_condition in initial_conditions:
            # Find optimum angles and polarisation for OX conversion for
            # both windows (n_parallel > 0 and n_parallel < 0).
            (
                optimum_refractive_index,
                optimum_launch_angles_geometric,
                optimum_launch_angles_imas,
                optimum_polarisation_angles,
            ) = optimise_ox_conversion(
                system_data,
                options_ray_tracing,
                options_integrator,
                initial_condition,
            )

            output[initial_condition.name] = {
                "refractive_index_cartesian": np.round(
                    optimum_refractive_index, decimals=3
                ),
                "launch_angles_geometric_deg": RAD_TO_DEG
                * np.round(optimum_launch_angles_geometric, decimals=1),
                "optimum_launch_angles_imas_deg": RAD_TO_DEG
                * np.round(optimum_launch_angles_imas, decimals=1),
                "polarisation_angles_deg": RAD_TO_DEG
                * np.round(optimum_polarisation_angles, decimals=1),
            }

        # Write optimal angles to file.
        logger.info("Writing %s", output_file)
        with open(output_file, "w") as fh:
            toml.dump(output, fh, encoder=IOToml.numpy_encoder)
