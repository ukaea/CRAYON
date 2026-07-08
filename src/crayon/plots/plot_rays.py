"""
Standard plots for ray trajectories in xy, xz, yz and rz coordinate planes.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import matplotlib.pyplot as plt
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.coordinates import CoordinateCoordinator, CoordinateSystem
from crayon.ray_tracing.ray_tracer import RayTracingOutput
from crayon.system_data import Limiters
from crayon.system_data.limiter import CappedCone, Cylinder, Disk, Plane

logger = logging.getLogger(__name__)

labels_cartesian = {0: "x [m]", 1: "y [m]", 2: "z [m]"}


def plot_rays_cartesian(
    input_data_file: pathlib.Path,
    ray_tracing_file: pathlib.Path,
    ix: int,
    iy: int,
    iz: int,
    /,
    *rays: tuple[str],
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot rays in 2d Cartesian plane.

    Parameters
    ----------
    input_data_file : pathlib.Path
        Path to file containing Crayon input data.
    ray_tracing_file : pathlib.Path
        Path to file containing ray tracing output.
    ix, iy : int
        Index of Cartesian coordinate component to show on each axis.
    iz : int
        Index of Cartesian coordinate component not being shown.
    *rays : list[str], optional
        Name of rays to plot. If not provided, all rays are shown.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(labels_cartesian[ix])
    ax.set_ylabel(labels_cartesian[iy])
    ax.set_aspect("equal")

    # Plot limiter if available.
    with nc4.Dataset(input_data_file, "r", auto_complex=True) as dset:
        limiters = Limiters.read_netcdf(dset["system_data/limiters"])

    xy = np.empty((2, 2))

    for limiter in limiters.limiters.values():
        for element in limiter.elements:
            # Check element is 2D Plane.
            if not (
                isinstance(element, Plane)
                and np.allclose(element.direction_2, 0.0)  # 2D plane.
            ):
                continue

            # Check plane matches projection being plotted.
            if np.isclose(element.direction_1[iz], 0.0):
                xy[0, :] = element.origin[ix], element.origin[iy]
                xy[1, 0] = xy[0, 0] + element.direction_1[ix]
                xy[1, 1] = xy[0, 1] + element.direction_1[iy]
                ax.plot(xy[:, 0], xy[:, 1], color="black")

    # Plot rays.
    with nc4.Dataset(ray_tracing_file, auto_complex=True) as dset:
        if len(rays) == 0:
            rays = dset.groups.keys()

        for ray_name in rays:
            if ray_name not in dset.groups:
                logger.warning("No group '%s'", ray_name)
                continue

            trace = RayTracingOutput.read_netcdf(dset[ray_name])
            _x = trace.position[CoordinateSystem.CARTESIAN]

            ax.plot(_x[:, ix], _x[:, iy], color="black")
            ax.scatter(_x[0, ix], _x[0, iy], color="red", marker="x")

    fig.tight_layout()

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()


def plot_rays_xy(
    input_data_file: pathlib.Path,
    ray_tracing_file: pathlib.Path,
    /,
    *rays: tuple[str],
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot rays in Cartesian x-y plane.

    Parameters
    ----------
    input_data_file : pathlib.Path
        Path to file containing Crayon input data.
    ray_tracing_file : pathlib.Path
        Path to file containing ray tracing output.
    *rays : list[str], optional
        Name of rays to plot. If not provided, all rays are shown.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    plot_rays_cartesian(
        input_data_file,
        ray_tracing_file,
        0,
        1,
        2,
        *rays,
        show=show,
        save_name=save_name,
    )


def plot_rays_xz(
    input_data_file: pathlib.Path,
    ray_tracing_file: pathlib.Path,
    /,
    *rays: tuple[str],
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot rays in Cartesian x-z plane.

    Parameters
    ----------
    input_data_file : pathlib.Path
        Path to file containing Crayon input data.
    ray_tracing_file : pathlib.Path
        Path to file containing ray tracing output.
    *rays : list[str], optional
        Name of rays to plot. If not provided, all rays are shown.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    plot_rays_cartesian(
        input_data_file,
        ray_tracing_file,
        0,
        2,
        1,
        *rays,
        show=show,
        save_name=save_name,
    )


def plot_rays_yz(
    input_data_file: pathlib.Path,
    ray_tracing_file: pathlib.Path,
    /,
    *rays: tuple[str],
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot rays in Cartesian y-z plane.

    Parameters
    ----------
    input_data_file : pathlib.Path
        Path to file containing Crayon input data.
    ray_tracing_file : pathlib.Path
        Path to file containing ray tracing output.
    *rays : list[str], optional
        Name of rays to plot. If not provided, all rays are shown.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    plot_rays_cartesian(
        input_data_file,
        ray_tracing_file,
        1,
        2,
        0,
        *rays,
        show=show,
        save_name=save_name,
    )


def plot_rays_rz(
    input_data_file: pathlib.Path,
    ray_tracing_file: pathlib.Path,
    /,
    *rays: tuple[str],
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot rays in cylindrical r-z plane.

    Parameters
    ----------
    input_data_file : pathlib.Path
        Path to file containing Crayon input data.
    ray_tracing_file : pathlib.Path
        Path to file containing ray tracing output.
    *rays : list[str], optional
        Name of rays to plot. If not provided, all rays are shown.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")

    # Load coordinates and limiters.
    with nc4.Dataset(input_data_file, "r", auto_complex=True) as dset:
        coordinate_coordinator = CoordinateCoordinator.read_netcdf(
            dset["system_data/coordinate_coordinator"]
        )
        limiters = Limiters.read_netcdf(dset["system_data/limiters"])

    # If 2d flux coordinate system available plot isocontours.
    if CoordinateSystem.RHO_POLOIDAL in coordinate_coordinator.coordinates:
        rho_poloidal = coordinate_coordinator.coordinates[
            CoordinateSystem.RHO_POLOIDAL
        ]

        # Plot rho poloidal isocontours.
        _rho = rho_poloidal.rho_1d[0]

        for i, rho in enumerate(rho_poloidal.rho_1d):
            if rho >= _rho:
                ax.plot(
                    rho_poloidal.isocontours_rz[i, :, 0],
                    rho_poloidal.isocontours_rz[i, :, 1],
                    color="black",
                    alpha=0.5,
                )

                _rho += 0.1

        # Plot limiter contour.
        ax.plot(
            rho_poloidal.isocontours_rz[-1, :, 0],
            rho_poloidal.isocontours_rz[-1, :, 1],
            color="black",
            alpha=0.5,
        )

    # If (r, z) limiter available plot elements.
    rz = np.empty((2, 2))

    for limiter in limiters.limiters.values():
        for element in limiter.elements:
            if isinstance(element, Disk):
                rz[0, :] = element.r_min, element.z
                rz[1, :] = element.r_max, element.z
            elif isinstance(element, Cylinder):
                rz[0, :] = element.r, element.z_min
                rz[1, :] = element.r, element.z_max
            elif isinstance(element, CappedCone):
                rz[0, :] = element.r_a, element.z_a
                rz[1, :] = element.r_b, element.z_b
            else:
                continue

            ax.plot(rz[:, 0], rz[:, 1], color="black")

    # Plot rays.
    with nc4.Dataset(ray_tracing_file, auto_complex=True) as dset:
        if len(rays) == 0:
            rays = dset.groups.keys()

        for ray_name in rays:
            if ray_name not in dset.groups:
                logger.warning("No group '%s'", ray_name)
                continue

            trace = RayTracingOutput.read_netcdf(dset[ray_name])
            _x = trace.position[CoordinateSystem.CYLINDRICAL]

            ax.plot(_x[:, 0], _x[:, 2], color="black")
            ax.scatter(_x[0, 0], _x[0, 2], color="red", marker="x")

    fig.tight_layout()

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()
