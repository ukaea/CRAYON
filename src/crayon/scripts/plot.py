"""
Wrapper scripts for standard plots of output data.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import matplotlib.pyplot as plt

# Local imports
from crayon.plots import (
    plot_damping_vs_rho_poloidal,
    plot_hamiltonian,
    plot_mode_conversion,
    plot_optical_depth,
    plot_osculating_plane,
    plot_plasma_parameters,
    plot_power,
    plot_rays_rz,
    plot_rays_xy,
    plot_rays_xz,
    plot_rays_yz,
)
from crayon.shared.constants import (
    INPUT_DATA_NETCDF,
    OUTPUT,
    PLOTS,
    RAYS_NETCDF,
)

logger = logging.getLogger(__name__)


def plot_single(
    run_directory: pathlib.Path,
    ray_name: str,
    times: list[float],
    /,
    *,
    show: bool = True,
    xy: bool = False,
    xz: bool = False,
    yz: bool = False,
    rz: bool = False,
    plasma_parameters: bool = False,
    hamiltonian: bool = False,
    mode_conversion: bool = False,
    power: bool = False,
    optical_depth: bool = False,
    damping: bool = False,
    osculating_plane: bool = False,
):
    """
    Wrapper for parameter plots for a single ray.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    ray_name : str
        Name of ray to plot.
    times : list[float]
        Times to plot at.
    show : bool, optional
        If True, show plots in interactive window. Default = True.
    xy : bool, optional
        If True, show x-y projection of ray trajectory.
    xz : bool, optional
        If True, show y-z projection of ray trajectory.
    yz : bool, optional
        If True, show x-z projection of ray trajectory.
    rz : bool, optional
        If True, show r-z projection of ray trajectory.
    plasma_parameters : bool, optional
        If True, show normalised plasma parameters along ray.
    hamiltonian : bool, optional
        If True, show Hamiltonian parameters along ray.
    mode_conversion : bool, optional
        If True, show mode conversion parameters along ray.
    power : bool, optional
        If True, show power along ray.
    optical_depth : bool, optional
        If True, show optical depth along ray.
    damping : bool, optional
        If True, show power in ray vs root normalised poloidal flux. Should
        be removed once linear current drive model ready!
    """
    plots_master = run_directory.joinpath(PLOTS)
    plots_master.mkdir(exist_ok=True, parents=True)

    for time_s in times:
        plots_dir = plots_master.joinpath(f"{time_s}s")
        plots_dir.mkdir(exist_ok=True)

        input_data_file = run_directory.joinpath(
            OUTPUT, f"{time_s}s", INPUT_DATA_NETCDF
        )
        output_file = run_directory.joinpath(OUTPUT, f"{time_s}s", RAYS_NETCDF)

        if xy:
            plot_rays_xy(
                input_data_file,
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_xy.png"),
            )

        if xz:
            plot_rays_xz(
                input_data_file,
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_xz.png"),
            )

        if yz:
            plot_rays_yz(
                input_data_file,
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_yz.png"),
            )

        if rz:
            plot_rays_rz(
                input_data_file,
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_rz.png"),
            )

        if plasma_parameters:
            plot_plasma_parameters(
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(
                    f"{ray_name}_plasma-parameters.png"
                ),
            )

        if hamiltonian:
            plot_hamiltonian(
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_hamiltonian.png"),
            )

        if mode_conversion:
            plot_mode_conversion(
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(
                    f"{ray_name}_mode-conversion.png"
                ),
            )

        if power:
            plot_power(
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_power.png"),
            )

        if optical_depth:
            plot_optical_depth(
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_optical-depth.png"),
            )

        if damping:
            plot_damping_vs_rho_poloidal(
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(f"{ray_name}_damping.png"),
            )

        if osculating_plane:
            plot_osculating_plane(
                input_data_file,
                output_file,
                ray_name,
                show=False,
                save_name=plots_dir.joinpath(
                    f"{ray_name}_osculating-plane.png"
                ),
            )

        if show:
            plt.show()

        plt.close("all")


def plot_all(
    run_directory: pathlib.Path,
    times: list[float],
    /,
    *,
    show: bool = True,
    xy: bool = False,
    xz: bool = False,
    yz: bool = False,
    rz: bool = False,
):
    """
    Wrapper for parameter plots for all rays.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    times : list[float]
        Times to plot at.
    show : bool, optional
        If True, show plots in interactive window. Default = True.
    xy : bool, optional
        If True, show x-y projection of ray trajectories.
    xz : bool, optional
        If True, show y-z projection of ray trajectories.
    yz : bool, optional
        If True, show x-z projection of ray trajectories.
    rz : bool, optional
        If True, show r-z projection of ray trajectories.
    """
    plots_master = run_directory.joinpath(PLOTS)
    plots_master.mkdir(exist_ok=True, parents=True)

    for time_s in times:
        plots_dir = plots_master.joinpath(f"{time_s}s")
        plots_dir.mkdir(exist_ok=True)

        input_data_file = run_directory.joinpath(
            OUTPUT, f"{time_s}s", INPUT_DATA_NETCDF
        )
        output_file = run_directory.joinpath(OUTPUT, f"{time_s}s", RAYS_NETCDF)

        if xy:
            plot_rays_xy(
                input_data_file,
                output_file,
                show=False,
                save_name=plots_dir.joinpath("rays_xy.png"),
            )

        if xz:
            plot_rays_xz(
                input_data_file,
                output_file,
                show=False,
                save_name=plots_dir.joinpath("rays_xz.png"),
            )

        if yz:
            plot_rays_yz(
                input_data_file,
                output_file,
                show=False,
                save_name=plots_dir.joinpath("rays_yz.png"),
            )

        if rz:
            plot_rays_rz(
                input_data_file,
                output_file,
                show=False,
                save_name=plots_dir.joinpath("rays_rz.png"),
            )

        if show:
            plt.show()

        plt.close("all")
