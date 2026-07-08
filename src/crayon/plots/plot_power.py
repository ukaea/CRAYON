"""
Standard plots for damped power along ray.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import matplotlib.pyplot as plt
import netCDF4 as nc4  # noqa: N813

# Local imports
from crayon.ray_tracing.ray_tracer import RayTracingOutput
from crayon.shared.constants import CoordinateSystem

logger = logging.getLogger(__name__)


def plot_power(
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot breakdown of power flow along ray.

    Parameters
    ----------
    output_file : pathlib.Path
        Path to file containing ray tracing output.
    ray_name : str
        Name of ray to plot.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    fig, axes = plt.subplots(ncols=2, figsize=(10, 8), sharex=True)

    for ax in axes.flatten():
        ax.grid()

    ax_power = axes[0]
    ax_power.set_ylabel("Power [W]")

    ax_cum = axes[1]
    ax_cum.set_ylabel("Cum. Damped Power [W]")

    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    t = trace.time_ns

    # Plot power.
    ax_power.plot(t, trace.power_w, color="black")
    ax_power.set_ylim(bottom=-0.01)

    # Plot cumulative damped power.
    ax_cum.plot(
        t, trace.cumulative_damped_power_w, color="black", label="Total"
    )
    ax_cum.plot(
        t,
        trace.cumulative_damped_power_resonance_w,
        color="red",
        ls="--",
        label="Resonance",
    )
    ax_cum.plot(
        t,
        trace.cumulative_damped_power_collisional_w,
        color="blue",
        ls=":",
        label="Collisional",
    )
    ax_cum.plot(
        t,
        trace.cumulative_damped_power_external_w,
        color="green",
        ls="-.",
        label="External",
    )
    ax_cum.legend(loc="upper left")

    fig.supxlabel("Time [ns]")
    fig.tight_layout()

    fig.subplots_adjust(top=0.95)
    fig.suptitle(ray_name)

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()


def plot_optical_depth(
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot optical depth and fraction of damped power in each channel along ray.

    Parameters
    ----------
    output_file : pathlib.Path
        Path to file containing ray tracing output.
    ray_name : str
        Name of ray to plot.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    fig, axes = plt.subplots(ncols=2, figsize=(10, 8), sharex=True)

    axes[0].set_ylabel("Optical Depth []")
    axes[0].grid()

    axes[1].set_ylabel("Damping Fraction []")
    axes[1].grid()

    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    t = trace.time_ns

    axes[0].plot(t, trace.optical_depth, color="black", label="Total")
    axes[0].plot(
        t, trace.optical_depth_internal, color="red", label="Internal", ls="--"
    )
    axes[0].plot(
        t,
        trace.optical_depth_external,
        color="green",
        label="External",
        ls=":",
    )
    axes[0].legend(loc="upper left")

    axes[1].plot(
        t, trace.damping_fraction_resonance, color="red", label="Resonance"
    )
    axes[1].plot(
        t,
        trace.damping_fraction_collisional,
        color="blue",
        label="Collisions",
        ls="--",
    )
    axes[1].plot(
        t,
        trace.damping_fraction_external,
        color="green",
        label="External",
        ls="--",
    )

    axes[1].legend(loc="upper left")

    fig.supxlabel("Time [ns]")
    fig.tight_layout()

    fig.subplots_adjust(top=0.95)
    fig.suptitle(ray_name)

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()


def plot_damping_vs_rho_poloidal(
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot damped power vs root normalised poloidal flux.

    Parameters
    ----------
    output_file : pathlib.Path
        Path to file containing ray tracing output.
    ray_name : str
        Name of ray to plot.
    show : bool, optional
        If True, show the plot in an interactive window.
    save_name : str, optional
        If provided, save the plot using this name.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\rho_p$ []")
    ax.set_ylabel("Power [W]")
    ax.grid()

    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    rho_poloidal = trace.position[CoordinateSystem.RHO_POLOIDAL][:, 0]

    ax.plot(rho_poloidal, trace.power_w)

    fig.tight_layout()

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()
