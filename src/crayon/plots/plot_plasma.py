"""
Standard plots for plasma parameters along ray.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.ray_tracing.initial_conditions import InitialConditions
from crayon.ray_tracing.ray import Ray
from crayon.ray_tracing.ray_tracer import RayTracingOutput
from crayon.shared.dimensions import Dimensions
from crayon.system_data import SystemData

logger = logging.getLogger(__name__)


def plot_plasma_parameters(
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot normalised plasma parameters along ray:
        Normalised electron density aka Stix X
        Normalised magnetic field strength aka Stix Y
        Normalised electron-ion collision rate aka Stix Z.
        Normalised electron temperature aka theta
        Perpendicular refractive index component n_perp
        Parallel refractive index component n_parallel

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
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 8), sharex=True)

    for ax in axes.flatten():
        ax.grid()

    ax_x = axes[0, 0]
    ax_x.set_ylabel("Normalised Density []")

    ax_y = axes[0, 1]
    ax_y.set_ylabel("Normalised Magnetic Field Strength []")

    ax_z = axes[0, 2]
    ax_z.set_ylabel("Normalised Electron-Ion Collision Rate []")

    ax_theta = axes[1, 0]
    ax_theta.set_ylabel("Normalised Temperature []")

    ax_n_perp = axes[1, 1]
    ax_n_perp.set_ylabel("Perpendicular Refractive Index []")

    ax_n_par = axes[1, 2]
    ax_n_par.set_ylabel("Parallel Refractive Index []")

    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    t = trace.time_ns

    ax_x.plot(t, trace.normalised_electron_density, color="black")
    ax_y.plot(t, trace.normalised_magnetic_field_strength, color="black")
    ax_z.plot(t, trace.normalised_collision_rate, color="black")
    ax_theta.plot(t, trace.normalised_electron_temperature, color="black")
    ax_n_perp.plot(t, trace.n_perp, color="black")
    ax_n_par.plot(t, trace.n_parallel, color="black")

    fig.supxlabel("Time [ns]")
    fig.tight_layout()

    fig.subplots_adjust(top=0.95)
    fig.suptitle(ray_name)

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()


def plot_hamiltonian(
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot hamiltonian along ray (a metric for error in the integration).

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

    for ax in axes:
        ax.grid()
        ax.set_yscale("log")

    ax_hreal = axes[0]
    ax_hreal.set_ylabel("|Re[H]| []")

    ax_freal = ax_hreal.twinx()
    ax_freal.set_ylabel("|H / dH/df| [GHz]")

    ax_himag = axes[1]
    ax_himag.set_ylabel("|Im[H]| []")

    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    t = trace.time_ns

    hamiltonian = trace.eigenvalue
    ax_hreal.plot(t, abs(hamiltonian.real), color="black")
    ax_himag.plot(t, abs(hamiltonian.imag), color="black")

    determinant = trace.determinant
    ax_hreal.plot(t, abs(determinant.real), color="red", ls="--")
    ax_himag.plot(t, abs(determinant.imag), color="red", ls="--")

    ax_freal.plot(
        t, abs(trace.eigenvalue_error_frequency), color="black", ls=":"
    )
    ax_freal.plot(
        t, abs(trace.determinant_error_frequency), color="red", ls=":"
    )

    fig.supxlabel("Time [ns]")
    fig.tight_layout()

    fig.subplots_adjust(top=0.92)
    fig.suptitle(ray_name)

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()


def plot_mode_conversion(
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: str | None = None,
):
    """
    Plot parameters related to mode conversion along ray.

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
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), sharex=True)

    for ax in axes.flatten():
        ax.grid()

    ax_x = axes[0, 0]
    ax_x.set_ylabel("Normalised Density []")

    ax_alarm = axes[0, 1]
    ax_alarm.set_ylabel("Mode Conversion Alarm []")

    ax_wkb = axes[1, 0]
    ax_wkb.set_ylabel("WKB Validity []")

    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    t = trace.time_ns

    ax_x.plot(t, trace.normalised_electron_density, color="black")
    ax_alarm.plot(t, trace.mode_conversion_alarm, color="black")
    ax_wkb.plot(t, trace.wkb_validity, color="black")

    fig.supxlabel("Time [ns]")
    fig.tight_layout()

    fig.subplots_adjust(top=0.92)
    fig.suptitle(ray_name)

    if save_name is not None:
        fig.savefig(save_name)

    if show:
        plt.show()


def plot_osculating_plane(
    input_data_file: pathlib.Path,
    output_file: pathlib.Path,
    ray_name: str,
    /,
    *,
    show: bool = False,
    save_name: pathlib.Path | None = None,
):
    """
    Plot dispersion curves in osculating plane at mode conversion.

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
    # Load system data.
    with nc4.Dataset(input_data_file, "r", auto_complex=True) as dset:
        system_data = SystemData.read_netcdf(dset["system_data"])
        initial_conditions = InitialConditions.read_netcdf(
            dset[f"initial_conditions/{ray_name}"]
        )

    # Load ray tracing output.
    with nc4.Dataset(output_file, auto_complex=True) as dset:
        trace = RayTracingOutput.read_netcdf(dset[ray_name])

    mc_size = trace.k0ln_saddle.shape[0]

    if mc_size == 0:
        logger.warning("No mode conversion for %s", ray_name)
        return

    # Calculate Hamiltonian in osculating plane.
    n_p, n_q = 51, 51
    pp = 0.1 * np.linspace(-1, 1, n_p)
    qq = 0.1 * np.linspace(-1, 1, n_q)

    hh = np.empty((n_p, n_q), dtype=float)

    ray = Ray(system_data, initial_conditions)

    for i in range(mc_size):
        fig, ax = plt.subplots()
        ax.set_xlabel("p []")
        ax.set_ylabel("q []")
        ax.set_aspect("equal")

        xk_closest = trace.closest_to_conversion[i, :]
        xk_saddle = trace.saddle_at_conversion[i, :]
        xk = xk_closest.copy()

        osculating_plane_basis = trace.osculating_plane_basis[i]
        e_p = osculating_plane_basis[0]
        e_q = osculating_plane_basis[1]

        for j, p in enumerate(pp):
            for k, q in enumerate(qq):
                xk[:] = xk_closest + p * e_p + q * e_q

                ray.set_xk_position(
                    xk[Dimensions.slice_x],
                    xk[Dimensions.slice_k],
                )
                ray.calculate_hamiltonian(derivatives=0, determinant=True)

                hh[j, k] = ray.hamiltonian_cache.determinant.real

        cs = ax.pcolormesh(pp, qq, abs(hh.T), norm=mcolors.LogNorm())
        fig.colorbar(cs, label="Hamiltonian")
        ax.contour(pp, qq, hh.T, levels=[0], colors="black")

        pq_saddle = np.linalg.lstsq(
            osculating_plane_basis.T, xk_saddle - xk_closest, rcond=None
        )[0]

        ax.scatter(*pq_saddle, marker="x", color="black")
        ax.scatter(0.0, 0.0, marker="o", color="black")
        fig.tight_layout()

        if save_name is not None:
            savename = save_name.parent.joinpath(
                f"{save_name.stem}_{i + 1}{save_name.suffix}"
            )
            fig.savefig(savename)

    if show:
        plt.show()
