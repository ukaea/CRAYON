"""
Mirror EBW integration test.
"""

# Standard imports
import logging
import pathlib

import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.scripts import optimise, plot_all, plot_single, trace
from crayon.shared.constants import INPUT

logger = logging.getLogger(__name__)

run_directory = pathlib.Path(__file__).parent


def write_netcdf():
    """
    Write input data for test case to netCDF4 file.
    """
    # Magnetic field.
    r = np.linspace(0.0, 2.0, 95)
    z = np.linspace(-1.5, 1.5, 85)

    rmesh, zmesh = np.meshgrid(r, z, indexing="ij")

    br = np.zeros((r.size, z.size))
    bphi = (1 + 0.5 * rmesh) * np.sqrt(1 + zmesh * zmesh)
    bz = np.zeros((r.size, z.size))

    # Core profiles.
    decay = 0.5 * (1 - np.tanh(20 * (r - 1.5)))
    ne = 5.0e19 * decay
    te = 3.0e3 * decay
    zeff = (1.0 + 0.5 * r) * decay

    # Write to database.
    plasma_data_file = run_directory.joinpath(INPUT, "plasma_data.nc")

    with nc4.Dataset(plasma_data_file, "w") as dset:
        # Create dimensions.
        dset.createDimension("x", 3)
        dset.createDimension("time", 1)
        dset.createDimension("r", r.size)
        dset.createDimension("z", z.size)

        # Write abscissas.
        v = dset.createVariable("time_s", "f8", "time")
        v[:] = 0.0

        v = dset.createVariable("r", "f8", "r")
        v[:] = r

        v = dset.createVariable("z", "f8", "z")
        v[:] = z

        # Write electron density.
        g = dset.createGroup("electron_density_per_m3")
        g.setncattr("coordinate_system", "cylindrical")

        v = g.createVariable("dependent_components", "u4", "x")
        v[:] = 1, 0, 0

        v = g.createVariable("data", "f8", ("time", "r"))
        v[0, :] = ne

        # Write electron temperature.
        g = dset.createGroup("electron_temperature_ev")
        g.setncattr("coordinate_system", "cylindrical")

        v = g.createVariable("dependent_components", "u4", "x")
        v[:] = 1, 0, 0

        v = g.createVariable("data", "f8", ("time", "r"))
        v[0, :] = te

        # Write effective charge.
        g = dset.createGroup("effective_charge")
        g.setncattr("coordinate_system", "cylindrical")

        v = g.createVariable("dependent_components", "u4", "x")
        v[:] = 1, 0, 0

        v = g.createVariable("data", "f8", ("time", "r"))
        v[0, :] = zeff

        # Write magnetic field.
        g = dset.createGroup("magnetic_field_t")
        g.setncattr("coordinate_system", "cylindrical")

        v = g.createVariable("dependent_components", "u4", "x")
        v[:] = 1, 0, 1

        v = g.createVariable("data", "f8", ("time", "r", "z", "x"))
        v[0, :, :, 0] = br
        v[0, :, :, 1] = bphi
        v[0, :, :, 2] = bz


def main():
    """
    Run mirror EBW test case.
    """
    write_netcdf()

    optimise(run_directory, [0.0], overwrite=True)

    trace(run_directory, [0.0], overwrite=True)

    plot_all(
        run_directory,
        [0.0],
        show=True,
        rz=True,
    )

    plot_single(
        run_directory,
        "ray_npar_gt_0-0-0",
        [0.0],
        show=True,
        rz=True,
        plasma_parameters=True,
        hamiltonian=True,
        power=True,
        optical_depth=True,
        mode_conversion=True,
    )

    plot_single(
        run_directory,
        "ray_npar_lt_0-0",
        [0.0],
        show=True,
        rz=True,
        plasma_parameters=True,
        hamiltonian=True,
        power=True,
        optical_depth=True,
        mode_conversion=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S",
    )
    main()
