"""
Circular EBW integration test.
"""

# Standard imports
import logging
import pathlib

import imas
import numpy as np

# Local imports
from crayon.imas import imasdef
from crayon.scripts import optimise, plot_all, plot_single, trace
from crayon.shared.constants import INPUT

logger = logging.getLogger(__name__)

this_directory = pathlib.Path(__file__).parent
run_directory = this_directory.joinpath("ebw")


def psi(radius: np.ndarray[float]) -> np.ndarray[float]:
    """
    Poloidal flux function.

    Parameters
    ----------
    radius : np.array[float]
        Radius from magnetic axis.

    Returns
    -------
    psi : np.array[float]
        Poloidal flux array.

    Notes
    -----
    Looks like x^2 for small x but then flattens off to stop poloidal field
    diverging.
    """
    radius2 = np.square(radius)
    return radius2 / (1.0 + radius2)


def c1_ramp_down(x0: float, f0: float, f0p: float) -> np.polynomial.Polynomial:
    """
    Construct polynomial for C1 ramp from f0 at x0 to 1.0 such that the
    derivative at x0 = f0p and the derivative at 1.0 is zero.

    Parameters
    ----------
    x0 : float
        Reference position.
    f0 : float
        Value at x0.
    f0p : float
        First derivative at x0.

    Returns
    -------
    poly : np.polynomial.Polynomial
        Polynomial object.
    """
    m = np.array([
        [1.0, x0, x0**2, x0**3],
        [0.0, 1.0, 2 * x0, 3 * x0**2],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0, 3.0],
    ])

    x = np.array([f0, f0p, 0.0, 0.0])
    c = np.linalg.solve(m, x)

    return np.polynomial.Polynomial(c)


def create_profile(
    rho_poloidal: np.ndarray[float], rho_top: float, y0: float, y1: float
):
    """
    Create profile with a linear dependence in the core then a C1 ramp down
    to zero at rho = 1.

    Parameters
    ----------
    rho_poloidal : np.array[float]
        Grid of root normalised poloidal flux values to evaluate function.
    rho_top : float
        rho at top of ramp down
    y0 : float
        Value at rho = 0.0
    y1 : float
        Value at rho = rho_top

    Returns
    -------
    value : np.array[float]
        Profile values.
    """
    y1p = -(y0 - y1) / rho_top
    p = c1_ramp_down(rho_top, y1, y1p)

    value = np.empty_like(rho_poloidal)

    mask = rho_poloidal <= rho_top
    value[mask] = y0 + y1p * rho_poloidal[mask]
    value[~mask] = p(rho_poloidal[~mask])

    return value


def write_imas():
    """
    Write input data for test case to IMAS database.
    """
    # Equilibrium.
    ids_equilibrium = imas.equilibrium()
    ids_equilibrium.ids_properties.homogeneous_time = (
        imasdef.IDS_TIME_MODE_HOMOGENEOUS
    )

    ids_equilibrium.time.resize(1)
    ids_equilibrium.time = np.zeros(1)
    ids_equilibrium.time_slice.resize(1)

    r_maj, r_min = 2.0, 1.0
    r0, r1 = r_maj - 1.1 * r_min, r_maj + 1.1 * r_min
    z0, z1 = -1.1 * r_min, 1.1 * r_min

    r = np.linspace(r0, r1, 95)
    z = np.linspace(z0, z1, 85)
    _r, _z = np.meshgrid(r, z, indexing="ij")
    a = np.sqrt((_r - r_maj) ** 2 + _z**2)
    psi_norm_2d = psi(a) / psi(r_min)

    psi_axis = 0.0
    psi_sep = 0.01
    psi_2d = psi_sep * psi_norm_2d
    b0 = 1.0

    rho_poloidal = np.linspace(0, 1, 51)
    rho_toroidal = rho_poloidal
    psi_norm_1d = np.square(rho_poloidal)
    f_toroidal = np.full(rho_poloidal.shape, r_maj * b0)
    safety_factor = np.ones_like(rho_poloidal)

    ids_equilibrium.vacuum_toroidal_field.r0 = r_maj
    ids_equilibrium.vacuum_toroidal_field.b0 = b0 * np.ones(1)

    _profiles_1d = ids_equilibrium.time_slice[0].profiles_1d
    _profiles_1d.psi = psi_sep * psi_norm_1d
    _profiles_1d.f = f_toroidal
    _profiles_1d.psi_norm = psi_norm_1d
    _profiles_1d.rho_tor_norm = rho_toroidal
    _profiles_1d.q = safety_factor

    _profiles_2d = ids_equilibrium.time_slice[0].profiles_2d
    _profiles_2d.resize(1)
    _profiles_2d[0].grid_type.index = 1
    _profiles_2d[0].grid.dim1 = r
    _profiles_2d[0].grid.dim2 = z
    _profiles_2d[0].psi = psi_2d

    _globals = ids_equilibrium.time_slice[0].global_quantities
    _globals.psi_axis = psi_axis
    _globals.psi_boundary = psi_sep
    _globals.magnetic_axis.r = r_maj
    _globals.magnetic_axis.z = 0.0

    ids_equilibrium.validate()

    # Core profiles.
    ids_core_profiles = imas.core_profiles()
    ids_core_profiles.ids_properties.homogeneous_time = (
        imasdef.IDS_TIME_MODE_HOMOGENEOUS
    )
    ids_core_profiles.time.resize(1)
    ids_core_profiles.time = np.zeros(1)
    ids_core_profiles.profiles_1d.resize(1)

    rho_top = 0.8
    ne0, ne_top, te0, te_top = 5.0e19, 4.0e19, 3.0e3, 1.0e3

    rho_poloidal = np.linspace(0, 1, 101)
    rho_toroidal = rho_poloidal

    ne = create_profile(rho_poloidal, rho_top, ne0, ne_top)
    te = create_profile(rho_poloidal, rho_top, te0, te_top)

    _profiles_1d = ids_core_profiles.profiles_1d[0]
    _profiles_1d.grid.rho_pol_norm = rho_poloidal
    _profiles_1d.grid.rho_tor_norm = rho_toroidal
    _profiles_1d.electrons.density = ne
    _profiles_1d.electrons.temperature = te

    ids_core_profiles.validate()

    # Write to database.
    path = run_directory.joinpath(INPUT, "imas")

    logger.info("Writing %s", path)
    with imas.DBEntry(f"imas:hdf5?path={path}", "w") as dbase:
        dbase.put(ids_equilibrium, 0)
        dbase.put(ids_core_profiles, 0)


def main():
    """
    Run circular plasma EBW test case.
    """
    write_imas()

    optimise(run_directory, [0.0], overwrite=True)

    trace(run_directory, [0.0], overwrite=True)

    plot_all(
        run_directory,
        [0.0],
        show=True,
        xy=True,
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
