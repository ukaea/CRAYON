"""
End-to-end checks of O-X mode conversion on the slab OXB case of the
integration tests: the power split, for a single ray and a ray bundle, and
that a ray converts only once.
"""

# Standard imports
import pathlib
import shutil

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.ray_tracing.caches.mode_conversion import (
    mjolhus_gaussian_beam,
    mjolhus_plane_wave,
)
from crayon.scripts import trace

OXB_INPUT = pathlib.Path(__file__).parents[2].joinpath(
    "integration", "slab", "oxb", "input"
)
BEAM_WAIST_RADIUS_M = 0.05  # as in OXB_INPUT / "rays.toml"


def first(variable: nc4.Variable) -> float:
    """
    First value of a netCDF variable.

    Parameters
    ----------
    variable : nc4.Variable
        Variable.

    Returns
    -------
    value : float
        First value.
    """
    return float(np.ravel(variable[:])[0])


def run_oxb(
    run_directory: pathlib.Path, replacements: list[tuple[str, str, str]]
) -> pathlib.Path:
    """
    Trace the slab OXB case with text replacements in its input files.

    Parameters
    ----------
    run_directory : pathlib.Path
        Directory to copy the inputs to and trace in.
    replacements : list[tuple[str, str, str]]
        (input file name, old text, new text) for each replacement.

    Returns
    -------
    rays_file : pathlib.Path
        Rays output file.
    """
    shutil.copytree(OXB_INPUT, run_directory / "input")
    for toml, old, new in replacements:
        path = run_directory / "input" / toml
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))
    trace(run_directory, [0.0], overwrite=True)
    return run_directory / "output" / "0.0s" / "rays.nc"


def power_splits(
    rays_file: pathlib.Path, n_radial_zones: int
) -> dict[str, tuple[float, float, float, float]]:
    """
    Power split at the first conversion of every ray that converted.

    The parent's power at the conversion is its power after the internal
    damping of the step on which it converted, before the converted power
    is removed (as external optical depth).

    Parameters
    ----------
    rays_file : pathlib.Path
        Rays output file.
    n_radial_zones : int
        Number of radial zones; 1 is a single ray (beam transmission),
        more is a ray bundle (plane-wave transmission).

    Returns
    -------
    splits : dict[str, tuple[float, float, float, float]]
        For each parent: (its power at conversion, its power after the
        conversion, the child's initial power, the transmission T computed
        from the conversion values the parent stored).
    """
    splits = {}
    with nc4.Dataset(rays_file) as data:
        for name in data.groups:
            child = f"{name}-0"
            if child not in data.groups:
                continue
            ray = data[name]
            power = np.ravel(ray["power"][:])
            tau_internal = np.ravel(ray["optical_depth_internal"][:])
            tau_external = np.ravel(ray["optical_depth_external"][:])
            # The conversion is the largest drop in power along the ray.
            step = int(np.argmax(power[:-1] - power[1:]))
            power_at_conversion = first(ray["initial_power"]) * np.exp(
                -(tau_internal[step + 1] + tau_external[step])
            )
            args = (
                first(ray["k0ln_saddle"]),
                first(ray["y_saddle"]),
                first(ray["n_parallel_at_conversion"]),
                first(ray["n_y_at_conversion"]),
            )
            if n_radial_zones == 1:
                k0w = first(ray["vacuum_wavenumber"]) * BEAM_WAIST_RADIUS_M
                transmission = mjolhus_gaussian_beam(*args, k0w)
            else:
                transmission = mjolhus_plane_wave(*args)
            splits[name] = (
                power_at_conversion,
                power[step + 1],
                first(data[child]["initial_power"]),
                transmission,
            )
    return splits


class TestModeConversionPower:
    """
    End-to-end tests of O-X mode conversion on the slab OXB case.
    """

    @staticmethod
    @pytest.mark.parametrize(
        ("n_radial_zones", "n_parallel"), [(1, 0.655), (2, 0.6)]
    )
    def test_power_split(
        tmp_path: pathlib.Path, n_radial_zones: int, n_parallel: float
    ):
        """
        Test the power split at conversion: the child (X mode) ray starts
        with T times the parent's power there, and the parent keeps the rest.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Run directory.
        n_radial_zones : int
            Number of radial zones.
        n_parallel : float
            Launch parallel refractive index.
        """
        rays_file = run_oxb(
            tmp_path,
            [
                (
                    "rays.toml",
                    "n_radial_zones = 1",
                    f"n_radial_zones = {n_radial_zones}",
                ),
                (
                    "rays.toml",
                    "n_parallel = 0.655",
                    f"n_parallel = {n_parallel}",
                ),
                # The parents convert near node 19; stop the children soon
                # after.
                ("options.toml", "max_ray_nodes = 100", "max_ray_nodes = 25"),
            ],
        )
        splits = power_splits(rays_file, n_radial_zones)

        assert splits, "no ray converted"
        for at_conversion, after, child, transmission in splits.values():
            assert 0.0 < transmission <= 1.0
            nptest.assert_allclose(
                child, transmission * at_conversion, rtol=1e-8
            )
            nptest.assert_allclose(
                after, (1 - transmission) * at_conversion, rtol=1e-8
            )

    @staticmethod
    @pytest.mark.parametrize(
        ("fidelity", "n_parallel"),
        [("HIGH", 0.655), ("MEDIUM", 0.6546), ("MEDIUM", 0.6544)],
    )
    def test_conversion_happens_once(
        tmp_path: pathlib.Path, fidelity: str, n_parallel: float
    ):
        """
        Test that the single O mode ray of the slab OXB case, launched close
        to the optimal parallel refractive index, converts once and is then
        reflected: it has exactly one converted child, it never reaches
        densities above its cutoff, X = 1, and it ends outside the plasma.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Run directory.
        fidelity : str
            Fidelity option.
        n_parallel : float
            Launch parallel refractive index.
        """
        rays_file = run_oxb(
            tmp_path,
            [
                (
                    "options.toml",
                    'fidelity = "MEDIUM"',
                    f'fidelity = "{fidelity}"',
                ),
                (
                    "rays.toml",
                    "n_parallel = 0.655",
                    f"n_parallel = {n_parallel}",
                ),
            ],
        )

        with nc4.Dataset(rays_file) as data:
            children = [
                name for name in data.groups if name.startswith("ray_1-0-")
            ]
            density = np.ravel(
                data["ray_1-0"]["normalised_electron_density"][:]
            )
            k_x = np.reshape(data["ray_1-0"]["wavevector"][:], (-1, 3))[:, 0]

        assert children == ["ray_1-0-0"]
        assert np.max(density) < 1.01  # noqa: PLR2004
        # It leaves the plasma (towards smaller x) through the limiter.
        assert k_x[-1] < 0.0
        assert density[-1] < 1e-3  # noqa: PLR2004
