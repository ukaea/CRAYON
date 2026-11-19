"""
End-to-end check of the power split at O-X mode conversion, on the slab OXB
case of the integration tests, for a single ray and for a ray bundle.
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

OXB_INPUT = (
    pathlib.Path(__file__).parents[2] / "integration" / "slab" / "oxb" / "input"
)
BEAM_WAIST_RADIUS_M = 0.05  # as in OXB_INPUT / "rays.toml"


def first(variable):
    """
    First value of a netCDF variable.
    """
    return float(np.ravel(variable[:])[0])


def trace_oxb(run_directory, n_radial_zones):
    """
    Trace the slab OXB case with the given number of radial zones and
    return {parent: (power before, power after, child initial power, T)}
    for every ray that converted, T computed from the conversion values
    the ray stored.
    """
    shutil.copytree(OXB_INPUT, run_directory / "input")
    for toml, old, new in [
        ("rays.toml", "n_radial_zones = 1", f"n_radial_zones = {n_radial_zones}"),
        # The parents convert near node 19; stop the children soon after.
        ("options.toml", "max_ray_nodes = 100", "max_ray_nodes = 25"),
    ]:
        path = run_directory / "input" / toml
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))
    trace(run_directory, [0.0], overwrite=True)

    splits = {}
    with nc4.Dataset(run_directory / "output" / "0.0s" / "rays.nc") as data:
        for name in data.groups:
            child = f"{name}-0"
            if child not in data.groups:
                continue
            ray = data[name]
            power = np.ravel(ray["power"][:])
            # The conversion is the largest drop in power along the ray.
            step = int(np.argmax(power[:-1] - power[1:]))
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
                power[step],
                power[step + 1],
                first(data[child]["initial_power"]),
                transmission,
            )
    return splits


@pytest.mark.parametrize("n_radial_zones", [1, 2])
def test_ox_conversion_power_split(tmp_path, n_radial_zones):
    """
    At conversion the child (X mode) ray starts with T times the parent's
    power there, and the parent keeps the rest.
    """
    splits = trace_oxb(tmp_path, n_radial_zones)

    assert splits, "no ray converted"
    for before, after, child, transmission in splits.values():
        assert 0.0 < transmission <= 1.0
        nptest.assert_allclose(child, transmission * before, rtol=1e-4)
        nptest.assert_allclose(child + after, before, rtol=1e-4)
