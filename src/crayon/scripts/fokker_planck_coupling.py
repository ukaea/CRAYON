"""
Wrapper scripts for coupling to Fokker-Planck codes.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.ray_tracing.ray_tracer import RayTracingOutput

logger = logging.getLogger(__name__)

M_TO_CM = 1.0e-2
PER_M_TO_PER_CM = 1 / M_TO_CM
PER_M3_TO_PER_CM3 = 1.0e-6
B_TO_GAUSS = 1.0e4
W_TO_ERG_PER_S = 1.0e7


def couple_cql3d(run_directory: pathlib.Path):
    """
    Couple Crayon output to CQL3D input file.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    """
    raise NotImplementedError


def write_genray_style(
    crayon_output: list[RayTracingOutput],
    output_file: pathlib.Path,
):
    """
    Write selected ray data in GENRAY style output file.
    Used for coupling to CQL3D and LUKE.

    Attributes
    ----------
    crayon_output : list[RayTracingOutput]
        List of ray tracing output to write.
    output_file : pathlib.Path
        Name of file to write.
    """
    raise NotImplementedError("Reimplement")

    with nc4.Dataset(crayon_output, "r", auto_complex=True) as dset:
        trajectory = RayTracingOutput.read_netcdf(dset)

    neltmax = trajectory.time_ns.size

    arc_length = trajectory.arc_length_m * M_TO_CM
    eikonal_phase = trajectory.eikonal_phase_rad

    ray_r = trajectory.position[CoordinateSystem.CYLINDRICAL][:, 0] * M_TO_CM
    ray_phi = trajectory.position[CoordinateSystem.CYLINDRICAL][:, 1]
    ray_z = trajectory.position[CoordinateSystem.CYLINDRICAL][:, 2] * M_TO_CM
    ray_rho_poloidal = trajectory.position[CoordinateSystem.RHO_POLOIDAL][:, 0]
    n_par = trajectory.n_parallel
    n_perp = trajectory.n_perp
    power = trajectory.power_w * W_TO_ERG_PER_S
    magnetic_field_strength = trajectory.magnetic_field_strength_t * B_TO_GAUSS
    ray_density = trajectory.electron_density_per_m3 * PER_M3_TO_PER_CM3
    polarisation = trajectory.polarisation_stix
    fluxn = trajectory.normalised_em_energy_density

    # Damping rate in perpendicular direction.
    velocity = np.linalg.norm(trajectory.velocity, axis=1)
    velocity_perpendicular = np.empty_like(velocity)

    for i in range(len(velocity)):
        cos_theta = np.dot(
            trajectory.velocity[i, :], trajectory.magnetic_field_t[i, :]
        ) / (velocity[i] * trajectory.magnetic_field_strength_t[i])
        velocity_perpendicular[i] = velocity[i] * np.sqrt(
            1 - cos_theta * cos_theta
        )

    linear_damping_wavenumber = PER_M_TO_PER_CM * (
        trajectory.damping_rate / velocity_perpendicular
    )

    _scalar = "scalar"
    _scalar_rays = "nrays"
    _scalar_ray = ("nrays", "neltmax")
    _complex_vector_ray = ("two", "nrays", "neltmax")

    with nc4.Dataset(output_file, "w") as dset:
        dset.createDimension("nrays", 1)
        dset.createDimension("neltmax", neltmax)
        dset.createDimension("scalar", 1)
        dset.createDimension("two", 2)
        dset.createDimension("char8dim", 8)
        dset.createDimension("char64dim", 64)
        dset.createDimension("char128dim", 128)
        dset.createDimension("char256dim", 256)

        # Scalars.
        _nray = dset.createVariable("nray", "i4", _scalar)
        _nharm = dset.createVariable("nharm", "i4", _scalar)
        _freqcy = dset.createVariable("freqcy", "f8", _scalar)
        _nrayelt = dset.createVariable("nrayelt", "i4", _scalar_rays)

        _nray[...] = 1
        _nharm[...] = 0
        _freqcy[...] = 1
        _nrayelt[...] = neltmax

        # Ray scalars.
        _arc_length = dset.createVariable("ws", "f8", _scalar_ray)
        _eikonal_phase = dset.createVariable("seikon", "f8", _scalar_ray)
        _rho_poloidal = dset.createVariable("spsi", "f8", _scalar_ray)
        _ray_r = dset.createVariable("wr", "f8", _scalar_ray)
        _ray_phi = dset.createVariable("wphi", "f8", _scalar_ray)
        _ray_z = dset.createVariable("wz", "f8", _scalar_ray)
        _n_par = dset.createVariable("wnpar", "f8", _scalar_ray)
        _n_perp = dset.createVariable("wnper", "f8", _scalar_ray)
        _power = dset.createVariable("delpwr", "f8", _scalar_ray)
        _sdpwr = dset.createVariable("sdpwr", "f8", _scalar_ray)
        _linear_damping_wavenumber = dset.createVariable(
            "salphal", "f8", _scalar_ray
        )
        _n_par_width = dset.createVariable("wdnpar", "f8", _scalar_ray)
        _magnetic_field_strength = dset.createVariable(
            "sbtot", "f8", _scalar_ray
        )
        _ray_density = dset.createVariable("sene", "f8", _scalar_ray)
        _fluxn = dset.createVariable("fluxn", "f8", _scalar_ray)

        _arc_length[0, :] = arc_length
        _eikonal_phase[0, :] = eikonal_phase
        _rho_poloidal[0, :] = ray_rho_poloidal
        _ray_r[0, :] = ray_r
        _ray_phi[0, :] = ray_phi
        _ray_z[0, :] = ray_z
        _n_par[0, :] = n_par
        _n_perp[0, :] = n_perp
        _power[0, :] = power
        _sdpwr[:] = 0.0
        _linear_damping_wavenumber[0, :] = linear_damping_wavenumber
        _n_par_width[0, :] = 0.05 * n_par
        _magnetic_field_strength[0, :] = magnetic_field_strength
        _ray_density[0, :] = ray_density
        _fluxn[0, :] = fluxn

        # Ray vectors.
        _polarisation_x = dset.createVariable(
            "cwexde", "f8", _complex_vector_ray
        )
        _polarisation_y = dset.createVariable(
            "cweyde", "f8", _complex_vector_ray
        )
        _polarisation_z = dset.createVariable(
            "cwezde", "f8", _complex_vector_ray
        )

        _polarisation_x[0, 0, :] = polarisation[:, 0].real
        _polarisation_x[1, 0, :] = polarisation[:, 0].imag
        _polarisation_y[0, 0, :] = polarisation[:, 1].real
        _polarisation_y[1, 0, :] = polarisation[:, 1].imag
        _polarisation_z[0, 0, :] = polarisation[:, 2].real
        _polarisation_z[1, 0, :] = polarisation[:, 2].imag
