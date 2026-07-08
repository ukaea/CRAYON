"""
Unit tests for system_data.system_data_provider.
"""

# Standard imports
import logging
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Third party imports
# Local imports
from crayon.coordinates import CoordinateCoordinator
from crayon.imas import DBEntry, imas, imasdef
from crayon.shared.constants import CoordinateSystem
from crayon.shared.dimensions import Dimensions
from crayon.system_data.limiter import (
    Cylinder,
    Disk,
    Limiter,
    LimiterEffect,
    Plane,
)
from crayon.system_data.schemas import (
    EFFECTIVE_CHARGE,
    ELECTRON_DENSITY_PER_M3,
    ELECTRON_TEMPERATURE_EV,
    CoordinateToroidal,
    DataSourceImas,
    DataSourceNetcdf,
    DataSourceVmec,
    LimiterAnalyticBoundingBox2D,
    LimiterAnalyticBoundingBox3D,
    LimiterImas2D,
    LimiterImasBoundingBox2D,
    MagneticModelTokamak,
    ModelAnalyticConstant,
    ModelAnalyticQuadraticBowl,
    ModelAnalyticQuadraticChannel,
    ModelAnalyticQuadraticWell,
    ModelAnalyticRamp,
    ModelImas,
    ModelNetcdf,
)
from crayon.system_data.system_data_provider import (
    Kinetic,
    Limiters,
    Magnetic,
    MagneticTokamak,
    SystemData,
    SystemDataProvider,
)
from crayon.value_model.analytic import (
    C1Ramp,
    Constant,
    QuadraticBowl,
    QuadraticChannel,
    QuadraticWell,
)
from crayon.value_model.magnetic_field import AxisymmetricMagneticField
from crayon.value_model.models import ValueModel
from crayon.value_model.splines import (
    Spline1D,
    Spline2D,
)

logger = logging.getLogger(__name__)


class TestKinetic:
    """
    Unit tests for Kinetic.
    """

    @staticmethod
    def make_core_profiles(
        rho_poloidal: np.ndarray[float],
        rho_toroidal: np.ndarray[float],
        time_index: int,
    ):
        """
        Create IDS core_profiles.

        Parameters
        ----------
        rho_poloidal: np.ndarray[float]
        Root normalised poloidal flux grid.
        rho_toroidal: np.ndarray[float]
            Root normalised toroidal flux grid.
        time_index: int
            Time slice index to save at.

        Returns
        -------
        core_profiles
            IDS core_profiles.
        """
        core_profiles = imas.core_profiles()
        core_profiles.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        core_profiles.time.resize(time_index + 1, refcheck=False)
        core_profiles.time = np.arange(time_index + 1)

        core_profiles.profiles_1d.resize(time_index + 1)
        _profiles_1d = core_profiles.profiles_1d[time_index]
        _profiles_1d.grid.rho_pol_norm = rho_poloidal
        _profiles_1d.grid.rho_tor_norm = rho_toroidal

        return core_profiles

    def test_from_imas(self):
        """
        Test creation from IMAS.
        """
        rho_poloidal = np.linspace(0, 1, 51)
        rho_toroidal = np.square(rho_poloidal)

        electron_density_per_m3 = np.linspace(0, 2, rho_poloidal.size)
        electron_temperature_ev = np.linspace(0, 3, rho_poloidal.size)
        effective_charge = np.linspace(0, 4, rho_poloidal.size)

        time_index_ne = 0
        core_profiles_ne = self.make_core_profiles(
            rho_poloidal, rho_toroidal, time_index_ne
        )
        core_profiles_ne.profiles_1d[
            time_index_ne
        ].electrons.density = electron_density_per_m3
        scale_factor_ne = 1.2

        time_index_te = 1
        core_profiles_te = self.make_core_profiles(
            rho_poloidal, rho_toroidal, time_index_te
        )
        core_profiles_te.profiles_1d[
            time_index_te
        ].electrons.temperature = electron_temperature_ev
        scale_factor_te = 1.3

        time_index_zeff = 2
        core_profiles_zeff = self.make_core_profiles(
            rho_poloidal, rho_toroidal, time_index_zeff
        )
        core_profiles_zeff.profiles_1d[time_index_zeff].zeff = effective_charge
        core_profiles_zeff.profiles_1d[
            time_index_zeff
        ].grid.rho_pol_norm = np.full(rho_poloidal.shape, imasdef.EMPTY_FLOAT)

        scale_factor_zeff = 1.4

        core_profiles_ne.validate()
        core_profiles_te.validate()
        core_profiles_zeff.validate()

        ne = Kinetic.from_imas_profiles_1d(
            ELECTRON_DENSITY_PER_M3,
            core_profiles_ne,
            time_index_ne,
            scale_factor_ne,
        )

        assert isinstance(ne, Spline1D)
        assert ne.coordinate_system == CoordinateSystem.RHO_POLOIDAL
        nptest.assert_allclose(ne._abscissas[0], rho_poloidal)
        nptest.assert_allclose(ne._data, electron_density_per_m3)
        assert ne.scale_factor == scale_factor_ne

        te = Kinetic.from_imas_profiles_1d(
            ELECTRON_TEMPERATURE_EV,
            core_profiles_te,
            time_index_te,
            scale_factor_te,
        )

        assert isinstance(te, Spline1D)
        assert te.coordinate_system == CoordinateSystem.RHO_POLOIDAL

        nptest.assert_allclose(te._abscissas[0], rho_poloidal)
        nptest.assert_allclose(te._data, electron_temperature_ev)
        assert te.scale_factor == scale_factor_te

        zeff = Kinetic.from_imas_profiles_1d(
            EFFECTIVE_CHARGE,
            core_profiles_zeff,
            time_index_zeff,
            scale_factor_zeff,
        )

        assert isinstance(zeff, Spline1D)
        assert zeff.coordinate_system == CoordinateSystem.RHO_TOROIDAL
        nptest.assert_allclose(zeff._abscissas[0], rho_toroidal)
        nptest.assert_allclose(zeff._data, effective_charge)
        assert zeff.scale_factor == scale_factor_zeff

    @staticmethod
    def test_from_netcdf():
        """
        Test creation from netCDF4.
        """
        n0, n1, n2 = 3, 11, 12

        time = np.linspace(0, 2, n0)
        dim_1 = np.linspace(0.0, 1.0, n1)
        dim_2 = np.linspace(1.0, 2.0, n2)

        data_1d = np.linspace(3.0, 4.0, n0 * n1).reshape((n0, n1))
        data_2d = np.linspace(4.0, 5.0, n0 * n1 * n2).reshape((n0, n1, n2))

        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            # Create Dimensions.
            dset.createDimension("x", 3)

            dset.createDimension("time", 3)
            v = dset.createVariable("time_s", "f4", "time")
            v[:] = time

            dset.createDimension("dim_1", n1)
            v = dset.createVariable("dim_1", "f4", "dim_1")
            v[:] = dim_1

            # Test 1d.
            g = dset.createGroup("electron_density_per_m3")
            g.setncattr("coordinate_system", "cartesian")

            v = g.createVariable("dependent_components", "u4", "x")
            v[:] = 0, 1, 0

            v = g.createVariable("data", "f4", ("time", "dim_1"))
            v[:, :] = data_1d

            ne = Kinetic.from_netcdf(ELECTRON_DENSITY_PER_M3, 0.0, dset, 1.1)

            assert isinstance(ne, Spline1D)
            assert ne.coordinate_system == CoordinateSystem.CARTESIAN
            nptest.assert_allclose(ne._abscissas[0], dim_1)
            nptest.assert_allclose(ne._data, data_1d[0, :])
            nptest.assert_allclose(ne.scale_factor, 1.1)

            # Test 2d.
            dset.createDimension("dim_2", n2)
            v = dset.createVariable("dim_2", "f4", "dim_2")
            v[:] = dim_2

            g = dset.createGroup("electron_temperature_ev")
            g.setncattr("coordinate_system", "cylindrical")

            v = g.createVariable("dependent_components", "u4", "x")
            v[:] = 0, 1, 1

            v = g.createVariable("data", "f4", ("time", "dim_1", "dim_2"))
            v[:, :, :] = data_2d

            te = Kinetic.from_netcdf(ELECTRON_TEMPERATURE_EV, 1.0, dset, 1.2)

            assert isinstance(te, Spline2D)
            assert te.coordinate_system == CoordinateSystem.CYLINDRICAL
            nptest.assert_allclose(te._abscissas[0], dim_1)
            nptest.assert_allclose(te._abscissas[1], dim_2)
            nptest.assert_allclose(te._data, data_2d[1, :, :])
            nptest.assert_allclose(te.scale_factor, 1.2)

    @staticmethod
    def test_round_trip_netcdf():
        """
        Test serialising and deserialising to netCDF4 file gives same object.
        """
        electron_density_per_m3 = (
            ValueModel.electron_density_per_m3().constant(
                CoordinateSystem.CARTESIAN, 1.0
            )
        )
        electron_temperature_ev = (
            ValueModel.electron_temperature_ev().constant(
                CoordinateSystem.CARTESIAN, 2.0
            )
        )
        effective_charge = ValueModel.effective_charge().constant(
            CoordinateSystem.CARTESIAN, 3.0
        )

        kinetic = Kinetic(
            electron_density_per_m3, electron_temperature_ev, effective_charge
        )

        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            kinetic.write_netcdf(dset)
            kinetic_2 = Kinetic.read_netcdf(dset)

        assert isinstance(kinetic_2.electron_density_per_m3, Constant)
        nptest.assert_allclose(
            kinetic_2.electron_density_per_m3.constant_value, 1.0
        )

        assert isinstance(kinetic_2.electron_temperature_ev, Constant)
        nptest.assert_allclose(
            kinetic_2.electron_temperature_ev.constant_value, 2.0
        )

        assert isinstance(kinetic_2.effective_charge, Constant)
        nptest.assert_allclose(kinetic_2.effective_charge.constant_value, 3.0)


class TestMagnetic:
    """
    Unit tests for Magnetic.
    """

    @staticmethod
    def test_round_trip_netcdf():
        """
        Test serialising and deserialising to netCDF4 file gives same object.
        """
        magnetic = Magnetic(
            ValueModel.magnetic_field_t().constant(
                CoordinateSystem.CYLINDRICAL, [1.0, 0.0, 0.0]
            )
        )

        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            magnetic.write_netcdf(dset)
            magnetic_2 = Magnetic.read_netcdf(dset)

        assert isinstance(magnetic_2.magnetic_field_t, Constant)
        assert (
            magnetic.magnetic_field_t.coordinate_system
            == magnetic_2.magnetic_field_t.coordinate_system
        )
        nptest.assert_allclose(
            magnetic.magnetic_field_t.constant_value,
            magnetic_2.magnetic_field_t.constant_value,
        )
        nptest.assert_allclose(
            magnetic.magnetic_field_t.scale_factor,
            magnetic_2.magnetic_field_t.scale_factor,
        )

    @staticmethod
    def test_from_imas():
        """
        Test creation from IMAS database
        """
        ids_equilibrium = imas.equilibrium()
        ids_equilibrium.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        ids_equilibrium.time.resize(1, refcheck=False)
        ids_equilibrium.time[0] = 1.0

        r = np.linspace(1.0, 3.0, 51)
        z = np.linspace(-1.0, 1.0, 41)
        r_mesh, z_mesh = np.meshgrid(r, z, indexing="ij")
        radius = np.sqrt(r_mesh * r_mesh + z_mesh * z_mesh)

        br = 1 / (1 + radius)
        bphi = radius * radius
        bz = -1 / 1 / (1 + radius)

        ids_equilibrium.time_slice.resize(1)
        _profiles_2d = ids_equilibrium.time_slice[0].profiles_2d
        _profiles_2d.resize(1)
        _profiles_2d[0].grid_type.index = 1
        _profiles_2d[0].grid.dim1 = r
        _profiles_2d[0].grid.dim2 = z
        _profiles_2d[0].b_field_r = br
        _profiles_2d[0].b_field_phi = bphi
        _profiles_2d[0].b_field_z = bz

        ids_equilibrium.validate()

        magnetic = Magnetic.from_imas(ids_equilibrium, 0, 1.1)

        assert isinstance(magnetic.magnetic_field_t, Spline2D)
        nptest.assert_allclose(magnetic.magnetic_field_t._abscissas[0], r)
        nptest.assert_allclose(magnetic.magnetic_field_t._abscissas[1], z)
        nptest.assert_allclose(magnetic.magnetic_field_t._data[:, :, 0], br)
        nptest.assert_allclose(
            magnetic.magnetic_field_t._data[:, :, 1], bphi / r[:, np.newaxis]
        )
        nptest.assert_allclose(magnetic.magnetic_field_t._data[:, :, 2], bz)
        nptest.assert_allclose(magnetic.magnetic_field_t.scale_factor, 1.1)


class TestMagneticTokamak:
    """
    Unit tests for MagneticTokamak.
    """

    @staticmethod
    def test_from_imas():
        """
        Test creation from IMAS database.
        """
        equilibrium = imas.equilibrium()
        equilibrium.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        equilibrium.time.resize(1, refcheck=False)
        equilibrium.time[0] = 1.0
        equilibrium.time_slice.resize(1)

        rho_poloidal = np.linspace(0, 1, 51)
        rho_toroidal = np.square(rho_poloidal)
        f_toroidal = 2.0 * rho_poloidal

        r = np.linspace(1.0, 3.0, 21)
        z = np.linspace(-1.0, 1.0, 31)
        r_grid, z_grid = np.meshgrid(r, z, indexing="ij")

        r0, z0 = 2.0, 0.0
        psi = np.square(r_grid - 2.0) + np.square(z_grid)
        psi_axis, psi_sep = 0.0, 0.5**2
        total_toroidal_flux = 2.0
        rho_poloidal_2d = np.sqrt(psi / psi_sep)

        _profiles_1d = equilibrium.time_slice[0].profiles_1d
        _profiles_1d.psi = np.zeros_like(rho_poloidal)
        _profiles_1d.f = f_toroidal
        _profiles_1d.psi_norm = np.square(rho_poloidal)
        _profiles_1d.phi = total_toroidal_flux * np.square(rho_toroidal)
        _profiles_1d.rho_tor_norm = rho_toroidal

        cross_sectional_area_m2 = np.linspace(0, 2, rho_poloidal.size)
        volume_m3 = np.linspace(0, 3, rho_poloidal.size)
        fsa_1_over_r = np.linspace(0, 4, rho_poloidal.size)
        fsa_1_over_r2 = np.linspace(0, 5, rho_poloidal.size)
        fsa_b = np.linspace(0, 6, rho_poloidal.size)
        b_max = np.linspace(0, 7, rho_poloidal.size)
        trapped_particle_fraction = np.linspace(0, 8, rho_poloidal.size)

        _profiles_1d.area = cross_sectional_area_m2
        _profiles_1d.volume = volume_m3
        _profiles_1d.gm9 = fsa_1_over_r
        _profiles_1d.gm1 = fsa_1_over_r2
        _profiles_1d.b_field_average = fsa_b
        _profiles_1d.b_field_max = b_max
        _profiles_1d.trapped_fraction = trapped_particle_fraction

        _profiles_2d = equilibrium.time_slice[0].profiles_2d
        _profiles_2d.resize(1)
        _profiles_2d[0].grid_type.index = 1
        _profiles_2d[0].grid.dim1 = r
        _profiles_2d[0].grid.dim2 = z
        _profiles_2d[0].psi = psi

        _globals = equilibrium.time_slice[0].global_quantities
        _globals.psi_axis = psi_axis
        _globals.psi_boundary = psi_sep
        _globals.magnetic_axis.r = r0
        _globals.magnetic_axis.z = z0

        equilibrium.vacuum_toroidal_field.r0 = 1.0
        equilibrium.vacuum_toroidal_field.b0 = 2.0 * np.ones(1)

        equilibrium.validate()

        scale_factor = 1.3

        magnetic = MagneticTokamak.from_imas(equilibrium, 0, scale_factor)

        assert isinstance(magnetic.rho_poloidal_2d, Spline2D)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._abscissas[0], r)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._abscissas[1], z)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._data, rho_poloidal_2d)
        nptest.assert_allclose(magnetic.magnetic_axis_m, (r0, z0))
        nptest.assert_allclose(magnetic.total_poloidal_flux_wb, psi_sep)

        assert isinstance(magnetic.rho_poloidal_to_toroidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._data, rho_toroidal
        )

        assert isinstance(magnetic.rho_toroidal_to_poloidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._abscissas[0], rho_toroidal
        )
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._data, rho_poloidal
        )

        nptest.assert_allclose(
            magnetic.total_toroidal_flux_wb, total_toroidal_flux
        )

        assert isinstance(magnetic.f_toroidal_tm, Spline1D)
        nptest.assert_allclose(
            magnetic.f_toroidal_tm._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.f_toroidal_tm._data, f_toroidal)

        assert isinstance(magnetic.cross_sectional_area_m2, Spline1D)
        nptest.assert_allclose(
            magnetic.cross_sectional_area_m2._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.cross_sectional_area_m2._data, cross_sectional_area_m2
        )

        assert isinstance(magnetic.volume_m3, Spline1D)
        nptest.assert_allclose(magnetic.volume_m3._abscissas[0], rho_poloidal)
        nptest.assert_allclose(magnetic.volume_m3._data, volume_m3)

        assert isinstance(magnetic.fsa_1_over_r, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_1_over_r._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.fsa_1_over_r._data, fsa_1_over_r)

        assert isinstance(magnetic.fsa_1_over_r2, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_1_over_r2._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.fsa_1_over_r2._data, fsa_1_over_r2)

        assert isinstance(magnetic.fsa_b, Spline1D)
        nptest.assert_allclose(magnetic.fsa_b._abscissas[0], rho_poloidal)
        nptest.assert_allclose(magnetic.fsa_b._data, fsa_b)

        assert isinstance(magnetic.b_max, Spline1D)
        nptest.assert_allclose(magnetic.b_max._abscissas[0], rho_poloidal)
        nptest.assert_allclose(magnetic.b_max._data, b_max)

        assert isinstance(magnetic.trapped_particle_fraction, Spline1D)
        nptest.assert_allclose(
            magnetic.trapped_particle_fraction._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.trapped_particle_fraction._data, trapped_particle_fraction
        )

        assert isinstance(magnetic.magnetic_field_t, AxisymmetricMagneticField)
        assert magnetic.magnetic_field_t.scale_factor == scale_factor

    test_circles = ((1.0, 0.2), (0.8, 0.5), (2.4, 0.9), (3.0, 2.2))

    @staticmethod
    @pytest.mark.parametrize(("r_maj", "r_min"), test_circles)
    def test_calculate_area_volume(r_maj: float, r_min: float):
        """
        Test calculation of flux surface area and flux tube volume of a
        torus.

        Parameters
        ----------
        r_maj : float
            Major radius of torus
        r_min : float
            Minor radius of torus.
        """
        theta = np.linspace(0, 2 * np.pi, 51)
        radius = r_maj + r_min * np.cos(theta)
        polar_radius = np.full(radius.shape, r_min)

        area, volume = MagneticTokamak._calculate_area_volume(
            theta, radius, polar_radius
        )

        expected_area = np.pi * r_min * r_min
        expected_volume = 2 * np.pi * r_maj * expected_area

        nptest.assert_allclose(expected_area, area)
        nptest.assert_allclose(expected_volume, volume)

    @staticmethod
    @pytest.mark.parametrize(("r_maj", "r_min"), test_circles)
    def test_get_fsa_kernel_denominator(r_maj: float, r_min: float):
        """
        Test calculation of flux surface averaging kernel of a torus.

        Parameters
        ----------
        r_maj : float
            Major radius of torus
        r_min : float
            Minor radius of torus.
        """
        theta = np.linspace(0, 2 * np.pi, 101)
        radius = r_maj + r_min * np.cos(theta)
        polar_radius = np.full(theta.shape, r_min)
        grad_rho = np.full(theta.shape, 1.0)

        kernel, denominator = MagneticTokamak._get_fsa_kernel_denominator(
            theta, radius, polar_radius, grad_rho
        )

        nptest.assert_allclose(kernel, 2 * np.pi * radius * polar_radius)
        nptest.assert_allclose(denominator, 4 * np.pi**2 * r_min * r_maj)

    @staticmethod
    @pytest.mark.parametrize(("r_maj", "r_min"), test_circles)
    def test_flux_surface_integrals_circle(r_maj: float, r_min: float):
        """
        Test calculation of flux surface averaging integrals of a torus.

        Parameters
        ----------
        r_maj : float
            Major radius of torus
        r_min : float
            Minor radius of torus.
        """
        eps = r_min / r_maj
        eps2 = eps * eps

        theta = np.linspace(0, 2 * np.pi, 101)
        radius = r_maj + r_min * np.cos(theta)
        polar_radius = np.full(theta.shape, r_min)
        grad_rho = np.full(theta.shape, 1.0 / r_min)

        b0 = 1.4
        magnetic_field_strength = b0 / (1 + eps * np.cos(theta))

        (
            surface_area_m2,
            volume_m3,
            fsa_1_over_r,
            fsa_1_over_r2,
            fsa_b,
            b_max,
            _,
        ) = MagneticTokamak._flux_surface_integrals(
            theta, radius, polar_radius, grad_rho, magnetic_field_strength
        )

        expected_surface_area_m2 = np.pi * r_min * r_min
        expected_volume_m3 = 2 * np.pi * r_maj * expected_surface_area_m2
        expected_fsa_1_over_r = 1 / r_maj
        expected_fsa_1_over_r2 = 1 / (r_maj * r_maj * np.sqrt(1 - eps2))
        expected_fsa_b = b0
        expected_b_max = b0 / (1 - eps)

        nptest.assert_allclose(surface_area_m2, expected_surface_area_m2)
        nptest.assert_allclose(volume_m3, expected_volume_m3)
        nptest.assert_allclose(fsa_1_over_r, expected_fsa_1_over_r)
        nptest.assert_allclose(fsa_1_over_r2, expected_fsa_1_over_r2)
        nptest.assert_allclose(fsa_b, expected_fsa_b)
        nptest.assert_allclose(b_max, expected_b_max)

        # Circulating particle fraction can only be analytically solved for
        # constant magnetic field strength ('square well').
        # As h in (0, 1), use epsilon to mock various values.
        h = eps
        expected_value = 1 - (1 + 0.5 * h) * np.sqrt(1 - h)

        fsa_kernel, fsa_denominator = (
            MagneticTokamak._get_fsa_kernel_denominator(
                theta, radius, polar_radius, grad_rho
            )
        )

        actual_value = MagneticTokamak._get_circulating_particle_fraction(
            theta,
            fsa_kernel,
            fsa_denominator,
            h,
            h * h,
        )

        nptest.assert_allclose(expected_value, actual_value, atol=5e-5)

    @staticmethod
    def test_from_imas_calculate_integrals():
        """
        Test calculation of flux surface integrals from IMAS.
        """
        r_maj, r_min = 1.0, 0.5
        eps = r_min / r_maj
        total_poloidal_flux = 1.4
        f = 1.2

        equilibrium = imas.equilibrium()
        equilibrium.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        equilibrium.time.resize(1, refcheck=False)
        equilibrium.time[0] = 1.0
        equilibrium.time_slice.resize(1)

        r0, r1 = r_maj - 1.1 * r_min, r_maj + 1.1 * r_min
        z0, z1 = -1.1 * r_min, 1.1 * r_min

        r = np.linspace(r0, r1, 95)
        z = np.linspace(z0, z1, 85)
        _dr, _dz = np.meshgrid(r - r_maj, z, indexing="ij")
        psi_norm_2d = (_dr**2 + _dz**2) / r_min**2
        psi_2d = total_poloidal_flux * psi_norm_2d
        rho_poloidal_2d = np.sqrt(psi_norm_2d)

        rho_poloidal = np.linspace(0, 1, 51)
        rho_toroidal = np.square(rho_poloidal)
        f_toroidal = np.full(rho_poloidal.shape, f)

        btor0 = f / r_maj
        bpol0 = total_poloidal_flux * rho_poloidal * eps / (np.pi * r_min**2)
        b0 = np.sqrt(np.square(btor0) + np.square(bpol0))

        psi_axis, psi_sep = 0.0, total_poloidal_flux
        total_toroidal_flux = 2.0

        _profiles_1d = equilibrium.time_slice[0].profiles_1d
        _profiles_1d.psi = np.zeros_like(rho_poloidal)
        _profiles_1d.f = f_toroidal
        _profiles_1d.psi_norm = np.square(rho_poloidal)
        _profiles_1d.phi = total_toroidal_flux * np.square(rho_toroidal)
        _profiles_1d.rho_tor_norm = rho_toroidal

        _profiles_2d = equilibrium.time_slice[0].profiles_2d
        _profiles_2d.resize(1)
        _profiles_2d[0].grid_type.index = 1
        _profiles_2d[0].grid.dim1 = r
        _profiles_2d[0].grid.dim2 = z
        _profiles_2d[0].psi = psi_2d

        _globals = equilibrium.time_slice[0].global_quantities
        _globals.psi_axis = psi_axis
        _globals.psi_boundary = psi_sep
        _globals.magnetic_axis.r = r_maj
        _globals.magnetic_axis.z = 0.0

        equilibrium.vacuum_toroidal_field.r0 = 1.0
        equilibrium.vacuum_toroidal_field.b0 = [f]

        equilibrium.validate()

        scale_factor = 1.0
        magnetic = MagneticTokamak.from_imas(equilibrium, 0, scale_factor)

        assert isinstance(magnetic.rho_poloidal_2d, Spline2D)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._abscissas[0], r)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._abscissas[1], z)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._data, rho_poloidal_2d)
        nptest.assert_allclose(magnetic.magnetic_axis_m, (r_maj, 0.0))
        nptest.assert_allclose(magnetic.total_poloidal_flux_wb, psi_sep)

        assert isinstance(magnetic.rho_poloidal_to_toroidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._data, rho_toroidal
        )

        assert isinstance(magnetic.rho_toroidal_to_poloidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._abscissas[0], rho_toroidal
        )
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._data, rho_poloidal
        )

        nptest.assert_allclose(
            magnetic.total_toroidal_flux_wb, total_toroidal_flux
        )

        assert isinstance(magnetic.f_toroidal_tm, Spline1D)
        nptest.assert_allclose(
            magnetic.f_toroidal_tm._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.f_toroidal_tm._data, f_toroidal)

        # Test derived flux surface averages.
        _x = rho_poloidal.reshape((-1, 1))
        a = r_min * rho_poloidal
        a2 = a * a

        assert isinstance(magnetic.cross_sectional_area_m2, Spline1D)
        cross_sectional_area_m2 = magnetic.cross_sectional_area_m2.value(_x)
        nptest.assert_allclose(cross_sectional_area_m2, np.pi * a2, atol=5e-4)

        assert isinstance(magnetic.volume_m3, Spline1D)
        volume_m3 = magnetic.volume_m3.value(_x)
        nptest.assert_allclose(
            volume_m3, 2 * np.pi * r_maj * np.pi * a2, atol=2e-3
        )

        assert isinstance(magnetic.fsa_1_over_r, Spline1D)
        fsa_1_over_r = magnetic.fsa_1_over_r.value(_x)
        nptest.assert_allclose(fsa_1_over_r, 1 / r_maj, atol=5e-5)

        assert isinstance(magnetic.fsa_1_over_r2, Spline1D)
        fsa_1_over_r2 = magnetic.fsa_1_over_r2.value(_x)
        nptest.assert_allclose(
            fsa_1_over_r2,
            1 / (r_maj * r_maj * np.sqrt(1 - np.square(rho_poloidal * eps))),
            atol=5e-4,
        )

        assert isinstance(magnetic.fsa_b, Spline1D)
        fsa_b = magnetic.fsa_b.value(_x)
        nptest.assert_allclose(fsa_b, b0, atol=5e-4)

        assert isinstance(magnetic.b_max, Spline1D)
        b_max = magnetic.b_max.value(_x)
        # Bad atol because derivatives close to axis are bad.
        nptest.assert_allclose(b_max, b0 / (1 - rho_poloidal * eps), atol=2e-2)

        assert isinstance(magnetic.trapped_particle_fraction, Spline1D)

        assert isinstance(magnetic.magnetic_field_t, AxisymmetricMagneticField)
        assert magnetic.magnetic_field_t.scale_factor == scale_factor

    @staticmethod
    def test_round_trip_netcdf():
        """
        Test serialising and deserialising to netCDF4 file gives same object.
        """
        scale_factor = 1.2
        magnetic_axis_m = (1.0, 0.0)
        total_poloidal_flux_wb = 2.0
        total_toroidal_flux_wb = 3.0

        r = np.linspace(0.5, 1.5, 31)
        z = np.linspace(-1.0, 1.0, 51)

        r_grid, z_grid = np.meshgrid(r, z, indexing="ij")
        rho_2d = np.sqrt(np.square(r_grid - 1.0) + np.square(z_grid)) / 0.4

        rho_poloidal_1d = np.linspace(0, 1, 41)
        rho_toroidal_1d = np.square(rho_poloidal_1d)

        f_toroidal_tm = np.linspace(0, 2, rho_poloidal_1d.size)
        cross_sectional_area_m2 = np.linspace(0, 3, rho_poloidal_1d.size)
        volume_m3 = np.linspace(0, 4, rho_poloidal_1d.size)
        fsa_1_over_r = np.linspace(0, 5, rho_poloidal_1d.size)
        fsa_1_over_r2 = np.linspace(0, 6, rho_poloidal_1d.size)
        fsa_b = np.linspace(0, 7, rho_poloidal_1d.size)
        b_max = np.linspace(0, 8, rho_poloidal_1d.size)
        trapped_particle_fraction = np.linspace(0, 9, rho_poloidal_1d.size)

        model_rho_poloidal_2d = (
            ValueModel.root_normalised_poloidal_flux().spline_2d(
                CoordinateSystem.CYLINDRICAL, r, z, rho_2d, (True, False, True)
            )
        )
        model_rho_poloidal_to_toroidal_1d = (
            ValueModel.root_normalised_toroidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal_1d,
                rho_toroidal_1d,
                (True,),
            )
        )
        model_rho_toroidal_to_poloidal_1d = (
            ValueModel.root_normalised_poloidal_flux_1d().spline_1d(
                CoordinateSystem.RHO_TOROIDAL,
                rho_toroidal_1d,
                rho_poloidal_1d,
                (True,),
            )
        )
        model_f_toroidal_tm = ValueModel.f_toroidal_tm().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal_1d,
            f_toroidal_tm,
            (True,),
        )
        model_cross_sectional_area_m2 = (
            ValueModel.cross_sectional_area_m2().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal_1d,
                cross_sectional_area_m2,
                (True,),
            )
        )
        model_volume_m3 = ValueModel.volume_m3().spline_1d(
            CoordinateSystem.RHO_POLOIDAL, rho_poloidal_1d, volume_m3, (True,)
        )
        model_fsa_1_over_r = ValueModel.fsa_1_over_r().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal_1d,
            fsa_1_over_r,
            (True,),
        )
        model_fsa_1_over_r2 = ValueModel.fsa_1_over_r2().spline_1d(
            CoordinateSystem.RHO_POLOIDAL,
            rho_poloidal_1d,
            fsa_1_over_r2,
            (True,),
        )
        model_fsa_b = ValueModel.fsa_b().spline_1d(
            CoordinateSystem.RHO_POLOIDAL, rho_poloidal_1d, fsa_b, (True,)
        )
        model_b_max = ValueModel.b_max().spline_1d(
            CoordinateSystem.RHO_POLOIDAL, rho_poloidal_1d, b_max, (True,)
        )
        model_trapped_particle_fraction = (
            ValueModel.trapped_particle_fraction().spline_1d(
                CoordinateSystem.RHO_POLOIDAL,
                rho_poloidal_1d,
                trapped_particle_fraction,
                (True,),
            )
        )

        magnetic = MagneticTokamak(
            model_rho_poloidal_2d,
            magnetic_axis_m,
            model_rho_poloidal_to_toroidal_1d,
            model_rho_toroidal_to_poloidal_1d,
            total_poloidal_flux_wb,
            total_toroidal_flux_wb,
            model_f_toroidal_tm,
            model_cross_sectional_area_m2,
            model_volume_m3,
            model_fsa_1_over_r,
            model_fsa_1_over_r2,
            model_fsa_b,
            model_b_max,
            model_trapped_particle_fraction,
            scale_factor_magnetic_field=scale_factor,
        )

        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)
            magnetic.write_netcdf(dset)
            magnetic_2 = MagneticTokamak.read_netcdf(
                dset,
                model_rho_poloidal_2d,
                magnetic_axis_m,
                model_rho_poloidal_to_toroidal_1d,
                model_rho_toroidal_to_poloidal_1d,
            )

        assert isinstance(magnetic_2.rho_poloidal_2d, Spline2D)
        nptest.assert_allclose(
            magnetic.rho_poloidal_2d._abscissas[0],
            magnetic_2.rho_poloidal_2d._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.rho_poloidal_2d._abscissas[1],
            magnetic_2.rho_poloidal_2d._abscissas[1],
        )
        nptest.assert_allclose(
            magnetic.rho_poloidal_2d._data, magnetic_2.rho_poloidal_2d._data
        )

        nptest.assert_allclose(
            magnetic.magnetic_axis_m, magnetic_2.magnetic_axis_m
        )
        nptest.assert_allclose(
            magnetic.total_poloidal_flux_wb, magnetic_2.total_poloidal_flux_wb
        )
        nptest.assert_allclose(
            magnetic.total_toroidal_flux_wb, magnetic_2.total_toroidal_flux_wb
        )

        assert isinstance(magnetic_2.rho_poloidal_to_toroidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._abscissas[0],
            magnetic_2.rho_poloidal_to_toroidal_1d._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._data,
            magnetic_2.rho_poloidal_to_toroidal_1d._data,
        )

        assert isinstance(magnetic_2.rho_toroidal_to_poloidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._abscissas[0],
            magnetic_2.rho_toroidal_to_poloidal_1d._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._data,
            magnetic_2.rho_toroidal_to_poloidal_1d._data,
        )

        assert isinstance(magnetic_2.f_toroidal_tm, Spline1D)
        nptest.assert_allclose(
            magnetic.f_toroidal_tm._abscissas[0],
            magnetic_2.f_toroidal_tm._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.f_toroidal_tm._data, magnetic_2.f_toroidal_tm._data
        )

        assert isinstance(magnetic_2.cross_sectional_area_m2, Spline1D)
        nptest.assert_allclose(
            magnetic.cross_sectional_area_m2._abscissas[0],
            magnetic_2.cross_sectional_area_m2._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.cross_sectional_area_m2._data,
            magnetic_2.cross_sectional_area_m2._data,
        )

        assert isinstance(magnetic_2.volume_m3, Spline1D)
        nptest.assert_allclose(
            magnetic.volume_m3._abscissas[0],
            magnetic_2.volume_m3._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.volume_m3._data, magnetic_2.volume_m3._data
        )

        assert isinstance(magnetic_2.fsa_1_over_r, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_1_over_r._abscissas[0],
            magnetic_2.fsa_1_over_r._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.fsa_1_over_r._data, magnetic_2.fsa_1_over_r._data
        )

        assert isinstance(magnetic_2.fsa_1_over_r2, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_1_over_r2._abscissas[0],
            magnetic_2.fsa_1_over_r2._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.fsa_1_over_r2._data, magnetic_2.fsa_1_over_r2._data
        )

        assert isinstance(magnetic_2.fsa_b, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_b._abscissas[0], magnetic_2.fsa_b._abscissas[0]
        )
        nptest.assert_allclose(magnetic.fsa_b._data, magnetic_2.fsa_b._data)

        assert isinstance(magnetic_2.b_max, Spline1D)
        nptest.assert_allclose(
            magnetic.b_max._abscissas[0], magnetic_2.b_max._abscissas[0]
        )
        nptest.assert_allclose(magnetic.b_max._data, magnetic_2.b_max._data)

        assert isinstance(magnetic_2.trapped_particle_fraction, Spline1D)
        nptest.assert_allclose(
            magnetic.trapped_particle_fraction._abscissas[0],
            magnetic_2.trapped_particle_fraction._abscissas[0],
        )
        nptest.assert_allclose(
            magnetic.trapped_particle_fraction._data,
            magnetic_2.trapped_particle_fraction._data,
        )


class TestLimiters:
    """
    Unit tests for Limiters.
    """

    @staticmethod
    @pytest.fixture
    def limiters() -> Limiters:
        """
        Test limiters.

        Returns
        -------
        limiters : Limiters
            Limiters.
        """
        return Limiters({
            "limiter_1": Limiter(
                [
                    Plane.xy([0.0, 0.0], [1.0, 0.0], LimiterEffect.STOP),
                    Plane.xy(
                        [1.0, 0.0], [0.0, 1.0], LimiterEffect.REFLECT_SPECULAR
                    ),
                    Plane.xy([1.0, 1.0], [-1.0, 0.0], LimiterEffect.STOP),
                    Plane.xy([0.0, 1.0], [0.0, -1.0], LimiterEffect.STOP),
                ],
                0.1,
            ),
            "limiter_2": Limiter(
                [
                    Disk(-2.0, 1.0, 2.0, LimiterEffect.STOP),
                    Cylinder(2.0, -2.0, -1.0, LimiterEffect.STOP),
                    Disk(-1.0, 1.0, 2.0, LimiterEffect.STOP),
                    Cylinder(1.0, -2.0, -1.0, LimiterEffect.STOP),
                ],
                0.2,
            ),
        })

    @staticmethod
    def test_round_trip_netcdf(limiters: Limiters):
        """
        Test serialising and deserialising to netCDF4 file gives same object.

        Parameters
        ----------
        limiters : Limiters
            Test limiters.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions().write_netcdf(dset)
            limiters.write_netcdf(dset)
            limiters_2 = Limiters.read_netcdf(dset)

        assert len(limiters.limiters) == len(limiters_2.limiters)

        for name, limiter_1 in limiters.limiters.items():
            limiter_2 = limiters_2.limiters[name]

            assert type(limiter_1) is type(limiter_2)
            assert (
                limiter_1.extinction_coefficient_nepers
                == limiter_2.extinction_coefficient_nepers
            )

    @staticmethod
    def test_intersection(limiters: Limiters):
        """
        Test intersection with limiters.

        Parameters
        ----------
        limiters : Limiters
            Test limiters.
        """
        ray_origin = [0.9, 0.0, 0.5]
        ray_direction = [0.4, 0.0, 0.0]
        intersects, name, idx, s, normal, effect, extinction_coefficient = (
            limiters.intersects(ray_origin, ray_direction)
        )

        assert intersects
        assert name == "limiter_1"
        assert idx == 1
        nptest.assert_allclose(s, 0.25)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])
        assert effect == LimiterEffect.REFLECT_SPECULAR
        nptest.assert_allclose(extinction_coefficient, 0.1)

        # Test ignoring element.
        intersects, *_ = limiters.intersects(
            ray_origin, ray_direction, ignore=(name, idx)
        )

        assert not intersects

    @staticmethod
    def test_add_from_imas_bounding_box():
        """
        Test adding limiter from bounding box of IMAS equilibrium data.
        """
        ids_equilibrium = imas.equilibrium()
        ids_equilibrium.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        ids_equilibrium.time.resize(1, refcheck=False)
        ids_equilibrium.time[0] = 1.0
        ids_equilibrium.time_slice.resize(1)

        _profiles_2d = ids_equilibrium.time_slice[0].profiles_2d
        _profiles_2d.resize(1)
        _profiles_2d[0].grid_type.index = 1
        _profiles_2d[0].grid.dim1 = [0.0, 1.0]
        _profiles_2d[0].grid.dim2 = [-1.0, 1.0]

        ids_equilibrium.validate()

        limiters = Limiters({})
        limiters.add_from_imas_bounding_box(
            "test", ids_equilibrium, 0, LimiterEffect.STOP, 0.1
        )

        limiter = limiters.limiters["test"]
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        for element in limiter.elements:
            assert element.effect == LimiterEffect.STOP

    @staticmethod
    def test_add_from_imas_2d():
        """
        Test adding limiter from IMAS wall.wall_2d.
        """
        ids_wall = imas.wall()
        ids_wall.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_INDEPENDENT
        )

        ids_wall.description_2d.resize(1)

        theta = np.linspace(0.0, 2 * np.pi, 11)
        r = 1.0 + 0.2 * np.cos(theta)
        z = 0.2 * np.sin(theta)

        _units = ids_wall.description_2d[0].limiter.unit
        _units.resize(1)
        _units[0].outline.r = r
        _units[0].outline.z = z

        ids_wall.validate()

        limiters = Limiters({})
        limiters.add_from_imas_2d("test", ids_wall, LimiterEffect.STOP, 0.1)

        limiter = limiters.limiters["test"]
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        for element in limiter.elements:
            assert element.effect == LimiterEffect.STOP

    @staticmethod
    @pytest.mark.skip("todo")
    def test_add_from_imas_3d():
        """
        Test adding limiter from IMAS wall.wall_3d.
        """

    @staticmethod
    def test_add_bounding_box_2d():
        """
        Test adding limiter from analytic 2D bounding box description.
        """
        limiters = Limiters({})

        limiters.add_bounding_box_2d(
            "test_xy",
            LimiterAnalyticBoundingBox2D.XY,
            [0.0, 0.1],
            [-1.0, 1.0],
            LimiterEffect.STOP,
            0.1,
        )
        limiters.add_bounding_box_2d(
            "test_xz",
            LimiterAnalyticBoundingBox2D.XZ,
            [0.0, 0.2],
            [-2.0, 2.0],
            LimiterEffect.STOP,
            0.1,
        )
        limiters.add_bounding_box_2d(
            "test_yz",
            LimiterAnalyticBoundingBox2D.YZ,
            [0.0, 0.3],
            [-3.0, 3.0],
            LimiterEffect.STOP,
            0.1,
        )
        limiters.add_bounding_box_2d(
            "test_rz",
            LimiterAnalyticBoundingBox2D.RZ,
            [0.0, 0.3],
            [-3.0, 3.0],
            LimiterEffect.STOP,
            0.1,
        )

        limiter = limiters.limiters["test_xy"]
        assert isinstance(limiter.elements[0], Plane)
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        limiter = limiters.limiters["test_xz"]
        assert isinstance(limiter.elements[0], Plane)
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        limiter = limiters.limiters["test_yz"]
        assert isinstance(limiter.elements[0], Plane)
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        limiter = limiters.limiters["test_rz"]
        assert isinstance(limiter.elements[0], (Disk, Cylinder))
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

    @staticmethod
    def test_add_bounding_box_3d():
        """
        Test adding limiter from analytic 3D bounding box description.
        """
        limiters = Limiters({})

        limiters.add_bounding_box_3d(
            "test_xyz",
            LimiterAnalyticBoundingBox3D.XYZ,
            [-0.1, 0.1],
            [-0.2, 0.2],
            [-0.3, 0.3],
            LimiterEffect.STOP,
            0.1,
        )

        limiter = limiters.limiters["test_xyz"]
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        for element in limiter.elements:
            assert element.effect == LimiterEffect.STOP


class TestSystemData:
    """
    Unit tests for SystemData.
    """

    @pytest.fixture
    @staticmethod
    def system_data() -> SystemData:
        """
        System data object.

        Returns
        -------
        system_data : SystemData
            System data.
        """
        # Coordinates.
        coordinate_coordinator = CoordinateCoordinator()

        r = np.linspace(1.0, 3.0, 61)
        ne = 2.0 * r
        te = 3.0 * r
        zeff = 4.0 * r

        kinetic = Kinetic(
            ValueModel.electron_density_per_m3().spline_1d(
                CoordinateSystem.CARTESIAN, r, ne, (True, False, False)
            ),
            ValueModel.electron_temperature_ev().spline_1d(
                CoordinateSystem.CARTESIAN, r, te, (True, False, False)
            ),
            ValueModel.effective_charge().spline_1d(
                CoordinateSystem.CARTESIAN, r, zeff, (True, False, False)
            ),
        )

        magnetic = Magnetic(
            ValueModel.magnetic_field_t().constant(
                CoordinateSystem.CARTESIAN,
                [1.0, 0.0, 0.0],
            )
        )

        limiters = Limiters({})
        limiters.add_bounding_box_2d(
            "test", "xy", [0.0, 1.0], [-1.0, 1.0], LimiterEffect.STOP, 0.1
        )

        return SystemData(coordinate_coordinator, kinetic, magnetic, limiters)

    @staticmethod
    def test_round_trip_netcdf(system_data: SystemData):
        """
        Test serialising and deserialising to netCDF4 file gives same object.

        Parameters
        ----------
        system_data : SystemData
            System data.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions().write_netcdf(dset)
            system_data.write_netcdf(dset)
            system_data_2 = SystemData.read_netcdf(dset)

        kinetic, kinetic_2 = system_data.kinetic, system_data_2.kinetic

        nptest.assert_allclose(
            kinetic.electron_density_per_m3._abscissas,
            kinetic_2.electron_density_per_m3._abscissas,
        )
        nptest.assert_allclose(
            kinetic.electron_density_per_m3._data,
            kinetic_2.electron_density_per_m3._data,
        )

        nptest.assert_allclose(
            kinetic.electron_temperature_ev._abscissas,
            kinetic_2.electron_temperature_ev._abscissas,
        )
        nptest.assert_allclose(
            kinetic.electron_temperature_ev._data,
            kinetic_2.electron_temperature_ev._data,
        )

        nptest.assert_allclose(
            kinetic.effective_charge._abscissas,
            kinetic_2.effective_charge._abscissas,
        )
        nptest.assert_allclose(
            kinetic.effective_charge._data,
            kinetic_2.effective_charge._data,
        )

        magnetic, magnetic_2 = system_data.magnetic, system_data_2.magnetic

        nptest.assert_allclose(
            magnetic.magnetic_field_t.constant_value,
            magnetic_2.magnetic_field_t.constant_value,
        )


class TestSystemDataProvider:
    """
    Unit tests for SystemDataProvider.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test serialising and deserialising to netCDF4 file gives same object.
        """
        sdp = SystemDataProvider(
            {
                "imas_test": DataSourceImas("/some/uri", 0, 0, 0),
                "netcdf_test": DataSourceNetcdf("/some/netcdf/file"),
            },
            {CoordinateSystem.TOROIDAL: CoordinateToroidal(1.0, 2.0)},
            ModelImas("imas_test", 1.0),
            ModelNetcdf("netcdf_test", 1.0),
            ModelAnalyticConstant(CoordinateSystem.CARTESIAN, 2.0, 1.0),
            MagneticModelTokamak("imas_test", 1.0),
            {
                "limiter_test": LimiterImas2D(
                    LimiterEffect.STOP, "imas_test", 0.0
                )
            },
        )

        document = sdp.to_dict_toml()
        sdp_2 = SystemDataProvider.from_dict_toml(document)

        # Check data sources.
        data_sources = sdp.data_sources
        data_sources_2 = sdp_2.data_sources

        assert data_sources["imas_test"].uri == data_sources_2["imas_test"].uri
        assert (
            data_sources["imas_test"].occurrence_equilibrium
            == data_sources_2["imas_test"].occurrence_equilibrium
        )
        assert (
            data_sources["imas_test"].occurrence_core_profiles
            == data_sources_2["imas_test"].occurrence_core_profiles
        )
        assert (
            data_sources["imas_test"].occurrence_wall
            == data_sources_2["imas_test"].occurrence_wall
        )

        assert (
            data_sources["netcdf_test"].filepath
            == data_sources_2["netcdf_test"].filepath
        )

        # Check coordinates.
        coordinates = sdp.coordinates
        coordinates_2 = sdp_2.coordinates

        assert (
            coordinates[CoordinateSystem.TOROIDAL].r0
            == coordinates_2[CoordinateSystem.TOROIDAL].r0
        )
        assert (
            coordinates[CoordinateSystem.TOROIDAL].z0
            == coordinates_2[CoordinateSystem.TOROIDAL].z0
        )

        # Check kinetic models.
        assert type(sdp_2.electron_density_per_m3) is ModelImas
        assert (
            sdp.electron_density_per_m3.source
            == sdp_2.electron_density_per_m3.source
        )

        assert type(sdp_2.electron_temperature_ev) is ModelNetcdf
        assert (
            sdp.electron_temperature_ev.source
            == sdp_2.electron_temperature_ev.source
        )

        assert type(sdp_2.effective_charge) is ModelAnalyticConstant
        assert (
            sdp.effective_charge.coordinate_system
            == sdp_2.effective_charge.coordinate_system
        )

        # Check magnetic.
        assert type(sdp_2.magnetic_field_t) is MagneticModelTokamak
        assert sdp.magnetic_field_t.source == sdp_2.magnetic_field_t.source

        # Check limiters.
        limiters = sdp.limiters
        limiters_2 = sdp_2.limiters

        assert type(limiters_2["limiter_test"]) is LimiterImas2D
        assert limiters_2["limiter_test"].effect == LimiterEffect.STOP
        assert (
            limiters["limiter_test"].source
            == limiters_2["limiter_test"].source
        )

    imas_test = "imas_test"
    netcdf_test = "netcdf_test"
    vmec_test = "vmec_test"

    @pytest.fixture
    def system_data_provider(self) -> SystemDataProvider:
        """
        System data provider.

        Returns
        -------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        data_sources = {
            self.imas_test: DataSourceImas("/imas/database", 1, 2, 3),
            self.netcdf_test: DataSourceNetcdf("/netcdf/file"),
            self.vmec_test: DataSourceVmec("/vmec/file"),
        }
        coordinates = {CoordinateSystem.TOROIDAL: CoordinateToroidal(1.0, 0.6)}
        electron_density_per_m3 = ModelImas(self.imas_test, 1.0)
        electron_temperature_ev = ModelImas(self.imas_test, 1.0)
        effective_charge = ModelImas(self.imas_test, 1.0)
        magnetic_field_t = ModelImas(self.imas_test, 1.0)

        limiters = {
            "imas_bounding_box": LimiterImasBoundingBox2D(
                LimiterEffect.STOP, self.imas_test, 0.0
            ),
            "imas_2d": LimiterImas2D(LimiterEffect.STOP, self.imas_test, 0.0),
            "bounding_box_2d_xy": LimiterAnalyticBoundingBox2D(
                LimiterEffect.STOP,
                LimiterAnalyticBoundingBox2D.XY,
                [-1.0, 1.0],
                [-2.0, 2.0],
                0.0,
            ),
            "bounding_box_2d_rz": LimiterAnalyticBoundingBox2D(
                LimiterEffect.STOP,
                LimiterAnalyticBoundingBox2D.RZ,
                [-1.0, 1.0],
                [-2.0, 2.0],
                0.0,
            ),
            "bounding_box_3d": LimiterAnalyticBoundingBox3D(
                LimiterEffect.STOP,
                LimiterAnalyticBoundingBox3D.XYZ,
                [-1.0, 1.0],
                [-2.0, 2.0],
                [-3.0, 3.0],
                0.0,
            ),
        }

        return SystemDataProvider(
            data_sources,
            coordinates,
            electron_density_per_m3,
            electron_temperature_ev,
            effective_charge,
            magnetic_field_t,
            limiters,
        )

    @staticmethod
    def test_get_coordinate_coordinator(
        system_data_provider: SystemDataProvider,
    ):
        """
        Test constructing coordinate coordinator.

        Parameters
        ----------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        cc = system_data_provider.get_coordinate_coordinator()

        assert CoordinateSystem.TOROIDAL in cc.coordinates
        nptest.assert_allclose(
            cc.coordinates[CoordinateSystem.TOROIDAL].axis_m, (1.0, 0.6)
        )

    @staticmethod
    def test_get_analytic_model(system_data_provider: SystemDataProvider):
        """
        Test constructing analytic model.

        Parameters
        ----------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        coordinate_system = CoordinateSystem.CARTESIAN
        constant_value = 1.1
        origin = [0.0, 0.1, 0.2]
        direction = [0.36, 0.48, 0.8]
        y0 = 0.4
        y1 = 0.9
        ramp_width = 1.5
        smoothness = 1
        scale_factor = 1.2

        # Test constant.
        system_data_provider.electron_density_per_m3 = ModelAnalyticConstant(
            coordinate_system, constant_value, scale_factor
        )

        model = system_data_provider.get_kinetic_model(
            0.0, ELECTRON_DENSITY_PER_M3
        )

        assert isinstance(model, Constant)
        assert model.coordinate_system == coordinate_system
        nptest.assert_allclose(model.constant_value, constant_value)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test ramp.
        system_data_provider.electron_density_per_m3 = ModelAnalyticRamp(
            coordinate_system,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            smoothness,
            scale_factor,
        )

        model = system_data_provider.get_kinetic_model(
            0.0, ELECTRON_DENSITY_PER_M3
        )

        assert isinstance(model, C1Ramp)
        assert model.coordinate_system == coordinate_system
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.direction, direction)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test quadratic well.
        system_data_provider.electron_density_per_m3 = (
            ModelAnalyticQuadraticWell(
                origin, y0, y1, ramp_width, scale_factor
            )
        )

        model = system_data_provider.get_kinetic_model(
            0.0, ELECTRON_DENSITY_PER_M3
        )

        assert isinstance(model, QuadraticWell)
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)

        # Test quadratic channel.
        system_data_provider.electron_density_per_m3 = (
            ModelAnalyticQuadraticChannel(
                origin, direction, y0, y1, ramp_width, scale_factor
            )
        )

        model = system_data_provider.get_kinetic_model(
            0.0, ELECTRON_DENSITY_PER_M3
        )

        assert isinstance(model, QuadraticChannel)
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.direction, direction)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)

        # Test quadratic bowl.
        system_data_provider.electron_density_per_m3 = (
            ModelAnalyticQuadraticBowl(
                origin, direction, y0, y1, ramp_width, scale_factor
            )
        )

        model = system_data_provider.get_kinetic_model(
            0.0, ELECTRON_DENSITY_PER_M3
        )

        assert isinstance(model, QuadraticBowl)
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.direction, direction)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)

    def test_get_kinetic_model_imas(
        self, system_data_provider: SystemDataProvider
    ):
        """
        Test loading kinetic model from IMAS.

        Parameters
        ----------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        # Mock IMAS data.
        core_profiles = imas.core_profiles()
        core_profiles.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        core_profiles.time.resize(2, refcheck=False)
        core_profiles.time = np.array([0.0, 1.0])

        core_profiles.profiles_1d.resize(2)

        rho_poloidal = np.linspace(0, 1, 51)
        rho_toroidal = np.zeros_like(rho_poloidal)
        ne_1 = np.linspace(0, 2, rho_poloidal.size)
        ne_2 = np.linspace(0, 3, rho_poloidal.size)
        te_1 = np.linspace(0, 4, rho_poloidal.size)
        zeff_1 = np.linspace(0, 5, rho_poloidal.size)

        core_profiles.profiles_1d[0].grid.rho_pol_norm = rho_poloidal
        core_profiles.profiles_1d[0].grid.rho_tor_norm = rho_toroidal
        core_profiles.profiles_1d[0].electrons.density = ne_1
        core_profiles.profiles_1d[0].electrons.temperature = te_1
        core_profiles.profiles_1d[0].zeff = zeff_1

        core_profiles.profiles_1d[1].grid.rho_pol_norm = rho_poloidal
        core_profiles.profiles_1d[1].grid.rho_tor_norm = rho_toroidal
        core_profiles.profiles_1d[1].electrons.density = ne_2

        core_profiles.validate()

        with tempfile.TemporaryDirectory() as tmpdir:
            dbentry = DBEntry(f"imas:hdf5?path={tmpdir}", "a")
            dbentry.put(
                core_profiles,
                occurrence=(
                    system_data_provider.data_sources[
                        self.imas_test
                    ].occurrence_core_profiles
                ),
            )

            # Mock opening database.
            system_data_provider._data_source_handles[self.imas_test] = dbentry

            # Load density.
            model = system_data_provider.get_kinetic_model(
                0.0, ELECTRON_DENSITY_PER_M3
            )

            assert isinstance(model, Spline1D)
            nptest.assert_allclose(model._abscissas[0], rho_poloidal)
            nptest.assert_allclose(model._data, ne_1)

            model = system_data_provider.get_kinetic_model(
                1.0, ELECTRON_DENSITY_PER_M3
            )

            assert isinstance(model, Spline1D)
            nptest.assert_allclose(model._abscissas[0], rho_poloidal)
            nptest.assert_allclose(model._data, ne_2)

            # Load temperature.
            model = system_data_provider.get_kinetic_model(
                0.0, ELECTRON_TEMPERATURE_EV
            )

            assert isinstance(model, Spline1D)
            nptest.assert_allclose(model._abscissas[0], rho_poloidal)
            nptest.assert_allclose(model._data, te_1)

            # Load effective charge.
            model = system_data_provider.get_kinetic_model(
                0.0, EFFECTIVE_CHARGE
            )

            assert isinstance(model, Spline1D)
            nptest.assert_allclose(model._abscissas[0], rho_poloidal)
            nptest.assert_allclose(model._data, zeff_1)

    @pytest.mark.skip
    @staticmethod
    def test_get_kinetic_model_netcdf(
        system_data_provider: SystemDataProvider,
    ):
        """
        Test loading kinetic model from netCDF.

        Parameters
        ----------
        system_data_provider : SystemDataProvider
            System data provider.
        """

    def test_get_magnetic_tokamak_imas(
        self, system_data_provider: SystemDataProvider
    ):
        """
        Test loading magnetic model from netCDF.

        Parameters
        ----------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        system_data_provider.magnetic_field_t = MagneticModelTokamak(
            "imas_test", 1.2
        )

        # Create IDS.
        equilibrium = imas.equilibrium()
        equilibrium.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        equilibrium.time.resize(1, refcheck=False)
        equilibrium.time[0] = 1.0
        equilibrium.time_slice.resize(1)

        rho_poloidal = np.linspace(0, 1, 51)
        rho_toroidal = np.square(rho_poloidal)
        f_toroidal = 2.0 * rho_poloidal

        r = np.linspace(1.0, 3.0, 21)
        z = np.linspace(0.0, 2.0, 31)
        r_grid, z_grid = np.meshgrid(r, z, indexing="ij")

        r0, z0 = 2.0, 1.0
        psi = np.square(r_grid - r0) + np.square(z_grid - z0)
        psi_axis, psi_sep = 0.0, 0.5**2
        total_toroidal_flux = 2.0
        rho_poloidal_2d = np.sqrt(psi / psi_sep)

        _profiles_1d = equilibrium.time_slice[0].profiles_1d
        _profiles_1d.psi = np.zeros_like(rho_poloidal)
        _profiles_1d.f = f_toroidal
        _profiles_1d.psi_norm = np.square(rho_poloidal)
        _profiles_1d.phi = total_toroidal_flux * np.square(rho_toroidal)
        _profiles_1d.rho_tor_norm = rho_toroidal

        cross_sectional_area_m2 = np.linspace(0, 2, rho_poloidal.size)
        volume_m3 = np.linspace(0, 3, rho_poloidal.size)
        fsa_1_over_r = np.linspace(0, 4, rho_poloidal.size)
        fsa_1_over_r2 = np.linspace(0, 5, rho_poloidal.size)
        fsa_b = np.linspace(0, 6, rho_poloidal.size)
        b_max = np.linspace(0, 7, rho_poloidal.size)
        trapped_particle_fraction = np.linspace(0, 8, rho_poloidal.size)

        _profiles_1d.area = cross_sectional_area_m2
        _profiles_1d.volume = volume_m3
        _profiles_1d.gm9 = fsa_1_over_r
        _profiles_1d.gm1 = fsa_1_over_r2
        _profiles_1d.b_field_average = fsa_b
        _profiles_1d.b_field_max = b_max
        _profiles_1d.trapped_fraction = trapped_particle_fraction

        _profiles_2d = equilibrium.time_slice[0].profiles_2d
        _profiles_2d.resize(1)
        _profiles_2d[0].grid_type.index = 1
        _profiles_2d[0].grid.dim1 = r
        _profiles_2d[0].grid.dim2 = z
        _profiles_2d[0].psi = psi

        _globals = equilibrium.time_slice[0].global_quantities
        _globals.psi_axis = psi_axis
        _globals.psi_boundary = psi_sep
        _globals.magnetic_axis.r = r0
        _globals.magnetic_axis.z = z0

        equilibrium.vacuum_toroidal_field.r0 = 1.0
        equilibrium.vacuum_toroidal_field.b0 = 2.0 * np.ones(1)

        equilibrium.validate()

        # Mock opening database.
        with tempfile.TemporaryDirectory() as tmpdir:
            dbentry = DBEntry(f"imas:hdf5?path={tmpdir}", "a")
            dbentry.put(
                equilibrium,
                occurrence=(
                    system_data_provider.data_sources[
                        self.imas_test
                    ].occurrence_equilibrium
                ),
            )

            # Mock opening database.
            system_data_provider._data_source_handles[self.imas_test] = dbentry

            # Load magnetic.
            magnetic = system_data_provider.get_magnetic(0.0)

        assert isinstance(magnetic, MagneticTokamak)

        assert isinstance(magnetic.rho_poloidal_2d, Spline2D)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._abscissas[0], r)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._abscissas[1], z)
        nptest.assert_allclose(magnetic.rho_poloidal_2d._data, rho_poloidal_2d)
        nptest.assert_allclose(magnetic.magnetic_axis_m, (r0, z0))
        nptest.assert_allclose(magnetic.total_poloidal_flux_wb, psi_sep)

        assert isinstance(magnetic.rho_poloidal_to_toroidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.rho_poloidal_to_toroidal_1d._data, rho_toroidal
        )

        assert isinstance(magnetic.rho_toroidal_to_poloidal_1d, Spline1D)
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._abscissas[0], rho_toroidal
        )
        nptest.assert_allclose(
            magnetic.rho_toroidal_to_poloidal_1d._data, rho_poloidal
        )

        nptest.assert_allclose(
            magnetic.total_toroidal_flux_wb, total_toroidal_flux
        )

        assert isinstance(magnetic.f_toroidal_tm, Spline1D)
        nptest.assert_allclose(
            magnetic.f_toroidal_tm._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.f_toroidal_tm._data, f_toroidal)

        assert isinstance(magnetic.cross_sectional_area_m2, Spline1D)
        nptest.assert_allclose(
            magnetic.cross_sectional_area_m2._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.cross_sectional_area_m2._data, cross_sectional_area_m2
        )

        assert isinstance(magnetic.volume_m3, Spline1D)
        nptest.assert_allclose(magnetic.volume_m3._abscissas[0], rho_poloidal)
        nptest.assert_allclose(magnetic.volume_m3._data, volume_m3)

        assert isinstance(magnetic.fsa_1_over_r, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_1_over_r._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.fsa_1_over_r._data, fsa_1_over_r)

        assert isinstance(magnetic.fsa_1_over_r2, Spline1D)
        nptest.assert_allclose(
            magnetic.fsa_1_over_r2._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(magnetic.fsa_1_over_r2._data, fsa_1_over_r2)

        assert isinstance(magnetic.fsa_b, Spline1D)
        nptest.assert_allclose(magnetic.fsa_b._abscissas[0], rho_poloidal)
        nptest.assert_allclose(magnetic.fsa_b._data, fsa_b)

        assert isinstance(magnetic.b_max, Spline1D)
        nptest.assert_allclose(magnetic.b_max._abscissas[0], rho_poloidal)
        nptest.assert_allclose(magnetic.b_max._data, b_max)

        assert isinstance(magnetic.trapped_particle_fraction, Spline1D)
        nptest.assert_allclose(
            magnetic.trapped_particle_fraction._abscissas[0], rho_poloidal
        )
        nptest.assert_allclose(
            magnetic.trapped_particle_fraction._data, trapped_particle_fraction
        )

        assert isinstance(magnetic.magnetic_field_t, AxisymmetricMagneticField)
        nptest.assert_allclose(magnetic.magnetic_field_t.scale_factor, 1.2)

    def test_get_limiters(self, system_data_provider: SystemDataProvider):
        """
        Test loading magnetic model from netCDF.

        Parameters
        ----------
        system_data_provider : SystemDataProvider
            System data provider.
        """
        ids_equilibrium = imas.equilibrium()
        ids_equilibrium.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_HOMOGENEOUS
        )
        ids_equilibrium.time.resize(1, refcheck=False)
        ids_equilibrium.time[0] = 1.0

        ids_equilibrium.time_slice.resize(1)
        _profiles_2d = ids_equilibrium.time_slice[0].profiles_2d
        _profiles_2d.resize(1)
        _profiles_2d[0].grid_type.index = 1
        _profiles_2d[0].grid.dim1 = np.array([0.5, 2.5])
        _profiles_2d[0].grid.dim2 = np.array([-1.5, 1.5])

        ids_equilibrium.validate()

        ids_wall = imas.wall()
        ids_wall.ids_properties.homogeneous_time = (
            imasdef.IDS_TIME_MODE_INDEPENDENT
        )

        ids_wall.description_2d.resize(1)

        theta = np.linspace(0.0, 2 * np.pi, 11)
        r = 1.0 + 0.2 * np.cos(theta)
        z = 0.2 * np.sin(theta)

        _units = ids_wall.description_2d[0].limiter.unit
        _units.resize(1)
        _units[0].outline.r = r
        _units[0].outline.z = z

        ids_wall.validate()

        # Mock opening data sources.
        with tempfile.TemporaryDirectory() as tmpdir:
            dbentry = DBEntry(f"imas:hdf5?path={tmpdir}", "a")
            dbentry.put(
                ids_equilibrium,
                occurrence=(
                    system_data_provider.data_sources[
                        self.imas_test
                    ].occurrence_equilibrium
                ),
            )
            dbentry.put(
                ids_wall,
                occurrence=(
                    system_data_provider.data_sources[
                        self.imas_test
                    ].occurrence_wall
                ),
            )

            # Mock opening database.
            system_data_provider._data_source_handles[self.imas_test] = dbentry

            # Load magnetic.
            limiters = system_data_provider.get_limiters(0.0)

        assert isinstance(limiters, Limiters)
