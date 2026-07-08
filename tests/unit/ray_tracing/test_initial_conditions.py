"""
Unit tests for ray_tracing.initial_conditions.
"""

# Standard imports
import logging
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import CoordinateCoordinator
from crayon.ray_tracing.caches import CoordinateCache, PlasmaCache
from crayon.ray_tracing.initial_conditions import (
    InitialConditions,
    InitialConditionsSchema,
    PolarisationEllipseAngles,
    PolarisationWaveMode,
    RefractiveIndexComponents,
    RefractiveIndexLaunchAnglesGeometric,
    RefractiveIndexLaunchAnglesImas,
    RefractiveIndexNparallel,
    RefractiveIndexOptimal,
    read_initial_conditions_toml,
    write_initial_conditions_toml,
)
from crayon.shared.constants import CoordinateSystem, WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.system_data import Kinetic, Magnetic
from crayon.value_model.models import ValueModel

logger = logging.getLogger(__name__)


class TestInitialConditions:
    """
    Unit tests for InitialConditions.
    """

    @staticmethod
    def test_round_trip_netcdf():
        """
        Test writing and reading object through netCDF4 file gives same object.
        """
        initial_conditions = InitialConditions(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            9.0,
            10.0,
            [11.0, 12.0, 13.0],
            WaveMode.O,
            14.0,
            15.0,
            16.0,
            bundle=True,
        )

        with (
            tempfile.TemporaryFile("r+") as f,
            nc4.Dataset(f, "w", auto_complex=True) as dset,
        ):
            Dimensions().write_netcdf(dset)
            group = dset.createGroup("test")
            initial_conditions.write_netcdf(group)
            initial_conditions_2 = InitialConditions.read_netcdf(group)

        ic, ic2 = initial_conditions, initial_conditions_2
        assert ic.name == ic2.name
        assert ic.time_ns == ic2.time_ns
        assert ic.frequency_ghz == ic2.frequency_ghz
        nptest.assert_allclose(ic.position_cartesian, ic2.position_cartesian)
        nptest.assert_allclose(
            ic.refractive_index_cartesian, ic2.refractive_index_cartesian
        )
        assert ic.eikonal_phase_rad == ic2.eikonal_phase_rad
        assert ic.adiabatic_phase_rad == ic2.adiabatic_phase_rad
        nptest.assert_allclose(ic.polarisation_stix, ic2.polarisation_stix)
        assert ic.wave_mode == ic2.wave_mode
        assert ic.power_w == ic2.power_w
        assert ic.intensity_w_per_m2 == ic2.intensity_w_per_m2
        assert ic.beam_waist_radius_m == ic2.beam_waist_radius_m
        assert ic.bundle == ic2.bundle


@pytest.fixture(scope="module")
def coordinate_coordinator() -> CoordinateCoordinator:
    """
    Coordinate coordinator holding coordinate system information.

    Returns
    -------
    coordinate_coordinator : CoordinateCoordinator
        Coordinate coordinator.
    """
    cc = CoordinateCoordinator()
    cc.calculate_conversion_paths()
    return cc


@pytest.fixture
def coordinate_cache(
    coordinate_coordinator: CoordinateCoordinator,
) -> CoordinateCache:
    """
    Cache holding coordinate system data.

    Returns
    -------
    coordinate_cache : CoordinateCache
        Coordinate cache.
    """
    return CoordinateCache(coordinate_coordinator)


@pytest.fixture
def plasma_cache() -> PlasmaCache:
    """
    Plasma parameter cache.

    Returns
    -------
    plasma_cache : PlasmaCache
        Plasma parameter cache.
    """
    electron_density_per_m3 = ValueModel.electron_density_per_m3().constant(
        CoordinateSystem.CARTESIAN, 0.0
    )
    electron_temperature_ev = ValueModel.electron_density_per_m3().constant(
        CoordinateSystem.CARTESIAN, 0.0
    )
    effective_charge = ValueModel.effective_charge().constant(
        CoordinateSystem.CARTESIAN, 0.0
    )
    magnetic_field_t = ValueModel.magnetic_field_t().constant(
        CoordinateSystem.CARTESIAN, np.array([0.0, 0.0, 1.0])
    )

    kinetic = Kinetic(
        electron_density_per_m3, electron_temperature_ev, effective_charge
    )
    magnetic = Magnetic(magnetic_field_t)

    return PlasmaCache(kinetic, magnetic)


class TestRefractiveIndexComponents:
    """
    Unit tests for TestRefractiveIndexComponents.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = RefractiveIndexComponents(
            [1.0, 2.0, 3.0], CoordinateSystem.CYLINDRICAL, holonomic=False
        )

        d = obj.to_dict_toml()
        obj2 = RefractiveIndexComponents.from_dict_toml(d)

        nptest.assert_allclose(obj.refractive_index, obj2.refractive_index)
        assert obj.coordinate_system == obj2.coordinate_system
        assert obj.holonomic == obj2.holonomic

    @staticmethod
    def test_unpack(coordinate_cache: CoordinateCache):
        """
        Test unpacking initial conditions.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Coordinate cache.
        """
        # Test Cartesian.
        coordinate_cache.set_position(
            CoordinateSystem.CARTESIAN, [1.0, 0.0, 0.0]
        )

        obj = RefractiveIndexComponents(
            [1.0, 2.0, 3.0], CoordinateSystem.CARTESIAN, holonomic=False
        )

        expected_value = [1.0, 2.0, 3.0]
        actual_value = obj.unpack(coordinate_cache)

        nptest.assert_allclose(expected_value, actual_value)

        obj = RefractiveIndexComponents(
            [1.0, 2.0, 3.0], CoordinateSystem.CARTESIAN, holonomic=True
        )

        expected_value = [1.0, 2.0, 3.0]
        actual_value = obj.unpack(coordinate_cache)

        nptest.assert_allclose(expected_value, actual_value)

        # Test cylindrical.
        coordinate_cache.set_position(
            CoordinateSystem.CYLINDRICAL, [1.2, 0.0, 0.0]
        )

        obj = RefractiveIndexComponents(
            [0.0, 1.0, 0.0], CoordinateSystem.CYLINDRICAL, holonomic=True
        )

        expected_value = [0.0, 1.0 / 1.2, 0.0]
        actual_value = obj.unpack(coordinate_cache)

        nptest.assert_allclose(expected_value, actual_value)

        obj = RefractiveIndexComponents(
            [0.0, 1.0, 0.0], CoordinateSystem.CYLINDRICAL, holonomic=False
        )

        expected_value = [0.0, 1.0, 0.0]
        actual_value = obj.unpack(coordinate_cache)

        nptest.assert_allclose(expected_value, actual_value)


class TestRefractiveIndexLaunchAnglesGeometric:
    """
    Unit tests for RefractiveIndexLaunchAnglesGeometric.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = RefractiveIndexLaunchAnglesGeometric(10.0, -20.0, radians=False)

        d = obj.to_dict_toml()
        obj2 = RefractiveIndexLaunchAnglesGeometric.from_dict_toml(
            d, radians=True
        )

        nptest.assert_allclose(obj.toroidal_angle_rad, np.pi / 18.0)
        nptest.assert_allclose(obj.poloidal_angle_rad, -np.pi / 9.0)
        assert obj.toroidal_angle_rad == obj2.toroidal_angle_rad
        assert obj.poloidal_angle_rad == obj2.poloidal_angle_rad

    @staticmethod
    def test_unpack(coordinate_cache: CoordinateCache):
        """
        Test unpacking initial conditions.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Coordinate cache.
        """
        obj = RefractiveIndexLaunchAnglesGeometric(10.0, -20.0, radians=False)

        coordinate_cache.set_position(
            CoordinateSystem.CYLINDRICAL, [1.2, 0.0, 0.0]
        )

        expected_value = [-0.92541658, 0.16317591, -0.34202014]
        actual_value = obj.unpack(coordinate_cache)

        nptest.assert_allclose(expected_value, actual_value)


class TestRefractiveIndexLaunchAnglesImas:
    """
    Unit tests for RefractiveIndexLaunchAnglesImas.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = RefractiveIndexLaunchAnglesImas(10.0, -20.0, radians=False)

        d = obj.to_dict_toml()
        obj2 = RefractiveIndexLaunchAnglesImas.from_dict_toml(d, radians=True)

        nptest.assert_allclose(obj.toroidal_angle_rad, np.pi / 18.0)
        nptest.assert_allclose(obj.poloidal_angle_rad, -np.pi / 9.0)
        assert obj.toroidal_angle_rad == obj2.toroidal_angle_rad
        assert obj.poloidal_angle_rad == obj2.poloidal_angle_rad

    @staticmethod
    def test_unpack(coordinate_cache: CoordinateCache):
        """
        Test unpacking initial conditions.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Coordinate cache.
        """
        obj = RefractiveIndexLaunchAnglesImas(10.0, -20.0, radians=False)

        coordinate_cache.set_position(
            CoordinateSystem.CYLINDRICAL, [1.2, 0.0, 0.0]
        )

        expected_value = [-0.92541658, 0.17364818, 0.33682409]
        actual_value = obj.unpack(coordinate_cache)

        nptest.assert_allclose(expected_value, actual_value)


class TestRefractiveIndexNparallel:
    """
    Unit tests for RefractiveIndexNparallel.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = RefractiveIndexNparallel(0.6, -20.0, radians=False)

        d = obj.to_dict_toml()
        obj2 = RefractiveIndexNparallel.from_dict_toml(d, radians=True)

        nptest.assert_allclose(obj.angle_perp_rad, -np.pi / 9.0)
        assert obj.n_parallel == obj2.n_parallel
        assert obj.angle_perp_rad == obj2.angle_perp_rad

    @staticmethod
    def test_unpack(
        coordinate_cache: CoordinateCache, plasma_cache: PlasmaCache
    ):
        """
        Test unpacking initial conditions.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Coordinate cache.
        plasma_cache : PlasmaCache
            Plasma cache.
        """
        obj = RefractiveIndexNparallel(0.6, -20.0, radians=False)

        coordinate_cache.set_position(
            CoordinateSystem.CARTESIAN, [1.0, 0.0, 0.0]
        )
        plasma_cache.set_frequency(1.0)
        plasma_cache.calculate(coordinate_cache, derivatives=0)

        expected_value = [0.75175410, -0.27361611, 0.6]
        actual_value = obj.unpack(plasma_cache)

        nptest.assert_allclose(expected_value, actual_value)


class TestRefractiveIndexOptimal:
    """
    Unit tests for RefractiveIndexOptimal.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = RefractiveIndexOptimal(n_parallel_positive=True)

        d = obj.to_dict_toml()
        obj2 = RefractiveIndexOptimal.from_dict_toml(d)

        assert obj.n_parallel_positive == obj2.n_parallel_positive

    @staticmethod
    def test_unpack():
        """
        Test unpacking initial conditions.
        """
        optimal_refractive_index = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]

        obj = RefractiveIndexOptimal(n_parallel_positive=True)
        expected_value = optimal_refractive_index[0]
        actual_value = obj.unpack(optimal_refractive_index)

        nptest.assert_allclose(expected_value, actual_value)

        obj = RefractiveIndexOptimal(n_parallel_positive=False)
        expected_value = optimal_refractive_index[1]
        actual_value = obj.unpack(optimal_refractive_index)

        nptest.assert_allclose(expected_value, actual_value)


class TestPolarisationWaveMode:
    """
    Unit tests for PolarisationWaveMode.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = PolarisationWaveMode(WaveMode.O)

        d = obj.to_dict_toml()
        obj2 = PolarisationWaveMode.from_dict_toml(d)

        assert obj.wave_mode == obj2.wave_mode


class TestPolarisationEllipseAngles:
    """
    Unit tests for PolarisationEllipseAngles.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        obj = PolarisationEllipseAngles(10.0, -20.0, radians=False)

        d = obj.to_dict_toml()
        obj2 = PolarisationEllipseAngles.from_dict_toml(d, radians=True)

        assert np.isclose(obj.orientation_angle_rad, np.pi / 18.0)
        assert np.isclose(obj.ellipticity_angle_rad, -np.pi / 9.0)
        assert obj.orientation_angle_rad == obj2.orientation_angle_rad
        assert obj.ellipticity_angle_rad == obj2.ellipticity_angle_rad


class TestInitialConditionsSchema:
    """
    Unit tests for InitialConditionsSchema.
    """

    @staticmethod
    def test_round_trip_toml():
        """
        Test writing and reading object through TOML file gives same object.
        """
        # Schema 1.
        obj = InitialConditionsSchema(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexComponents(
                [6.0, 7.0, 8.0], CoordinateSystem.CYLINDRICAL, holonomic=False
            ),
            PolarisationWaveMode(WaveMode.O),
            9.0,
            10.0,
            11,
        )

        d = obj.to_dict_toml()
        obj2 = InitialConditionsSchema.from_dict_toml(d, "test", radians=True)

        assert obj.name == obj2.name
        assert obj.time_ns == obj2.time_ns
        assert obj.frequency_ghz == obj2.frequency_ghz
        nptest.assert_allclose(obj.position, obj2.position)
        assert (
            obj.coordinate_system_position == obj2.coordinate_system_position
        )
        assert obj.power_w == obj2.power_w
        assert obj.beam_waist_radius_m == obj2.beam_waist_radius_m
        assert obj.n_radial_zones == obj2.n_radial_zones

        assert isinstance(obj2.refractive_index, RefractiveIndexComponents)
        nptest.assert_allclose(
            obj.refractive_index.refractive_index,
            obj2.refractive_index.refractive_index,
        )
        assert (
            obj.refractive_index.coordinate_system
            == obj2.refractive_index.coordinate_system
        )
        assert (
            obj.refractive_index.holonomic == obj2.refractive_index.holonomic
        )

        assert isinstance(obj2.polarisation, PolarisationWaveMode)
        assert obj.polarisation.wave_mode == obj2.polarisation.wave_mode

        # Schema 2.
        obj = InitialConditionsSchema(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexLaunchAnglesGeometric(10.0, -20.0, radians=False),
            PolarisationEllipseAngles(15.0, -25.0, radians=False),
            9.0,
            10.0,
            11,
        )

        d = obj.to_dict_toml()
        obj2 = InitialConditionsSchema.from_dict_toml(d, "test", radians=True)

        assert isinstance(
            obj2.refractive_index, RefractiveIndexLaunchAnglesGeometric
        )
        nptest.assert_allclose(
            obj.refractive_index.toroidal_angle_rad,
            obj2.refractive_index.toroidal_angle_rad,
        )
        nptest.assert_allclose(
            obj.refractive_index.poloidal_angle_rad,
            obj2.refractive_index.poloidal_angle_rad,
        )

        assert isinstance(obj2.polarisation, PolarisationEllipseAngles)
        assert (
            obj.polarisation.orientation_angle_rad
            == obj2.polarisation.orientation_angle_rad
        )
        assert (
            obj.polarisation.ellipticity_angle_rad
            == obj2.polarisation.ellipticity_angle_rad
        )

        # Schema 3.
        obj = InitialConditionsSchema(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexLaunchAnglesImas(10.0, -20.0, radians=False),
            PolarisationWaveMode(WaveMode.O),
            9.0,
            10.0,
            11,
        )

        d = obj.to_dict_toml()
        obj2 = InitialConditionsSchema.from_dict_toml(d, "test", radians=True)

        assert isinstance(
            obj2.refractive_index, RefractiveIndexLaunchAnglesImas
        )
        nptest.assert_allclose(
            obj.refractive_index.toroidal_angle_rad,
            obj2.refractive_index.toroidal_angle_rad,
        )
        nptest.assert_allclose(
            obj.refractive_index.poloidal_angle_rad,
            obj2.refractive_index.poloidal_angle_rad,
        )

        # Schema 4.
        obj = InitialConditionsSchema(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexNparallel(0.6, -20.0, radians=False),
            PolarisationWaveMode(WaveMode.O),
            9.0,
            10.0,
            11,
        )

        d = obj.to_dict_toml()
        obj2 = InitialConditionsSchema.from_dict_toml(d, "test", radians=True)

        assert isinstance(obj2.refractive_index, RefractiveIndexNparallel)
        nptest.assert_allclose(
            obj.refractive_index.n_parallel, obj2.refractive_index.n_parallel
        )
        nptest.assert_allclose(
            obj.refractive_index.angle_perp_rad,
            obj2.refractive_index.angle_perp_rad,
        )

        # Schema 5.
        obj = InitialConditionsSchema(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexOptimal(n_parallel_positive=True),
            PolarisationWaveMode(WaveMode.O),
            9.0,
            10.0,
            11,
        )

        d = obj.to_dict_toml()
        obj2 = InitialConditionsSchema.from_dict_toml(d, "test", radians=True)

        assert isinstance(obj2.refractive_index, RefractiveIndexOptimal)
        assert (
            obj.refractive_index.n_parallel_positive
            == obj2.refractive_index.n_parallel_positive
        )


def test_read_write_initial_conditions_toml():
    """
    Test functions to read / write initial conditions from TOML file.
    """
    schemas = [
        InitialConditionsSchema(
            "test",
            1.0,
            2.0,
            [3.0, 4.0, 5.0],
            CoordinateSystem.CARTESIAN,
            RefractiveIndexComponents(
                [6.0, 7.0, 8.0], CoordinateSystem.CYLINDRICAL, holonomic=False
            ),
            PolarisationWaveMode(WaveMode.O),
            9.0,
            10.0,
            11,
        ),
    ]

    with tempfile.TemporaryFile("r+") as fh:
        write_initial_conditions_toml(fh, schemas)
        fh.seek(0)
        schemas_2 = read_initial_conditions_toml(fh)

    obj = schemas[0]
    obj2 = schemas_2[0]

    assert obj.name == obj2.name
    assert obj.time_ns == obj2.time_ns
    assert obj.frequency_ghz == obj2.frequency_ghz
    nptest.assert_allclose(obj.position, obj2.position)
    assert obj.coordinate_system_position == obj2.coordinate_system_position
    assert obj.power_w == obj2.power_w
    assert obj.beam_waist_radius_m == obj2.beam_waist_radius_m
    assert obj.n_radial_zones == obj2.n_radial_zones

    assert isinstance(obj2.refractive_index, RefractiveIndexComponents)
    nptest.assert_allclose(
        obj.refractive_index.refractive_index,
        obj2.refractive_index.refractive_index,
    )
    assert (
        obj.refractive_index.coordinate_system
        == obj2.refractive_index.coordinate_system
    )
    assert obj.refractive_index.holonomic == obj2.refractive_index.holonomic

    assert isinstance(obj2.polarisation, PolarisationWaveMode)
    assert obj.polarisation.wave_mode == obj2.polarisation.wave_mode
