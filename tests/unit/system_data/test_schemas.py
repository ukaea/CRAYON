"""
Unit tests for system_data_provider.schemas
"""

# Standard imports
import logging
import pathlib

import numpy as np
import numpy.testing as nptest

# Third party imports
# Local imports
from crayon.shared.constants import CoordinateSystem
from crayon.system_data.schemas import (
    COORDINATES,
    DATA_SOURCES,
    KINETIC,
    LIMITERS,
    MAGNETIC_FIELD_T,
    CoordinateToroidal,
    DataSourceImas,
    DataSourceNetcdf,
    DataSourceVmec,
    LimiterAnalyticBoundingBox2D,
    LimiterAnalyticBoundingBox3D,
    LimiterEffect,
    LimiterImas2D,
    LimiterImas3D,
    LimiterImasBoundingBox2D,
    LimiterNetcdf,
    MagneticModelStellarator,
    MagneticModelTokamak,
    ModelAnalyticConstant,
    ModelAnalyticQuadraticBowl,
    ModelAnalyticQuadraticChannel,
    ModelAnalyticQuadraticWell,
    ModelAnalyticRamp,
    ModelImas,
    ModelNetcdf,
    parse_coordinates,
    parse_data_sources,
    parse_kinetic,
    parse_limiters,
    parse_magnetic,
    parse_model,
    parse_schema,
)

logger = logging.getLogger(__name__)


class TestDataSource:
    """
    Unit tests for data sources.
    """

    @staticmethod
    def test_imas_round_trip_toml():
        """
        Test TOML serialising and deserialising gives same object.
        """
        data_source = DataSourceImas("uri", 1, 2, 3)
        toml_dict = data_source.to_dict_toml()
        data_source_2 = DataSourceImas.from_dict_toml(toml_dict)

        assert isinstance(data_source_2, DataSourceImas)
        assert data_source.uri == data_source_2.uri
        assert (
            data_source.occurrence_core_profiles
            == data_source_2.occurrence_core_profiles
        )
        assert (
            data_source.occurrence_equilibrium
            == data_source_2.occurrence_equilibrium
        )
        assert data_source.occurrence_wall == data_source_2.occurrence_wall

    @staticmethod
    def test_netcdf_round_trip_toml():
        """
        Test TOML serialising and deserialising gives same object.
        """
        data_source = DataSourceNetcdf("/some/file")
        toml_dict = data_source.to_dict_toml()
        data_source_2 = DataSourceNetcdf.from_dict_toml(toml_dict)

        assert isinstance(data_source_2, DataSourceNetcdf)
        assert (
            data_source.filepath.resolve() == data_source_2.filepath.resolve()
        )

    @staticmethod
    def test_vmec_round_trip_toml():
        """
        Test TOML serialising and deserialising gives same object.
        """
        data_source = DataSourceVmec("/some/file")
        toml_dict = data_source.to_dict_toml()
        data_source_2 = DataSourceVmec.from_dict_toml(toml_dict)

        assert isinstance(data_source_2, DataSourceVmec)
        assert (
            data_source.filepath.resolve() == data_source_2.filepath.resolve()
        )

    @staticmethod
    def test_parse_data_sources():
        """
        Test parsing data sources from dictionary.
        """
        # Check no data sources.
        data_sources = parse_data_sources({})
        assert data_sources == {}

        # Check all data sources.
        imas = {
            "uri": "/some/imas/database",
            "occurrence_core_profiles": 1,
            "occurrence_equilibrium": 2,
            "occurrence_wall": 3,
        }
        netcdf = {"filepath": "/some/netcdf/file"}
        vmec = {"filepath": "/some/vmec/file"}

        document = {
            "imas_test": imas,
            "netcdf_test": netcdf,
            "vmec_test": vmec,
            "NETCDF_CAPS_TEST": netcdf,
        }

        data_sources = parse_data_sources(document)

        imas_2 = data_sources["imas_test"]
        assert isinstance(imas_2, DataSourceImas)
        assert imas["uri"] == imas_2.uri
        assert (
            imas["occurrence_core_profiles"] == imas_2.occurrence_core_profiles
        )
        assert imas["occurrence_equilibrium"] == imas_2.occurrence_equilibrium
        assert imas["occurrence_wall"] == imas_2.occurrence_wall

        netcdf_2 = data_sources["netcdf_test"]
        assert isinstance(netcdf_2, DataSourceNetcdf)
        assert (
            pathlib.Path(netcdf["filepath"]).resolve()
            == netcdf_2.filepath.resolve()
        )

        vmec_2 = data_sources["vmec_test"]
        assert isinstance(vmec_2, DataSourceVmec)
        assert (
            pathlib.Path(vmec["filepath"]).resolve()
            == vmec_2.filepath.resolve()
        )

        netcdf_3 = data_sources["netcdf_caps_test"]
        assert isinstance(netcdf_3, DataSourceNetcdf)


class TestCoordinates:
    """
    Unit tests for coordinate system definitions.
    """

    @staticmethod
    def test_toroidal_round_trip_toml():
        """
        Test TOML serialising and deserialising gives same object.
        """
        r0, z0 = 1.3, 1.6

        coordinate = CoordinateToroidal(r0, z0)
        toml_dict = coordinate.to_dict_toml()
        coordinate_2 = CoordinateToroidal.from_dict_toml(toml_dict)

        assert isinstance(coordinate_2, CoordinateToroidal)
        assert coordinate.r0 == coordinate_2.r0
        assert coordinate.z0 == coordinate_2.z0

    @staticmethod
    def test_parse_coordinates():
        """
        Test parsing from dictionary.
        """
        # Check no coordinates.
        coordinates = parse_coordinates({})
        assert coordinates == {}

        # Check all coordinates.
        toroidal = {"r0": 1.2, "z0": 1.5}

        document = {"toroidal": toroidal}

        coordinates = parse_coordinates(document)

        logger.warning(coordinates)

        toroidal_2 = coordinates[CoordinateSystem.TOROIDAL]
        assert toroidal["r0"] == toroidal_2.r0
        assert toroidal["z0"] == toroidal_2.z0


class TestModels:
    """
    Unit tests for models.
    """

    scale_factor = 1.2

    def test_imas_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        model = ModelImas("imas_test", self.scale_factor)
        toml_dict = model.to_dict_toml()
        model_2 = ModelImas.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelImas)
        assert model.scale_factor == model_2.scale_factor

    def test_netcdf_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        model = ModelNetcdf("netcdf_test", self.scale_factor)
        toml_dict = model.to_dict_toml()
        model_2 = ModelNetcdf.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelNetcdf)
        assert model.scale_factor == model_2.scale_factor

    def test_constant_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        coordinate_system = CoordinateSystem.CARTESIAN
        constant_value = np.ones(3)

        model = ModelAnalyticConstant(
            coordinate_system, constant_value, self.scale_factor
        )
        toml_dict = model.to_dict_toml()
        model_2 = ModelAnalyticConstant.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelAnalyticConstant)
        assert model.coordinate_system == model_2.coordinate_system
        assert np.all(model.constant_value == model_2.constant_value)
        assert model.scale_factor == model_2.scale_factor

    def test_ramp_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        coordinate_system = CoordinateSystem.CARTESIAN
        origin = np.ones(3)
        direction = 2.0 * np.ones(3)
        y0 = 3.0 * np.ones(3)
        y1 = 4.0 * np.ones(3)
        ramp_width = 1.5
        smoothness = 2

        model = ModelAnalyticRamp(
            coordinate_system,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            smoothness,
            self.scale_factor,
        )
        toml_dict = model.to_dict_toml()
        model_2 = ModelAnalyticRamp.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelAnalyticRamp)
        assert model.coordinate_system == model_2.coordinate_system
        assert np.all(model.origin == model_2.origin)
        assert np.all(model.direction == model_2.direction)
        assert np.all(model.y0 == model_2.y0)
        assert np.all(model.y1 == model_2.y1)
        assert model.ramp_width == model_2.ramp_width
        assert model.smoothness == model_2.smoothness
        assert model.scale_factor == model_2.scale_factor

    def test_quadratic_well_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        origin = np.ones(3)
        y0 = 3.0 * np.ones(3)
        y1 = 4.0 * np.ones(3)
        ramp_width = 1.5

        model = ModelAnalyticQuadraticWell(
            origin, y0, y1, ramp_width, self.scale_factor
        )
        toml_dict = model.to_dict_toml()
        model_2 = ModelAnalyticQuadraticWell.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelAnalyticQuadraticWell)
        assert np.all(model.origin == model_2.origin)
        assert np.all(model.y0 == model_2.y0)
        assert np.all(model.y1 == model_2.y1)
        assert model.ramp_width == model_2.ramp_width
        assert model.scale_factor == model_2.scale_factor

    def test_quadratic_channel_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        origin = np.ones(3)
        direction = 2.0 * np.ones(3)
        y0 = 3.0 * np.ones(3)
        y1 = 4.0 * np.ones(3)
        ramp_width = 1.5

        model = ModelAnalyticQuadraticChannel(
            origin, direction, y0, y1, ramp_width, self.scale_factor
        )
        toml_dict = model.to_dict_toml()
        model_2 = ModelAnalyticQuadraticChannel.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelAnalyticQuadraticChannel)
        assert np.all(model.origin == model_2.origin)
        assert np.all(model.direction == model_2.direction)
        assert np.all(model.y0 == model_2.y0)
        assert np.all(model.y1 == model_2.y1)
        assert model.ramp_width == model_2.ramp_width
        assert model.scale_factor == model_2.scale_factor

    def test_quadratic_bowl_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        origin = np.ones(3)
        direction = 2.0 * np.ones(3)
        y0 = 3.0 * np.ones(3)
        y1 = 4.0 * np.ones(3)
        ramp_width = 1.5

        model = ModelAnalyticQuadraticBowl(
            origin, direction, y0, y1, ramp_width, self.scale_factor
        )
        toml_dict = model.to_dict_toml()
        model_2 = ModelAnalyticQuadraticBowl.from_dict_toml(toml_dict)

        assert isinstance(model_2, ModelAnalyticQuadraticBowl)
        assert np.all(model.origin == model_2.origin)
        assert np.all(model.direction == model_2.direction)
        assert np.all(model.y0 == model_2.y0)
        assert np.all(model.y1 == model_2.y1)
        assert model.ramp_width == model_2.ramp_width
        assert model.scale_factor == model_2.scale_factor

    @staticmethod
    def test_parse_model():
        """
        Test parsing from dictionary.
        """
        # Test IMAS.
        document = {"source": "imas_test"}
        model = parse_model(document)
        assert isinstance(model, ModelImas)
        assert model.source == document["source"]
        nptest.assert_allclose(model.scale_factor, 1.0)

        # Test netCDF.
        document = {"source": "netcdf_test"}
        model = parse_model(document)
        assert isinstance(model, ModelNetcdf)
        assert model.source == document["source"]
        nptest.assert_allclose(model.scale_factor, 1.0)

        # Test constant.
        document = {
            "source": "constant",
            "coordinate_system": "cartesian",
            "constant_value": 1.0,
        }
        model = parse_model(document)
        assert isinstance(model, ModelAnalyticConstant)
        assert model.source == ModelAnalyticConstant.source
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        assert model.constant_value == document["constant_value"]
        nptest.assert_allclose(model.scale_factor, 1.0)

        # Test ramp.
        document = {
            "source": "ramp",
            "coordinate_system": "cartesian",
            "origin": [1.0, 0.0, 0.0],
            "direction": [2.0, 0.0, 0.0],
            "y0": 3.0,
            "y1": 4.0,
            "ramp_width": 5.0,
            "smoothness": 2,
        }
        model = parse_model(document)
        assert isinstance(model, ModelAnalyticRamp)
        assert model.source == ModelAnalyticRamp.source
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(model.origin, document["origin"])
        nptest.assert_allclose(model.direction, document["direction"])
        nptest.assert_allclose(model.y0, document["y0"])
        nptest.assert_allclose(model.y1, document["y1"])
        nptest.assert_allclose(model.ramp_width, document["ramp_width"])
        assert model.smoothness == document["smoothness"]
        nptest.assert_allclose(model.scale_factor, 1.0)

        # Test quadratic well.
        document = {
            "source": "quadratic_well",
            "origin": [1.0, 0.0, 0.0],
            "y0": 3.0,
            "y1": 4.0,
            "ramp_width": 5.0,
        }
        model = parse_model(document)
        assert isinstance(model, ModelAnalyticQuadraticWell)
        assert model.source == ModelAnalyticQuadraticWell.source
        nptest.assert_allclose(model.origin, document["origin"])
        nptest.assert_allclose(model.y0, document["y0"])
        nptest.assert_allclose(model.y1, document["y1"])
        nptest.assert_allclose(model.ramp_width, document["ramp_width"])
        nptest.assert_allclose(model.scale_factor, 1.0)

        # Test quadratic channel.
        document = {
            "source": "quadratic_channel",
            "origin": [1.0, 0.0, 0.0],
            "direction": [2.0, 0.0, 0.0],
            "y0": 3.0,
            "y1": 4.0,
            "ramp_width": 5.0,
        }
        model = parse_model(document)
        assert isinstance(model, ModelAnalyticQuadraticChannel)
        assert model.source == ModelAnalyticQuadraticChannel.source
        nptest.assert_allclose(model.origin, document["origin"])
        nptest.assert_allclose(model.direction, document["direction"])
        nptest.assert_allclose(model.y0, document["y0"])
        nptest.assert_allclose(model.y1, document["y1"])
        nptest.assert_allclose(model.ramp_width, document["ramp_width"])
        nptest.assert_allclose(model.scale_factor, 1.0)

        # Test quadratic bowl.
        document = {
            "source": "quadratic_bowl",
            "origin": [1.0, 0.0, 0.0],
            "direction": [2.0, 0.0, 0.0],
            "y0": 3.0,
            "y1": 4.0,
            "ramp_width": 5.0,
        }
        model = parse_model(document)
        assert isinstance(model, ModelAnalyticQuadraticBowl)
        assert model.source == ModelAnalyticQuadraticBowl.source
        nptest.assert_allclose(model.origin, document["origin"])
        nptest.assert_allclose(model.direction, document["direction"])
        nptest.assert_allclose(model.y0, document["y0"])
        nptest.assert_allclose(model.y1, document["y1"])
        nptest.assert_allclose(model.ramp_width, document["ramp_width"])
        nptest.assert_allclose(model.scale_factor, 1.0)

    @staticmethod
    def test_parse_kinetic():
        """
        Test parsing kinetic model from dictionary.
        """
        document = {
            "electron_density_per_m3": {"source": "imas_test"},
            "electron_temperature_ev": {"source": "netcdf_test"},
            "effective_charge": {
                "source": "constant",
                "coordinate_system": "cartesian",
                "constant_value": 1.0,
            },
        }

        (electron_density, electron_temperature, effective_charge) = (
            parse_kinetic(document)
        )

        assert isinstance(electron_density, ModelImas)
        assert electron_density.source == "imas_test"
        nptest.assert_allclose(electron_density.scale_factor, 1.0)

        assert isinstance(electron_temperature, ModelNetcdf)
        assert electron_temperature.source == "netcdf_test"
        nptest.assert_allclose(electron_temperature.scale_factor, 1.0)

        assert isinstance(effective_charge, ModelAnalyticConstant)
        assert effective_charge.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(effective_charge.constant_value, 1.0)
        nptest.assert_allclose(effective_charge.scale_factor, 1.0)


class TestMagnetic:
    """
    Test magnetic models.
    """

    scale_factor = 1.1

    def test_tokamak_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        model = MagneticModelTokamak("imas_test", self.scale_factor)
        toml_dict = model.to_dict_toml()
        model_2 = MagneticModelTokamak.from_dict_toml(toml_dict)

        assert isinstance(model_2, MagneticModelTokamak)
        assert model_2.source == "imas_test"
        assert model_2.scale_factor == self.scale_factor

    def test_stellarator_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        model = MagneticModelStellarator("vmec_test", self.scale_factor)
        toml_dict = model.to_dict_toml()
        model_2 = MagneticModelStellarator.from_dict_toml(toml_dict)

        assert isinstance(model_2, MagneticModelStellarator)
        assert model_2.source == "vmec_test"
        assert model_2.scale_factor == self.scale_factor

    @staticmethod
    def test_parse_magnetic():
        """
        Test parsing from dictionary.
        """
        # Test simple.
        document = {
            "topology": "simple",
            "source": "constant",
            "coordinate_system": "cartesian",
            "constant_value": [0.0, 0.0, 1.0],
        }
        model = parse_magnetic(document)

        assert isinstance(model, ModelAnalyticConstant)

        # Test tokamak.
        document = {"topology": "tokamak", "source": "imas_test"}
        model = parse_magnetic(document)

        assert isinstance(model, MagneticModelTokamak)
        assert model.source == "imas_test"

        # Test stellarator.
        document = {"topology": "stellarator", "source": "vmec_test"}
        model = parse_magnetic(document)

        assert isinstance(model, MagneticModelStellarator)
        assert model.source == "vmec_test"


class TestLimiters:
    """
    Test limiter models.
    """

    limiter_effect = LimiterEffect.STOP

    def test_imas_bounding_box_2d_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        limiter = LimiterImasBoundingBox2D(
            self.limiter_effect, "imas_test", 0.0
        )
        document = limiter.to_dict_toml()
        limiter_2 = LimiterImasBoundingBox2D.from_dict_toml(document)

        assert isinstance(limiter_2, LimiterImasBoundingBox2D)
        assert limiter_2.effect == self.limiter_effect
        assert limiter_2.source == limiter.source

    def test_imas_curve_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        limiter = LimiterImas2D(self.limiter_effect, "imas_test", 0.0)
        document = limiter.to_dict_toml()
        limiter_2 = LimiterImas2D.from_dict_toml(document)

        assert isinstance(limiter_2, LimiterImas2D)
        assert limiter_2.effect == self.limiter_effect
        assert limiter_2.source == limiter.source

    def test_imas_surface_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        limiter = LimiterImas3D(self.limiter_effect, "imas_test", 0.0)
        document = limiter.to_dict_toml()
        limiter_2 = LimiterImas3D.from_dict_toml(document)

        assert isinstance(limiter_2, LimiterImas3D)
        assert limiter_2.effect == self.limiter_effect
        assert limiter_2.source == limiter.source

    @staticmethod
    def test_netcdf_curve_round_trip_toml():
        """
        Test TOML serialising and deserialising gives same object.
        """
        limiter = LimiterNetcdf("netcdf_test", "test/group", 0.0)
        document = limiter.to_dict_toml()
        limiter_2 = LimiterNetcdf.from_dict_toml(document)

        assert isinstance(limiter_2, LimiterNetcdf)
        assert limiter_2.source == limiter.source
        assert limiter_2.group == limiter.group

    def test_analytic_bounding_box_2d_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        limiter = LimiterAnalyticBoundingBox2D(
            self.limiter_effect, "xy", (0.0, 1.0), (-1.0, 2.0), 0.0
        )
        document = limiter.to_dict_toml()
        limiter_2 = LimiterAnalyticBoundingBox2D.from_dict_toml(document)

        assert isinstance(limiter_2, LimiterAnalyticBoundingBox2D)
        assert limiter_2.effect == self.limiter_effect
        assert limiter_2.coordinate == LimiterAnalyticBoundingBox2D.XY
        nptest.assert_allclose(limiter.x_limits, limiter_2.x_limits)
        nptest.assert_allclose(limiter.y_limits, limiter_2.y_limits)

    def test_analytic_bounding_box_3d_round_trip_toml(self):
        """
        Test TOML serialising and deserialising gives same object.
        """
        limiter = LimiterAnalyticBoundingBox3D(
            self.limiter_effect,
            "xyz",
            (0.0, 1.0),
            (-1.0, 2.0),
            (1.0, 3.0),
            0.0,
        )
        document = limiter.to_dict_toml()
        limiter_2 = LimiterAnalyticBoundingBox3D.from_dict_toml(document)

        assert isinstance(limiter_2, LimiterAnalyticBoundingBox3D)
        assert limiter_2.effect == self.limiter_effect
        assert limiter_2.coordinate == LimiterAnalyticBoundingBox3D.XYZ
        nptest.assert_allclose(limiter.x_limits, limiter_2.x_limits)
        nptest.assert_allclose(limiter.y_limits, limiter_2.y_limits)
        nptest.assert_allclose(limiter.z_limits, limiter_2.z_limits)

    @staticmethod
    def test_parse_limiters():
        """
        Test parsing limiters.
        """
        # Test analytic bounding box.
        limiters = parse_limiters({
            "limiter": {
                "effect": "stop",
                "source": "analytic",
                "shape": "bounding_box_2d",
                "coordinate": "xy",
                "x_limits": (0.0, 1.0),
                "y_limits": (-1.0, 2.0),
            }
        })

        limiter = limiters["limiter"]
        assert isinstance(limiter, LimiterAnalyticBoundingBox2D)
        assert limiter.effect == LimiterEffect.STOP
        assert limiter.coordinate == LimiterAnalyticBoundingBox2D.XY
        nptest.assert_allclose(limiter.x_limits, (0.0, 1.0))
        nptest.assert_allclose(limiter.y_limits, (-1.0, 2.0))
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.0)

        limiters = parse_limiters({
            "limiter": {
                "effect": "stop",
                "source": "analytic",
                "shape": "bounding_box_3d",
                "coordinate": "xyz",
                "x_limits": (0.0, 1.0),
                "y_limits": (-1.0, 2.0),
                "z_limits": (2.0, 4.0),
                "extinction_coefficient_nepers": 0.1,
            }
        })

        limiter = limiters["limiter"]
        assert isinstance(limiter, LimiterAnalyticBoundingBox3D)
        assert limiter.effect == LimiterEffect.STOP
        assert limiter.coordinate == LimiterAnalyticBoundingBox3D.XYZ
        nptest.assert_allclose(limiter.x_limits, (0.0, 1.0))
        nptest.assert_allclose(limiter.y_limits, (-1.0, 2.0))
        nptest.assert_allclose(limiter.z_limits, (2.0, 4.0))
        nptest.assert_allclose(limiter.extinction_coefficient_nepers, 0.1)

        # Test IMAS.
        limiters = parse_limiters({
            "limiter": {
                "effect": "stop",
                "source": "imas_test",
                "shape": "bounding_box_2d",
            }
        })

        limiter = limiters["limiter"]
        assert isinstance(limiter, LimiterImasBoundingBox2D)
        assert limiter.effect == LimiterEffect.STOP

        limiters = parse_limiters({
            "limiter": {
                "effect": "stop",
                "source": "imas_test",
                "shape": "2d",
            }
        })

        limiter = limiters["limiter"]
        assert isinstance(limiter, LimiterImas2D)
        assert limiter.effect == LimiterEffect.STOP
        assert limiter.source == "imas_test"

        limiters = parse_limiters({
            "limiter": {
                "effect": "stop",
                "source": "imas_test",
                "shape": "3d",
            }
        })

        limiter = limiters["limiter"]
        assert isinstance(limiter, LimiterImas3D)
        assert limiter.effect == LimiterEffect.STOP
        assert limiter.source == "imas_test"

        # Test netCDF.
        limiters = parse_limiters({
            "limiter": {
                "source": "netcdf_test",
                "group": "/some/group",
            }
        })

        limiter = limiters["limiter"]
        assert isinstance(limiter, LimiterNetcdf)
        assert limiter.source == "netcdf_test"
        assert limiter.group == "/some/group"


def test_parse_schema():
    """
    Test parsing schema from dictionary.
    """
    document = {
        DATA_SOURCES: {"imas_test": {"uri": "/some/uri"}},
        COORDINATES: {"toroidal": {"r0": 1.2, "z0": 0.3}},
        KINETIC: {
            "electron_density_per_m3": {"source": "imas_test"},
            "electron_temperature_ev": {"source": "imas_test"},
            "effective_charge": {"source": "imas_test"},
        },
        MAGNETIC_FIELD_T: {"topology": "tokamak", "source": "imas_test"},
        LIMITERS: {
            "limiter_test": {
                "effect": "stop",
                "source": "imas_test",
                "shape": "2d",
            }
        },
    }

    (
        data_sources,
        coordinates,
        electron_density_per_m3,
        electron_temperature_ev,
        effective_charge,
        magnetic_field_t,
        limiters,
    ) = parse_schema(document)

    assert "imas_test" in data_sources
    assert type(data_sources["imas_test"]) is DataSourceImas
    assert data_sources["imas_test"].uri == "/some/uri"

    assert CoordinateSystem.TOROIDAL in coordinates
    nptest.assert_allclose(coordinates[CoordinateSystem.TOROIDAL].r0, 1.2)
    nptest.assert_allclose(coordinates[CoordinateSystem.TOROIDAL].z0, 0.3)

    assert electron_density_per_m3.source == "imas_test"
    assert electron_temperature_ev.source == "imas_test"
    assert effective_charge.source == "imas_test"

    assert type(magnetic_field_t) is MagneticModelTokamak
    assert magnetic_field_t.source == "imas_test"

    assert "limiter_test" in limiters
    assert type(limiters["limiter_test"]) is LimiterImas2D
    assert limiters["limiter_test"].source == "imas_test"
