"""
Unit tests for value_model.models
"""

# Standard imports
import logging

import numpy as np
import numpy.testing as nptest
import pytest

# Third party imports
# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.shared.dimensions import Dimensions
from crayon.value_model.models import (
    CnRamp,
    Constant,
    QuadraticBowl,
    QuadraticChannel,
    QuadraticWell,
    Spline1D,
    Spline2D,
    Spline3D,
    SplineMethod,
    ValueModel,
)

logger = logging.getLogger(__name__)


class TestValueModel:
    """
    Unit tests for ValueModel.
    """

    @pytest.fixture
    @staticmethod
    def scalar() -> ValueModel:
        """
        Scalar model.

        Returns
        -------
        scalar : ValueModel
            Scalar value model.
        """
        return ValueModel("test", Dimensions.x, (), "")

    @pytest.fixture
    @staticmethod
    def vector() -> ValueModel:
        """
        Vector model.

        Returns
        -------
        vector : ValueModel
            Vector value model.
        """
        return ValueModel("test", Dimensions.x, (Dimensions.x,), "")

    @staticmethod
    def test_analytic_constant(scalar: ValueModel, vector: ValueModel):
        """
        Test constant analytic model.

        Parameters
        ----------
        scalar : ValueModel
            Scalar value model.
        vector : ValueModel
            Vector value model.
        """
        # Test scalar value.
        constant_value = 1.3
        scale_factor = 1.2

        model = scalar.constant(
            CoordinateSystem.CARTESIAN,
            constant_value,
            scale_factor=scale_factor,
        )

        assert isinstance(model, Constant)
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(model.constant_value, constant_value)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test array value.
        constant_value = [1.0, 2.0, 3.0]

        model = vector.constant(
            CoordinateSystem.CARTESIAN,
            constant_value,
            scale_factor=scale_factor,
        )

        nptest.assert_allclose(model.constant_value, constant_value)

    @staticmethod
    def test_analytic_ramp(scalar: ValueModel, vector: ValueModel):
        """
        Test analytic ramp model.

        Parameters
        ----------
        scalar : ValueModel
            Scalar value model.
        vector : ValueModel
            Vector value model.
        """
        # Test scalar value.
        origin = [1.0, 2.0, 3.0]
        direction = [0.6, 0.8, 0.0]
        y0 = 0.2
        y1 = 1.3
        ramp_width = 2.4
        scale_factor = 1.2
        smoothness = 1

        model = scalar.ramp(
            CoordinateSystem.CARTESIAN,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            smoothness,
            scale_factor=scale_factor,
        )

        assert isinstance(model, CnRamp)
        assert model.smoothness == smoothness
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.direction, direction)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test array value.
        y0 = [1.0, 3.0, 2.0]
        y1 = [0.5, 1.4, 1.8]

        model = vector.ramp(
            CoordinateSystem.CARTESIAN,
            origin,
            direction,
            y0,
            y1,
            ramp_width,
            smoothness,
            scale_factor=scale_factor,
        )

        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)

    @staticmethod
    def test_analytic_quadratic_channel(
        scalar: ValueModel, vector: ValueModel
    ):
        """
        Test analytic quadratic channel model.

        Parameters
        ----------
        scalar : ValueModel
            Scalar value model.
        vector : ValueModel
            Vector value model.
        """
        # Test scalar value.
        origin = [1.0, 2.0, 3.0]
        direction = [0.6, 0.8, 0.0]
        y0 = 0.2
        y1 = 1.3
        ramp_width = 2.8
        scale_factor = 1.2

        model = scalar.quadratic_channel(
            origin, direction, y0, y1, ramp_width, scale_factor=scale_factor
        )

        assert isinstance(model, QuadraticChannel)
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.direction, direction)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test array value.
        y0 = [1.0, 3.0, 2.0]
        y1 = [0.5, 1.4, 1.8]

        model = vector.quadratic_channel(
            origin,
            direction,
            y0,
            y1,
            ramp_width,
        )

        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)

    @staticmethod
    def test_analytic_quadratic_bowl(scalar: ValueModel, vector: ValueModel):
        """
        Test analytic quadratic bowl model.

        Parameters
        ----------
        scalar : ValueModel
            Scalar value model.
        vector : ValueModel
            Vector value model.
        """
        # Test scalar value.
        origin = [1.0, 2.0, 3.0]
        direction = [0.6, 0.8, 0.0]
        y0 = 0.2
        y1 = 1.3
        ramp_width = 2.8
        scale_factor = 1.2

        model = scalar.quadratic_bowl(
            origin, direction, y0, y1, ramp_width, scale_factor=scale_factor
        )

        assert isinstance(model, QuadraticBowl)
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.direction, direction)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test array value.
        y0 = [1.0, 3.0, 2.0]
        y1 = [0.5, 1.4, 1.8]

        model = vector.quadratic_bowl(
            origin,
            direction,
            y0,
            y1,
            ramp_width,
        )

        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)

    @staticmethod
    def test_analytic_quadratic_well(scalar: ValueModel, vector: ValueModel):
        """
        Test analytic quadratic well model.

        Parameters
        ----------
        scalar : ValueModel
            Scalar value model.
        vector : ValueModel
            Vector value model.
        """
        # Test scalar value.
        origin = [1.0, 2.0, 3.0]
        y0 = 0.2
        y1 = 1.3
        ramp_width = 2.8
        scale_factor = 1.2

        model = scalar.quadratic_well(
            origin, y0, y1, ramp_width, scale_factor=scale_factor
        )

        assert isinstance(model, QuadraticWell)
        nptest.assert_allclose(model.origin, origin)
        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)
        nptest.assert_allclose(model.ramp_width, ramp_width)
        nptest.assert_allclose(model.scale_factor, scale_factor)

        # Test array value.
        y0 = [1.0, 3.0, 2.0]
        y1 = [0.5, 1.4, 1.8]

        model = vector.quadratic_well(
            origin,
            y0,
            y1,
            ramp_width,
        )

        nptest.assert_allclose(model.y0, y0)
        nptest.assert_allclose(model.y1, y1)

    @staticmethod
    def test_splines():
        """
        Test spline creation.
        """
        size = 11
        x = np.linspace(0, 1, size)
        y_1d = np.linspace(0, 1, size)
        y_2d = np.linspace(0, 1, size**2).reshape((size, size))
        y_3d = np.linspace(0, 1, size**3).reshape((size, size, size))
        scale_factor = 1.7

        # Test 1D spline.
        model = ValueModel.electron_density_per_m3().spline_1d(
            CoordinateSystem.CARTESIAN,
            x,
            y_1d,
            (True, False, False),
            scale_factor=scale_factor,
            method=SplineMethod.LINEAR,
        )

        assert isinstance(model, Spline1D)
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(model._abscissas[0], x)
        nptest.assert_allclose(model._data, y_1d)
        nptest.assert_allclose(
            model._dependent_components, (True, False, False)
        )

        # Test 2D spline.
        model = ValueModel.electron_density_per_m3().spline_2d(
            CoordinateSystem.CARTESIAN,
            x,
            2.0 * x,
            y_2d,
            (True, False, True),
            scale_factor=scale_factor,
            method=SplineMethod.LINEAR,
        )

        assert isinstance(model, Spline2D)
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(model._abscissas[0], x)
        nptest.assert_allclose(model._abscissas[1], 2.0 * x)
        nptest.assert_allclose(model._data, y_2d)
        nptest.assert_allclose(
            model._dependent_components, (True, False, True)
        )

        # Test 3D spline.
        model = ValueModel.electron_density_per_m3().spline_3d(
            CoordinateSystem.CARTESIAN,
            x,
            2.0 * x,
            3.0 * x,
            y_3d,
            (True, True, True),
            scale_factor=scale_factor,
            method=SplineMethod.LINEAR,
        )

        assert isinstance(model, Spline3D)
        assert model.coordinate_system == CoordinateSystem.CARTESIAN
        nptest.assert_allclose(model._abscissas[0], x)
        nptest.assert_allclose(model._abscissas[1], 2.0 * x)
        nptest.assert_allclose(model._abscissas[2], 3.0 * x)
        nptest.assert_allclose(model._data, y_3d)
        nptest.assert_allclose(model._dependent_components, (True, True, True))
