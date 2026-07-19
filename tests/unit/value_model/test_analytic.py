"""
Tests for value_model.analytic
"""

# Standard imports
import itertools
import logging
import pathlib
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.value_model import ValueCache
from crayon.value_model.analytic import (
    C0Ramp,
    C1Ramp,
    C2Ramp,
    Constant,
    QuadraticBowl,
    QuadraticChannel,
    QuadraticWell,
)
from crayon.value_model.base import ValueModelBase

logger = logging.getLogger(__name__)


def assert_model_equal(model_1: ValueModelBase, model_2: ValueModelBase):
    """
    Assert models are equal.

    Parameters
    ----------
    model_1, model_2: ValueModelBase
        Models to check.
    """
    for var in (
        "coordinate_system",
        "input_dimension",
        "output_dimensions",
        "units",
        "input_size",
        "output_shape",
        "dtype",
    ):
        value_1 = getattr(model_1, var)
        value_2 = getattr(model_2, var)
        assert value_1 == value_2

    nptest.assert_allclose(model_1.scale_factor, model_2.scale_factor)
    nptest.assert_allclose(model_1.input_bounds, model_2.input_bounds)


class TestConstant:
    """
    Unit tests for Constant.
    """

    scalar_const = np.array(2.0)
    vector_const = np.array([1.0, 0.5, -0.5])
    tensor_const = np.linspace(0, 1, 9).reshape((3, 3))

    @pytest.fixture(scope="class")
    def scalar_model(self) -> Constant:
        """
        Scalar constant model.

        Returns
        -------
        scalar_model : Constant
            Constant model.
        """
        return Constant(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (),
            "unit",
            self.scalar_const,
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> Constant:
        """
        Vector constant model.

        Returns
        -------
        vector_model : Constant
            Constant model.
        """
        return Constant(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x,),
            "unit",
            self.vector_const,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> Constant:
        """
        Tensor constant model.

        Returns
        -------
        tensor_model : Constant
            Constant model.
        """
        return Constant(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "unit",
            self.tensor_const,
        )

    @staticmethod
    def _test_value(
        model: Constant,
        position: np.ndarray[float],
        constant_value: np.ndarray[float],
    ):
        """
        Test value for model.

        Parameters
        ----------
        model : Constant
            Constant model.
        position : np.ndarray[float]
            Test position.
        constant_value : np.ndarray[float]
            Constant model.
        """
        # Test value
        expected_value = constant_value
        actual_value = model(position)
        nptest.assert_allclose(actual_value, expected_value)

        # Test jacobian.
        expected_value = np.zeros((*constant_value.shape, 3))
        actual_value = model(position, nu=1)

        nptest.assert_allclose(actual_value, expected_value)

        # Test hessian.
        expected_value = np.zeros((*constant_value.shape, 3, 3))
        actual_value = model(position, nu=2)

        nptest.assert_allclose(actual_value, expected_value)

        # Test jerk.
        expected_value = np.zeros((*constant_value.shape, 3, 3, 3))
        actual_value = model(position, nu=3)

        nptest.assert_allclose(actual_value, expected_value)

        # Test multidimensional input.
        position_multidimensional = np.empty((5, 5, 3))
        position_multidimensional[..., :] = position
        expected_value = np.tile(
            constant_value, (5, 5, *itertools.repeat(1, constant_value.ndim))
        )

        actual_value = model(position_multidimensional)

        nptest.assert_allclose(actual_value, expected_value)

    test_positions = (
        np.asarray([0.9, 0.3, 0.7]),
        np.asarray([0.5, -0.7, 1.3]),
        np.asarray([0.4, -0.2, 0.9]),
        np.asarray([-0.1, 0.8, -1.5]),
    )

    @pytest.mark.parametrize("position", test_positions)
    def test_value(
        self,
        scalar_model: Constant,
        vector_model: Constant,
        tensor_model: Constant,
        position: np.ndarray[float],
    ):
        """
        Test value for all models.

        Parameters
        ----------
        scalar_model: Constant
            Scalar constant model.
        vector_model: Constant
            Vector constant model.
        tensor_model: Constant
            Tensor constant model.
        position: np.ndarray[float]
            Test position.
        """
        self._test_value(scalar_model, position, self.scalar_const)
        self._test_value(vector_model, position, self.vector_const)
        self._test_value(tensor_model, position, self.tensor_const)

    @staticmethod
    def _test_fill_cache(model: Constant, position: np.ndarray[float]):
        """
        Test for filling cache.

        Parameters
        ----------
        model : Constant
            Test model.
        position : np.ndarray[float]
            Test position.
        """
        cache = ValueCache.for_model(model)
        model.fill_cache(cache, position, derivatives=3)

        # Test value.
        expected_value = model(position, nu=0)
        actual_value = cache.value

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test jacobian.
        expected_value = model(position, nu=1)
        actual_value = cache.jacobian

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test hessian.
        expected_value = model(position, nu=2)
        actual_value = cache.hessian

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        # Test jerk.
        expected_value = model(position, nu=3)
        actual_value = cache.jerk

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @classmethod
    @pytest.mark.parametrize("position", test_positions)
    def test_fill_cache(
        cls,
        scalar_model: Constant,
        vector_model: Constant,
        tensor_model: Constant,
        position: np.ndarray[float],
    ):
        """
        Test filling cache for all models.

        Parameters
        ----------
        scalar_model: Constant
            Scalar valued model.
        vector_model: Constant
            Vector valued model.
        tensor_model: Constant
            Tensor valued model.
        position: np.ndarray[float]
            Test position.
        """
        cls._test_fill_cache(scalar_model, position)
        cls._test_fill_cache(vector_model, position)
        cls._test_fill_cache(tensor_model, position)

    @staticmethod
    def assert_model_equal(model_1: Constant, model_2: Constant):
        """
        Assert 2 constant models equal.

        Parameters
        ----------
        model_1, model_2 : Constant
            Models to compare.
        """
        assert_model_equal(model_1, model_2)
        nptest.assert_allclose(model_1.constant_value, model_2.constant_value)

    def test_round_trip_netcdf(
        self,
        scalar_model: Constant,
        vector_model: Constant,
        tensor_model: Constant,
    ):
        """
        Test serialising and deserialising through netCDF4 gives same object.

        Parameters
        ----------
        scalar_model: Constant
            Scalar valued model.
        vector_model: Constant
            Vector valued model.
        tensor_model: Constant
            Tensor valued model.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            nc4.Dataset(pathlib.Path(tmpdir).joinpath("tmp.nc"), "w") as dset,
        ):
            Dimensions.write_netcdf(dset)

            scalar_model.write_netcdf(dset.createGroup("scalar_model"))
            scalar_model_2 = Constant.read_netcdf(dset["scalar_model"])

            vector_model.write_netcdf(dset.createGroup("vector_model"))
            vector_model_2 = Constant.read_netcdf(dset["vector_model"])

            tensor_model.write_netcdf(dset.createGroup("tensor_model"))
            tensor_model_2 = Constant.read_netcdf(dset["tensor_model"])

        self.assert_model_equal(scalar_model, scalar_model_2)
        self.assert_model_equal(vector_model, vector_model_2)
        self.assert_model_equal(tensor_model, tensor_model_2)


class TestC0Ramp:
    """
    Unit tests for C0Ramp.
    """

    _cls = C0Ramp
    origin = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    ramp_width = 2.0

    y0_scalar, y1_scalar = -1.0, 1.0
    y0_vector, y1_vector = np.full(3, -1.0), np.full(3, 1.0)
    y0_tensor, y1_tensor = -np.identity(3), np.identity(3)

    @pytest.fixture(scope="class")
    def scalar_model(self):
        """
        Scalar ramp model.

        Returns
        -------
        scalar_model : C0Ramp
            Ramp model.
        """
        return C0Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (),
            "",
            self.origin,
            self.direction,
            self.y0_scalar,
            self.y1_scalar,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def vector_model(self):
        """
        Vector ramp model.

        Returns
        -------
        vector_model : C0Ramp
            Ramp model.
        """
        return C0Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x,),
            "",
            self.origin,
            self.direction,
            self.y0_vector,
            self.y1_vector,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self):
        """
        Tensor ramp model.

        Returns
        -------
        tensor_model : C0Ramp
            Ramp model.
        """
        return C0Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "",
            self.origin,
            self.direction,
            self.y0_tensor,
            self.y1_tensor,
            self.ramp_width,
        )

    @staticmethod
    def assert_model_equal(model_1: C0Ramp, model_2: C0Ramp):
        """
        Assert ramp models equal.

        Parameters
        ----------
        model_1, model_2 : C0Ramp
            Models to compare.
        """
        nptest.assert_allclose(model_1.origin, model_2.origin)
        nptest.assert_allclose(model_1.direction, model_2.direction)
        nptest.assert_allclose(model_1.y0, model_2.y0)
        nptest.assert_allclose(model_1.y1, model_2.y1)
        nptest.assert_allclose(model_1.ramp_width, model_2.ramp_width)

    test_positions = (
        np.array([-0.1, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([1.6, 0.0, 0.0]),
        np.array([2.2, 0.0, 0.0]),
    )

    @staticmethod
    @pytest.fixture(params=test_positions)
    def position(request) -> np.ndarray[float]:
        """
        Test position.

        Parameters
        ----------
        request
            Request.

        Returns
        -------
        position : np.ndarray[float]
            Position.
        """
        return request.param

    @staticmethod
    @pytest.mark.parametrize(
        "model", ["scalar_model", "vector_model", "tensor_model"]
    )
    def test_derivatives(request, position: np.ndarray[float], model: str):
        """
        Test derivative calculation.

        Parameters
        ----------
        request
            Request
        position : np.ndarray[float]
            Test position.
        model : str
            Name of model.
        """
        _model = request.getfixturevalue(model)

        expected_value = _model.jacobian(position)
        actual_value = first_derivative_finite_difference(
            position,
            _model.value,
            _model.output_shape,
        )

        logger.warning((expected_value, actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = _model.hessian(position)
        actual_value = second_derivative_finite_difference(
            position,
            _model.value,
            _model.output_shape,
        )

        logger.warning((expected_value, actual_value))
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = _model.jerk(position)
        actual_value = second_derivative_finite_difference(
            position,
            _model.jacobian,
            (*_model.output_shape, _model.input_size),
        )

        logger.warning((expected_value, actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    @pytest.mark.parametrize("position", test_positions)
    def test_fill_cache(
        scalar_model: C0Ramp,
        vector_model: C0Ramp,
        tensor_model: C0Ramp,
        position: np.ndarray[float],
    ):
        """
        Test fill_cache.

        Parameters
        ----------
        scalar_model: Constant
            Scalar valued model.
        vector_model: Constant
            Vector valued model.
        tensor_model: Constant
            Tensor valued model.
        position : np.ndarray[float]
            Test position.
        """
        TestConstant._test_fill_cache(scalar_model, position)
        TestConstant._test_fill_cache(vector_model, position)
        TestConstant._test_fill_cache(tensor_model, position)

    def test_round_trip_netcdf(
        self,
        scalar_model: C0Ramp,
        vector_model: C0Ramp,
        tensor_model: C0Ramp,
    ):
        """
        Test serialising and deserialising through netCDF4 gives same object.

        Parameters
        ----------
        scalar_model: Constant
            Scalar valued model.
        vector_model: Constant
            Vector valued model.
        tensor_model: Constant
            Tensor valued model.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            nc4.Dataset(pathlib.Path(tmpdir).joinpath("tmp.nc"), "w") as dset,
        ):
            Dimensions.write_netcdf(dset)

            scalar_model.write_netcdf(dset.createGroup("scalar_model"))
            scalar_model_2 = self._cls.read_netcdf(dset["scalar_model"])

            vector_model.write_netcdf(dset.createGroup("vector_model"))
            vector_model_2 = self._cls.read_netcdf(dset["vector_model"])

            tensor_model.write_netcdf(dset.createGroup("tensor_model"))
            tensor_model_2 = self._cls.read_netcdf(dset["tensor_model"])

        self.assert_model_equal(scalar_model, scalar_model_2)
        self.assert_model_equal(vector_model, vector_model_2)
        self.assert_model_equal(tensor_model, tensor_model_2)


class TestC1Ramp(TestC0Ramp):
    """
    Unit tests for C1Ramp.
    """

    _cls = C1Ramp

    @pytest.fixture(scope="class")
    def scalar_model(self) -> C1Ramp:
        """
        Scalar ramp model.

        Returns
        -------
        scalar_model : C1Ramp
            Ramp model.
        """
        return C1Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (),
            "",
            self.origin,
            self.direction,
            self.y0_scalar,
            self.y1_scalar,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> C1Ramp:
        """
        Vector ramp model.

        Returns
        -------
        vector_model : C1Ramp
            Ramp model.
        """
        return C1Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x,),
            "",
            self.origin,
            self.direction,
            self.y0_vector,
            self.y1_vector,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> C1Ramp:
        """
        Tensor ramp model.

        Returns
        -------
        tensor_model : C1Ramp
            Ramp model.
        """
        return C1Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "",
            self.origin,
            self.direction,
            self.y0_tensor,
            self.y1_tensor,
            self.ramp_width,
        )


class TestC2Ramp(TestC0Ramp):
    """
    Unit tests for C2Ramp.
    """

    _cls = C2Ramp

    @pytest.fixture(scope="class")
    def scalar_model(self) -> C2Ramp:
        """
        Scalar ramp model.

        Returns
        -------
        scalar_model : C0Ramp
            Ramp model.
        """
        return C2Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (),
            "",
            self.origin,
            self.direction,
            self.y0_scalar,
            self.y1_scalar,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> C2Ramp:
        """
        Vector ramp model.

        Returns
        -------
        vector_model : C2Ramp
            Ramp model.
        """
        return C2Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x,),
            "",
            self.origin,
            self.direction,
            self.y0_vector,
            self.y1_vector,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> C2Ramp:
        """
        Tensor ramp model.

        Returns
        -------
        tensor_model : C2Ramp
            Ramp model.
        """
        return C2Ramp(
            CoordinateSystem.CARTESIAN,
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "",
            self.origin,
            self.direction,
            self.y0_tensor,
            self.y1_tensor,
            self.ramp_width,
        )


class TestQuadraticChannel(TestC0Ramp):
    """
    Unit tests for QuadraticChannel.
    """

    _cls = QuadraticChannel

    @pytest.fixture(scope="class")
    def scalar_model(self) -> QuadraticChannel:
        """
        Scalar quadratic channel model.

        Returns
        -------
        scalar_model : QuadraticChannel
            Quadratic channel model.
        """
        return QuadraticChannel(
            Dimensions.x,
            (),
            "",
            self.origin,
            self.direction,
            self.y0_scalar,
            self.y1_scalar,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> QuadraticChannel:
        """
        Vector quadratic channel model.

        Returns
        -------
        vector_model : QuadraticChannel
            Quadratic channel model.
        """
        return QuadraticChannel(
            Dimensions.x,
            (Dimensions.x,),
            "",
            self.origin,
            self.direction,
            self.y0_vector,
            self.y1_vector,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> QuadraticChannel:
        """
        Tensor quadratic channel model.

        Returns
        -------
        tensor_model : QuadraticChannel
            Quadratic channel model.
        """
        return QuadraticChannel(
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "",
            self.origin,
            self.direction,
            self.y0_tensor,
            self.y1_tensor,
            self.ramp_width,
        )

    def test_value(self):
        """
        Test value.
        """


class TestQuadraticBowl(TestC0Ramp):
    """
    Unit tests for QuadraticBowl.
    """

    _cls = QuadraticBowl

    @pytest.fixture(scope="class")
    def scalar_model(self) -> QuadraticBowl:
        """
        Scalar quadratic bowl model.

        Returns
        -------
        scalar_model : QuadraticBowl
            Quadratic bowl model.
        """
        return QuadraticBowl(
            Dimensions.x,
            (),
            "",
            self.origin,
            self.direction,
            self.y0_scalar,
            self.y1_scalar,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> QuadraticBowl:
        """
        Vector quadratic bowl model.

        Returns
        -------
        vector_model : QuadraticBowl
            Quadratic bowl model.
        """
        return QuadraticBowl(
            Dimensions.x,
            (Dimensions.x,),
            "",
            self.origin,
            self.direction,
            self.y0_vector,
            self.y1_vector,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> QuadraticBowl:
        """
        Tensor quadratic bowl model.

        Returns
        -------
        tensor_model : QuadraticBowl
            Quadratic bowl model.
        """
        return QuadraticBowl(
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "",
            self.origin,
            self.direction,
            self.y0_tensor,
            self.y1_tensor,
            self.ramp_width,
        )

    def test_value(self):
        """
        Test value.
        """


class TestQuadraticWell(TestC0Ramp):
    """
    Unit tests for QuadraticWell.
    """

    _cls = QuadraticWell

    @pytest.fixture(scope="class")
    def scalar_model(self) -> QuadraticWell:
        """
        Scalar quadratic well model.

        Returns
        -------
        scalar_model : QuadraticWell
            Quadratic well model.
        """
        return QuadraticWell(
            Dimensions.x,
            (),
            "",
            self.origin,
            self.y0_scalar,
            self.y1_scalar,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> QuadraticWell:
        """
        Vector quadratic well model.

        Returns
        -------
        vector_model : QuadraticWell
            Quadratic well model.
        """
        return QuadraticWell(
            Dimensions.x,
            (Dimensions.x,),
            "",
            self.origin,
            self.y0_vector,
            self.y1_vector,
            self.ramp_width,
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> QuadraticWell:
        """
        Tensor quadratic well model.

        Returns
        -------
        tensor_model : QuadraticWell
            Quadratic well model.
        """
        return QuadraticWell(
            Dimensions.x,
            (Dimensions.x, Dimensions.x),
            "",
            self.origin,
            self.y0_tensor,
            self.y1_tensor,
            self.ramp_width,
        )

    def test_value(self):
        """
        Test value.
        """

    @staticmethod
    def assert_model_equal(model_1: QuadraticWell, model_2: QuadraticWell):
        """
        Assert models are equal.

        Parameters
        ----------
        model_1, model_2: QuadraticWell
            Models to check.
        """
        assert_model_equal(model_1, model_2)

        nptest.assert_allclose(model_1.origin, model_2.origin)
        nptest.assert_allclose(model_1.y0, model_2.y0)
        nptest.assert_allclose(model_1.y1, model_2.y1)
        nptest.assert_allclose(model_1.ramp_width, model_2.ramp_width)
