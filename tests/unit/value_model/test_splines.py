"""
Unit tests for value_model.splines.
"""

# Standard imports
import itertools
import logging
import tempfile
import typing

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest
import pytest
import scipy

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.shared.dimensions import Dimensions
from crayon.shared.numerics import (
    first_derivative_finite_difference,
    second_derivative_finite_difference,
)
from crayon.value_model import ValueCache
from crayon.value_model.splines import Spline1D, Spline2D, Spline3D, SplineBase

logger = logging.getLogger(__name__)


def assert_model_equal(model_1: SplineBase, model_2: SplineBase):
    """
    Assert spline models are equal.

    Parameters
    ----------
    model_1, model_2: SplineBase
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


class _TestSpline:
    """
    Base class for testing spline model.
    """

    x_dim = NotImplemented
    data_dim = Dimensions.x

    def scalar_value(self, x: np.ndarray[float]) -> float:
        """
        Calculate scalar model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_value : np.ndarray[float]
            Scalar model value.
        """
        raise NotImplementedError

    def scalar_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jacobian : np.ndarray[float]
            First derivative of scalar model.
        """
        raise NotImplementedError

    def scalar_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_hessian : np.ndarray[float]
            Second derivative of scalar model.
        """
        raise NotImplementedError

    def scalar_jerk(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jerk : np.ndarray[float]
            Third derivative of scalar model.
        """
        raise NotImplementedError

    def scalar_snap(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate fourth derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_snap : np.ndarray[float]
            Fourth derivative of scalar model.
        """
        raise NotImplementedError

    def vector_value(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate vector model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_value : np.ndarray[float]
            Vector model value.
        """
        raise NotImplementedError

    def vector_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jacobian : np.ndarray[float]
            First derivative of vector model.
        """
        raise NotImplementedError

    def vector_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_hessian : np.ndarray[float]
            Second derivative of vector model.
        """
        raise NotImplementedError

    def vector_jerk(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jerk : np.ndarray[float]
            Third derivative of vector model.
        """
        raise NotImplementedError

    def tensor_value(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate tensor model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_value : np.ndarray[float]
            Tensor model value
        """
        raise NotImplementedError

    def tensor_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jacobian : np.ndarray[float]
            First derivative of tensor model.
        """
        raise NotImplementedError

    def tensor_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_hessian : np.ndarray[float]
            Second derivative of tensor model.
        """
        raise NotImplementedError

    def tensor_jerk(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jerk : np.ndarray[float]
            Third derivative of tensor model.
        """
        raise NotImplementedError

    def test_functions(self, position: np.ndarray[float]):
        """
        Test derivatives of functions against finite difference.

        Parameters
        ----------
        position : np.ndarray[float]
            Position.
        """
        # Jacobians.
        expected_value = self.scalar_jacobian(position)
        actual_value = first_derivative_finite_difference(
            position, self.scalar_value, ()
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = self.vector_jacobian(position)
        actual_value = first_derivative_finite_difference(
            position, self.vector_value, (self.data_dim.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value)

        expected_value = self.tensor_jacobian(position)
        actual_value = first_derivative_finite_difference(
            position,
            self.tensor_value,
            (self.data_dim.size, self.data_dim.size),
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value)

        # Hessians.
        expected_value = self.scalar_hessian(position)
        actual_value = second_derivative_finite_difference(
            position, self.scalar_value, ()
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

        expected_value = self.vector_hessian(position)
        actual_value = second_derivative_finite_difference(
            position, self.vector_value, (self.data_dim.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

        expected_value = self.tensor_hessian(position)
        actual_value = second_derivative_finite_difference(
            position,
            self.tensor_value,
            (self.data_dim.size, self.data_dim.size),
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

        # Jerks.
        expected_value = self.scalar_jerk(position)
        actual_value = second_derivative_finite_difference(
            position, self.scalar_jacobian, (self.x_dim.size,)
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

        expected_value = self.vector_jerk(position)
        actual_value = second_derivative_finite_difference(
            position,
            self.vector_jacobian,
            (self.data_dim.size, self.x_dim.size),
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

        expected_value = self.tensor_jerk(position)
        actual_value = second_derivative_finite_difference(
            position,
            self.tensor_jacobian,
            (self.data_dim.size, self.data_dim.size, self.x_dim.size),
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

    def scalar_model(self) -> SplineBase:
        """
        Scalar spline model.

        Returns
        -------
        spline : SplineBase
            Spline model.
        """
        raise NotImplementedError

    def vector_model(self) -> SplineBase:
        """
        Vector spline model.

        Returns
        -------
        spline : SplineBase
            Spline model.
        """
        raise NotImplementedError

    def tensor_model(self) -> SplineBase:
        """
        Tensor spline model.

        Returns
        -------
        spline : SplineBase
            Spline model.
        """
        raise NotImplementedError

    test_positions = (
        np.array([0.9, 0.5, 0.3]),
        np.array([0.8, -0.4, 0.1]),
        np.array([-0.5, 0.7, -0.5]),
        np.array([-0.7, -0.8, 0.9]),
    )

    @pytest.fixture(params=test_positions)
    @staticmethod
    def position(request) -> np.ndarray[float]:
        """
        Position.

        Parameters
        ----------
        request
            Request

        Returns
        -------
        position : np.ndarray[float]
            Position.
        """
        return request.param

    def _test_value(
        self,
        model: SplineBase,
        position: np.ndarray[float],
        f_value: typing.Callable[[np.ndarray[float]], np.ndarray[float]],
        f_jacobian: typing.Callable[[np.ndarray[float]], np.ndarray[float]],
        f_hessian: typing.Callable[[np.ndarray[float]], np.ndarray[float]],
        f_jerk: typing.Callable[[np.ndarray[float]], np.ndarray[float]],
        shape: tuple[int],
    ):
        """
        Test value for single model.

        Parameters
        ----------
        model : SplineBase
            Test model.
        position : np.ndarray[float]
            Test position.
        f_value : callable[[np.ndarray[float]], np.ndarray[float]]
            Function giving value.
        f_jacobian : callable[[np.ndarray[float]], np.ndarray[float]]
            Function giving jacobian.
        f_hessian : callable[[np.ndarray[float]], np.ndarray[float]]
            Function giving hessian.
        f_jerk : callable[[np.ndarray[float]], np.ndarray[float]]
            Function giving jerk.
        shape : tuple[int]
            Shape of value.

        Notes
        -----
        Generous tolerances are for 3D spline fits.
        """
        # Test value
        expected_value = f_value(position)
        actual_value = model.value(position)
        nptest.assert_allclose(actual_value, expected_value, atol=6e-5)

        # Test jacobian.
        expected_value = f_jacobian(position)
        actual_value = model.jacobian(position)

        nptest.assert_allclose(actual_value, expected_value, atol=2e-4)

        # Test hessian.
        expected_value = f_hessian(position)
        actual_value = model.hessian(position)

        nptest.assert_allclose(actual_value, expected_value, atol=1e-2)

        # Test multidimensional input.
        position_multidimensional = np.empty((5, 5, self.x_dim.size))
        position_multidimensional[..., :] = position

        expected_value = np.empty((5, 5, *shape))
        for i, j in itertools.product(range(5), range(5)):
            expected_value[i, j, ...] = f_value(
                position_multidimensional[i, j]
            )

        actual_value = model.value(position_multidimensional)

        nptest.assert_allclose(actual_value, expected_value, atol=6e-5)

        # Test jerk.
        expected_value = f_jerk(position)
        actual_value = model.jerk(position)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-2)

        # Test multidimensional input.
        position_multidimensional = np.empty((5, 5, self.x_dim.size))
        position_multidimensional[..., :] = position

        expected_value = np.empty((5, 5, *shape))
        for i, j in itertools.product(range(5), range(5)):
            expected_value[i, j, ...] = f_value(
                position_multidimensional[i, j]
            )

        actual_value = model.value(position_multidimensional)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(actual_value, expected_value, atol=6e-5)

    def test_value(
        self,
        scalar_model: SplineBase,
        vector_model: SplineBase,
        tensor_model: SplineBase,
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
        _y = self.data_dim.size

        self._test_value(
            scalar_model,
            position,
            self.scalar_value,
            self.scalar_jacobian,
            self.scalar_hessian,
            self.scalar_jerk,
            (),
        )

        self._test_value(
            vector_model,
            position,
            self.vector_value,
            self.vector_jacobian,
            self.vector_hessian,
            self.vector_jerk,
            (_y,),
        )

        self._test_value(
            tensor_model,
            position,
            self.tensor_value,
            self.tensor_jacobian,
            self.tensor_hessian,
            self.tensor_jerk,
            (_y, _y),
        )

    def test_scalar_snap(
        self, scalar_model: SplineBase, position: np.ndarray[float]
    ):
        """
        Test snap of scalar model.

        Parameters
        ----------
        scalar_model: SplineBase
            Scalar valued spline model.
        position: np.ndarray[float]
            Test position.
        """
        # Snap function.
        expected_value = self.scalar_snap(position)
        actual_value = second_derivative_finite_difference(
            position, self.scalar_hessian, (self.x_dim.size, self.x_dim.size)
        )

        logger.warning(abs(expected_value - actual_value))
        assert expected_value.shape == actual_value.shape
        nptest.assert_allclose(expected_value, actual_value, atol=5e-7)

        # Spline fit.
        expected_value = self.scalar_snap(position)
        actual_value = scalar_model.snap(position)

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(actual_value, expected_value, atol=1e-2)

        # Fill cache for snap
        cache = ValueCache.for_model(scalar_model)
        scalar_model.fill_cache(cache, position, derivatives=4)

        expected_value = scalar_model(position, nu=4)
        actual_value = cache.snap

        logger.warning(abs(expected_value - actual_value))
        nptest.assert_allclose(expected_value, actual_value)

    @staticmethod
    def _test_fill_cache(model: SplineBase, position: np.ndarray[float]):
        """
        Test filling of cache.

        Parameters
        ----------
        model: SplineBase
            Spline model.
        position: np.ndarray[float]
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

    @pytest.mark.parametrize("position", test_positions)
    def test_fill_cache(
        self,
        scalar_model: SplineBase,
        vector_model: SplineBase,
        tensor_model: SplineBase,
        position: np.ndarray[float],
    ):
        """
        Test fill_cache for all models.

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
        self._test_fill_cache(scalar_model, position)
        self._test_fill_cache(vector_model, position)
        self._test_fill_cache(tensor_model, position)

    @staticmethod
    def assert_model_equal(model_1: SplineBase, model_2: SplineBase):
        """
        Assert models equal.

        Parameters
        ----------
        model_1, model_2 : SplineBase
            Models to check.
        """
        assert_model_equal(model_1, model_2)

        assert model_1.method == model_2.method
        assert model_1._dim == model_2._dim

        for i in range(model_1._dim):
            nptest.assert_allclose(
                model_1._abscissas[i], model_2._abscissas[i]
            )

        nptest.assert_allclose(model_1._data, model_2._data)
        nptest.assert_allclose(
            model_1._dependent_components, model_2._dependent_components
        )

    def test_round_trip_netcdf(
        self,
        scalar_model: SplineBase,
        vector_model: SplineBase,
        tensor_model: SplineBase,
    ):
        """
        Test serialising and deserialising through netCDF4 gives same object.

        Parameters
        ----------
        scalar_model: Constant
            Scalar constant model.
        vector_model: Constant
            Vector constant model.
        tensor_model: Constant
            Tensor constant model.
        """
        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions.write_netcdf(dset)

            scalar_model.write_netcdf(dset.createGroup("scalar_model"))
            scalar_model_2 = SplineBase.read_netcdf(dset["scalar_model"])

            vector_model.write_netcdf(dset.createGroup("vector_model"))
            vector_model_2 = SplineBase.read_netcdf(dset["vector_model"])

            tensor_model.write_netcdf(dset.createGroup("tensor_model"))
            tensor_model_2 = SplineBase.read_netcdf(dset["tensor_model"])

        self.assert_model_equal(scalar_model, scalar_model_2)
        self.assert_model_equal(vector_model, vector_model_2)
        self.assert_model_equal(tensor_model, tensor_model_2)


class TestSpline1D(_TestSpline):
    """
    Unit tests for Spline1D.
    """

    x_dim = Dimensions.x

    @staticmethod
    def scalar_value(x: np.ndarray[float]) -> float:
        """
        Calculate scalar model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_value : np.ndarray[float]
            Scalar model value.
        """
        return_value = np.zeros(())

        return_value.fill(x[0] ** 4)

        return return_value

    def scalar_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jacobian : np.ndarray[float]
            First derivative of scalar model.
        """
        return_value = np.zeros(self.x_dim.size)

        return_value[0] = 4 * x[0] ** 3

        return return_value

    def scalar_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_hessian : np.ndarray[float]
            Second derivative of scalar model.
        """
        return_value = np.zeros((self.x_dim.size, self.x_dim.size))

        return_value[0, 0] = 12 * x[0] ** 2

        return return_value

    def scalar_jerk(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jerk : np.ndarray[float]
            Third derivative of scalar model.
        """
        return_value = np.zeros((
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0] = 24 * x[0]

        return return_value

    def scalar_snap(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate fourth derivative of scalar model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_snap : np.ndarray[float]
            Fourth derivative of scalar model.
        """
        return_value = np.zeros((
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0, 0] = 24

        return return_value

    @staticmethod
    def vector_value(x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate vector model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_value : np.ndarray[float]
            Vector model value.
        """
        return_value = np.zeros(3)

        return_value[0] = x[0] ** 2
        return_value[1] = 2 * x[0] ** 2
        return_value[2] = 3 * x[0] ** 3

        return return_value

    def vector_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jacobian : np.ndarray[float]
            First derivative of vector model.
        """
        return_value = np.zeros((self.data_dim.size, self.x_dim.size))

        return_value[0, 0] = 2 * x[0]
        return_value[1, 0] = 4 * x[0]
        return_value[2, 0] = 9 * x[0] ** 2

        return return_value

    def vector_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_hessian : np.ndarray[float]
            Second derivative of vector model.
        """
        return_value = np.zeros((
            self.data_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0] = 2
        return_value[1, 0, 0] = 4
        return_value[2, 0, 0] = 18 * x[0]

        return return_value

    def vector_jerk(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of vector model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jerk : np.ndarray[float]
            Third derivative of vector model.
        """
        return_value = np.zeros((
            self.data_dim.size,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[2, 0, 0, 0] = 18

        return return_value

    def tensor_value(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate tensor model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_value : np.ndarray[float]
            Tensor model value
        """
        return_value = np.zeros((self.data_dim.size, self.data_dim.size))
        return_value[0, 0] = x[0] ** 2
        return_value[1, 0] = 2 * x[0] ** 2
        return_value[2, 2] = 3 * x[0] ** 3

        return return_value

    def tensor_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jacobian : np.ndarray[float]
            First derivative of tensor model.
        """
        return_value = np.zeros((
            self.data_dim.size,
            self.data_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0] = 2 * x[0]
        return_value[1, 0, 0] = 4 * x[0]
        return_value[2, 2, 0] = 9 * x[0] ** 2

        return return_value

    def tensor_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_hessian : np.ndarray[float]
            Second derivative of tensor model.
        """
        return_value = np.zeros((
            self.data_dim.size,
            self.data_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0, 0] = 2
        return_value[1, 0, 0, 0] = 4
        return_value[2, 2, 0, 0] = 18 * x[0]

        return return_value

    def tensor_jerk(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of tensor model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jerk : np.ndarray[float]
            Third derivative of tensor model.
        """
        return_value = np.zeros((
            self.data_dim.size,
            self.data_dim.size,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[2, 2, 0, 0, 0] = 18.0

        return return_value

    @pytest.fixture(scope="class")
    def scalar_model(self) -> Spline1D:
        """
        Scalar spline model.

        Returns
        -------
        spline : Spline1D
            Spline model.
        """
        x = np.linspace(-1, 1, 51)
        data = np.empty(x.size)

        _x = np.empty(self.x_dim.size)
        for i, x_value in enumerate(x):
            _x[0] = x_value
            data[i] = self.scalar_value(_x)

        return Spline1D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (),
            "unit",
            x,
            data,
            (True, False, False),
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> Spline1D:
        """
        Vector spline model.

        Returns
        -------
        spline : Spline1D
            Spline model.
        """
        x = np.linspace(-1, 1, 51)
        data = np.empty((x.size, self.data_dim.size))

        _x = np.empty(self.x_dim.size)
        for i, x_value in enumerate(x):
            _x[0] = x_value
            data[i, :] = self.vector_value(_x)

        return Spline1D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (self.data_dim,),
            "unit",
            x,
            data,
            (True, False, False),
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> Spline1D:
        """
        Tensor spline model.

        Returns
        -------
        spline : Spline1D
            Spline model.
        """
        x = np.linspace(-1, 1, 51)
        data = np.empty((x.size, self.data_dim.size, self.data_dim.size))

        _x = np.empty(self.x_dim.size)
        for i, x_value in enumerate(x):
            _x[0] = x_value
            data[i, :, :] = self.tensor_value(_x)

        return Spline1D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (self.data_dim, self.data_dim),
            "unit",
            x,
            data,
            (True, False, False),
        )


class TestSpline2D(_TestSpline):
    """
    Unit tests for Spline2D.
    """

    x_dim = Dimensions.x

    @staticmethod
    def scalar_value(x: np.ndarray[float]) -> float:
        """
        Calculate scalar model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_value : np.ndarray[float]
            Scalar model value.
        """
        return_value = np.zeros(())
        return_value.fill((x[0] * x[1]) ** 3)

        return return_value

    def scalar_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jacobian : np.ndarray[float]
            First derivative of scalar model.
        """
        return_value = np.zeros(self.x_dim.size)

        return_value[0] = 3 * x[0] ** 2 * x[1] ** 3
        return_value[1] = 3 * x[0] ** 3 * x[1] ** 2
        return return_value

    def scalar_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_hessian : np.ndarray[float]
            Second derivative of scalar model.
        """
        return_value = np.zeros((self.x_dim.size, self.x_dim.size))

        return_value[0, 0] = 6 * x[0] * x[1] ** 3
        return_value[0, 1] = 9 * x[0] ** 2 * x[1] ** 2
        return_value[1, 0] = return_value[0, 1]
        return_value[1, 1] = 6 * x[0] ** 3 * x[1]

        return return_value

    def scalar_jerk(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jerk : np.ndarray[float]
            Third derivative of scalar model.
        """
        return_value = np.zeros((
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0] = 6 * x[1] ** 3
        return_value[0, 0, 1] = 18 * x[0] * x[1] ** 2
        return_value[0, 1, 0] = return_value[0, 0, 1]
        return_value[1, 0, 0] = return_value[0, 0, 1]
        return_value[0, 1, 1] = 18 * x[0] ** 2 * x[1]
        return_value[1, 0, 1] = return_value[0, 1, 1]
        return_value[1, 1, 0] = return_value[0, 1, 1]
        return_value[1, 1, 1] = 6 * x[0] ** 3

        return return_value

    def scalar_snap(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate fourth derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_snap : np.ndarray[float]
            Fourth derivative of scalar model.
        """
        return_value = np.zeros((
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 0, 1] = 18 * x[1] ** 2
        return_value[0, 0, 1, 0] = return_value[0, 0, 0, 1]
        return_value[0, 1, 0, 0] = return_value[0, 0, 0, 1]
        return_value[1, 0, 0, 0] = return_value[0, 0, 0, 1]

        return_value[0, 0, 1, 1] = 36 * x[0] * x[1]
        return_value[0, 1, 0, 1] = return_value[0, 0, 1, 1]
        return_value[1, 0, 0, 1] = return_value[0, 0, 1, 1]
        return_value[0, 1, 1, 0] = return_value[0, 0, 1, 1]
        return_value[1, 0, 1, 0] = return_value[0, 0, 1, 1]
        return_value[1, 1, 0, 0] = return_value[0, 0, 1, 1]

        return_value[0, 1, 1, 1] = 18 * x[0] ** 2
        return_value[1, 0, 1, 1] = return_value[0, 1, 1, 1]
        return_value[1, 1, 0, 1] = return_value[0, 1, 1, 1]
        return_value[1, 1, 1, 0] = return_value[0, 1, 1, 1]

        return return_value

    @staticmethod
    def vector_value(x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate vector model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_value : np.ndarray[float]
            Vector model value.
        """
        return_value = np.zeros(3)

        return_value[0] = x[0] ** 2 + 2 * x[1] ** 2
        return_value[1] = 2 * (x[0] ** 2 + 2 * x[1] ** 2)
        return_value[2] = 3 * (x[0] ** 3 + 2 * x[1] ** 3)

        return return_value

    def vector_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jacobian : np.ndarray[float]
            First derivative of vector model.
        """
        return_value = np.zeros((3, self.x_dim.size))

        return_value[0, 0] = 2 * x[0]
        return_value[0, 1] = 4 * x[1]
        return_value[1, 0] = 4 * x[0]
        return_value[1, 1] = 8 * x[1]
        return_value[2, 0] = 9 * x[0] ** 2
        return_value[2, 1] = 18 * x[1] ** 2

        return return_value

    def vector_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_hessian : np.ndarray[float]
            Second derivative of vector model.
        """
        return_value = np.zeros((3, self.x_dim.size, self.x_dim.size))

        return_value[0, 0, 0] = 2
        return_value[0, 1, 1] = 4
        return_value[1, 0, 0] = 4
        return_value[1, 1, 1] = 8
        return_value[2, 0, 0] = 18 * x[0]
        return_value[2, 1, 1] = 36 * x[1]

        return return_value

    def vector_jerk(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of vector model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jerk : np.ndarray[float]
            Third derivative of vector model.
        """
        return_value = np.zeros((
            3,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[2, 0, 0, 0] = 18
        return_value[2, 1, 1, 1] = 36

        return return_value

    @staticmethod
    def tensor_value(x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate tensor model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_value : np.ndarray[float]
            Tensor model value
        """
        return_value = np.zeros((3, 3))

        return_value[0, 0] = x[0] ** 2 + 2 * x[1] ** 2
        return_value[1, 0] = 2 * (x[0] ** 2 + 2 * x[1] ** 2)
        return_value[2, 2] = 3 * (x[0] ** 3 + 2 * x[1] ** 3)

        return return_value

    def tensor_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jacobian : np.ndarray[float]
            First derivative of tensor model.
        """
        return_value = np.zeros((3, 3, self.x_dim.size))

        return_value[0, 0, 0] = 2 * x[0]
        return_value[0, 0, 1] = 4 * x[1]
        return_value[1, 0, 0] = 4 * x[0]
        return_value[1, 0, 1] = 8 * x[1]
        return_value[2, 2, 0] = 9 * x[0] ** 2
        return_value[2, 2, 1] = 18 * x[1] ** 2

        return return_value

    def tensor_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_hessian : np.ndarray[float]
            Second derivative of tensor model.
        """
        return_value = np.zeros((3, 3, self.x_dim.size, self.x_dim.size))

        return_value[0, 0, 0, 0] = 2
        return_value[0, 0, 1, 1] = 4
        return_value[1, 0, 0, 0] = 4
        return_value[1, 0, 1, 1] = 8
        return_value[2, 2, 0, 0] = 18 * x[0]
        return_value[2, 2, 1, 1] = 36 * x[1]

        return return_value

    def tensor_jerk(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of tensor model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jerk : np.ndarray[float]
            Third derivative of tensor model.
        """
        return_value = np.zeros((
            3,
            3,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[2, 2, 0, 0, 0] = 18
        return_value[2, 2, 1, 1, 1] = 36

        return return_value

    @pytest.fixture(scope="class")
    def scalar_model(self) -> Spline2D:
        """
        Scalar spline model.

        Returns
        -------
        spline : Spline2D
            Spline model.
        """
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 61)
        data = np.empty((x.size, y.size))

        _x = np.empty(2)
        for i, j in itertools.product(range(x.size), range(y.size)):
            _x[0] = x[i]
            _x[1] = y[j]
            data[i, j] = self.scalar_value(_x)

        return Spline2D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (),
            "unit",
            x,
            y,
            data,
            (True, True, False),
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> Spline2D:
        """
        Vector spline model.

        Returns
        -------
        spline : Spline2D
            Spline model.
        """
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 61)
        data = np.empty((x.size, y.size, self.data_dim.size))

        _x = np.empty(2)
        for i, j in itertools.product(range(x.size), range(y.size)):
            _x[0] = x[i]
            _x[1] = y[j]
            data[i, j, :] = self.vector_value(_x)

        return Spline2D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (self.data_dim,),
            "unit",
            x,
            y,
            data,
            (True, True, False),
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> Spline2D:
        """
        Tensor spline model.

        Returns
        -------
        spline : Spline2D
            Spline model.
        """
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 61)
        data = np.empty((
            x.size,
            y.size,
            self.data_dim.size,
            self.data_dim.size,
        ))

        _x = np.empty(2)
        for i, j in itertools.product(range(x.size), range(y.size)):
            _x[0] = x[i]
            _x[1] = y[j]
            data[i, j, :, :] = self.tensor_value(_x)

        return Spline2D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (self.data_dim, self.data_dim),
            "unit",
            x,
            y,
            data,
            (True, True, False),
        )


class TestSpline3D(_TestSpline):
    """
    Unit tests for Spline3D.
    """

    x_dim = Dimensions.x

    @staticmethod
    def scalar_value(x: np.ndarray[float]) -> float:
        """
        Calculate scalar model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_value : np.ndarray[float]
            Scalar model value.
        """
        return_value = np.zeros(())

        return_value.fill(
            x[0] ** 2
            + 2 * x[1] ** 2
            + 3 * x[2] ** 2
            + (x[0] * x[1] * x[2]) ** 2
        )

        return return_value

    def scalar_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jacobian : np.ndarray[float]
            First derivative of scalar model.
        """
        return_value = np.zeros(self.x_dim.size)

        return_value[0] = 2 * x[0] + 2 * x[0] * (x[1] * x[2]) ** 2
        return_value[1] = 4 * x[1] + 2 * x[1] * (x[0] * x[2]) ** 2
        return_value[2] = 6 * x[2] + 2 * x[2] * (x[0] * x[1]) ** 2

        return return_value

    def scalar_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_hessian : np.ndarray[float]
            Second derivative of scalar model.
        """
        return_value = np.zeros((self.x_dim.size, self.x_dim.size))

        return_value[0, 0] = 2 + 2 * (x[1] * x[2]) ** 2
        return_value[0, 1] = 4 * x[0] * x[1] * x[2] ** 2
        return_value[0, 2] = 4 * x[0] * x[1] ** 2 * x[2]
        return_value[1, 0] = return_value[0, 1]
        return_value[1, 1] = 4 + 2 * (x[0] * x[2]) ** 2
        return_value[1, 2] = 4 * x[1] * x[0] ** 2 * x[2]
        return_value[2, 0] = return_value[0, 2]
        return_value[2, 1] = return_value[1, 2]
        return_value[2, 2] = 6 + 2 * (x[0] * x[1]) ** 2

        return return_value

    def scalar_jerk(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_jerk : np.ndarray[float]
            Third derivative of scalar model.
        """
        return_value = np.zeros((
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 1] = 4 * x[1] * x[2] ** 2
        return_value[0, 1, 0] = return_value[0, 0, 1]
        return_value[1, 0, 0] = return_value[0, 0, 1]

        return_value[0, 0, 2] = 4 * x[1] ** 2 * x[2]
        return_value[0, 2, 0] = return_value[0, 0, 2]
        return_value[2, 0, 0] = return_value[0, 0, 2]

        return_value[0, 1, 1] = 4 * x[0] * x[2] ** 2
        return_value[1, 0, 1] = return_value[0, 1, 1]
        return_value[1, 1, 0] = return_value[0, 1, 1]

        return_value[0, 1, 2] = 8 * x[0] * x[1] * x[2]
        return_value[0, 2, 1] = return_value[0, 1, 2]
        return_value[1, 0, 2] = return_value[0, 1, 2]
        return_value[2, 0, 1] = return_value[0, 1, 2]
        return_value[1, 2, 0] = return_value[0, 1, 2]
        return_value[2, 1, 0] = return_value[0, 1, 2]

        return_value[0, 2, 2] = 4 * x[0] * x[1] ** 2
        return_value[2, 0, 2] = return_value[0, 2, 2]
        return_value[2, 2, 0] = return_value[0, 2, 2]

        return_value[1, 1, 2] = 4 * x[0] ** 2 * x[2]
        return_value[1, 2, 1] = return_value[1, 1, 2]
        return_value[2, 1, 1] = return_value[1, 1, 2]

        return_value[1, 2, 2] = 4 * x[1] * x[0] ** 2
        return_value[2, 1, 2] = return_value[1, 2, 2]
        return_value[2, 2, 1] = return_value[1, 2, 2]

        return return_value

    def scalar_snap(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate fourth derivative of scalar model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        scalar_snap : np.ndarray[float]
            Fourth derivative of scalar model.
        """
        return_value = np.zeros((
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[0, 0, 1, 1] = 4 * x[2] ** 2
        return_value[0, 1, 0, 1] = return_value[0, 0, 1, 1]
        return_value[1, 0, 0, 1] = return_value[0, 0, 1, 1]
        return_value[0, 1, 1, 0] = return_value[0, 0, 1, 1]
        return_value[1, 0, 1, 0] = return_value[0, 0, 1, 1]
        return_value[1, 1, 0, 0] = return_value[0, 0, 1, 1]

        return_value[0, 0, 1, 2] = 8 * x[1] * x[2]
        return_value[0, 1, 0, 2] = return_value[0, 0, 1, 2]
        return_value[1, 0, 0, 2] = return_value[0, 0, 1, 2]
        return_value[0, 1, 2, 0] = return_value[0, 0, 1, 2]
        return_value[1, 0, 2, 0] = return_value[0, 0, 1, 2]
        return_value[1, 2, 0, 0] = return_value[0, 0, 1, 2]
        return_value[0, 0, 2, 1] = return_value[0, 0, 1, 2]
        return_value[0, 2, 0, 1] = return_value[0, 0, 1, 2]
        return_value[2, 0, 0, 1] = return_value[0, 0, 1, 2]
        return_value[0, 2, 1, 0] = return_value[0, 0, 1, 2]
        return_value[2, 0, 1, 0] = return_value[0, 0, 1, 2]
        return_value[2, 1, 0, 0] = return_value[0, 0, 1, 2]

        return_value[0, 0, 2, 2] = 4 * x[1] ** 2
        return_value[0, 2, 0, 2] = return_value[0, 0, 2, 2]
        return_value[2, 0, 0, 2] = return_value[0, 0, 2, 2]
        return_value[0, 2, 2, 0] = return_value[0, 0, 2, 2]
        return_value[2, 0, 2, 0] = return_value[0, 0, 2, 2]
        return_value[2, 2, 0, 0] = return_value[0, 0, 2, 2]

        return_value[0, 1, 1, 2] = 8 * x[0] * x[2]
        return_value[1, 0, 1, 2] = return_value[0, 1, 1, 2]
        return_value[1, 1, 0, 2] = return_value[0, 1, 1, 2]
        return_value[1, 1, 2, 0] = return_value[0, 1, 1, 2]
        return_value[0, 1, 2, 1] = return_value[0, 1, 1, 2]
        return_value[1, 0, 2, 1] = return_value[0, 1, 1, 2]
        return_value[1, 2, 0, 1] = return_value[0, 1, 1, 2]
        return_value[1, 2, 1, 0] = return_value[0, 1, 1, 2]
        return_value[0, 2, 1, 1] = return_value[0, 1, 1, 2]
        return_value[2, 0, 1, 1] = return_value[0, 1, 1, 2]
        return_value[2, 1, 0, 1] = return_value[0, 1, 1, 2]
        return_value[2, 1, 1, 0] = return_value[0, 1, 1, 2]

        return_value[0, 1, 2, 2] = 8 * x[0] * x[1]
        return_value[1, 0, 2, 2] = return_value[0, 1, 2, 2]
        return_value[1, 2, 0, 2] = return_value[0, 1, 2, 2]
        return_value[1, 2, 2, 0] = return_value[0, 1, 2, 2]
        return_value[0, 2, 1, 2] = return_value[0, 1, 2, 2]
        return_value[2, 0, 1, 2] = return_value[0, 1, 2, 2]
        return_value[2, 1, 0, 2] = return_value[0, 1, 2, 2]
        return_value[2, 1, 2, 0] = return_value[0, 1, 2, 2]
        return_value[0, 2, 2, 1] = return_value[0, 1, 2, 2]
        return_value[2, 0, 2, 1] = return_value[0, 1, 2, 2]
        return_value[2, 2, 0, 1] = return_value[0, 1, 2, 2]
        return_value[2, 2, 1, 0] = return_value[0, 1, 2, 2]

        return_value[1, 1, 2, 2] = 4 * x[0] ** 2
        return_value[1, 2, 1, 2] = return_value[1, 1, 2, 2]
        return_value[2, 1, 1, 2] = return_value[1, 1, 2, 2]
        return_value[1, 2, 2, 1] = return_value[1, 1, 2, 2]
        return_value[2, 1, 2, 1] = return_value[1, 1, 2, 2]
        return_value[2, 2, 1, 1] = return_value[1, 1, 2, 2]

        return return_value

    @staticmethod
    def vector_value(x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate vector model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_value : np.ndarray[float]
            Vector model value.
        """
        return_value = np.zeros(3)

        return_value[0] = x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2
        return_value[1] = 2 * (x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2)
        return_value[2] = 3 * (x[0] ** 3 + 2 * x[1] ** 3 + 3 * x[2] ** 3)

        return return_value

    def vector_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jacobian : np.ndarray[float]
            First derivative of vector model.
        """
        return_value = np.zeros((3, self.x_dim.size))

        return_value[0, 0] = 2 * x[0]
        return_value[0, 1] = 4 * x[1]
        return_value[0, 2] = 6 * x[2]
        return_value[1, 0] = 4 * x[0]
        return_value[1, 1] = 8 * x[1]
        return_value[1, 2] = 12 * x[2]
        return_value[2, 0] = 9 * x[0] ** 2
        return_value[2, 1] = 18 * x[1] ** 2
        return_value[2, 2] = 27 * x[2] ** 2

        return return_value

    def vector_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of vector model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_hessian : np.ndarray[float]
            Second derivative of vector model.
        """
        return_value = np.zeros((3, self.x_dim.size, self.x_dim.size))

        return_value[0, 0, 0] = 2
        return_value[0, 1, 1] = 4
        return_value[0, 2, 2] = 6
        return_value[1, 0, 0] = 4
        return_value[1, 1, 1] = 8
        return_value[1, 2, 2] = 12
        return_value[2, 0, 0] = 18 * x[0]
        return_value[2, 1, 1] = 36 * x[1]
        return_value[2, 2, 2] = 54 * x[2]

        return return_value

    def vector_jerk(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of vector model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        vector_jerk : np.ndarray[float]
            Third derivative of vector model.
        """
        return_value = np.zeros((
            3,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[2, 0, 0, 0] = 18
        return_value[2, 1, 1, 1] = 36
        return_value[2, 2, 2, 2] = 54

        return return_value

    @staticmethod
    def tensor_value(x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate tensor model value.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_value : np.ndarray[float]
            Tensor model value
        """
        return_value = np.zeros((3, 3))

        return_value[0, 0] = x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2
        return_value[1, 0] = 2 * (x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2)
        return_value[2, 2] = 3 * (x[0] ** 3 + 2 * x[1] ** 3 + 3 * x[2] ** 3)

        return return_value

    def tensor_jacobian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate first derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jacobian : np.ndarray[float]
            First derivative of tensor model.
        """
        return_value = np.zeros((3, 3, self.x_dim.size))

        return_value[0, 0, 0] = 2 * x[0]
        return_value[0, 0, 1] = 4 * x[1]
        return_value[0, 0, 2] = 6 * x[2]
        return_value[1, 0, 0] = 4 * x[0]
        return_value[1, 0, 1] = 8 * x[1]
        return_value[1, 0, 2] = 12 * x[2]
        return_value[2, 2, 0] = 9 * x[0] ** 2
        return_value[2, 2, 1] = 18 * x[1] ** 2
        return_value[2, 2, 2] = 27 * x[2] ** 2

        return return_value

    def tensor_hessian(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate second derivative of tensor model.

        Parameters
        ----------
        x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_hessian : np.ndarray[float]
            Second derivative of tensor model.
        """
        return_value = np.zeros((3, 3, self.x_dim.size, self.x_dim.size))

        return_value[0, 0, 0, 0] = 2
        return_value[0, 0, 1, 1] = 4
        return_value[0, 0, 2, 2] = 6
        return_value[1, 0, 0, 0] = 4
        return_value[1, 0, 1, 1] = 8
        return_value[1, 0, 2, 2] = 12
        return_value[2, 2, 0, 0] = 18 * x[0]
        return_value[2, 2, 1, 1] = 36 * x[1]
        return_value[2, 2, 2, 2] = 54 * x[2]

        return return_value

    def tensor_jerk(self, _x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate third derivative of tensor model.

        Parameters
        ----------
        _x : np.ndarray[float]
            Position.

        Returns
        -------
        tensor_jerk : np.ndarray[float]
            Third derivative of tensor model.
        """
        return_value = np.zeros((
            3,
            3,
            self.x_dim.size,
            self.x_dim.size,
            self.x_dim.size,
        ))

        return_value[2, 2, 0, 0, 0] = 18
        return_value[2, 2, 1, 1, 1] = 36
        return_value[2, 2, 2, 2, 2] = 54

        return return_value

    @pytest.fixture(scope="class")
    def scalar_model(self) -> Spline3D:
        """
        Scalar spline model.

        Returns
        -------
        spline : Spline3D
            Spline model.
        """
        x = np.linspace(-1, 1, 6)
        y = np.linspace(-1, 1, 9)
        z = np.linspace(-1, 1, 14)
        data = np.empty((x.size, y.size, z.size))

        _x = np.empty(3)
        for i, j, k in itertools.product(
            range(x.size), range(y.size), range(z.size)
        ):
            _x[0] = x[i]
            _x[1] = y[j]
            _x[2] = z[k]
            data[i, j, k] = self.scalar_value(_x)

        return Spline3D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (),
            "unit",
            x,
            y,
            z,
            data,
            (True, True, True),
        )

    @pytest.fixture(scope="class")
    def vector_model(self) -> Spline3D:
        """
        Vector spline model.

        Returns
        -------
        spline : Spline3D
            Spline model.
        """
        x = np.linspace(-1, 1, 6)
        y = np.linspace(-1, 1, 9)
        z = np.linspace(-1, 1, 14)
        data = np.empty((x.size, y.size, z.size, self.data_dim.size))

        _x = np.empty(3)
        for i, j, k in itertools.product(
            range(x.size), range(y.size), range(z.size)
        ):
            _x[0] = x[i]
            _x[1] = y[j]
            _x[2] = z[k]
            data[i, j, k, :] = self.vector_value(_x)

        return Spline3D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (self.data_dim,),
            "unit",
            x,
            y,
            z,
            data,
            (True, True, True),
        )

    @pytest.fixture(scope="class")
    def tensor_model(self) -> Spline3D:
        """
        Tensor spline model.

        Returns
        -------
        spline : Spline3D
            Spline model.
        """
        x = np.linspace(-1, 1, 6)
        y = np.linspace(-1, 1, 9)
        z = np.linspace(-1, 1, 14)
        data = np.empty((
            x.size,
            y.size,
            z.size,
            self.data_dim.size,
            self.data_dim.size,
        ))

        _x = np.empty(3)
        for i, j, k in itertools.product(
            range(x.size), range(y.size), range(z.size)
        ):
            _x[0] = x[i]
            _x[1] = y[j]
            _x[2] = z[k]
            data[i, j, k, :, :] = self.tensor_value(_x)

        return Spline3D(
            CoordinateSystem.CARTESIAN,
            self.x_dim,
            (self.data_dim, self.data_dim),
            "unit",
            x,
            y,
            z,
            data,
            (True, True, True),
        )

    def test_value(
        self,
        scalar_model: Spline3D,
        vector_model: Spline3D,
        tensor_model: Spline3D,
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
        if scipy.__version__ > "1.13.0":
            super().test_value(
                scalar_model, vector_model, tensor_model, position
            )
        else:
            pytest.skip("Requires scipy >= 1.13.0")

    def test_scalar_snap(
        self, scalar_model: Spline3D, position: np.ndarray[float]
    ):
        """
        Test snap of scalar model.

        Parameters
        ----------
        scalar_model: SplineBase
            Scalar valued spline model.
        position: np.ndarray[float]
            Test position.
        """
        if scipy.__version__ > "1.13.0":
            super().test_scalar_snap(scalar_model, position)

        pytest.skip("Requires scipy >= 1.13.0")

    def test_fill_cache(
        self,
        scalar_model: SplineBase,
        vector_model: SplineBase,
        tensor_model: SplineBase,
        position: np.ndarray[float],
    ):
        """
        Test fill_cache for all models.

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
        if scipy.__version__ > "1.13.0":
            super().test_fill_cache(
                scalar_model, vector_model, tensor_model, position
            )

        pytest.skip("Requires scipy >= 1.13.0")
