"""
Tests for system_data.limiter
"""

# Standard imports
import logging
import tempfile

import netCDF4 as nc4  # noqa: N813
import numpy as np
import numpy.testing as nptest

# Local imports
from crayon.shared.dimensions import Dimensions
from crayon.system_data.limiter import (
    CappedCone,
    Cylinder,
    Disk,
    Limiter,
    LimiterEffect,
    Plane,
)

logger = logging.getLogger(__name__)


class TestPlane:
    """
    Unit tests for Plane.
    """

    @staticmethod
    def test_constructors():
        """
        Test helper methods for constructing planes.
        """
        # xy.
        plane = Plane.xy([1.0, 2.0], [3.0, 4.0], LimiterEffect.STOP)

        nptest.assert_allclose(plane.origin, [1.0, 2.0, 0.0])
        nptest.assert_allclose(plane.direction_1, [3.0, 4.0, 0.0])
        nptest.assert_allclose(
            np.dot(plane.direction_1, plane.normal), 0.0, atol=1e-8
        )
        nptest.assert_allclose(
            np.dot(plane.direction_2, plane.normal), 0.0, atol=1e-8
        )

        # xz.
        plane = Plane.xz([1.0, 2.0], [3.0, 4.0], LimiterEffect.STOP)

        nptest.assert_allclose(plane.origin, [1.0, 0.0, 2.0])
        nptest.assert_allclose(plane.direction_1, [3.0, 0.0, 4.0])
        nptest.assert_allclose(
            np.dot(plane.direction_1, plane.normal), 0.0, atol=1e-8
        )
        nptest.assert_allclose(
            np.dot(plane.direction_2, plane.normal), 0.0, atol=1e-8
        )

        # yz.
        plane = Plane.yz([1.0, 2.0], [3.0, 4.0], LimiterEffect.STOP)

        nptest.assert_allclose(plane.origin, [0.0, 1.0, 2.0])
        nptest.assert_allclose(plane.direction_1, [0.0, 3.0, 4.0])
        nptest.assert_allclose(
            np.dot(plane.direction_1, plane.normal), 0.0, atol=1e-8
        )
        nptest.assert_allclose(
            np.dot(plane.direction_2, plane.normal), 0.0, atol=1e-8
        )

        # xyz.
        plane = Plane.xyz(
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            LimiterEffect.STOP,
        )

        nptest.assert_allclose(plane.origin, [1.0, 2.0, 3.0])
        nptest.assert_allclose(plane.direction_1, [4.0, 5.0, 6.0])
        nptest.assert_allclose(plane.direction_2, [7.0, 8.0, 9.0])
        nptest.assert_allclose(
            np.dot(plane.direction_1, plane.normal), 0.0, atol=1e-8
        )
        nptest.assert_allclose(
            np.dot(plane.direction_2, plane.normal), 0.0, atol=1e-8
        )

    @staticmethod
    def test_intersects():
        """
        Test intersections
        """
        element = Plane.xy([0.0, 0.0], [1.0, 0.0], LimiterEffect.STOP)

        # Check intersection.
        ray_origin = np.array([0.5, -0.5, 0.0])
        ray_direction = np.array([0.0, 1.0, 0.0])
        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [0.0, -1.0, 0.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [0.0, 1.0, 0.0])

        # Check no intersection.
        ray_origin = np.array([-0.5, -0.5, 0.0])
        ray_direction = np.array([0.0, 1.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        ray_origin = np.array([-0.5, 1.5, 0.0])
        ray_direction = np.array([0.0, 1.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        # Check parallel lines.
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert not intersects

        # Vertical element.
        element = Plane.xy([0.0, 0.0], [0.0, 1.0], LimiterEffect.STOP)

        # Check intersection.
        ray_origin = np.array([-0.5, 0.5, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [1.0, 0.0, 0.0])

        # Check no intersection.
        ray_origin = np.array([-0.5, 0.5, 0.0])
        ray_direction = np.array([0.0, 1.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        ray_origin = np.array([-0.5, 1.5, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        # Check parallel lines.
        ray_origin = np.array([0.0, -0.5, 0.0])
        ray_direction = np.array([0.0, 1.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        # Diagonal element.
        inv_sqrt_2 = 1.0 / np.sqrt(2)

        # Vertical element.
        element = Plane.xy([0.0, 0.0], [1.0, 1.0], LimiterEffect.STOP)

        # Check intersection.
        ray_origin = np.array([0.0, 1.0, 0.0])
        ray_direction = np.array([1.0, -1.0, 0.0])
        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-inv_sqrt_2, inv_sqrt_2, 0.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [inv_sqrt_2, -inv_sqrt_2, 0.0])

        # Check no intersection.
        ray_origin = np.array([-0.2, -0.1, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        ray_origin = np.array([0.9, 1.1, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        # Check parallel lines.
        ray_origin = np.array([0.0, 0.1, 0.0])
        ray_direction = np.array([1.0, 1.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects


class TestDisk:
    """
    Unit tests for Disk.
    """

    @staticmethod
    def test_disk_intersects():
        """
        Test intersection with disk.
        """
        element = Disk(0.0, 1.0, 2.0, LimiterEffect.STOP)

        # Test intersection.
        ray_origin = np.array([1.5, 0.0, -0.5])
        ray_direction = np.array([0.0, 0.0, 1.0])

        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [0.0, 0.0, -1.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [0.0, 0.0, 1.0])

        # Test intersection with diagonal ray.
        ray_origin = np.array([1.0, 0.0, -0.5])
        ray_direction = np.array([1.0, 0.0, 1.0])

        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [0.0, 0.0, -1.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [0.0, 0.0, 1.0])

        # Test non intersection.
        ray_origin = np.array([0.5, 0.0, -0.5])
        ray_direction = np.array([0.0, 0.0, 1.0])

        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects

        ray_origin = np.array([2.5, 0.0, -0.5])
        ray_direction = np.array([0.0, 0.0, 1.0])

        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        intersects, *_ = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert not intersects


class TestCylinder:
    """
    Unit tests for Cylinder.
    """

    @staticmethod
    def test_intersects():
        """
        Test intersection with cylinder.
        """
        element = Cylinder(1.0, 0.0, 1.0, LimiterEffect.STOP)

        # Test intersection.
        ray_origin = np.array([0.5, 0.0, 0.5])
        ray_direction = np.array([1.0, 0.0, 0.0])

        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [1.0, 0.0, 0.0])

        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_direction = np.array([2.0, 0.0, 2.0])

        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [1.0, 0.0, 0.0])

        ray_origin = np.array([0.0, 0.0, -0.5])
        ray_direction = np.array([2.0, 0.0, 1.5])

        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])

        # Test no intersection.
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_direction = np.array([0.0, 0.0, 1.0])

        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        ray_origin = np.array([0.0, 0.0, -0.5])
        ray_direction = np.array([1.0, 0.0, 0.0])

        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects


class TestCappedCone:
    """
    Unit tests for CappedCone.
    """

    @staticmethod
    def test_cone_intersects():
        """
        Test intersections with element.
        """
        inv_sqrt_2 = 1 / np.sqrt(2.0)

        element = CappedCone(1.0, 2.0, 0.0, 1.0, LimiterEffect.STOP)

        nptest.assert_allclose(element._sign_axis, 1.0)
        nptest.assert_allclose(element._z_min, 0.0)
        nptest.assert_allclose(element._z_max, 1.0)
        nptest.assert_allclose(element._cos_theta2, 0.5)
        nptest.assert_allclose(element._z_vertex, -1.0)

        # Test intersection.
        ray_origin = np.array([1.0, 0.0, 0.1])
        ray_direction = np.array([0.2, 0.0, 0.0])
        intersects, s, normal = element.intersects(ray_origin, ray_direction)

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-inv_sqrt_2, 0.0, inv_sqrt_2])

        intersects, s, normal = element.intersects(
            ray_origin + ray_direction, -ray_direction
        )

        assert intersects
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [inv_sqrt_2, 0.0, -inv_sqrt_2])

        # Test no intersection.
        ray_origin = np.array([0.9, 0.0, 0.3])
        ray_direction = np.array([0.2, 0.0, 0.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_direction = np.array([0.2, 0.0, -0.3])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_direction = np.array([0.0, 0.0, 1.0])
        intersects, *_ = element.intersects(ray_origin, ray_direction)

        assert not intersects

    @staticmethod
    def test_from_rz():
        """
        Test CappedCone.from_rz gives correct elements.
        """
        element = CappedCone.from_rz(
            [1.0, 0.0], [1.0, 0.0], LimiterEffect.STOP
        )
        assert isinstance(element, Disk)

        element = CappedCone.from_rz(
            [1.0, 0.0], [0.0, 1.0], LimiterEffect.STOP
        )
        assert isinstance(element, Cylinder)

        element = CappedCone.from_rz(
            [1.0, 0.0], [1.0, 1.0], LimiterEffect.STOP
        )
        assert isinstance(element, CappedCone)


class TestLimiter:
    """
    Unit tests for Limiter.
    """

    @staticmethod
    def test_round_trip_netcdf():
        """
        Test writing and reading object through netCDF4 file gives the same
        object.
        """
        elements = [
            Plane.xy([0.0, 0.0], [1.0, 0.0], LimiterEffect.STOP),
            Plane.xy([1.0, 0.0], [0.0, 1.0], LimiterEffect.REFLECT_SPECULAR),
            Plane.xy([1.0, 1.0], [-1.0, 0.0], LimiterEffect.REFLECT_POLARISER),
            Plane.xy([0.0, 1.0], [0.0, -1.0], LimiterEffect.STOP),
            Disk(-2.0, 1.0, 2.0, LimiterEffect.STOP),
            Cylinder(2.0, -2.0, -1.0, LimiterEffect.STOP),
            CappedCone(1.0, 2.0, -5.0, -4.0, LimiterEffect.STOP),
        ]

        limiter = Limiter(elements, 0.1)

        with tempfile.TemporaryFile("r+") as f, nc4.Dataset(f, "w") as dset:
            Dimensions().write_netcdf(dset)
            limiter.write_netcdf(dset)
            limiter_2 = Limiter.read_netcdf(dset)

        assert np.isclose(
            limiter.extinction_coefficient_nepers,
            limiter_2.extinction_coefficient_nepers,
        )

        assert len(limiter.elements) == len(limiter_2.elements)

        for element_1, element_2 in zip(
            limiter.elements, limiter_2.elements, strict=True
        ):
            assert type(element_1) is type(element_2)
            assert element_1.effect == element_2.effect

    @staticmethod
    def test_intersects():
        """
        Test intersections with element.
        """
        limiter = Limiter(
            [
                Plane.xy([-1.0, 0.0], [1.0, 0.0], LimiterEffect.STOP),
                Plane.xy([0.0, 0.0], [0.0, 1.0], LimiterEffect.STOP),
                Plane.xy([0.0, 1.0], [-1.0, 0.0], LimiterEffect.STOP),
            ],
            0.1,
        )

        # Test intersection.
        ray_origin = np.array([-0.5, 0.5, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])

        intersects, idx, s, normal, effect, extinction_coefficient = (
            limiter.intersects(ray_origin, ray_direction)
        )

        assert intersects
        assert idx == 1
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])
        assert effect == LimiterEffect.STOP
        nptest.assert_allclose(extinction_coefficient, 0.1)

        # Test ignoring intersection.
        intersects, *_ = limiter.intersects(
            ray_origin, ray_direction, ignore_idx=1
        )

        assert not intersects

    @staticmethod
    def test_intersects_double():
        """
        Test intersections with 2 elements correctly gives closest element.
        """
        limiter = Limiter(
            [
                Plane.xy([-0.1, 0.0], [0.0, 1.0], LimiterEffect.STOP),
                Plane.xy([0.0, 0.0], [0.0, 1.0], LimiterEffect.STOP),
                Plane.xy([0.1, 0.0], [0.0, 1.0], LimiterEffect.STOP),
            ],
            0.1,
        )

        point_1 = [-1.0, 0.5, 0.0]
        point_2 = [3.0, 0.5, 0.0]
        intersects, idx, s, normal, effect, extinction_coefficient = (
            limiter.intersects(point_1, point_2)
        )

        # Test intersection.
        ray_origin = np.array([-0.5, 0.5, 0.0])
        ray_direction = np.array([1.0, 0.0, 0.0])

        intersects, idx, s, normal, effect, extinction_coefficient = (
            limiter.intersects(ray_origin, ray_direction)
        )

        assert intersects
        assert idx == 0
        nptest.assert_allclose(s, 0.4)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])
        assert effect == LimiterEffect.STOP
        nptest.assert_allclose(extinction_coefficient, 0.1)

        # Test ignoring intersection.
        intersects, idx, s, normal, effect, extinction_coefficient = (
            limiter.intersects(ray_origin, ray_direction, ignore_idx=0)
        )

        assert intersects
        assert idx == 1
        nptest.assert_allclose(s, 0.5)
        nptest.assert_allclose(normal, [-1.0, 0.0, 0.0])
        assert effect == LimiterEffect.STOP
        nptest.assert_allclose(extinction_coefficient, 0.1)
