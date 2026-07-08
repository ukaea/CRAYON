"""
Classes for handling intersection of rays with limiter.
"""

# Standard imports
import abc
import logging
import typing

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.dimensions import Dimension, Dimensions
from crayon.shared.io import write_netcdf_variable
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)


class LimiterEffect(CrayonEnum):
    """
    Effect applied on intersection with limiter.

    Attributes
    ----------
    STOP
        Stop ray tracing if ray intersects with this element.
    REFLECT
        Apply a specular reflection to any ray intersecting this element.
    REFLECT_POLARISER
        Apply a specular reflection to any ray intersecting this element plus
        switch O and X mode mode fractions.
    """

    STOP = 1
    REFLECT_SPECULAR = 2
    REFLECT_POLARISER = 3


def reflect_specular(vector: FloatArray, normal: FloatArray) -> FloatArray:
    """
    Apply specular reflection to 2D vector about a normal vector.

    Parameters
    ----------
    vector : np.array[float]
        Vector to reflect.
    normal : np.array[float]
        Normal to reflect about.

    Returns
    -------
    reflected_vector : np.array[float]
        Reflected vector.
    """
    overlap = np.dot(vector, normal) / np.linalg.norm(normal)
    return vector - 2.0 * overlap * normal


# Tolerance on distances. Don't care about anything smaller than 0.1mm.
EPS = 1.0e-4
EPS2 = EPS * EPS


class LimiterElementType(CrayonEnum):
    """
    Enumeration for limiter element types.

    Attributes
    ----------
    PLANE
        Plane.
    DISK
        Disk aligned with z axis.
    CYLINDER
        Cylinder aligned with z axis.
    CAPPED_CONE
        Capped cone aligned with z axis.
    """

    PLANE = 1
    DISK = 2
    CYLINDER = 3
    CAPPED_CONE = 4


class LimiterElementBase(abc.ABC):
    """
    Base class for a limiter element.

    Attributes
    ----------
    effect : LimiterEffect
        Effect applied to ray on intersection.

    Methods
    -------
    intersects
        Check if line between two points intersects the limiter.
    """

    __slots__ = ("effect",)

    def __init__(
        self,
        effect: LimiterEffect,
    ):
        """
        Inits LimiterElementBase.

        Parameters
        ----------
        effect : LimiterEffect
            Effect applied to ray on intersection.
        """
        self.effect = effect

    @abc.abstractmethod
    def intersects(self, point_1: FloatArray, point_2: FloatArray):
        """
        Check if line between two points intersects the limiter.

        Parameters
        ----------
        point_1 : np.array[float]
            Start point.
        point_2 : np.array[float]
            End point.
        """

    @abc.abstractmethod
    def pack(
        self,
        origin: FloatArray,
        normal: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
    ):
        """
        Pack limiter element data into array.

        Parameters
        ----------
        origin : np.array[float]
            Information about origin of element.
        normal : np.array[float]
            Information about normal to element.
        direction_1 : np.array[float]
            Information about first direction along element.
        direction_2 : np.array[float]
            Information about second direction along element.
        """


class Plane(LimiterElementBase):
    """
    Plane limiter element.

    Attributes
    ----------
    origin : np.array[float]
        Origin of element.
    direction_1 : np.array[float]
        Direction along first side of element from origin.
    direction_2 : np.array[float]
        Direction along second side of element from origin.
    normal : np.array[float]
        Normal vector to plane at origin.

    Methods
    -------
    xy
        Create 2D Cartesian element in (x, y) plane.
    xz
        Create 2D Cartesian element in (x, z) plane.
    yz
        Create 2D Cartesian element in (y, z) plane.
    xyz
        Create Plane from origin and two directions.
    intersects
        Determine if ray intersects plane.
    pack
        Pack limiter element data into arrays.
    """

    __slots__ = (
        "direction_1",
        "direction_2",
        "normal",
        "origin",
    )

    def __init__(
        self,
        origin: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
        normal: FloatArray,
        effect: LimiterEffect,
    ):
        """
        Inits Plane.

        Parameters
        ----------
        origin : np.array[float]
            Origin of element.
        direction_1 : np.array[float]
            Direction along first side of element from origin.
        direction_2 : np.array[float]
            Direction along second side of element from origin.
        normal : np.array[float]
            Normal vector to plane at origin.
        effect : LimiterEffect
            Effect applied to ray on intersection.

        Raises
        ------
        ValueError
            normal has zero magnitude.
        """
        super().__init__(effect)

        self.origin = np.asarray(origin).reshape((3,))
        self.direction_1 = np.asarray(direction_1).reshape((3,))
        self.direction_2 = np.asarray(direction_2).reshape((3,))

        # Construct normal.
        self.normal = np.array(normal)
        magnitude = np.linalg.norm(self.normal)

        if np.isclose(magnitude, 0.0):
            raise ValueError("normal must have non-zero size.")

        self.normal /= magnitude

    @classmethod
    def _2d(
        cls,
        origin: FloatArray,
        direction: FloatArray,
        effect: LimiterEffect,
        ix: int,
        iy: int,
    ) -> "Plane":
        """
        Create 2D Cartesian element extended across entire third dimension.

        Parameters
        ----------
        origin : np.array[float]
            Origin of element. Must have shape (2,).
        direction : np.array[float]
            Direction along element from origin. Must have shape (2,).
        effect : LimiterEffect
            Effect applied to ray on intersection.
        ix, iy : int
            Index of Cartesian coordinate components defining limiter elements
            e.g. ix=0, iy=1 would be (x, y), ix=1, iy=2 would be (y, z).

        Returns
        -------
        limiter_element : Plane
            Limiter element in 2d plane.
        """
        origin = np.asarray(origin).reshape((2,))
        direction = np.asarray(direction).reshape((2,))

        _origin = np.zeros(3)
        _origin[ix] = origin[0]
        _origin[iy] = origin[1]

        _direction_1 = np.zeros(3)
        _direction_1[ix] = direction[0]
        _direction_1[iy] = direction[1]

        # Dummy direction ensures intersection cannot be out of range.
        _direction_2 = np.zeros(3)

        normal = np.zeros(3)
        normal[ix] = direction[1]
        normal[iy] = -direction[0]

        return cls(_origin, _direction_1, _direction_2, normal, effect)

    @classmethod
    def xy(
        cls, origin: FloatArray, direction: FloatArray, effect: LimiterEffect
    ) -> "Plane":
        """
        Create 2D Cartesian element in (x, y) plane. This extends for all
        z values.

        Parameters
        ----------
        origin : np.array[float]
            Origin of element. Must have shape (2,).
        direction : np.array[float]
            Direction along element from origin. Must have shape (2,).
        effect : LimiterEffect
            Effect applied to ray on intersection.

        Returns
        -------
        limiter_element : Plane
            Limiter element in (x, y) plane.
        """
        return cls._2d(origin, direction, effect, 0, 1)

    @classmethod
    def xz(
        cls, origin: FloatArray, direction: FloatArray, effect: LimiterEffect
    ) -> "Plane":
        """
        Create 2D Cartesian element in (x, z) plane. This extends for all
        y values.

        Parameters
        ----------
        origin : np.array[float]
            Origin of element. Must have shape (2,).
        direction : np.array[float]
            Direction along element from origin. Must have shape (2,).
        effect : LimiterEffect
            Effect applied to ray on intersection.

        Returns
        -------
        limiter_element : Plane
            Limiter element in (x, z) plane.
        """
        return cls._2d(origin, direction, effect, 0, 2)

    @classmethod
    def yz(
        cls, origin: FloatArray, direction: FloatArray, effect: LimiterEffect
    ) -> "Plane":
        """
        Create 2D Cartesian element in (y, z) plane. This extends for all
        x values.

        Parameters
        ----------
        origin : np.array[float]
            Origin of element. Must have shape (2,).
        direction : np.array[float]
            Direction along element from origin. Must have shape (2,).
        effect : LimiterEffect
            Effect applied to ray on intersection.

        Returns
        -------
        limiter_element : Plane
            Limiter element in (y, z) plane.
        """
        return cls._2d(origin, direction, effect, 1, 2)

    @classmethod
    def xyz(
        cls,
        origin: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
        effect: LimiterEffect,
    ) -> "Plane":
        """
        Create Plane from origin and two directions.

        Parameters
        ----------
        origin : np.array[float]
            Oirigin
        direction_1 : np.array[float]
            Direction along first side of element from origin.
        direction_2 : np.array[float]
            Direction along second side of element from origin.
        effect : LimiterEffect
            Effect applied to ray on intersection.

        Returns
        -------
        limiter_element : LimiterElementCartesian
            Limiter element.

        Raises
        ------
        ValueError
            direction_1 or direction_2 has size zero.
            direction_1 and direction_2 are parallel.
        """
        origin = np.asarray(origin).reshape((3,))
        direction_1 = np.asarray(direction_1).reshape((3,))
        direction_2 = np.asarray(direction_2).reshape((3,))

        if np.isclose(np.dot(direction_1, direction_1), 0.0):
            raise ValueError("direction_1 must have non-zero size.")

        if np.isclose(np.dot(direction_2, direction_2), 0.0):
            raise ValueError("direction_2 must have non-zero size.")

        # Construct normal.
        normal = np.cross(direction_1, direction_2)
        magnitude = np.linalg.norm(normal)

        if np.isclose(magnitude, 0.0):
            raise ValueError("direction_1 and direction_2 cannot be parallel.")

        normal /= magnitude

        return cls(origin, direction_1, direction_2, normal, effect)

    def intersects(
        self,
        ray_origin: FloatArray,
        ray_direction: FloatArray,
    ) -> tuple[bool, float, FloatArray]:
        """
        Determine if ray intersects plane.

        Parameters
        ----------
        ray_origin : np.array[float]
            Starting point of ray.
        ray_direction : np.array[float]
            Direction of ray.

        Returns
        -------
        intersects : bool
            Flag if the element is intersected.
        s: float
            Normalised distance along line between point_1 and point_2 the
            intersection occurs.
        normal : array
            3D vector giving normal vector for applying reflections.
        """
        ray_origin = np.asarray(ray_origin)
        ray_direction = np.asarray(ray_direction)
        crossover = np.dot(ray_direction, self.normal)

        if np.isclose(crossover, 0.0):
            # Plane and line are co-planar and never intersect.
            return False, np.inf, None

        # Lines are not parallel.
        origin_diff = ray_origin - self.origin
        s = -np.dot(origin_diff, self.normal) / crossover

        if s < 0 or s > 1:
            # Intersection occurs outside line segment.
            return False, np.inf, None

        # Check if intersection occurs within limiter element.
        intersection_minus_origin = origin_diff + s * ray_direction
        t = np.dot(intersection_minus_origin, self.direction_1)

        if t < 0 or t > np.dot(self.direction_1, self.direction_1):
            # Intersection occurs outside limiter element in direction 1.
            return False, np.inf, None

        t = np.dot(intersection_minus_origin, self.direction_2)

        if t < 0 or t > np.dot(self.direction_2, self.direction_2):
            # Intersection occurs outside limiter element in direction 2.
            return False, np.inf, None

        # Get correct normal.
        normal = self.normal.copy()

        if np.dot(self.normal, ray_direction) > 0:
            normal *= -1.0

        return True, s, normal

    def pack(
        self,
        origin: FloatArray,
        normal: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
    ):
        """
        Pack limiter element data into arrays.

        Parameters
        ----------
        origin : np.array[float]
            Information about origin of element.
        normal : np.array[float]
            Information about normal to element.
        direction_1 : np.array[float]
            Information about first direction along element.
        direction_2 : np.array[float]
            Information about second direction along element.
        """
        origin[:] = self.origin
        normal[:] = self.normal
        direction_1[:] = self.direction_1
        direction_2[:] = self.direction_2


class Disk(LimiterElementBase):
    """
    Disk parallel to z axis.

    Attributes
    ----------
    z : float
        Z value of plane disk lies in.
    r_min : float
        Minimum radius of disk.
    r_max : float
        Maximum radius of disk.
    """

    __slots__ = ("r_max", "r_min", "z")

    def __init__(
        self, z: float, r_min: float, r_max: float, effect: LimiterEffect
    ):
        """
        Inits Disk.

        Parameters
        ----------
        z : float
            Z value of plane disk lies in.
        r_min : float
            Minimum radius of disk.
        r_max : float
            Maximum radius of disk.
        effect : LimiterEffect
            Effect applied to ray on intersection.
        """
        super().__init__(effect)

        self.z = float(z)
        self.r_min = float(r_min)
        self.r_max = float(r_max)

    def intersects(
        self,
        ray_origin: FloatArray,
        ray_direction: FloatArray,
    ) -> tuple[bool, float, FloatArray]:
        """
        Determine if ray intersects plane.

        Parameters
        ----------
        ray_origin : np.array[float]
            Starting point of ray.
        ray_direction : np.array[float]
            Direction of ray.

        Returns
        -------
        intersects : bool
            Flag if the element is intersected.
        s: float
            Normalised distance along line between point_1 and point_2 the
            intersection occurs.
        normal : array
            3D vector giving normal vector for applying reflections.
        """
        ray_origin = np.asarray(ray_origin)
        ray_direction = np.asarray(ray_direction)

        if np.isclose(ray_direction[2], 0.0):
            # Ray is parallel to plane of disk.
            return False, np.inf, None

        # Calculate radius at intersection.
        s = (self.z - ray_origin[2]) / ray_direction[2]

        if s < 0 or s > 1:
            # Intersection occurs outside line segment.
            return False, np.inf, None

        r = np.linalg.norm(ray_origin[:2] + s * ray_direction[:2])

        if r < self.r_min or r > self.r_max:
            # Ray doesn't intersect disk.
            return False, np.inf, None

        normal = np.zeros(3)

        if ray_direction[2] > 0.0:
            normal[2] = -1.0
        else:
            normal[2] = 1.0

        return True, s, normal

    def pack(
        self,
        origin: FloatArray,
        normal: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
    ):
        """
        Pack limiter element data into array.

        Parameters
        ----------
        origin : np.array[float]
            Information about origin of element.
        normal : np.array[float]
            Information about normal to element.
        direction_1 : np.array[float]
            Information about first direction along element.
        direction_2 : np.array[float]
            Information about second direction along element.
        """
        origin[:] = 0.0, 0.0, self.z
        normal[:] = 0.0, 0.0, 1.0
        direction_1[:] = self.r_min, 0.0, 0.0
        direction_2[:] = self.r_max, 0.0, 0.0


class Cylinder(LimiterElementBase):
    """
    Cylinder with axis parallel to z axis.

    Attributes
    ----------
    r : float
        Radius of cylinder.
    z_min : float
        Minimum z extent of cylinder.
    z_max : float
        Maximum z extent of cylinder.
    """

    __slots__ = ("r", "z_max", "z_min")

    def __init__(
        self, r: float, z_min: float, z_max: float, effect: LimiterEffect
    ):
        """
        Inits Cylinder.

        Parameters
        ----------
        r : float
            Radius of cylinder.
        z_min : float
            Minimum z extent of cylinder.
        z_max : float
            Maximum z extent of cylinder.
        effect : LimiterEffect
            Effect applied to ray on intersection.
        """
        super().__init__(effect)

        self.r = float(r)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def intersects(
        self,
        ray_origin: FloatArray,
        ray_direction: FloatArray,
    ) -> tuple[bool, float, FloatArray]:
        """
        Determine if ray intersects plane.

        Parameters
        ----------
        ray_origin : np.array[float]
            Starting point of ray.
        ray_direction : np.array[float]
            Direction of ray.

        Returns
        -------
        intersects : bool
            Flag if the element is intersected.
        s: float
            Normalised distance along line between point_1 and point_2 the
            intersection occurs.
        normal : array
            3D vector giving normal vector for applying reflections.
        """
        ray_origin = np.array(ray_origin)
        ray_direction = np.array(ray_direction)

        x0, y0, z0 = ray_origin
        dx, dy, dz = ray_direction

        if np.allclose((dx, dy), 0.0):
            # Ray parallel to z axis.
            return False, np.inf, None

        a = dx * dx + dy * dy
        b = x0 * dx + y0 * dy
        c = x0 * x0 + y0 * y0 - self.r * self.r

        disc = b * b - a * c

        if np.isclose(disc, 0.0, rtol=0.0, atol=EPS2):
            s = -0.5 * c / b if np.isclose(a, 0.0) else -b / a

            if s < 0 or s > 1:
                # Intersection occurs out of ray.
                return False, np.inf, None

            z_int = z0 + s * dz

            if self.z_min or self.z_max < z_int:
                # Intersection occurs out of cylinder.
                return False, np.inf, None

        elif disc < 0.0:
            # Ray doesn't intersect cylinder.
            return False, np.inf, None
        else:
            sqrt_disc = np.sqrt(disc)
            s_plus = (-b + sqrt_disc) / a
            s_minus = (-b - sqrt_disc) / a

            s_plus_in_range = s_plus > -EPS and s_plus < 1 + EPS
            s_minus_in_range = s_minus > -EPS and s_minus < 1 + EPS

            if not (s_plus_in_range or s_minus_in_range):
                # Intersection occurs out of ray.
                return False, np.inf, None

            z_plus = z0 + s_plus * dz
            z_minus = z0 + s_minus * dz

            z_plus_in_range = (
                s_plus_in_range
                and z_plus >= self.z_min
                and z_plus <= self.z_max
            )
            z_minus_in_range = (
                s_minus_in_range
                and z_minus >= self.z_min
                and z_minus <= self.z_max
            )

            if not (z_plus_in_range or z_minus_in_range):
                # No valid intersections.
                return False, np.inf, None
            if z_plus_in_range and not z_minus_in_range:
                # Only plus intersection valid.
                s = s_plus
            elif z_plus_in_range and not z_minus_in_range:
                # Only minus intersection valid.
                s = s_minus
            else:
                # Both intersection points valid.
                s = min(s_plus, s_minus)

        # Calculate normal vector.
        normal = np.zeros(3)

        x_int, y_int, z_int = ray_origin + s * ray_direction
        r_int = np.sqrt(x_int * x_int + y_int * y_int)

        normal[0] = -x_int / r_int
        normal[1] = -y_int / r_int

        if np.dot(normal, ray_direction) > 0.0:
            normal *= -1

        return True, s, normal

    def pack(
        self,
        origin: FloatArray,
        normal: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
    ):
        """
        Pack limiter element data into array.

        Parameters
        ----------
        origin : np.array[float]
            Information about origin of element.
        normal : np.array[float]
            Information about normal to element.
        direction_1 : np.array[float]
            Information about first direction along element.
        direction_2 : np.array[float]
            Information about second direction along element.
        """
        origin[:] = 0.0, 0.0, self.z_min
        normal[:] = 0.0, 0.0, 0.0
        direction_1[:] = self.r, 0.0, 0.0
        direction_2[:] = 0.0, 0.0, self.z_max - self.z_min


class CappedCone(LimiterElementBase):
    """
    Capped cone.

    Attributes
    ----------
    r_a : float
        Radius at one end point of capped cone.
    r_b : float
        Radius at other end point of capped cone.
    z_a : float
        Height at one end point of capped cone.
    z_b : float
        Height at one end point of capped cone.
    """

    __slots__ = (
        "_cos_theta2",
        "_sign_axis",
        "_z_max",
        "_z_min",
        "_z_vertex",
        "r_a",
        "r_b",
        "z_a",
        "z_b",
    )

    def __init__(
        self,
        r_a: float,
        r_b: float,
        z_a: float,
        z_b: float,
        effect: LimiterEffect,
    ):
        """
        Inits CappedCone.

        Parameters
        ----------
        r_a : float
            Radius at one end point of capped cone.
        r_b : float
            Radius at other end point of capped cone.
        z_a : float
            Height at one end point of capped cone.
        z_b : float
            Height at one end point of capped cone.
        effect : LimiterEffect
            Effect applied to ray on intersection.
        """
        super().__init__(effect)

        self.r_a = float(r_a)
        self.r_b = float(r_b)
        self.z_a = float(z_a)
        self.z_b = float(z_b)

        self._z_min = min(self.z_a, self.z_b)
        self._z_max = max(self.z_a, self.z_b)

        # Direction along axis.
        self._sign_axis = np.sign(z_b - z_a)

        # Calculate cos(theta)**2 where theta is slant angle of cone.
        axis_size2 = np.square(self._z_max - self._z_min)
        rr = np.square(self.r_b - self.r_a)
        self._cos_theta2 = axis_size2 / (rr * rr + axis_size2)

        # Calculate z position at vertex of cone.
        dz = self.r_a * (self.z_b - self.z_a) / (self.r_b - self.r_a)
        self._z_vertex = z_a - dz

    @classmethod
    def from_rz(
        cls, origin: FloatArray, direction: FloatArray, effect: LimiterEffect
    ) -> typing.Union["CappedCone", "Disk", "Cylinder"]:
        """
        Construct limiter element from (r, z) origin and direction. May return
        a Disk, Cylinder or CappedCone depending on values.

        Parameters
        ----------
        origin : np.array[float]
            (r, z) origin of element. Must have shape (2,).
        direction : np.array[float]
            (r, z) direction along element from origin. Must have shape (2,).
        effect : LimiterEffect
            Effect applied to ray on intersection.

        Returns
        -------
        limiter_element : CappedCone | Disk | Cylinder
            Limiter element. May return a Disk or Cylinder for special cases.
        """
        origin = np.asarray(origin).reshape((2,))
        direction = np.asarray(direction).reshape((2,))

        r_a, z_a = origin
        r_b, z_b = r_a + direction[0], z_a + direction[1]

        if abs(r_a - r_b) < EPS:
            return Cylinder(r_a, min(z_a, z_b), max(z_a, z_b), effect)

        if abs(z_a - z_b) < EPS:
            return Disk(z_a, min(r_a, r_b), max(r_a, r_b), effect)

        return cls(r_a, r_b, z_a, z_b, effect)

    def _cone_function(self, x: np.ndarray[float]) -> float:
        """
        Function which is zero on the surface of the cone.

        Parameters
        ----------
        x : np.array[float]
            Position.

        Returns
        -------
        cone_function : float
            Value which is zero on the surface of the cone.
        """
        return (1 - self._cos_theta2) * np.square(
            x[2] - self._z_vertex
        ) - self._cos_theta2 * (x[0] * x[0] + x[1] * x[1])

    def intersects(
        self,
        ray_origin: FloatArray,
        ray_direction: FloatArray,
    ) -> tuple[bool, float, FloatArray]:
        """
        Determine if line intersects capped cone.

        Parameters
        ----------
        ray_origin : np.array[float]
            Starting point of ray.
        ray_direction : np.array[float]
            Direction of ray.

        Returns
        -------
        intersects : bool
            Flag if the element is intersected.
        s: float
            Normalised distance along line between point_1 and point_2 the
            intersection occurs.
        normal : array
            3D vector giving normal vector for applying reflections.
        """
        ray_origin = np.asarray(ray_origin)
        ray_direction = np.asarray(ray_direction)
        direction2 = np.dot(ray_direction, ray_direction)

        delta = ray_origin.copy()
        delta[2] -= self._z_vertex

        d_dot_u = self._sign_axis * ray_direction[2]
        d_dot_delta = self._sign_axis * delta[2]
        u_dot_delta = np.dot(ray_direction, delta)
        delta_dot_delta = np.dot(delta, delta)

        a = d_dot_u * d_dot_u - direction2 * self._cos_theta2
        b = d_dot_u * d_dot_delta - self._cos_theta2 * u_dot_delta
        c = d_dot_delta * d_dot_delta - self._cos_theta2 * delta_dot_delta

        disc = b * b - c * a

        if np.isclose(a, 0.0):
            # Single intersection.
            t = -0.5 * c / b
            intersection = ray_origin + t * ray_direction

            if self._cone_function(intersection) < 0:
                return False, np.inf, None

            z = t * d_dot_u + d_dot_delta

            if z < self._z_min or z > self._z_max:
                return False, np.inf, None
        elif np.isclose(disc, 0.0):
            # Tangent intersection (double root).
            t = -0.5 * b / a

            intersection = ray_origin + t * ray_direction

            if self._cone_function(intersection) < 0:
                return False, np.inf, None

            z = ray_origin[2] + t * ray_direction[2]

            if z < self._z_min or z > self._z_max:
                return False, np.inf, None
        else:
            # Double intersection.
            if disc < 0.0:
                return False, np.inf, None

            sqrt_disc = np.sqrt(disc)
            t_plus = (-b + sqrt_disc) / a
            t_minus = (-b - sqrt_disc) / a

            t_plus_in_range = t_plus >= -EPS and t_plus <= 1.0 + EPS
            t_minus_in_range = t_minus >= -EPS and t_minus <= 1.0 + EPS

            if not (t_plus_in_range or t_minus_in_range):
                return False, np.inf, None

            z_plus = ray_origin[2] + t_plus * ray_direction[2]
            z_minus = ray_origin[2] + t_minus * ray_direction[2]

            z_plus_in_range = (
                t_plus_in_range
                and z_plus >= self._z_min
                and z_plus <= self._z_max
            )
            z_minus_in_range = (
                t_minus_in_range
                and z_minus >= self._z_min
                and z_minus <= self._z_max
            )

            if not (z_plus_in_range or z_minus_in_range):
                # No valid intersections.
                return False, np.inf, None
            if z_plus_in_range and not z_minus_in_range:
                # Only plus intersection valid.
                t = t_plus
            elif z_minus_in_range and not z_plus_in_range:
                # Only minus intersection valid.
                t = t_minus
            else:
                # Both intersection points valid.
                t = min(t_plus, t_minus)

            intersection = ray_origin + t * ray_direction

        # Calculate normal vector.
        ox = intersection - ray_origin
        vx = intersection.copy()
        vx[2] -= self._z_vertex
        normal = ox - np.dot(ox, vx) * vx / np.dot(vx, vx)
        normal /= np.linalg.norm(normal)

        # Correct normal to point against direction.
        if np.dot(normal, ray_direction) > 0.0:
            normal *= -1

        return True, t, normal

    def pack(
        self,
        origin: FloatArray,
        normal: FloatArray,
        direction_1: FloatArray,
        direction_2: FloatArray,
    ):
        """
        Pack limiter element data into array.

        Parameters
        ----------
        origin : np.array[float]
            Information about origin of element.
        normal : np.array[float]
            Information about normal to element.
        direction_1 : np.array[float]
            Information about first direction along element.
        direction_2 : np.array[float]
            Information about second direction along element.
        """
        origin[:] = 0.0, 0.0, self._z_vertex
        normal[:] = 0.0, 0.0, 0.0
        direction_1[:] = self.r_a, 0.0, self.z_a
        direction_2[:] = self.r_b, 0.0, self.z_b


LimiterElement = Plane | Disk | Cylinder | CappedCone


class Limiter:
    """
    Collection of limiter elements.

    Attributes
    ----------
    elements : list[LimiterElementBase]
        List of limiter elements.
    extinction_coefficient_nepers : float
        Extinction coefficient applied on reflection [nepers].

    Methods
    -------
    write_netcdf
        Write to netCDF4 dataset.
    read_netcdf
        Read from netCDF4 dataset.
    intersects
        Determine if line connecting point_1 to point_2 intersects the element.
    """

    __slots__ = (
        "elements",
        "extinction_coefficient_nepers",
    )

    def __init__(
        self,
        elements: list[LimiterElement],
        extinction_coefficient_nepers: float,
    ):
        """
        Inits Limiter.

        Parameters
        ----------
        elements : list[LimiterElement]
            Limiter elements.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on reflection [nepers].

        Raises
        ------
        ValueError
            extinction_coefficient_nepers is negative.
        """
        self.elements = elements
        self.extinction_coefficient_nepers = float(
            extinction_coefficient_nepers
        )

        if self.extinction_coefficient_nepers < 0:
            raise ValueError(
                "extinction_coefficient_nepers must be positive: "
                f"{extinction_coefficient_nepers}"
            )

    def write_netcdf(self, group: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to write data to.

        Raises
        ------
        TypeError
            Unknown limiter type.
        """
        group.setncattr(
            "extinction_coefficient_nepers", self.extinction_coefficient_nepers
        )

        # Create dimension for elements.
        element_number = Dimension("element_number", len(self.elements))
        group.createDimension(element_number.name, element_number.size)

        # Save element type, origin and dimension.
        element_type = np.empty(element_number.size, dtype=int)
        effect_type = np.empty(element_number.size, dtype=int)
        origin = np.empty((element_number.size, 3), dtype=float)
        normal = np.empty((element_number.size, 3), dtype=float)
        direction_1 = np.empty((element_number.size, 3), dtype=float)
        direction_2 = np.empty((element_number.size, 3), dtype=float)

        for i, element in enumerate(self.elements):
            if isinstance(element, Plane):
                element_type[i] = LimiterElementType.PLANE.value
            elif isinstance(element, Disk):
                element_type[i] = LimiterElementType.DISK.value
            elif isinstance(element, Cylinder):
                element_type[i] = LimiterElementType.CYLINDER.value
            elif isinstance(element, CappedCone):
                element_type[i] = LimiterElementType.CAPPED_CONE.value
            else:
                raise TypeError(element)

            effect_type[i] = element.effect.value
            element.pack(origin[i], normal[i], direction_1[i], direction_2[i])

        write_netcdf_variable(
            group,
            "element_type",
            (element_number,),
            element_type,
            "Type of limiter element",
            "",
        )

        write_netcdf_variable(
            group,
            "effect_type",
            (element_number,),
            effect_type,
            "Effect applied on intersection with limiter",
            "",
        )

        write_netcdf_variable(
            group,
            "origin",
            (element_number, Dimensions.x),
            origin,
            "Origin of limiter element",
            "m",
        )

        write_netcdf_variable(
            group,
            "normal",
            (element_number, Dimensions.x),
            normal,
            "Normal of limiter element",
            "m",
        )

        write_netcdf_variable(
            group,
            "direction_1",
            (element_number, Dimensions.x),
            direction_1,
            "First direction of limiter element",
            "m",
        )

        write_netcdf_variable(
            group,
            "direction_2",
            (element_number, Dimensions.x),
            direction_2,
            "Second direction of limiter element",
            "m",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Limiter":
        """
        Read from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to write data to.

        Returns
        -------
        limiter : Limiter
            Collection of limiter elements.

        Raises
        ------
        ValueError
            Unknown limiter element type.
        """
        extinction_coefficient_nepers = group.getncattr(
            "extinction_coefficient_nepers"
        )

        element_type = group["element_type"][:].data
        effect_type = group["effect_type"][:].data
        origin = group["origin"][:].data
        normal = group["normal"][:].data
        direction_1 = group["direction_1"][:].data
        direction_2 = group["direction_2"][:].data

        elements = []

        for (
            _element_type,
            _effect,
            _origin,
            _normal,
            _direction_1,
            _direction_2,
        ) in zip(
            element_type,
            effect_type,
            origin,
            normal,
            direction_1,
            direction_2,
            strict=True,
        ):
            __element_type = LimiterElementType(_element_type)
            __effect = LimiterEffect(_effect)

            if __element_type == LimiterElementType.PLANE:
                element = Plane(
                    _origin, _direction_1, _direction_2, _normal, __effect
                )
            elif __element_type == LimiterElementType.DISK:
                element = Disk(
                    _origin[2], _direction_1[0], _direction_2[0], __effect
                )
            elif __element_type == LimiterElementType.CYLINDER:
                element = Cylinder(
                    _direction_1[0],
                    _origin[2],
                    _origin[2] + _direction_2[2],
                    __effect,
                )
            elif __element_type == LimiterElementType.CAPPED_CONE:
                element = CappedCone(
                    _direction_1[0],
                    _direction_2[0],
                    _direction_1[2],
                    _direction_2[2],
                    __effect,
                )
            else:
                raise ValueError(__element_type)

            elements.append(element)

        return cls(elements, extinction_coefficient_nepers)

    @abc.abstractmethod
    def intersects(
        self,
        ray_origin: FloatArray,
        ray_direction: FloatArray,
        /,
        *,
        ignore_idx: int | None = None,
    ) -> tuple[bool, int, float, FloatArray, LimiterEffect, float]:
        """
        Determine if line connecting point_1 to point_2 intersects the element.

        Parameters
        ----------
        ray_origin : np.array[float]
            Starting point of ray.
        ray_direction : np.array[float]
            Direction of ray.
        ignore_idx : int, optional
            If provided, intersections with the given element index will be
            ignored.

        Returns
        -------
        intersects : bool
            Flag if the element is intersected.
        idx : int
            Index of intersected element.
        s: float
            Normalised distance along line between point_1 and point_2 the
            intersection occurs.
        normal : array
            2D vector giving normal vector for applying reflections.
        effect : LimiterEffect
            Effect applied to ray on intersection.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].
        """
        intersects = False
        idx = -1
        s = np.inf
        normal = np.empty(3)
        effect = LimiterEffect.STOP

        for i, element in enumerate(self.elements):
            if i == ignore_idx:
                continue

            _intersects, _s, _normal = element.intersects(
                ray_origin, ray_direction
            )

            if _intersects and _s < s:
                intersects = True
                idx = i
                s = _s
                normal[:] = _normal
                effect = element.effect

        return (
            intersects,
            idx,
            s,
            normal,
            effect,
            self.extinction_coefficient_nepers,
        )
