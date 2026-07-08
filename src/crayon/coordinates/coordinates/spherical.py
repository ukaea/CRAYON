"""
Object for a spherical coordinate system.
"""

# Standard imports
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np

# Local imports
from crayon.coordinates.coordinates.base import (
    BackwardTransformDerivatives,
    CoordinateSystem,
    map_angle_minus_pi_to_pi,
)
from crayon.coordinates.coordinates.cartesian import CARTESIAN
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class Spherical(BackwardTransformDerivatives):
    """
    Global Spherical coordinate system (r, theta, phi).
        r in [0, inf) is the radius from the origin.
        theta in [0, pi] is the poloidal angle measured from the Cartesian
            z axis.
        phi in [-pi, pi] is the aximuthal angle measured from the Cartesian
            x axis in the Cartesian x-y plane.
    """

    __slots__ = ()

    coordinate_system = CoordinateSystem.SPHERICAL
    orthogonal = True

    backward_transform_preferred_coordinate = CoordinateSystem.SPHERICAL

    def __init__(self):
        """
        Inits spherical coordinate system.
        """
        super().__init__(CARTESIAN)

    @staticmethod
    def bound_components(
        position_spherical: FloatArray, /, *, return_array: FloatArray = None
    ) -> FloatArray:
        """
        Bound the coordinate components to within coordinate system limits.

        Parameters
        ----------
        position_spherical : np.array[float]
            Coordinate components (r, phi, theta).
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        bounded_position
            Position components bounded to coordinate system limits.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), FloatType
        )

        r, theta, phi = position_spherical

        return_array[0] = abs(r)

        if r < 0.0:
            # Passing through r = 0 is mirror across origin.
            theta += np.pi

        return_array[1] = map_angle_minus_pi_to_pi(theta)

        if return_array[1] < 0.0:
            # Passing through theta = 0 gives pi shift in phi.
            return_array[1] = abs(return_array[1])
            phi += np.pi

        return_array[2] = map_angle_minus_pi_to_pi(phi)

        return return_array

    def forward_transform(
        self,
        position_cartesian: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert Cartesian position to spherical.

        Parameters
        ----------
        position_cartesian : np.array[float]
            Position components in Cartesian coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_spherical : np.array[float]
            Position components in spherical coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        x, y, z = position_cartesian

        perp2 = x * x + y * y
        perp = np.sqrt(perp2)

        return_array[0] = np.sqrt(perp2 + z * z)
        return_array[1] = np.arctan2(perp, z)
        return_array[2] = np.arctan2(y, x)

        self.bound_components(return_array, return_array=return_array)

        return return_array

    def backward_transform(
        self,
        position_spherical: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert spherical position to Cartesian.

        Parameters
        ----------
        position_spherical : np.array[float]
            Position components in spherical coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_cartesian : np.array[float]
            Position components in Cartesian coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        r, theta, phi = position_spherical
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        return_array[0] = r * sin_theta * cos_phi
        return_array[1] = r * sin_theta * sin_phi
        return_array[2] = r * cos_theta

        self.parent_coordinate.bound_components(
            return_array, return_array=return_array
        )

        return return_array

    def forward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
        backward_transform_dx: FloatArray = None,
    ):
        """
        First derivative of spherical position with respect to Cartesian
        position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        backward_transform_dx : np.array[float], optional
            First derivative of backward transform.

        Returns
        -------
        forward_transform_dx : np.array[float]
            First derivative of forward transform.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), FloatType
        )

        if backward_transform_dx is not None:
            return_array[:, :] = np.linalg.inv(backward_transform_dx)
            return return_array

        if self.is_coordinate(coordinate_system):
            position_cartesian = self.backward_transform(position)
        else:
            position_cartesian = position

        x, y, z = position_cartesian

        perp2 = x * x + y * y
        perp = np.sqrt(perp2)

        r = np.sqrt(perp2 + z * z)
        r_inv = 1 / r
        r2_inv = r_inv * r_inv

        return_array.fill(0.0)

        return_array[0, 0] = x * r_inv
        return_array[0, 1] = y * r_inv
        return_array[0, 2] = z * r_inv

        return_array[1, 2] = -perp * r2_inv

        perp_inv = 1 / perp
        return_array[1, 0] = x * z * perp_inv * r2_inv
        return_array[1, 1] = y * z * perp_inv * r2_inv

        perp_inv2 = perp_inv * perp_inv
        return_array[2, 0] = -y * perp_inv2
        return_array[2, 1] = x * perp_inv2

        return return_array

    def backward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
        forward_transform_dx: FloatArray = None,
    ):
        """
        First derivative of Cartesian position with respect to spherical
        position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        forward_transform_dx : np.array[float], optional
            First derivative of forward transform.

        Returns
        -------
        backward_transform_dx : np.array[float]
            First derivative of backward transform.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), FloatType
        )

        if forward_transform_dx is not None:
            return_array[:, :] = np.linalg.inv(forward_transform_dx)
            return return_array

        if self.is_coordinate(coordinate_system):
            position_spherical = position
        else:
            position_spherical = self.forward_transform(position)

        r, theta, phi = position_spherical
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        return_array.fill(0.0)

        return_array[0, 0] = sin_theta * cos_phi
        return_array[0, 1] = r * cos_theta * cos_phi
        return_array[0, 2] = -r * sin_theta * sin_phi

        return_array[1, 0] = sin_theta * sin_phi
        return_array[1, 1] = r * cos_theta * sin_phi
        return_array[1, 2] = r * sin_theta * cos_phi

        return_array[2, 0] = cos_theta
        return_array[2, 1] = -r * sin_theta

        return return_array

    def backward_transform_dx2(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Second derivative of Cartesian position with respect to spherical
        position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        backward_transform_dx2 : np.array[float]
            Second derivative of backward transform.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
            FloatType,
        )

        if self.is_coordinate(coordinate_system):
            position_spherical = position
        else:
            position_spherical = self.forward_transform(position)

        r, theta, phi = position_spherical
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        return_array.fill(0.0)

        return_array[0, 0, 1] = cos_theta * cos_phi
        return_array[0, 0, 2] = -sin_theta * sin_phi
        return_array[0, 1, 1] = -r * sin_theta * cos_phi
        return_array[0, 1, 2] = -r * cos_theta * sin_phi
        return_array[0, 2, 2] = -r * sin_theta * cos_phi

        return_array[0, 1, 0] = return_array[0, 0, 1]
        return_array[0, 2, 0] = return_array[0, 0, 2]
        return_array[0, 2, 1] = return_array[0, 1, 2]

        return_array[1, 0, 1] = cos_theta * sin_phi
        return_array[1, 0, 2] = sin_theta * cos_phi
        return_array[1, 1, 1] = -r * sin_theta * sin_phi
        return_array[1, 1, 2] = r * cos_theta * cos_phi
        return_array[1, 2, 2] = -r * sin_theta * sin_phi

        return_array[1, 1, 0] = return_array[1, 0, 1]
        return_array[1, 2, 0] = return_array[1, 0, 2]
        return_array[1, 2, 1] = return_array[1, 1, 2]

        return_array[2, 0, 1] = -sin_theta
        return_array[2, 1, 1] = -r * cos_theta

        return_array[2, 1, 0] = return_array[2, 0, 1]

        return return_array

    def backward_transform_dx3(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Third derivative of Cartesian position with respect to spherical
        position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        backward_transform_dx3 : np.array[float]
            Third derivative of backward transform.
        """
        return_array = get_return_array(
            return_array,
            (
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.x.size,
            ),
            FloatType,
        )

        if self.is_coordinate(coordinate_system):
            position_spherical = position
        else:
            position_spherical = self.forward_transform(position)

        r, theta, phi = position_spherical
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        return_array.fill(0.0)

        return_array[0, 0, 1, 1] = -sin_theta * cos_phi
        return_array[0, 0, 1, 2] = -cos_theta * sin_phi
        return_array[0, 0, 2, 2] = -sin_theta * cos_phi
        return_array[0, 1, 1, 1] = -r * cos_theta * cos_phi
        return_array[0, 1, 1, 2] = r * sin_theta * sin_phi
        return_array[0, 1, 2, 2] = -r * cos_theta * cos_phi
        return_array[0, 2, 2, 2] = r * sin_theta * sin_phi

        return_array[0, 1, 0, 1] = return_array[0, 0, 1, 1]
        return_array[0, 1, 1, 0] = return_array[0, 0, 1, 1]
        return_array[0, 0, 2, 1] = return_array[0, 0, 1, 2]
        return_array[0, 1, 0, 2] = return_array[0, 0, 1, 2]
        return_array[0, 1, 2, 0] = return_array[0, 0, 1, 2]
        return_array[0, 2, 0, 1] = return_array[0, 0, 1, 2]
        return_array[0, 2, 1, 0] = return_array[0, 0, 1, 2]
        return_array[0, 2, 0, 2] = return_array[0, 0, 2, 2]
        return_array[0, 2, 2, 0] = return_array[0, 0, 2, 2]
        return_array[0, 1, 2, 1] = return_array[0, 1, 1, 2]
        return_array[0, 2, 1, 1] = return_array[0, 1, 1, 2]
        return_array[0, 2, 1, 2] = return_array[0, 1, 2, 2]
        return_array[0, 2, 2, 1] = return_array[0, 1, 2, 2]

        return_array[1, 0, 1, 1] = -sin_theta * sin_phi
        return_array[1, 0, 1, 2] = cos_theta * cos_phi
        return_array[1, 0, 2, 2] = -sin_theta * sin_phi
        return_array[1, 1, 1, 1] = -r * cos_theta * sin_phi
        return_array[1, 1, 1, 2] = -r * sin_theta * cos_phi
        return_array[1, 1, 2, 2] = -r * cos_theta * sin_phi
        return_array[1, 2, 2, 2] = -r * sin_theta * cos_phi

        return_array[1, 1, 0, 1] = return_array[1, 0, 1, 1]
        return_array[1, 1, 1, 0] = return_array[1, 0, 1, 1]
        return_array[1, 0, 2, 1] = return_array[1, 0, 1, 2]
        return_array[1, 1, 0, 2] = return_array[1, 0, 1, 2]
        return_array[1, 1, 2, 0] = return_array[1, 0, 1, 2]
        return_array[1, 2, 0, 1] = return_array[1, 0, 1, 2]
        return_array[1, 2, 1, 0] = return_array[1, 0, 1, 2]
        return_array[1, 2, 0, 2] = return_array[1, 0, 2, 2]
        return_array[1, 2, 2, 0] = return_array[1, 0, 2, 2]
        return_array[1, 1, 2, 1] = return_array[1, 1, 1, 2]
        return_array[1, 2, 1, 1] = return_array[1, 1, 1, 2]
        return_array[1, 2, 1, 2] = return_array[1, 1, 2, 2]
        return_array[1, 2, 2, 1] = return_array[1, 1, 2, 2]

        return_array[2, 0, 1, 1] = -cos_theta
        return_array[2, 1, 1, 1] = r * sin_theta

        return_array[2, 1, 0, 1] = return_array[2, 0, 1, 1]
        return_array[2, 1, 1, 0] = return_array[2, 0, 1, 1]

        return return_array

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        dset.createGroup(self.coordinate_system.name)

    @classmethod
    def read_netcdf(cls) -> "Spherical":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        spherical : Spherical
            Spherical coordinate system object.
        """
        return cls()


SPHERICAL = Spherical()
