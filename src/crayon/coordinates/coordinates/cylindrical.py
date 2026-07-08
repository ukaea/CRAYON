"""
Object for a cylindrical coordinate system.
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
    ForwardTransformDerivatives,
    map_angle_minus_pi_to_pi,
)
from crayon.coordinates.coordinates.cartesian import CARTESIAN
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class Cylindrical(ForwardTransformDerivatives, BackwardTransformDerivatives):
    """
    Global cylindrical coordinate system (r, phi, z).
    """

    __slots__ = ()

    coordinate_system = CoordinateSystem.CYLINDRICAL
    orthogonal = True

    forward_transform_preferred_coordinate = CoordinateSystem.CARTESIAN
    backward_transform_preferred_coordinate = CoordinateSystem.CYLINDRICAL

    def __init__(self):
        """
        Inits cylindrical coordinate system.
        """
        super().__init__(CARTESIAN)

    @staticmethod
    def bound_components(
        position_cylindrical: FloatArray, /, *, return_array: FloatArray = None
    ) -> FloatArray:
        """
        Bound the coordinate components to within coordinate system limits.

        Parameters
        ----------
        position_cylindrical : np.array[float]
            Coordinate components (r, phi, z).
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

        r, phi, z = position_cylindrical

        return_array[0] = abs(r)

        if r < 0.0:
            # Passing through r = 0 results in pi rotation.
            phi += np.pi

        return_array[1] = map_angle_minus_pi_to_pi(phi)
        return_array[2] = z

        return return_array

    def forward_transform(
        self,
        position_cartesian: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert Cartesian position to cylindrical.

        Parameters
        ----------
        position_cartesian : np.array[float]
            Position components in Cartesian coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_cylindrical : np.array[float]
            Position components in cylindrical coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        x, y, z = position_cartesian

        return_array[0] = np.sqrt(x * x + y * y)
        return_array[1] = np.arctan2(y, x)
        return_array[2] = z

        self.bound_components(return_array, return_array=return_array)

        return return_array

    def backward_transform(
        self,
        position_cylindrical: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert cylindrical position to Cartesian.

        Parameters
        ----------
        position_cylindrical : np.array[float]
            Position components in cylindrical coordinate system.
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

        r, phi, z = position_cylindrical

        return_array[0] = r * np.cos(phi)
        return_array[1] = r * np.sin(phi)
        return_array[2] = z

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
        First derivative of cylindrical position with respect to Cartesian
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

        x, y, _ = position_cartesian
        x2, y2 = x * x, y * y
        r2 = x2 + y2
        r = np.sqrt(np.square(x) + np.square(y))
        r_inv = 1 / r
        r2_inv = 1 / r2

        return_array.fill(0.0)
        return_array[0, 0] = x * r_inv
        return_array[0, 1] = y * r_inv
        return_array[1, 0] = -y * r2_inv
        return_array[1, 1] = x * r2_inv
        return_array[2, 2] = 1.0

        return return_array

    def forward_transform_dx2(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Second derivative of cylindrical position with respect to Cartesian
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
        forward_transform_dx2 : np.array[float]
            Second derivative of forward transform.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
            FloatType,
        )

        if self.is_coordinate(coordinate_system):
            position_cartesian = self.backward_transform(position)
        else:
            position_cartesian = position

        x, y, _ = position_cartesian
        x2, y2 = x * x, y * y
        r2 = x2 + y2
        r = np.sqrt(r2)
        r3_inv = 1 / (r * r2)
        r4_inv = 1 / (r2 * r2)

        return_array.fill(0.0)

        return_array[0, 0, 0] = y2 * r3_inv
        return_array[0, 0, 1] = -x * y * r3_inv

        return_array[0, 1, 0] = -x * y * r3_inv
        return_array[0, 1, 1] = x2 * r3_inv

        return_array[1, 0, 0] = 2 * x * y * r4_inv
        return_array[1, 0, 1] = (y2 - x2) * r4_inv

        return_array[1, 1, 0] = (y2 - x2) * r4_inv
        return_array[1, 1, 1] = -2 * x * y * r4_inv

        return return_array

    def forward_transform_dx3(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Third derivative of cylindrical position with respect to Cartesian
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
        forward_transform_dx3 : np.array[float]
            Third derivative of forward transform.
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
            position_cartesian = self.backward_transform(position)
        else:
            position_cartesian = position

        x, y, _ = position_cartesian
        x2, y2 = x * x, y * y
        x3, y3 = x * x2, y * y2
        r2 = x2 + y2
        r = np.sqrt(r2)
        r4 = r2 * r2
        r5_inv = 1 / (r * r4)
        r6_inv = 1 / (r2 * r4)

        return_array.fill(0.0)

        return_array[0, 0, 0, 0] = -3 * x * y2 * r5_inv
        return_array[0, 0, 0, 1] = (2 * x2 * y - y3) * r5_inv
        return_array[0, 0, 1, 0] = return_array[0, 0, 0, 1]
        return_array[0, 0, 1, 1] = (2 * x * y2 - x3) * r5_inv

        return_array[0, 1, 0, 0] = (2 * x2 * y - y3) * r5_inv
        return_array[0, 1, 0, 1] = (2 * x * y2 - x3) * r5_inv
        return_array[0, 1, 1, 0] = return_array[0, 1, 0, 1]
        return_array[0, 1, 1, 1] = -3 * x2 * y * r5_inv

        return_array[1, 0, 0, 0] = 2 * y * (y2 - 3 * x2) * r6_inv
        return_array[1, 0, 0, 1] = 2 * x * (x2 - 3 * y2) * r6_inv
        return_array[1, 0, 1, 0] = return_array[1, 0, 0, 1]
        return_array[1, 0, 1, 1] = -2 * y * (y2 - 3 * x2) * r6_inv

        return_array[1, 1, 0, 0] = 2 * x * (x2 - 3 * y2) * r6_inv
        return_array[1, 1, 0, 1] = 2 * y * (3 * x2 - y2) * r6_inv
        return_array[1, 1, 1, 0] = return_array[1, 1, 0, 1]
        return_array[1, 1, 1, 1] = 2 * x * (3 * y2 - x2) * r6_inv

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
        First derivative of Cartesian position with respect to cylindrical
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
            position_cylindrical = position
        else:
            position_cylindrical = self.forward_transform(position)

        r, phi, _ = position_cylindrical
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        return_array.fill(0.0)

        return_array[0, 0] = cos_phi
        return_array[0, 1] = -r * sin_phi
        return_array[1, 0] = sin_phi
        return_array[1, 1] = r * cos_phi
        return_array[2, 2] = 1

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
        Second derivative of Cartesian position with respect to cylindrical
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
            position_cylindrical = position
        else:
            position_cylindrical = self.forward_transform(position)

        r, phi, _ = position_cylindrical
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        return_array.fill(0.0)

        return_array[0, 0, 1] = -sin_phi

        return_array[0, 1, 0] = -sin_phi
        return_array[0, 1, 1] = -r * cos_phi

        return_array[1, 0, 1] = cos_phi

        return_array[1, 1, 0] = cos_phi
        return_array[1, 1, 1] = -r * sin_phi

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
        Third derivative of Cartesian position with respect to cylindrical
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
            position_cylindrical = position
        else:
            position_cylindrical = self.forward_transform(position)

        r, phi, _ = position_cylindrical
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)

        return_array.fill(0.0)

        return_array[0, 0, 1, 1] = -cos_phi

        return_array[0, 1, 0, 1] = -cos_phi
        return_array[0, 1, 1, 0] = return_array[0, 1, 0, 1]
        return_array[0, 1, 1, 1] = r * sin_phi

        return_array[1, 0, 1, 1] = -sin_phi

        return_array[1, 1, 0, 1] = -sin_phi
        return_array[1, 1, 1, 0] = return_array[1, 1, 0, 1]
        return_array[1, 1, 1, 1] = -r * cos_phi

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
    def read_netcdf(cls) -> "Cylindrical":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        cylindrical : Cylindrical
            Cylindrical coordinate system object.
        """
        return cls()


CYLINDRICAL = Cylindrical()
