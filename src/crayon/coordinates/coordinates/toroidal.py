"""
Object for a toroidal coordinate system.
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
from crayon.coordinates.coordinates.cylindrical import CYLINDRICAL
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.io import write_netcdf_variable
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class Toroidal(BackwardTransformDerivatives):
    """
    Global Toroidal coordinate system (r, phi, theta).
    """

    __slots__ = ("axis_m",)

    coordinate_system = CoordinateSystem.TOROIDAL
    orthogonal = True

    backward_transform_preferred_coordinate = CoordinateSystem.TOROIDAL

    def __init__(self, axis_m: tuple[float, float]):
        """
        Inits toroidal coordinate system.

        Raises
        ------
        ValueError
            axis_m doesn't have shape (2,)
        """
        super().__init__(CYLINDRICAL)

        self.axis_m = np.asarray(axis_m, dtype=FloatType)

        if self.axis_m.shape != (2,):
            raise ValueError(
                "axis_m has incorrect shape."
                f"Expected (2,), got {self.axis_m.shape}"
            )

    @staticmethod
    def bound_components(
        position_toroidal: FloatArray, /, *, return_array: FloatArray = None
    ) -> FloatArray:
        """
        Bound the coordinate components to within coordinate system limits.

        Parameters
        ----------
        position_toroidal : np.array[float]
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

        r, phi, theta = position_toroidal

        return_array[0] = abs(r)
        return_array[1] = map_angle_minus_pi_to_pi(phi)
        return_array[2] = map_angle_minus_pi_to_pi(theta)

        return return_array

    def forward_transform(
        self,
        position_cylindrical: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert Cartesian position to toroidal.

        Parameters
        ----------
        position_cylindrical : np.array[float]
            Position components in cylindrical coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_toroidal : np.array[float]
            Position components in toroidal coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        r, phi, z = position_cylindrical
        r_ax, z_ax = self.axis_m
        dr, dz = r - r_ax, z - z_ax

        return_array[0] = np.sqrt(np.square(dr) + np.square(dz))
        return_array[1] = phi
        return_array[2] = np.arctan2(dz, dr)

        self.bound_components(return_array, return_array=return_array)

        return return_array

    def backward_transform(
        self,
        position_toroidal: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert toroidal position to Cartesian.

        Parameters
        ----------
        position_toroidal : np.array[float]
            Position components in toroidal coordinate system.
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

        r, phi, theta = position_toroidal
        r_ax, z_ax = self.axis_m

        return_array[0] = r_ax + r * np.cos(theta)
        return_array[1] = phi
        return_array[2] = z_ax + r * np.sin(theta)

        self.parent_coordinate.bound_components(
            return_array, return_array=return_array
        )

        return return_array

    def backward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        First derivative of Cartesian position with respect to toroidal
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

        if self.is_coordinate(coordinate_system):
            position_toroidal = position
        else:
            position_toroidal = self.forward_transform(position)

        r, _, theta = position_toroidal
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        return_array.fill(0.0)

        return_array[0, 0] = cos_theta
        return_array[0, 2] = -r * sin_theta

        return_array[1, 1] = 1.0

        return_array[2, 0] = sin_theta
        return_array[2, 2] = r * cos_theta

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
        Second derivative of Cartesian position with respect to toroidal
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
            position_toroidal = position
        else:
            position_toroidal = self.forward_transform(position)

        r, _, theta = position_toroidal
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        return_array.fill(0.0)

        return_array[0, 0, 2] = -sin_theta

        return_array[0, 2, 0] = return_array[0, 0, 2]
        return_array[0, 2, 2] = -r * cos_theta

        return_array[2, 0, 2] = cos_theta
        return_array[2, 2, 0] = return_array[2, 0, 2]
        return_array[2, 2, 2] = -r * sin_theta

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
        Third derivative of Cartesian position with respect to toroidal
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
            position_toroidal = position
        else:
            position_toroidal = self.forward_transform(position)

        r, _, theta = position_toroidal
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        return_array.fill(0.0)

        return_array[0, 0, 2, 2] = -cos_theta

        return_array[0, 2, 0, 2] = return_array[0, 0, 2, 2]
        return_array[0, 2, 2, 0] = return_array[0, 0, 2, 2]
        return_array[0, 2, 2, 2] = r * sin_theta

        return_array[2, 0, 2, 2] = -sin_theta
        return_array[2, 2, 0, 2] = return_array[2, 0, 2, 2]
        return_array[2, 2, 2, 0] = return_array[2, 0, 2, 2]
        return_array[2, 2, 2, 2] = -r * cos_theta

        return return_array

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.coordinate_system.name)

        write_netcdf_variable(
            group,
            "axis",
            (Dimensions.two,),
            self.axis_m,
            "r-z position of origin of toroidal coordinates",
            "m",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "Toroidal":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        toroidal : Toroidal
            Toroidal coordinate system object.
        """
        axis_m = group["axis"][:].data
        return cls(axis_m)
