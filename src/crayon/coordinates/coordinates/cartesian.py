"""
Object for a Cartesian coordinate system.
"""

# Standard imports
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813

# Local imports
from crayon.coordinates.coordinates.base import (
    CoordinateSystem,
    CurvilinearCoordinate,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class Cartesian(CurvilinearCoordinate):
    """
    Global Cartesian coordinate system (x, y, z).
    """

    __slots__ = ()

    coordinate_system = CoordinateSystem.CARTESIAN
    parent_coordinate_system = CoordinateSystem.CARTESIAN
    orthogonal = True

    def __init__(self):
        """
        Inits Cartesian coordinate.
        """

    @staticmethod
    def bound_components(
        position_cartesian: FloatArray, /, *, return_array: FloatArray = None
    ) -> FloatArray:
        """
        Bound the coordinate components to within coordinate system limits.

        Parameters
        ----------
        position_cartesian : np.array[float]
            Coordinate components (x, y, z).
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

        return_array[:] = position_cartesian

        return return_array

    def forward_transform(self, *args, **kwargs):
        """
        Cartesian has no parent so this is not defined.
        """
        raise NotImplementedError("Should not call this")

    def backward_transform(self, *args, **kwargs):
        """
        Cartesian has no parent so this is not defined.
        """
        raise NotImplementedError("Should not call this")

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
    def read_netcdf(cls) -> "Cartesian":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        cartesian : Cartesian
            Cartesian coordinate system object.
        """
        return cls()


CARTESIAN = Cartesian()
