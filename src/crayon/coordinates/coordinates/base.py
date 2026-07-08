"""
Base classes for objects representing curvilinear coordinate systems.
"""

# Standard imports
import abc
import logging
import math

# Third party imports
import numpy as np

# Local imports
from crayon.shared.constants import CoordinateSystem
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.io import IONetcdf
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


def map_angle_minus_pi_to_pi(angle):
    """
    Map an angle in radians to its equivalent value on [-pi, pi).

    Parameters
    ----------
    angle : float
        Angle to be mapped.

    Returns
    -------
    angle_mapped : float
        Angle mapped to equivalent value on [-pi, pi).
    """
    return math.remainder(angle, 2 * np.pi)


class CurvilinearCoordinate(IONetcdf):
    """
    Curvilinear coordinates system base class.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        Coordinate system enum this object represents.
    orthogonal : bool
        If True, the coordinate system is orthogonal.
    forward_transform_derivatives : bool
        If True, the coordinate system implements higher than first order
        derivatives of the forward transform.
    backward_transform_derivatives : bool
        If True, the coordinate system implements higher than first order
        derivatives of the backward transform.

    Methods
    -------
    bound_components
        Bound components of the coordinate system to correct range.
    forward_transform
        Transform coordinate system components from parent coordinate.
    backward_transform
        Transform coordinate system components to parent coordinate.
    contravariant_transform
        Calculate contravariant forward transform.
    covariant_transform
        Calculate covariant forward transform.
    write_netcdf
        Write coordinate data into netCDF format.
    read_netcdf
        Create coordinate object from netCDF format data.
    """

    __slots__ = ("parent_coordinate",)

    # Coordinate system corresponding to this object.
    coordinate_system = NotImplemented
    orthogonal = NotImplemented

    forward_transform_derivatives = False
    backward_transform_derivatives = False

    def __init__(self, parent_coordinate: "CurvilinearCoordinate"):
        """
        Inits CurvilinearCoordinate.

        Parameters
        ----------
        parent_coordinate : CurvilinearCoordinate
            Coordinate system this coordinate system is derived from.
        """
        self.parent_coordinate = parent_coordinate

    @property
    def parent_coordinate_system(self) -> CoordinateSystem:
        """CoordinateSystem for parent coordinate."""
        return self.parent_coordinate.coordinate_system

    def is_coordinate(self, coordinate_system: CoordinateSystem) -> bool:
        """
        Flag if coordinate system in pair is this coordinate system or the
        parent.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system to check.

        Returns
        -------
        is_coordinate : bool
            True if the provided coordinate system is this coordinate, False
            if it is the parent coordinate.

        Raises
        ------
        ValueError
            Provided coordinate system is neither this coordinate nor its
            parent.
        """
        if coordinate_system == self.coordinate_system:
            return True
        if coordinate_system == self.parent_coordinate_system:
            return False
        raise ValueError(coordinate_system)

    @abc.abstractmethod
    def bound_components(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Bound the coordinate components to within the defined ranges. Useful
        for mapping periodic coordinates.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        bounded_position
            Position components bounded to coordinate system limits.
        """

    @abc.abstractmethod
    def forward_transform(
        self,
        position: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert parent coordinate components p to this coordinate system q.

        Parameters
        ----------
        position : np.array[float]
            Position components in parent coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_new : np.array[float]
            Position components in this coordinate system.
        """

    @abc.abstractmethod
    def backward_transform(
        self,
        position: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert coordinate components q to parent coordinate components p.

        Parameters
        ----------
        position : np.array[float]
            Position components in this coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_new : np.array[float]
            Position components in parent coordinate system.
        """

    @staticmethod
    def contravariant_transform(
        forward_transform_dx: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Forward transform for contravariantly transfoming quantities
        f^i_j = dp^i / dq^j.

        Parameters
        ----------
        forward_transform_dx : np.array[float]
            First derivative of forward transform.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        contravariant_transform : np.array[float]
            Contravariant forward transform.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), FloatType
        )

        np.copyto(return_array, forward_transform_dx)

        return return_array

    @staticmethod
    def covariant_transform(
        backward_transform_dx: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Forward transform for covariantly transforming quantities
        f_i^j = dq^i / dp^j.

        Parameters
        ----------
        backward_transform_dx : np.array[float]
            First derivative of backward transform.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        contravariant_transform : np.array[float]
            Contravariant forward transform.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), FloatType
        )

        # NOTE: the transpose is very important.
        np.copyto(return_array, backward_transform_dx.T)

        return return_array


class ForwardTransformDerivatives(CurvilinearCoordinate):
    """
    Curvilinear coordinate defining higher order derivatives of the forward
    transform.

    Attributes
    ----------
    forward_transform_preferred_coordinate : bool
        Preferred coordinate system components used to calculate the forward
        transform and its derivatives.

    Methods
    -------
    forward_transform_dx
        First derivative of coordinate components with respect to parent
        coordinate components.
    forward_transform_dx2
        Second derivative of coordinate components with respect to parent
        coordinate components.
    forward_transform_dx3
        Third derivative of coordinate components with respect to parent
        coordinate components.
    """

    __slots__ = ()

    forward_transform_derivatives = True
    forward_transform_preferred_coordinate = NotImplemented

    def __init__(self, parent_coordinate):
        super().__init__(parent_coordinate)

    @abc.abstractmethod
    def forward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        First derivative of position components q with respect to components
        in parent coordinate system p
        i.e. forward_transform_dx[i, j] = dq^i / dp^j

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx : np.array[float]
            First derivative of forward transform.
        """

    @abc.abstractmethod
    def forward_transform_dx2(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Second derivative of position components q with respect to components
        in parent coordinate system p
        i.e. forward_transform_dx2[i, j, k] = d^2 q^i / dp^j dp^k

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx2 : np.array[float]
            Second derivative of forward transform.
        """

    @abc.abstractmethod
    def forward_transform_dx3(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Third derivative of position components q with respect to components
        in parent coordinate system p
        i.e. forward_transform_dx2[i, j, k, l] = d^3 q^i / dp^j dp^k dp^l

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx3 : np.array[float]
            Third derivative of forward transform.
        """

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
        First derivative of components in parent coordinate system p with
        respect to components q
        i.e. backward_transform_dx[i, j] = dp^i / dq^i

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
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

        if forward_transform_dx is None:
            forward_transform_dx = self.forward_transform_dx(
                position, coordinate_system
            )

        return_array[:, :] = np.linalg.inv(forward_transform_dx)

        return return_array


class BackwardTransformDerivatives(CurvilinearCoordinate):
    """
    Curvilinear coordinate defining higher order derivatives of the backward
    transform.

    Attributes
    ----------
    backward_transform_preferred_coordinate : bool
        Preferred coordinate system components used to calculate the backward
        transform and its derivatives.

    Methods
    -------
    backward_transform_dx
        First derivative of parent coordinate components with respect to
        coordinate components.
    backward_transform_dx2
        Second derivative of parent coordinate components with respect to
        coordinate components.
    backward_transform_dx3
        Third derivative of parent coordinate components with respect to
        coordinate components.
    """

    __slots__ = ()

    backward_transform_derivatives = True
    backward_transform_preferred_coordinate = NotImplemented

    def __init__(self, parent_coordinate):
        super().__init__(parent_coordinate)

    @abc.abstractmethod
    def backward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        First derivative of components in parent coordinate system p with
        respect to components q
        i.e. backward_transform_dx[i, j] = dp^i / dq^i

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        backward_transform_dx : np.array[float]
            First derivative of backward transform.
        """

    @abc.abstractmethod
    def backward_transform_dx2(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Second derivative of components in parent coordinate system p with
        respect to components q
        i.e. backward_transform_dx[i, j, k] = d^2 p^i / dq^j dq^k

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        backward_transform_dx2 : np.array[float]
            Second derivative of backward transform.
        """

    @abc.abstractmethod
    def backward_transform_dx3(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Third derivative of components in parent coordinate system p with
        respect to components q
        i.e. backward_transform_dx[i, j, k, l] = d^3 p^i / dq^j dq^k dq^l

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        backward_transform_dx3 : np.array[float]
            Third derivative of backward transform.
        """

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
        First derivative of position components q with respect to components
        in parent coordinate system p
        i.e. forward_transform_dx[i, j] = dq^i / dp^j

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either this coordinate or the parent coordinate.
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

        if backward_transform_dx is None:
            backward_transform_dx = self.backward_transform_dx(
                position, coordinate_system
            )

        return_array[:, :] = np.linalg.inv(backward_transform_dx)

        return return_array
