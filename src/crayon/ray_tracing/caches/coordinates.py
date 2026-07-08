"""
Caches for coordinate system data for ray tracing.
"""

# Standard imports
import itertools
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import (
    TensorType,
    first_covariant_derivative,
    first_derivative,
    second_covariant_derivative,
    second_derivative,
    third_derivative,
    transform_tensor_field,
)
from crayon.coordinates import (
    CoordinateCoordinator,
    CoordinateSystem,
    CurvilinearCoordinate,
    connection_coefficients,
    connection_coefficients_dx,
    forward_transform_dx2,
    forward_transform_dx3,
    metric_tensor,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)

X = Dimensions.x.size


class ForwardTransformCache:
    """
    Cache holding derivatives of forward transforms between coordinate systems.
    The forward transform relates coordinate components in a new coordinate
    system to its parent while the backward transform is the reverse.

    Attributes
    ----------
    backward_transform_dx : np.array[float]
        First derivative of backward transform.
    forward_transform_dx : np.array[float]
        First derivative of forward transform.
    forward_transform_dx2 : np.array[float]
        Second derivative of forward transform.
    forward_transform_dx3 : np.array[float]
        Third derivative of forward transform.

    Methods
    -------
    calculate_transform_derivatives
    """

    __slots__ = (
        "backward_transform_dx",
        "forward_transform_dx",
        "forward_transform_dx2",
        "forward_transform_dx3",
    )

    def __init__(self):
        """
        Inits ForwardTransformCache.
        """
        self.forward_transform_dx = np.empty((X, X), dtype=float)
        self.backward_transform_dx = np.empty((X, X), dtype=float)
        self.forward_transform_dx2 = np.empty((X, X, X), dtype=float)
        self.forward_transform_dx3 = np.empty((X, X, X, X), dtype=float)

    def calculate_transform_derivatives(
        self,
        coordinate: CurvilinearCoordinate,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        position_parent: FloatArray,
        parent_coordinate_system: CoordinateSystem,
    ):
        """
        Calculate derivatives of transforms.

        Parameters
        ----------
        coordinate : CurvilinearCoordinate
            Object representing curvilinear coordinate.
        position : np.array[float]
            Position in new coordinate system.
        coordinate_system : CoordinateSystem
            Coordinate system of new coordinate.
        position_parent : np.array[float]
            Position in parent coordinate system.
        parent_coordinate_system : CoordinateSystem
            Coordinate system of parent coordinate.

        Raises
        ------
        ValueError
            coordinate defines neither forward or backward transform higher
            order derivatives.
        """
        if coordinate.forward_transform_derivatives:
            # Select correct input position
            _preferred = coordinate.forward_transform_preferred_coordinate

            if coordinate_system == _preferred:
                _position = position
            elif parent_coordinate_system == _preferred:
                _position = position_parent
            else:
                raise ValueError(_preferred)

            # Calculate forward transform derivatives directly.
            coordinate.forward_transform_dx(
                _position, _preferred, return_array=self.forward_transform_dx
            )

            coordinate.forward_transform_dx2(
                _position, _preferred, return_array=self.forward_transform_dx2
            )

            coordinate.forward_transform_dx3(
                _position, _preferred, return_array=self.forward_transform_dx3
            )

            # Calculate backward transform from forward transform.
            coordinate.backward_transform_dx(
                _position,
                _preferred,
                forward_transform_dx=self.forward_transform_dx,
                return_array=self.backward_transform_dx,
            )

            return

        if coordinate.backward_transform_derivatives:
            # Select correct input position
            _preferred = coordinate.backward_transform_preferred_coordinate

            if coordinate_system == _preferred:
                _position = position
            elif parent_coordinate_system == _preferred:
                _position = position_parent
            else:
                raise ValueError(_preferred)

            coordinate.backward_transform_dx(
                _position, _preferred, return_array=self.backward_transform_dx
            )

            _backward_transform_dx2 = coordinate.backward_transform_dx2(
                _position,
                _preferred,
            )

            _backward_transform_dx3 = coordinate.backward_transform_dx3(
                _position,
                _preferred,
            )

            # Calculate forward transform derivatives from backward transform.
            coordinate.forward_transform_dx(
                _position,
                _preferred,
                backward_transform_dx=self.backward_transform_dx,
                return_array=self.forward_transform_dx,
            )

            forward_transform_dx2(
                self.forward_transform_dx,
                _backward_transform_dx2,
                return_array=self.forward_transform_dx2,
            )

            forward_transform_dx3(
                self.forward_transform_dx,
                self.forward_transform_dx2,
                _backward_transform_dx2,
                _backward_transform_dx3,
                return_array=self.forward_transform_dx3,
            )

            return

        raise ValueError(coordinate)


class TransformCache(ForwardTransformCache):
    """
    Cache holding objects required for transforming tensor fields between
    coordinate systems.

    Attributes
    ----------
    connection_coefficients : np.array[float]
        Connection coefficients aka Christoffel symbols of 2nd kind.
    connection_coefficients_dx : np.array[float]
        First derivative of connection coefficients with respect to x.
    contravariant_transform : np.array[float]
        Forward transform for contravariant tensor component.
    covariant_transform : np.array[float]
        Forward transform for covariant tensor component.
    inverse_metric : np.array[float]
        Inverse metric tensorl
    metric : np.array[float]
        Metric tensor.

    Methods
    -------
    compose
        Compose forward and backward transform derivatives with another
        transform cache.
    finalise
        Compute final tensor field transform objects.
    """

    __slots__ = (
        "connection_coefficients",
        "connection_coefficients_dx",
        "contravariant_transform",
        "covariant_transform",
        "inverse_metric",
        "metric",
    )

    def __init__(self):
        """
        Inits TransformCache.
        """
        super().__init__()

        self.covariant_transform = np.empty((X, X), dtype=float)
        self.contravariant_transform = np.empty((X, X), dtype=float)
        self.metric = np.empty((X, X), dtype=float)
        self.inverse_metric = np.empty((X, X), dtype=float)
        self.connection_coefficients = np.empty((X, X, X), dtype=float)
        self.connection_coefficients_dx = np.empty((X, X, X, X), dtype=float)

    def compose(self, other: ForwardTransformCache):
        """
        Compose forward and backward transform derivatives with another
        transform cache.

        Parameters
        ----------
        other : ForwardTransformCache
            Another cache of forward and backward transforms.

        Notes
        -----
        self holds derivatives from coordinate 2 to 3 while other holds
        derivatives from coordinate 1 to 2. The derivatives are then
        composed so self holds derivatives from coordinate 1 to 3.
        """
        # Take copies of the transforms for coordinate 2 to 3.
        # These will be overwritten but need all the old values available.
        _forward_transform_dx = np.copy(self.forward_transform_dx)
        _backward_transform_dx = np.copy(self.backward_transform_dx)
        _forward_transform_dx2 = np.copy(self.forward_transform_dx2)
        _forward_transform_dx3 = np.copy(self.forward_transform_dx3)

        _n = Dimensions.x.size
        _f_shape = (_n,)
        _g_size = _n
        _x_size = _n

        first_derivative(
            _forward_transform_dx,
            other.forward_transform_dx,
            _f_shape,
            _g_size,
            _x_size,
            return_array=self.forward_transform_dx,
        )

        first_derivative(
            other.backward_transform_dx,
            _backward_transform_dx,
            _f_shape,
            _g_size,
            _x_size,
            return_array=self.backward_transform_dx,
        )

        second_derivative(
            _forward_transform_dx,
            _forward_transform_dx2,
            other.forward_transform_dx,
            other.forward_transform_dx2,
            _f_shape,
            _g_size,
            _x_size,
            return_array=self.forward_transform_dx2,
        )

        third_derivative(
            _forward_transform_dx,
            _forward_transform_dx2,
            _forward_transform_dx3,
            other.forward_transform_dx,
            other.forward_transform_dx2,
            other.forward_transform_dx3,
            _f_shape,
            _g_size,
            _x_size,
            return_array=self.forward_transform_dx3,
        )

    def finalise(self):
        """
        Compute final tensor field transform objects. These are the
        contravariant transform, covariant transform, metric tensor,
        inverse metric tensor, connection coefficients and connection
        coefficients jacobian.

        Notes
        -----
        Only call once the forward and backward transforms have Cartesian
        as the parent coordinate.
        """
        CurvilinearCoordinate.contravariant_transform(
            self.forward_transform_dx,
            return_array=self.contravariant_transform,
        )
        CurvilinearCoordinate.covariant_transform(
            self.backward_transform_dx, return_array=self.covariant_transform
        )
        metric_tensor(self.covariant_transform, return_array=self.metric)
        connection_coefficients(
            self.backward_transform_dx,
            self.forward_transform_dx2,
            return_array=self.connection_coefficients,
        )
        connection_coefficients_dx(
            self.connection_coefficients,
            self.backward_transform_dx,
            self.forward_transform_dx3,
            return_array=self.connection_coefficients_dx,
        )

        self.inverse_metric = np.linalg.inv(self.metric)


class CoordinateCache:
    """
    Cache for coordinate system data.

    Attributes
    ----------
    position : dict[CoordinateSystem, np.array[float]]
        Position in different coordinate systems.
    transforms : dict[CoordinateSystem, TransformCache]
        Transform caches for each coordinate system.

    Methods
    -------
    set_position
        Calculate equivalent position in all coordinate systems.
    calculate_transforms
        Calculate coordinate transform objects for all coordinate systems.
    transform_basis
        Transform a tensor field between holonomic and physical basis.
    transform_tensor_field
        Transform a tensor field between coordinate systems.
    first_covariant_derivative
        Calculate first covariant derivative of a tensor field.
    second_covariant_derivative
        Calculate second covariant derivative of a tensor field.
    """

    __slots__ = (
        "_coordinate_coordinator",
        "_tmp_transform",
        "position",
        "transforms",
    )

    def __init__(
        self,
        coordinate_coordinator: CoordinateCoordinator,
    ):
        """
        Inits CoordinateCache.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Object coordinating transforms between coordinate systems.
        """
        self._coordinate_coordinator = coordinate_coordinator

        self.position: dict[CoordinateSystem, FloatArray] = {}
        self.transforms: dict[CoordinateSystem, TransformCache] = {}

        for coordinate_system in coordinate_coordinator.coordinates:
            self.position[coordinate_system] = np.empty(X, dtype=float)

            if coordinate_system == CoordinateSystem.CARTESIAN:
                continue

            self.transforms[coordinate_system] = TransformCache()

        # A temporary transform used to hold intermediate values.
        self._tmp_transform = ForwardTransformCache()

    @property
    def position_cartesian(self):
        """Position in Cartesian coordinate system."""
        return self.position[CoordinateSystem.CARTESIAN]

    def set_position(
        self,
        input_coordinate: CoordinateSystem,
        position: FloatArray,
        /,
        *,
        calculate_transforms: bool = True,
    ):
        """
        Calculate equivalent position in all coordinate systems.

        Parameters
        ----------
        input_coordinate : CoordinateSystem
            Coordinate system position components are provided in.
        position : np.array[float]
            Position components.
        calculate_transforms : bool
            If True, calculate coordinate transforms.
        """
        # List of converted coordinates.
        available_coordinates = dict.fromkeys(self.position, False)
        available_coordinates[input_coordinate] = True
        self.position[input_coordinate][:] = position

        # Convert input position to Cartesian if not provided.
        if input_coordinate != CoordinateSystem.CARTESIAN:
            # Get conversion path from input coordinate to Cartesian.
            # i.e. [input_coordinate, ..., Cartesian]
            conversion_path = self._coordinate_coordinator.get_conversion_path(
                input_coordinate, to_target=False
            )

            # Convert input position to Cartesian.
            for input_coordinate_system in itertools.islice(
                conversion_path, 0, len(conversion_path) - 1
            ):
                _root_coordinate = self._coordinate_coordinator.coordinates[
                    input_coordinate_system
                ].parent_coordinate_system
                self._coordinate_coordinator.convert_coordinate(
                    self.position[input_coordinate_system],
                    input_coordinate_system,
                    forward=False,
                    return_array=self.position[_root_coordinate][:],
                )

                # Mark coordinate as available.
                available_coordinates[_root_coordinate] = True

        # Convert Cartesian to all other coordinates.
        for target_coordinate in available_coordinates.copy():
            # If coordinate already available then skip.
            if available_coordinates[target_coordinate]:
                continue

            # Get conversion path from Cartesian to input coordinate.
            # i.e. [Cartesian, ..., input_coordinate]
            conversion_path = self._coordinate_coordinator.get_conversion_path(
                target_coordinate, to_target=True
            )

            # Convert Cartesian to target coordinate.
            for output_coordinate_system in itertools.islice(
                conversion_path, 1, None
            ):
                _root_coordinate = self._coordinate_coordinator.coordinates[
                    output_coordinate_system
                ].parent_coordinate_system

                self._coordinate_coordinator.convert_coordinate(
                    self.position[_root_coordinate],
                    output_coordinate_system,
                    forward=True,
                    return_array=self.position[output_coordinate_system][:],
                )

                # Mark coordinate as available.
                available_coordinates[output_coordinate_system] = True

        # If requested calculate covariant / contravariant transforms.
        if calculate_transforms:
            self.calculate_transforms()

    def calculate_transforms(self):
        """
        Calculate coordinate transform objects for all coordinate systems.
        """
        available_transforms = dict.fromkeys(self.transforms, False)

        for coordinate_system, transform in self.transforms.items():
            coordinate = self._coordinate_coordinator.coordinates[
                coordinate_system
            ]
            parent_coordinate_system = coordinate.parent_coordinate_system

            # Calculate transforms between coordinate and root.
            transform.calculate_transform_derivatives(
                coordinate,
                self.position[coordinate_system],
                coordinate_system,
                self.position[parent_coordinate_system],
                parent_coordinate_system,
            )

            # Get conversion path from input coordinate to Cartesian.
            # i.e. [coordinate_system, ..., Cartesian]
            conversion_path = self._coordinate_coordinator.get_conversion_path(
                coordinate_system, to_target=False
            )

            # Compose transforms for intermediate coordinates.
            # If there are no intermediate coordinates the iterator is empty.
            for input_coordinate_system in itertools.islice(
                conversion_path, 1, len(conversion_path) - 1
            ):
                # If transform from input coordinate to Cartesian already
                # calculated we can reuse it to jump to final answer.
                if available_transforms[input_coordinate_system]:
                    transform.compose(self.transforms[input_coordinate_system])
                    break

                _coordinate = self._coordinate_coordinator.coordinates[
                    input_coordinate_system
                ]
                _parent_coordinate = _coordinate.parent_coordinate_system

                # Calculate transforms between coordinate and root.
                self._tmp_transform.calculate_transform_derivatives(
                    _coordinate,
                    self.position[coordinate_system],
                    coordinate_system,
                    self.position[_parent_coordinate],
                    _parent_coordinate,
                )

                # Compose transforms between coordinates.
                transform.compose(self._tmp_transform)

            # Mark transform as available.
            available_transforms[coordinate_system] = True

            # Calculate final coordinate objects.
            transform.finalise()

    def transform_basis(
        self,
        coordinate_system: CoordinateSystem,
        tensor_field: FloatArray,
        tensor_type: TensorType,
        /,
        *,
        to_holonomic: bool,
        return_array: FloatArray = None,
    ) -> FloatArray:
        """
        Transform a tensor field between holonomic and physical basis. Only
        valid for orthogonal coordinate systems.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system the tensor field components are provided in.
        tensor_field : np.array[float]
            Tensor field components.
        tensor_type : TensorType
            Tensor type.
        to_holonomic : bool
            If True, convert from physical to holonomic basis and vice versa.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        tensor_field_new : np.array[float]
            Tensor field in new basis.

        Raises
        ------
        ValueError
            Coordinate system is not orthogonal.
        """
        return_array = get_return_array(
            return_array, tensor_field.shape, tensor_field.dtype
        )

        if tensor_type.rank == 1:
            if not (
                self._coordinate_coordinator.coordinates[
                    coordinate_system
                ].orthogonal
            ):
                raise ValueError(
                    "Physical basis only defined for orthogonal coordinate "
                    f"systems: {coordinate_system.name}"
                )

            # Get Lame coefficients.
            metric = self.transforms[coordinate_system].metric
            lame_coefficients = np.sqrt(np.diagonal(metric))

            if tensor_type.index_covariance[0]:
                # Covector.
                if to_holonomic:
                    return_array[:] = tensor_field * lame_coefficients
                else:
                    return_array[:] = tensor_field / lame_coefficients
            # Vector.
            elif to_holonomic:
                return_array[:] = tensor_field / lame_coefficients
            else:
                return_array[:] = tensor_field * lame_coefficients
        else:
            raise NotImplementedError(tensor_type.rank)

        return return_array

    def transform_tensor_field(
        self,
        input_coordinate: CoordinateSystem,
        output_coordinate: CoordinateSystem,
        tensor_field: FloatArray,
        tensor_type: TensorType,
        /,
        *,
        return_array: FloatArray = None,
    ) -> FloatArray:
        """
        Transform a tensor field between coordinate systems.

        Parameters
        ----------
        input_coordinate : CoordinateSystem
            Coordinate system the tensor field components are provided in.
        output_coordinate : CoordinateSystem
            Desired output coordinate system.
        tensor_field : np.array[float]
            Tensor field components.
        tensor_type : TensorType
            Tensor type.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        tensor_field_new : np.array[float]
            Tensor field components in new coordinate system.
        """
        return_array = get_return_array(
            return_array, tensor_field.shape, tensor_field.dtype
        )

        if input_coordinate == output_coordinate:
            np.copyto(return_array, tensor_field)
            return return_array

        # Transform to input to Cartesian.
        if input_coordinate == CoordinateSystem.CARTESIAN:
            np.copyto(return_array, tensor_field)
        else:
            transform = self.transforms[input_coordinate]

            covariant_transform = transform.covariant_transform

            contravariant_transform = transform.contravariant_transform

            # NOTE: The covariant / contravariant transforms are defined for
            # Cartesian -> coordinate. As we are doing the inverse transform
            # direction we swap the order i.e. provide the covariant in the
            # contravariant slot and vice versa.
            transform_tensor_field(
                tensor_type,
                tensor_field,
                covariant_transform,
                contravariant_transform,
                reverse=True,
                return_array=return_array,
            )

        # Transform from Cartesian to output coordinate.
        if output_coordinate != CoordinateSystem.CARTESIAN:
            transform = self.transforms[output_coordinate]

            _tensor_field = np.copy(return_array)

            covariant_transform = transform.covariant_transform

            contravariant_transform = transform.contravariant_transform

            transform_tensor_field(
                tensor_type,
                _tensor_field,
                covariant_transform,
                contravariant_transform,
                reverse=False,
                return_array=return_array,
            )

        return return_array

    def first_covariant_derivative(
        self,
        coordinate_system: CoordinateSystem,
        tensor_field: FloatArray,
        tensor_field_dx: FloatArray,
        tensor_type: TensorType,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Calculate first covariant derivative of a tensor field.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system the tensor field components are provided in.
        tensor_field : np.array[float]
            Tensor field components.
        tensor_field_dx : np.array[float]
            First derivative (jacobian) of tensor field components.
        tensor_type : TensorType
            Tensor type.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        first_covariant_derivative : np.array[float]
            First covariant derivative of tensor field.
        """
        return_array = get_return_array(
            return_array, tensor_field_dx.shape, tensor_field.dtype
        )

        if coordinate_system == CoordinateSystem.CARTESIAN:
            np.copyto(return_array, tensor_field_dx)
            return return_array

        # Calculate first covariant derivative in local coordinate.
        _connection_coefficients = self.transforms[
            coordinate_system
        ].connection_coefficients

        return_array[..., :] = first_covariant_derivative(
            tensor_type,
            tensor_field,
            tensor_field_dx,
            _connection_coefficients,
            return_array=return_array,
        )

        return return_array

    def second_covariant_derivative(
        self,
        coordinate_system: CoordinateSystem,
        tensor_field: FloatArray,
        tensor_field_dx: FloatArray,
        tensor_field_dx2: FloatArray,
        tensor_field_first_covariant_derivative: FloatArray,
        tensor_type: TensorType,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Calculate second covariant derivative of a tensor field.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system the tensor field components are provided in.
        tensor_field : np.array[float]
            Tensor field components.
        tensor_field_dx : np.array[float]
            First derivative (jacobian) of tensor field components.
        tensor_field_dx2 : np.array[float]
            Second derivative (hessian) of tensor field components.
        tensor_field_first_covariant_derivative : np.array[float]
            First covariant derivative of tensor field components.
        tensor_type : TensorType
            Tensor type.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        second_covariant_derivative : np.array[float]
            Second covariant derivative of tensor field.
        """
        return_array = get_return_array(
            return_array, tensor_field_dx2.shape, tensor_field.dtype
        )

        if coordinate_system == CoordinateSystem.CARTESIAN:
            np.copyto(return_array, tensor_field_dx2)
            return return_array

        _connection_coefficients = self.transforms[
            coordinate_system
        ].connection_coefficients

        _connection_coefficients_dx = self.transforms[
            coordinate_system
        ].connection_coefficients_dx

        return_array[..., :, :] = second_covariant_derivative(
            tensor_type,
            tensor_field,
            tensor_field_dx,
            tensor_field_dx2,
            tensor_field_first_covariant_derivative,
            _connection_coefficients,
            _connection_coefficients_dx,
            return_array=return_array,
        )

        return return_array
