"""
Objects for coordinating transforms between multiple coordinate systems.
"""

# Standard imports
import itertools
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813
import numpy as np
from scipy.sparse import csgraph, csr_matrix

# Local imports
from crayon.calculus import first_derivative
from crayon.coordinates.coordinate_system import metric_tensor
from crayon.coordinates.coordinates.base import (
    CoordinateSystem,
    CurvilinearCoordinate,
)
from crayon.coordinates.coordinates.cartesian import CARTESIAN
from crayon.coordinates.coordinates.cylindrical import CYLINDRICAL
from crayon.coordinates.coordinates.flux_coordinate import (
    AxisymmetricFluxCoordinate,
    AxisymmetricFluxCoordinateRebase,
)
from crayon.coordinates.coordinates.toroidal import Toroidal
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.io import IONetcdf
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


class CoordinatePair:
    """
    A data structure for a pair of coordinate systems.

    Attributes
    ----------
    coordinate_1, coordinate_2 : CoordinateSystem
        Coordinate systems in pair.
    """

    __slots__ = ("coordinate_1", "coordinate_2")

    def __init__(
        self, coordinate_1: CoordinateSystem, coordinate_2: CoordinateSystem
    ):
        """
        Inits CoordinatePair.

        Parameters
        ----------
        coordinate_1, coordinate_2 : CoordinateSystem
            Coordinate systems in pair.
        """
        self.coordinate_1 = coordinate_1
        self.coordinate_2 = coordinate_2

    def __str__(self) -> str:
        """
        Convert to string.

        Returns
        -------
        s : str
            String containing names of coordinates in pair.
        """
        return f"{self.coordinate_1.name}, {self.coordinate_2.name}"

    def __hash__(self) -> int:
        """
        Create hash value for object.

        Returns
        -------
        hash : int
            Hash value for coordinate pair.

        Notes
        -----
        Use XOR so the hash is symmetric in coordinate i.e.
        (cartesian, cylindrical) and (cylindrical, cartesian) are the same.
        """
        return hash(self.coordinate_1) ^ hash(self.coordinate_2)

    def __eq__(self, other) -> bool:
        """
        Check equality with other object.

        Parameters
        ----------
        other : any
            Other object to compare equality with.

        Returns
        -------
        equal : bool
            Flag if objects are equal.
        """
        if isinstance(other, CoordinatePair):
            return hash(self) == hash(other)
        return False

    @classmethod
    def parse(cls, s: str) -> "CoordinatePair":
        """
        Parse CoordinatePair from string representation.

        Parameters
        ----------
        s : str
            String to parse from.

        Returns
        -------
        coordinate_pair : CoordinatePair
            New coordinate pair object.
        """
        _c1, _c2 = s.split(", ")
        coordinate_1 = CoordinateSystem.parse(_c1)
        coordinate_2 = CoordinateSystem.parse(_c2)

        return cls(coordinate_1, coordinate_2)

    @property
    def unique_name(self) -> str:
        """
        Return a string which contains the names of both coordinates but is
        independent of the order of coordinate_1 and coordinate_2.
        """
        if self.coordinate_1.value < self.coordinate_2.value:
            return f"{self.coordinate_1.name}, {self.coordinate_2.name}"
        return f"{self.coordinate_2.name}, {self.coordinate_1.name}"


DIJKSTRA_NOT_CONNECTED = -9999


class CoordinateCoordinator(IONetcdf):
    """
    A coordinator transforms between coordinate systems.

    Attributes
    ----------
    coordinates : dict[CoordinateSystem, CurvilinearCoordinate]
        A mapping of coordinate objects defined.
    conversion_paths : dict[CoordinateSystem, list[CoordinateSystem]]
        A mapping showing intermediate coordinates required to convert
        from a coordinate system to Cartesian.
    """

    __slots__ = ("conversion_paths", "coordinates")

    section_name = "coordinates"

    def __init__(self):
        """
        Inits CoordinateCoordinator. A global Cartesian and cylindrical
        coordinate system is added by default.
        """
        self.coordinates: dict[CoordinateSystem, CurvilinearCoordinate] = {}
        self.conversion_paths: dict[
            CoordinateSystem, list[CoordinateSystem]
        ] = {}

        # Always have Cartesian and cylindrical coordinate systems.
        self.register_coordinate(CARTESIAN)
        self.register_coordinate(CYLINDRICAL)

    def register_coordinate(self, coordinate: CurvilinearCoordinate):
        """
        Register a coordinate system model.

        Parameters
        ----------
        coordinate : CurvilinearCoordinateSystem
            Coordinate system object.

        Raises
        ------
        ValueError
            Coordinate already exists in coordinator.
        """
        c = coordinate.coordinate_system
        if c in self.coordinates:
            raise ValueError(f"Coordinate already exists: {c.name}")
        self.coordinates[c] = coordinate

    def calculate_conversion_paths(self):
        """
        Calculate paths to convert from each coordinate system to Cartesian.

        Raises
        ------
        ValueError
            Unable to find conversion paths between all coordinates.
        """
        # Get list of all coordinate systems.
        all_coordinates = list(self.coordinates.keys())

        # Assign an index to each coordinate system.
        index_map = {
            coordinate: i for i, coordinate in enumerate(all_coordinates)
        }

        # Set up a directed graph showing which coordinates we can map between.
        n = len(index_map)
        directed_graph = np.zeros((n, n), dtype=int)

        # Map parent coordinate.
        for coordinate_system, coordinate in self.coordinates.items():
            parent_coordinate_system = coordinate.parent_coordinate_system
            i = index_map[coordinate_system]
            j = index_map[parent_coordinate_system]
            directed_graph[i, j] = 1
            directed_graph[j, i] = 1

        # Use diijkstra's algorithm to find the minimum spanning tree.
        # predecessors[i, j] gives the index of the previous node in the
        # shortest path from point i to point j.
        directed_graph = csr_matrix(directed_graph)
        _, predecessors = csgraph.dijkstra(
            directed_graph, directed=True, return_predecessors=True
        )

        # conversion_paths is a mapping of which intermediate coordinates to
        # pass through while transforming.
        self.conversion_paths.clear()

        start_index = index_map[CoordinateSystem.CARTESIAN]
        for coordinate_system in all_coordinates:
            # Don't calculate conversion path from coordinate to itself.
            if coordinate_system == CoordinateSystem.CARTESIAN:
                continue

            # -9999 = no path from node i to j.
            k = index_map[coordinate_system]
            path = [coordinate_system]
            while predecessors[start_index, k] != DIJKSTRA_NOT_CONNECTED:
                k = predecessors[start_index, k]
                path.append(all_coordinates[k])

            # If we didn't make it to the start coordinate the nodes are
            # not connected.
            if k != start_index:
                raise ValueError(
                    f"No conversion path from {coordinate_system.name} to "
                    f"'{CoordinateSystem.CARTESIAN.name}'"
                )

            # Add conversion path in terms of coordinate pairs.
            self.conversion_paths[coordinate_system] = path

    def get_conversion_path(
        self,
        target_coordinate: CoordinateSystem,
        /,
        *,
        to_target: bool,
    ) -> list[CoordinatePair]:
        """
        Get conversion path between Cartesian and provided coordinate system.

        Parameters
        ----------
        target_coordinate : CoordinateSystem
            Target coordinate system for conversion path.
        to_target : bool
            If True, the coversion path will start at Cartesian and end at
            target_coordinate. Otherwise, the path will be reversed.

        Returns
        -------
        conversion_path : list[CoordinateSystem]
            Intermediate coordinate systems required to transform from
            Cartesian <-> target coordinate.

        Raises
        ------
        ValueError
            No conversion path found for target coordinate.
        """
        if target_coordinate == CoordinateSystem.CARTESIAN:
            return []
        if target_coordinate in self.conversion_paths:
            # Conversion path goes from target coordinate to Cartesian.
            conversion_path = self.conversion_paths[target_coordinate]
        else:
            raise ValueError(
                f"No conversion path for {target_coordinate}. "
                "Did calculate_conversion_paths get called?"
            )

        # If we want Cartesian to target coordinate then reverse.
        if to_target:
            return reversed(conversion_path)
        return conversion_path

    def convert_coordinate(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        forward: bool,
        return_array: FloatArray = None,
    ) -> FloatArray:
        """
        Convert a position from an input coordinate to an output coordinate.

        Parameters
        ----------
        position : np.array[float]
            Position components
        coordinate_system : CoordinateSystem
            Coordinate system model to use for conversion.
        forward : bool
            If True, position is assumed to be in the root coordinate system
            of the coordinate model and the equivalent position in the new
            coordinate system is returned. Otherwise, position is assumed to be
            in the coordinate system and the equivalent postion in the root
            coordinate system is returned.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        new_position
            Position components in target coordinate system.
        """
        return_array = get_return_array(
            return_array, position.shape, position.dtype
        )

        coordinate = self.coordinates[coordinate_system]

        if forward:
            coordinate.forward_transform(position, return_array=return_array)
        else:
            coordinate.backward_transform(position, return_array=return_array)

        return return_array

    def metric_tensor(
        self,
        coordinate_system: CoordinateSystem,
        position: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ) -> FloatArray:
        """
        Calculate metric tensor.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system to calculate metric tensor for.
        position : np.array[float]
            Position in the provided coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        metric_tensor : np.array[float]
            Metric tensor for given coordinate system at given position.

        Raises
        ------
        ValueError
            Coordinate implements neither forward nor backward transform
            derivatives.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), FloatType
        )

        # If Cartesian return Euclidean metric.
        if coordinate_system == CoordinateSystem.CARTESIAN:
            return_array[:, :] = np.identity(Dimensions.x.size)
            return return_array

        # Get conversion path from coordinate system to Cartesian.
        conversion_path = self.get_conversion_path(
            coordinate_system, to_target=False
        )

        # Evaluate positions along the conversion path.
        _positions = {coordinate_system: position}

        for _coordinate_system in itertools.islice(
            conversion_path, len(conversion_path) - 1
        ):
            _parent_coordinate = self.coordinates[
                _coordinate_system
            ].parent_coordinate_system
            _positions[_parent_coordinate] = self.convert_coordinate(
                _positions[_coordinate_system],
                _coordinate_system,
                forward=False,
            )

        # Calculate backward transform from Cartesian to coordinate system.
        coordinate = self.coordinates[coordinate_system]

        _n = Dimensions.x.size
        _f_shape = (_n,)
        _g_size = _n
        _x_size = _n

        backward_transform_dx = np.identity(_n)
        _backward_transform_dx = np.empty_like(backward_transform_dx)

        for input_coordinate_system in itertools.islice(
            conversion_path, len(conversion_path) - 1
        ):
            _coordinate = self.coordinates[input_coordinate_system]

            if _coordinate.forward_transform_derivatives:
                _preferred = _coordinate.forward_transform_preferred_coordinate
            elif coordinate.backward_transform_derivatives:
                _preferred = (
                    _coordinate.backward_transform_preferred_coordinate
                )
            else:
                # Shouldn't reach here!
                raise ValueError(coordinate)

            _coordinate.backward_transform_dx(
                _positions[_preferred],
                _preferred,
                return_array=_backward_transform_dx,
            )

            first_derivative(
                _backward_transform_dx,
                backward_transform_dx,
                _f_shape,
                _g_size,
                _x_size,
                return_array=backward_transform_dx,
            )

        # Calculate covariant transform.
        covariant_transform = _backward_transform_dx
        CurvilinearCoordinate.covariant_transform(
            backward_transform_dx, return_array=covariant_transform
        )

        # Calculate metric tensor.
        metric_tensor(covariant_transform, return_array=return_array)

        return return_array

    def write_netcdf(self, group: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Group
            netCDF4 group to write data to.
        """
        for model in self.coordinates.values():
            model.write_netcdf(group)

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "CoordinateCoordinator":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Group
            netCDF4 group to read data from.

        Returns
        -------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator containing coordinates defined in file.
        """
        obj = cls()

        toroidal_coordinate_group = ""

        for name, g in group.groups.items():
            coordinate_system = CoordinateSystem.parse(name)

            if coordinate_system in {
                CoordinateSystem.CARTESIAN,
                CoordinateSystem.CYLINDRICAL,
            }:
                # Always defined.
                continue
            if coordinate_system == CoordinateSystem.TOROIDAL:
                model = Toroidal.read_netcdf(g)
            elif coordinate_system == CoordinateSystem.RHO_POLOIDAL:
                model = AxisymmetricFluxCoordinate.read_netcdf(g)
            elif coordinate_system == CoordinateSystem.RHO_TOROIDAL:
                toroidal_coordinate_group = name
                continue
            else:
                raise NotImplementedError(coordinate_system)

            obj.register_coordinate(model)

        # Need to read this after we have read rho poloidal coordinate.
        if toroidal_coordinate_group:
            model = AxisymmetricFluxCoordinateRebase.read_netcdf(
                group[toroidal_coordinate_group],
                obj.coordinates[CoordinateSystem.RHO_POLOIDAL],
            )
            obj.register_coordinate(model)

        return obj
