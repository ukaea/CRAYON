"""
Coordinate dimensions.
"""

# Standard imports
import logging
import typing

# Third party imports
import netCDF4 as nc4  # noqa: N813

# Local import
from crayon.shared.data_structures import Dimension
from crayon.shared.io import IONetcdf

logger = logging.getLogger(__name__)

_D: int = 3
_MAX_HARMONIC: int = 40
_MAX_RAY_NODES: int = 1000
_MAX_RAY_CHILDREN: int = 10


class Dimensions(IONetcdf):
    """
    Object collecting all available Dimensions.

    Attributes
    ----------
    one : Dimension
        Size 1 dimension.
    two : Dimension
        Size 2 dimension.
    three : Dimension
        Size 3 dimension.
    x : Dimension
        Spatial dimension x.
    slice_x : slice
        Slice for extracting x component from extended phase space z.
    k : Dimension
        Wavenumber dimension k.
    slice_k : slice
        Slice for extracting k component from extended phase space z.
    xk : Dimension
        Phase space dimension (x, k).
    slice_xk : slice
        Slice for extracting xk component from extended phase space z.
    z : Dimension
        Extended phase space dimension (x, k, f, t).
    slice_f : slice
        Slice for extracting f component from extended phase space z.
    slice_t : slice
        Slice for extracting t component from extended phase space z.
    q : Dimension
        Hamiltonian arguments dimension q.
    IDX_X : int
        Index of X in Hamiltonian arguments q.
    IDX_Y : int
        Index of Y in Hamiltonian arguments q.
    IDX_Z : int
        Index of Z in Hamiltonian arguments q.
    IDX_THETA : int
        Index of theta in Hamiltonian arguments q.
    IDX_N_PERP : int
        Index of n_perp in Hamiltonian arguments q.
    IDX_N_PARALLEL : int
        Index of n_parallel in Hamiltonian arguments q.
    ray_node : str
        Name of ray node dimension. Size is set dynamically.
    max_harmonic : Dimension
        Maximum cyclotron harmonic dimension.
    max_ray_nodes : Dimension
        Max ray nodes dimension.
    max_ray_children : Dimension
        Max ray children dimension.
    """

    __slots__ = ()

    # Dummy dimension for 1 sized data.
    one = Dimension("one", 1)

    # netCDF dimensions.
    # Two related values e.g. lower and upper limit.
    two = Dimension("two", 2)
    three = Dimension("three", 3)

    # x = position.
    x = Dimension("x", _D)
    slice_x = slice(0, _D)

    # k = wavevector.
    k = Dimension("k", _D)
    slice_k = slice(_D, 2 * _D)

    # xk = 6D phase space (x, k).
    xk = Dimension("xk", 2 * _D)
    slice_xk = slice(0, 6)

    # z = 8D phase space (x, k, f, t).
    z = Dimension("z", 2 * _D + 2)
    slice_f = 6
    slice_t = 7

    # q = normalised plasma parameters X, Y, Z, theta, N_perp, N_parallel.
    q = Dimension("q", 6)
    IDX_X = 0
    IDX_Y = 1
    IDX_Z = 2
    IDX_THETA = 3
    IDX_N_PERP = 4
    IDX_N_PARALLEL = 5

    # Ray node. Set dynamically for each ray in output.
    ray_node = "ray_node"

    # Mode conversion.
    mode_conversion = "mode_conversion"

    # Max cyclotron harmonic number for computing kinetic dispersion relations.
    max_harmonic = Dimension("max_harmonic", _MAX_HARMONIC)

    # Max number of ray elements.
    max_ray_nodes = Dimension("max_ray_node", _MAX_RAY_NODES)

    # Maximum number of children a ray can generate.
    max_ray_children = Dimension("max_ray_children", _MAX_RAY_CHILDREN)

    _DIMS: typing.ClassVar[dict[str, int]] = {
        x.name: x,
        k.name: k,
        xk.name: xk,
        z.name: z,
        q.name: q,
        max_harmonic.name: max_harmonic,
        max_ray_nodes.name: max_ray_nodes,
        one.name: one,
        two.name: two,
        three.name: three,
    }

    @classmethod
    def get_dim(cls, name: str) -> Dimension:
        """
        Get dimension from name.

        Attributes
        ----------
        name : str
            Name of dimension.

        Returns
        -------
        dim : Dimension
            Dimension object.
        """
        return cls._DIMS[name]

    @classmethod
    def get_shape(cls, *dimensions: Dimension) -> tuple[int]:
        """
        Get shape of array with given dimensions.

        Attributes
        ----------
        *dimensions : list[Dimension]
            Dimensions.

        Returns
        -------
        shape : tuple[int]
            Shape of array.
        """
        return tuple(d.size for d in dimensions)

    @classmethod
    def write_netcdf(cls, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        # Write data dimensions.
        for dim in (
            cls.x,
            cls.k,
            cls.xk,
            cls.z,
            cls.q,
            cls.one,
            cls.two,
            cls.three,
        ):
            dset.createDimension(dim.name, dim.size)

    @classmethod
    def read_netcdf(cls, *_args, **_kwargs):
        """
        Load from netCDF4 dataset.
        """
        raise NotImplementedError
