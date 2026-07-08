"""
Classes and methods for options for ray tracing.
"""

# Standard imports
import logging
import typing

# Third party imports
import netCDF4 as nc4  # noqa: N813
import toml

# Local imports
from crayon.ray_tracing.integrator import OptionsIntegrator
from crayon.shared.dimensions import Dimensions
from crayon.shared.io import (
    IONetcdf,
    IOToml,
    TomlValidator,
    write_netcdf_variable,
)

logger = logging.getLogger(__name__)

MAX_RAY_NODES = 500
MAX_RAY_CHILDREN = 5
MAX_GENERATIONS = 2
MAX_OPTICAL_DEPTH = 23.0  # == 10^-10 * original power.
MAX_REFLECTIONS = 1
MIN_POWER_FRACTION_NEW_RAY = 0.001

_schema = {
    "max_ray_nodes": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "min": 1,
        "max": Dimensions.max_ray_nodes.size,
        "default": MAX_RAY_NODES,
    },
    "max_ray_children": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "min": 0,
        "max": Dimensions.max_ray_children.size,
        "default": MAX_RAY_CHILDREN,
    },
    "max_generations": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "min": 1,
        "default": MAX_GENERATIONS,
    },
    "max_optical_depth": {
        "type": "float",
        "required": False,
        "coerce": float,
        "min": 0.0,
        "default": MAX_OPTICAL_DEPTH,
    },
    "max_reflections": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "min": 0,
        "default": MAX_REFLECTIONS,
    },
    "min_power_fraction_new_ray": {
        "type": "float",
        "required": False,
        "coerce": float,
        "min": 0.0,
        "default": MIN_POWER_FRACTION_NEW_RAY,
    },
}


class OptionsRayTracing(IONetcdf, IOToml):
    """
    Options for ray tracing.

    Attributes
    ----------
    max_generations : int
        Maximum number of generations of children each parent ray can spawn.
    max_optical_depth : float
        Maximum allowed optical depth along a ray.
    max_ray_children : int
        Maximum number of children each ray can spawn.
    max_ray_nodes : int
        Maximum number of nodes on the ray trajectory.
    max_reflections : int
        Maximum number of reflections for each ray.
    min_power_fraction_new_ray : int
        Minimum fraction of power which must be transferred to spawn a new ray.
    """

    __slots__ = (
        "max_generations",
        "max_optical_depth",
        "max_ray_children",
        "max_ray_nodes",
        "max_reflections",
        "min_power_fraction_new_ray",
    )

    section_name = "ray_tracing"

    def __init__(
        self,
        max_ray_nodes: int = MAX_RAY_NODES,
        max_ray_children: int = MAX_RAY_CHILDREN,
        max_generations: int = MAX_GENERATIONS,
        max_optical_depth: float = MAX_OPTICAL_DEPTH,
        max_reflections: int = MAX_REFLECTIONS,
        min_power_fraction_new_ray: float = MIN_POWER_FRACTION_NEW_RAY,
    ):
        """
        Inits OptionsRayTracing.

        Parameters
        ----------
        max_ray_nodes : int
            Maximum number of nodes on the ray trajectory.
        max_ray_children : int
            Maximum number of children each ray can spawn.
        max_generations : int
            Maximum number of generations of children each parent ray can
            spawn.
        max_optical_depth : float
            Maximum allowed optical depth along a ray.
        max_reflections : int
            Maximum number of reflections for each ray.
        min_power_fraction_new_ray : int
            Minimum fraction of power which must be transferred to spawn a new
            ray.
        """
        self.max_ray_nodes = int(max_ray_nodes)
        self.max_ray_children = int(max_ray_children)
        self.max_generations = int(max_generations)
        self.max_optical_depth = float(max_optical_depth)
        self.max_reflections = int(max_reflections)
        self.min_power_fraction_new_ray = float(min_power_fraction_new_ray)

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Group
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.section_name)

        write_netcdf_variable(
            group,
            "max_ray_nodes",
            (),
            self.max_ray_nodes,
            "Maximum number of nodes in a ray",
            "",
        )

        write_netcdf_variable(
            group,
            "max_ray_children",
            (),
            self.max_ray_children,
            "Maximum number of child rays each ray can spawn",
            "",
        )

        write_netcdf_variable(
            group,
            "max_generations",
            (),
            self.max_generations,
            "Maximum number of generations of children each ray can spawn",
            "",
        )

        write_netcdf_variable(
            group,
            "max_optical_depth",
            (),
            self.max_optical_depth,
            "Maximum allowed optical depth along a ray",
            "Nepers",
        )

        write_netcdf_variable(
            group,
            "max_reflections",
            (),
            self.max_reflections,
            "Maximum allowed reflections for a ray",
            "",
        )

        write_netcdf_variable(
            group,
            "min_power_fraction_new_ray",
            (),
            self.min_power_fraction_new_ray,
            (
                "Minimum fraction of power which must be transferred to "
                "spawn a new ray."
            ),
            "",
        )

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "OptionsRayTracing":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Group
            netCDF4 dataset or group to read data from.

        Returns
        -------
        options_ray_tracing : OptionsRayTracing
            Options for ray tracing.
        """
        group = dset[cls.section_name]

        max_ray_nodes = group["max_ray_nodes"][...].item()
        max_ray_children = group["max_ray_children"][...].item()
        max_generations = group["max_generations"][...].item()
        max_optical_depth = group["max_optical_depth"][...].item()
        max_reflections = group["max_reflections"][...].item()
        min_power_fraction_new_ray = group["min_power_fraction_new_ray"][
            ...
        ].item()

        return cls(
            max_ray_nodes=max_ray_nodes,
            max_ray_children=max_ray_children,
            max_generations=max_generations,
            max_optical_depth=max_optical_depth,
            max_reflections=max_reflections,
            min_power_fraction_new_ray=min_power_fraction_new_ray,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "max_ray_nodes": self.max_ray_nodes,
            "max_ray_children": self.max_ray_children,
            "max_generations": self.max_generations,
            "max_optical_depth": self.max_optical_depth,
            "max_reflections": self.max_reflections,
            "min_power_fraction_new_ray": self.min_power_fraction_new_ray,
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "OptionsRayTracing":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        options_ray_tracing : OptionsRayTracing
            Options for ray tracing.
        """
        validator = TomlValidator()
        validator.validate(d, _schema)

        _d = validator.document
        _max_ray_nodes = _d.pop("max_ray_nodes")
        _max_ray_children = _d.pop("max_ray_children")
        _max_generations = _d.pop("max_generations")
        _max_optical_depth = _d.pop("max_optical_depth")
        _max_reflections = _d.pop("max_reflections")
        _min_power_fraction_new_ray = _d.pop("min_power_fraction_new_ray")

        return cls(
            max_ray_nodes=_max_ray_nodes,
            max_ray_children=_max_ray_children,
            max_generations=_max_generations,
            max_optical_depth=_max_optical_depth,
            max_reflections=_max_reflections,
            min_power_fraction_new_ray=_min_power_fraction_new_ray,
        )


RAYS = "rays"
INTEGRATOR = "integrator"


def read_options_ray_tracing_toml(
    fh: typing.TextIO,
) -> tuple[OptionsRayTracing, OptionsIntegrator]:
    """
    Load all ray tracing options objects for from TOMl file.

    Parameters
    ----------
    fh : TextIO
        Handle to TOML file to read from.

    Returns
    -------
    options_ray_tracing : OptionsRayTracing
        Options for ray tracing.
    options_integrator : OptionsIntegrator
        Options for ray tracing integrator.
    """
    d = toml.load(fh)

    options_ray_tracing = OptionsRayTracing.from_dict_toml(d[RAYS])
    options_integrator = OptionsIntegrator.from_dict_toml(d[INTEGRATOR])

    return options_ray_tracing, options_integrator


def write_options_ray_tracing_toml(
    fh: typing.TextIO,
    options_ray_tracing: OptionsRayTracing,
    options_integrator: OptionsIntegrator,
):
    """
    Write all ray tracing options objects to TOMl file.

    Parameters
    ----------
    fh : TextIO
        Handle to TOML file to write to.
    options_ray_tracing : OptionsRayTracing
        Options for ray tracing.
    options_integrator : OptionsIntegrator
        Options for ray tracing integrator.
    """
    d = {
        RAYS: options_ray_tracing.to_dict_toml(),
        INTEGRATOR: options_integrator.to_dict_toml(),
    }

    toml.dump(d, fh, encoder=IOToml.numpy_encoder)
