"""
Classes for defining construction of objects describing plasma system data.
"""

# Standard imports
import logging
import pathlib
import typing

# Third party imports
import cerberus
import numpy as np

# Local imports
from crayon.shared.constants import CoordinateSystem
from crayon.shared.io import IOToml, TomlValidator
from crayon.shared.types import FloatArray
from crayon.system_data.limiter import LimiterEffect
from crayon.value_model.base import ModelType

logger = logging.getLogger(__name__)


class SystemDataTomlValidator(TomlValidator):
    """
    Extended cerberus.Validator containing enums used in TOML files for
    defining system data.
    """

    types_mapping = TomlValidator.types_mapping.copy()

    types_mapping["CoordinateSystem"] = cerberus.TypeDefinition(
        "CoordinateSystem", (CoordinateSystem,), ()
    )
    types_mapping["LimiterEffect"] = cerberus.TypeDefinition(
        "LimiterEffect", (LimiterEffect,), ()
    )


IMAS = "imas"
NETCDF = "netcdf"
VMEC = "vmec"


class DataSourceImas(IOToml):
    """
    IMAS database source.

    Attributes
    ----------
    occurrence_core_profiles : int
        Occurrence of core_profiles to read.
    occurrence_equilibrium : int
        Occurrence of equilibrium to read.
    occurrence_wall : int
        Occurrence of wall to read.
    uri : str
        Uniform resource identifier for IMAS database. See [1].

    Methods
    -------
    from_dict_toml
        Create object from dictionary of data read from toml file.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.

    References
    ----------
    [1] https://imas-data-dictionary.readthedocs.io/en/4.1.1/
        IMAS-URI-scheme.html
    """

    __slots__ = (
        "occurrence_core_profiles",
        "occurrence_equilibrium",
        "occurrence_wall",
        "uri",
    )

    prefix = IMAS

    toml_schema: typing.ClassVar[dict] = {
        "uri": {"type": "string", "required": True},
        "occurrence_core_profiles": {
            "type": "integer",
            "required": False,
            "coerce": int,
            "min": 0,
            "default": 0,
        },
        "occurrence_equilibrium": {
            "type": "integer",
            "required": False,
            "coerce": int,
            "min": 0,
            "default": 0,
        },
        "occurrence_wall": {
            "type": "integer",
            "required": False,
            "coerce": int,
            "min": 0,
            "default": 0,
        },
    }

    def __init__(
        self,
        uri: str,
        occurrence_core_profiles: int,
        occurrence_equilibrium: int,
        occurrence_wall: int,
    ):
        """
        Inits DataSourceImas.

        Parameters
        ----------
        uri : str
            Uniform resource identifier for IMAS database.
        occurrence_core_profiles : int
            Occurrence of core_profiles to read.
        occurrence_equilibrium : int
            Occurrence of equilibrium to read.
        occurrence_wall : int
            Occurrence of wall to read.
        """
        super().__init__()

        self.uri = uri
        self.occurrence_core_profiles = int(occurrence_core_profiles)
        self.occurrence_equilibrium = int(occurrence_equilibrium)
        self.occurrence_wall = int(occurrence_wall)

    @classmethod
    def from_dict_toml(cls, d: dict) -> "DataSourceImas":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        data_source_imas : DataSourceImas
            IMAS database source.
        """
        validator = SystemDataTomlValidator()
        validator.validate(d, cls.toml_schema)

        return cls(
            validator.document["uri"],
            validator.document["occurrence_core_profiles"],
            validator.document["occurrence_equilibrium"],
            validator.document["occurrence_wall"],
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
            "uri": self.uri,
            "occurrence_core_profiles": self.occurrence_core_profiles,
            "occurrence_equilibrium": self.occurrence_equilibrium,
            "occurrence_wall": self.occurrence_wall,
        }


class DataSourceNetcdf(IOToml):
    """
    NetCDF4 file source.

    Attributes
    ----------
    filepath : str
        Path to netCDF4 file.

    Methods
    -------
    from_dict_toml
        Create object from dictionary of data read from toml file.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    """

    __slots__ = ("filepath",)

    prefix = NETCDF

    toml_schema: typing.ClassVar[dict] = {
        "filepath": {"type": "string", "required": True}
    }

    def __init__(self, filepath: pathlib.Path):
        """
        Inits DataSourceNetcdf.

        Parameters
        ----------
        filepath : str
            Path to netCDF4 file.
        """
        super().__init__()

        self.filepath = pathlib.Path(filepath)

    @classmethod
    def from_dict_toml(cls, d: dict) -> "DataSourceNetcdf":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        data_source_netcdf : DataSourceNetcdf
            Netcdf4 file source.
        """
        return cls(d["filepath"])

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "filepath": str(self.filepath.resolve()),
        }


class DataSourceVmec(IOToml):
    """
    VMEC output file source.

    Attributes
    ----------
    filepath : str
        Path to VMEC output file.

    Methods
    -------
    from_dict_toml
        Create object from dictionary of data read from toml file.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    """

    __slots__ = ("filepath",)

    prefix = VMEC

    toml_schema: typing.ClassVar[dict] = {
        "filepath": {"type": "string", "required": True}
    }

    def __init__(self, filepath: pathlib.Path):
        """
        Inits DataSourceVmec.

        Parameters
        ----------
        filepath : str
            Path to VMEC output file.
        """
        super().__init__()

        self.filepath = pathlib.Path(filepath)

    @classmethod
    def from_dict_toml(cls, d: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        data_source_vmec : DataSourceVMEC
            VMEC output file source.
        """
        return cls(d["filepath"])

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "filepath": str(self.filepath.resolve()),
        }


DataSourceType = DataSourceImas | DataSourceNetcdf | DataSourceVmec


def parse_data_sources(document: dict) -> dict[str, DataSourceType]:
    """
    Parse data sources from document loaded from TOML file.

    Parameters
    ----------
    document : dict
        Document defining data sources.

    Returns
    -------
    data_sources : dict[str, DataSourceType]
        Dictionary of data sources.

    Raises
    ------
    ValueError
        Unknown data source type.
    """
    data_sources = {}

    for name, config in document.items():
        _name = str(name).lower()

        if _name.startswith(DataSourceImas.prefix):
            data_sources[_name] = DataSourceImas.from_dict_toml(config)
        elif _name.startswith(DataSourceNetcdf.prefix):
            data_sources[_name] = DataSourceNetcdf.from_dict_toml(config)
        elif _name.startswith(DataSourceVmec.prefix):
            data_sources[_name] = DataSourceVmec.from_dict_toml(config)
        else:
            raise ValueError(
                f"Invalid data source prefix '{name}'. Allowed values: "
                ", ".join(
                    DataSourceImas.prefix,
                    DataSourceNetcdf.prefix,
                    DataSourceVmec.prefix,
                )
            )

    return data_sources


class CoordinateToroidal(IOToml):
    """
    Definition of a toroidal coordinate system.

    Attributes
    ----------
    r0, z0 : float
        Cylindrical (r, z) position of axis of toroidal coordinate system.

    Methods
    -------
    from_dict_toml
        Create object from dictionary of data read from toml file.
    to_dict_toml
        Write object data into dictionary for serialisation to toml.
    """

    __slots__ = ("r0", "z0")

    toml_schema: typing.ClassVar[dict] = {
        "r0": {
            "type": "float",
            "required": True,
        },
        "z0": {
            "type": "float",
            "required": True,
        },
    }

    def __init__(
        self,
        r0: float,
        z0: float,
    ):
        """
        Inits CoordinateToroidal.

        Parameters
        ----------
        r0, z0 : float
            Cylindrical (r, z) position of axis of toroidal coordinate system.
        """
        self.r0 = float(r0)
        self.z0 = float(z0)

    @classmethod
    def from_dict_toml(cls, document: dict) -> "CoordinateToroidal":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        coordinate_toroidal : CoordinateToroidal
            Definition of toroidal coordinate system.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(validator.document["r0"], validator.document["z0"])

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {"r0": self.r0, "z0": self.z0}


CoordinateType = CoordinateToroidal


def parse_coordinates(
    document: dict,
) -> dict[CoordinateSystem, CoordinateType]:
    """
    Parse coordinates from document loaded from TOML file.

    Parameters
    ----------
    document : dict
        Document defining coordinates.

    Returns
    -------
    coordinates : dict[CoordinateSystem, CoordinateType]
        Dictionary of data sources.
    """
    coordinates = {}

    for name, config in document.items():
        _name = name.lower()

        if _name == CoordinateSystem.TOROIDAL.name.lower():
            coordinates[CoordinateSystem.TOROIDAL] = (
                CoordinateToroidal.from_dict_toml(config)
            )

    return coordinates


class Model(IOToml):
    """
    Base class for definition of plasma parameter model.

    Attributes
    ----------
    scale_factor : float
        Scale factor for parameter value.
    """

    __slots__ = ("scale_factor",)

    source_prefix: str = NotImplemented

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
        },
        "scale_factor": {
            "required": False,
            "type": "float",
            "coerce": float,
            "default": 1.0,
        },
    }

    def __init__(self, scale_factor: float):
        """
        Inits Model.

        Parameters
        ----------
        scale_factor : float
            Scale factor for parameter value.
        """
        self.scale_factor = scale_factor

    @classmethod
    def from_dict_toml(cls, document: dict) -> "Model":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model : Model
            Plasma parameter model.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(validator.document["scale_factor"])

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {"source": self.source, "scale_factor": self.scale_factor}


class ModelImas(Model):
    """
    Plasma parameter model defined from IMAS database.

    Attributes
    ----------
    source : str
        Name of IMAS data source.
    """

    __slots__ = ("source",)

    source_prefix = IMAS

    def __init__(self, source: str, scale_factor: float):
        """
        Inits ModelImas.

        Parameters
        ----------
        source : str
            Name of IMAS data source.
        scale_factor : float
            Scale factor for parameter value.

        Raises
        ------
        ValueError
            source does not start with IMAS prefix.
        """
        self.source = str(source).lower()
        super().__init__(scale_factor)

        if not self.source.startswith(self.source_prefix):
            raise ValueError(
                f"source must start with '{self.source_prefix}': {source}"
            )

    @classmethod
    def from_dict_toml(cls, document: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model_imas : ModelImas
            Plasma parameter model from IMAS data.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["source"], validator.document["scale_factor"]
        )


class ModelNetcdf(ModelImas):
    """
    Plasma parameter model defined from netCDF4 file.
    """

    __slots__ = ()

    source_prefix = NETCDF


def coerce_float_ndarray(x) -> FloatArray:
    """
    Coerce value into np.array with float datatype.

    Parameters
    ----------
    x : any
        Value to be coerced.

    Returns
    -------
    x_coerced : np.array[float]
        Value as np.array with float datatype
    """
    return np.asarray(x, dtype=float)


schema_array = {
    "required": True,
    "type": "ndarray",
    "coerce": coerce_float_ndarray,
}

schema_coordinate = {
    "coordinate_system": {
        "required": True,
        "type": "CoordinateSystem",
        "coerce": CoordinateSystem.parse,
    },
}

schema_origin = {
    "origin": schema_array,
}

schema_direction = {
    "direction": schema_array,
}

schema_y0_y1 = {
    "y0": schema_array,
    "y1": schema_array,
}

schema_ramp_width = {
    "ramp_width": {
        "required": True,
        "type": "float",
        "coerce": float,
    },
}

schema_smoothness = {
    "smoothness": {
        "required": True,
        "type": "integer",
        "allowed": (0, 1, 2),
    }
}


class ModelAnalyticConstant(Model):
    """
    Analytic plasma parameter model for a constant value.

    Attributes
    ----------
    constant_value : FloatArray
        Constant value.
    coordinate_system : CoordinateSystem
        Coordinate system used to define model.
    """

    __slots__ = (
        "constant_value",
        "coordinate_system",
    )

    source = ModelType.CONSTANT.name.lower()

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            **Model.toml_schema["source"],
            "allowed": (source,),
        },
        "scale_factor": Model.toml_schema["scale_factor"],
        **schema_coordinate,
        "constant_value": schema_array,
    }

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        constant_value: FloatArray,
        scale_factor: float,
    ):
        """
        Inits ModelAnalyticConstant.

        Parameters
        ----------
        constant_value : FloatArray
            Constant value.
        coordinate_system : CoordinateSystem
            Coordinate system used to define model.
        scale_factor : float
            Scale factor for parameter value.
        """
        super().__init__(scale_factor)

        self.coordinate_system = coordinate_system
        self.constant_value = np.asarray(constant_value, dtype=float)
        self.scale_factor = float(scale_factor)

    @classmethod
    def from_dict_toml(cls, document: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model : ModelAnalyticConstant
            Analytic plasma parameter model for a constant.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["coordinate_system"],
            validator.document["constant_value"],
            validator.document["scale_factor"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()

        d["coordinate_system"] = self.coordinate_system.name
        d["constant_value"] = (
            self.constant_value.item()
            if self.constant_value.size == 1
            else self.constant_value
        )
        d["scale_factor"] = self.scale_factor

        return d


class ModelAnalyticRamp(Model):
    """
    Analytic plasma parameter model for a value ramp.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        Coordinate system used to define model.
    direction : np.array[float]
        Direction in which value ramp occurs.
    origin : np.array[float]
        Origin of ramp where value = y0.
    ramp_width : float
        Width of ramp.
    smoothness : int
        Smoothness class of ramp.
    y0 : np.array[float]
        Value at start of ramp.
    y1 : np.array[float]
        Value at end of ramp.
    """

    __slots__ = (
        "coordinate_system",
        "direction",
        "origin",
        "ramp_width",
        "smoothness",
        "y0",
        "y1",
    )

    source = ModelType.RAMP.name.lower()

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            **Model.toml_schema["source"],
            "allowed": (source,),
        },
        "scale_factor": Model.toml_schema["scale_factor"],
        **schema_coordinate,
        **schema_origin,
        **schema_direction,
        **schema_y0_y1,
        **schema_ramp_width,
        **schema_smoothness,
    }

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        smoothness: int,
        scale_factor: float,
    ):
        """
        Inits ModelAnalyticRamp.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system used to define model.
        origin : np.array[float]
            Origin of ramp where value = y0.
        direction : np.array[float]
            Direction in which value ramp occurs.
        y0 : np.array[float]
            Value at start of ramp.
        y1 : np.array[float]
            Value at end of ramp.
        ramp_width : float
            Width of ramp.
        smoothness : int
            Smoothness class of ramp.
        scale_factor : float
            Scale factor for parameter value.
        """
        super().__init__(scale_factor)

        self.coordinate_system = coordinate_system
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.y0 = np.asarray(y0, dtype=float)
        self.y1 = np.asarray(y1, dtype=float)
        self.ramp_width = float(ramp_width)
        self.smoothness = int(smoothness)

    @classmethod
    def from_dict_toml(cls, document: dict) -> "ModelAnalyticRamp":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model : ModelAnalyticRamp
            Analytic plasma parameter model for a ramp.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["coordinate_system"],
            validator.document["origin"],
            validator.document["direction"],
            validator.document["y0"],
            validator.document["y1"],
            validator.document["ramp_width"],
            validator.document["smoothness"],
            validator.document["scale_factor"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()

        d["coordinate_system"] = self.coordinate_system.name
        d["origin"] = self.origin
        d["direction"] = self.direction
        d["y0"] = self.y0.item() if self.y0.size == 1 else self.y0
        d["y1"] = self.y1.item() if self.y1.size == 1 else self.y1
        d["ramp_width"] = self.ramp_width
        d["smoothness"] = self.smoothness

        return d


class ModelAnalyticQuadraticWell(Model):
    """
    Analytic plasma parameter model for a Cartesian quadratic well.

    Attributes
    ----------
    origin : np.array[float]
        Origin of well where value = y0.
    ramp_width : float
        Width of well.
    y0 : np.array[float]
        Value at bottom of well.
    y1 : np.array[float]
        Value one ramp_width away from origin.
    """

    __slots__ = (
        "origin",
        "ramp_width",
        "y0",
        "y1",
    )

    source = ModelType.QUADRATIC_WELL.name.lower()

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            **Model.toml_schema["source"],
            "allowed": (source,),
        },
        "scale_factor": Model.toml_schema["scale_factor"],
        **schema_origin,
        **schema_y0_y1,
        **schema_ramp_width,
    }

    def __init__(
        self,
        origin: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        scale_factor: float,
    ):
        """
        Inits ModelAnalyticQuadraticWell.

        Parameters
        ----------
        origin : np.array[float]
            Origin of well where value = y0.
        y0 : np.array[float]
            Value at bottom of well.
        y1 : np.array[float]
            Value one ramp_width away from origin
        ramp_width : float
            Width of well.
        scale_factor : float
            Scale factor for parameter value.
        """
        super().__init__(scale_factor)

        self.origin = np.asarray(origin, dtype=float)
        self.y0 = np.asarray(y0, dtype=float)
        self.y1 = np.asarray(y1, dtype=float)
        self.ramp_width = float(ramp_width)

    @classmethod
    def from_dict_toml(cls, document: dict) -> "ModelAnalyticQuadraticWell":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model : ModelAnalyticQuadraticWell.
            Analytic plasma parameter model for a Cartesian quadratic well.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["origin"],
            validator.document["y0"],
            validator.document["y1"],
            validator.document["ramp_width"],
            validator.document["scale_factor"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()

        d["origin"] = self.origin
        d["y0"] = self.y0.item() if self.y0.size == 1 else self.y0
        d["y1"] = self.y1.item() if self.y1.size == 1 else self.y1
        d["ramp_width"] = self.ramp_width

        return d


class ModelAnalyticQuadraticChannel(Model):
    """
    Analytic plasma parameter model for a Cartesian quadratic channel.

    Attributes
    ----------
    direction : np.array[float]
        Direction from origin in which value stays at y0.
    origin : np.array[float]
        Origin of channel where value = y0.
    ramp_width : float
        Width of channel.
    y0 : np.array[float]
        Value at bottom of channel.
    y1 : np.array[float]
        Value one ramp_width away from origin.
    """

    __slots__ = (
        "direction",
        "origin",
        "ramp_width",
        "y0",
        "y1",
    )

    source = ModelType.QUADRATIC_CHANNEL.name.lower()

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            **Model.toml_schema["source"],
            "allowed": (source,),
        },
        "scale_factor": Model.toml_schema["scale_factor"],
        **schema_direction,
        **schema_origin,
        **schema_y0_y1,
        **schema_ramp_width,
    }

    def __init__(
        self,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        scale_factor: float,
    ):
        """
        Inits ModelAnalyticQuadraticChannel.

        Attributes
        ----------
        origin : np.array[float]
            Origin of channel where value = y0.
        direction : np.array[float]
            Direction from origin in which value stays at y0.
        y0 : np.array[float]
            Value at bottom of channel.
        y1 : np.array[float]
            Value one ramp_width away from origin.
        ramp_width : float
            Width of channel.
        scale_factor : float
            Scale factor for parameter value.
        """
        super().__init__(scale_factor)

        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.y0 = np.asarray(y0, dtype=float)
        self.y1 = np.asarray(y1, dtype=float)
        self.ramp_width = float(ramp_width)

    @classmethod
    def from_dict_toml(cls, document: dict) -> "ModelAnalyticQuadraticChannel":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model : ModelAnalyticQuadraticChannel
            Analytic plasma parameter model for a quadratic channel.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["origin"],
            validator.document["direction"],
            validator.document["y0"],
            validator.document["y1"],
            validator.document["ramp_width"],
            validator.document["scale_factor"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()

        d["origin"] = self.origin
        d["direction"] = self.direction
        d["y0"] = self.y0.item() if self.y0.size == 1 else self.y0
        d["y1"] = self.y1.item() if self.y1.size == 1 else self.y1
        d["ramp_width"] = self.ramp_width

        return d


class ModelAnalyticQuadraticBowl(Model):
    """
    Analytic plasma parameter model for a Cartesian quadratic bowl.

    Attributes
    ----------
    direction : np.array[float]
        Direction in which value ramp occurs.
    origin : np.array[float]
        Origin of bowl where value = y0.
    ramp_width : float
        Width of bowl.
    y0 : np.array[float]
        Value at bottom of bowl.
    y1 : np.array[float]
        Value one ramp_width away from origin.
    """

    __slots__ = (
        "direction",
        "origin",
        "ramp_width",
        "y0",
        "y1",
    )

    source = ModelType.QUADRATIC_BOWL.name.lower()

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            **Model.toml_schema["source"],
            "allowed": (source,),
        },
        "scale_factor": Model.toml_schema["scale_factor"],
        **schema_direction,
        **schema_origin,
        **schema_y0_y1,
        **schema_ramp_width,
    }

    def __init__(
        self,
        origin: FloatArray,
        direction: FloatArray,
        y0: FloatArray,
        y1: FloatArray,
        ramp_width: float,
        scale_factor: float,
    ):
        """
        Inits ModelAnalyticQuadraticBowl.

        Attributes
        ----------
        origin : np.array[float]
            Origin of bowl where value = y0.
        direction : np.array[float]
            Direction in which value ramp occurs.
        y0 : np.array[float]
            Value at bottom of bowl.
        y1 : np.array[float]
            Value one ramp_width away from origin.
        ramp_width : float
            Width of bowl.
        scale_factor : float
            Scale factor for parameter value.
        """
        super().__init__(scale_factor)

        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.y0 = np.asarray(y0, dtype=float)
        self.y1 = np.asarray(y1, dtype=float)
        self.ramp_width = float(ramp_width)

    @classmethod
    def from_dict_toml(cls, document: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        model : ModelAnalyticQuadraticChannel
            Analytic plasma parameter model for a quadratic channel.
        """
        validator = SystemDataTomlValidator()
        logger.warning(cls.toml_schema)
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["origin"],
            validator.document["direction"],
            validator.document["y0"],
            validator.document["y1"],
            validator.document["ramp_width"],
            validator.document["scale_factor"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()

        d["origin"] = self.origin
        d["direction"] = self.direction
        d["y0"] = self.y0.item() if self.y0.size == 1 else self.y0
        d["y1"] = self.y1.item() if self.y1.size == 1 else self.y1
        d["ramp_width"] = self.ramp_width

        return d


ModelAnalyticType = (
    ModelAnalyticConstant
    | ModelAnalyticRamp
    | ModelAnalyticQuadraticWell
    | ModelAnalyticQuadraticChannel
    | ModelAnalyticQuadraticBowl
)

KineticModelType = (
    ModelImas
    | ModelNetcdf
    | ModelAnalyticConstant
    | ModelAnalyticRamp
    | ModelAnalyticQuadraticWell
    | ModelAnalyticQuadraticChannel
    | ModelAnalyticQuadraticBowl
)

schema_model = {
    "source": {
        "required": True,
        "type": "string",
        "coerce": (str, str.lower),
    },
}


def parse_model(document: dict) -> KineticModelType:
    """
    Parse model from document read from TOML file.

    Parameters
    ----------
    document : dict
        Document defining model.

    Returns
    -------
    model : KineticModelType
        Loaded model.

    Raises
    ------
    ValueError
        Unknown data source.
    """
    validator = SystemDataTomlValidator()
    validator.validate(document, schema_model, allow_unknown=True)

    source = validator.document["source"]

    if source.startswith(ModelImas.source_prefix):
        return ModelImas.from_dict_toml(validator.document)
    if source.startswith(ModelNetcdf.source_prefix):
        return ModelNetcdf.from_dict_toml(validator.document)
    if source == ModelAnalyticConstant.source:
        return ModelAnalyticConstant.from_dict_toml(validator.document)
    if source == ModelAnalyticRamp.source:
        return ModelAnalyticRamp.from_dict_toml(validator.document)
    if source == ModelAnalyticQuadraticChannel.source:
        return ModelAnalyticQuadraticChannel.from_dict_toml(validator.document)
    if source == ModelAnalyticQuadraticBowl.source:
        return ModelAnalyticQuadraticBowl.from_dict_toml(validator.document)
    if source == ModelAnalyticQuadraticWell.source:
        return ModelAnalyticQuadraticWell.from_dict_toml(validator.document)

    raise ValueError(source)


ELECTRON_DENSITY_PER_M3 = "electron_density_per_m3"
ELECTRON_TEMPERATURE_EV = "electron_temperature_ev"
EFFECTIVE_CHARGE = "effective_charge"

schema_kinetic = {
    ELECTRON_DENSITY_PER_M3: {
        "required": True,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
    ELECTRON_TEMPERATURE_EV: {
        "required": True,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
    EFFECTIVE_CHARGE: {
        "required": True,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
}


def parse_kinetic(
    document: dict,
) -> tuple[KineticModelType, KineticModelType, KineticModelType]:
    """
    Parse all kinetic models from document read from TOML file.

    Parameters
    ----------
    document : dict
        Document defining model.

    Returns
    -------
    electron_density_per_m3 : KineticModelType
        Loaded electron density model.
    electron_temperature_ev : KineticModelType
        Loaded electron density model.
    effective_charge : KineticModelType
        Loaded electron density model.
    """
    validator = SystemDataTomlValidator()
    validator.validate(document, schema_kinetic)

    electron_density_per_m3 = parse_model(
        validator.document[ELECTRON_DENSITY_PER_M3]
    )
    electron_temperature_ev = parse_model(
        validator.document[ELECTRON_TEMPERATURE_EV]
    )
    effective_charge = parse_model(validator.document[EFFECTIVE_CHARGE])

    return (electron_density_per_m3, electron_temperature_ev, effective_charge)


SIMPLE = "simple"
TOKAMAK = "tokamak"
STELLARATOR = "stellarator"


class MagneticModelTokamak(Model):
    """
    Model for a tokamak magnetic field.

    Attributes
    ----------
    source : str
        Name of data source.
    """

    __slots__ = ("source",)

    topology = TOKAMAK

    toml_schema: typing.ClassVar[dict] = {
        **Model.toml_schema,
        "topology": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
            "allowed": (topology,),
        },
    }

    def __init__(self, source: str, scale_factor: float):
        """
        Inits MagneticModelTokamak.

        Parameters
        ----------
        source : str
            Name of data source.
        scale_factor : float
            Scale factor for magnetic field.

        Raises
        ------
        ValueError
            Source has unknown prefix.
        """
        self.source = str(source).lower()
        self.scale_factor = float(scale_factor)

        if not self.source.startswith(DataSourceImas.prefix):
            raise ValueError(
                "Invalid source prefix for topology='tokamak'. "
                "Allowed prefixes: " + ", ".join(DataSourceImas.prefix)
            )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = {"topology": self.topology}
        d.update(super().to_dict_toml())

        return d

    @classmethod
    def from_dict_toml(cls, document: dict) -> "MagneticModelTokamak":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        magnetic_model_tokamak : MagneticModelTokamak
            Model for a tokamak magnetic field.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["source"], validator.document["scale_factor"]
        )


class MagneticModelStellarator(MagneticModelTokamak):
    """
    Model for a stellarator magnetic field.
    """

    __slots__ = ()

    topology = STELLARATOR

    toml_schema: typing.ClassVar[dict] = {
        **Model.toml_schema,
        "topology": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
            "allowed": (topology,),
        },
    }

    def __init__(self, source: str, scale_factor: float):
        """
        Inits MagneticModelStellarator.

        Parameters
        ----------
        source : str
            Name of data source.
        scale_factor : float
            Scale factor for magnetic field.

        Raises
        ------
        ValueError
            Source has unknown prefix.
        """
        self.source = str(source).lower()
        self.scale_factor = float(scale_factor)

        if not self.source.startswith(DataSourceVmec.prefix):
            raise ValueError(
                "Invalid source prefix for topology='tokamak'. "
                "Allowed prefixes: " + f"{DataSourceVmec.prefix}"
            )


MagneticModelType = (
    ModelAnalyticConstant
    | ModelAnalyticRamp
    | ModelAnalyticQuadraticWell
    | ModelAnalyticQuadraticChannel
    | ModelAnalyticQuadraticBowl
    | MagneticModelTokamak
    | MagneticModelStellarator
)

schema_magnetic = {
    "topology": {
        "required": True,
        "type": "string",
        "coerce": (str, str.lower),
        "allowed": (
            SIMPLE,
            MagneticModelTokamak.topology,
            MagneticModelStellarator.topology,
        ),
    },
}


def parse_magnetic(document: dict) -> MagneticModelType:
    """
    Parse magnetic model from document read from TOML file.

    Parameters
    ----------
    document : dict
        Document defining model.

    Returns
    -------
    magnetic_field_model : MagneticModelType
        Loaded electron density model.

    Raises
    ------
    ValueError
        Unknown magnetic field topology.
    """
    validator = SystemDataTomlValidator()
    validator.validate(document, schema_magnetic, allow_unknown=True)

    topology = validator.document["topology"]

    if topology == SIMPLE:
        validator.document.pop("topology")
        return parse_model(validator.document)
    if topology == MagneticModelTokamak.topology:
        return MagneticModelTokamak.from_dict_toml(validator.document)
    if topology == MagneticModelStellarator.topology:
        return MagneticModelStellarator.from_dict_toml(validator.document)

    raise ValueError(topology)


ANALYTIC = "analytic"
BOUNDING_BOX_2D = "bounding_box_2d"
BOUNDING_BOX_3D = "bounding_box_3d"
_2D = "2d"
_3D = "3d"


class Limiter(IOToml):
    """
    Base class for limiter model.

    Attributes
    ----------
    extinction_coefficient_nepers : float
        Extinction coefficient applied on intersection [nepers].
    """

    __slots__ = ("extinction_coefficient_nepers",)

    toml_schema: typing.ClassVar[dict] = {
        "source": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
        },
        "extinction_coefficient_nepers": {
            "required": False,
            "type": "float",
            "default": 0.0,
            "coerce": float,
            "min": 0.0,
        },
    }

    def __init__(self, extinction_coefficient_nepers: float):
        """
        Inits Limiter.

        Parameters
        ----------
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].
        """
        self.extinction_coefficient_nepers = float(
            extinction_coefficient_nepers
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
            "extinction_coefficient_nepers": self.extinction_coefficient_nepers
        }


class LimiterImas(Limiter):
    """
    Limiter loaded from IMAS database.

    Attributes
    ----------
    effect : LimiterEffect
        Effect applied on intersection with limiter.
    source : str
        Name of data source.
    """

    __slots__ = ("effect", "source")

    source_prefix = IMAS

    toml_schema: typing.ClassVar[dict] = {
        **Limiter.toml_schema,
        "shape": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
        },
        "effect": {
            "required": True,
            "type": "LimiterEffect",
            "coerce": (str, LimiterEffect.parse),
            "allowed": (LimiterEffect.STOP, LimiterEffect.REFLECT_SPECULAR),
        },
    }

    def __init__(
        self,
        effect: LimiterEffect,
        source: str,
        extinction_coefficient_nepers: float,
    ):
        """
        Inits LimiterImas.

        Parameters
        ----------
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        source : str
            Name of data source.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        ValueError
            Data source does not have IMAS prefix.
        """
        super().__init__(extinction_coefficient_nepers)
        self.effect = effect
        self.source = str(source).lower()

        if not self.source.startswith(self.source_prefix):
            raise ValueError(
                f"source must start with '{self.source_prefix}': {source}"
            )

    @classmethod
    def from_dict_toml(cls, document: dict) -> "LimiterImas":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        limiter_imas : LimiterImas
            Limiter loaded from IMAS database.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["effect"],
            validator.document["source"],
            validator.document["extinction_coefficient_nepers"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["effect"] = self.effect.name
        d["source"] = self.source

        return d


class LimiterImasBoundingBox2D(LimiterImas):
    """
    Limiter constructed from bounding box of magnetic equilibrium data.
    """

    __slots__ = ()

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["shape"] = BOUNDING_BOX_2D

        return d


class LimiterImas2D(LimiterImas):
    """
    Cylindrical (r, z) limiter loaded from IMAS wall.wall_2d.
    """

    __slots__ = ()

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["shape"] = _2D

        return d


class LimiterImas3D(LimiterImas):
    """
    Cylindrical (r, phi, z) limiter loaded from IMAS wall.wall_3d.
    """

    __slots__ = ()

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["shape"] = _3D

        return d


class LimiterNetcdf(Limiter):
    """
    Limiter loaded from netCDF4 file.

    Attributes
    ----------
    group : str
        Name of group in file to load from.
    source : str
        Name of data source.
    """

    __slots__ = (
        "group",
        "source",
    )

    source_prefix = NETCDF

    toml_schema: typing.ClassVar[dict] = {
        **Limiter.toml_schema,
        "group": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
        },
    }

    def __init__(
        self, source: str, group: str, extinction_coefficient_nepers: float
    ):
        """
        Inits LimiterNetcdf.

        Parameters
        ----------
        source : str
            Name of data source.
        group : str
            Name of group in file to load from.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].
        """
        super().__init__(extinction_coefficient_nepers)
        self.source = str(source)
        self.group = str(group)

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["source"] = self.source
        d["group"] = self.group

        return d

    @classmethod
    def from_dict_toml(cls, document: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        limiter_netcdf : LimiterNetcdf
            Limiter loaded from netCDF4 file.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["source"],
            validator.document["group"],
            validator.document["extinction_coefficient_nepers"],
        )


class LimiterAnalyticBoundingBox2D(Limiter):
    """
    Limiter defined as a 2D bounding box.

    Attributes
    ----------
    coordinate : literal['xy', 'xz', 'yz', 'rz']
        Coordinate pair being used.
    effect : LimiterEffect
        Effect applied on intersection with limiter.
    x_limits : tuple[float, float]
        Minimum and maximum x coordinate value.
    y_limits : tuple[float, float]
        Minimum and maximum y coordinate value.
    """

    __slots__ = ("coordinate", "effect", "x_limits", "y_limits")

    XY = "xy"
    XZ = "xz"
    YZ = "yz"
    RZ = "rz"

    toml_schema: typing.ClassVar[dict] = {
        "source": {**Limiter.toml_schema["source"], "allowed": (ANALYTIC,)},
        "extinction_coefficient_nepers": Limiter.toml_schema[
            "extinction_coefficient_nepers"
        ],
        "shape": LimiterImas.toml_schema["shape"],
        "effect": LimiterImas.toml_schema["effect"],
        "coordinate": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
            "allowed": (XY, XZ, YZ, RZ),
        },
        "x_limits": schema_array,
        "y_limits": schema_array,
    }

    def __init__(
        self,
        effect: LimiterEffect,
        coordinate: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        extinction_coefficient_nepers: float,
    ):
        """
        Inits LimiterAnalyticBoundingBox2D.

        Parameters
        ----------
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        coordinate : literal['xy', 'xz', 'yz', 'rz']
            Coordinate pair being used.
        x_limits : tuple[float, float]
            Minimum and maximum x coordinate value.
        y_limits : tuple[float, float]
            Minimum and maximum y coordinate value.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        ValueError
            Invalid coordinate.
        """
        super().__init__(extinction_coefficient_nepers)
        self.effect = effect
        self.coordinate = str(coordinate).lower()
        self.x_limits = np.asarray(x_limits, dtype=float).reshape(2)
        self.y_limits = np.asarray(y_limits, dtype=float).reshape(2)

        if self.coordinate not in {self.XY, self.XZ, self.YZ, self.RZ}:
            raise ValueError(
                f"Invalid coordinate '{coordinate}'. "
                "Valid coordinates: "
                f"{self.XY}, {self.XZ}, {self.YZ}, {self.RZ}"
            )

    @classmethod
    def from_dict_toml(cls, document: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        limiter : LimiterAnalyticBoundingBox2D
            Limiter defined as a 2D bounding box.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["effect"],
            validator.document["coordinate"],
            validator.document["x_limits"],
            validator.document["y_limits"],
            validator.document["extinction_coefficient_nepers"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["effect"] = self.effect.name
        d["source"] = ANALYTIC
        d["shape"] = BOUNDING_BOX_2D
        d["coordinate"] = self.coordinate
        d["x_limits"] = self.x_limits
        d["y_limits"] = self.y_limits

        return d


class LimiterAnalyticBoundingBox3D(Limiter):
    """
    Limiter defined as a 3D bounding box.

    Attributes
    ----------
    coordinate : literal['xyz']
        Coordinate being used.
    effect : LimiterEffect
        Effect applied on intersection with limiter.
    x_limits : tuple[float, float]
        Minimum and maximum x coordinate value.
    y_limits : tuple[float, float]
        Minimum and maximum y coordinate value.
    z_limits : tuple[float, float]
        Minimum and maximum z coordinate value.
    """

    __slots__ = ("coordinate", "effect", "x_limits", "y_limits", "z_limits")

    XYZ = "xyz"

    toml_schema: typing.ClassVar[dict] = {
        "source": {**Limiter.toml_schema["source"], "allowed": (ANALYTIC,)},
        "extinction_coefficient_nepers": Limiter.toml_schema[
            "extinction_coefficient_nepers"
        ],
        "shape": LimiterImas.toml_schema["shape"],
        "effect": LimiterImas.toml_schema["effect"],
        "coordinate": {
            "required": True,
            "type": "string",
            "coerce": (str, str.lower),
            "allowed": (XYZ,),
        },
        "x_limits": schema_array,
        "y_limits": schema_array,
        "z_limits": schema_array,
    }

    def __init__(
        self,
        effect: LimiterEffect,
        coordinate: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        z_limits: tuple[float, float],
        extinction_coefficient_nepers: float,
    ):
        """
        Inits LimiterAnalyticBoundingBox3D.

        Parameters
        ----------
        effect : LimiterEffect
            Effect applied on intersection with limiter.
        coordinate : literal['xy', 'xz', 'yz', 'rz']
            Coordinate pair being used.
        x_limits : tuple[float, float]
            Minimum and maximum x coordinate value.
        y_limits : tuple[float, float]
            Minimum and maximum y coordinate value.
        z_limits : tuple[float, float]
            Minimum and maximum z coordinate value.
        extinction_coefficient_nepers : float
            Extinction coefficient applied on intersection [nepers].

        Raises
        ------
        ValueError
            Invalid coordinate.
        """
        super().__init__(extinction_coefficient_nepers)
        self.effect = effect
        self.coordinate = str(coordinate).lower()
        self.x_limits = np.asarray(x_limits, dtype=float).reshape(2)
        self.y_limits = np.asarray(y_limits, dtype=float).reshape(2)
        self.z_limits = np.asarray(z_limits, dtype=float).reshape(2)

        if self.coordinate != self.XYZ:
            raise ValueError(
                f"Invalid coordinate '{coordinate}'. "
                "Valid coordinates: " + f"{self.XYZ}"
            )

    @classmethod
    def from_dict_toml(cls, document: dict):
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        document : dict
            Dictionary of data read from toml file.

        Returns
        -------
        limiter : LimiterAnalyticBoundingBox3D
            Limiter defined as a 3D bounding box.
        """
        validator = SystemDataTomlValidator()
        validator.validate(document, cls.toml_schema)

        return cls(
            validator.document["effect"],
            validator.document["coordinate"],
            validator.document["x_limits"],
            validator.document["y_limits"],
            validator.document["z_limits"],
            validator.document["extinction_coefficient_nepers"],
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        d = super().to_dict_toml()
        d["effect"] = self.effect.name
        d["source"] = ANALYTIC
        d["shape"] = BOUNDING_BOX_3D
        d["coordinate"] = self.coordinate
        d["x_limits"] = self.x_limits
        d["y_limits"] = self.y_limits
        d["z_limits"] = self.z_limits

        return d


LimiterSchemaType = (
    LimiterImasBoundingBox2D
    | LimiterImas2D
    | LimiterImas3D
    | LimiterNetcdf
    | LimiterAnalyticBoundingBox2D
    | LimiterAnalyticBoundingBox3D
)

schema_source = {
    "source": {"required": True, "type": "string", "coerce": (str, str.lower)}
}

schema_shape = {
    "shape": {"required": True, "type": "string", "coerce": (str, str.lower)},
}


def parse_limiters(document: dict) -> dict[str, LimiterSchemaType]:
    """
    Parse limiter models from document read from TOML file.

    Parameters
    ----------
    document : dict
        Document defining model.

    Returns
    -------
    limiter_models : dict[str, LimiterSchemaType]
        Loaded limiter models.

    Raises
    ------
    ValueError
        Unknown limiter type.
        Incorrect limiter shape for data source.
    """
    validator = SystemDataTomlValidator()

    limiters = {}

    for name, config in document.items():
        _name = str(name).lower()
        validator.validate(config, schema_source, allow_unknown=True)

        source = validator.document["source"]

        if source.startswith(LimiterImas.source_prefix):
            validator.validate(config, schema_shape, allow_unknown=True)
            shape = validator.document["shape"]

            if shape == BOUNDING_BOX_2D:
                limiters[_name] = LimiterImasBoundingBox2D.from_dict_toml(
                    validator.document
                )
            elif shape == _2D:
                limiters[_name] = LimiterImas2D.from_dict_toml(
                    validator.document
                )
            elif shape == _3D:
                limiters[_name] = LimiterImas3D.from_dict_toml(
                    validator.document
                )
            elif shape == BOUNDING_BOX_3D:
                raise ValueError(
                    f"limiters.{name} has invalid shape '{shape}'"
                    f"for source '{ANALYTIC}'"
                )
            else:
                raise ValueError(
                    f"limiters.{name} has unknown shape '{shape}'"
                )
        elif source.startswith(LimiterNetcdf.source_prefix):
            limiters[_name] = LimiterNetcdf.from_dict_toml(validator.document)
        elif source == ANALYTIC:
            validator.validate(config, schema_shape, allow_unknown=True)
            shape = validator.document["shape"]

            if shape == BOUNDING_BOX_2D:
                limiters[_name] = LimiterAnalyticBoundingBox2D.from_dict_toml(
                    validator.document
                )
            elif shape == BOUNDING_BOX_3D:
                limiters[_name] = LimiterAnalyticBoundingBox3D.from_dict_toml(
                    validator.document
                )
            elif shape in {_2D, _3D}:
                raise ValueError(
                    f"limiters.{name} has invalid shape '{shape}'"
                    f"for source '{ANALYTIC}'"
                )
            else:
                raise ValueError(
                    f"limiters.{name} has unknown shape '{shape}'"
                )
        else:
            raise ValueError(f"limiters.{name} has unknown source '{source}'")

    return limiters


DATA_SOURCES = "data_sources"
COORDINATES = "coordinates"
KINETIC = "kinetic"
MAGNETIC_FIELD_T = "magnetic_field_t"
LIMITERS = "limiters"

schema_top_level = {
    DATA_SOURCES: {
        "required": False,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
    COORDINATES: {
        "required": False,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
    KINETIC: {
        "required": True,
        "type": "dict",
        "schema": {
            ELECTRON_DENSITY_PER_M3: {"required": True, "type": "dict"},
            ELECTRON_TEMPERATURE_EV: {"required": True, "type": "dict"},
            EFFECTIVE_CHARGE: {"required": True, "type": "dict"},
        },
    },
    MAGNETIC_FIELD_T: {
        "required": True,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
    LIMITERS: {
        "required": False,
        "type": "dict",
        "keysrules": {"type": "string"},
    },
}


def parse_schema(
    document: dict,
) -> tuple[
    dict[str, DataSourceType],
    dict[CoordinateSystem, CoordinateType],
    KineticModelType,
    KineticModelType,
    KineticModelType,
    MagneticModelType,
    dict[str, LimiterSchemaType],
]:
    """
    Parse system data schema from document read from TOML file.

    Parameters
    ----------
    document : dict
        Document defining model.

    Returns
    -------
    data_sources : dict[str, DataSourceType]
        Dictionary of data sources.
    coordinates : dict[CoordinateSystem, CoordinateType]
        Dictionary of data sources.
    electron_density_per_m3 : KineticModelType
        Loaded electron density model.
    electron_temperature_ev : KineticModelType
        Loaded electron density model.
    effective_charge : KineticModelType
        Loaded electron density model.
    magnetic_field_model : MagneticModelType
        Loaded electron density model.
    limiter_models : dict[str, LimiterSchemaType]
        Loaded limiter models.
    """
    validator = SystemDataTomlValidator()
    validator.validate(document, schema_top_level)

    # Parse external data sources.
    if DATA_SOURCES in validator.document:
        data_sources = parse_data_sources(validator.document[DATA_SOURCES])
    else:
        data_sources = {}

    # Parse coordinate systems.
    if COORDINATES in validator.document:
        coordinates = parse_coordinates(validator.document[COORDINATES])
    else:
        coordinates = {}

    # Parse kinetic models.
    (electron_density_per_m3, electron_temperature_ev, effective_charge) = (
        parse_kinetic(validator.document[KINETIC])
    )

    # Parse magnetic model.
    magnetic_field_t = parse_magnetic(validator.document[MAGNETIC_FIELD_T])

    # Parse limiters.
    limiters = parse_limiters(validator.document[LIMITERS])

    return (
        data_sources,
        coordinates,
        electron_density_per_m3,
        electron_temperature_ev,
        effective_charge,
        magnetic_field_t,
        limiters,
    )
