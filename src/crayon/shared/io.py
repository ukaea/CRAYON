"""
Helpers for file input output.
"""

# Standard imports
import abc
import logging
import typing

# Third party imports
import cerberus
import netCDF4 as nc4  # noqa: N813
import numpy as np
import toml

# Local imports
from crayon.shared.constants import AngleFormat, DispersionType, WaveMode
from crayon.shared.data_structures import Dimension
from crayon.shared.types import Array, FloatArray

logger = logging.getLogger(__name__)

_NC4_FLOAT: str = "f8"
_NC4_COMPLEX: str = "c16"
_NC4_BYTE: str = "u1"
_NC4_INT: str = "i8"

_NC4_COMPRESSION_ARGS = {
    "zlib": True,
    "complevel": 7,
    "fletcher32": True,
}


def get_netcdf_dimensions(
    dset: nc4.Dataset | nc4.Group,
) -> list[nc4.Dimension]:
    """
    Get dimensions from netCDF dataset / group.

    Parameters
    ----------
    dset : netCDF4.Dataset | netCDF4.Group
        netCDF4 dataset or group.

    Returns
    -------
    dimensions : list[netCDF4.Dimension]

    Raises
    ------
    ValueError
        netCDF4 dataset has > 128 levels.
    """
    dimensions = dset.dimensions.copy()

    _dset = dset
    for _ in range(128):
        dimensions.update(_dset.dimensions)

        if _dset.parent is None:
            break
        _dset = _dset.parent

    if _dset.parent is not None:
        raise ValueError("netCDF4 file depth > 128")

    return dimensions


def create_netcdf_variable(
    dset: nc4.Dataset,
    name: str,
    dimensions: tuple[Dimension],
    dtype: type,
    description: str,
    units: str,
) -> nc4.Variable:
    """
    Create netCDF4 dataset variable.

    Parameters
    ----------
    dset : nc4.Dataset
        netCDF4 dataset.
    name : str
        Name of variable.
    dimensions : tuple[Dimension]
        Dimensions of signals.
    dtype : type
        Datatype of value.
    description : str
        Description of signal.
    units : str
        Units of signal.

    Returns
    -------
    variable : netCD4.Variable
        netCDF4 variable.
    """
    if np.issubdtype(dtype, np.complexfloating):
        dtype = _NC4_COMPLEX
    elif np.issubdtype(dtype, np.floating):
        dtype = _NC4_FLOAT
    elif np.issubdtype(dtype, np.integer):
        dtype = _NC4_INT
    elif np.issubdtype(dtype, np.bool_):
        dtype = _NC4_BYTE
    else:
        raise NotImplementedError(dtype)

    var = dset.createVariable(
        name, dtype, (d.name for d in dimensions), **_NC4_COMPRESSION_ARGS
    )

    # Metadata.
    var.setncattr("description", description)
    var.setncattr("units", units)

    return var


def set_netcdf_variable(
    dset: nc4.Dataset,
    name: str,
    data: Array,
    /,
    *,
    variable: nc4.Variable = None,
    index: int | None = None,
):
    """
    Set data in netCDF4 variable.

    Parameters
    ----------
    dset : nc4.Dataset
        netCDF4 dataset.
    name : str
        Name of variable.
    data : Array
        Array.
    variable : netCDF4.Variable, optional
        netCDF4 variable to put data in.
    index : int | None, optional
        Index in array to put array.
    """
    if variable is None:
        variable = dset[name]

    _data = np.asarray(data)
    if np.issubdtype(_data.dtype, np.bool_):
        # Boolean data has to be saved as integers.
        _data = _data.astype(int)

    if index:
        variable[index, ...] = _data
    else:
        variable[...] = _data


def write_netcdf_variable(
    dset: nc4.Dataset,
    name: str,
    dimensions: tuple[Dimension],
    data: FloatArray,
    description: str,
    units: str,
):
    """
    Create and write data into netCDF4 dataset variable.

    Parameters
    ----------
    dset : nc4.Dataset
        netCDF4 dataset.
    name : str
        Name of variable.
    dimensions : tuple[Dimension]
        Dimensions of signals.
    data : Array
        Array.
    description : str
        Description of signal.
    units : str
        Units of signal.

    Notes
    -----
    Need to have opened the dset using auto_complex=True.
    """
    _type = data.dtype if isinstance(data, np.ndarray) else type(data)

    var = create_netcdf_variable(
        dset, name, dimensions, _type, description, units
    )

    set_netcdf_variable(dset, name, data, variable=var)


def read_netcdf_variable(
    variable: nc4.Variable,
) -> tuple[FloatArray, tuple[str], str, str]:
    """
    Read netcdf variable and return data, dimensions, description and units.

    Parameters
    ----------
    variable : netCD4.Variable
        netCDF4 variable.

    Returns
    -------
    data : array
        Data.
    dimensions : list[str]
        List of dimension names.
    description : str
        Description of variable.
    units : str
        Units of variable
    """
    data = variable[...].data
    dimensions = variable.dimensions
    description = variable.description
    units = variable.units

    return data, dimensions, description, units


class IONetcdf(abc.ABC):
    """
    A class that can be read / written to netCDF4 files.

    Methods
    -------
    write_netcdf
        Write object data to netCDF4.
    read_netcdf
        Create object from netCDF4 data.
    """

    __slots__ = ()

    @abc.abstractmethod
    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write object contents to netCDF4 dataset.
        """

    @classmethod
    @abc.abstractmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "IONetcdf":
        """
        Create object from netCDF4 dataset.
        """


def _dict_keys_to_lower_case(d: dict):
    """
    Convert all dictionary keys to lower case.

    Parameters
    ----------
    d : dict
        Dictionary to convert.

    Returns
    -------
    d_converted : dict
        Converted dictionary.
    """
    new_dict = {}

    for key, value in d.items():
        new_key = key.lower()

        if isinstance(value, dict):
            new_dict[new_key] = _dict_keys_to_lower_case(value)
        else:
            new_dict[new_key] = value

    return new_dict


class IOToml(abc.ABC):
    """
    A class that can be read / written to TOML files.

    Methods
    -------
    numpy_encoder

    write_netcdf
        Write object data to TOML.
    read_netcdf
        Create object from TOML data
    """

    __slots__ = ()

    numpy_encoder = toml.TomlNumpyEncoder(preserve=True)

    @abc.abstractmethod
    def to_dict_toml(self) -> dict:
        """
        Create dictionary containing object contents that can be serialised to
        TOML i.e. containing only python built-ins.

        Returns
        -------
        d : dict
            Object data
        """

    @classmethod
    @abc.abstractmethod
    def from_dict_toml(cls, d: dict) -> "IOToml":
        """
        Create object from dictionary de-serialised from a TOML file i.e.
        containing only python built-ins.

        Parameters
        ----------
        d : dict
            Object data

        Returns
        -------
        obj : IOToml
            Object with data from dictionary.
        """

    def write_toml(self, fh: typing.IO[str]):
        """
        Write object contents to file stream in TOML format.

        Parameters
        ----------
        fh : TextIO
            File handle to write to.
        """
        toml.dump(self.to_dict_toml(), fh, encoder=self.numpy_encoder)

    @classmethod
    def read_toml(cls, fh: typing.IO[str]) -> "IOToml":
        """
        Create object from file stream in TOML file.

        Parameters
        ----------
        fh : TextIO
            File handle to read from.

        Returns
        -------
        obj : IOToml
            Object with data from file.
        """
        data = _dict_keys_to_lower_case(toml.load(fh))
        return cls.from_dict_toml(data)


class TomlValidator(cerberus.Validator):
    """
    Extended cerberus.Validator to help TOML read/write.
    """

    types_mapping = cerberus.Validator.types_mapping.copy()

    types_mapping["ndarray"] = cerberus.TypeDefinition(
        "ndarray", (np.ndarray,), ()
    )
    types_mapping["AngleFormat"] = cerberus.TypeDefinition(
        "AngleFormat", (AngleFormat,), ()
    )
    types_mapping["DispersionType"] = cerberus.TypeDefinition(
        "DispersionType", (DispersionType,), ()
    )
    types_mapping["WaveMode"] = cerberus.TypeDefinition(
        "WaveMode", (WaveMode,), ()
    )

    def validate(self, *args, allow_unknown: bool = False, **kwargs) -> dict:
        """
        Validate dictionary against schema.

        Parameters
        ----------
        allow_unknown : bool, optional
            If True, ignore items in the dictionary that don't appear in the
            schema.

        Returns
        -------
        document : dict
            Validated dictionary.

        Raises
        ------
        ValueError
            Document failed validation.
        """
        allow_unknown_original = self.allow_unknown

        self.allow_unknown = allow_unknown
        return_value = super().validate(*args, **kwargs)
        self.allow_unknown = allow_unknown_original

        if self.errors:
            for key, error_list in self.errors.items():
                for error in error_list:
                    logger.error("%s: %s", key, error)

            raise ValueError("TOML failed validation")

        return return_value
