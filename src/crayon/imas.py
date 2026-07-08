"""
Helpers for IMAS coupling.
"""

# Standard imports
import logging
import warnings

# Third party imports
import imas as _imas
import numpy as np
from imas_core import imasdef

# Local imports

logger = logging.getLogger(__name__)

DD_VERSION = "3.42.0"

# Check version of imas.
if hasattr(_imas, "al_dd_version"):
    # Python bindings from imas-core.
    if _imas.al_dd_version != DD_VERSION:
        warnings.warn(
            "This package is compatible with data dictionary version "
            "3.42.0. Using other versions may result in errors.",
            stacklevel=2,
        )

    imas = _imas
    DBEntry = _imas.DBEntry
else:
    # imas-python package.
    imas = _imas.IDSFactory(DD_VERSION)

    class DBEntry(_imas.DBEntry):
        """
        IMAS Database entry with a fixed data dictionary version. Mocks
        version obtained from python bindings to IMAS core.
        """

        def __init__(self, uri: str, mode: str):
            """
            Inits DBEntry.

            Parameters
            ----------
            uri : str
                Uniform resource identifier for IMAS database.
            mode : str
                File open mode.
            """
            super().__init__(uri, mode, dd_version=DD_VERSION)


def ids_empty(value) -> bool:
    """
    Check if IDS field is empty.

    Parameters
    ----------
    value : any
        Value to be checked.

    Returns
    -------
    ids_empty : bool
        Flag if IDS field is empty.

    Notes
    -----
    IDS fields with no value are set with a dummy value. This function returns
    True if the field still has that dummy value. For arrays only the first
    value is checked.
    """
    _value = np.asarray(value)
    return _value.size == 0 or (
        _value.size > 0 and _value.item(0) == imasdef.EMPTY_FLOAT
    )


def check_ids_empty(value: np.ndarray, field: str):
    """
    Raise exception if IDS field is empty.

    Parameters
    ----------
    field : str
        Name of IDS field to appear in exception.

    Raises
    ------
    IDSEmptyError
        IDS field is empty.
    """
    if ids_empty(value):
        raise IDSEmptyError(field)


class IDSEmptyError(Exception):
    """
    Raised when data field read from IDS is empty.
    """

    __slots__ = ()

    def __init__(self, field: str):
        """
        Inits IDSEmptyError.

        Parameters
        ----------
        field : str
            Name of IDS field.
        """
        super().__init__(f"IDS {field} is empty")
