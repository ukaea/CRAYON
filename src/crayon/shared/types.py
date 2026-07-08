"""
Shared array datatypes and type hints.
"""

# Standard imports
import logging
import pathlib

# Third party imports
import numpy as np
import numpy.typing as npt

# Local imports

logger = logging.getLogger(__name__)

# Types used for arrays.
BoolType = bool
IntType = np.intp
FloatType = np.double
ComplexType = np.cdouble

# Type hints for Python code.
NumericType = int | float | complex | np.number
FilepathType = pathlib.Path | str

ArrayLike = npt.ArrayLike
Array = npt.NDArray[np.float64 | np.complex128]
BooleanArray = npt.NDArray[BoolType]
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.int64]
