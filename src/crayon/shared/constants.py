"""
Useful constants.
"""

# Standard imports
import logging

# Third party imports
import numpy as np
import scipy.constants as const

# Local imports
from crayon.shared.data_structures import CrayonEnum

logger = logging.getLogger(__name__)

TWO_PI = 2 * np.pi
TWO_PI3 = TWO_PI * TWO_PI * TWO_PI
TWO_PI_INV = 1 / TWO_PI
SQRT_PI = np.sqrt(np.pi)

RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0

NONE = "NONE"

# Speed of light
C_M_PER_S = const.speed_of_light
C_M_PER_NS = const.speed_of_light * 1e-9

# 6D Symplectic matrix.
SYMPLECTIC_MATRIX_J6 = np.eye(6, k=3, dtype=float) - np.eye(
    6, k=-3, dtype=float
)

# Limits on normalised plasma parameters.
# Minimum normalised density before considered vacuum.
MIN_NORM_ELECTRON_DENSITY = 1.0e-4

# Minimum normalised electron temperature before considered cold.
MIN_NORM_ELECTRON_TEMPERATURE = 5.0e-5

# Maximum normalised electron temperature before relativistic effects
# considered. Corresponds to Te = 5.11keV.
MAX_NONRELATIVISTIC_NORM_ELECTRON_TEMPERATURE = 0.01

# Minimum normalised magnetic field strength before considered unmagnetised.
MIN_NORM_MAGNETIC_FIELD_STRENGTH = 1.0e-4

# Impedance of free space [Ohms].
Z0 = const.physical_constants["characteristic impedance of vacuum"][0]

# Admittance of free space [Siemens].
Y0 = 1 / Z0

# Directory names.
INPUT = "input"
OUTPUT = "output"
PLOTS = "plots"

# File names.
OPTIONS_TOML = "options.toml"
RAYS_TOML = "rays.toml"
SYSTEM_DATA_TOML = "system_data.toml"
INPUT_DATA_NETCDF = "input.nc"
OPTIMAL_OX_TOML = "optimal_ox.toml"
RAYS_NETCDF = "rays.nc"


# Shared enumerations.
class AngleFormat(CrayonEnum):
    """
    Angle format.

    Attributes
    ----------
    DEGREES
        Degrees.
    RADIANS
        Radians.
    """

    DEGREES = 1
    RADIANS = 2


class DispersionType(CrayonEnum):
    """
    Dispersion tensor model type.

    Attributes
    ----------
    COLD
        Cold plasma model.
    NON_RELATIVISTIC
        Kinetic plasma model without relativistic effects.
    FULLY_RELATIVISTIC
        Kinetic plasma model with relativistic effects.
    """

    COLD = 1
    NON_RELATIVISTIC = 2
    FULLY_RELATIVISTIC = 3


class WaveMode(CrayonEnum):
    """
    Plasma microwave mode.

    Attributes
    ----------
    ANY
        Either O or X mode. Used in vacuum.
    O
        Ordinary mode.
    X
        Extraordinary mode.
    """

    ANY = 0
    ORDINARY = 1
    O = ORDINARY  # noqa: E741 - This is well defined in literature.
    EXTRAORDINARY = 2
    X = EXTRAORDINARY


class CoordinateSystem(CrayonEnum):
    """
    Coordinate system.

    Attributes
    ----------
    CARTESIAN
        Cartesian (x, y, z).
    CYLINDRICAL
        Cylindrical (r, phi, z).
    SPHERICAL
        Spherical (r, theta, phi).
    TOROIDAL
        Toroidal (r, phi, theta).
    RHO_POLOIDAL
        Root normalised poloidal flux (rho_p, phi, theta).
    RHO_TOROIDAL
        Root normalised toroidal flux (rho_t, phi, theta).

    Properties
    ----------
    symbols
        Symbols for coordinate components.
    units
        Units of coordinate components.
    angular_components
        Indicies of components which are angles.
    """

    CARTESIAN = 0
    CYLINDRICAL = 1
    SPHERICAL = 3
    TOROIDAL = 4
    RHO_POLOIDAL = 5
    RHO_TOROIDAL = 6

    # Aliases.
    POLOIDAL_FLUX_ROOT_NORMALISED = RHO_POLOIDAL
    TOROIDAL_FLUX_ROOT_NORMALISED = RHO_TOROIDAL

    @property
    def symbols(self) -> tuple[str, str, str]:
        """Symbols for coordinate components."""
        if self == CoordinateSystem.CARTESIAN:
            return ("x", "y", "z")
        if self == CoordinateSystem.CYLINDRICAL:
            return ("r", "phi", "z")
        if self == CoordinateSystem.TOROIDAL:
            return ("r", "phi", "theta")
        if self == CoordinateSystem.RHO_POLOIDAL:
            return ("rho_poloidal", "phi", "theta")
        if self == CoordinateSystem.RHO_TOROIDAL:
            return ("rho_toroidal", "phi", "theta")
        raise NotImplementedError(self.name)

    @property
    def units(self) -> str:
        """Units of coordinate components."""
        if self == CoordinateSystem.CARTESIAN:
            return "m"
        if self in {CoordinateSystem.CYLINDRICAL, CoordinateSystem.TOROIDAL}:
            return "m, rad"
        if self in {
            CoordinateSystem.RHO_POLOIDAL,
            CoordinateSystem.RHO_TOROIDAL,
        }:
            return ", rad"
        raise NotImplementedError(self.name)

    @property
    def angular_components(self) -> tuple[int]:
        """Indicies of components which are angles."""
        if self == CoordinateSystem.CARTESIAN:
            return ()
        if self == CoordinateSystem.CYLINDRICAL:
            return (1,)
        if self in {
            CoordinateSystem.TOROIDAL,
            CoordinateSystem.RHO_POLOIDAL,
            CoordinateSystem.RHO_TOROIDAL,
        }:
            return (1, 2)
        raise NotImplementedError(self.name)
