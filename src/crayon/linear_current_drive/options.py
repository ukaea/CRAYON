"""
Object for options controlling linear current drive model.
"""

# Standard imports
import logging

# Third party imports
import netCDF4 as nc4  # noqa: N813

# Local imports
from crayon.shared.io import (
    IONetcdf,
    IOToml,
    TomlValidator,
    write_netcdf_variable,
)

logger = logging.getLogger(__name__)

N_RHO = 51

_schema = {
    "n_rho": {
        "required": True,
        "type": "integer",
        "coerce": int,
        "min": 1,
    },
    "rho_poloidal_grid": {
        "required": True,
        "type": "boolean",
    },
}


class OptionsLinearCurrentDrive(IONetcdf, IOToml):
    """
    Options for linear current drive.

    Attributes
    ----------
    n_rho : int
        Number of radial points used to evaluate flux function profiles
        e.g current density, power density.
    rho_poloidal_grid : bool
        If True, results are evaluated on a equispaced root normalised
        poloidal flux grid. Otherwise it is evaluated on a equispaced root
        normalised toroidal flux grid.
    """

    __slots__ = ("n_rho", "rho_poloidal_grid")

    section_name = "linear_current_drive"

    def __init__(
        self, /, *, n_rho: int = N_RHO, rho_poloidal_grid: bool = True
    ):
        """
        Inits OptionsLinearCurrentDrive.

        Parameters
        ----------
        n_rho : int, optional
            Number of radial points used to evaluate flux function profiles
            e.g current density, power density.
        rho_poloidal_grid : bool, optional
            If True, results are evaluated on a equispaced root normalised
            poloidal flux grid. Otherwise it is evaluated on a equispaced root
            normalised toroidal flux grid.
        """
        self.n_rho = int(n_rho)
        self.rho_poloidal_grid = bool(rho_poloidal_grid)

    def write_netcdf(self, group: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Group
            netCDF4 dataset or group to write data to.
        """
        write_netcdf_variable(
            group,
            "n_rho",
            (),
            self.n_rho,
            (
                "Number of radial points used to evaluate flux function "
                "profiles e.g current density, power density."
            ),
            "",
        )

        write_netcdf_variable(
            group,
            "rho_poloidal_grid",
            (),
            int(self.rho_poloidal_grid),
            (
                "If True, calculations are performed on equispaced normalised "
                "poloidal flux grid, otherwise toroidal flux."
            ),
            "",
        )

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "OptionsLinearCurrentDrive":
        """
        Load from netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Group
            netCDF4 dataset or group to read data from.

        Returns
        -------
        options_linear_current_drive : OptionsLinearCurrentDrive
            Options for linear current drive model.
        """
        n_rho = group["n_rho"][...].item()
        rho_poloidal_grid = group["rho_poloidal_grid"][...].item() == 1

        return cls(n_rho=n_rho, rho_poloidal_grid=rho_poloidal_grid)

    def to_dict_toml(self) -> dict:
        """
        Create dictionary containing object contents that can be serialised to
        TOML i.e. containing only python built-ins.

        Returns
        -------
        dict_toml
            Object data in dictionary for serialisation to TOML.
        """
        return {
            "n_rho": self.n_rho,
            "rho_poloidal_grid": self.rho_poloidal_grid,
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "OptionsLinearCurrentDrive":
        """
        Create object from dictionary de-serialised from a TOML file i.e.
        containing only python built-ins.

        Parameters
        ----------
        d : dict
            Dictionary of object data loaded from TOML.

        Returns
        -------
        options_linear_current_drive : OptionsLinearCurrentDrive
            Options for linear current drive model.
        """
        validator = TomlValidator()
        validator.validate(d, _schema)

        return cls(
            n_rho=validator.document["n_rho"],
            rho_poloidal_grid=validator.document["rho_poloidal_grid"],
        )
