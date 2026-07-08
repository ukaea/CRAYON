"""
Wrapper scripts for electron cyclotron emission (ECE) and electron Bernstein
emission (EBE) models.
"""

# Standard imports
import logging
import pathlib

# Local imports

logger = logging.getLogger(__name__)


def emission(run_directory: pathlib.Path):
    """
    Wrapper script for electron cyclotron emission (ECE) and electron Bernstein
    emission (EBE) models.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    """
    raise NotImplementedError
