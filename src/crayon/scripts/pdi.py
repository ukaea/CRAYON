"""
Wrapper scripts for parametric decay instability (PDI) models.
"""

# Standard imports
import logging
import pathlib

# Local imports

logger = logging.getLogger(__name__)


def pdi(run_directory: pathlib.Path):
    """
    Wrapper for parametric decay instability (PDI) models.

    Attributes
    ----------
    run_directory : pathlib.Path
        Crayon run directory.
    """
