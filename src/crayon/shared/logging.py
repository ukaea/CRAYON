"""
Helpers for logging.
"""

# Standard imports
import argparse
import logging

# Local imports

logger = logging.getLogger(__name__)


def add_logging_arguments(parser: argparse.ArgumentParser):
    """
    Add arguments to argparse.ArgumentParser to control log level.

    Parameters
    ----------
    parser : argparse.Parser
        Argparse parser.
    """
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--debug",
        dest="log_level",
        action="store_const",
        default=logging.WARNING,
        const=logging.DEBUG,
        help="Set DEBUG logging level for terminal output",
    )
    logging_group.add_argument(
        "--info",
        dest="log_level",
        action="store_const",
        const=logging.INFO,
        help="Set INFO logging level for terminal output",
    )
    logging_group.add_argument(
        "--warning",
        dest="log_level",
        action="store_const",
        const=logging.WARNING,
        help="Set WARNING logging level for terminal output",
    )
    logging_group.add_argument(
        "--error",
        dest="log_level",
        action="store_const",
        const=logging.ERROR,
        help="Set ERROR logging level for terminal output",
    )
    logging_group.add_argument(
        "--critical",
        dest="log_level",
        action="store_const",
        const=logging.CRITICAL,
        help="Set CRITICAL logging level for terminal output",
    )
