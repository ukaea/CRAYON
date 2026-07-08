"""
Vacuum integration test.
"""

# Standard imports
import logging
import pathlib

# Local imports
from crayon.scripts import plot_all, plot_single, trace

logger = logging.getLogger(__name__)

run_directory = pathlib.Path(__file__).parent


def main():
    """
    Run vacuum integration test.
    """
    trace(run_directory, [0.0], overwrite=True)
    plot_single(run_directory, "ray_1-0", [0.0], xy=True, power=True)
    plot_single(run_directory, "ray_2-0", [0.0], xy=True, power=True)
    plot_all(
        run_directory,
        [0.0],
        xy=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S",
    )
    main()
