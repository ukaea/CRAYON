"""
Unmagnetised slab EC integration test.
"""

# Standard imports
import logging
import pathlib

# Local imports
from crayon.scripts import plot_single, trace

logger = logging.getLogger(__name__)

this_directory = pathlib.Path(__file__).parent
run_directory = this_directory.joinpath("unmagnetised")


def main():
    """
    Run unmagnetised slab EC integration test.
    """
    trace(run_directory, [0.0], overwrite=True)
    plot_single(
        run_directory,
        "ray_1-0",
        [0.0],
        xy=True,
        plasma_parameters=True,
        hamiltonian=True,
        power=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S",
    )
    main()
