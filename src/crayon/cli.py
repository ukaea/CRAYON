"""
Crayon command line interface.
"""

# Standard imports
import argparse
import logging
import pathlib
import sys

# Third party imports.
import numpy as np

# Local imports
from crayon.scripts import (
    couple_cql3d,
    current,
    emission,
    new,
    optimise,
    pdi,
    plot_all,
    plot_single,
    trace,
)

logger = logging.getLogger(__name__)

# Raise an error on floating point errors.
np.seterr(divide="raise", over="raise", invalid="raise")


def cli(args):
    """
    Crayon command line interface.

    Available Commands
    ------------------
    crayon new = initialise code directory structure
    crayon optimise = optimise launch angles and polarisation for OX conversion
    crayon trace = run ray tracing
    crayon plot <> = Create standard plots.

    crayon current = run linear current drive
    crayon emission = run ECE / EBE model
    crayon pdi = run parametric decay model

    crayon couple-cql3d = re-write output into GENRAY style
    crayon couple-luke
    crayon dump-data = write netcdf file containing all input data
    """
    """
    crayon new <path>
    crayon optimise --dir <path> --times 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    crayon trace
        --dir <path>
        --times 0.2 0.3 0.4 0.5 0.6 0.7 0.8
        --rays <>
    """
    parser = argparse.ArgumentParser(
        description="Crayon fully relativistic kinetic ray tracing code"
    )

    parser.add_argument("path", help="Crayon run directory")

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing data.",
    )

    # Logging arguments.
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

    # Subcommand parsers.
    subparsers = parser.add_subparsers(required=True, dest="command")

    # Crayon run initialisation subparser.
    _parser_new = subparsers.add_parser(
        "new", help="Initialise blank Crayon run directory."
    )

    # Optimise OX conversion angles / polarisation subparser.
    parser_optimise = subparsers.add_parser(
        "optimise",
        help="Optimise launch angles and polarisation for OX conversion.",
    )

    parser_optimise.add_argument(
        "times",
        type=float,
        action="store",
        nargs="+",
        help="Time slices for calculation [s].",
    )

    # Ray tracing subparser.
    parser_trace = subparsers.add_parser("trace", help="Run ray tracing.")

    parser_trace.add_argument(
        "times",
        action="store",
        nargs="+",
        help="Time slices for calculation [s].",
    )

    parser_trace.add_argument(
        "--max-workers",
        dest="max_workers",
        action="store",
        type=int,
        help="Maximum number of simultaneous processes to run rays.",
        default=1,
    )

    parser_trace.add_argument(
        "--current",
        action="store_true",
        help="Run linear current drive model.",
    )

    parser_trace.add_argument(
        "--emission", action="store_true", help="Run ECE / EBE emission model."
    )

    parser_trace.add_argument(
        "--pdi", action="store_true", help="Run parametric decay model."
    )

    # Linear current drive model subparser.
    _parser_current = subparsers.add_parser(
        "current", help="Run linear current drive model on ray tracing output"
    )

    # ECE / EBE emission model subparser.
    _parser_emission = subparsers.add_parser(
        "emission", help="Run ECE / EBE emission model on ray tracing output"
    )

    # Parametric decay model subparser.
    _parser_pdi = subparsers.add_parser(
        "pdi",
        help="Run parametric decay instability model on ray tracing output",
    )

    # Plotting subparser.
    parser_plot = subparsers.add_parser(
        "plot", help="Generate standard plots of Crayon output for all rays."
    )

    parser_plot.add_argument(
        "times",
        action="store",
        nargs="+",
        help="Time slices for calculation [s].",
    )

    parser_plot.add_argument(
        "--xy",
        dest="xy",
        action="store_true",
        default=False,
        help="Plot ray position x vs y.",
    )

    parser_plot.add_argument(
        "--xz",
        dest="xz",
        action="store_true",
        default=False,
        help="Plot ray position x vs z.",
    )

    parser_plot.add_argument(
        "--yz",
        dest="yz",
        action="store_true",
        default=False,
        help="Plot ray position y vs z.",
    )

    parser_plot.add_argument(
        "--rz",
        dest="rz",
        action="store_true",
        default=False,
        help="Plot ray position r vs z.",
    )

    parser_plot_single = subparsers.add_parser(
        "plot_single",
        help="Generate standard plots of Crayon output for a single ray.",
    )

    parser_plot_single.add_argument("ray_name", help="Name of ray to plot.")

    parser_plot_single.add_argument(
        "times",
        action="store",
        nargs="+",
        help="Time slices for calculation [s].",
    )

    parser_plot_single.add_argument(
        "--plasma-params",
        dest="plasma_params",
        action="store_true",
        default=False,
        help="Plot plasma parameters along ray.",
    )

    parser_plot_single.add_argument(
        "--hamiltonian",
        dest="hamiltonian",
        action="store_true",
        default=False,
        help="Plot Hamiltonian along ray.",
    )

    parser_plot_single.add_argument(
        "--mode-conversion",
        dest="mode_conversion",
        action="store_true",
        default=False,
        help="Plot mode conversion alarm.",
    )

    parser_plot_single.add_argument(
        "--power",
        dest="power",
        action="store_true",
        default=False,
        help="Plot power along ray.",
    )

    parser_plot_single.add_argument(
        "--damping",
        dest="damping",
        action="store_true",
        default=False,
        help="Plot power against flux coordinate.",
    )

    parser_plot_single.add_argument(
        "--xy",
        dest="xy",
        action="store_true",
        default=False,
        help="Plot ray position x vs y.",
    )

    parser_plot_single.add_argument(
        "--xz",
        dest="xz",
        action="store_true",
        default=False,
        help="Plot ray position x vs z.",
    )

    parser_plot_single.add_argument(
        "--yz",
        dest="yz",
        action="store_true",
        default=False,
        help="Plot ray position y vs z.",
    )

    parser_plot_single.add_argument(
        "--rz",
        dest="rz",
        action="store_true",
        default=False,
        help="Plot ray position r vs z.",
    )

    parser_plot_single.add_argument(
        "--osculating-plane",
        dest="osculating_plane",
        action="store_true",
        default=False,
        help="Plot dispersion curves in mode conversion osculating plane.",
    )

    # Cql3d coupling subparser.
    _parser_cql3d = subparsers.add_parser(
        "couple-cql3d", help="Convert to GENRAY style output to run CQL3D."
    )

    # plot ray <ray_name> --rz --xy --power --plasma-params etc.
    # plot current <ray_name> --q --j --ql-threshold

    # Parse arguments.
    args = parser.parse_args(args)
    run_directory = pathlib.Path(args.path).resolve()

    # Configure logging.
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S",
    )

    # Run command.
    if args.command == "new":
        new(run_directory, overwrite=args.overwrite)
    elif args.command == "optimise":
        optimise(run_directory, args.times, overwrite=args.overwrite)
    elif args.command == "trace":
        trace(
            run_directory,
            [float(t) for t in args.times],
            overwrite=args.overwrite,
            max_workers=args.max_workers,
        )
    elif args.command == "current":
        current(run_directory)
    elif args.command == "emission":
        emission(run_directory)
    elif args.command == "pdi":
        pdi(run_directory)
    elif args.command == "plot":
        plot_all(
            run_directory,
            [float(t) for t in args.times],
            xy=args.xy,
            xz=args.xz,
            yz=args.yz,
            rz=args.rz,
        )
    elif args.command == "plot_single":
        plot_single(
            run_directory,
            args.ray_name,
            [float(t) for t in args.times],
            xy=args.xy,
            xz=args.xz,
            yz=args.yz,
            rz=args.rz,
            plasma_parameters=args.plasma_params,
            hamiltonian=args.hamiltonian,
            mode_conversion=args.mode_conversion,
            power=args.power,
            damping=args.damping,
            osculating_plane=args.osculating_plane,
        )
    elif args.command == "couple-cql3d":
        couple_cql3d(
            run_directory,
        )
    else:
        raise NotImplementedError(args.command)


def main():
    """
    Run command line interface.
    """
    # Pass args directly to enable unit testing.
    cli(sys.argv[1:])
