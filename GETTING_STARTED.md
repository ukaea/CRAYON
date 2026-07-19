## Getting Started ##

Firstly follow the [installation guide](docs/installation.md).

Once installed Crayon provides a single command line script `crayon` to run the code which has several subcommands.

- `crayon <dir> new`: Create an empty run directory.
- `crayon <dir> optimise`: Optimise launch conditions for OX conversion.
- `crayon <dir> trace`: Run ray tracing.
- `crayon <dir> plot_single`: Plot output for a single ray.
- `crayon <dir> plot`: Plot output for all rays.

Use `crayon -h` to see details all available subcommands.

## Quick Tutorial ##

Create and cd to a new directory.

Run `crayon . new`. This generates an empty run directory and writes 3 template input files `options.toml`, `rays.toml` and `system_data.toml` in `./input/` which contain information for a plasma slab O-X-B conversion case. For more details on inputs, see [inputs](INPUTS.md).

Run `crayon . trace 0.0` to start the ray tracing at time slice $0.0$ seconds. This will generate an output file in `./output/0.0s/rays.nc` containing the ray data.

To plot the ray data run `crayon . plot_single ray_1-0 0.0 --plasma --xy`
