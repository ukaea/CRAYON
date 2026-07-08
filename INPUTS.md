# Inputs #

A Crayon run directory requires a directory called `input/` containing 3 input files in TOML format named `options.toml`, `rays.toml` and `system_data.toml`.

## system_data.toml ##

Information about the plasma and system the rays are launched into is provided in `system_data.toml`. This file contains 6 sections:

### data_sources ###

`data_sources` contains information about data to be read from files on disk. Crayon can read data from IMAS databases and correctly structed netCDF files. If no data is read from files on disk leave this section blank.

#### IMAS ####

IMAS data sources must have section names starting with `imas`.

- `uri`: A Uniform Resource Identifier (URI) for a IMAS data-entry. For example, a database using HDF5 backend would have form `imas:hdf5?path=/some/imas/database`. See IMAS Access-Layer documentation for more details.

- `occurrence_core_profiles` (optional): An integer specifying which occurrence of the `core_profiles` IDS will be read from. Defaults to $`0`$ if not provided.

- `occurrence_equilibrium` (optional): An integer specifying which occurrence of the `equilibrium` IDS will be read from. Defaults to $`0`$ if not provided.

- `occurrence_wall` (optional): An integer specifying which occurrence of the `wall` IDS will be read from. Defaults to $`0`$ if not provided.

Example of a IMAS data source:

```toml
[data_sources.imas_example]
uri = "example-imas-uri"
occurrence_core_profiles = 0  # Defaults to 0 if not provided.
occurrence_equilibrium = 0  # Defaults to 0 if not provided.
occurrence_wall = 0  # Defaults to 0 if not provided.
```

#### NetCDF ####

NetCDF data sources must have section names starting with `netcdf`.

Example of a netCDF data source:

```toml
[data_sources.netcdf_example]
filepath = "example-netcdf-file.nc"
```

### coordinates ###

`coordinates` is used to define additional coordinate systems used for the input plasma models. If no additional coordinate systems as defined leave this section blank.

#### toroidal ####

Used to define a toroidal coordinate system $`(r, \phi, \theta)`$ derived from a cylindrical coordinate system $`(R, \phi, Z)`$. $`r`$ and $`\theta`$ are respectively the radius and poloidal angle in the $`(R, Z)`$ plane from the axis $`(r_0, z_0)`$.

- `r0`: Cylindrical radius of axis [m].

- `z0`: Cylindrical height axis [m].

Example of a toroidal coordinate definition:

```toml
[coordinates.toroidal]
r0 = 1.0
z0 = 0.0
```

### magnetic_field_t ###

`magnetic_field_t` contains information about the magnetic field.

NOTE: If providing vector components in coordinate systems other than Cartesian these must be given in the holonomic / unnormalised basis!

- `topology`: String indicating

  - `simple`: Magnetic field is provided using vector components.

  - `tokamak`: Magnetic field is axisymmetric and provided using a poloidal flux function $`\psi(r, z)`$ and a diamagnetic function $`F(\psi)`$.

- `source`: String indicating what model is being used.
  - `imas<>`: Set to the name of an IMAS data source. The magnetic field will be constructed from a poloidal flux grid inside the `equilibrium` IDS. Only valid for `topology='tokamak'`

  - `constant`: An analytic constant model. See Analytic Models below. Only valid for `topology='simple'`.

  - `ramp`: An analytic ramp model. See Analytic Models. Only valid for `topology='simple'`.

  - `quadratic_well`: An analytic quadratic well model. See Analytic Models. Only valid for `topology='simple'`.

  - `quadratic_channel`: An analytic quadratic channel model. See Analytic Models. Only valid for `topology='simple'`.

  - `quadratic_bowl`: An analytic bowl model. See Analytic Models. Only valid for `topology='simple'`.

- `scale_factor` (optional): All values will be scaled by this factor. Defaults to $`1.0`$ if not provided.

Example of a constant magnetic field:

```toml
[magnetic_field_t]
topology = 'simple'
source = 'constant'
coordinate_system = 'cartesian'
constant_value = [0.0, 0.0, 0.8]
scale_factor = 1.0
```

Example of a tokamak magnetic field read from IMAS:

```toml
[magnetic_field_t]
topology = 'tokamak'
source = 'imas_example'
scale_factor = 1.0
```

### kinetic ###

`kinetic` contains information about the plasma kinetic profiles.

- `kinetic.electron_density_per_m3` defines the electron density [$`\text{m}^{-3}`$].
- `kinetic.electron_temperature_ev` defines the electron temperature [$`\text{eV}`$].
- `kinetic.effective_charge` defines the effective charge a.k.a $`\text{Z}_{\text{eff}}`$ of all ion species.

Each model is defined using the same options.

- `source`: String indicating what model is being used.
  - `imas<>`: Set to the name of an IMAS data source. The kinetic profile will be constructed from a 1d profile against a flux coordinate from the `core_profiles` IDS.

  - `netcdf<>`: Set to the name of a netCDF data source. The kinetic profile will be constructed from a spline fit to the data from the file.

  - `constant`: An analytic constant model. See Analytic Models below. Only valid for `topology='simple'`.

  - `ramp`: An analytic ramp model. See Analytic Models. Only valid for `topology='simple'`.

  - `quadratic_well`: An analytic quadratic well model. See Analytic Models. Only valid for `topology='simple'`.

  - `quadratic_channel`: An analytic quadratic channel model. See Analytic Models. Only valid for `topology='simple'`.

  - `quadratic_bowl`: An analytic bowl model. See Analytic Models. Only valid for `topology='simple'`.

- `scale_factor`: All values will be scaled by this factor. Defaults to $`1.0`$ if not provided.

Example of a $`C^2`$ ramp in electron density and a constant electron temperature and effective charge.

```toml
[kinetic.electron_density_per_m3]
source = "ramp"
coordinate_system = "cartesian"
origin = [1.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
y0 = 0.0
y1 = 1.2
ramp_width = 0.3
smoothness = 2
scale_factor = 1.0e18

[kinetic.electron_temperature_ev]
source = 'constant'
coordinate_system = 'cartesian'
constant_value = 1.2
scale_factor = 1.0e3

[kinetic.effective_charge]
source = 'constant'
coordinate_system = 'cartesian'
constant_value = 1.0
```

### limiters ###

`limiters` contains information about walls and other solid surfaces the rays must interact with. Each limiter is provided in its own subsection i.e. the section names are `limiters.a`, `limiters.b`, etc.

- `source`: String indicating what model is being used.
  - `analytic`:

  - `imas<>`: Set to the name of an IMAS data source. The limiter will be read from the `wall` IDS.

- `effect`: Effect that will be applied when rays intersect with the limiter element.
  - `effect = 'stop'`: Ray will stop at the limiter element.
  - `effect = 'reflect_specular'`: Ray will reflect specularly off the limiter element.

If `source = 'analytic'`:

- `shape`: String indicating which analytic model is being used.
  - `bounding_box_2d`: A rectangular bounding box on 2 coordinate components.
  - `bounding_box_2d`: A cuboid bounding box on 3 coordinate components.

- `coordinate`: String indiciating which coordinate components the limiter applies to.
  - `xy`: 2D Cartesian limiter in $`(x, y)`$ plane. Only valid for `shape = 'bounding_box_2d'`.

  - `xz`: 2D Cartesian limiter in $`(x, z)`$ plane. Only valid for `shape = 'bounding_box_2d'`.

  - `yz`: 2D Cartesian limiter in $`(y, z)`$ plane. Only valid for `shape = 'bounding_box_2d'`.

  - `yz`: 2D cylindrical limiter in $`(R, Z)`$ plane. Only valid for `shape = 'bounding_box_2d'`.

  - `xyz`: 3D Cartesian limiter in $`(x, y, z)`$. Only valid for `shape = 'bounding_box_3d'`.

- `x_limits`: Length 2 array containing minimum and maximum value of bounding box in first coordinate component. This depends on `coordinate` i.e. it could be $`x`$, $`y`$ or $`r`$ for `coordinate = 'xy'`, `'yz'` or `'rz'` respectively.

- `y_limits`: Length 2 array containing minimum and maximum value of bounding box in second coordinate component.

- `z_limits`: Length 2 array containing minimum and maximum value of bounding box in third coordinate component. Only valid for `shape = 'bounding_box_2d'`.

- `extinction_coefficient_nepers`: Determines how much ray power will be removed on intersection with a limiter element. Given in Nepers.

If `source = 'imas<>`:

- `shape`: String indicating which sub-model is being used.
  - `2d`: A 2D $`(R, Z)`$ limiter will be read from the `wall` IDS.

  - `bounding_box_2d`: A 2D $`(R, Z)`$ bounding box limiter will be created from the limits of the 2D poloidal flux function in `equilibrium` IDS.

Example of a 2D bounding box in $`(R, Z)`$ for $`R \in (0.2, 1.0)`$ and $`Z \in (-0.5, 0.5)`$:

```toml
[limiters.example_rz]
source = 'analytic'
shape = 'bounding_box_2d'
effect = 'stop'
coordinate = 'rz'
x_limits = [0.2, 1.0]
y_limits = [-0.5, 0.5]
extinction_coefficient_nepers = 0.1
```

Example of a 2D limiter in $`(R, Z)`$ read from IMAS:

```toml
[limiters.example_imas_2d]
source = 'imas_example'
shape = '2d'
effect = 'reflect_specular'
extinction_coefficient_nepers = 0.0
```

### Analytic Models ###

Each analytic model must be defined as below.

#### constant ####

A model which returns a constant value for all positions. Components must be given in the holonomic / unnormalised basis. Note for non Cartesian coordinate systems this means the derivative may be non-zero!

- `coordinate_system`: Coordinate system the model is defined in.

- `constant_value`: Constant value of the model. This must match the dimension i.e. if used for electron density it must have size $`1`$ whereas for the magnetic field it must have size $`3`$.

Example fields for a constant model:

```toml
coordinate_system = 'cylindrical'
constant_value = [0.0, 1.0, 0.0]
```

#### ramp ####

A ramp model with $`C^n`$ smoothness.

- `coordinate_system`: Coordinate system the model is defined in.

- `origin`: Point at which the value equals $`y_0`$.

- `direction`: Direction in which the value changes.

- `y0`: Value of the model at the origin.

- `y1`: Value of the model at the far side of the ramp.

- `ramp_width`: Distance over which the value transitions from $`y_0`$ to $`y_1`$.

- `smoothness`: Smoothness class of the ramp $n$ i.e. the value and up to the $`n`$-th derivative are continuous. Can be $`0`$, $`1`$ and $`2`$.

Example fields for a ramp model:

```toml
coordinate_system = 'cartesian'
origin = [1.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
y0 = 0.0
y1 = 1.0
ramp_width = 0.3
smoothness = 1
```

#### Quadratic Well ####

A model whose value is given by a quadratic well i.e. the value increases quadratically will radius from the origin. Cartesian coordinate only.

- `origin`: Point at which the value equals $`y_0`$.

- `y0`: Value of the model at the origin.

- `y1`: Reference value of the model.

- `ramp_width`: Distance over which the value transitions from $`y_0`$ to $`y_1`$.

Example fields for a quadratic well model:

```toml
origin = [1.0, 0.0, 0.0]
y0 = 0.0
y1 = 1.0
ramp_width = 0.5
```

#### Quadratic Channel ####

A model whose value is given by a quadratic channel i.e. the value increases quadratically perpendicular to a given direction. Cartesian coordinate only.

- `origin`: Point at which the value equals $`y_0`$.

- `direction`: Direction in which the value stays constant.

- `y0`: Value of the model at the origin.

- `y1`: Reference value of the model.

- `ramp_width`: Distance over which the value transitions from $`y_0`$ to $`y_1`$.

Example fields for a quadratic channel model:

```toml
origin = [1.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
y0 = 0.0
y1 = 1.0
ramp_width = 0.5
```

#### Quadratic Bowl ####

A model whose value is given by a quadratic bowl i.e. the value increases quadratically parallel to a given direction. Cartesian coordinate only.

- `origin`: Point at which the value equals $`y_0`$.

- `direction`: Direction in which the value changes.

- `y0`: Value of the model at the origin.

- `y1`: Reference value of the model.

- `ramp_width`: Distance over which the value transitions from $`y_0`$ to $`y_1`$.

Example fields for a quadratic channel model:

```toml
origin = [1.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
y0 = 0.0
y1 = 1.0
ramp_width = 0.5
```

## rays.toml ##

Information about the ray initial conditions is provided in `rays.toml`. Each subsection corresponds to a single parent ray. Ray section names must not contain hyphens as these are appended to track child rays.

As a single section can spawn multiple rays an index is appended to the ray name e.g. `ray_1` will generate rays in the output file named `ray_1-0`, `ray_1-1`, etc.

At the top of the file provide

- `angle_format`: Set to `'radians'` if provided angles are in radians. Set to `'degrees` if provided angles are in degrees.

In each ray section provide

- `time_ns`: Time at which the ray will be launched [ns].

- `frequency_ghz`: Frequency of the wave [GHz]. Note this is not angular frequency.

- `position`: Initial position of the ray.

- `coordinate_system_position`: Coordinate system the position is provided in.

- `refractive_index_source`: String indicating how the initial wavevector / refractive index is defined.

  - `components`: Components of refractive index provided. The magnitude of $`N`$ will be set to satisfy dispersion relation at initial position.

  - `launch_angles_geometric`: Launch angles using geometric definition as polar and azimuthal angles of a spherical coordinate system. The magnitude of $`N`$ will be set to satisfy dispersion relation at initial position.

  - `launch_angles_imas`: Launch angles using IMAS definition. The magnitude of $`N`$ will be set to satisfy dispersion relation at initial position.

  - `n_parallel`: Set initial parallel refractive index and angle between $`N_\perp`$ and $`x`$ axis in rotated frame where $`B \parallel z`$. The magnitude of $`N_\perp`$ will be set to satisfy dispersion relation at initial position.

- `polarisation_source`: String indicating how the initial electric field polarisation is provided.

  - `wave_mode`: Polarisation will be set as pure mode.

  - `ellipse_angles`: Polarisation set using orientation and ellipticity angles $`\psi`$ and $`\chi`$ of polarisation ellipse.

- `power_w`: Power in ray channel [W].

- `beam_waist_radius_m`: 1/e electric field beam waist radius of Gaussian beam.

- `n_radial_zones`: Number of radial zones in ray bundle. The power is split into a series of radial zones with rays automatically distributed azimuthally.

#### Initial Refractive Index ####

If `refractive_index_source = 'components'`:

- `refractive_index`: 3 element array giving refractive index components.

- `coordinate_system_refractive_index`: Coordinate system refractive index components are provided in.

- `holonomic`: Set `true` if components are provided in holonomic basis. Set `false` if components are provided in normalised basis. Setting `holonomic = 'false'` only valid for orthogonal coordinate systems like Cartesian, Cylindrical and Toroidal.

If `refractive_index_source = 'launch_angles_geometric'` or `'launch_angles_imas'`:

- `toroidal_angle`: Toroidal launch angle.

- `poloidal_angle`: Poloidal launch angle.

If `refractive_index_source = 'n_parallel'`:

- `n_parallel`: Initial value of parallel refractive index.

- `angle_perp`: Angle between $`N_\perp`$ and $`x`$ axis in rotated frame where $`B \parallel z`$.

#### Initial Polarisation ####

If `polarisation_source = 'wave_mode'`:

- `wave_mode`: Pure wave mode. can be `'O'` or  `'X'`.

If `polarisation_source = 'ellipse_angles'`:

- `orientation_angle`: Orientation angle $`\psi`$ of polarisation ellipse.
- `ellipticity_angle`: Ellipticity angle $`\chi`$ of polarisation ellipse.

Examples:

```toml
angle_format = "degrees"

[ray_1]
time_ns = 0.0
frequency_ghz = 10.0
position = [ 0.1, 0.0, 0.0,]
coordinate_system_position = "cartesian"
refractive_index_source = "components"
refractive_index = [0.0, 0.0, 1.0]
coordinate_system_refractive_index = 'cartesian'
holonomic = true
polarisation_source = "wave_mode"
wave_mode = "O"
power_w = 1.0
beam_waist_radius_m = 0.05
n_radial_zones = 1

[ray_2]
time_ns = 0.0
frequency_ghz = 10.0
position = [ 0.1, 0.0, 0.0,]
coordinate_system_position = "cylindrical"
refractive_index_source = "launch_angles_geometric"
toroidal_angle = 10.0
poloidal_angle = -20.0
polarisation_source = "wave_mode"
wave_mode = "X"
power_w = 1.0
beam_waist_radius_m = 0.05
n_radial_zones = 1

[ray_3]
time_ns = 0.0
frequency_ghz = 10.0
position = [ 0.1, 0.0, 0.0,]
coordinate_system_position = "cartesian"
refractive_index_source = "launch_angles_imas"
toroidal_angle = 10.0
poloidal_angle = -20.0
polarisation_source = "ellipse_angles"
orientation_angle = 0.0
ellipticity_angle = -15.0
power_w = 1.0
beam_waist_radius_m = 0.05
n_radial_zones = 1

[ray_4]
time_ns = 0.0
frequency_ghz = 10.0
position = [ 0.1, 0.0, 0.0,]
coordinate_system_position = "cartesian"
refractive_index_source = "n_parallel"
n_parallel = 0.655
angle_perp = 0.0
polarisation_source = "wave_mode"
wave_mode = "X"
power_w = 1.0
beam_waist_radius_m = 0.05
n_radial_zones = 1
```

### Launch Angles Definitions ###

The toroidal angle $`\alpha`$ and poloidal angle $`\beta`$ define a cylindrical refractive index in the normalised basis such that $`\alpha = \beta = 0`$ corresponds to pointing towards the origin at constant $`z`$. Setting $`\alpha > 0`$ causes the refractive index toroidal component to point counter-clockwise from above.

In the geometric definition we define the initial refractive index vector as a spherical unit vector with polar angle $`\beta`$ and azimuthal angle $`\alpha`$. The normalised refractive components are then

```math
\hat{N}_r = - \cos \beta \cos \alpha \qquad
\hat{N}_\phi = \cos \beta \sin \alpha \qquad
\hat{N}_z = \sin \beta
```

In the IMAS definition the normalised refractive index components are

```math
\hat{N}_r = - \cos \alpha \cos \beta \qquad
\hat{N}_\phi = \sin \alpha \qquad
\hat{N}_z = -\cos \alpha \sin \beta
```

## options.toml ##

Additional options for controlling behaviour of ray tracing.

### rays ###

These options control the behaviour of the ray tracing algorithm/

- `max_ray_nodes`: Maximum number of points in a ray before it is stopped. Includes the start point so the ray tracer will take a maximum of `max_ray_nodes - 1` steps.

- `max_ray_children`: Maximum number of child rays each ray can spawn.

- `max_generations`: Maximum number of generations of rays to trace. The first set of rays are generation 1.

- `max_optical_depth`: Maximum optical depth along a ray before it is stopped [nepers].

- `max_reflections`: Maximum reflections off the wall for a ray before it is stopped. Includes any reflections from parent rays.

- `min_power_fraction_new_ray`: Minimum fraction of power a new ray must have to be spawned. If below this fraction the power will be deducted from the parent ray but no child ray will be created.

### integrator ###

These options control the integration algorithm used to solve the ray trajectories.

- `solver_type`: Type of solver used. Can be
`'EXPLICIT'`,
`'IMPLICIT'` or
`'HYBRID'`.

  - `'EXPLICIT'`: Explicit time stepping.
  - `'IMPLICIT'`: Implicit time stepping.
  - `'HYBRID'`: Switches between explicit and implicit methods.

- `fidelity`: Controls accuracy of integration. Higher fidelity reduces error but increases computation time. Can be
`'LOW'`,
`'MEDIUM'`
or `'HIGH'`.

- `max_timestep`: Maximum timestep in integrator [ns].

- `min_timestep`: Minimum timestep in integrator [ns].

- `timestep_controller`: Algorithm used to set timestep for adaptive solvers based on integration error estimate. Can be
`'NONE'`,
`'I_NUMERICAL_RECIPES'`,
`'PI_42'`,
`'PI_33'`,
`'PI_34'`,
`'PI_H211'`,
`'PID_H312'`.

  - `'NONE'`: No control, all timesteps are accepted regardless of error.
  - `'I_NUMERICAL_RECIPES'`: Integral controller from numerical recipes in Fortran / C / etc.
  - `'PI_42'`: Proportional-integral controller.
  - `'PI_33'`: Proportional-integral controller.
  - `'PI_34'`: Proportional-integral controller.
  - `'PI_H211'`: Proportional-integral controller.
  - `'PID_H312'`: Proportional-integral-derivative controller.

- `norm`: Norm used to calculate error estimate from integration state vector. Can be `'HARRIER'` or `'INFINITY'`.
  - `'HARRIER'`: Use $`L_2`$ norm divided by size of state vector.
  - `'INFINITY'`: Use largest error in state vector.

### integrator.explicit ###

Controls for explicit integration methods.

- `solver`: Explicit solver to use. Can be
`'RK4_RUNGE'`,
`'RK45_DORMAND_PRINCE'`,
`'RK45_CASH_KARP'`,
or `'RK45_TSITSORAS'`.

  - `'RK4_RUNGE'`: Runge-Kutta 4th order with 'classic' coefficients from *M. Kutta, Zeitschrift für Mathematik und Physik, Vol. 46, pp. 435-453, 1901*. Do not use for real problems!
  - `'RK45_DORMAND_PRINCE'`: Runge-Kutta 4(5) with coefficients from *J. R. Dormand, P. J. Prince, Journal of Computational and Applied Mathematics, Vol. 6, No. 1, pp. 19-26, 1980*.
  - `'RK45_CASH_KARP'`: Runge-Kutta 4(5) with coefficients from *J. R. Cash, A. H. Karp, ACM Transactions on Mathematical Software, Vol. 16, No. 3, pp. 201-222 1990.*.
  - `'RK45_TSITSORAS'`: Runge-Kutta 4(5) with coefficients from *Ch. Tsitoras, Computers and Mathematics with Applications, Vol. 62, No. 2, pp. 770-775, 2011*.

- `max_iterations`: Maximum interations to find acceptable time step before ray is stopped.
