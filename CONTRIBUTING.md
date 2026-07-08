# Contributing #

For development it can be more convenient to install in editable mode. Also install the development dependencies and the git pre-commit hook.

```
pip install -e .'[dev]'
pre-commit install -f
```

Run the test suite using ```pytest```

```
python3 -m pytest
```

```pytest``` also supports running individual test files and selecting tests by name or using a regular expression. See the
[pytest documentation](https://docs.pytest.org/en/latest/usage.html).

After a test run a test coverage report can be found in the `htmlcov` directory.

## Contributing to Crayon ##

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

## Code of Conduct ##

This project and everyone participating in it is governed by the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [thomas.wilson@ukaea.uk](mailto:thomas.wilson@ukaea.uk) and/or [simon.freethy@ukaea.uk](mailto:simon.freethy@ukaea.uk).

## Submitting an issue

In order to help us address your issue effectively, we ask that you follow this
procedure when creating an issue:

* Check that the issue is not already described elsewhere in [Issues]()
* Write a fairly complete description of the bug/feature/problem in the issue.

## Submitting a bug report

``Crayon`` is software in development and is therefore likely to contain bugs. If you
discover bugs, please follow this procedure:

* Raise an issue including a way to reproduce the bug in the issue, let us know the expected result and what actually happens

## Submitting a pull request

Please discuss any feature ideas you have with the developers before submitting them, as you may not be aware of parallel development work taking place, or implementation decisions / subtleties which are relevant. The ideal workflow for submitting a pull request is as follows:

* Discuss the feature with the developers first
* Submit an issue documenting the intent of the feature you wish to develop
* Fork our repository (see the [GitHub documentation on forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks))
* Make a branch of the `develop` branch which bears a similar name as your issue (e.g.
  `new_feature`)
* Develop your feature(s) in your `new_feature` branch
* Discuss any problems that arise in the associated issue, perhaps documenting your
  progress
* Finally, as the author of the `new_feature` branch, you must submit a pull request
  onto the `develop` branch
  * Link the relevant issue(s) and project(s)
  * Add an assignee, who will be responsible for reviewing and merging the PR. This
    should be the person you feel has the most relevant technical background and/or the most familiar with the underlying issue or code.
  * The reviewers will be automatically selected based on which module(s) of the code your pull request affects.

The merge request will be reviewed by the core development team before potentially being accepted.

## Dependency Management ##

`Crayon` uses [uv](https://docs.astral.sh/uv/) for dependency management. To maintain stability, all collaborators must adhere to the following rules:

Do not edit the [project.dependencies] section of `pyproject.toml` or the `uv.lock` file manually. Use the `uv` command line tool for all changes.

To add a dependency:
```
uv add <package-name>
uv add --group dev <package-name>  # Development packages.
```

To remove a dependency:
```
uv remove <package-name>
```

To upgrade a dependency:
```
uv lock --upgrade-package <package-name>
```

Use `uv run` to run arbitrary scripts or commands in the project environment. Prior to every `uv run` invocation, uv will verify that the lockfile is up-to-date with the pyproject.toml, and that the environment is up-to-date with the lockfile, keeping your project in-sync without the need for manual intervention.

## Python style guide

``Crayon`` is an object-oriented code and is designed to run relatively fast. We use
objects to represent physical entities, as well as for certain solvers where it is
useful to persist the state of some stored calculations in order to save run-time.
When writing objects, make sure all attributes are initialised in the ``__init__`` method, and that all methods and variables that are not user-facing are made protected with a leading underscore: ``_protected_method``.

We try to keep functions as single-purposes as possible, with as few arguments and
keyword arguments as possible. Similarly, with classes, we try to keep the number of
attributes and methods relatively low.

Python can run relatively slowly when compared with compiled languages,
and in key areas we use a couple of tricks to speed things up:
* Using ``__slots__`` on classes to save memory

Please try and follow these relatively loose guidelines when developing ``Crayon``.
We also recommend you use an integrated development environment with appropriate code
linting to improve the code you contribute.

``Crayon`` is strictly auto-formatted using the [ruff](https://docs.astral.sh/ruff/) format module which is very similar to [black](https://pypi.org/project/black/). In both cases this is an opinionated subset of the [Python PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).

If you don't like how ``ruff`` formats your code, remember the loss of
aesthetics is the price to pay for uniformity and consistency! The point of using it is that all code should more or less look the same, regardless of who writes it.

Please read up on this if you need more details. If ``ruff`` breaks your one-line or chain calls into multiple lines, consider breaking these down for readability.

Crayon is also checked for quality using the [ruff](https://beta.ruff.rs/docs/) linter which implements [flake8](https://flake8.pycqa.org/en/latest/) rules and various extensions to it. This is more a question of code style, which just the formatter doesn't cover in full.

Code that is committed to a branch is automatically checked for quality using pre-commit. Violations detected by our code quality checks are printed to the console for information. To avoid checks for a single commit you can use the `--no-verify` flag when committing. The checks do not prevent you from pushing code, but the continuous integration checks will fail.

Please address issues raised by the checks prior to pushing your code.

When writing code for ``Crayon``, please adhere to the following Python naming
conventions:

* `ALL_CAPITALS_SNAKE_CASE` for global constants

* `CapitalisedWords` for classes

* `lower_case_snake_case` for functions and methods

* `lower_case_snake_case` for arguments and variables

We try to stick to descriptive `lower_case_snake_case` argument and local variable names with the general rule that names less than three characters should not be used.
This is not a hard rule, and there are some notable exceptions:
* `i`, `j`, `k` for integers (e.g. looping, indexing)
* `m`, `n` for integers (e.g. array sizes)
* `x`, `y`, `z`, `r` for floats or arrays describing coordinates
* Where it makes sense to reflect mathematical notation (e.g. `R_0`)
* Please include physical constants on the end of physical variables e.g. `electron_density_per_m3` rather than `ne`.

### Import style

* All imports should be absolute imports, eg:

    ```python
    from bluemira.base.components import Component
    ```

    not this

    ```python
    from ..base.components import Component
    ```
    We enforce this with the `flake8` rules in `ruff`.
* Imports between bluemira modules should access individual methods directly.
* Wildcard imports should not be used as it pollutes the namespace and makes it hard to work out where a method originates.
* Some external modules such as `numpy` and `matplotlib.pyplot` have specific import styles widely used elsewhere, please look for examples in `Crayon` if unsure:
   ```python
   import netCDF4 as nc4
   import numpy as np
   import matplotlib.pyplot as plt
    ```
    If only a few methods are used from these types of modules directly importing the method is preferred.
* Formatting of imports is automatically organised by the rules provided by `isort` and `ruff`. `isort` organises imports into three sections; builtin modules, external modules and internal modules.

All of the above means that this is bad:

```python

import crayon
from ..geometry import tools
import os
from scipy import *
```
and this is good:

```python
from os import chdir

from scipy.interpolate import interp1d
import numpy as np

import crayon
```

## Documentation style guide

Please write all Python docstrings in `numpydoc` style, see details [here](
https://numpydoc.readthedocs.io/en/latest/format.html). Typing should be specified
with type-hints. An example is shown below:

```python
def coulomb_logarithm(
    density_per_m3: float, temperature_ev: float, effective_charge: float
) -> float:
    """
    Calculate Coulomb logarithm for thermal electron-ion collisions. Formulas
    taken from [1].

    Parameters
    ----------
    density_per_m3 : float
        Electron density [m^-3].
    temperature_ev : float
        Electron temperature [eV].
    effective_charge : float
        Effective charge.

    Returns
    -------
    coulomb_logarithm : float
        Coulomb logarithm.

    Notes
    -----
    We use density in m^-3 rather than cm^-3 so there is the extra
    factor ln(1000) compared to [1].
    ln(sqrt(n_e[cm^-3])) = ln(sqrt(10**-6 * n_e[m^-3]))
    = ln(10**-3 * sqrt(n_e[m^-3])) = -ln(10**3) + ln(sqrt(n_e[m^-3]))

    References
    ----------
    [1] https://farside.ph.utexas.edu/teaching/plasma/Plasma/node39.html#sclog
    """
```

## Testing

[Pytest](https://docs.pytest.org) is used for all testing of `Crayon`. Tests are grouped into classes for testing multiple pieces of functionality of a single class or function.

> [!WARNING]
> Do not commit private data to the repository there is no simple way to remove that
data once pushed. You will have to contact github support to fully remove the data
as noted [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository#fully-removing-the-data-from-github).

## Releases and Packaging

Release versions of ``Crayon`` are generated using ```uv version```. The most recent
tag is dynamically pulled into the ``Crayon`` itself to set `__version__` correctly. The ``main`` branch will always contain the newest release.
