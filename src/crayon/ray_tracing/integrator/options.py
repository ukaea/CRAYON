"""
Options for integrator.
"""

# Standard imports
import logging

# Third party imports
import cerberus
import netCDF4 as nc4  # noqa: N813

# Local imports
from crayon.ray_tracing.integrator.explicit import ExplicitIntegratorType
from crayon.ray_tracing.integrator.implicit import ImplicitIntegratorType
from crayon.ray_tracing.integrator.timestep_controller import (
    ErrorEstimateNorm,
    TimestepControllerType,
)
from crayon.shared.data_structures import CrayonEnum
from crayon.shared.io import IONetcdf, IOToml, TomlValidator

logger = logging.getLogger(__name__)


class IntegratorType(CrayonEnum):
    """


    EXPLICIT
        Explicit integration.
    IMPLICIT
        Implicit integration.
    HYBRID
        Dynamically switch between explicit and implicit method.
    """

    EXPLICIT = 1
    IMPLICIT = 2
    HYBRID = 3


class IntegratorFidelity(CrayonEnum):
    """
    Fidelity for integration.

    Attributes
    ----------
    LOW
        Low fidelity, high speed.
    MEDIUM
        Medium fidelity, medium speed.
    HIGH
        High fidelity, low speed.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class IntegratorTomlValidator(TomlValidator):
    """
    TomlValidator for integration options enumerations.
    """

    types_mapping = TomlValidator.types_mapping.copy()

    types_mapping["TimestepControllerType"] = cerberus.TypeDefinition(
        "TimestepControllerType", (TimestepControllerType,), ()
    )
    types_mapping["ErrorEstimateNorm"] = cerberus.TypeDefinition(
        "ErrorEstimateNorm", (ErrorEstimateNorm,), ()
    )
    types_mapping["IntegratorType"] = cerberus.TypeDefinition(
        "IntegratorType", (IntegratorType,), ()
    )
    types_mapping["ExplicitIntegratorType"] = cerberus.TypeDefinition(
        "ExplicitIntegratorType", (ExplicitIntegratorType,), ()
    )
    types_mapping["ImplicitIntegratorType"] = cerberus.TypeDefinition(
        "ImplicitIntegratorType", (ImplicitIntegratorType,), ()
    )
    types_mapping["IntegratorFidelity"] = cerberus.TypeDefinition(
        "IntegratorFidelity", (IntegratorFidelity,), ()
    )


DEFAULT_EXPLICIT_SOLVER: ExplicitIntegratorType = (
    ExplicitIntegratorType.RK45_TSITOURAS
)
MAX_ITERATIONS: int = 32

_explicit_schema = {
    "solver": {
        "type": "ExplicitIntegratorType",
        "required": False,
        "coerce": ExplicitIntegratorType.parse,
        "default": DEFAULT_EXPLICIT_SOLVER,
    },
    "max_iterations": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "default": MAX_ITERATIONS,
        "min": 1,
    },
}


class OptionsIntegratorExplicit(IONetcdf, IOToml):
    """
    Options for explicit integrator

    Parameters
    ----------
    solver
        Solver used for integration.
    max_iterations : int
        Maximum allowed iterations to find an acceptable timestep.
    """

    __slots__ = ("max_iterations", "solver")

    section_name = "explicit"

    def __init__(
        self,
        solver: ExplicitIntegratorType = DEFAULT_EXPLICIT_SOLVER,
        max_iterations: int = MAX_ITERATIONS,
    ):
        """
        Inits OptionsIntegratorExplicit

        Attributes
        ----------
        max_iterations
            Maximum number of iterations to find an acceptable step.
        """
        super().__init__()

        self.solver = solver
        self.max_iterations = max_iterations

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.section_name)

        group.setncattr("solver", self.solver.name)
        group.setncattr("max_iterations", self.max_iterations)

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "OptionsIntegratorExplicit":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        options : OptionsIntegratorExplicit
            Options for explicit integrator.
        """
        group = dset[cls.section_name]

        _solver = ExplicitIntegratorType.parse(group.getncattr("solver"))
        _max_iterations = group.getncattr("max_iterations")

        return cls(
            solver=_solver,
            max_iterations=_max_iterations,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            self.section_name: {
                "solver": self.solver.name,
                "max_iterations": self.max_iterations,
            }
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "OptionsIntegratorExplicit":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        options : OptionsIntegratorExplicit
            Options for explicit integrator.
        """
        validator = IntegratorTomlValidator()
        validator.validate(d, _explicit_schema)

        _d = validator.document

        _solver = _d.pop("solver")
        _max_iterations = _d.pop("max_iterations")

        return cls(
            solver=_solver,
            max_iterations=_max_iterations,
        )


DEFAULT_IMPLICIT_SOLVER: ImplicitIntegratorType = (
    ImplicitIntegratorType.BACKWARDS_EULER
)

_implicit_schema = {
    "solver": {
        "type": "ImplicitIntegratorType",
        "required": False,
        "coerce": ImplicitIntegratorType.parse,
        "default": DEFAULT_IMPLICIT_SOLVER,
    },
    "max_iterations": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "default": MAX_ITERATIONS,
        "min": 1,
    },
}


class OptionsIntegratorImplicit(IONetcdf, IOToml):
    """
    Options for implicit integrator

    Attributes
    ----------
    solver
        Solver used for integration.
    max_iterations : int
        Maximum allowed iterations to find solution.
    """

    __slots__ = ("max_iterations", "solver")

    section_name = "implicit"

    def __init__(
        self,
        solver: ImplicitIntegratorType = DEFAULT_IMPLICIT_SOLVER,
        max_iterations: int = MAX_ITERATIONS,
    ):
        """
        Inits OptionsIntegratorImplicit

        Attributes
        ----------
        max_iterations
            Maximum number of iterations to find an acceptable step.
        """
        super().__init__()

        self.solver = solver
        self.max_iterations = max_iterations

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.section_name)

        group.setncattr("solver", self.solver.name)
        group.setncattr("max_iterations", self.max_iterations)

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "OptionsIntegratorImplicit":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        options : OptionsIntegratorImplicit
            Options for implicit integrator.
        """
        group = dset[cls.section_name]

        _solver = ImplicitIntegratorType.parse(group.getncattr("solver"))
        _max_iterations = group.getncattr("max_iterations")

        return cls(
            solver=_solver,
            max_iterations=_max_iterations,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            self.section_name: {
                "solver": self.solver.name,
                "max_iterations": self.max_iterations,
            }
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "OptionsIntegratorImplicit":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        options : OptionsIntegratorImplicit
            Options for implicit integrator.
        """
        validator = IntegratorTomlValidator()
        validator.validate(d, _implicit_schema)

        _d = validator.document

        _solver = ImplicitIntegratorType.parse(_d.pop("solver"))
        _max_iterations = _d.pop("max_iterations")

        return cls(
            solver=_solver,
            max_iterations=_max_iterations,
        )


STEP_FAILURE_THRESHOLD: float = 0.25
STEP_FAILURE_WINDOW: int = 20

_hybrid_schema = {
    "step_failure_threshold": {
        "type": "float",
        "required": False,
        "coerce": float,
        "default": STEP_FAILURE_THRESHOLD,
        "min": 0.0,
        "max": 1.0,
    },
    "step_failure_window": {
        "type": "integer",
        "required": False,
        "coerce": int,
        "default": STEP_FAILURE_WINDOW,
        "min": 1,
    },
}


class OptionsIntegratorHybrid(IONetcdf, IOToml):
    """
    Options for hybrid integrator.

    Attributes
    ----------
    step_failure_threshold : float
        If fraction of previous steps which have failed exceeds this threshold,
        the hybrid solver will check the stiffness of the problem.
    step_failure_window : int
        Window of steps the failure check is performed over.
    """

    __slots__ = ("step_failure_threshold", "step_failure_window")

    section_name = "hybrid"

    def __init__(
        self,
        step_failure_threshold: float = STEP_FAILURE_THRESHOLD,
        step_failure_window: int = STEP_FAILURE_WINDOW,
    ):
        """
        Inits OptionsIntegratorHybrid.

        Attributes
        ----------
        step_failure_threshold : float
            Fraction of previous steps which have failed above which the hybrid
            solver will check for stiffness problems if the step fails.
        step_failure_window : int
            Number of steps the step failure fraction is calculated over.
        """
        self.step_failure_threshold = step_failure_threshold
        self.step_failure_window = step_failure_window

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.section_name)

        group.setncattr("step_failure_threshold", self.step_failure_threshold)
        group.setncattr("step_failure_window", self.step_failure_window)

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "OptionsIntegratorHybrid":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        options : OptionsIntegratorHybrid
            Options for hybrid integrator.
        """
        group = dset[cls.section_name]

        _step_failure_threshold = group.getncattr("step_failure_threshold")
        _step_failure_window = group.getncattr("step_failure_window")

        return cls(
            step_failure_threshold=_step_failure_threshold,
            step_failure_window=_step_failure_window,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            self.section_name: {
                "step_failure_threshold": self.step_failure_threshold,
                "step_failure_window": self.step_failure_window,
            }
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "OptionsIntegratorHybrid":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        options : OptionsIntegratorHybrid
            Options for hybrid integrator.
        """
        validator = IntegratorTomlValidator()
        validator.validate(d, _hybrid_schema)

        _d = validator.document

        _step_failure_threshold = _d.pop("step_failure_threshold")
        _step_failure_window = _d.pop("step_failure_window")

        return cls(
            step_failure_threshold=_step_failure_threshold,
            step_failure_window=_step_failure_window,
        )


DEFAULT_SOLVER: IntegratorType = IntegratorType.EXPLICIT
DEFAULT_TIMESTEP_CTRL: TimestepControllerType = TimestepControllerType.PI_42
DEFAULT_NORM: ErrorEstimateNorm = ErrorEstimateNorm.HARRIER
DEFAULT_FIDELITY: IntegratorFidelity = IntegratorFidelity.MEDIUM
MAX_TIMESTEP: float = 1000.0  # ns => ~300m for ray in vacuum.

ATOL_LOW: float = 1.0e-2
ATOL_MEDIUM: float = 1.0e-4
ATOL_HIGH: float = 1.0e-6

RTOL_LOW: float = 1.0e-4
RTOL_MEDIUM: float = 1.0e-6
RTOL_HIGH: float = 1.0e-8

MIN_TIMESTEP_LOW: float = 1.0e-6
MIN_TIMESTEP_MEDIUM: float = 1.0e-8
MIN_TIMESTEP_HIGH: float = 1.0e-10

_schema = {
    "solver_type": {
        "type": "IntegratorType",
        "required": False,
        "coerce": IntegratorType.parse,
        "default": DEFAULT_SOLVER,
    },
    "fidelity": {
        "type": "IntegratorFidelity",
        "required": False,
        "coerce": IntegratorFidelity.parse,
        "default": DEFAULT_FIDELITY,
    },
    "max_timestep": {
        "type": "float",
        "required": False,
        "coerce": float,
        "default": MAX_TIMESTEP,
    },
    "initial_timestep": {
        "type": "float",
        "required": False,
        "coerce": float,
        "default": 0.0,
    },
    "timestep_controller": {
        "type": "TimestepControllerType",
        "required": False,
        "coerce": TimestepControllerType.parse,
        "default": DEFAULT_TIMESTEP_CTRL,
    },
    "norm": {
        "type": "ErrorEstimateNorm",
        "required": False,
        "coerce": ErrorEstimateNorm.parse,
        "default": DEFAULT_NORM,
    },
    "explicit": {
        "type": "dict",
        "required": False,
    },
    "implicit": {
        "type": "dict",
        "required": False,
    },
    "hybrid": {
        "type": "dict",
        "required": False,
    },
}


class OptionsIntegrator(IONetcdf, IOToml):
    """
    Options for integrator.

    Attributes
    ----------
    solver_type
        Type of ODE solver used to integrate rays.
    fidelity
        Fidelity of ODE solution. The higher the fidelity the more accurate
        the solution but the slower the computation.
    max_timestep : float
        Maximum allowed timestep [ns].
    initial_timestep : float
        If > 0.0, the initial timestep of the ODE integration. Otherwise a
        default timestep will be computed automatically.
    timestep_controller
        Algorithm used for dynamic timestep control.
    norm
        Norm used for error calculations.
    """

    __slots__ = (
        "absolute_tolerance",
        "explicit",
        "fidelity",
        "hybrid",
        "implicit",
        "initial_timestep",
        "max_timestep",
        "min_timestep",
        "norm",
        "relative_tolerance",
        "solver_type",
        "timestep_controller",
    )

    section_name = "integrator"

    def __init__(
        self,
        solver_type: IntegratorType = DEFAULT_SOLVER,
        fidelity: IntegratorFidelity = DEFAULT_FIDELITY,
        max_timestep: float = MAX_TIMESTEP,
        initial_timestep: float = 0.0,
        timestep_controller: TimestepControllerType = DEFAULT_TIMESTEP_CTRL,
        norm: ErrorEstimateNorm = DEFAULT_NORM,
        explicit: OptionsIntegratorExplicit = None,
        implicit: OptionsIntegratorImplicit = None,
        hybrid: OptionsIntegratorHybrid = None,
    ):
        """
        Inits OptionsIntegrator.

        Attributes
        ----------
        solver_type : IntegratorType
            Type of solver used for integration.
        fidelity : IntegratorFidelity
            Fidelity of ODE solution.
        max_timestep : float
            Maximum step the ODE solver can take. By default the step is
            unbounded.
        initial_timestep : float
            Initial timestep used in integrator.
        timestep_controller : TimestepControllerType
            Algorithm used to adaptively set timestep based on error estimate.
        norm : ErrorEstimateNorm
            Norm used for error estimate of state vector.
        explicit
            Options for the explicit ODE solver.
        implicit
            Options for the implicit ODE solver.
        hybrid
            Options for the hybrid ODE solver.
        """
        super().__init__()

        self.solver_type = solver_type
        self.fidelity = IntegratorFidelity.parse(fidelity)
        self.max_timestep = max_timestep
        self.initial_timestep = initial_timestep
        self.timestep_controller = timestep_controller
        self.norm = norm

        if explicit is None:
            self.explicit = OptionsIntegratorExplicit()
        else:
            self.explicit = explicit

        if implicit is None:
            self.implicit = OptionsIntegratorImplicit()
        else:
            self.implicit = implicit

        if hybrid is None:
            self.hybrid = OptionsIntegratorHybrid()
        else:
            self.hybrid = hybrid

        # Unpack fidelity.
        if self.fidelity == IntegratorFidelity.LOW:
            self.absolute_tolerance = ATOL_LOW
            self.relative_tolerance = RTOL_LOW
            self.min_timestep = MIN_TIMESTEP_LOW
        elif self.fidelity == IntegratorFidelity.MEDIUM:
            self.absolute_tolerance = ATOL_MEDIUM
            self.relative_tolerance = RTOL_MEDIUM
            self.min_timestep = MIN_TIMESTEP_MEDIUM
        elif self.fidelity == IntegratorFidelity.HIGH:
            self.absolute_tolerance = ATOL_HIGH
            self.relative_tolerance = RTOL_HIGH
            self.min_timestep = MIN_TIMESTEP_HIGH
        else:
            raise NotImplementedError(self.fidelity)

    def write_netcdf(self, group: nc4.Group):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        group : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group.setncattr("solver_type", self.solver_type.name)
        group.setncattr("fidelity", self.fidelity.name)
        group.setncattr("max_timestep", self.max_timestep)
        group.setncattr("initial_timestep", self.initial_timestep)
        group.setncattr("timestep_controller", self.timestep_controller.name)

        group.setncattr("norm", self.norm.name)

        self.explicit.write_netcdf(group)
        self.implicit.write_netcdf(group)
        self.hybrid.write_netcdf(group)

    @classmethod
    def read_netcdf(cls, group: nc4.Group) -> "OptionsIntegrator":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        options : OptionsIntegrator
            Options for integrator.
        """
        _solver_type = IntegratorType.parse(group.getncattr("solver_type"))
        _fidelity = IntegratorFidelity.parse(group.getncattr("fidelity"))
        _max_timestep = group.getncattr("max_timestep")
        _initial_timestep = group.getncattr("initial_timestep")
        _timestep_controller = TimestepControllerType.parse(
            group.getncattr("timestep_controller")
        )
        _norm = ErrorEstimateNorm.parse(group.getncattr("norm"))

        _explicit = OptionsIntegratorExplicit.read_netcdf(group)
        _implicit = OptionsIntegratorImplicit.read_netcdf(group)
        _hybrid = OptionsIntegratorHybrid.read_netcdf(group)

        return cls(
            solver_type=_solver_type,
            fidelity=_fidelity,
            max_timestep=_max_timestep,
            initial_timestep=_initial_timestep,
            timestep_controller=_timestep_controller,
            norm=_norm,
            explicit=_explicit,
            implicit=_implicit,
            hybrid=_hybrid,
        )

    def to_dict_toml(self) -> dict:
        """
        Write object data into dictionary for serialisation to toml.

        Returns
        -------
        dict_toml : dict
            Dictionary of object data that can be serialised to toml.
        """
        return {
            "solver_type": self.solver_type.name,
            "fidelity": self.fidelity.name,
            "max_timestep": self.max_timestep,
            "initial_timestep": self.initial_timestep,
            "timestep_controller": self.timestep_controller.name,
            "norm": self.norm.name,
            **self.explicit.to_dict_toml(),
            **self.implicit.to_dict_toml(),
            **self.hybrid.to_dict_toml(),
        }

    @classmethod
    def from_dict_toml(cls, d: dict) -> "OptionsIntegrator":
        """
        Create object from dictionary of data read from toml file.

        Parameters
        ----------
        d : dict
            Dictionary of data read from toml file.

        Returns
        -------
        options : OptionsIntegrator
            Options for integrator.
        """
        validator = IntegratorTomlValidator()
        validator.validate(d, _schema)

        _d = validator.document

        _solver_type = _d.pop("solver_type")
        _fidelity = _d.pop("fidelity")
        _max_timestep = _d.pop("max_timestep")
        _initial_timestep = _d.pop("initial_timestep")
        _timestep_controller = _d.pop("timestep_controller")
        _norm = _d.pop("norm")

        _explicit = OptionsIntegratorExplicit.from_dict_toml(
            _d.pop("explicit", None)
        )
        _implicit = OptionsIntegratorImplicit.from_dict_toml(
            _d.pop("implicit", None)
        )
        _hybrid = OptionsIntegratorHybrid.from_dict_toml(
            _d.pop("hybrid", None)
        )

        return cls(
            solver_type=_solver_type,
            fidelity=_fidelity,
            max_timestep=_max_timestep,
            initial_timestep=_initial_timestep,
            timestep_controller=_timestep_controller,
            norm=_norm,
            explicit=_explicit,
            implicit=_implicit,
            hybrid=_hybrid,
        )
