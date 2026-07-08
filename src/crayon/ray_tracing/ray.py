"""
Classes for defining a position on a ray trajectory.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.coordinates import CoordinateSystem
from crayon.dispersion import (
    DispersionType,
    polarisation_phase_convention_factor,
    vacuum_stix_polarisation,
)
from crayon.ray_tracing.caches import (
    CoordinateCache,
    HamiltonianCache,
    PlasmaCache,
    State,
    StateDt,
    WaveCache,
)
from crayon.ray_tracing.initial_conditions import InitialConditions
from crayon.shared.constants import WaveMode
from crayon.shared.dimensions import Dimensions
from crayon.shared.types import ComplexArray, FloatArray
from crayon.system_data import SystemData

logger = logging.getLogger(__name__)


class Ray:
    """
    Ray object representing position on ray trajectory.

    Attributes
    ----------
    beam_waist_radius_m
        1 / e electric field waist radius of beam.
    bundle : bool
        Flag if the ray is part of a bundle of rays.
    conversions : int
        Number of mode conversions.
    coordinate_cache : CoordinateCache
        Cache containing coordinate system data.
    damping_model : DispersionType
        Dispersion model used for damping.
    dy_dt_evaluations : int
        Number of dy/dt evaluations along ray.
    force_damping_model : DispersionType
        If set, forces dispersion model used for damping.
    force_propagation_model : DispersionType
        If set, forces dispersion model used for propagation.
    hamiltonian_cache : HamiltonianCache
        Cache containing ray hamiltonian data.
    index : int
        Index of current node on ray trajectory.
    name : str
        Name of ray.
    output : RayTracingOutput
        Output data from ray tracing.
    plasma_cache : PlasmaCache
        Cache containing plasma parameter data.
    propagation_model : DispersionType
        Dispersion model used for propagation.
    state : State
        Ray tracing integrator state vector.
    state_dt : StateDt
        Ray tracing integrator state vector time derivative.
    step_accepted : bool
        Flag if current step has been accepted.
    stop_condition : str
        Message containing details why ray tracing stopped.
    wave_cache : WaveCache
        Cache containing wave parameter data.
    wave_mode : WaveMode
        Mode ray represents.

    Methods
    -------
    set_time
        Set time [ns].
    set_frequency
        Set wave frequency [GHz].
    set_xk_position
        Set Cartesian position x [m] and wavevector k [m^-1].
    set_xn_position
        Set Cartesian position x[m] and refractive index vector N.
    find_root_n
        Rescale magnitude of refractive index to satisfy dispersion relation.
    set_polarisation_stix
        Set Stix polarisation (vacuum only).
    set_polarisation_cartesian
        Set Cartesian polarisation (vacuum only).
    vacuum_power_fraction_in_ox
        Return fraction of wave power contained in O and X mode components
        (vacuum only).
    force_dispersion_models
        Force dispersion models for propagation and/or damping.
    set_dispersion_models
        Set dispersion models for propagation and/or damping.
    calculate_hamiltonian
        Calculate hamiltonian data at current position.
    calculate_determinant
        Calculate determinant of dispersion tensor.
    calculate_state_vector_dt
        Calculate time derivative of integrator state vector.
    set_stop_condition
        Set stop condition message for ray tracing.
    accept_step
        Accept step to current state.
    """

    __slots__ = (
        "beam_waist_radius_m",
        "bundle",
        "conversions",
        "coordinate_cache",
        "damping_model",
        "dy_dt_evaluations",
        "force_damping_model",
        "force_propagation_model",
        "hamiltonian_cache",
        "index",
        "name",
        "output",
        "plasma_cache",
        "propagation_model",
        "state",
        "state_dt",
        "step_accepted",
        "stop_condition",
        "wave_cache",
        "wave_mode",
    )

    def __init__(
        self,
        system_data: SystemData,
        initial_conditions: InitialConditions,
    ):
        """
        Inits Ray.

        Parameters
        ----------
        system_data : SystemData
            Object containing information about plasma system.
        initial_conditions : InitialConditions
            Initial conditions of ray.
        """
        # Initialise caches.
        self.coordinate_cache = CoordinateCache(
            system_data.coordinate_coordinator
        )
        self.plasma_cache = PlasmaCache(
            system_data.kinetic, system_data.magnetic
        )
        self.wave_cache = WaveCache()
        self.hamiltonian_cache = HamiltonianCache()

        # State vector for time integration.
        self.state = State()
        self.state_dt = StateDt()

        # Hamiltonian models.
        self.propagation_model = DispersionType.COLD
        self.damping_model = DispersionType.COLD
        self.force_propagation_model = None
        self.force_damping_model = None

        # Ray name.
        self.name = initial_conditions.name

        # Mode this ray represents.
        self.wave_mode = initial_conditions.wave_mode

        self.beam_waist_radius_m = initial_conditions.beam_waist_radius_m
        self.bundle = initial_conditions.bundle

        # Index of node of ray trajectory.
        self.index: int = 0

        # Number of mode conversions.
        self.conversions: int = 0

        # Counter for number of evaluations of state vector derivative.
        # A crude measure of performance of integrator.
        self.dy_dt_evaluations: int = 0

        # Message describing ray stop condition.
        self.stop_condition: str = ""

        # Flag if we accepted the step to the new position.
        self.step_accepted: bool = True

        # Set initial conditions.
        if initial_conditions:
            self._set_initial_conditions(initial_conditions)

    def _set_initial_conditions(self, initial_conditions: InitialConditions):
        """
        Set initial conditions on ray.

        Parameters
        ----------
        initial_conditions : InitialConditions
            Ray initial conditions.

        Raises
        ------
        ValueError
            Stix polarisation is provided but zero.
        """
        # Set initial phase space position.
        self.set_time(initial_conditions.time_ns)
        self.set_frequency(initial_conditions.frequency_ghz)
        self.set_xn_position(
            initial_conditions.position_cartesian,
            initial_conditions.refractive_index_cartesian,
        )

        # Set initial phase and power.
        self.state.set_auxilliaries(
            initial_conditions.eikonal_phase_rad, initial_conditions.power_w
        )

        # Calculate plasma parameters.
        self.plasma_cache.calculate(self.coordinate_cache, derivatives=0)

        # Calculate wave parameters.
        self.wave_cache.calculate_k_components(
            self.plasma_cache.magnetic_field_unit, derivatives=0
        )

        # Initial polarisation.
        if (
            self.plasma_cache.vacuum
            and initial_conditions.wave_mode != WaveMode.ANY
        ):
            # Calculate vacuum polarisation for eigenmode.
            polarisation_stix = vacuum_stix_polarisation(
                self.plasma_cache.normalised_magnetic_field_strength.value.item(),
                self.wave_cache.n_perp,
                self.wave_cache.n_parallel,
                self.wave_mode,
            )

            self.set_polarisation_stix(polarisation_stix)
        else:
            # The polarisation gets calculated again in calculate_hamiltonian
            # as not in vacuum. However, this acts as a 'nearby value' to help
            # select correct polarisation if modes are nearly degenerate.
            self.hamiltonian_cache.polarisation.stix.value[:] = (
                initial_conditions.polarisation_stix
            )

            # Indicates polarisation not being set correctly.
            if np.allclose(initial_conditions.polarisation_stix, 0.0):
                raise ValueError("polarisation_stix is zero.")

        # Root find to satisfy dispersion relation.
        self.calculate_hamiltonian(derivatives=0, determinant=False)
        self.find_root_n(kinetic=False)

    @property
    def time_ns(self) -> float:
        """Current time on ray."""
        return self.state.time_ns

    @property
    def state_vector(self) -> FloatArray:
        """Ray integrator state vector."""
        return self.state.state_vector

    def _synchronise_xk(self, /, *, to_state: bool):
        """
        Synchronise (x, k) values held on state and the position and wave
        caches. If to_state=True, the values in the caches are copied to state
        and vice versa.
        """
        if to_state:
            self.state.position_cartesian[:] = (
                self.coordinate_cache.position_cartesian
            )
            self.state.wavevector_cartesian[:] = (
                self.wave_cache.wavevector_per_m
            )
        else:
            self.coordinate_cache.set_position(
                CoordinateSystem.CARTESIAN, self.state.position_cartesian
            )
            self.wave_cache.set_wavevector(self.state.wavevector_cartesian)

    def set_time(self, time_ns: float):
        """
        Set time [ns].

        Parameters
        ----------
        time_ns : float
            Time [ns].
        """
        self.state.time_ns = time_ns

    def set_frequency(self, frequency_ghz: float):
        """
        Set frequency [GHz].

        Parameters
        ----------
        frequency_ghz : float
            Frequency [GHz].
        """
        self.plasma_cache.set_frequency(frequency_ghz)
        self.wave_cache.set_frequency(frequency_ghz)
        self.state.frequency_ghz = frequency_ghz

    def set_xk_position(
        self,
        position_cartesian: FloatArray,
        wavevector_cartesian: FloatArray,
    ):
        """
        Set Cartesian position x [m] and wavevector k [m^-1].

        Parameters
        ----------
        position_cartesian : np.array[float]
            Cartesian position x [m].
        wavevector_cartesian : np.array[float]
            Cartesian wavevector k [m^-1].
        """
        self.coordinate_cache.set_position(
            CoordinateSystem.CARTESIAN, position_cartesian
        )
        self.wave_cache.set_wavevector(wavevector_cartesian)

        # Synchronise caches and state vector.
        self._synchronise_xk(to_state=True)

    def set_xn_position(
        self,
        position_cartesian: FloatArray,
        refractive_index_cartesian: FloatArray,
    ):
        """
        Set Cartesian position x[m] and refractive index vector N.

        Parameters
        ----------
        position_cartesian : np.array[float]
            Cartesian position x [m].
        refractive_index_cartesian : np.array[float]
            Cartesian refractive index vector N.
        """
        self.coordinate_cache.set_position(
            CoordinateSystem.CARTESIAN, position_cartesian
        )
        self.wave_cache.set_refractive_index(refractive_index_cartesian)

        # Synchronise caches and state vector.
        self._synchronise_xk(to_state=True)

    def find_root_n(self, /, *, kinetic: bool):
        """
        Rescale magnitude of refractive index to satisfy dispersion relation.
        """
        if self.plasma_cache.vacuum:
            n = 1.0
        else:
            result = self.hamiltonian_cache.find_root_n(
                self.propagation_model, self.wave_mode, kinetic=kinetic
            )

            if result.message:
                self.set_stop_condition(result.message)
                return

            n = result.value

        # Re-normalise refractive index.
        logger.info("Renormalise refractive index to %s", n)
        _norm = np.linalg.norm(self.wave_cache.refractive_index)
        self.state.wavevector_cartesian *= n / _norm

        # Synchronise state vector with caches.
        self._synchronise_xk(to_state=False)

    def set_polarisation_stix(self, polarisation_stix: ComplexArray):
        """
        Set Stix polarisation. Only use in vacuum.

        Parameters
        ----------
        polarisation_stix : np.array[complex]
            Stix frame polarisation.

        Raises
        ------
        ValueError
            Not in vacuum.
        """
        if not self.plasma_cache.vacuum:
            raise ValueError("Must be in vacuum.")

        # Apply phase convention.
        polarisation_stix *= polarisation_phase_convention_factor(
            polarisation_stix
        )

        self.hamiltonian_cache.polarisation.stix.value[:] = polarisation_stix
        self.hamiltonian_cache.calculate_cartesian_polarisation_from_stix(
            self.wave_cache
        )

        # Polarisation does not evolve in vacuum.
        self.hamiltonian_cache.polarisation.stix.first_derivative.q.fill(0.0)
        self.hamiltonian_cache.polarisation.stix.first_derivative.z.fill(0.0)

    def set_polarisation_cartesian(self, polarisation_cartesian: ComplexArray):
        """
        Set Cartesian polarisation. Only use in vacuum.

        Parameters
        ----------
        polarisation_cartesian : np.array[complex]
            Cartesian polarisation.

        Raises
        ------
        ValueError
            Not in vacuum.
        """
        if not self.plasma_cache.vacuum:
            raise ValueError("Must be in vacuum.")

        self.hamiltonian_cache.polarisation.cartesian[:] = (
            polarisation_cartesian
        )
        self.hamiltonian_cache.calculate_stix_polarisation_from_cartesian(
            self.wave_cache
        )

        # Apply phase convention.
        factor = polarisation_phase_convention_factor(
            self.hamiltonian_cache.polarisation.stix.value
        )

        self.hamiltonian_cache.polarisation.cartesian *= factor
        self.hamiltonian_cache.polarisation.stix.value *= factor

        # Polarisation does not evolve in vacuum.
        self.hamiltonian_cache.polarisation.stix.first_derivative.q.fill(0.0)
        self.hamiltonian_cache.polarisation.stix.first_derivative.z.fill(0.0)

    def vacuum_power_fraction_in_ox(self) -> tuple[float, float]:
        """
        Return fraction of wave power contained in O and X mode components
        (vacuum only).

        Returns
        -------
        o_mode_fraction : float
            Fraction of ray power in O polarisation.
        x_mode_fraction : float
            Fraction of ray power in X polarisation.

        Raises
        ------
        ValueError
            Not in vacuum.
        """
        if not self.plasma_cache.vacuum:
            raise ValueError("Must be in vacuum.")

        arguments = self.hamiltonian_cache.arguments
        stix_polarisation_o_mode = vacuum_stix_polarisation(
            arguments.value[Dimensions.IDX_Y],
            arguments.value[Dimensions.IDX_N_PERP],
            arguments.value[Dimensions.IDX_N_PARALLEL],
            WaveMode.O,
        )

        # In vacuum E_perp = 0 so E_y and E_parallel are Jones vector.
        o_mode = abs(
            np.vdot(
                stix_polarisation_o_mode,
                self.hamiltonian_cache.polarisation.stix.value,
            )
        )

        return o_mode, max(0, min(1.0, 1.0 - o_mode))

    def force_dispersion_models(
        self,
        /,
        *,
        propagation_model: DispersionType = None,
        damping_model: DispersionType = None,
    ):
        """
        Force dispersion models for propagation and/or damping.

        Parameters
        ----------
        propagation_model : DispersionType, optional.
            If provided, forces dispersion model used for propagation.
        damping_model : DispersionType, optional.
            If provided, forces dispersion model used for damping.
        """
        self.force_propagation_model = propagation_model
        self.force_damping_model = damping_model

    def set_dispersion_models(
        self,
        propagation_model: DispersionType,
        damping_model: DispersionType,
    ):
        """
        Set models used for propagation and damping.

        Parameters
        ----------
        propagation_model : DispersionType, optional.
            Dispersion model used for propagation.
        damping_model : DispersionType, optional.
            Dispersion model used for damping.
        """
        if self.force_propagation_model is not None:
            self.propagation_model = self.force_propagation_model
        else:
            if propagation_model != self.propagation_model:
                logger.info(
                    "Switch propagation model to %s", propagation_model.name
                )

            self.propagation_model = propagation_model

        if self.force_damping_model is not None:
            self.damping_model = self.force_damping_model
        else:
            if damping_model != self.damping_model:
                logger.info("Switch damping model to %s", damping_model.name)

            self.damping_model = damping_model

    def calculate_hamiltonian(self, /, *, derivatives: int, determinant: bool):
        """
        Calculate hamiltonian data at current position. Includes eigenvalue
        and determinant of dispersion tensor and polarisation if selected /
        appropriate.

        Parameters
        ----------
        derivatives : int
            Number of derivatives to evaluate.
        determinant : bool
            If True, also calculate determinant of dispersion tensor.

        Raises
        ------
        ValueError
            Not in vacuum but wave_mode is WaveMode.ANY
        """
        # Calculate plasma parameters.
        self.plasma_cache.calculate(
            self.coordinate_cache, derivatives=derivatives
        )

        # Calculate perpendicular and parallel wavenumber.
        self.wave_cache.calculate_k_components(
            self.plasma_cache.magnetic_field_unit, derivatives=derivatives
        )

        # Set hamiltonian arguments.
        self.hamiltonian_cache.set_hamiltonian_arguments(
            self.plasma_cache, self.wave_cache, derivatives=derivatives
        )

        # Calculate recommended models for hamiltonian.
        self.hamiltonian_cache.calculate_recommended_models()

        self.set_dispersion_models(
            self.hamiltonian_cache.recommended_propagation_model,
            self.hamiltonian_cache.recommended_damping_model,
        )

        # Calculate dispersion tensor and eigenspace.
        self.hamiltonian_cache.calculate_dispersion_tensor(
            self.propagation_model, self.damping_model, derivatives=derivatives
        )

        # Calculate polarisation if not in vacuum. In vacuum the polarisation
        # stays constant.
        if not self.plasma_cache.vacuum:
            # Polarisation chosen as eigenvector of smallest eigenvalue.
            # Need 1 order smaller derivative than eigenvalue.
            self.hamiltonian_cache.calculate_stix_polarisation(
                self.plasma_cache,
                self.wave_cache,
                derivatives=max(0, derivatives - 1),
            )

        # Calculate eigenvalue and derivatives.
        self.hamiltonian_cache.calculate_eigenvalue(derivatives=derivatives)

        # Calculate determinant and derivatives if requested.
        if determinant:
            self.calculate_determinant(derivatives=derivatives)

    def calculate_determinant(self, /, *, derivatives: int):
        """
        Calculate determinant of dispersion tensor.

        Parameters
        ----------
        derivatives : int
            Number of derivatives to evaluate.
        """
        self.hamiltonian_cache.calculate_determinant(derivatives=derivatives)

    def calculate_state_vector_dt(
        self,
        time_ns: float,
        state_vector: FloatArray,
    ) -> FloatArray:
        """
        Calculate time derivative of integrator state vector.

        Parameters
        ----------
        time_ns : float
            Time [ns].
        state_vector : np.array[float]
            Integrator state vector.

        Returns
        -------
        state_vector_dt : np.array[float]
            Time derivative of integrator state vector.
        """
        # Increment counter on dy/dt evaluations.
        self.dy_dt_evaluations += 1

        # Unpack state vector and synchronise to caches.
        self.state.unpack(time_ns, state_vector)
        self._synchronise_xk(to_state=False)

        # Calculate hamiltonian.
        self.calculate_hamiltonian(derivatives=2, determinant=False)

        # Calculate and pack state vector derivative.
        self.state_dt.calculate(
            self.state, self.plasma_cache, self.hamiltonian_cache
        )

        # Pack state vector derivative.
        self.state_dt.pack()

        return self.state_dt.state_vector_dt

    def set_stop_condition(self, message: str, /, *, error: bool = True):
        """
        Set stop condition message for ray tracing.

        Parameters
        ----------
        message : str
            Stop condition message.
        error : bool, optional
            If True, log the stop condition as an error (something went wrong).
            Otherwise log the stop condition as a warning (benign stop
            condition).
        """
        self.stop_condition = message

        if error:
            logger.error("[%s] Stop: %s", self.name, message)
        else:
            logger.warning("[%s] Stop: %s", self.name, message)

    def accept_step(self):
        """
        Accept step to current state.
        """
        # Calculate determinant of dispersion tensor.
        self.calculate_determinant(derivatives=0)

        # If state vector is not in pure x representation calculate it.
        if not all(self.state.x_representation):
            self.state.calculate_x_representation()

        # Increment node index.
        self.step_accepted = True
        self.index += 1
