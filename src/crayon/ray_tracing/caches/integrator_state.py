"""
Caches for integrator state vector.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import mirror_upper_triangular_to_lower_triangular
from crayon.ray_tracing.caches.hamiltonian import HamiltonianCache
from crayon.ray_tracing.caches.plasma import PlasmaCache
from crayon.shared.constants import (
    C_M_PER_NS,
    SYMPLECTIC_MATRIX_J6,
    TWO_PI_INV,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.types import (
    BooleanArray,
    ComplexArray,
    FloatArray,
    FloatType,
)

logger = logging.getLogger(__name__)

_x = Dimensions.slice_x
_k = Dimensions.slice_k
_f = Dimensions.slice_f
_xk = Dimensions.slice_xk


def _pack_symmetric_tensor(tensor: FloatArray, unique: FloatArray):
    """
    Extract unique elements of a symmetric rank 2 tensor.

    Parameters
    ----------
    tensor : np.array[float]
        Symmetric rank 2 tensor.
    unique : np.array[float]
        Array to store unique elements into.
    """
    unique[0] = tensor[0, 0]
    unique[1] = tensor[0, 1]
    unique[2] = tensor[0, 2]
    unique[3] = tensor[1, 1]
    unique[4] = tensor[1, 2]
    unique[5] = tensor[2, 2]


def _unpack_symmetric_tensor(unique: FloatArray, tensor: FloatArray):
    """
    Fill symmetric rank 2 tensor from array of unique elements.

    Parameters
    ----------
    unique : np.array[float]
        Array tof unique elements.
    tensor : np.array[float]
        Symmetric rank 2 tensor to store elements in.
    """
    tensor[0, 0] = unique[0]
    tensor[0, 1] = unique[1]
    tensor[0, 2] = unique[2]
    tensor[1, 1] = unique[3]
    tensor[1, 2] = unique[4]
    tensor[2, 2] = unique[5]

    mirror_upper_triangular_to_lower_triangular(tensor)


class State:
    """
    Ray tracing integrator state vector.

    Attributes
    ----------
    adiabatic_phase_rad : float
        Adiabatic contribution to wave phase [radians].
    arc_length_m : float
        Arc length along ray [m].
    damping_fraction_collisional : float
        Fraction of damped power due to collisional damping.
    damping_fraction_external : float
        Fraction of damped power due to external power losses.
    damping_fraction_resonance : float
        Fraction of damped power due to resonant damping.
    eikonal_phase_rad : float
        Eikonal contribution to wave phase [radians].
    eikonal_phase_x_rad : float
        Eikonal phase in x representation [radians].
    focusing_tensor : np.array[float]
        Focusing tensor.
    focusing_tensor_x : np.array[float]
        Focusing tensor in x representation [m^-2].
    frequency_ghz : float
        Wave frequency [GHz].
    initial_power_w : float
        Initial power in ray [W].
    magnification : float
        Magnification factor for wave intensity.
    magnification_x : float
        Magnification factor for wave intensity in x representation.
    optical_depth_external : float
        Optical depth due to external power losses [nepers]
    optical_depth_internal : float
        Optical depth due to internal damping [nepers]
    position_cartesian : np.array[float]
        Ray position [m].
    state_vector : np.array[float]
        State vector evolved in integrator.
    time_ns : float
        Time along ray [ns].
    wave_action_density : float
        Wave action density in ray.
    wavevector_cartesian : np.array[float]
        Wavevector [m^-1].
    x_representation : np.array[bool]
        Representation fo coordinate components. If True x representation,
        if False k representation.

    Properties
    ----------
    optical_depth : float
        Total optical depth [nepers].
    phase_rad : float
        Total phase [radians]
    power_w : float
        Power in ray [W].

    Methods
    -------
    set_auxilliaries
        Set initial eikonal phase and initial power.
    increment_optical_depth_external
        Increment external optical depth.
    pack
        Pack state object contents into state vector.
    unpack
        Unpack state vector contents into state object.
    change_representation
        Change coordinate representation x <-> k.
    calculate_x_representation
        Calculate all quantities in x representation.
    """

    __slots__ = (
        "adiabatic_phase_rad",
        "arc_length_m",
        "damping_fraction_collisional",
        "damping_fraction_external",
        "damping_fraction_resonance",
        "eikonal_phase_rad",
        "eikonal_phase_x_rad",
        "focusing_tensor",
        "focusing_tensor_x",
        "frequency_ghz",
        "initial_power_w",
        "magnification",
        "magnification_x",
        "optical_depth_external",
        "optical_depth_internal",
        "position_cartesian",
        "state_vector",
        "time_ns",
        "wave_action_density",
        "wavevector_cartesian",
        "x_representation",
    )

    FOCUSING_TENSOR_SIZE = Dimensions.x.size * (Dimensions.x.size + 1) // 2
    STATE_VECTOR_SIZE = Dimensions.xk.size + 7 + FOCUSING_TENSOR_SIZE

    IDX_X = Dimensions.slice_x
    IDX_K = Dimensions.slice_k
    IDX_XK = Dimensions.slice_xk
    IDX_TAU = 2 * Dimensions.x.size
    IDX_FRACTION_RESONANCE = IDX_TAU + 1
    IDX_FRACTION_COLLISIONAL = IDX_TAU + 2
    IDX_ARC_LENGTH = IDX_TAU + 3
    IDX_EIKONAL_PHASE = IDX_TAU + 4
    IDX_ADIABATIC_PHASE = IDX_TAU + 5
    IDX_MAGNIFICATION = IDX_TAU + 6
    IDX_FOCUSING_TENSOR = slice(
        IDX_MAGNIFICATION + 1, IDX_MAGNIFICATION + 1 + FOCUSING_TENSOR_SIZE
    )

    def __init__(self):
        """
        Inits State.
        """
        # Phase space position.
        self.time_ns = 0.0
        self.frequency_ghz = 0.0
        self.position_cartesian = np.empty(Dimensions.x.size, dtype=FloatType)
        self.wavevector_cartesian = np.empty(
            Dimensions.k.size, dtype=FloatType
        )

        # Distance along the ray.
        self.arc_length_m = 0.0

        # Phase function.
        self.eikonal_phase_rad = 0.0
        self.adiabatic_phase_rad = 0.0

        # Optical depth and contributions of damped power.
        self.optical_depth_internal = 0.0
        self.damping_fraction_resonance = 0.0
        self.damping_fraction_collisional = 0.0
        self.damping_fraction_external = 0.0

        # Amplitude transport terms.
        self.magnification = 0.0
        self.focusing_tensor = np.empty(
            (Dimensions.x.size, Dimensions.x.size), dtype=FloatType
        )
        self.focusing_tensor.fill(0.0)

        # Representation (x = True, k = False) for each coordinate component.
        # Use mixed representations in x and k space to handle fold caustics.
        self.x_representation = np.ones(Dimensions.x.size, dtype=bool)

        # Pure x representations.
        self.eikonal_phase_x_rad = 0.0
        self.magnification_x = 0.0
        self.focusing_tensor_x = np.zeros_like(self.focusing_tensor)

        # Vector containing all evolved state components.
        self.state_vector = np.empty(self.STATE_VECTOR_SIZE, dtype=FloatType)

        # Initial power in ray [W].
        self.initial_power_w = 0.0

        # Optical depth encurred due to external sources of power loss e.g.
        # extinction at reflection, ray splitting at mode conversion, etc.
        self.optical_depth_external = 0.0

        # Wave action density.
        self.wave_action_density = 0.0

    @property
    def optical_depth(self) -> float:
        """Total optical depth [nepers]."""
        return self.optical_depth_internal + self.optical_depth_external

    @property
    def phase_rad(self) -> float:
        """Total wave phase [radians]."""
        return self.eikonal_phase_x_rad + self.adiabatic_phase_rad

    @property
    def power_w(self) -> float:
        """Remaining power [W]"""
        return self.initial_power_w * np.exp(-self.optical_depth)

    def set_auxilliaries(
        self,
        eikonal_phase_rad: float,
        initial_power_w: float,
    ):
        """
        Set other variables not related to phase space position.

        Parameters
        ----------
        eikonal_phase_rad : float
            Eikonal contribution to wave phase [radians].
        initial_power_w : float
            Initial power in ray [W].
        """
        self.eikonal_phase_rad = eikonal_phase_rad
        self.initial_power_w = initial_power_w

    def increment_optical_depth_external(self, dtau: float):
        """
        Increment external optical depth.

        Parameters
        ----------
        dtau : float
            Amount to increase external optical depth by.
        """
        self.damping_fraction_external += (1 - np.exp(-dtau)) * np.exp(
            -self.optical_depth
        )
        self.optical_depth_external += dtau

    def pack(self):
        """
        Pack contents of self into state vector.
        """
        y = self.state_vector

        y[self.IDX_X] = self.position_cartesian
        y[self.IDX_K] = self.wavevector_cartesian
        y[self.IDX_TAU] = self.optical_depth_internal
        y[self.IDX_FRACTION_RESONANCE] = self.damping_fraction_resonance
        y[self.IDX_FRACTION_COLLISIONAL] = self.damping_fraction_collisional

        y[self.IDX_ARC_LENGTH] = self.arc_length_m
        y[self.IDX_EIKONAL_PHASE] = self.eikonal_phase_rad
        y[self.IDX_ADIABATIC_PHASE] = self.adiabatic_phase_rad
        y[self.IDX_MAGNIFICATION] = self.magnification

        _pack_symmetric_tensor(
            self.focusing_tensor, y[self.IDX_FOCUSING_TENSOR]
        )

    def unpack(self, time_ns: float, state_vector: FloatArray):
        """
        Unpack contents of state vector into self.

        Parameters
        ----------
        time_ns : float
            Time [ns].
        state_vector : np.array[float]
            State vector of evolved variables.
        """
        y = state_vector

        self.time_ns = time_ns
        self.position_cartesian[:] = y[self.IDX_X]
        self.wavevector_cartesian[:] = y[self.IDX_K]
        self.optical_depth_internal = y[self.IDX_TAU]
        self.damping_fraction_resonance = y[self.IDX_FRACTION_RESONANCE]
        self.damping_fraction_collisional = y[self.IDX_FRACTION_COLLISIONAL]

        self.arc_length_m = y[self.IDX_ARC_LENGTH]
        self.eikonal_phase_rad = y[self.IDX_EIKONAL_PHASE]
        self.adiabatic_phase_rad = y[self.IDX_ADIABATIC_PHASE]
        self.magnification = y[self.IDX_MAGNIFICATION]

        _unpack_symmetric_tensor(
            y[self.IDX_FOCUSING_TENSOR], self.focusing_tensor
        )

        # Guard against overdamping. Sum of damped power in each channel must
        # be the same as the total damped power.
        # Any error due to numerical issues during integration.
        damping_fraction_internal = (
            self.damping_fraction_resonance + self.damping_fraction_collisional
        )

        if damping_fraction_internal + self.damping_fraction_external > 1.0:
            correction = (
                1.0 - self.damping_fraction_external
            ) / damping_fraction_internal

            self.damping_fraction_resonance *= correction
            self.damping_fraction_collisional *= correction

    def change_representation(self, /, *, x: bool, y: bool, z: bool):
        """
        Switch x / k representation used for a coordinate component.

        Parameters
        ----------
        x : bool
            If True, switch representation of x component.
        y : bool
            If True, switch representation of y component.
        z : bool
            If True, switch representation of z component.
        """
        for i, change in enumerate((x, y, z)):
            if change:
                # Flip representation flag.
                self.x_representation[i] = not self.x_representation[i]

        x_ = self.position_cartesian
        k_ = self.wavevector_cartesian

        _dim = int(x) + int(y) + int(z)

        if _dim == 1:
            # 1D change in representation.
            if x:
                ix = 0
            elif y:
                ix = 1
            elif z:
                ix = 2
            else:
                raise NotImplementedError((x, y, z))

            # Fourier transform component using stationary phase integral.
            self.eikonal_phase_rad -= x_[ix] * k_[ix]
            # self.magnification
            # self.focusing_tensor

        elif _dim == 2:  # noqa: PLR2004
            # 2D change in representation.
            if x and y:
                ix, iy = 0, 1
            elif x and z:
                ix, iy = 0, 2
            elif y and z:
                ix, iy = 1, 2
            else:
                raise NotImplementedError((x, y, z))

            # Fourier transform components using stationary phase integral.
            self.eikonal_phase_rad -= x_[ix] * k_[ix] + x_[iy] * k_[iy]
            # self.magnification
            # self.focusing_tensor

        elif _dim == 3:  # noqa: PLR2004
            # 3D change in representation.
            # Fourier transform components using stationary phase integral.
            self.eikonal_phase_rad -= np.dot(x_, k_)

        else:
            raise NotImplementedError((x, y, z))

    def calculate_x_representation(self):
        """
        Calculate all components in x representation.
        """


class StateDt:
    """
    Ray tracing integrator state vector time derivative.

    Attributes
    ----------
    adiabatic_phase_dt : float
        Rate of change of adiabatic phase [radians.s^-1].
    damping_fraction_collisional_dt : float
        Rate of change of damped power fraction due to collisional damping
        [s^-1].
    damping_fraction_resonance_dt : float
        Rate of change of damped power fraction due to resonant damping
        [s^-1].
    damping_rate : float
        Rate of change of optical depth [nepers.s^-1].
    eikonal_phase_dt : float
        Rate of change of eikonal phase [radians.s^-1].
    focusing_tensor_dt : np.array[float]
        Rate of change of focusing tensor [m^-2.s^-1].
    magnification_dt : float
        Rate of change of intensity magnification [s^-1].
    state_vector_dt : np.array[float]
        Rate of change of state vector.
    velocity : float
        Magnitude of velocity in x space [m.s^-1].
    velocity_xk : np.array[float]
        Rate of change of phase space position (x, k) [m.s^-1, m^-1.s^-1].

    Properties
    ----------
    velocity_x : np.array[float]
        Rate of change of position x [m.s^-1].
    velocity_k : np.array[float]
        Rate of change of wavevector k [m^-1.s^-1].

    Methods
    -------
    calculate

    pack
        Pack object contents into array.
    """

    __slots__ = (
        "adiabatic_phase_dt",
        "damping_fraction_collisional_dt",
        "damping_fraction_resonance_dt",
        "damping_rate",
        "eikonal_phase_dt",
        "focusing_tensor_dt",
        "magnification_dt",
        "state_vector_dt",
        "velocity",
        "velocity_xk",
    )

    def __init__(self):
        """
        Inits StateDt.
        """
        # Phase space position.
        self.velocity_xk = np.empty(Dimensions.xk.size, dtype=FloatType)

        # Optical depth and contributions of damped power.
        self.damping_rate = 0.0
        self.damping_fraction_resonance_dt = 0.0
        self.damping_fraction_collisional_dt = 0.0

        # Distance along the ray.
        self.velocity = 0.0

        # Phase function.
        self.eikonal_phase_dt = 0.0
        self.adiabatic_phase_dt = 0.0

        # Amplitude transport terms.
        self.magnification_dt = 0.0
        self.focusing_tensor_dt = np.zeros(
            (Dimensions.x.size, Dimensions.x.size), dtype=FloatType
        )
        self.state_vector_dt = np.empty(
            State.STATE_VECTOR_SIZE, dtype=FloatType
        )

    @property
    def velocity_x(self):
        """Velocity in x space [m.s^-1]."""
        return self.velocity_xk[State.IDX_X]

    @property
    def velocity_k(self):
        """Velocity in k space [m^-1.s^-1]."""
        return self.velocity_xk[State.IDX_K]

    def _calculate_velocity_vacuum(self, wavevector_cartesian: FloatArray):
        """
        Calculate phase space velocity in vacuum.

        Parameters
        ----------
        wavevector_cartesian : np.array[float]
            Cartesian wavevector [m^-1].
        """
        # Ray equations seem to generate funny stuff in vacuum if there is
        # numerical error.
        self.velocity_xk[_x] = (
            C_M_PER_NS
            * wavevector_cartesian
            / np.linalg.norm(wavevector_cartesian)
        )
        self.velocity_xk[_k] = 0.0

    def _calculate_velocity(
        self,
        eigenvalue_real_dz: FloatArray,
        eigenvalue_real_dz2: FloatArray,
        x_representation: BooleanArray,
    ):
        """
        Calculate phase space velocity.

        Parameters
        ----------
        eigenvalue_real_dz : np.array[float]
            First derivative of real part of eigenvalue with respect to
            extended phase space (x, k, f).
        eigenvalue_real_dz2 : np.array[float]
            Second derivative of real part of eigenvalue with respect to
            extended phase space (x, k, f).
        x_representation : np.array[bool
            Representation of each coordinate. True means x representation,
            False means k representation.
        """
        h_dxk = eigenvalue_real_dz[_xk]
        h_domega = TWO_PI_INV * eigenvalue_real_dz[_f]
        h_dxk2 = eigenvalue_real_dz2[_xk, _xk]

        # Mixed representation.
        _h_dxk = np.empty(Dimensions.xk.size)
        _h_dxk2 = np.empty((Dimensions.xk.size, Dimensions.xk.size))

        for idx_x in range(Dimensions.x.size):
            idx_k = idx_x + Dimensions.x.size

            if x_representation[idx_x]:
                # Use x representation.
                _h_dxk[idx_x] = h_dxk[idx_x]
                _h_dxk[idx_k] = h_dxk[idx_k]

                _h_dxk2[idx_x, idx_x] = h_dxk2[idx_x, idx_x]
                _h_dxk2[idx_x, idx_k] = h_dxk2[idx_x, idx_k]
                _h_dxk2[idx_k, idx_k] = h_dxk2[idx_k, idx_k]
            else:
                # Use k representation.
                _h_dxk[idx_x] = h_dxk[idx_k]
                _h_dxk[idx_k] = -h_dxk[idx_x]

                _h_dxk2[idx_x, idx_x] = h_dxk2[idx_k, idx_k]
                _h_dxk2[idx_x, idx_k] = -h_dxk2[idx_k, idx_x]
                _h_dxk2[idx_k, idx_k] = h_dxk2[idx_x, idx_x]

            # Second mixed derivatives commute.
            _h_dxk2[idx_k, idx_x] = h_dxk2[idx_x, idx_k]

        # Derivative of ray position.
        # Follows from dz/dt = {H, z} where {., .} is Poisson bracket.
        # {f, g} = df/dz . J6 . dg/dz.
        self.velocity_xk = (
            np.einsum("j, ji -> i", _h_dxk, SYMPLECTIC_MATRIX_J6) / h_domega
        )

    def _calculate_damping(
        self,
        eigenvalue_real_dz: FloatArray,
        eigenvalue_imaginary_resonance: float,
        eigenvalue_imaginary_collisional: float,
        optical_depth: float,
    ):
        """
        Calculate damping.

        Parameters
        ----------
        eigenvalue_real_dz : np.array[float]
            First derivative of real part of eigenvalue with respect to
            extended phase space (x, k, f).
        eigenvalue_imaginary_resonance : float
            Imaginary part of eigenvalue due to resonant damping.
        eigenvalue_imaginary_collisional : float
            Imaginary part of eigenvalue due to collisional damping.
        optical_depth : float
            Accumulated optical depth [nepers].
        """
        h_domega = TWO_PI_INV * eigenvalue_real_dz[_f]

        # Derivative of optical depth.
        damping_rate_resonance = (2.0 / h_domega) * max(
            0.0, eigenvalue_imaginary_resonance
        )
        damping_rate_collisional = (2.0 / h_domega) * max(
            0.0, eigenvalue_imaginary_collisional
        )
        self.damping_rate = damping_rate_resonance + damping_rate_collisional

        # Derivative of power damped in resonance and collisions normalised
        # to input power.
        # max prevents overflow during integration when optical depth can
        # become very large and negative.
        decay = np.exp(-max(0.0, optical_depth))
        self.damping_fraction_resonance_dt = decay * damping_rate_resonance
        self.damping_fraction_collisional_dt = decay * damping_rate_collisional

    def _calculate_eikonal_phase(self, wavevector_cartesian: FloatArray):
        """
        Calculate rate of change of eikonal wave phase.

        Parameters
        ----------
        wavevector_cartesian : np.array[float]
            Cartesian wavevector [m^-1].
        """
        # Rate of change of eikonal phase.
        self.eikonal_phase_dt = np.dot(wavevector_cartesian, self.velocity_x)

    def _calculate_adiabatic_phase(
        self,
        eigenvalue_real_dz: FloatArray,
        stix_polarisation: ComplexArray,
        stix_polarisation_dz: ComplexArray,
        dispersion_tensor_hermitian: ComplexArray,
    ):
        """
        Calculate rate of change of adiabatic wave phase aka Berry phase,
        geometric phase, etc.

        Parameters
        ----------
        stix_polarisation : np.array[complex]
            Stix frame polarisation.
        stix_polarisation_dz : np.array[complex]
            First derivative of Stix frame polarisation with respect to
            extended phase space position z = [x, k, f].
        """
        h_domega = TWO_PI_INV * eigenvalue_real_dz[_f]

        stix_polarisation_dx = stix_polarisation_dz[:, _x]
        stix_polarisation_dk = stix_polarisation_dz[:, _k]
        stix_polarisation_dxk = stix_polarisation_dz[:, _xk]

        poisson_bracket = (
            np.einsum(
                "ia, ja -> ij",
                stix_polarisation_dx,
                np.conj(stix_polarisation_dk),
            )
            - np.einsum(
                "ia, ja -> ij",
                stix_polarisation_dk,
                np.conj(stix_polarisation_dx),
            )
        ) / h_domega

        self.adiabatic_phase_dt = (
            np.einsum(
                "i, ij, j",
                np.conj(stix_polarisation),
                stix_polarisation_dxk,
                self.velocity_xk,
            ).imag
            + 0.5
            * np.einsum(
                "ij, ij", dispersion_tensor_hermitian, poisson_bracket
            ).imag
        )

    def calculate(
        self,
        state: State,
        plasma_cache: PlasmaCache,
        hamiltonian_cache: HamiltonianCache,
    ):
        """
        Calculate state vector time derivative.

        Parameters
        ----------
        state : State
            Ray state object.
        plasma_cache : PlasmaCache
            Cache containing plasma parameter data.
        hamiltonian_cache : HamiltonianCache
            Cache containing ray hamiltonian data.
        """
        # Calculate phase space velocity.
        if plasma_cache.vacuum:
            self._calculate_velocity_vacuum(state.wavevector_cartesian)
        else:
            self._calculate_velocity(
                hamiltonian_cache.eigenvalue.first_derivative.z,
                hamiltonian_cache.eigenvalue.second_derivative.z,
                state.x_representation,
            )

        # Derivative of arclength ds / dt = ||dx / dt||.
        self.velocity = np.linalg.norm(self.velocity_x)

        # Calculate damping.
        self._calculate_damping(
            hamiltonian_cache.eigenvalue.first_derivative.z,
            hamiltonian_cache.eigenvalue.imag_resonance,
            hamiltonian_cache.eigenvalue.imag_collisional,
            state.optical_depth,
        )

        # Calculate wave phase.
        self._calculate_eikonal_phase(state.wavevector_cartesian)
        self._calculate_adiabatic_phase(
            hamiltonian_cache.eigenvalue.first_derivative.z,
            hamiltonian_cache.polarisation.stix.value,
            hamiltonian_cache.polarisation.stix.first_derivative.z,
            hamiltonian_cache.dispersion_tensor.hermitian.value,
        )

        self.magnification_dt = 0.0
        self.focusing_tensor_dt.fill(0.0)

        # Below doesn't work yet.
        return

        h_dz = hamiltonian_cache.eigenvalue.first_derivative.z
        h_domega = TWO_PI_INV * h_dz[Dimensions.slice_f]
        h_dz2 = hamiltonian_cache.eigenvalue.second_derivative.z

        if self.x_representation:
            # Derivative of Eikonal phase dphi/dt = k . dx/dt.
            self.eikonal_phase_dt = np.dot(
                self.wavevector_cartesian, self.velocity_x
            )

            # Derivative of magnification M.
            # Wave intensity I(t) = I0 * exp[M(t)] where I0 = vacuum amplitude.
            # i.e. amplification of wave amplitude due to non-dissapative
            # processes e.g. acceleration of ray, convergence of nearby rays.
            h_dz2 = hamiltonian_cache.eigenvalue.second_derivative.real
            _x, _k = Dimensions.slice_x, Dimensions.slice_k

            self.magnification_dt = (
                np.trace(h_dz2[_x, _k])
                + np.einsum("ij, ij", h_dz2[_k, _k], self.focusing_tensor_x)
            ) / h_domega

            # Derivative of focusing tensor.
            self.focusing_tensor_dt[:, :] = (
                h_dz2[_x, _x]
                + np.einsum(
                    "ab, ia, bj -> ij",
                    h_dz2[_k, _k],
                    self.focusing_tensor_x,
                    self.focusing_tensor_x,
                )
                + np.einsum(
                    "ia, ja -> ij", h_dz2[_x, _k], self.focusing_tensor_x
                )
                + np.einsum(
                    "ja, ia -> ij", h_dz2[_x, _k], self.focusing_tensor_x
                )
            ) / h_domega
        else:
            # k representation of evolution equations.
            # Eikonal phase.
            self.eikonal_phase_dt = -np.dot(
                self.position_cartesian, self.velocity_k
            )

            # Magnification.
            self.magnification_dt = (
                -(
                    np.trace(h_dz2[_x, _k])
                    + np.einsum(
                        "ij, ij", h_dz2[_x, _x], self.focusing_tensor_k
                    )
                )
                / h_domega
            )

            # Focusing tensor.
            self.focusing_tensor_dt[:, :] = (
                -(
                    h_dz2[_k, _k]
                    + np.einsum(
                        "ab, ia, bj -> ij",
                        h_dz2[_x, _x],
                        self.focusing_tensor_k,
                        self.focusing_tensor_k,
                    )
                    + np.einsum(
                        "ia, ja -> ij", h_dz2[_k, _x], self.focusing_tensor_k
                    )
                    + np.einsum(
                        "ja, ia -> ij", h_dz2[_k, _x], self.focusing_tensor_k
                    )
                )
                / h_domega
            )

    def pack(self):
        """
        Pack contents of self into state vector.
        """
        y = self.state_vector_dt

        y[State.IDX_X] = self.velocity_x
        y[State.IDX_K] = self.velocity_k
        y[State.IDX_TAU] = self.damping_rate
        y[State.IDX_FRACTION_RESONANCE] = self.damping_fraction_resonance_dt
        y[State.IDX_FRACTION_COLLISIONAL] = (
            self.damping_fraction_collisional_dt
        )

        y[State.IDX_ARC_LENGTH] = self.velocity
        y[State.IDX_EIKONAL_PHASE] = self.eikonal_phase_dt
        y[State.IDX_ADIABATIC_PHASE] = self.adiabatic_phase_dt
        y[State.IDX_MAGNIFICATION] = self.magnification_dt

        _pack_symmetric_tensor(
            self.focusing_tensor_dt, y[State.IDX_FOCUSING_TENSOR]
        )
