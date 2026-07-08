"""
Caches for ray Hamiltonian.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import (
    adjugate_3x3_cofactors,
    first_derivative,
    second_derivative,
)
from crayon.dispersion import (
    ColdDispersion,
    DispersionType,
    FullyRelativisticDispersion,
    NonRelativisticDispersion,
    SusceptibilityCache,
    calculate_harmonic_range,
    determinant,
    determinant_dx,
    determinant_dx2,
    eigenvalue,
    eigenvalue_dx,
    eigenvalue_dx2,
    eigenvector_dx,
    polarisation,
    polarisation_phase_convention_factor,
    vacuum_dispersion_tensor,
    vacuum_dispersion_tensor_dq,
    vacuum_dispersion_tensor_dq2,
    vacuum_stix_polarisation,
)
from crayon.ray_tracing.caches.base import DerivativeCacheZ
from crayon.ray_tracing.caches.plasma import PlasmaCache
from crayon.ray_tracing.caches.wave import WaveCache
from crayon.shared.constants import (
    MAX_NONRELATIVISTIC_NORM_ELECTRON_TEMPERATURE,
    MIN_NORM_ELECTRON_TEMPERATURE,
    WaveMode,
)
from crayon.shared.data_structures import Result
from crayon.shared.dimensions import Dimensions
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)

dispersion_models = {
    DispersionType.COLD: ColdDispersion,
    DispersionType.NON_RELATIVISTIC: NonRelativisticDispersion,
    DispersionType.FULLY_RELATIVISTIC: FullyRelativisticDispersion,
}

# Sizes of dimensions.
X = Dimensions.x.size
Q = Dimensions.q.size
Z = Dimensions.z.size

# Slices into derivative dimension.
_x = Dimensions.slice_x
_xk = Dimensions.slice_xk
_f = Dimensions.slice_f

# Indicies in q.
_X = Dimensions.IDX_X
_Y = Dimensions.IDX_Y
_Z = Dimensions.IDX_Z
_THETA = Dimensions.IDX_THETA
_N_PERP = Dimensions.IDX_N_PERP
_N_PARALLEL = Dimensions.IDX_N_PARALLEL

MAX_NONRELATIVISTIC_GAMMA = 1 + MAX_NONRELATIVISTIC_NORM_ELECTRON_TEMPERATURE


class FirstDerivativeQZ:
    """
    Cache for first derivative of an array with respect to Hamiltonian
    arguments q and extended phase space z. All coordinate dependent objects
    are in given in Cartesian.

    Attributes
    ----------
    q : np.array[complex]
        First derivative with respect to Hamiltonian arguments q.
    z : np.array[complex]
        First derivative with respect to extended phase space z
    """

    def __init__(self, shape: tuple[int], /, *, is_complex: bool = False):
        """
        Inits FirstDerivative.
        """
        dtype = complex if is_complex else float

        self.q = np.zeros((*shape, Q), dtype=dtype)
        self.z = np.zeros((*shape, Z), dtype=dtype)


class SecondDerivativeQZ:
    """
    Cache for first derivative of an array with respect to Hamiltonian
    arguments q and extended phase space z. All coordinate dependent objects
    are in given in Cartesian.

    Attributes
    ----------
    q : np.array[complex]
        Second derivative with respect to Hamiltonian arguments q.
    z : np.array[complex]
        Second derivative with respect to extended phase space z
    """

    def __init__(self, shape: tuple[int], /, *, is_complex: bool = False):
        """
        Inits FirstDerivative.
        """
        dtype = complex if is_complex else float

        self.q = np.zeros((*shape, Q, Q), dtype=dtype)
        self.z = np.zeros((*shape, Z, Z), dtype=dtype)


class TensorCache:
    """
    Cache for a rank 2 tensor and its derivatives with respect to Hamiltonian
    arguments q and extended phase space position z. All coordinate dependent
    objects are in given in Cartesian.

    Attributes
    ----------
    value : np.array[complex]
        Parameter value.
    first_derivative : np.array[complex]
        First derivative with respect to q and z.
    second_derivative : np.array[complex]
        Second derivative with respect to q and z.
    """

    __slots__ = (
        "first_derivative",
        "second_derivative",
        "value",
    )

    def __init__(self):
        """
        Inits TensorCache.
        """
        self.value = np.zeros((X, X), dtype=complex)
        self.first_derivative = FirstDerivativeQZ((X, X), is_complex=True)
        self.second_derivative = SecondDerivativeQZ((X, X), is_complex=True)


class DispersionTensorCache:
    """
    Cache for dispersion tensor.

    Attributes
    ----------
    antihermitian_collisional : np.array[complex]
        Anti-Hermitian part of dispersion tensor associated with collisional
        damping.
    antihermitian_resonance : np.array[complex]
        Anti-Hermitian part of dispersion tensor associated with resonant
        damping.
    hermitian : np.array[complex]
        Hermitian part of dispersion tensor and its derivatives with respect
        to Hamiltonian arguments q and extended phase space z.

    Methods
    -------
    calculate
        Calculate dispersion tensor from susceptibility cache.
    """

    __slots__ = (
        "antihermitian_collisional",
        "antihermitian_resonance",
        "hermitian",
    )

    def __init__(self):
        """
        Inits DispersionTensorCache.
        """
        self.hermitian = TensorCache()
        self.antihermitian_resonance = np.empty((X, X), dtype=complex)
        self.antihermitian_collisional = np.empty((X, X), dtype=complex)

    def calculate(
        self,
        q: FloatArray,
        susceptibility_cache: SusceptibilityCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate dispersion tensor from susceptibility cache.

        Parameters
        ----------
        q : np.array[float]
            Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel].
        susceptibility_cache : SusceptibilityCache
            Cache of electric susceptibility tensor values.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Calculate hermitian part.
        vacuum_dispersion_tensor(q, return_array=self.hermitian.value)
        self.hermitian.value += susceptibility_cache.hermitian

        self.antihermitian_resonance[:, :] = (
            susceptibility_cache.antihermitian_resonance
        )

        self.antihermitian_collisional[:, :] = (
            susceptibility_cache.antihermitian_collisional
        )

        if derivatives == 0:
            return

        # Calculate first derivative.
        vacuum_dispersion_tensor_dq(
            q, return_array=self.hermitian.first_derivative.q
        )
        self.hermitian.first_derivative.q += susceptibility_cache.hermitian_dq

        if derivatives == 1:
            return

        # Calculate second derivative.
        vacuum_dispersion_tensor_dq2(
            return_array=self.hermitian.second_derivative.q
        )
        self.hermitian.second_derivative.q += (
            susceptibility_cache.hermitian_dq2
        )

        if derivatives == 2:  # noqa: PLR2004
            return

        # Calculate third derivative.
        raise ValueError(derivatives)


class DeterminantCache:
    """
    Cache for determinant of dispersion tensor.

    Attributes
    ----------
    first_derivative : FirstDerivativeQZ
        First derivative with respect to q and z.
    imag : float
        Imaginary part.
    real : float
        Real part.
    second_derivative : SecondDerivativeQZ
        Second derivative with respect to q and z.
    """

    __slots__ = (
        "first_derivative",
        "imag",
        "real",
        "second_derivative",
    )

    def __init__(self):
        """
        Inits DeterminantCache.
        """
        self.real = 0.0
        self.imag = 0.0
        self.first_derivative = FirstDerivativeQZ((), is_complex=False)
        self.second_derivative = SecondDerivativeQZ((), is_complex=False)


class EigenvalueCache(DeterminantCache):
    """
    Cache for eigenvalue of dispersion tensor.

    Attributes
    ----------
    first_derivative : FirstDerivativeQZ
        First derivative with respect to q and z.
    imag : float
        Imaginary part.
    imag_collisional
        Imaginary part due to collisional effects.
    imag_resonance
        Imaginary part due to resonant effects.
    real : float
        Real part.
    second_derivative : SecondDerivativeQZ
        Second derivative with respect to q and z.
    """

    __slots__ = ("imag_collisional", "imag_resonance")

    def __init__(self):
        """
        Inits EigenvalueCache.
        """
        super().__init__()
        self.imag_resonance = 0.0
        self.imag_collisional = 0.0


class StixPolarisationCache:
    """
    Cache for Stix polarisation and its derivatives with respect to Hamiltonian
    arguments q and extended phase space position z. All coordinate dependent
    objects are in given in Cartesian.

    Attributes
    ----------
    value : np.array[complex]
        Stix polarisation.
    first_derivative : np.array[complex]
        First derivative with respect to q and z.
    second_derivative : np.array[complex]
        Second derivative with respect to q and z.
    """

    __slots__ = ("first_derivative", "second_derivative", "value")

    def __init__(self):
        """
        Inits StixPolarisationCache.
        """
        self.value = np.empty(X, dtype=complex)
        self.first_derivative = FirstDerivativeQZ((X,), is_complex=True)
        self.second_derivative = SecondDerivativeQZ((X,), is_complex=True)


class PolarisationCache:
    """
    Cache for electric field polarisation and its derivatives with respect to
    Hamiltonian arguments q and extended phase space position z. All
    coordinate dependent objects are in given in Cartesian.

    Attributes
    ----------
    cartesian : np.array[complex]
        Cartesian polarisation.
    stix : StixPolarisationCache
        Cache for Stix polarisation and its derivatives with respect to q and
        z.
    """

    __slots__ = ("cartesian", "stix")

    def __init__(self):
        """
        Inits PolarisationCache.
        """
        self.cartesian = np.empty(X, dtype=complex)
        self.stix = StixPolarisationCache()


class HamiltonianCache:
    """
    Cache for ray Hamiltonian data.

    Attributes
    ----------
    arguments : DerivativeCacheZ
        Hamiltonian arguments q = [X, Y, Z, theta, n_perp, n_parallel] and
        derivatives with respect to extended phase space z.
    determinant : DeterminantCache
        Determinant of dispersion tensor and derivatives with respect to
        Hamiltonian arguments q and extended phase space z.
    determinant_error_frequency : float
        Deviation of real part of determinant of dispersion tensor from zero
        expressed as a frequency shift [GHz].
    dispersion_tensor : DispersionTensorCache
        Dispersion tensor and derivatives with respect to Hamiltonian arguments
        q and extended phase space z.
    eigenvalue : EigenvalueCache
        Eigenvalue of dispersion tensor and derivatives with respect to
        Hamiltonian arguments q and extended phase space z.
    eigenvalue_error_frequency : float
        Deviation of real part of eigenvalue of dispersion tensor from zero
        expressed as a frequency shift [GHz].
    eigenvalues : np.array[complex]
        All eigenvalues of dispersion tensor.
    eigenvectors : np.array[complex]
        All eigenvectors of dispersion tensor.
    normalised_em_energy_density : float
        Normalised electromagnetic energy density aka fluxn.
    polarisation : PolarisationCache
        Electric field polarisation and derivatives with respect to
        Hamiltonian arguments q and extended phase space z.
    recommended_damping_model : DispersionType
        Recommended dispersion model for damping based on plasma and wave
        parameters.
    recommended_propagation_model : DispersionType
        Recommended dispersion model for propagation based on plasma and wave
        parameters.
    susceptibility : SusceptibilityCache
        Cache for electric susceptibility and derivatives with respect to
        Hamiltonian arguments q.

    Methods
    -------
    set_hamiltonian_arguments
        Set Hamiltonian arguments q.
    calculate_recommended_models
        Calculate recommended dispersion tensor models for propagation and
        damping.
    find_root_n
        Calculate magnitude of refractive index required to satisfy dispersion
        relation.
    find_root_n_perp
        Calculate magnitude of perpendicular refractive index required to
        satisfy dispersion relation.
    calculate_dispersion_tensor
        Calculate dispersion tensor and derivatives with respect to Hamiltonian
        arguments q and extended phase space z.
    calculate_stix_polarisation
        Calculate stix_polarisation as null space of dispersion tensor.
    calculate_vacuum_stix_polarisation
        Calculate stix polarisation of O and X mode in vacuum.
    calculate_cartesian_polarisation_from_stix
        Calculate Cartesian polarisation from Stix polarisation.
    calculate_stix_polarisation_from_cartesian
        Calculate Stix polarisation from Cartesian polarisation.
    calculate_eigenvalue
        Calculate eigenvalue for given mode and derivative with respect to
        Hamiltonian arguments q and extended phase space z.
    calculate_determinant
        Calculate determinant of dispersion tensor and derivative with respect
        to Hamiltonian arguments q and extended phase space z.
    calculate_normalised_em_flux
        Calculate normalised electromagnetic flux.
    """

    __slots__ = (
        "arguments",
        "determinant",
        "determinant_error_frequency",
        "dispersion_tensor",
        "eigenvalue",
        "eigenvalue_error_frequency",
        "eigenvalues",
        "eigenvectors",
        "normalised_em_energy_density",
        "polarisation",
        "recommended_damping_model",
        "recommended_propagation_model",
        "susceptibility",
    )

    def __init__(self):
        """
        Inits HamiltonianCache.
        """
        super().__init__()

        # Arguments of susceptibility models.
        self.arguments = DerivativeCacheZ((Q,))

        # Recommended models for Hamiltonian.
        self.recommended_propagation_model = DispersionType.COLD
        self.recommended_damping_model = DispersionType.COLD

        # Electric susceptibility model.
        self.susceptibility = SusceptibilityCache()

        # Dispersion tensor.
        self.dispersion_tensor = DispersionTensorCache()

        # All eigenvalues and eigenvectors.
        self.eigenvalues = np.empty(X, dtype=complex)
        self.eigenvectors = np.empty((X, X), dtype=complex)

        # Eigenvalue, determinant and polarisation of mode.
        self.eigenvalue = EigenvalueCache()
        self.determinant = DeterminantCache()
        self.polarisation = PolarisationCache()

        # Fraction of wave energy density carried in electromagnetic field.
        self.normalised_em_energy_density = 0.0

        # Error in Hamiltonian expressed as a frequency.
        self.eigenvalue_error_frequency = 0.0
        self.determinant_error_frequency = 0.0

    def set_hamiltonian_arguments(
        self,
        plasma_cache: PlasmaCache,
        wave_cache: WaveCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Set Hamiltonian arguments q.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Cache containing plasma parameter data.
        wave_cache : WaveCache
            Cache containing wave parameter data.
        derivatives : int
            Order of derivatives to set.

        Raises
        ------
        ValueError
            Derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Plasma data.
        x = plasma_cache.normalised_electron_density
        y = plasma_cache.normalised_magnetic_field_strength
        z = plasma_cache.normalised_collision_rate
        theta = plasma_cache.normalised_electron_temperature

        # Wave data.
        inv_f_ghz = 1.0 / wave_cache.frequency_ghz
        inv_f_ghz2 = inv_f_ghz * inv_f_ghz
        k0_inv = 1.0 / wave_cache.vacuum_wavenumber_per_m
        k_perp = wave_cache.k_perp
        k_parallel = wave_cache.k_parallel

        # Set values.
        _q = self.arguments.value
        _q.fill(0.0)

        _q[_X] = x.value
        _q[_Y] = y.value
        _q[_Z] = z.value
        _q[_THETA] = theta.value
        _q[_N_PERP] = k_perp.value * k0_inv
        _q[_N_PARALLEL] = k_parallel.value * k0_inv

        if derivatives == 0:
            return

        # Set first derivatives.
        _q_dz = self.arguments.first_derivative
        _q_dz.fill(0.0)

        _q_dz[_X, _x] = x.first_derivative[:]
        _q_dz[_X, _f] = -2.0 * _q[_X] * inv_f_ghz

        _q_dz[_Y, _x] = y.first_derivative[:]
        _q_dz[_Y, _f] = -_q[_Y] * inv_f_ghz

        _q_dz[_Z, _x] = z.first_derivative[:]
        _q_dz[_Z, _f] = -_q[_Z] * inv_f_ghz

        _q_dz[_THETA, _x] = theta.first_derivative[:]

        _q_dz[_N_PERP, _xk] = k_perp.first_derivative[:] * k0_inv
        _q_dz[_N_PERP, _f] = -_q[_N_PERP] * inv_f_ghz

        _q_dz[_N_PARALLEL, _xk] = k_parallel.first_derivative[:] * k0_inv
        _q_dz[_N_PARALLEL, _f] = -_q[_N_PARALLEL] * inv_f_ghz

        if derivatives == 1:
            return

        # Set second derivatives.
        _q_dz2 = self.arguments.second_derivative
        _q_dz2.fill(0.0)

        _q_dz2[_X, _x, _x] = x.second_derivative[:, :]
        _q_dz2[_X, _x, _f] = -2.0 * _q_dz[_X, _x] * inv_f_ghz
        _q_dz2[_X, _f, _f] = 6 * _q[_X] * inv_f_ghz2

        _q_dz2[_Y, _x, _x] = y.second_derivative[:, :]
        _q_dz2[_Y, _x, _f] = -_q_dz[_Y, _x] * inv_f_ghz
        _q_dz2[_Y, _f, _f] = 2 * _q[_Y] * inv_f_ghz2

        _q_dz2[_Z, _x, _x] = z.second_derivative[:, :]
        _q_dz2[_Z, _x, _f] = -_q_dz[_Z, _x] * inv_f_ghz
        _q_dz2[_Z, _f, _f] = 2 * _q[_Z] * inv_f_ghz2

        _q_dz2[_THETA, _x, _x] = theta.second_derivative[:, :]

        _q_dz2[_N_PERP, _xk, _xk] = k_perp.second_derivative[:, :] * k0_inv
        _q_dz2[_N_PERP, _xk, _f] = -_q_dz[_N_PERP, _xk] * inv_f_ghz
        _q_dz2[_N_PERP, _f, _f] = 2 * _q[_N_PERP] * inv_f_ghz2

        _q_dz2[_N_PARALLEL, _xk, _xk] = (
            k_parallel.second_derivative[:, :] * k0_inv
        )
        _q_dz2[_N_PARALLEL, _xk, _f] = -_q_dz[_N_PARALLEL, _xk] * inv_f_ghz
        _q_dz2[_N_PARALLEL, _f, _f] = 2 * _q[_N_PARALLEL] * inv_f_ghz2

        # Second partial derivatives commute.
        _q_dz2[:, _f, :] = _q_dz2[:, :, _f]

        if derivatives == 2:  # noqa: PLR2004
            return

        # Set third derivatives.
        raise ValueError(derivatives)

    def calculate_recommended_models(self):
        """
        Calculate recommended dispersion tensor models for propagation and
        damping.
        """
        _q = self.arguments.value

        x = _q[_X]
        y = abs(_q[_Y])
        theta = _q[_THETA]
        n_perp = _q[_N_PERP]
        n_parallel = abs(_q[_N_PARALLEL])

        # Find bounding harmonics of current magnetic field strength.
        n_below, n_above = calculate_harmonic_range(y)

        # Calculate recommended dispersion models.
        # Use cold dispersion if
        #   1. FLR parameter is very small => N_perp * beta << Y
        #   2. Doppler parameter is very large => 1 - nY >> N_parallel * beta
        delta_ny = min(abs(1 - n_below * y), abs(1 - n_above * y))

        # If density very small use cold to avoid issues with branches.
        if x < 1e-2 or y < 1e-4:  # noqa: PLR2004
            self.recommended_propagation_model = DispersionType.COLD
        else:
            beta_thermal = np.sqrt(2 * theta)
            flr_small = 10.0 * n_perp * beta_thermal < y
            doppler_large = delta_ny > 10.0 * abs(n_parallel) * beta_thermal
            doppler_large = True

            if flr_small and doppler_large:
                self.recommended_propagation_model = DispersionType.COLD
            elif theta > MAX_NONRELATIVISTIC_NORM_ELECTRON_TEMPERATURE:
                # Te > 5 keV.
                self.recommended_propagation_model = (
                    DispersionType.FULLY_RELATIVISTIC
                )
            else:
                self.recommended_propagation_model = (
                    DispersionType.NON_RELATIVISTIC
                )

        # Damping model.
        if theta < MIN_NORM_ELECTRON_TEMPERATURE:
            # If temperature extremely small ignore resonant effects.
            self.recommended_damping_model = DispersionType.COLD
        else:
            # Check for resonance.
            alpha = 4.0
            q_thermal = np.sqrt(theta * (2 + theta))
            q_thermal_max = alpha * q_thermal
            gamma_thermal_max = np.sqrt(1 + q_thermal_max**2)

            n_parallel2 = n_parallel * n_parallel

            # Should move this to dedicated function about resonance curves.
            discriminant_below = np.square(n_below * y) - (1 - n_parallel2)
            discriminant_above = np.square(n_above * y) - (1 - n_parallel2)

            if discriminant_below < 0.0:
                resonance_below = False
            else:
                resonance_below = (
                    n_below * y * n_parallel
                    - np.sign(n_parallel) * np.sqrt(discriminant_below)
                ) / (1 - n_parallel2) / q_thermal_max < 1

            if discriminant_above < 0.0:
                resonance_above = False
            else:
                resonance_above = (
                    n_below * y * n_parallel
                    - np.sign(n_parallel) * np.sqrt(discriminant_above)
                ) / (1 - n_parallel2) / q_thermal_max < 1

            if resonance_below or resonance_above:
                # Unless very small relativistic shift use relativistic model.
                if gamma_thermal_max < MAX_NONRELATIVISTIC_GAMMA:
                    self.recommended_damping_model = (
                        DispersionType.FULLY_RELATIVISTIC
                    )
                else:
                    self.recommended_damping_model = (
                        DispersionType.NON_RELATIVISTIC
                    )
            else:
                self.recommended_damping_model = DispersionType.COLD

        # While fully relativistic dispersion not available. See issue #33.
        if (
            self.recommended_propagation_model
            == DispersionType.FULLY_RELATIVISTIC
        ):
            self.recommended_propagation_model = (
                DispersionType.NON_RELATIVISTIC
            )

        if self.recommended_damping_model == DispersionType.FULLY_RELATIVISTIC:
            self.recommended_damping_model = DispersionType.NON_RELATIVISTIC

    def find_root_n(
        self,
        propagation_model: DispersionType,
        wave_mode: WaveMode,
        /,
        *,
        kinetic: bool,
    ) -> Result:
        """
        Calculate magnitude of refractive index required to satisfy dispersion
        relation.

        Parameters
        ----------
        propagation_model : DispersionType
            Model to use for Hermitian part of susceptibility.
        wave_mode : WaveMode
            Wave mode to solve for.
        kinetic : bool
            If True, search for large n_perp solution. Otherwise search for
            small n_perp solution.

        Returns
        -------
        n : Result
            Result containing magnitude of refractive index.
        """
        return dispersion_models[propagation_model].calculate_n(
            self.arguments.value, wave_mode, kinetic=kinetic
        )

    def find_root_n_perp(
        self,
        propagation_model: DispersionType,
        wave_mode: WaveMode,
        /,
        *,
        kinetic: bool,
    ) -> Result:
        """
        Calculate magnitude of perpendicular refractive index required to
        satisfy dispersion relation.

        Parameters
        ----------
        propagation_model : DispersionType
            Model to use for Hermitian part of susceptibility.
        wave_mode : WaveMode
            Wave mode to solve for.
        kinetic : bool
            If True, search for large n_perp solution. Otherwise search for
            small n_perp solution.

        Returns
        -------
        n_perp : Result
            Result containing magnitude of perpendicular refractive index.
        """
        return dispersion_models[propagation_model].calculate_n_perp(
            self.arguments.value, wave_mode, kinetic=kinetic
        )

    def calculate_dispersion_tensor(
        self,
        propagation_model: DispersionType,
        damping_model: DispersionType,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate dispersion tensor and derivatives with respect to Hamiltonian
        arguments q and extended phase space z.

        Parameters
        ----------
        propagation_model : DispersionType
            Model to use for Hermitian part of susceptibility.
        damping_model : DispersionType
            Model to use for anti-Hermitian part of susceptibility.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            Derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        if propagation_model == damping_model:
            # Same model for hermitian and anti-hermitian part.
            dispersion_models[propagation_model].calculate_susceptibility(
                self.arguments.value,
                self.susceptibility,
                derivatives,
                hermitian=True,
                antihermitian=True,
            )
        else:
            # Different model for hermitian and anti-hermitian part.
            dispersion_models[damping_model].calculate_susceptibility(
                self.arguments.value,
                self.susceptibility,
                derivatives,
                hermitian=False,
                antihermitian=True,
            )

            dispersion_models[propagation_model].calculate_susceptibility(
                self.arguments.value,
                self.susceptibility,
                derivatives,
                hermitian=True,
                antihermitian=False,
            )

        # Construct dispersion tensor.
        self.dispersion_tensor.calculate(
            self.arguments.value, self.susceptibility, derivatives=derivatives
        )

        # Calculate all eigenvalues and eigenvectors of the dispersion tensor.
        # i-th eigenvector returned by eig is eigenvectors[:, i] so transpose.
        _l, _e = np.linalg.eig(self.dispersion_tensor.hermitian.value)
        _e = _e.T

        # Sort in ascending order by absolute value of eigenvalue.
        _order = np.argsort(abs(_l))
        self.eigenvalues[:] = _l[_order]
        self.eigenvectors[:, :] = _e[_order]

        # Apply phase convention to eigenvectors.
        for i in range(self.eigenvectors.shape[0]):
            self.eigenvectors[i] *= polarisation_phase_convention_factor(
                self.eigenvectors[i]
            )

        if derivatives == 0:
            return

        # First derivative of dispersion tensor with respect to z.
        first_derivative(
            self.dispersion_tensor.hermitian.first_derivative.q,
            self.arguments.first_derivative,
            (Dimensions.x.size, Dimensions.x.size),
            Dimensions.q.size,
            Dimensions.z.size,
            return_array=self.dispersion_tensor.hermitian.first_derivative.z,
        )

        if derivatives == 1:
            return

        # Second derivative of dispersion tensor with respect to z.
        second_derivative(
            self.dispersion_tensor.hermitian.first_derivative.q,
            self.dispersion_tensor.hermitian.second_derivative.q,
            self.arguments.first_derivative,
            self.arguments.second_derivative,
            (Dimensions.x.size, Dimensions.x.size),
            Dimensions.q.size,
            Dimensions.z.size,
            return_array=self.dispersion_tensor.hermitian.second_derivative.z,
        )

        if derivatives == 2:  # noqa: PLR2004
            return

        raise ValueError(derivatives)

    def calculate_stix_polarisation(
        self,
        plasma_cache: PlasmaCache,
        wave_cache: WaveCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate stix_polarisation as null space of dispersion tensor.
        Do not call if dispersion tensor is degenerate (e.g. in vacuum) as
        the stix_polarisation will be in-determinate.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Cache containing plasma parameter data.
        wave_cache : WaveCache
            Cache containing wave parameter data.
        derivatives : int
            Order of derivatives to set.

        Raises
        ------
        ValueError
            Derivatives < 0 or > 1.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Calculate polarisation in Stix frame.
        polarisation(
            self.dispersion_tensor.hermitian.value,
            return_array=self.polarisation.stix.value,
            nearby_polarisation=self.polarisation.stix.value,
        )

        # Calculate Cartesian polarisation components.
        self.calculate_cartesian_polarisation_from_stix(wave_cache)

        if derivatives == 0:
            return

        # First derivative of Stix polarisation.
        if plasma_cache.unmagnetised:
            # If unmagnetised there are always 2 null eigenvalues which
            # causes divide by zero in the eigenvector derivative.
            # Anyway unmagnetised polarisation is constant.
            self.polarisation.stix.first_derivative.q.fill(0.0)
            self.polarisation.stix.first_derivative.z.fill(0.0)
        else:
            eigenvector_dx(
                self.eigenvalues,
                self.eigenvectors,
                self.dispersion_tensor.hermitian.first_derivative.q,
                0,
                return_array=self.polarisation.stix.first_derivative.q,
            )

            first_derivative(
                self.polarisation.stix.first_derivative.q,
                self.arguments.first_derivative,
                (Dimensions.x.size,),
                Dimensions.q.size,
                Dimensions.z.size,
                return_array=self.polarisation.stix.first_derivative.z,
            )

        if derivatives == 1:
            return

        raise ValueError(derivatives)

    def calculate_vacuum_stix_polarisation(
        self,
        wave_mode: WaveMode,
        plasma_cache: PlasmaCache,
        wave_cache: WaveCache,
    ):
        """
        Calculate stix polarisation of O and X mode in vacuum.

        Parameters
        ----------
        wave_mode : WaveMode
            Wave mode to solve for.
        plasma_cache : PlasmaCache
            Cache containing plasma parameter data.
        wave_cache : WaveCache
            Cache containing wave parameter data.

        Raises
        ------
        ValueError
            Not in vacuum.
        """
        if not plasma_cache:
            raise ValueError("Not in vacuum.")

        # Calculate vacuum limit of eigenmode polarisation.
        vacuum_stix_polarisation(
            self.arguments.value[_Y],
            self.arguments.value[_N_PERP],
            self.arguments.value[_N_PARALLEL],
            wave_mode,
            return_array=self.polarisation.stix.value,
        )

        # Calculate Cartesian polarisation components.
        self.calculate_cartesian_polarisation_from_stix(wave_cache)

        # Polarisation does not evolve in vacuum.
        self.polarisation.stix.first_derivative.q.fill(0.0)
        self.polarisation.stix.first_derivative.z.fill(0.0)

    def calculate_cartesian_polarisation_from_stix(
        self, wave_cache: WaveCache
    ):
        """
        Calculate Cartesian polarisation from Stix polarisation.

        Parameters
        ----------
        wave_cache : WaveCache
            Cache containing wave parameter data.
        """
        wave_cache.stix_to_cartesian(
            self.polarisation.stix.value,
            return_array=self.polarisation.cartesian,
        )

    def calculate_stix_polarisation_from_cartesian(
        self, wave_cache: WaveCache
    ):
        """
        Calculate Stix polarisation from Cartesian polarisation.

        Parameters
        ----------
        wave_cache : WaveCache
            Cache containing wave parameter data.
        """
        wave_cache.cartesian_to_stix(
            self.polarisation.cartesian,
            return_array=self.polarisation.stix.value,
        )

    def calculate_eigenvalue(self, /, *, derivatives: int):
        """
        Calculate eigenvalue for given mode and derivative with respect to
        Hamiltonian arguments q and extended phase space z. Also calculates
        deviation of eigenvalue from zero expressed as a frequency.

        Parameters
        ----------
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Eigenvalue of polarisation.
        self.eigenvalue.real = eigenvalue(
            self.dispersion_tensor.hermitian.value,
            self.polarisation.stix.value,
        ).real

        self.eigenvalue.imag_resonance = eigenvalue(
            self.dispersion_tensor.antihermitian_resonance,
            self.polarisation.stix.value,
        ).imag

        self.eigenvalue.imag_collisional = eigenvalue(
            self.dispersion_tensor.antihermitian_collisional,
            self.polarisation.stix.value,
        ).imag

        self.eigenvalue.imag = (
            self.eigenvalue.imag_resonance + self.eigenvalue.imag_collisional
        )

        if derivatives == 0:
            return

        # First derivative of real part of eigenvalue.
        self.eigenvalue.first_derivative.q[:] = eigenvalue_dx(
            self.dispersion_tensor.hermitian.first_derivative.q,
            self.polarisation.stix.value,
        ).real

        first_derivative(
            self.eigenvalue.first_derivative.q,
            self.arguments.first_derivative,
            (),
            Dimensions.q.size,
            Dimensions.z.size,
            return_array=self.eigenvalue.first_derivative.z,
        )

        # Error in Hamiltonian expressed as a frequency.
        dx_df = self.eigenvalue.first_derivative.z[Dimensions.slice_f]

        if np.isclose(dx_df, 0.0):
            self.eigenvalue_error_frequency = 0.0
        else:
            self.eigenvalue_error_frequency = self.eigenvalue.real / dx_df

        if derivatives == 1:
            return

        # Second derivative of real part of eigenvalue.
        self.eigenvalue.second_derivative.q[:, :] = eigenvalue_dx2(
            self.dispersion_tensor.hermitian.first_derivative.q,
            self.dispersion_tensor.hermitian.second_derivative.q,
            self.polarisation.stix.value,
            self.polarisation.stix.first_derivative.q,
        ).real

        second_derivative(
            self.eigenvalue.first_derivative.q,
            self.eigenvalue.second_derivative.q,
            self.arguments.first_derivative,
            self.arguments.second_derivative,
            (),
            Dimensions.q.size,
            Dimensions.z.size,
            return_array=self.eigenvalue.second_derivative.z,
        )

        if derivatives == 2:  # noqa: PLR2004
            return

        raise ValueError(derivatives)

    def calculate_determinant(self, /, *, derivatives: int):
        """
        Calculate determinant of dispersion tensor and derivative with respect
        to Hamiltonian arguments q and extended phase space z. Also calculates
        deviation of determinant from zero expressed as a frequency.

        Parameters
        ----------
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        d = determinant(
            self.dispersion_tensor.hermitian.value
            + self.dispersion_tensor.antihermitian_resonance
            + self.dispersion_tensor.antihermitian_collisional
        )

        self.determinant.real = d.real
        self.determinant.imag = d.imag

        if derivatives == 0:
            return

        # First derivative of determinant of Hermitian part.
        dispersion_tensor_adjugate = adjugate_3x3_cofactors(
            self.dispersion_tensor.hermitian.value
        )

        self.determinant.first_derivative.q[:] = determinant_dx(
            self.dispersion_tensor.hermitian.value,
            self.dispersion_tensor.hermitian.first_derivative.q,
            dispersion_tensor_adjugate=dispersion_tensor_adjugate,
        ).real

        first_derivative(
            self.determinant.first_derivative.q,
            self.arguments.first_derivative,
            (),
            Dimensions.q.size,
            Dimensions.z.size,
            return_array=self.determinant.first_derivative.z,
        )

        # Error in Hamiltonian expressed as a frequency.
        dx_df = self.determinant.first_derivative.z[Dimensions.slice_f].real

        if dx_df == 0:
            self.determinant_error_frequency = 0.0
        else:
            self.determinant_error_frequency = self.determinant.real / dx_df

        if derivatives == 1:
            return

        # Second derivative of determinant.
        self.determinant.second_derivative.q[:, :] = determinant_dx2(
            self.dispersion_tensor.hermitian.value,
            self.dispersion_tensor.hermitian.first_derivative.q,
            self.dispersion_tensor.hermitian.second_derivative.q,
            dispersion_tensor_adjugate=dispersion_tensor_adjugate,
        ).real

        second_derivative(
            self.determinant.first_derivative.q,
            self.determinant.second_derivative.q,
            self.arguments.first_derivative,
            self.arguments.second_derivative,
            (),
            Dimensions.q.size,
            Dimensions.z.size,
            return_array=self.determinant.second_derivative.z,
        )

        if derivatives == 2:  # noqa: PLR2004
            return

        raise ValueError(derivatives)

    def calculate_normalised_em_flux(self, frequency_ghz: float):
        """
        Calculate normalised electromagnetic flux i.e. fraction of wave energy
        being carried in electromagnetic fields. The remaining energy is in
        coherent motion of charged particles.

        Parameters
        ----------
        frequency_ghz : float
            Wave frequency [GHz].
        """
        d_dx = first_derivative(
            self.dispersion_tensor.hermitian.first_derivative.q,
            self.arguments.first_derivative[:, Dimensions.slice_x],
            (Dimensions.x.size, Dimensions.x.size),
            Dimensions.q.size,
            Dimensions.x.size,
        )

        self.normalised_em_energy_density = (
            0.5
            * frequency_ghz
            * eigenvalue_dx(d_dx, self.polarisation.stix.value)
        )
