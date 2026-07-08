"""
Caches for linear mode conversion.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.ray_tracing.caches.plasma import PlasmaCache
from crayon.ray_tracing.caches.wave import WaveCache
from crayon.shared.dimensions import Dimensions
from crayon.shared.types import FloatArray

logger = logging.getLogger(__name__)

_x = Dimensions.slice_x
_k = Dimensions.slice_k
_xk = Dimensions.slice_xk

_X = Dimensions.IDX_X
_Y = Dimensions.IDX_Y


def mjolhus_plane_wave(
    normalised_density_gradient_scale_length: float,
    normalised_magnetic_field_strength: float,
    n_parallel: float,
    n_y: float,
) -> float:
    """
    Calculate transmission exponent for OX conversion from a plane wave given
    by Mjolhus [1].

    Parameters
    ----------
    normalised_density_gradient_scale_length : float
        Electron density scale length Ln normalised by vacuum wavenumber k0
        aka k0Ln.
    normalised_magnetic_field_strength : float
        Normalised magnetic field strength aka Stix Y.
    n_parallel : float
        Parallel refractive index.
    n_y : float
        Refractive index component perpendicular to magnetic field and
        density gradient.

    Returns
    -------
    transmission_exponent : float
        Exponent of the transmission factor.

    References
    ----------
    [1] Mjølhus, E. "Coupling to Z mode near critical angle." Journal of
        plasma physics 31.1 (1984): 7-28.
    """
    k0ln = normalised_density_gradient_scale_length
    y = normalised_magnetic_field_strength
    npar_opt = np.sqrt(y / (1 + y))

    return (
        np.pi
        * k0ln
        * np.sqrt(0.5 * y)
        * (2.0 * (1 + y) * np.square(abs(n_parallel) - npar_opt) + n_y * n_y)
    )


def mjolhus_gaussian_beam(
    normalised_density_gradient_scale_length: float,
    normalised_magnetic_field_strength: float,
    n_parallel: float,
    n_y: float,
    normalised_beam_waist_radius: float,
) -> float:
    """
    Mjolhus formula for transmission for a plane wave integrated over the
    spectral density for a Gaussian beam.

    Parameters
    ----------
    normalised_density_gradient_scale_length : float
        Electron density scale length Ln normalised by vacuum wavenumber k0
        aka k0Ln.
    normalised_magnetic_field_strength : float
        Normalised magnetic field strength aka Stix Y.
    n_parallel : float
        Parallel refractive index.
    n_y : float
        Refractive index component perpendicular to magnetic field and
        density gradient.
    normalised_beam_waist_radius : float
        1 / e electric field waist radius W normalised by vacuum wavenumber k0.

    Returns
    -------
    transmission_exponent : float
        Exponent of the transmission factor.
    """
    y = normalised_magnetic_field_strength
    npar_opt = np.sqrt(y / (1 + y))

    w_norm = 0.5 * np.square(normalised_beam_waist_radius)
    l_norm = (
        np.pi * np.sqrt(0.5 * y) * normalised_density_gradient_scale_length
    )

    # Integral related to n_y offset.
    a = w_norm + l_norm
    b = -2 * w_norm * n_y
    c = w_norm * n_y * n_y

    ny_term_exponent = b**2 / (4 * a) - c
    ny_term = (1 / a) ** 0.5 * np.exp(ny_term_exponent)

    # Integral related to n_parallel offset.
    a = w_norm + 2 * (1 + y) * l_norm
    b = -2 * (n_parallel * w_norm + 2 * (1 + y) * l_norm * npar_opt)
    c = (
        w_norm * n_parallel * n_parallel
        + 2 * (1 + y) * l_norm * npar_opt * npar_opt
    )

    nz_term_exponent = b**2 / (4 * a) - c
    nz_term = (1 / a) ** 0.5 * np.exp(nz_term_exponent)

    # Pick up sqrt(pi) from the ny and nz term. This cancels with pi in
    # normalising factor Wnorm / pi for the spectral power.
    return ny_term * nz_term * w_norm


class ModeConversionCache:
    """
    Cache holding information about linear mode conversion.

    Parameters
    ----------
    acceleration_xk : np.array[float]
        Ray acceleration in phase space [m.s^-2, m^-1.s^-2].
    alarm_history : np.array[float]
        Mode conversion alarm values for previous ray steps.
    conversions : int
        Counter of detected mode conversions.
    timeout : int
        Number of ray steps mode conversion calculations are paused for.
    xk_osculating_plane : np.array[float]
        Basis vectors of osculating plane at mode conversion.
    xk_closest : np.array[float]
        Phase space position (x, k) of closest approach to mode conversion.
    xk_saddle : np.array[float]
        Phase space position (x, k) of saddle point.
    k0ln_saddle : float
        Density gradient scale length Ln at saddle point * vacuum wavenumber
        k0.
    y_saddle : float
        Normalised magnetic field strength at saddle point.
    n_parallel : float
        N_parallel at closest approach to mode conversion.
    n_y : float
        N_y (perpendicular to magnetic field and density gradient) at closest
        approach to mode conversion.
    """

    __slots__ = (
        "acceleration_xk",
        "alarm_history",
        "conversions",
        "k0ln_saddle",
        "n_parallel",
        "n_y",
        "timeout",
        "xk_closest",
        "xk_osculating_plane",
        "xk_saddle",
        "y_saddle",
    )

    TIMEOUT = 5

    def __init__(self):
        """
        Inits ModeConversionCache.
        """
        self.alarm_history = np.zeros(4, dtype=float)
        self.acceleration_xk = np.empty(Dimensions.xk.size)
        self.timeout = 0
        self.conversions = 0
        self.xk_osculating_plane = np.empty((2, Dimensions.xk.size))
        self.xk_closest = np.empty(Dimensions.xk.size)
        self.xk_saddle = np.empty(Dimensions.xk.size)
        self.k0ln_saddle = 0.0
        self.y_saddle = 0.0
        self.n_parallel = 0.0
        self.n_y = 0.0

    def update_alarm_value(
        self, alarm_value: float, /, *, override: bool = False
    ):
        """
        Update mode conversion alarm value.

        Parameters
        ----------
        alarm_value : bool
            Value indicating proximity to mode conversion. Must be >= 0.
        override : bool, optional
            If True, override last alarm value. Default = False.
        """
        # Update history of alarm values.
        if not override:
            self.alarm_history[3] = self.alarm_history[2]
            self.alarm_history[2] = self.alarm_history[1]
            self.alarm_history[1] = self.alarm_history[0]

        self.alarm_history[0] = alarm_value

    def mode_conversion_detected(self) -> bool:
        """
        Detect closest approach to mode conversion point.

        Returns
        -------
        mode_conversion_detected : bool
            If at closest approach to mode conversion point.
        """
        if self.timeout > 0:
            self.timeout -= 1
            mode_conversion_detected = False
        else:
            # Check if we are at a local minimum in the mode conversion alarm.
            mode_conversion_detected = (
                self.alarm_history[0] > self.alarm_history[1]
                and self.alarm_history[2] > self.alarm_history[1]
                and self.alarm_history[3] > self.alarm_history[2]
            )

        return mode_conversion_detected

    def save_saddle_plasma_parameters(
        self, plasma_cache: PlasmaCache, wave_cache: WaveCache
    ):
        """
        Save plasma parameters at saddle point used in calculation of mode
        conversion coefficient.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Cache containing plasma data.
        wave_cache : WaveCache
            Cache containing wave data.
        """
        self.y_saddle = abs(
            plasma_cache.normalised_magnetic_field_strength.value
        )
        self.k0ln_saddle = abs(
            wave_cache.vacuum_wavenumber_per_m
            * plasma_cache.normalised_electron_density.value
            / np.linalg.norm(
                plasma_cache.normalised_electron_density.first_derivative
            )
        )

    def save_n_components(
        self, plasma_cache: PlasmaCache, wave_cache: WaveCache
    ):
        """
        Save refractive index components at closest approach to mode
        conversion.

        Parameters
        ----------
        plasma_cache : PlasmaCache
            Cache containing plasma data.
        wave_cache : WaveCache
            Cache containing wave data.
        """
        self.n_parallel = wave_cache.n_parallel

        b_hat = plasma_cache.magnetic_field_unit.value
        ne_dx = plasma_cache.normalised_electron_density.first_derivative
        y_direction = np.cross(b_hat, ne_dx)
        _norm = np.linalg.norm(y_direction)

        if _norm == 0:
            self.n_y = 0.0
        else:
            y_direction /= _norm
            self.n_y = np.dot(wave_cache.refractive_index, y_direction)

    def complete(self):
        """
        Register mode conversion calculations ran successfully.
        """
        self.conversions += 1
        self.timeout = self.TIMEOUT

    @property
    def save_conversion(self) -> bool:
        """
        Flag if mode conversion happened on this step.
        """
        return self.timeout == self.TIMEOUT

    def calculate_acceleration_xk(
        self,
        hamiltonian_real_dz: FloatArray,
        hamiltonian_real_dz2: FloatArray,
    ):
        """
        Calculate acceleration of the ray in phase space.

        Parameters
        ----------
        hamiltonian_real_dz : np.array[float]
            First derivative of ray Hamiltonian with respect to extended phase
            space z = [x, k, f].
        hamiltonian_real_dz2 : np.array[float]
            First derivative of ray Hamiltonian with respect to extended phase
            space z = [x, k, f].
        """
        _x = Dimensions.slice_x
        _k = Dimensions.slice_k
        _f = Dimensions.slice_f

        h_dx = hamiltonian_real_dz[_x]
        h_dk = hamiltonian_real_dz[_k]
        h_df = hamiltonian_real_dz[_f]

        h_dx2 = hamiltonian_real_dz2[_x, _x]
        h_dxdk = hamiltonian_real_dz2[_x, _k]
        h_dxdf = hamiltonian_real_dz2[_x, _f]
        h_dkdk = hamiltonian_real_dz2[_k, _k]
        h_dkdf = hamiltonian_real_dz2[_k, _f]

        # Second derivatives wrt curve parameter tau.
        d2tau_dt2 = np.dot(h_dkdf, h_dx) - np.dot(h_dxdf, h_dk)
        d2x_dtau2 = np.einsum("ji,j -> i", h_dxdk, h_dk) - np.einsum(
            "ij,j -> i", h_dkdk, h_dx
        )
        d2k_dtau2 = np.einsum("ij,j -> i", h_dxdk, h_dx) - np.einsum(
            "ij,j -> i", h_dx2, h_dk
        )

        # Convert from acceleration wrt curve parameter tau to time t [ns].
        h_df_2 = h_df * h_df

        dx_dt = -h_dk / h_df
        dk_dt = h_dx / h_df

        self.acceleration_xk[_x] = (d2x_dtau2 - dx_dt * d2tau_dt2) / h_df_2
        self.acceleration_xk[_k] = (d2k_dtau2 - dk_dt * d2tau_dt2) / h_df_2
