"""
Useful functions for physics parameters.
"""

# Standard imports
import logging

# Third party imports
import numpy as np
import scipy.constants as const

# Local imports
from crayon.shared.constants import TWO_PI, TWO_PI_INV
from crayon.shared.helpers import get_return_array
from crayon.shared.types import FloatArray, FloatType

logger = logging.getLogger(__name__)


def vacuum_wavelength_m(frequency_ghz: float) -> float:
    """
    Calculate vacuum wavelength for given frequency [m].

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency [GHz].

    Returns
    -------
    vacuum_wavelength_m : float
        Vacuum wavelength [m].
    """
    return 1e-9 * const.speed_of_light / frequency_ghz


def vacuum_wavenumber_per_m(frequency_ghz: float) -> float:
    """
    Calculate vacuum wavenumber for given frequency [m^-1].

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency [GHz].

    Returns
    -------
    vacuum_wavenumber_per_m : float
        Vacuum wavenumber [m^-1].
    """
    return 1e9 * TWO_PI * frequency_ghz / const.speed_of_light


def refractive_index(frequency_ghz: float, wavenumber_per_m: float) -> float:
    """
    Calculate refractive index from wavenumber.

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency [GHz].
    wavenumber_per_m : float
        Wavenumber [m^-1].

    Returns
    -------
    refractive_index : float
        Refractive index.
    """
    k0 = vacuum_wavenumber_per_m(frequency_ghz)
    return wavenumber_per_m / k0


_cyclotron_frequency_ghz_per_t = (
    1e-9 * abs(const.elementary_charge / const.electron_mass) * TWO_PI_INV
)


def cyclotron_frequency_ghz(magnetic_field_strength_t: float) -> float:
    """
    Calculate electron cyclotron frequency [GHz].

    Parameters
    ----------
    magnetic_field_strength_t : float
        Magnetic field strength [T].

    Returns
    -------
    cyclotron_frequency_ghz : float
        Cyclotron frequency [GHz].
    """
    return _cyclotron_frequency_ghz_per_t * magnetic_field_strength_t


def critical_magnetic_field_strength_t(cyclotron_frequency_ghz: float):
    """
    Calculate required magnetic field strength [T] to generate a given
    cyclotron frequency [GHz].

    Parameters
    ----------
    cyclotron_frequency_ghz : float
        Cyclotron frequency [GHz].

    Returns
    -------
    critical_magnetic_field_strength_t : float
        Critical magnetic field strength [T].
    """
    return cyclotron_frequency_ghz / _cyclotron_frequency_ghz_per_t


plasma_frequency_ghz_per_sqrt_m3 = (
    1e-9
    * abs(const.elementary_charge)
    / np.sqrt(const.epsilon_0 * const.electron_mass)
    * TWO_PI_INV
)


def plasma_frequency_ghz(density_per_m3: float) -> float:
    """
    Calculate electron plasma frequency [GHz].

    Parameters
    ----------
    density_per_m3 : float
        Electron density [m^-3].

    Returns
    -------
    plasma_frequency_ghz : float
        Plasma frequency [GHz].
    """
    return plasma_frequency_ghz_per_sqrt_m3 * np.sqrt(density_per_m3)


def critical_density_per_m3(plasma_frequency_ghz: float) -> float:
    """
    Calculate required electron density [m^-3] to generate a given plasma
    frequency [GHz].

    Parameters
    ----------
    plasma_frequency_ghz : float
        Plasma frequency [GHz].

    Returns
    -------
    critical_density_per_m3 : float
        Critical electron density [m^-3].
    """
    return np.square(plasma_frequency_ghz / plasma_frequency_ghz_per_sqrt_m3)


ELECTRON_REST_MASS_ENERGY_EV = (
    const.electron_mass * const.speed_of_light**2 / const.elementary_charge
)

PROTON_REST_MASS_ENERGY_EV = (
    const.proton_mass * const.speed_of_light**2 / const.elementary_charge
)


def normalised_electron_energy(electron_energy_ev: float) -> float:
    """
    Electron energy normalised to rest mass energy 511keV [].

    Parameters
    ----------
    electron_energy_ev : float
        Electron kinetic energy [eV].

    Returns
    -------
    normalised_electron_energy : float
        Normalised electron energy theta.
    """
    return electron_energy_ev / ELECTRON_REST_MASS_ENERGY_EV


def lorentz_factor_from_energy(electron_energy_ev: float) -> float:
    """
    Calculate Lorentz factor gamma [] for an electron in terms of energy [eV].

    Parameters
    ----------
    electron_energy_ev : float
        Electron kinetic energy [eV].

    Returns
    -------
    lorentz_factor : float
        Lorentz factor gamma.
    """
    theta = normalised_electron_energy(electron_energy_ev)
    return lorentz_factor_from_normalised_energy(theta)


def lorentz_factor_from_normalised_energy(
    normalised_electron_energy: float,
) -> float:
    """
    Calculate Lorentz factor gamma [] for an electron in terms of energy as a
    normalised to the rest mass energy theta = E / (m0 * c**2).

    Parameters
    ----------
    normalised_electron_energy : float
        Electron kinetic energy normalised to rest mass energy.

    Returns
    -------
    lorentz_factor : float
        Lorentz factor gamma.
    """
    return 1 + normalised_electron_energy


def lorentz_factor_from_normalised_momentum(
    normalised_momentum: float,
) -> float:
    """
    Calculate Lorentz factor gamma [] for an electron in terms of momentum
    normalised to the rest mass momentum q = p / (m0 * c).

    Parameters
    ----------
    normalised_momentum : float
        Electron momentum normalised to rest mass momentum.

    Returns
    -------
    lorentz_factor : float
        Lorentz factor gamma.
    """
    q = normalised_momentum
    return (1 + q**2) ** 0.5


def normalised_momentum_from_energy(electron_energy_ev: float) -> float:
    """
    Calculate electron momentum normalised to rest mass momentum
    q = p / (m0 * c) in terms of the electron energy [eV].

    Parameters
    ----------
    electron_energy_ev : float
        Electron kinetic energy [eV].

    Returns
    -------
    normalised_momentum : float
        Electron momentum normalised to rest mass momentum.
    """
    theta = normalised_electron_energy(electron_energy_ev)
    return normalised_momentum_from_normalised_energy(theta)


def normalised_momentum_from_normalised_energy(
    normalised_electron_energy: float,
) -> float:
    """
    Calculate electron momentum normalised to rest mass momentum
    q = p / (m0 * c) in terms of the electron energy normalised to the rest
    mass energy theta = E / (m0 * c**2).

    Parameters
    ----------
    normalised_electron_energy : float
        Electron kinetic energy normalised to rest mass energy.

    Returns
    -------
    normalised_momentum : float
        Electron momentum normalised to rest mass momentum.
    """
    theta = normalised_electron_energy
    return np.sqrt(theta * (2 + theta))


def normalised_velocity(
    normalised_momentum: float, lorentz_factor: float
) -> float:
    """
    Calculate velocity beta = v_thermal / c for a relativistic particle.

    Parameters
    ----------
    normalised_momentum : float
        Electron momentum normalised to rest mass momentum.
    lorentz_factor : float
        Lorentz factor gamma.

    Returns
    -------
    normalised_velocity : float
        Velocity normalised to speed of light.
    """
    return normalised_momentum / lorentz_factor


def normalised_electron_energy_from_normalised_velocity(
    normalised_velocity: float,
) -> float:
    """
    Calculate the normalised electron energy from the velocity normalised to
    the speed of light [].

    Parameters
    ----------
    normalised_velocity : float
        Velocity normalised to speed of light.

    Returns
    -------
    normalised_electron_energy : float
        Electron kinetic energy normalised to rest mass energy.
    """
    beta2 = normalised_velocity * normalised_velocity
    return np.sqrt(1 + (beta2 / (1 - beta2))) - 1


def normalised_thermal_velocity_nonrelativistic(
    temperature_ev: float, /, *, half: bool = False, theta: float | None = None
) -> float:
    """
    Calculate thermal velocity beta = v_thermal / c for a Maxwell
    distribution []. Only valid for non-relativistic distribution.

    Parameters
    ----------
    temperature_ev : float
        Electron temperature [eV].
    half : bool, optional
        If True, use sqrt(theta) instead of sqrt(2 * theta). Default = False.
    theta : float, optional.
        Electron kinetic energy normalised to rest mass energy.

    Returns
    -------
    normalised_thermal_velocity : float
        Thermal velocity normalised to speed of light.
    """
    if theta is None:
        theta = normalised_electron_energy(temperature_ev)

    if half:
        return np.sqrt(theta)

    return np.sqrt(2 * theta)


def normalised_thermal_velocity_relativistic(temperature_ev: float) -> float:
    """
    Calculate thermal velocity beta = v_thermal / c for a Maxwell Juttner
    distribution []. This is valid for relativistic electrons for small
    Te/mc^2. Taken from Karney 1985.

    Parameters
    ----------
    temperature_ev : float
        Electron temperature [eV].

    Returns
    -------
    normalised_thermal_velocity : float
        Thermal velocity normalised to speed of light.
    """
    theta = normalised_electron_energy(temperature_ev)
    return np.sqrt(3 * theta * (1 - 2.5 * theta + 6.875 * theta * theta))


_debye_radius_const = (const.epsilon_0 / const.elementary_charge) ** 0.5


def debye_radius_m(density_per_m3: float, temperature_ev: float) -> float:
    """
    Calculate Debye radius [m].

    Parameters
    ----------
    density_per_m3 : float
        Electron density [m^-3].
    temperature_ev : float
        Electron temperature [eV].

    Returns
    -------
    debye_radius_m : float
        Debye radius [m].
    """
    return _debye_radius_const * (temperature_ev / density_per_m3) ** 0.5


ln1000 = np.log(1.0e3)


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
    if temperature_ev < 10 * effective_charge**2:
        return (
            23
            + ln1000
            + 1.5 * np.log(temperature_ev)
            - 0.5 * np.log(density_per_m3)
        )
    return 24 + ln1000 + np.log(temperature_ev) - 0.5 * np.log(density_per_m3)


_electron_ion_collision_frequency_ghz_const = (
    1.0e-9
    * (2**0.5 / 12 / np.pi**1.5)
    * const.elementary_charge**2.5
    / const.electron_mass**0.5
    / const.epsilon_0**2
)


def electron_ion_collision_frequency_ghz(
    density_per_m3: float, temperature_ev: float, effective_charge: float
) -> float:
    """
    Calculate electron-ion collision frequency [GHz].

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
    electron_ion_collision_frequency_ghz : float
        Electron ion collision frequency [GHz].
    """
    if (
        density_per_m3 <= 0.0
        or temperature_ev <= 0.0
        or effective_charge <= 0.0
    ):
        return 0.0

    _const = _electron_ion_collision_frequency_ghz_const
    ne, te, zeff = density_per_m3, temperature_ev, effective_charge

    # Avoid unphysical negative value at extremely low temperatures.
    c_log = max(0.0, coulomb_logarithm(ne, te, zeff))

    return _const * c_log * (ne * zeff / te**1.5)


def zeff_from_electron_ion_collision_frequency_ghz(
    collision_frequency_ghz: float,
    density_per_m3: float,
    temperature_ev: float,
) -> float:
    """
    Calculate Zeff from electron-ion collision frequency, density and
    temperature.

    Parameters
    ----------
    collision_frequency_ghz : float
        Electron ion collision frequency [GHz].
    density_per_m3 : float
        Electron density [m^-3].
    temperature_ev : float
        Electron temperature [eV].

    Returns
    -------
    effective_charge : float
        Effective charge.
    """
    if density_per_m3 <= 0.0 or temperature_ev <= 0.0:
        return 0.0

    log_lambda_c = coulomb_logarithm(density_per_m3, temperature_ev)
    return np.sqrt(
        collision_frequency_ghz
        / (
            _electron_ion_collision_frequency_ghz_const
            * log_lambda_c
            * density_per_m3
            / temperature_ev**1.5
        )
    )


def electron_ion_collision_frequency_first_derivative(
    density_per_m3: float,
    temperature_ev: float,
    effective_charge: float,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Return first derivative of electron ion collision frequency with
    respect to the parameter vector q = [ne, te, zeff].

    Parameters
    ----------
    density_per_m3 : float
        Electron density [m^-3].
    temperature_ev : float
        Electron temperature [eV].
    effective_charge : float
        Effective charge.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    electron_ion_collision_frequency_ghz_dq : float
        First derivative of electron ion collision frequency [GHz].
    """
    return_array = get_return_array(return_array, (3,), FloatType)

    # If density or temperature is zero this causes issues so return zero.
    if density_per_m3 <= 0.0 or temperature_ev <= 0.0:
        return_array.fill(0.0)
        return return_array

    _const = _electron_ion_collision_frequency_ghz_const
    ne, te, zeff = density_per_m3, temperature_ev, effective_charge

    c_log = coulomb_logarithm(ne, te, zeff)
    c_log_dne = -0.5 / ne
    c_log_dte = 1 / te

    if te < 10 * zeff * zeff:
        c_log_dte *= 1.5

    # Density derivative.
    return_array[0] = _const * zeff / te**1.5 * (c_log + ne * c_log_dne)

    # Temperature derivative.
    return_array[1] = (
        _const * ne * zeff * ((c_log_dte - 1.5 * c_log / te) / te**1.5)
    )

    # Effective charge derivative.
    return_array[2] = _const * c_log * ne / te**1.5

    return return_array


def electron_ion_collision_frequency_second_derivative(
    density_per_m3: float,
    temperature_ev: float,
    effective_charge: float,
    /,
    *,
    return_array: FloatArray = None,
) -> FloatArray:
    """
    Return second derivative of electron ion collision frequency with
    respect to the parameter vector q = [ne, te, zeff].

    Parameters
    ----------
    density_per_m3 : float
        Electron density [m^-3].
    temperature_ev : float
        Electron temperature [eV].
    effective_charge : float
        Effective charge.
    return_array : np.array[float], optional
        Array into which the result is stored. If provided, must have same
        shape as output array. If not provided, an new array is allocated.

    Returns
    -------
    electron_ion_collision_frequency_ghz_dq2 : float
        Second derivative of electron ion collision frequency [GHz].
    """
    return_array = get_return_array(return_array, (3, 3), FloatType)

    # If density or temperature is zero this causes issues so return zero.
    if density_per_m3 <= 0.0 or temperature_ev <= 0.0:
        return_array.fill(0.0)
        return return_array

    _const = _electron_ion_collision_frequency_ghz_const
    ne, te, zeff = density_per_m3, temperature_ev, effective_charge
    te2 = te * te

    # Mixed derivatives are zero.
    c_log = coulomb_logarithm(ne, te, zeff)
    c_log_dne = -0.5 / ne
    c_log_dte = 1 / te
    c_log_dne2 = 0.5 / (ne * ne)
    c_log_dte2 = -1 / te2

    if te < 10 * zeff * zeff:
        c_log_dte *= 1.5
        c_log_dte2 *= 1.5

    return_array[0, 0] = (
        _const * zeff / te**1.5 * (2 * c_log_dne + ne * c_log_dne2)
    )

    return_array[0, 1] = (
        _const
        * zeff
        * ((c_log_dte - 1.5 * (c_log + ne * c_log_dne) / te) / te**1.5)
    )

    return_array[0, 2] = _const * (c_log + ne * c_log_dne) / te**1.5

    return_array[1, 1] = (
        _const
        * ne
        * zeff
        * (c_log_dte2 - 3.0 * c_log_dte / te + 3.75 * c_log / te2)
        / te**1.5
    )

    return_array[1, 2] = _const * ne / te**1.5 * (c_log_dte - 1.5 * c_log / te)

    return_array[2, 2] = 0.0

    # Second partial derivatives commute.
    return_array[1, 0] = return_array[0, 1]
    return_array[2, 0] = return_array[0, 2]
    return_array[2, 1] = return_array[1, 2]

    return return_array
