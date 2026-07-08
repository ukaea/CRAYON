"""
Caches for wave parameters.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import (
    component_parallel,
    component_perp,
    rotation_a_onto_b,
    v_parallel_first_derivative_v,
    v_parallel_first_derivative_x,
    v_parallel_second_derivative_v,
    v_parallel_second_derivative_x,
    v_parallel_second_derivative_xv,
    v_perp_first_derivative_v,
    v_perp_first_derivative_x,
    v_perp_second_derivative_v,
    v_perp_second_derivative_x,
    v_perp_second_derivative_xv,
)
from crayon.ray_tracing.caches.base import (
    DerivativeCache,
    DerivativeCacheXK,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.physics import vacuum_wavelength_m, vacuum_wavenumber_per_m
from crayon.shared.types import Array, FloatArray

logger = logging.getLogger(__name__)

K = Dimensions.k.size


class WaveCache:
    """
    Cache for wave parameters.

    Attributes
    ----------
    frequency_ghz : float
        Wave frequency [GHz].
    k_parallel : float
        Wavevector component parallel to magnetic field.
    k_perp : DerivativeCacheXK
        Wavevector component perpendicular to magnetic field and derivatives
        with respect to phase space xk.
    n_parallel : float
        Refractive index component parallel to magnetic field.
    n_perp : DerivativeCacheXK
        Refractive index component perpendicular to magnetic field and
        derivatives with respect to phase space xk.
    refractive_index : np.array[float]
        Refractive index vector.
    vacuum_wavelength_m : float
        Vacuum wavelength [m].
    vacuum_wavenumber_per_m : float
        Vacuum wavenumber [m^-1].
    wavevector_per_m : np.array[float]
        Wavevector [m^-1].

    Methods
    -------
    set_frequency
        Set wave frequency.
    set_refractive_index
        Set refractive index and calculate wavevector.
    set_wavevector
        Set wavevector and calculate refractive index.
    calculate_k_components
        Calculate components of the wavevector parallel and perpendicular to
        the magnetic field and derivatives with respect to phase space xk.
    cartesian_to_stix
        Transform vector from Cartesian into Stix frame.
    stix_to_cartesian
        Transform vector from Stix frame into Cartesian.
    """

    __slots__ = (
        "_stix_frame_rotation",
        "frequency_ghz",
        "k_parallel",
        "k_perp",
        "n_parallel",
        "n_perp",
        "refractive_index",
        "vacuum_wavelength_m",
        "vacuum_wavenumber_per_m",
        "wavevector_per_m",
    )

    def __init__(self):
        """
        Inits WaveCache.
        """
        self.frequency_ghz = 0.0
        self.vacuum_wavenumber_per_m = 0.0
        self.vacuum_wavelength_m = 0.0

        self.wavevector_per_m = np.empty(K, dtype=float)
        self.refractive_index = np.empty(K, dtype=float)

        self.k_perp = DerivativeCacheXK(())
        self.k_parallel = DerivativeCacheXK(())
        self.n_perp = 0.0
        self.n_parallel = 0.0

        self._stix_frame_rotation = np.empty((
            Dimensions.x.size,
            Dimensions.x.size,
        ))

    def set_frequency(
        self,
        frequency_ghz: float,
    ):
        """
        Set wave frequency.

        Parameters
        ----------
        frequency_ghz : float
            Wave frequency [GHz].
        """
        self.frequency_ghz = frequency_ghz
        self.vacuum_wavenumber_per_m = vacuum_wavenumber_per_m(
            self.frequency_ghz
        )
        self.vacuum_wavelength_m = vacuum_wavelength_m(frequency_ghz)

    def set_refractive_index(self, refractive_index: FloatArray):
        """
        Set refractive index and calculate wavevector.

        Parameters
        ----------
        refractive_index : np.array[float]
            Cartesian refractive index vector.
        """
        self.refractive_index[:] = refractive_index
        self.wavevector_per_m[:] = (
            self.vacuum_wavenumber_per_m * self.refractive_index
        )

    def set_wavevector(self, wavevector: FloatArray):
        """
        Set wavevector and calculate refractive index.

        Parameters
        ----------
        wavevector : np.array[float]
            Cartesian wavevector [m^-1].
        """
        self.wavevector_per_m[:] = wavevector
        self.refractive_index[:] = (
            self.wavevector_per_m / self.vacuum_wavenumber_per_m
        )

    def calculate_k_components(
        self, magnetic_field_unit: DerivativeCache, /, *, derivatives: int
    ):
        """
        Calculate components of the wavevector parallel and perpendicular to
        the magnetic field and derivatives with respect to phase space xk.

        Parameters
        ----------
        magnetic_field_unit: DerivativeCache
            Magnetic field unit vector and derivatives with respect to x.
        derivatives: int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Slices into arrays for x and k.
        _x, _k = Dimensions.slice_x, Dimensions.slice_k

        # Parallel wavenumber.
        self.k_parallel.value[...] = component_parallel(
            self.wavevector_per_m,
            magnetic_field_unit.value,
        )

        # Perpendicular wavenumber.
        self.k_perp.value[...] = component_perp(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            v_parallel=self.k_parallel.value,
        )

        # Parallel and perpendicular refractive index components.
        self.n_parallel = self.k_parallel.value / self.vacuum_wavenumber_per_m
        self.n_perp = self.k_perp.value / self.vacuum_wavenumber_per_m

        # Calculate rotation matrix for Stix basis i.e. rotation that maps
        # N -> [N_perp, 0.0, N_parallel].
        self._stix_frame_rotation[:, :] = rotation_a_onto_b(
            self.refractive_index,
            np.array([self.n_perp, 0.0, self.n_parallel]),
        )

        if derivatives == 0:
            return

        # First derivative of k_parallel wrt (x, k).
        v_parallel_first_derivative_x(
            self.wavevector_per_m,
            magnetic_field_unit.first_derivative,
            return_array=self.k_parallel.first_derivative[_x],
        )

        v_parallel_first_derivative_v(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            return_array=self.k_parallel.first_derivative[_k],
        )

        # First derivative of k_perp wrt (x, k).
        v_perp_first_derivative_x(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            magnetic_field_unit.first_derivative,
            v_perp=self.k_perp.value,
            v_parallel=self.k_parallel.value,
            v_parallel_dx=self.k_parallel.first_derivative[_x],
            return_array=self.k_perp.first_derivative[_x],
        )

        v_perp_first_derivative_v(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            v_perp=self.k_perp.value,
            v_parallel=self.k_parallel.value,
            return_array=self.k_perp.first_derivative[_k],
        )

        if derivatives == 1:
            return

        # Second derivative of k_parallel wrt (x, k).
        v_parallel_second_derivative_x(
            self.wavevector_per_m,
            magnetic_field_unit.second_derivative,
            return_array=self.k_parallel.second_derivative[_x, _x],
        )

        v_parallel_second_derivative_xv(
            self.wavevector_per_m,
            magnetic_field_unit.first_derivative,
            return_array=self.k_parallel.second_derivative[_x, _k],
        )

        # Partial second derivatives commute.
        self.k_parallel.second_derivative[_k, _x] = (
            self.k_parallel.second_derivative[_x, _k].T
        )

        v_parallel_second_derivative_v(
            self.wavevector_per_m,
            return_array=self.k_parallel.second_derivative[_k, _k],
        )

        # Second derivative of k_perp wrt (x, k).
        v_perp_second_derivative_x(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            magnetic_field_unit.first_derivative,
            magnetic_field_unit.second_derivative,
            v_perp=self.k_perp.value,
            v_parallel=self.k_parallel.value,
            v_parallel_dx=self.k_parallel.first_derivative[_x],
            v_parallel_dx2=self.k_parallel.second_derivative[_x, _x],
            return_array=self.k_perp.second_derivative[_x, _x],
        )

        v_perp_second_derivative_xv(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            magnetic_field_unit.first_derivative,
            v_perp=self.k_perp.value,
            v_parallel=self.k_parallel.value,
            v_parallel_dx=self.k_parallel.first_derivative[_x],
            return_array=self.k_perp.second_derivative[
                Dimensions.slice_x, Dimensions.slice_k
            ],
        )

        # Partial second derivatives commute.
        self.k_perp.second_derivative[_k, _x] = self.k_perp.second_derivative[
            _x, _k
        ].T

        v_perp_second_derivative_v(
            self.wavevector_per_m,
            magnetic_field_unit.value,
            v_perp=self.k_perp.value,
            v_parallel=self.k_parallel.value,
            return_array=self.k_perp.second_derivative[_k, _k],
        )

        if derivatives == 2:  # noqa: PLR2004
            return

        # Third derivatives.
        raise ValueError(derivatives)

    def cartesian_to_stix(
        self, vector_cartesian: Array, /, *, return_array: Array = None
    ) -> Array:
        """
        Transform vector from Cartesian into Stix frame (B // z, k_perp // x).

        Parameters
        ----------
        vector_cartesian : np.array
            Vector components in Cartesian global frame.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        vector_stix : np.array
            Vector components in Stix frame.
        """
        return_array = get_return_array(
            return_array, vector_cartesian.shape, vector_cartesian.dtype
        )

        # As unitary, the inverse rotation is the transpose.
        return_array[:] = self._stix_frame_rotation.T @ vector_cartesian

        return return_array

    def stix_to_cartesian(
        self, vector_stix: Array, /, *, return_array: Array = None
    ) -> Array:
        """
        Transform vector from Stix frame (B // z, k_perp // x) into Cartesian.

        Parameters
        ----------
        vector_stix : np.array
            Vector components in Stix frame.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        vector_cartesian : np.array
            Vector components in Cartesian global frame.
        """
        return_array = get_return_array(
            return_array, vector_stix.shape, vector_stix.dtype
        )

        return_array[:] = self._stix_frame_rotation @ vector_stix

        return return_array
