"""
Caches for plasma parameters for ray tracing.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import (
    TensorType,
    first_derivative,
    second_derivative,
    unit_vector,
    unit_vector_first_derivative_x,
    unit_vector_second_derivative_x,
    vector_magnitude,
    vector_magnitude_first_derivative_x,
    vector_magnitude_second_derivative_x,
)
from crayon.ray_tracing.caches.base import (
    DerivativeCacheX,
)
from crayon.ray_tracing.caches.coordinates import CoordinateCache
from crayon.shared.constants import (
    MIN_NORM_ELECTRON_DENSITY,
    MIN_NORM_ELECTRON_TEMPERATURE,
    MIN_NORM_MAGNETIC_FIELD_STRENGTH,
    CoordinateSystem,
)
from crayon.shared.dimensions import Dimensions
from crayon.shared.physics import (
    ELECTRON_REST_MASS_ENERGY_EV,
    critical_density_per_m3,
    critical_magnetic_field_strength_t,
    electron_ion_collision_frequency_first_derivative,
    electron_ion_collision_frequency_ghz,
    electron_ion_collision_frequency_second_derivative,
)
from crayon.shared.types import FloatType
from crayon.system_data import Kinetic, Magnetic
from crayon.value_model import ValueCache, ValueModelBase

logger = logging.getLogger(__name__)

X = Dimensions.x.size


class TwoCoordinateCache:
    """
    Cache for a tensor field with components in a local coordinate system
    and Cartesian.

    Attributes
    ----------
    cartesian : np.array
        Value in Cartesian.
    local : np.array
        Value in local coordinate system.
    local_coordinate_system : CoordinateSystem
        Local coordinate system.
    tensor_type : TensorType
        Tensor type of value.

    Methods
    -------
    calculate
        Compute Cartesian value from local value or vice versa.
    """

    __slots__ = (
        "cartesian",
        "local",
        "local_coordinate_system",
        "tensor_type",
    )

    def __init__(
        self,
        shape: tuple[int],
        tensor_type: TensorType,
        local_coordinate_system: CoordinateSystem,
        dtype: type,
    ):
        """
        Inits TwoCoordinateCache.

        Parameters
        ----------
        shape : tuple[int]
            Shape of value.
        tensor_type : TensorType
            Tensor type of value.
        local_coordinate_system : CoordinateSystem
            Local coordinate system.
        dtype : type
            Data type of value.
        """
        self.tensor_type = tensor_type
        self.local_coordinate_system = local_coordinate_system
        self.cartesian = np.empty(shape, dtype=dtype)
        self.local = np.empty(shape, dtype=dtype)

    def calculate(
        self, coordinate_cache: CoordinateCache, /, *, to_cartesian: bool
    ):
        """
        Compute Cartesian value from local value or vice versa.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        to_cartesian : bool
            If True, calculate Cartesian value from local value and vice versa.
        """
        if to_cartesian:
            coordinate_cache.transform_tensor_field(
                self.local_coordinate_system,
                CoordinateSystem.CARTESIAN,
                self.local,
                self.tensor_type,
                return_array=self.cartesian,
            )
        else:
            coordinate_cache.transform_tensor_field(
                CoordinateSystem.CARTESIAN,
                self.local_coordinate_system,
                self.cartesian,
                self.tensor_type,
                return_array=self.local,
            )


class PlasmaParameterCache:
    """
    Cache for a plasma parameter and its derivatives with respect position x.

    Attributes
    ----------
    cache : ValueCache
        Cache for holding values from model.
    first_derivative : TwoCoordinateCache
        First covariant derivative of plasma parameter with respect to x.
    max_value : float
        Maximum value of model.
    min_value : float
        Minimum value of model.
    model : ValueModelBase
        Object for plasma parameter value as a function of x.
    second_derivative : TwoCoordinateCache
        Second covariant derivative of plasma parameter with respect to x.
    third_derivative : TwoCoordinateCache
        Third covariant derivative of plasma parameter with respect to x.
    value : TwoCoordinateCache
        Value of plasma parameter.

    Methods
    -------
    calculate
        Evaluate model values and derivatives with respect to phase space xk.
    """

    __slots__ = (
        "cache",
        "first_derivative",
        "max_value",
        "min_value",
        "model",
        "second_derivative",
        "third_derivative",
        "value",
    )

    def __init__(
        self,
        model: ValueModelBase,
        tensor_type: TensorType,
        /,
        *,
        min_value: float = -np.inf,
        max_value: float = np.inf,
    ):
        """
        Inits PlasmaParameterCache.

        Parameters
        ----------
        model : ValueModelBase
            Object for plasma parameter value as a function of x.
        tensor_type : TensorType
            Tensor type of value.
        min_value : float
            Minimum value of model.
        max_value : float
            Maximum value of model.
        """
        self.model = model
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        n = model.input_size

        self.value = TwoCoordinateCache(
            model.output_shape,
            tensor_type,
            model.coordinate_system,
            model.dtype,
        )
        self.first_derivative = TwoCoordinateCache(
            (*model.output_shape, n),
            tensor_type.first_derivative,
            model.coordinate_system,
            model.dtype,
        )
        self.second_derivative = TwoCoordinateCache(
            (*model.output_shape, n, n),
            tensor_type.second_derivative,
            model.coordinate_system,
            model.dtype,
        )
        self.third_derivative = TwoCoordinateCache(
            (*model.output_shape, n, n, n),
            tensor_type.third_derivative,
            model.coordinate_system,
            model.dtype,
        )

        # Cache for efficient evaluation of value model.
        self.cache = ValueCache.for_model(model)

    def calculate(
        self,
        coordinate_cache: CoordinateCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Evaluate model values and derivatives with respect to phase space xk.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Calculate model values.
        position = coordinate_cache.position[self.model.coordinate_system]
        self.model.fill_cache(self.cache, position, derivatives=derivatives)

        # Clip value within allowed range.
        self.value.local[...] = np.clip(
            self.cache.value, self.min_value, self.max_value
        )

        # Transform value to Cartesian.
        self.value.calculate(coordinate_cache, to_cartesian=True)

        if derivatives == 0:
            return

        # Calculate first derivative.
        coordinate_cache.first_covariant_derivative(
            self.model.coordinate_system,
            self.value.local,
            self.cache.jacobian,
            self.value.tensor_type,
            return_array=self.first_derivative.local,
        )

        # Transform first derivative to Cartesian.
        self.first_derivative.calculate(coordinate_cache, to_cartesian=True)

        if derivatives == 1:
            return

        # Calculate second derivative.
        coordinate_cache.second_covariant_derivative(
            self.model.coordinate_system,
            self.value.local,
            self.cache.jacobian,
            self.cache.hessian,
            self.first_derivative.local,
            self.value.tensor_type,
            return_array=self.second_derivative.local,
        )

        # Transform second derivative to Cartesian.
        self.second_derivative.calculate(coordinate_cache, to_cartesian=True)

        if derivatives == 2:  # noqa : PLR2004
            return

        raise ValueError(derivatives)

        # Calculate third derivative.
        coordinate_cache.third_covariant_derivative(
            self.model.coordinate_system,
            self.value.local,
            self.cache.jacobian,
            self.cache.hessian,
            self.cache.jerk,
            self.first_derivative.local,
            self.second_derivative.local,
            self.value.tensor_type,
            return_array=self.third_derivative.local,
        )

        # Transform third derivative to Cartesian.
        self.third_derivative.calculate(coordinate_cache, to_cartesian=True)


class PlasmaCache:
    """
    Cache for plasma parameter data.

    Attributes
    ----------
    cold : bool
        Flag if normalised electron temperature low enough to consider plasma
        cold.
    effective_charge : PlasmaParameterCache
        Effective charge and derivatives with respect to x.
    electron_density_per_m3 : PlasmaParameterCache
        Electron density [m^-3] and derivatives with respect to x.
    electron_temperature_ev : PlasmaParameterCache
        Electron temperature [eV] and derivatives with respect to x.
    magnetic_field_strength_t : DerivativeCacheX
        Magnetic field strength [T] and derivatives with respect to x.
    magnetic_field_t : PlasmaParameterCache
        Magnetic field vector [T] and derivatives with respect to x.
    magnetic_field_unit : DerivativeCacheX
        Magnetic field unit vector and derivatives with respect to x.
    normalised_collision_rate : DerivativeCacheX
        Normalised electron ion collision rate aka Stix Z and derivatives with
        respect to x.
    normalised_electron_density : DerivativeCacheX
        Normalised electron density aka Stix X and derivatives with respect to
        x.
    normalised_electron_temperature : DerivativeCacheX
        Normalised electron temperature theta and derivatives with respect to
        x.
    normalised_magnetic_field_strength : DerivativeCacheX
        Normalised magnetic field strength aka Stix Y and derivatives with
        respect to x.
    unmagnetised : bool
        Flag if normalised magnetic field strength low enough to consider
        plasma unmagnetised.
    vacuum : bool
        Flag if normalised electron density low enough to consider in vacuum.

    Methods
    -------
    calculate
        Calculate all plasma parameters and derivatives with respect to x.
    set_frequency
        Set wave frequency for normalised plasma parameters.
    calculate_electron_density
        Calculate electron density and derivatives with respect to x.
    calculate_electron_temperature
        Calculate electron temperature and derivatives with respect to x.
    calculate_effective_charge
        Calculate effective charge and derivatives with respect to x.
    calculate_magnetic_field
        Calculate magnetic field and derivatives with respect to x.
    calculate_normalised_collision_rate
        Calculate normalised electron-ion collision rate and derivatives with
        respect to x.
    """

    __slots__ = (
        "_inv_critical_damping_frequency_ghz",
        "_inv_critical_density_per_m3",
        "_inv_critical_magnetic_field_strength_t",
        "_inv_critical_temperature_ev",
        "cold",
        "effective_charge",
        "electron_density_per_m3",
        "electron_temperature_ev",
        "magnetic_field_strength_t",
        "magnetic_field_t",
        "magnetic_field_unit",
        "normalised_collision_rate",
        "normalised_electron_density",
        "normalised_electron_temperature",
        "normalised_magnetic_field_strength",
        "unmagnetised",
        "vacuum",
    )

    def __init__(self, kinetic: Kinetic, magnetic: Magnetic):
        """
        Inits PlasmaCache.

        Parameters
        ----------
        kinetic : Kinetic
            Kinetic plasma parameter models.
        magnetic : Magnetic
            Magnetic plasma parameter models.
        """
        self.electron_density_per_m3 = PlasmaParameterCache(
            kinetic.electron_density_per_m3, TensorType.SCALAR, min_value=0.0
        )
        self.electron_temperature_ev = PlasmaParameterCache(
            kinetic.electron_temperature_ev, TensorType.SCALAR, min_value=0.0
        )
        self.effective_charge = PlasmaParameterCache(
            kinetic.effective_charge, TensorType.SCALAR, min_value=0.0
        )
        self.magnetic_field_t = PlasmaParameterCache(
            magnetic.magnetic_field_t,
            TensorType.VECTOR,
        )

        self.magnetic_field_strength_t = DerivativeCacheX(())
        self.magnetic_field_unit = DerivativeCacheX((X,))
        self.normalised_electron_density = DerivativeCacheX(())
        self.normalised_electron_temperature = DerivativeCacheX(())
        self.normalised_collision_rate = DerivativeCacheX(())
        self.normalised_magnetic_field_strength = DerivativeCacheX(())

        self._inv_critical_density_per_m3 = 0.0
        self._inv_critical_temperature_ev = 0.0
        self._inv_critical_magnetic_field_strength_t = 0.0
        self._inv_critical_damping_frequency_ghz = 0.0

        self.vacuum = False
        self.cold = False
        self.unmagnetised = False

    def calculate(
        self, coordinate_cache: CoordinateCache, /, *, derivatives: int
    ):
        """
        Calculate all plasma parameters and derivatives with respect to x.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        derivatives : int
            Number of derivatives to evaluate.
        """
        self.calculate_electron_density(
            coordinate_cache, derivatives=derivatives
        )
        self.calculate_electron_temperature(
            coordinate_cache, derivatives=derivatives
        )
        self.calculate_effective_charge(
            coordinate_cache, derivatives=derivatives
        )
        self.calculate_magnetic_field(
            coordinate_cache, derivatives=derivatives
        )
        self.calculate_normalised_collision_rate(derivatives=derivatives)

    def set_frequency(self, frequency_ghz: float):
        """
        Set wave frequency for normalised plasma parameters.

        Parameters
        ----------
        frequency_ghz : float
            Wave frequency [GHz]. Must be positive.

        Raises
        ------
        ValueError
            frequency_ghz is not positive.
        """
        if frequency_ghz <= 0.0:
            raise ValueError("frequency_ghz must be positive.")

        self._inv_critical_density_per_m3 = 1.0 / critical_density_per_m3(
            frequency_ghz
        )
        self._inv_critical_temperature_ev = 1.0 / ELECTRON_REST_MASS_ENERGY_EV
        self._inv_critical_magnetic_field_strength_t = (
            1.0 / critical_magnetic_field_strength_t(frequency_ghz)
        )
        self._inv_critical_damping_frequency_ghz = 1.0 / frequency_ghz

    def calculate_electron_density(
        self,
        coordinate_cache: CoordinateCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate electron density and derivatives with respect to x.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 3.
            set_frequency not called first.

        Notes
        -----
        set_frequency must be called before this function.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        if self._inv_critical_density_per_m3 <= 0.0:
            raise ValueError(
                "Critical electron density <= 0. Did you call set_frequency?"
            )

        # Evaluate electron density model.
        self.electron_density_per_m3.calculate(
            coordinate_cache,
            derivatives=derivatives,
        )

        # Calculate normalised density.
        self.normalised_electron_density.value[...] = (
            self._inv_critical_density_per_m3
            * self.electron_density_per_m3.value.cartesian
        )

        self.vacuum = (
            self.normalised_electron_density.value <= MIN_NORM_ELECTRON_DENSITY
        )

        if derivatives == 0:
            return

        # Calculate normalised density first derivative.
        self.normalised_electron_density.first_derivative[:] = (
            self._inv_critical_density_per_m3
            * self.electron_density_per_m3.first_derivative.cartesian
        )

        if derivatives == 1:
            return

        # Calculate normalised density second derivatives.
        self.normalised_electron_density.second_derivative[:, :] = (
            self._inv_critical_density_per_m3
            * self.electron_density_per_m3.second_derivative.cartesian
        )

        if derivatives == 2:  # noqa : PLR2004
            return

        # Calculate normalised density third derivative.
        self.normalised_electron_density.third_derivative[:, :, :] = (
            self._inv_critical_density_per_m3
            * self.electron_density_per_m3.third_derivative.cartesian
        )

        if derivatives == 3:  # noqa : PLR2004
            return

        raise ValueError(derivatives)

    def calculate_electron_temperature(
        self,
        coordinate_cache: CoordinateCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate electron temperature and derivatives with respect to x.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 3.
            set_frequency not called first.

        Notes
        -----
        set_frequency must be called before this function.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        if self._inv_critical_temperature_ev <= 0.0:
            raise ValueError(
                "Critical electron temperature <= 0. Did you call "
                "set_frequency?"
            )

        # Evaluate electron temperature model.
        self.electron_temperature_ev.calculate(
            coordinate_cache,
            derivatives=derivatives,
        )

        # Calculate normalised temperature.
        self.normalised_electron_temperature.value[...] = (
            self._inv_critical_temperature_ev
            * self.electron_temperature_ev.value.cartesian
        )

        self.cold = (
            self.normalised_electron_temperature.value
            <= MIN_NORM_ELECTRON_TEMPERATURE
        )

        if derivatives == 0:
            return

        # Calculate normalised temperature first derivative.
        self.normalised_electron_temperature.first_derivative[:] = (
            self._inv_critical_temperature_ev
            * self.electron_temperature_ev.first_derivative.cartesian
        )

        if derivatives == 1:
            return

        # Calculate normalised temperature second derivatives.
        self.normalised_electron_temperature.second_derivative[:, :] = (
            self._inv_critical_temperature_ev
            * self.electron_temperature_ev.second_derivative.cartesian
        )

        if derivatives == 2:  # noqa : PLR2004
            return

        # Calculate normalised temperature third derivative.
        self.normalised_electron_temperature.third_derivative[:, :, :] = (
            self._inv_critical_temperature_ev
            * self.electron_temperature_ev.third_derivative.cartesian
        )

        if derivatives == 3:  # noqa : PLR2004
            return

        raise ValueError(derivatives)

    def calculate_effective_charge(
        self,
        coordinate_cache: CoordinateCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate effective charge and derivatives with respect to x.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        # Evaluate effective charge model.
        self.effective_charge.calculate(
            coordinate_cache,
            derivatives=derivatives,
        )

    def calculate_magnetic_field(
        self,
        coordinate_cache: CoordinateCache,
        /,
        *,
        derivatives: int,
    ):
        """
        Calculate magnetic field vector and derivatives with respect to x.

        Parameters
        ----------
        coordinate_cache : CoordinateCache
            Cache containing coordinate system data.
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
            set_frequency not called first.

        Notes
        -----
        set_frequency must be called before this function.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        if self._inv_critical_magnetic_field_strength_t <= 0.0:
            raise ValueError(
                "Critical magnetic field strength <= 0. Did you call "
                "set_frequency?"
            )

        # Evaluate magnetic field model.
        self.magnetic_field_t.calculate(
            coordinate_cache,
            derivatives=derivatives,
        )

        # Calculate magnetic field strength.
        self.magnetic_field_strength_t.value[...] = vector_magnitude(
            self.magnetic_field_t.value.cartesian
        )

        # Take signed value i.e. for electrons < 0.
        # Allows using generalised formulas for susceptibility.
        self.normalised_magnetic_field_strength.value[...] = -(
            self._inv_critical_magnetic_field_strength_t
            * self.magnetic_field_strength_t.value
        )

        self.unmagnetised = (
            abs(self.normalised_magnetic_field_strength.value)
            <= MIN_NORM_MAGNETIC_FIELD_STRENGTH
        )

        # Calculate magnetic field unit vector.
        unit_vector(
            self.magnetic_field_t.value.cartesian,
            v_magnitude=self.magnetic_field_strength_t.value,
            return_array=self.magnetic_field_unit.value,
        )

        if derivatives == 0:
            return

        # Calculate first derivative.
        vector_magnitude_first_derivative_x(
            self.magnetic_field_t.value.cartesian,
            self.magnetic_field_t.first_derivative.cartesian,
            v_magnitude=self.magnetic_field_strength_t.value,
            return_array=self.magnetic_field_strength_t.first_derivative,
        )

        self.normalised_magnetic_field_strength.first_derivative[...] = -(
            self._inv_critical_magnetic_field_strength_t
            * self.magnetic_field_strength_t.first_derivative
        )

        unit_vector_first_derivative_x(
            self.magnetic_field_t.value.cartesian,
            self.magnetic_field_t.first_derivative.cartesian,
            v_magnitude=self.magnetic_field_strength_t.value,
            v_magnitude_dx=self.magnetic_field_strength_t.first_derivative,
            v_unit=self.magnetic_field_unit.value,
            return_array=self.magnetic_field_unit.first_derivative,
        )

        if derivatives == 1:
            return

        # Calculate second derivative.
        vector_magnitude_second_derivative_x(
            self.magnetic_field_t.value.cartesian,
            self.magnetic_field_t.first_derivative.cartesian,
            self.magnetic_field_t.second_derivative.cartesian,
            v_magnitude=self.magnetic_field_strength_t.value,
            v_magnitude_dx=self.magnetic_field_strength_t.first_derivative,
            return_array=self.magnetic_field_strength_t.second_derivative,
        )

        self.normalised_magnetic_field_strength.second_derivative[...] = -(
            self._inv_critical_magnetic_field_strength_t
            * self.magnetic_field_strength_t.second_derivative
        )

        unit_vector_second_derivative_x(
            self.magnetic_field_t.value.cartesian,
            self.magnetic_field_t.first_derivative.cartesian,
            self.magnetic_field_t.second_derivative.cartesian,
            v_magnitude=self.magnetic_field_strength_t.value,
            v_magnitude_dx=self.magnetic_field_strength_t.first_derivative,
            v_magnitude_dx2=self.magnetic_field_strength_t.second_derivative,
            v_unit=self.magnetic_field_unit.value,
            v_unit_dx=self.magnetic_field_unit.first_derivative,
            return_array=self.magnetic_field_unit.second_derivative,
        )

        if derivatives == 2:  # noqa : PLR2004
            return

        # Calculate third derivative.
        raise ValueError(derivatives)

    def calculate_normalised_collision_rate(self, /, *, derivatives: int):
        """
        Calculate normalised electron-ion collision rate and derivatives with
        respect to x.

        Parameters
        ----------
        derivatives : int
            Number of derivatives to evaluate.

        Raises
        ------
        ValueError
            derivatives < 0 or > 2.
            set_frequency not called first.

        Notes
        -----
        set_frequency must be called before this function.
        """
        if derivatives < 0:
            raise ValueError(derivatives)

        if self._inv_critical_damping_frequency_ghz <= 0.0:
            raise ValueError(
                "Critical electron density <= 0. Did you call set_frequency?"
            )

        _norm_inv = self._inv_critical_damping_frequency_ghz
        ne = self.electron_density_per_m3
        te = self.electron_temperature_ev
        zeff = self.effective_charge

        # Calculate value.
        self.normalised_collision_rate.value[...] = (
            self._inv_critical_damping_frequency_ghz
            * electron_ion_collision_frequency_ghz(
                self.electron_density_per_m3.value.cartesian,
                self.electron_temperature_ev.value.cartesian,
                self.effective_charge.value.cartesian,
            )
        )

        if derivatives == 0:
            return

        # Calculate first derivative.
        z_dq = np.empty(3, dtype=FloatType)
        dq_dx = np.empty((3, Dimensions.x.size), dtype=FloatType)

        electron_ion_collision_frequency_first_derivative(
            ne.value.cartesian,
            te.value.cartesian,
            zeff.value.cartesian,
            return_array=z_dq,
        )

        # Convert to derivatives of Z = nu / f.
        z_dq *= _norm_inv

        # Derivatives of [ne, te, zeff] wrt space.
        dq_dx[0, :] = ne.first_derivative.cartesian
        dq_dx[1, :] = te.first_derivative.cartesian
        dq_dx[2, :] = zeff.first_derivative.cartesian

        first_derivative(
            z_dq,
            dq_dx,
            (),
            3,
            Dimensions.x.size,
            return_array=self.normalised_collision_rate.first_derivative,
        )

        if derivatives == 1:
            return

        # Calculate second derivative.
        z_dq2 = np.empty((3, 3), dtype=FloatType)
        d2q_dx2 = np.empty(
            (3, Dimensions.x.size, Dimensions.x.size), dtype=FloatType
        )

        # Convert to derivatives of Z = nu / f.
        z_dq2 *= _norm_inv

        # Derivatives of [ne, te, zeff] wrt space.
        d2q_dx2[0, :, :] = ne.second_derivative.cartesian
        d2q_dx2[1, :, :] = te.second_derivative.cartesian
        d2q_dx2[2, :, :] = zeff.second_derivative.cartesian

        electron_ion_collision_frequency_second_derivative(
            ne.value.cartesian,
            te.value.cartesian,
            zeff.value.cartesian,
            return_array=z_dq2,
        )

        second_derivative(
            z_dq,
            z_dq2,
            dq_dx,
            d2q_dx2,
            (),
            3,
            Dimensions.x.size,
            return_array=self.normalised_collision_rate.second_derivative,
        )

        if derivatives == 2:  # noqa : PLR2004
            return

        # Calculate third derivative.
        raise ValueError(derivatives)
