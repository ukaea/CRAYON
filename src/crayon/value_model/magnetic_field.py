"""
Special magnetic field models.
"""

# Standard imports
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.calculus import (
    first_derivative,
    fourth_derivative,
    second_derivative,
    third_derivative,
)
from crayon.coordinates import CoordinateSystem
from crayon.shared.constants import TWO_PI_INV
from crayon.shared.dimensions import Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.types import BooleanArray, FloatArray, FloatType
from crayon.value_model.base import ModelType, ValueModelBase
from crayon.value_model.splines import Spline2D

logger = logging.getLogger(__name__)


class AxisymmetricMagneticField(ValueModelBase):
    """
    Axisymmetric cylindrical magnetic field.

    Attributes
    ----------
    f_toroidal_tm : ValueModelBase
        Model for diamagnetic function F.
    rho_poloidal : Spline2D
        Spline of root normalised poloidal flux rho (r, z) as function of
        cylindrical position.
    total_poloidal_flux_wb : float
        Total poloidal flux between magnetic axis and separatrix.
    """

    __slots__ = (
        "f_toroidal_tm",
        "rho_poloidal",
        "total_poloidal_flux_wb",
    )

    model_type = ModelType.AXISYMMETRIC_MAGNETIC_FIELD

    def __init__(
        self,
        rho_poloidal: Spline2D,
        f_toroidal_tm: ValueModelBase,
        total_poloidal_flux_wb: float,
        /,
        *,
        scale_factor: float = 1.0,
    ):
        """
        Cylindrical axisymmetric magnetic field.

        Parameters
        ----------
        rho_poloidal : Spline2D
            Spline of root normalised poloidal flux rho (r, z) as function of
            cylindrical position.
        f_toroidal_tm : ValueModelBase
            Model for diamagnetic function F.
        total_poloidal_flux_wb : float
            Total poloidal flux between magnetic axis and separatrix. This is
            total flux i.e. NOT flux / 2 pi.
        scale_factor : float, optional
            Scale factor for model.

        Raises
        ------
        ValueError
            rho_poloidal does not use cylindrical coordinate system.
            f_toroidal_tm does not use rho poloidal coordinate system.
        """
        self.rho_poloidal = rho_poloidal
        self.f_toroidal_tm = f_toroidal_tm

        if self.rho_poloidal.coordinate_system != CoordinateSystem.CYLINDRICAL:
            raise ValueError(
                "rho_poloidal must take coordinate system "
                f"{CoordinateSystem.CYLINDRICAL.name}. "
                f"Got {self.rho_poloidal.coordinate_system}"
            )

        if (
            self.f_toroidal_tm.coordinate_system
            != CoordinateSystem.RHO_POLOIDAL
        ):
            raise ValueError(
                "f_toroidal_tm must take coordinate system "
                f"{CoordinateSystem.RHO_POLOIDAL.name}. "
                f"Got {self.f_toroidal_tm.coordinate_system}"
            )

        # Total poloidal flux between magnetic axis and separatrix [Wb].
        # NOTE: this is total flux i.e. NOT flux / 2 pi.
        self.total_poloidal_flux_wb = float(total_poloidal_flux_wb)

        # init super class.
        super().__init__(
            CoordinateSystem.CYLINDRICAL,
            Dimensions.x,
            (Dimensions.x,),
            "T",
            scale_factor=scale_factor,
            input_bounds=self.rho_poloidal.input_bounds,
            dtype=float,
        )

    def _psi_norm(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
    ) -> FloatArray:
        """
        Normalised poloidal flux.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.

        Returns
        -------
        psi_norm : np.array[float]
            Normalised poloidal flux.
        """
        _n = abscissa.shape[0]

        return_array = get_return_array(
            return_array,
            (_n,),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        return_array[...] = np.square(rho)

        return return_array

    def _psi_norm_dx(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
    ) -> FloatArray:
        """
        First derivative of normalised poloidal with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.

        Returns
        -------
        psi_norm_dx : np.array[float]
            First derivative of normalised poloidal flux.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        psi_drho = 2 * rho

        first_derivative(
            psi_drho.reshape((_n, 1)),
            rho_dx.reshape((_n, 1, _x)),
            (),
            1,
            Dimensions.x.size,
            return_array=return_array,
        )

        return return_array

    def _psi_norm_dx2(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
        rho_dx2: FloatArray = None,
    ) -> FloatArray:
        """
        Second derivative of normalised poloidal with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.
        rho_dx2 : np.array[float], optional
            Second derivative of root normalised poloidal flux with respect to
            position.

        Returns
        -------
        psi_norm_dx2 : np.array[float]
            Second derivative of normalised poloidal flux.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x, _x),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        if rho_dx2 is None:
            rho_dx2 = self.rho_poloidal.hessian(abscissa)

        psi_drho = 2 * rho
        psi_drho2 = np.full(rho.shape, 2.0)

        second_derivative(
            psi_drho.reshape((_n, 1)),
            psi_drho2.reshape((_n, 1, 1)),
            rho_dx.reshape((_n, 1, _x)),
            rho_dx2.reshape((_n, 1, _x, _x)),
            (),
            1,
            Dimensions.x.size,
            return_array=return_array,
        )

        return return_array

    def _psi_norm_dx3(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
        rho_dx2: FloatArray = None,
        rho_dx3: FloatArray = None,
    ) -> FloatArray:
        """
        Third derivative of normalised poloidal with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.
        rho_dx2 : np.array[float], optional
            Second derivative of root normalised poloidal flux with respect to
            position.
        rho_dx3 : np.array[float], optional
            Third derivative of root normalised poloidal flux with respect to
            position.

        Returns
        -------
        psi_norm_dx3 : np.array[float]
            Third derivative of normalised poloidal flux.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x, _x, _x),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        if rho_dx2 is None:
            rho_dx2 = self.rho_poloidal.hessian(abscissa)

        if rho_dx3 is None:
            rho_dx3 = self.rho_poloidal.jerk(abscissa)

        psi_drho = 2 * rho
        psi_drho2 = np.full(rho.shape, 2.0)
        psi_drho3 = np.full(rho.shape, 0.0)

        third_derivative(
            psi_drho.reshape((_n, 1)),
            psi_drho2.reshape((_n, 1, 1)),
            psi_drho3.reshape((_n, 1, 1, 1)),
            rho_dx.reshape((_n, 1, _x)),
            rho_dx2.reshape((_n, 1, _x, _x)),
            rho_dx3.reshape((_n, 1, _x, _x, _x)),
            (),
            1,
            Dimensions.x.size,
            return_array=return_array,
        )

        return return_array

    def _psi_norm_dx4(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
        rho_dx2: FloatArray = None,
        rho_dx3: FloatArray = None,
        rho_dx4: FloatArray = None,
    ) -> FloatArray:
        """
        Fourth derivative of normalised poloidal with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.
        rho_dx2 : np.array[float], optional
            Second derivative of root normalised poloidal flux with respect to
            position.
        rho_dx3 : np.array[float], optional
            Third derivative of root normalised poloidal flux with respect to
            position.
        rho_dx4 : np.array[float], optional
            Fourth derivative of root normalised poloidal flux with respect to
            position.

        Returns
        -------
        psi_norm_dx4 : np.array[float]
            Fourth derivative of normalised poloidal flux.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x, _x, _x, _x),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        if rho_dx2 is None:
            rho_dx2 = self.rho_poloidal.hessian(abscissa)

        if rho_dx3 is None:
            rho_dx3 = self.rho_poloidal.jerk(abscissa)

        if rho_dx4 is None:
            rho_dx4 = self.rho_poloidal.snap(abscissa)

        psi_drho = 2 * rho
        psi_drho2 = np.full(rho.shape, 2.0)
        psi_drho3 = np.full(rho.shape, 0.0)
        psi_drho4 = psi_drho3

        fourth_derivative(
            psi_drho.reshape((_n, 1)),
            psi_drho2.reshape((_n, 1, 1)),
            psi_drho3.reshape((_n, 1, 1, 1)),
            psi_drho4.reshape((_n, 1, 1, 1, 1)),
            rho_dx.reshape((_n, 1, _x)),
            rho_dx2.reshape((_n, 1, _x, _x)),
            rho_dx3.reshape((_n, 1, _x, _x, _x)),
            rho_dx4.reshape((_n, 1, _x, _x, _x, _x)),
            (),
            1,
            Dimensions.x.size,
            return_array=return_array,
        )

        return return_array

    def _f_toroidal(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
    ) -> FloatArray:
        """
        Diamagnetic function F.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.

        Returns
        -------
        f_toroidal : np.array[float]
            Normalised poloidal flux.
        """
        _n = abscissa.shape[0]

        return_array = get_return_array(
            return_array,
            (_n,),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        # Interpolator expects array of shape (n, 3) but will only use first
        # column so an array of shape (n, 1) also works. This is an ugly hack
        # but avoids creating a pointless empty array.
        self.f_toroidal_tm.value(
            rho.reshape((-1, 1)), return_array=return_array
        )

        return return_array

    def _f_toroidal_dx(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
        f_toroidal_drho: FloatArray = None,
    ) -> FloatArray:
        """
        First derivative of diamagnetic function F with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.
        f_toroidal_drho : np.array[float], optional
            First derivative of diamagnetic function F with respect to rho.

        Returns
        -------
        f_toroidal_dx : np.array[float]
            First derivative of diamagnetic function F.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x),
            abscissa.dtype,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        if f_toroidal_drho is None:
            f_toroidal_drho = self.f_toroidal_tm.jacobian(rho.reshape(-1, 1))[
                :, 0
            ]

        first_derivative(
            f_toroidal_drho.reshape((_n, 1)),
            rho_dx.reshape((_n, 1, _x)),
            (),
            1,
            _x,
            return_array=return_array,
        )

        return return_array

    def _f_toroidal_dx2(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
        rho_dx2: FloatArray = None,
        f_toroidal_drho: FloatArray = None,
        f_toroidal_drho2: FloatArray = None,
    ) -> FloatArray:
        """
        Second derivative of diamagnetic function F with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.
        rho_dx2 : np.array[float], optional
            Second derivative of root normalised poloidal flux with respect to
            position.
        f_toroidal_drho : np.array[float], optional
            First derivative of diamagnetic function F with respect to rho.
        f_toroidal_drho2 : np.array[float], optional
            Second derivative of diamagnetic function F with respect to rho.

        Returns
        -------
        f_toroidal_dx2 : np.array[float]
            Second derivative of diamagnetic function F.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x, _x),
            FloatType,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        if rho_dx2 is None:
            rho_dx2 = self.rho_poloidal.hessian(abscissa)

        if f_toroidal_drho is None:
            f_toroidal_drho = self.f_toroidal_tm.jacobian(rho.reshape(-1, 1))[
                :, 0
            ]

        if f_toroidal_drho2 is None:
            f_toroidal_drho2 = self.f_toroidal_tm.hessian(rho.reshape(-1, 1))[
                :, 0, 0
            ]

        second_derivative(
            f_toroidal_drho.reshape((_n, 1)),
            f_toroidal_drho2.reshape((_n, 1, 1)),
            rho_dx.reshape((_n, 1, _x)),
            rho_dx2.reshape((_n, 1, _x, _x)),
            (),
            1,
            _x,
            return_array=return_array,
        )

        return return_array

    def _f_toroidal_dx3(
        self,
        abscissa: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
        rho: FloatArray = None,
        rho_dx: FloatArray = None,
        rho_dx2: FloatArray = None,
        rho_dx3: FloatArray = None,
        f_toroidal_drho: FloatArray = None,
        f_toroidal_drho2: FloatArray = None,
        f_toroidal_drho3: FloatArray = None,
    ) -> FloatArray:
        """
        Third derivative of diamagnetic function F with respect to position.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        rho_dx : np.array[float], optional
            First derivative of root normalised poloidal flux with respect to
            position.
        rho_dx2 : np.array[float], optional
            Second derivative of root normalised poloidal flux with respect to
            position.
        rho_dx3 : np.array[float], optional
            Third derivative of root normalised poloidal flux with respect to
            position.
        f_toroidal_drho : np.array[float], optional
            First derivative of diamagnetic function F with respect to rho.
        f_toroidal_drho2 : np.array[float], optional
            Second derivative of diamagnetic function F with respect to rho.
        f_toroidal_drho3 : np.array[float], optional
            Third derivative of diamagnetic function F with respect to rho.

        Returns
        -------
        f_toroidal_dx3 : np.array[float]
            Third derivative of diamagnetic function F.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        return_array = get_return_array(
            return_array,
            (_n, _x, _x, _x),
            FloatType,
        )

        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        if rho_dx is None:
            rho_dx = self.rho_poloidal.jacobian(abscissa)

        if rho_dx2 is None:
            rho_dx2 = self.rho_poloidal.hessian(abscissa)

        if rho_dx3 is None:
            rho_dx3 = self.rho_poloidal.jerk(abscissa)

        if f_toroidal_drho is None:
            f_toroidal_drho = self.f_toroidal_tm.jacobian(rho.reshape(-1, 1))[
                :, 0
            ]

        if f_toroidal_drho2 is None:
            f_toroidal_drho2 = self.f_toroidal_tm.hessian(rho.reshape(-1, 1))[
                :, 0, 0
            ]

        if f_toroidal_drho3 is None:
            f_toroidal_drho3 = self.f_toroidal_tm.jerk(rho.reshape(-1, 1))[
                :, 0, 0, 0
            ]

        third_derivative(
            f_toroidal_drho.reshape((_n, 1)),
            f_toroidal_drho2.reshape((_n, 1, 1)),
            f_toroidal_drho3.reshape((_n, 1, 1, 1)),
            rho_dx.reshape((_n, 1, _x)),
            rho_dx2.reshape((_n, 1, _x, _x)),
            rho_dx3.reshape((_n, 1, _x, _x, _x)),
            (),
            1,
            _x,
            return_array=return_array,
        )

        return return_array

    def value_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        rho: FloatArray = None,
        psi_norm_dx: FloatArray = None,
        f_toroidal: FloatArray = None,
    ):
        """
        Calculate magnetic field vector in holonomic basis.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        out_of_bounds : np.array[bool]
            Boolean array if input data in bounds of model.
        return_array : np.array[float]
            Array into which the result is stored.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        psi_norm_dx : np.array[float], optional
            First derivative of normalised poloidal flux.
        f_toroidal : np.array[float], optional
            Diamagnetic function F.

        Notes
        -----
        Evaluated in the holonomic basis so Bphi is divided by radius
        compared to physical basis.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        # Zero return array.
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        # Root normalised poloidal flux
        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        # Normalised poloidal flux first derivative wrt space.
        if psi_norm_dx is None:
            psi_norm_dx = self._psi_norm_dx(abscissa, rho=rho)
        else:
            psi_norm_dx = np.reshape(psi_norm_dx, (_n, _x))

        # F toroidal function.
        if f_toroidal is None:
            f_toroidal = self._f_toroidal(abscissa, rho=rho)
        else:
            f_toroidal = np.reshape(f_toroidal, (_n,))

        # Poloidal field components.
        r = abscissa[:, 0]
        _psi = self.total_poloidal_flux_wb * TWO_PI_INV

        psi_norm_dr = psi_norm_dx[:, 0]
        psi_norm_dz = psi_norm_dx[:, 2]

        # B_r
        return_array[:, 0] = _psi * psi_norm_dz / r

        # B_z
        return_array[:, 2] = -_psi * psi_norm_dr / r

        # Toroidal field component B_phi.
        # NOTE: This is in the holonomic basis so there is extra 1 / r.
        return_array[:, 1] = f_toroidal / np.square(r)

    def jacobian_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        rho: FloatArray = None,
        psi_norm_dx2: FloatArray = None,
        f_toroidal_dx: FloatArray = None,
        magnetic_field_t: FloatArray = None,
    ):
        """
        Calculate first derivative of magnetic field vector with respect to
        cylindrical position in holonomic basis.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        out_of_bounds : np.array[bool]
            Boolean array if input data in bounds of model.
        return_array : np.array[float]
            Array into which the result is stored.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        psi_norm_dx2 : np.array[float], optional
            Second derivative of normalised poloidal flux.
        f_toroidal_dx : np.array[float], optional
            First derivative of diamagnetic function F.
        magnetic_field_t : np.array[float], optional
            Magnetic field vector [T].
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        # Zero return array.
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        # Root normalised poloidal flux
        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        # Normalised poloidal flux second derivative wrt space.
        if psi_norm_dx2 is None:
            psi_norm_dx2 = self._psi_norm_dx2(abscissa, rho=rho)
        else:
            psi_norm_dx2 = np.reshape(psi_norm_dx2, (_n, _x, _x))

        # F toroidal first derivative wrt space.
        if f_toroidal_dx is None:
            f_toroidal_dx = self._f_toroidal_dx(abscissa, rho=rho)
        else:
            f_toroidal_dx = np.reshape(f_toroidal_dx, (_n, _x))

        if magnetic_field_t is None:
            magnetic_field_t = np.empty((_n, _x))

            self.value_function(
                abscissa, out_of_bounds, magnetic_field_t, rho=rho
            )
        else:
            magnetic_field_t = np.reshape(magnetic_field_t, (_n, _x))

        # Poloidal field components.
        r = abscissa[in_bounds, 0]
        _psi = self.total_poloidal_flux_wb * TWO_PI_INV

        psi_norm_dr2 = psi_norm_dx2[in_bounds, 0, 0]
        psi_norm_drdz = psi_norm_dx2[in_bounds, 0, 2]
        psi_norm_dz2 = psi_norm_dx2[in_bounds, 2, 2]

        b_r = magnetic_field_t[in_bounds, 0]
        b_z = magnetic_field_t[in_bounds, 2]

        # B_r.
        return_array[in_bounds, 0, 0] = (_psi * psi_norm_drdz - b_r) / r
        return_array[in_bounds, 0, 2] = _psi * psi_norm_dz2 / r

        # B_z.
        return_array[in_bounds, 2, 0] = -(_psi * psi_norm_dr2 + b_z) / r
        return_array[in_bounds, 2, 2] = -_psi * psi_norm_drdz / r

        # Toroidal field component B_phi.
        r2 = np.square(r)

        b_phi = magnetic_field_t[in_bounds, 1]

        f_dr = f_toroidal_dx[in_bounds, 0]
        f_dz = f_toroidal_dx[in_bounds, 2]

        # NOTE: This is in the holonomic basis so there is extra 1 / r.
        return_array[in_bounds, 1, 0] = (f_dr - 2 * r * b_phi) / r2
        return_array[in_bounds, 1, 2] = f_dz / r2

    def hessian_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        rho: FloatArray = None,
        psi_norm_dx3: FloatArray = None,
        f_toroidal_dx2: FloatArray = None,
        magnetic_field_t: FloatArray = None,
        magnetic_field_dx: FloatArray = None,
    ):
        """
        Calculate second derivative of magnetic field vector with respect to
        cylindrical position in holonomic basis.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        out_of_bounds : np.array[bool]
            Boolean array if input data in bounds of model.
        return_array : np.array[float]
            Array into which the result is stored.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        psi_norm_dx3 : np.array[float], optional
            Third derivative of normalised poloidal flux.
        f_toroidal_dx2 : np.array[float], optional
            Second derivative of diamagnetic function F.
        magnetic_field_t : np.array[float], optional
            Magnetic field vector [T].
        magnetic_field_dx : np.array[float], optional
            First derivative of magnetic field vector.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        # Zero return array.
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        # Root normalised poloidal flux
        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        # Normalised poloidal flux third derivative wrt space.
        if psi_norm_dx3 is None:
            psi_norm_dx3 = self._psi_norm_dx3(abscissa, rho=rho)
        else:
            psi_norm_dx3 = np.reshape(psi_norm_dx3, (_n, _x, _x, _x))

        # F toroidal second derivative wrt space.
        if f_toroidal_dx2 is None:
            f_toroidal_dx2 = self._f_toroidal_dx2(abscissa, rho=rho)
        else:
            f_toroidal_dx2 = np.reshape(f_toroidal_dx2, (_n, _x, _x))

        if magnetic_field_t is None:
            magnetic_field_t = np.empty((_n, _x))

            self.value_function(
                abscissa, out_of_bounds, magnetic_field_t, rho=rho
            )
        else:
            magnetic_field_t = np.reshape(magnetic_field_t, (_n, _x))

        if magnetic_field_dx is None:
            magnetic_field_dx = np.empty((_n, _x, _x))

            self.jacobian_function(
                abscissa,
                out_of_bounds,
                magnetic_field_dx,
                rho=rho,
                magnetic_field_t=magnetic_field_t,
            )
        else:
            magnetic_field_dx = np.reshape(magnetic_field_dx, (_n, _x, _x))

        # Poloidal field components.
        r = abscissa[in_bounds, 0]
        _psi = self.total_poloidal_flux_wb * TWO_PI_INV

        psi_norm_dr3 = psi_norm_dx3[in_bounds, 0, 0, 0]
        psi_norm_dr2dz = psi_norm_dx3[in_bounds, 0, 0, 2]
        psi_norm_drdz2 = psi_norm_dx3[in_bounds, 0, 2, 2]
        psi_norm_dz3 = psi_norm_dx3[in_bounds, 2, 2, 2]

        b_r_dr = magnetic_field_dx[in_bounds, 0, 0]
        b_r_dz = magnetic_field_dx[in_bounds, 0, 2]
        b_z_dr = magnetic_field_dx[in_bounds, 2, 0]
        b_z_dz = magnetic_field_dx[in_bounds, 2, 2]

        # B_r.
        return_array[in_bounds, 0, 0, 0] = (
            _psi * psi_norm_dr2dz - 2 * b_r_dr
        ) / r
        return_array[in_bounds, 0, 0, 2] = (_psi * psi_norm_drdz2 - b_r_dz) / r
        return_array[in_bounds, 0, 2, 0] = return_array[:, 0, 0, 2]
        return_array[in_bounds, 0, 2, 2] = _psi * psi_norm_dz3 / r

        # B_z.
        return_array[in_bounds, 2, 0, 0] = (
            -(_psi * psi_norm_dr3 + 2 * b_z_dr) / r
        )
        return_array[in_bounds, 2, 0, 2] = (
            -(_psi * psi_norm_dr2dz + b_z_dz) / r
        )
        return_array[in_bounds, 2, 2, 0] = return_array[:, 2, 0, 2]
        return_array[in_bounds, 2, 2, 2] = -_psi * psi_norm_drdz2 / r

        # Toroidal field component B_phi.
        r2 = np.square(r)

        b_phi = magnetic_field_t[in_bounds, 1]

        b_phi_dr = magnetic_field_dx[in_bounds, 1, 0]
        b_phi_dz = magnetic_field_dx[in_bounds, 1, 2]

        f_dr2 = f_toroidal_dx2[in_bounds, 0, 0]
        f_drdz = f_toroidal_dx2[in_bounds, 0, 2]
        f_dz2 = f_toroidal_dx2[in_bounds, 2, 2]

        # NOTE: This is in the holonomic basis so there is extra 1 / r.
        return_array[in_bounds, 1, 0, 0] = (
            -4 * b_phi_dr / r + (f_dr2 - 2 * b_phi) / r2
        )
        return_array[in_bounds, 1, 0, 2] = (f_drdz / r - 2 * b_phi_dz) / r
        return_array[in_bounds, 1, 2, 0] = return_array[:, 1, 0, 2]
        return_array[in_bounds, 1, 2, 2] = f_dz2 / r2

    def jerk_function(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        return_array: FloatArray,
        /,
        *,
        rho: FloatArray = None,
        psi_norm_dx4: FloatArray = None,
        f_toroidal_dx3: FloatArray = None,
        magnetic_field_dx: FloatArray = None,
        magnetic_field_dx2: FloatArray = None,
    ):
        """
        Calculate third derivative of magnetic field vector with respect to
        cylindrical position in holonomic basis.

        Parameters
        ----------
        abscissa : np.array[float]
            Cylindrical position.
        out_of_bounds : np.array[bool]
            Boolean array if input data in bounds of model.
        return_array : np.array[float]
            Array into which the result is stored.
        rho : np.array[float], optional
            Root normalised poloidal flux.
        psi_norm_dx4 : np.array[float], optional
            Fourth derivative of normalised poloidal flux.
        f_toroidal_dx3 : np.array[float], optional
            Third derivative of diamagnetic function F.
        magnetic_field_dx : np.array[float], optional
            First derivative of magnetic field vector.
        magnetic_field_dx2 : np.array[float], optional
            Second derivative of magnetic field vector.
        """
        _n = abscissa.shape[0]
        _x = Dimensions.x.size

        # Zero return array.
        return_array.fill(0.0)
        in_bounds = np.invert(out_of_bounds)

        if not np.any(in_bounds):
            return

        # Root normalised poloidal flux
        if rho is None:
            rho = self.rho_poloidal.value(abscissa)

        # Normalised poloidal flux fourth derivative wrt space.
        if psi_norm_dx4 is None:
            psi_norm_dx4 = self._psi_norm_dx4(abscissa, rho=rho)
        else:
            psi_norm_dx4 = np.reshape(psi_norm_dx4, (_n, _x, _x, _x, _x))

        # F toroidal third derivative wrt space.
        if f_toroidal_dx3 is None:
            f_toroidal_dx3 = self._f_toroidal_dx3(abscissa, rho=rho)
        else:
            f_toroidal_dx3 = np.reshape(f_toroidal_dx3, (_n, _x, _x, _x))

        if magnetic_field_dx is None:
            magnetic_field_dx = np.empty((_n, _x, _x))

            self.jacobian_function(
                abscissa, out_of_bounds, magnetic_field_dx, rho=rho
            )
        else:
            magnetic_field_dx = np.reshape(magnetic_field_dx, (_n, _x, _x))

        if magnetic_field_dx2 is None:
            magnetic_field_dx2 = np.empty((_n, _x, _x, _x))

            self.hessian_function(
                abscissa,
                out_of_bounds,
                magnetic_field_dx2,
                rho=rho,
                magnetic_field_dx=magnetic_field_dx,
            )
        else:
            magnetic_field_dx = np.reshape(magnetic_field_dx, (_n, _x, _x))

        # Poloidal field components.
        r = abscissa[in_bounds, 0]
        _psi = self.total_poloidal_flux_wb * TWO_PI_INV

        psi_norm_dr4 = psi_norm_dx4[in_bounds, 0, 0, 0, 0]
        psi_norm_dr3dz = psi_norm_dx4[in_bounds, 0, 0, 0, 2]
        psi_norm_dr2dz2 = psi_norm_dx4[in_bounds, 0, 0, 2, 2]
        psi_norm_drdz3 = psi_norm_dx4[in_bounds, 0, 2, 2, 2]
        psi_norm_dz4 = psi_norm_dx4[in_bounds, 2, 2, 2, 2]

        b_r_dr2 = magnetic_field_dx2[in_bounds, 0, 0, 0]
        b_r_drdz = magnetic_field_dx2[in_bounds, 0, 0, 2]
        b_r_dz2 = magnetic_field_dx2[in_bounds, 0, 2, 2]

        b_z_dr2 = magnetic_field_dx2[in_bounds, 2, 0, 0]
        b_z_drdz = magnetic_field_dx2[in_bounds, 2, 0, 2]
        b_z_dz2 = magnetic_field_dx2[in_bounds, 2, 2, 2]

        # B_r.
        return_array[in_bounds, 0, 0, 0, 0] = (
            _psi * psi_norm_dr3dz - 3 * b_r_dr2
        ) / r
        return_array[in_bounds, 0, 0, 0, 2] = (
            _psi * psi_norm_dr2dz2 - 2 * b_r_drdz
        ) / r
        return_array[in_bounds, 0, 0, 2, 2] = (
            _psi * psi_norm_drdz3 - b_r_dz2
        ) / r
        return_array[in_bounds, 0, 2, 2, 2] = _psi * psi_norm_dz4 / r

        return_array[in_bounds, 0, 0, 2, 0] = return_array[
            in_bounds, 0, 0, 0, 2
        ]
        return_array[in_bounds, 0, 2, 0, 0] = return_array[
            in_bounds, 0, 0, 0, 2
        ]
        return_array[in_bounds, 0, 2, 0, 2] = return_array[
            in_bounds, 0, 0, 2, 2
        ]
        return_array[in_bounds, 0, 2, 2, 0] = return_array[
            in_bounds, 0, 0, 2, 2
        ]

        # B_z.
        return_array[in_bounds, 2, 0, 0, 0] = -(
            (_psi * psi_norm_dr4 + 3 * b_z_dr2) / r
        )
        return_array[in_bounds, 2, 0, 0, 2] = -(
            (_psi * psi_norm_dr3dz + 2 * b_z_drdz) / r
        )
        return_array[in_bounds, 2, 0, 2, 2] = -(
            (_psi * psi_norm_dr2dz2 + b_z_dz2) / r
        )
        return_array[in_bounds, 2, 2, 2, 2] = -_psi * psi_norm_drdz3 / r

        return_array[in_bounds, 2, 0, 2, 0] = return_array[
            in_bounds, 2, 0, 0, 2
        ]
        return_array[in_bounds, 2, 2, 0, 0] = return_array[
            in_bounds, 2, 0, 0, 2
        ]
        return_array[in_bounds, 2, 2, 0, 2] = return_array[
            in_bounds, 2, 0, 2, 2
        ]
        return_array[in_bounds, 2, 2, 2, 0] = return_array[
            in_bounds, 2, 0, 2, 2
        ]

        # Toroidal field component.
        # NOTE: This is in the holonomic basis so there is extra 1 / r.
        r2 = np.square(r)

        b_phi_dr = magnetic_field_dx[in_bounds, 1, 0]
        b_phi_dz = magnetic_field_dx[in_bounds, 1, 2]

        b_phi_dr2 = magnetic_field_dx2[in_bounds, 1, 0, 0]
        b_phi_drdz = magnetic_field_dx2[in_bounds, 1, 0, 2]
        b_phi_dz2 = magnetic_field_dx2[in_bounds, 1, 2, 2]

        f_dr3 = f_toroidal_dx3[in_bounds, 0, 0, 0]
        f_dr2dz = f_toroidal_dx3[in_bounds, 0, 0, 2]
        f_drdz2 = f_toroidal_dx3[in_bounds, 0, 2, 2]
        f_dz3 = f_toroidal_dx3[in_bounds, 2, 2, 2]

        # B_phi.
        return_array[in_bounds, 1, 0, 0, 0] = (
            f_dr3 - 6 * b_phi_dr - 6 * r * b_phi_dr2
        ) / r2
        return_array[in_bounds, 1, 0, 0, 2] = (
            f_dr2dz - 2 * b_phi_dz - 4 * r * b_phi_drdz
        ) / r2
        return_array[in_bounds, 1, 0, 2, 2] = (
            f_drdz2 - 2 * r * b_phi_dz2
        ) / r2
        return_array[in_bounds, 1, 2, 2, 2] = f_dz3 / r2

        return_array[in_bounds, 1, 0, 2, 0] = return_array[
            in_bounds, 1, 0, 0, 2
        ]
        return_array[in_bounds, 1, 2, 0, 0] = return_array[
            in_bounds, 1, 0, 0, 2
        ]
        return_array[in_bounds, 1, 2, 0, 2] = return_array[
            in_bounds, 1, 0, 2, 2
        ]
        return_array[in_bounds, 1, 2, 2, 0] = return_array[
            in_bounds, 1, 0, 2, 2
        ]

    def _fill_cache(
        self,
        abscissa: FloatArray,
        out_of_bounds: BooleanArray,
        derivative: int,
        value: FloatArray,
        jacobian: FloatArray,
        hessian: FloatArray,
        jerk: FloatArray,
        _snap: FloatArray,
    ):
        """
        Evaluate up to nu-th derivative of model value.

        Parameters
        ----------
        abscissa : np.array[float]
            Input values to evaluate model at.
        out_of_bounds : np.array[bool]
            Flag indicating if provided input values were outside range of
            model.
        derivative : int
            Order of derivative. Must be > 0 or <= 3.
        value : np.array[float]
            Array to store value result.
        jacobian : np.array[float]
            Array to store first derivative result.
        hessian : np.array[float]
            Array to store second derivative result.
        jerk : np.array[float]
            Array to store third derivative result.
        _snap : np.array[float]
            Array to store fourth derivative result.

        Raises
        ------
        ValueError
            nu is < 0 or >= 4.
        """
        if derivative < 0 or derivative >= 4:  # noqa: PLR2004
            raise ValueError(derivative)

        _n = abscissa.shape[0]
        _x = self.input_dimension.size

        rho = np.empty(_n)
        rho_dx = np.empty((_n, _x))

        # Value.
        self.rho_poloidal.value_function(abscissa, out_of_bounds, rho)
        self.rho_poloidal.jacobian_function(abscissa, out_of_bounds, rho_dx)

        psi_norm_dx = self._psi_norm_dx(abscissa, rho=rho, rho_dx=rho_dx)
        f_toroidal = self._f_toroidal(abscissa, rho=rho)

        self.value_function(
            abscissa,
            out_of_bounds,
            value,
            psi_norm_dx=psi_norm_dx,
            f_toroidal=f_toroidal,
        )

        if derivative == 0:
            return

        # Jacobian.
        rho_dx2 = np.empty((_n, _x, _x))

        self.rho_poloidal.hessian_function(abscissa, out_of_bounds, rho_dx2)
        psi_norm_dx2 = self._psi_norm_dx2(
            abscissa, rho=rho, rho_dx=rho_dx, rho_dx2=rho_dx2
        )

        f_toroidal_drho = self.f_toroidal_tm.jacobian(rho.reshape(-1, 1))[:, 0]
        f_toroidal_dx = self._f_toroidal_dx(
            abscissa, rho=rho, rho_dx=rho_dx, f_toroidal_drho=f_toroidal_drho
        )

        self.jacobian_function(
            abscissa,
            out_of_bounds,
            jacobian,
            psi_norm_dx2=psi_norm_dx2,
            f_toroidal_dx=f_toroidal_dx,
            magnetic_field_t=value,
        )

        if derivative == 1:
            return

        # Hessian.
        rho_dx3 = np.empty((_n, _x, _x, _x))
        self.rho_poloidal.jerk_function(abscissa, out_of_bounds, rho_dx3)

        psi_norm_dx3 = self._psi_norm_dx3(
            abscissa, rho=rho, rho_dx=rho_dx, rho_dx2=rho_dx2, rho_dx3=rho_dx3
        )

        f_toroidal_drho2 = self.f_toroidal_tm.hessian(rho.reshape(-1, 1))[
            :, 0, 0
        ]
        f_toroidal_dx2 = self._f_toroidal_dx2(
            abscissa,
            rho=rho,
            rho_dx=rho_dx,
            rho_dx2=rho_dx2,
            f_toroidal_drho=f_toroidal_drho,
            f_toroidal_drho2=f_toroidal_drho2,
        )

        self.hessian_function(
            abscissa,
            out_of_bounds,
            hessian,
            rho=rho,
            psi_norm_dx3=psi_norm_dx3,
            f_toroidal_dx2=f_toroidal_dx2,
            magnetic_field_t=value,
            magnetic_field_dx=jacobian,
        )

        if derivative == 2:  # noqa: PLR2004
            return

        # Jerk.
        rho_dx4 = np.empty((
            abscissa.shape[0],
            self.input_dimension.size,
            self.input_dimension.size,
            self.input_dimension.size,
            self.input_dimension.size,
        ))
        self.rho_poloidal.snap_function(abscissa, out_of_bounds, rho_dx4)

        psi_norm_dx4 = self._psi_norm_dx4(
            abscissa,
            rho=rho,
            rho_dx=rho_dx,
            rho_dx2=rho_dx2,
            rho_dx3=rho_dx3,
            rho_dx4=rho_dx4,
        )

        f_toroidal_drho3 = self.f_toroidal_tm.jerk(rho.reshape(-1, 1))[
            :, 0, 0, 0
        ]
        f_toroidal_dx3 = self._f_toroidal_dx3(
            abscissa,
            rho=rho,
            rho_dx=rho_dx,
            rho_dx2=rho_dx2,
            rho_dx3=rho_dx3,
            f_toroidal_drho=f_toroidal_drho,
            f_toroidal_drho2=f_toroidal_drho2,
            f_toroidal_drho3=f_toroidal_drho3,
        )

        self.jerk_function(
            abscissa,
            out_of_bounds,
            jerk,
            rho=rho,
            psi_norm_dx4=psi_norm_dx4,
            f_toroidal_dx3=f_toroidal_dx3,
            magnetic_field_dx=jacobian,
            magnetic_field_dx2=hessian,
        )

        if derivative == 3:  # noqa: PLR2004
            return

    @staticmethod
    def write_netcdf(*_):
        """
        Write to netCDF4 dataset. Do not use.

        Raises
        ------
        ValueError
            Always. Write using SystemData.write_netcdf.
        """
        raise ValueError("Write using SystemData.write_netcdf")

    @classmethod
    def read_netcdf(cls, *_):
        """
        Load from netCDF4 dataset. Do not use.

        Raises
        ------
        ValueError
            Always. Load using SystemData.read_netcdf.
        """
        raise ValueError("Load using SystemData.read_netcdf")


class NonAxisymmetricMagneticField(ValueModelBase):
    """
    Non-axisymmetric cylindrical magnetic field.
    """

    __slots__ = ()

    def __init__(self):
        """
        Inits NonAxisymmetricMagneticField.
        """
        raise NotImplementedError
