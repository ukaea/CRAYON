"""
Objects for flux coordinate systems.
"""

# Standard imports
import logging

# Third party imports
import contourpy
import netCDF4 as nc4  # noqa: N813
import numpy as np
from scipy import integrate, interpolate

# Local imports
from crayon.coordinates.coordinates.base import (
    CoordinateSystem,
    ForwardTransformDerivatives,
    map_angle_minus_pi_to_pi,
)
from crayon.coordinates.coordinates.cylindrical import CYLINDRICAL
from crayon.shared.dimensions import Dimension, Dimensions
from crayon.shared.helpers import get_return_array
from crayon.shared.io import write_netcdf_variable
from crayon.shared.types import BooleanArray, FloatArray
from crayon.value_model.splines import Spline1D, Spline2D

logger = logging.getLogger(__name__)


class AxisymmetricFluxCoordinate(ForwardTransformDerivatives):
    """
    Axisymmetric flux coordinate system (rho, phi, theta).
        - rho is a flux surface label [0, inf)
        - phi is geometric toroidal angle (counter-clockwise > 0) [-pi, pi]
        - theta is geometric poloidal angle (counter-clockwise > 0) [-pi, pi]

    This system is derived from a cylindrical coordinate system (r, phi, z)
    around a magnetic axis (r0, z0) where rho = 0.

        rho = rho_func(R, Z)
        phi = phi
        theta = arctan((z - z0) / (r - r0))

    where rho_func is a given function. This can be inverted as

        r(rho, theta) = r0 + R_func(rho, theta) * cos(theta)
        phi = phi
        z(rho, theta) = z0 + R_func(rho, theta) * sin(theta)

    where R_func is a given function giving the radius from (r0, z0).

    NOTE: The theta and rho unit vectors are not always orthogonal so this
        coordinate system is skewed.

    NOTE: We can easily extend to a general flux coordinate system by allowing
        rho_func and R_func to depend on toroidal angle.

    NOTE: As we use geometric poloidal angle the flux surfaces must be convex
        to ensure the flux coordinate is bijective i.e. no 2 points have same
        poloidal angle.

        You could relax this by using an equal arc length coordinate s but it
        is numerically challenging to find s(R, Z). This would also break the
        radial remapping.
    """

    __slots__ = (
        "coordinate_system",
        "isocontours_rz",
        "magnetic_axis_m",
        "radius_spline",
        "rho_1d",
        "rho_spline",
        "theta_1d",
    )

    orthogonal = False
    forward_transform_preferred_coordinate = CoordinateSystem.CYLINDRICAL

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        rho_spline: Spline2D,
        magnetic_axis_m: tuple[float, float],
        rho_1d: FloatArray,
        theta_1d: FloatArray,
        isocontours_rz: FloatArray,
    ):
        """
        Inits flux coordinate system from numerical flux function rho(r, z).

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system this flux coordinate represents.
        rho_spline : Spline2D
            Spline object giving the radial coordinate rho as a function
            of cylindrical position (r, phi, z).
        magnetic_axis_m : tuple[float, float]
            Cylindrical coordinates (r, z) of magnetic axis.
        rho_1d : np.array[float]
            Radial coordinate array for isocontours of size n.
        theta_1d : np.array[float]
            Poloidal angle about magnetic axis array for isocontours of size m.
        isocontours_rz : np.array[float]
            Cylindrical (r, z) position of isocontours of rho. Must have
            shape (n, m, 2)

        Raises
        ------
        TypeError
            rho_spline is not Spline2D.

        ValueError
            rho_spline depends on a coordinate system that isn't cylindrical.
            magnetic_axis_m doesn't have shape (2,).
            rho_1d or theta_1d are not 1 dimensional.
            isocontours_rz has incorrect shape.
        """
        self.coordinate_system = coordinate_system
        super().__init__(CYLINDRICAL)

        if not isinstance(rho_spline, Spline2D):
            raise TypeError(
                f"rho_spline must be Spline2D. Got {type(rho_spline)}."
            )

        # Check rho provided against cylindrical coordinates.
        if rho_spline.coordinate_system != CoordinateSystem.CYLINDRICAL:
            raise ValueError(
                "rho spline must be in cylindrical coordinates. "
                f"Got {rho_spline.coordinate_system}"
            )

        self.rho_spline = rho_spline
        self.magnetic_axis_m = np.asarray(magnetic_axis_m)

        if self.magnetic_axis_m.shape != (2,):
            raise ValueError(
                "magnetic_axis_m has incorrect shape. "
                f"Expected (2,), got {self.magnetic_axis_m.shape}"
            )

        self.rho_1d = np.asarray(rho_1d)
        self.theta_1d = np.asarray(theta_1d)
        self.isocontours_rz = np.asarray(isocontours_rz)

        if self.rho_1d.ndim != 1:
            raise ValueError(
                f"rho_1d must be dimension 1. Got shape {self.rho_1d.shape}."
            )

        if self.theta_1d.ndim != 1:
            raise ValueError(
                "theta_1d must be dimension 1. "
                f"Got shape {self.theta_1d.shape}."
            )

        if self.isocontours_rz.shape != (
            self.rho_1d.size,
            self.theta_1d.size,
            2,
        ):
            raise ValueError(
                "isocontours_rz has incorrect shape. "
                f"Expected {(self.rho_1d.size, self.theta_1d.size)}, "
                f"got {self.isocontours_rz.shape}."
            )

        # Find radius of isocontour from magnetic axis.
        # Defines inverse map r(rho, theta), z(rho, theta).
        _rho_1d = np.empty(self.rho_1d.size + 1)
        _rho_1d[0] = 0.0
        _rho_1d[1:] = self.rho_1d

        isocontour_radius = np.empty((rho_1d.size + 1, theta_1d.size))
        isocontour_radius[0, :] = 0.0

        for i, contour in enumerate(self.isocontours_rz, start=1):
            isocontour_radius[i, :] = np.sqrt(
                np.square(contour[:, 0] - self.magnetic_axis_m[0])
                + np.square(contour[:, 1] - self.magnetic_axis_m[1])
            )

            # Enforce periodicity.
            isocontour_radius[i, 0] = 0.5 * (
                isocontour_radius[i, 0] + isocontour_radius[i, -1]
            )
            isocontour_radius[i, -1] = isocontour_radius[i, 0]

        # Create spline.
        self.radius_spline = Spline2D(
            self.coordinate_system,
            Dimensions.x,
            (),
            "m",
            _rho_1d,
            self.theta_1d,
            isocontour_radius,
            (True, False, True),
            method=self.rho_spline.method,
        )

    @staticmethod
    def _contour_winds(
        x_1d: FloatArray,
        y_1d: FloatArray,
        winding_point: tuple[float, float],
        /,
        *,
        threshold: float = 0.75,
    ) -> bool:
        """
        Return True if a 2D curve winds about a given point.

        Parameters
        ----------
        x_1d
            x position along curve.
        y_1d
            y position along curve.
        winding_point: (float, float)
            (x, y) position of point being checked the curve winds about.
        threshold : float, optional
            Fraction of 2*pi the wound angle must exceed. Numerical error means
            this doesn't always equal 1.

        Returns
        -------
        contour_winds : bool
            Flag if curve winds about winding_point.

        Notes
        -----
        This algorithm only works well if the radius from the winding point
        changes slowly along the contour as the change in radius is neglected
        in the integral.
        """
        dx = x_1d - winding_point[0]
        dy = y_1d - winding_point[1]
        r2 = np.square(dx) + np.square(dy)

        wound_angle = integrate.trapezoid(dx / r2, x=dy) - integrate.trapezoid(
            dy / r2, x=dx
        )
        critical_wound_angle = max(0.0, min(1.0, threshold)) * 2.0 * np.pi

        return abs(wound_angle) >= critical_wound_angle

    @staticmethod
    def _interpolate_closed_contour(
        theta: FloatArray,
        r: FloatArray,
        z: FloatArray,
        theta_target: FloatArray,
        r_out: FloatArray,
        z_out: FloatArray,
    ):
        """
        Interpolate a closed convex contour in poloidal angle theta.

        Parameters
        ----------
        theta : np.array[float]
            Poloidal angle about magnetic axis along contour.
        r, z : np.array[float]
            Cylindrical (r, z) position along contour.
        theta_target : np.array[float]
            Poloidal angle grid to interpolate contour on to.
        r_out : np.array[float]
            Array interpolated r values are stored. Must have same size as
            theta_target.
        z_out : np.array[float]
            Array interpolated z values are stored. Must have same size as
            theta_target.
        """
        # Remove any duplicates.
        _theta, index = np.unique(theta, return_index=True)
        _x = r[index]
        _y = z[index]

        # Spline fit requires theta data in ascending order.
        # This also enforces contour to be right handed.
        order = np.argsort(_theta[:-1])
        _theta[:-1] = _theta[order]
        _x[:-1] = _x[order]
        _y[:-1] = _y[order]

        # Enforce periodicity as periodic cubic spline requires this.
        _theta[-1] = _theta[0] + 2 * np.pi
        _x[-1] = _x[0]
        _y[-1] = _y[0]

        x_spline = interpolate.CubicSpline(_theta, _x, bc_type="periodic")
        y_spline = interpolate.CubicSpline(_theta, _y, bc_type="periodic")

        r_out[:] = x_spline(theta_target)
        z_out[:] = y_spline(theta_target)

    @staticmethod
    def _find_contours(
        r_1d: FloatArray,
        z_1d: FloatArray,
        rho_2d: FloatArray,
        axis: tuple[float, float],
        levels: FloatArray,
        theta_target: FloatArray,
        /,
        *,
        retry_boundary: bool = True,
    ) -> tuple[BooleanArray, FloatArray, FloatArray]:
        """
        Find 2D isocontours of radial coordinate rho which wind about axis
        using a provided grid of rho(r, z).

        Parameters
        ----------
        r_1d, z_1d : np.array[float]
            Cylindrical (r, z) arrays defining grid on which rho is provided.
        rho_2d : np.array[float]
            Array of radial coordinate rho.
        axis : tuple[float, float]

        levels : np.array[float]
            Levels of isocontours to find of size n.
        theta_target : np.array[float]
            Poloidal angle grid to interpolate contour on to of size m.

        Returns
        -------
        contour_found : np.array[bool]
            Flag if contour level was found succesfully.
        contour_levels : np.array[float]
            Level contour was found at.
        contour_rz : np.array[float]
            Cylindrical (r, z) values of isocontours. Has shape (n, m, 2).
        """
        # Calculate contours.
        contour_generator = contourpy.contour_generator(r_1d, z_1d, rho_2d.T)

        # Find closed winding contour and project onto theta grid.
        x0, y0 = axis
        n_theta = theta_target.size

        contour_found = np.zeros(len(levels), dtype=bool)
        contour_levels = np.copy(levels)
        contour_rz = np.zeros((len(levels), n_theta, 2))

        for i, level in enumerate(levels):
            # Generated contours wind anticlockwise if enclosed z > level.
            # Generated contours wind clockwise if enclosed z < level.
            line_set = contour_generator.lines(level)

            # Check if any contours were found.
            n_contours = len(line_set)

            if retry_boundary and i == len(levels) - 1 and len(line_set) == 0:
                # Retry final isocontour as it might fail because its very
                # close to a non-closed contour.
                for scaling in (0.999, 0.995, 0.99):
                    line_set = contour_generator.lines(scaling * level)

                    if len(line_set) > 0:
                        contour_levels[i] = scaling * level
                        break

            if len(line_set) == 0:
                msg = (
                    f"Unable to find contour for level = {level} "
                    "(found zero contours)."
                )
                logger.warning(msg)
                continue

            # Check if multiple contours were found.
            if len(line_set) > 1:
                # Use contour that winds about axis.
                for line in line_set:
                    if AxisymmetricFluxCoordinate._contour_winds(
                        line[:, 0], line[:, 1], axis
                    ):
                        contour_found[i] = True
                        break
            else:
                # Check found contour winds about axis.
                line = line_set[0]

                if AxisymmetricFluxCoordinate._contour_winds(
                    line[:, 0], line[:, 1], axis
                ):
                    contour_found[i] = True

            if not contour_found[i]:
                msg = (
                    f"Unable to find contour for level = {level} "
                    f"(found {n_contours} contour(s), none wind about axis)."
                )
                logger.warning(msg)
                continue

            # Interpolate contour onto theta grid using cubic spline.
            x, y = line[:, 0], line[:, 1]
            theta = np.arctan2(y - y0, x - x0)

            AxisymmetricFluxCoordinate._interpolate_closed_contour(
                theta,
                x,
                y,
                theta_target,
                contour_rz[i, :, 0],
                contour_rz[i, :, 1],
            )

        return contour_found, contour_levels, contour_rz

    @classmethod
    def find_contours(
        cls,
        coordinate_system: CoordinateSystem,
        rho_spline: Spline2D,
        magnetic_axis_m: tuple[float, float],
        /,
        *,
        n_contours: int = 51,
        n_theta: int = 101,
        boundary_rz: tuple[FloatArray, FloatArray] | None = None,
    ) -> "AxisymmetricFluxCoordinate":
        """
        Create AxisymmetricFluxCoordinate object by finding isocontours
        of radial coordinate rho.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Coordinate system this flux coordinate represents.
        rho_spline : Spline2D
            Spline object giving the radial coordinate rho as a function
            of cylindrical position (r, phi, z).
        magnetic_axis_m : tuple[float, float]
            Cylindrical coordinates (r, z) of magnetic axis.
        n_contours : int
            Number of isocontours in rho to find.
        n_theta : int
            Number of equispaced poloidal angle values between [-pi, pi] used
            to represent isocontours.
        boundary_rz : tuple[np.array[float], np.array[float]]
            Cylindrical (r, z) coordinates of boundary / separatrix contour.
            Often provided if boundary contour is numerically difficult to
            find e.g. tokamak plasmas with X points.

        Returns
        -------
        axisymmetric_flux_coordinate : AxisymmetricFluxCoordinate
            Axisymmetric flux coordinate object.

        Raises
        ------
        ValueError
            Unable to calculate > 50% of the isocontours.
        """
        # Sizes of meshes
        n_contours = int(n_contours)
        n_theta = int(n_theta)

        if n_theta % 2 == 0:
            n_theta += 1
            logger.warning(
                "n_theta must be odd. Incrementing by 1 to %s.", n_theta
            )

        r_1d, z_1d = rho_spline.abscissas
        rho_2d = rho_spline.data

        # Construct arrays to hold isocontour data.
        # Avoid levels with 10% of axis due to numerical difficulties.
        rho_1d = np.linspace(0.10, 1.0, n_contours)
        theta_1d = np.linspace(-np.pi, np.pi, n_theta)

        found_contour = np.empty_like(rho_1d, dtype=bool)
        isocontours = np.empty((n_contours, n_theta, 2))

        if boundary_rz is None:
            (found_contour[:], rho_1d[:], isocontours[:, :, :]) = (
                cls._find_contours(
                    r_1d, z_1d, rho_2d, magnetic_axis_m, rho_1d, theta_1d
                )
            )
        else:
            (found_contour[:-1], _, isocontours[:-1, :, :]) = (
                cls._find_contours(
                    r_1d, z_1d, rho_2d, magnetic_axis_m, rho_1d[:-1], theta_1d
                )
            )

            found_contour[-1] = True
            _r, _z = boundary_rz
            _theta = np.arctan2(
                _z - magnetic_axis_m[1], _r - magnetic_axis_m[0]
            )

            cls._interpolate_closed_contour(
                _theta,
                _r,
                _z,
                np.linspace(-np.pi, np.pi, n_theta),
                isocontours[-1, :, 0],
                isocontours[-1, :, 1],
            )

        # Check sufficient isocontours were found.
        if sum(found_contour) < 0.5 * n_contours:
            raise ValueError(r"Unable to calculate > 50% of isocontours")

        rho_1d = rho_1d[found_contour]
        isocontours_rz = isocontours[found_contour]

        # Use Newton method to improve isocontours.
        # This is important to get accurate derivatives of quantities
        # which depend on these contours.
        r0, z0 = magnetic_axis_m
        position_cylindrical = np.zeros((isocontours_rz.shape[1], 3))

        for i in range(rho_1d.size):
            position_cylindrical[:, 0] = isocontours_rz[i, :, 0]
            position_cylindrical[:, 2] = isocontours_rz[i, :, 1]

            dr = position_cylindrical[:, 0] - r0
            dz = position_cylindrical[:, 2] - z0

            radius = np.sqrt(np.square(dr) + np.square(dz))

            for k in range(16):
                rho = rho_spline.value(position_cylindrical)
                theta = np.arctan2(dz, dr)
                drho = rho - rho_1d[i]

                # Force a couple of rounds.
                if k > 2 and np.all(abs(drho) < 1e-6):  # noqa: PLR2004
                    break

                rho_jacobian = rho_spline.jacobian(position_cylindrical)

                c, s = np.cos(theta), np.sin(theta)
                drho_da = c * rho_jacobian[:, 0] + s * rho_jacobian[:, 2]

                radius -= drho / drho_da
                position_cylindrical[:, 0] = magnetic_axis_m[0] + radius * c
                position_cylindrical[:, 2] = magnetic_axis_m[1] + radius * s

            isocontours_rz[i, :, 0] = position_cylindrical[:, 0]
            isocontours_rz[i, :, 1] = position_cylindrical[:, 2]

        return cls(
            coordinate_system,
            rho_spline,
            magnetic_axis_m,
            rho_1d,
            theta_1d,
            isocontours_rz,
        )

    @staticmethod
    def bound_components(
        position_flux_coordinates: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ) -> FloatArray:
        """
        Bound the coordinate components to within coordinate system limits.

        Parameters
        ----------
        position_flux_coordinates : np.array[float]
            Coordinate components (rho, phi, theta).
        return_array : np.array[float], optional
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        bounded_position
            Position components bounded to coordinate system limits.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        rho_poloidal, phi, theta = position_flux_coordinates
        return_array[0] = abs(rho_poloidal)

        if rho_poloidal < 0.0:
            # Phase shift from crossing magnetic axis.
            theta += np.pi

        return_array[1] = map_angle_minus_pi_to_pi(phi)
        return_array[2] = map_angle_minus_pi_to_pi(theta)

        return return_array

    def forward_transform(
        self,
        position_cylindrical: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert cylindrical position to flux coordinate.

        Parameters
        ----------
        position_cylindrical : np.array[float]
            Position components in cylindrical coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_flux_coordinate : np.array[float]
            Position components in flux coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        r, phi, z = position_cylindrical

        return_array[0] = self.rho_spline(position_cylindrical)
        return_array[1] = phi
        return_array[2] = np.arctan2(
            z - self.magnetic_axis_m[1], r - self.magnetic_axis_m[0]
        )

        return self.bound_components(return_array)

    def backward_transform(
        self,
        position_flux_coordinate: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert flux coordinate position to cylindrical.

        Parameters
        ----------
        position_flux_coordinate : np.array[float]
            Position components in flux coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_cylindrical : np.array[float]
            Position components in cylindrical coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        _, phi, theta = position_flux_coordinate
        r0, z0 = self.magnetic_axis_m
        radius = self.radius_spline(position_flux_coordinate)

        return_array[0] = r0 + radius * np.cos(theta)
        return_array[1] = phi
        return_array[2] = z0 + radius * np.sin(theta)

        self.parent_coordinate.bound_components(
            return_array, return_array=return_array
        )

        return return_array

    def forward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        First derivative of flux coordinate position with respect to
        cylindrical position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx : np.array[float]
            First derivative of forward transform.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), float
        )

        if self.is_coordinate(coordinate_system):
            position_cylindrical = self.backward_transform(position)
        else:
            position_cylindrical = position

        r, _, z = position_cylindrical

        r0, z0 = self.magnetic_axis_m
        dr, dz = r - r0, z - z0
        r2 = dr * dr + dz * dz

        rho_jacobian = self.rho_spline.jacobian(position_cylindrical)

        # Need to fix jacobian being zero out of domain. See issue #38.
        if np.allclose(rho_jacobian, 0.0):
            rho_jacobian[:] = 1.0, 0.0, 0.0

        return_array.fill(0.0)

        return_array[0, 0] = rho_jacobian[0]
        return_array[0, 2] = rho_jacobian[2]
        return_array[1, 1] = 1.0
        return_array[2, 0] = -dz / r2
        return_array[2, 2] = dr / r2

        return return_array

    def forward_transform_dx2(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Second derivative of flux coordinate position with respect to
        cylindrical position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx2 : np.array[float]
            Second derivative of forward transform.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
            float,
        )

        if self.is_coordinate(coordinate_system):
            position_cylindrical = self.backward_transform(position)
        else:
            position_cylindrical = position

        r, _, z = position_cylindrical

        r0, z0 = self.magnetic_axis_m
        dr, dz = r - r0, z - z0
        dr2, dz2 = dr * dr, dz * dz
        r2 = dr2 + dz2
        r4 = r2 * r2

        rho_hessian = self.rho_spline.hessian(position_cylindrical)
        rho_dr2 = rho_hessian[0, 0]
        rho_drdz = rho_hessian[0, 2]
        rho_dz2 = rho_hessian[2, 2]

        return_array.fill(0.0)

        return_array[0, 0, 0] = rho_dr2
        return_array[0, 0, 2] = rho_drdz
        return_array[0, 2, 0] = return_array[0, 0, 2]
        return_array[0, 2, 2] = rho_dz2

        return_array[2, 0, 0] = 2 * dr * dz / r4
        return_array[2, 0, 2] = (dz2 - dr2) / r4
        return_array[2, 2, 0] = return_array[2, 0, 2]
        return_array[2, 2, 2] = -2 * dr * dz / r4

        return return_array

    def forward_transform_dx3(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Third derivative of flux coordinate position with respect to
        cylindrical position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx3 : np.array[float]
            Third derivative of forward transform.
        """
        return_array = get_return_array(
            return_array,
            (
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.x.size,
            ),
            float,
        )

        if self.is_coordinate(coordinate_system):
            position_cylindrical = self.backward_transform(position)
        else:
            position_cylindrical = position

        r, _, z = position_cylindrical

        r0, z0 = self.magnetic_axis_m
        dr, dz = r - r0, z - z0
        dr2, dz2 = dr * dr, dz * dz
        r2 = dr2 + dz2
        r6 = r2 * r2 * r2

        rho_jerk = self.rho_spline.jerk(position_cylindrical)

        rho_dr3 = rho_jerk[0, 0, 0]
        rho_dr2dz = rho_jerk[0, 0, 2]
        rho_drdz2 = rho_jerk[0, 2, 2]
        rho_dz3 = rho_jerk[2, 2, 2]

        return_array.fill(0.0)

        return_array[0, 0, 0, 0] = rho_dr3
        return_array[0, 0, 0, 2] = rho_dr2dz
        return_array[0, 0, 2, 0] = return_array[0, 0, 0, 2]
        return_array[0, 0, 2, 2] = rho_drdz2
        return_array[0, 2, 0, 0] = return_array[0, 0, 0, 2]
        return_array[0, 2, 0, 2] = return_array[0, 0, 2, 2]
        return_array[0, 2, 2, 0] = return_array[0, 0, 2, 2]
        return_array[0, 2, 2, 2] = rho_dz3

        return_array[2, 0, 0, 0] = -2 * dz * (3 * dr2 - dz2) / r6
        return_array[2, 0, 0, 2] = 2 * dr * (dr2 - 3 * dz2) / r6
        return_array[2, 0, 2, 0] = return_array[2, 0, 0, 2]
        return_array[2, 0, 2, 2] = 2 * dz * (3 * dr2 - dz2) / r6
        return_array[2, 2, 0, 0] = return_array[2, 0, 0, 2]
        return_array[2, 2, 0, 2] = return_array[2, 0, 2, 2]
        return_array[2, 2, 2, 0] = return_array[2, 0, 2, 2]
        return_array[2, 2, 2, 2] = -2 * dr * (dr2 - 3 * dz2) / r6

        return return_array

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.coordinate_system.name)

        # Write spline.
        self.rho_spline.write_netcdf(group.createGroup("rho_spline"))

        # Create dimensions.
        rho_dim = Dimension("rho", self.rho_1d.size)
        dset.createDimension(rho_dim.name, rho_dim.size)

        theta_dim = Dimension("theta", self.theta_1d.size)
        dset.createDimension(theta_dim.name, theta_dim.size)

        # Write additional data.
        write_netcdf_variable(
            group,
            "magnetic_axis",
            (Dimensions.two,),
            self.magnetic_axis_m,
            "r-z position of magnetic axis (origin of flux coordinates)",
            "m",
        )

        write_netcdf_variable(
            group, "rho", (rho_dim,), self.rho_1d, "Flux coordinate grid", ""
        )

        write_netcdf_variable(
            group,
            "theta",
            (theta_dim,),
            self.theta_1d,
            "Poloidal angle coordinate grid",
            "",
        )

        write_netcdf_variable(
            group,
            "isocontours",
            (rho_dim, theta_dim, Dimensions.two),
            self.isocontours_rz,
            "(R, Z) of flux coordinate isocontours",
            "",
        )

    @classmethod
    def read_netcdf(cls, dset: nc4.Dataset) -> "AxisymmetricFluxCoordinate":
        """
        Load from netCDF4 dataset.

        Returns
        -------
        axisymmetric_flux_coordinate : AxisymmetricFluxCoordinate
            Axisymmetric flux coordinate object.
        """
        coordinate_system = CoordinateSystem.parse(dset.name)

        rho_spline = Spline2D.read_netcdf(dset["rho_spline"])
        magnetic_axis_m = dset["magnetic_axis"][:].data
        rho_1d = dset["rho"][:].data
        theta_1d = dset["theta"][:].data
        isocontours_rz = dset["isocontours"][:].data

        return cls(
            coordinate_system,
            rho_spline,
            magnetic_axis_m,
            rho_1d,
            theta_1d,
            isocontours_rz,
        )


class AxisymmetricFluxCoordinateRebase(ForwardTransformDerivatives):
    """
    A flux coordinate system which is just a radial remapping i.e.
    (rho_1, phi, theta) -> (rho_2, phi, theta) where rho_2 = f(rho_1).
    """

    __slots__ = (
        "coordinate_system",
        "rho_spline_1_to_2",
        "rho_spline_2_to_1",
    )

    orthogonal = False

    def __init__(
        self,
        parent_coordinate: AxisymmetricFluxCoordinate,
        coordinate_system: CoordinateSystem,
        rho_spline_1_to_2: Spline1D,
        rho_spline_2_to_1: Spline1D,
    ):
        """
        Inits AxisymmetricFluxCoordinateRebase.

        Parameters
        ----------
        parent_coordinate : AxisymmetricFluxCoordinate
            Coordinate system this coordinate is a rebase of.
        coordinate_system : CoordinateSystem
            Coordinate system this coordinate represents.
        rho_spline_1_to_2 : Spline1D
            Spline giving transform from parent radial coordinate to this
            radial coordinate.
        rho_spline_2_to_1 : Spline1D
            Spline giving transform from this radial coordinate to parent
            radial coordinate.

        Raises
        ------
        TypeError
            rho_spline_1_to_2 or rho_spline_2_to_1 are not Spline1D.

        ValueError
            rho_spline_1_to_2 or rho_spline_2_to_1 have input dimension > 1.
        """
        self.coordinate_system = coordinate_system
        self.rho_spline_1_to_2 = rho_spline_1_to_2
        self.rho_spline_2_to_1 = rho_spline_2_to_1

        if not isinstance(rho_spline_1_to_2, Spline1D):
            raise TypeError("rho_spline_1_to_2 must be Spline1D")

        if not self.rho_spline_1_to_2.input_dimension.size == 1:
            raise ValueError(
                "rho_spline_1_to_2 input dimension must be size 1"
            )

        if not isinstance(rho_spline_2_to_1, Spline1D):
            raise TypeError("rho_spline_2_to_1 must be Spline1D")

        if not self.rho_spline_2_to_1.input_dimension.size == 1:
            raise ValueError(
                "rho_spline_2_to_1 input dimension must be size 1"
            )

        super().__init__(parent_coordinate)

    @property
    def forward_transform_preferred_coordinate(self):
        """
        Preferred coordinate system components used to calculate the forward
        transform and its derivatives.
        """
        return self.parent_coordinate_system

    bound_components = AxisymmetricFluxCoordinate.bound_components

    def forward_transform(
        self,
        position_flux_coordinate_1: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert from parent flux coordinate to this flux coordinate.

        Parameters
        ----------
        position_flux_coordinate_1 : np.array[float]
            Position components in parent flux coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_flux_coordinate_2 : np.array[float]
            Position components in this flux coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        rho_1, phi, theta = position_flux_coordinate_1

        return_array[0] = self.rho_spline_1_to_2(rho_1)
        return_array[1] = phi
        return_array[2] = theta

        return AxisymmetricFluxCoordinateRebase.bound_components(return_array)

    def backward_transform(
        self,
        position_flux_coordinate_2: FloatArray,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Convert from this flux coordinate to parent flux coordinate.

        Parameters
        ----------
        position_flux_coordinate_2 : np.array[float]
            Position components in this flux coordinate system.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        position_flux_coordinate_1 : np.array[float]
            Position components in parent flux coordinate system.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size,), float
        )

        rho_2, phi, theta = position_flux_coordinate_2

        return_array[0] = self.rho_spline_2_to_1(rho_2)
        return_array[1] = phi
        return_array[2] = theta

        self.parent_coordinate.bound_components(
            return_array, return_array=return_array
        )

        return return_array

    def forward_transform_dx(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        First derivative of flux coordinate 2 position with respect to
        flux coordinate 1 position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx : np.array[float]
            First derivative of forward transform.
        """
        return_array = get_return_array(
            return_array, (Dimensions.x.size, Dimensions.x.size), float
        )

        if self.is_coordinate(coordinate_system):
            position_flux_coordinate_1 = self.backward_transform(position)
        else:
            position_flux_coordinate_1 = position

        rho_1, _, _ = position_flux_coordinate_1
        return_array.fill(0.0)

        rho_jacobian = self.rho_spline_1_to_2(rho_1, nu=1).item()

        # Need to fix jacobian being zero out of domain. See issue #38.
        if np.allclose(rho_jacobian, 0.0):
            rho_jacobian = 1.0

        return_array[0, 0] = rho_jacobian
        return_array[1, 1] = 1.0
        return_array[2, 2] = 1.0

        return return_array

    def forward_transform_dx2(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Second derivative of flux coordinate 2 position with respect to
        flux coordinate 1 position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx2 : np.array[float]
            Second derivative of forward transform.
        """
        return_array = get_return_array(
            return_array,
            (Dimensions.x.size, Dimensions.x.size, Dimensions.x.size),
            float,
        )

        if self.is_coordinate(coordinate_system):
            position_flux_coordinate_1 = self.backward_transform(position)
        else:
            position_flux_coordinate_1 = position

        rho_1, _, _ = position_flux_coordinate_1
        return_array.fill(0.0)

        return_array[0, 0, 0] = self.rho_spline_1_to_2(rho_1, nu=2).item()

        return return_array

    def forward_transform_dx3(
        self,
        position: FloatArray,
        coordinate_system: CoordinateSystem,
        /,
        *,
        return_array: FloatArray = None,
    ):
        """
        Third derivative of flux coordinate 2 position with respect to
        flux coordinate 1 position.

        Parameters
        ----------
        position : np.array[float]
            Position components.
        coordinate_system : CoordinateSystem
            Coordinate system position components are provided in. Must be
            either Cartesian or cylindrical.
        return_array : np.array[float]
            Array into which the result is stored. If provided, must have same
            shape as output array. If not provided, an new array is allocated.

        Returns
        -------
        forward_transform_dx2 : np.array[float]
            Second derivative of forward transform.
        """
        return_array = get_return_array(
            return_array,
            (
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.x.size,
                Dimensions.x.size,
            ),
            float,
        )

        if self.is_coordinate(coordinate_system):
            position_flux_coordinate_1 = self.backward_transform(position)
        else:
            position_flux_coordinate_1 = position

        rho_1, _, _ = position_flux_coordinate_1
        return_array.fill(0.0)

        return_array[0, 0, 0, 0] = self.rho_spline_1_to_2(rho_1, nu=3).item()

        return return_array

    def write_netcdf(self, dset: nc4.Dataset):
        """
        Write to netCDF4 dataset.

        Parameters
        ----------
        dset : netCDF4.Dataset
            netCDF4 dataset or group to write data to.
        """
        group = dset.createGroup(self.coordinate_system.name)

        self.rho_spline_1_to_2.write_netcdf(
            group.createGroup("rho_spline_1_to_2")
        )
        self.rho_spline_2_to_1.write_netcdf(
            group.createGroup("rho_spline_2_to_1")
        )

    @classmethod
    def read_netcdf(
        cls, dset: nc4.Dataset, parent_coordinate: AxisymmetricFluxCoordinate
    ):
        """
        Load from netCDF4 dataset.

        Returns
        -------
        axisymmetric_flux_coordinate : AxisymmetricFluxCoordinate
            Axisymmetric flux coordinate object.
        """
        coordinate_system = CoordinateSystem.parse(dset.name)
        rho_spline_1_to_2 = Spline1D.read_netcdf(dset["rho_spline_1_to_2"])
        rho_spline_2_to_1 = Spline1D.read_netcdf(dset["rho_spline_2_to_1"])

        return cls(
            parent_coordinate,
            coordinate_system,
            rho_spline_1_to_2,
            rho_spline_2_to_1,
        )
