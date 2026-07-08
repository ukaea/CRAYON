"""
Objects for calculating linear current drive from ray tracing output.
"""

# Standard imports
import bisect
import logging

# Third party imports
import numpy as np

# Local imports
from crayon.coordinates import CoordinateCoordinator, CoordinateSystem
from crayon.linear_current_drive.options import OptionsLinearCurrentDrive
from crayon.ray_tracing.ray_tracer import RayTracingOutput
from crayon.system_data import SystemData
from crayon.system_data.system_data_provider import MagneticTokamak

logger = logging.getLogger(__name__)


class LinearCurrentDriveOutput:
    """
    Object holding outputs from linear current drive model.

    Attributes
    ----------
    bin_area_m2
        Cross sectional area of bin [m^2].
    bin_centres
        Radial coordinate at centre of bin.
    bin_damped_power_collisional_w
        Total damped power to collisional damping in bin [W].
    bin_damped_power_resonance_w
        Total damped power to resonant damping in bin [W].
    bin_edges
        Radial coordinate at edges of bins.
    bin_volume_m3
        Volume of bin [m^3].
    cumulative_power_collisional_w
        Cumulative damped power to collisional damping [W].
    cumulative_power_resonance_w
        Cumulative damped power to resonant damping [W].
    cumulative_power_w
        Cumulative damped power [W].
    cumulative_toroidal_current_a
        Cumulative toroidal current [A].
    edge_damped_power_w
        Total power damped outside bins [W].
    normalised_current_drive_efficiency
        Normalised current drive efficiency inside bin.
    normalised_current_drive_efficiency_luce
        Normalised current drive effiicency inside bin (Luce definition, large
        aspect ratio / on axis only).
    parallel_current_density_a_per_m2
        Flux surface averaged parallel current density [A.m^-2].
    power_density_collisional_w_per_m3
        Power density due to collisional damping in bin [W.m^-3].
    power_density_resonance_w_per_m3
        Power density due to resonant damping in bin [W.m^-3].
    power_density_w_per_m3
        Power density in bin [W.m^-3].
    rho_poloidal
        Root normalised poloidal flux at bin centre.
    rho_toroidal
        Root normalised toroidal flux at bin centre.
    toroidal_current_density_a_per_m2
        Toroidal current density [A.m^-2].
    total_damped_power_collisional_w
        Total damped power to collisional damping [W].
    total_damped_power_resonance_w
        Total damped power to resonant damping [W].
    total_damped_power_w
        Total damped power [W].
    total_toroidal_current_a
        Total toroidal current [A].
    """

    __slots__ = (
        "bin_area_m2",
        "bin_centres",
        "bin_damped_power_collisional_w",
        "bin_damped_power_resonance_w",
        "bin_edges",
        "bin_volume_m3",
        "cumulative_power_collisional_w",
        "cumulative_power_resonance_w",
        "cumulative_power_w",
        "cumulative_toroidal_current_a",
        "edge_damped_power_w",
        "normalised_current_drive_efficiency",
        "normalised_current_drive_efficiency_luce",
        "parallel_current_density_a_per_m2",
        "power_density_collisional_w_per_m3",
        "power_density_resonance_w_per_m3",
        "power_density_w_per_m3",
        "rho_poloidal",
        "rho_toroidal",
        "toroidal_current_density_a_per_m2",
        "total_damped_power_collisional_w",
        "total_damped_power_resonance_w",
        "total_damped_power_w",
        "total_toroidal_current_a",
    )

    def __init__(self, n_rho: int):
        """
        Inits LinearCurrentDriveOuput.

        Parameters
        ----------
        n_rho : int
            Number of values in flux coordinate mesh i.e. number of bins
            = n_rho - 1.
        """
        self.bin_edges = np.empty(n_rho)
        self.bin_centres = np.empty(n_rho - 1)
        self.bin_damped_power_resonance_w = np.empty(n_rho - 1)
        self.bin_damped_power_collisional_w = np.empty(n_rho - 1)

        # Flux function profiles.
        self.rho_poloidal = np.empty(n_rho)
        self.rho_toroidal = np.empty(n_rho)
        self.power_density_resonance_w_per_m3 = np.empty(n_rho)
        self.cumulative_power_resonance_w = np.empty(n_rho)
        self.power_density_collisional_w_per_m3 = np.empty(n_rho)
        self.cumulative_power_collisional_w = np.empty(n_rho)
        self.power_density_w_per_m3 = np.empty(n_rho)
        self.cumulative_power_w = np.empty(n_rho)
        self.power_density_collisional_w_per_m3 = np.empty(n_rho)
        self.parallel_current_density_a_per_m2 = np.empty(n_rho)
        self.toroidal_current_density_a_per_m2 = np.empty(n_rho)
        self.cumulative_toroidal_current_a = np.empty(n_rho)

        # 0d parameters.
        self.total_damped_power_resonance_w = 0.0
        self.total_damped_power_collisional_w = 0.0
        self.total_damped_power_w = 0.0
        self.total_toroidal_current_a = 0.0


class LinearCurrentDrive:
    """
    Linear current drive model using adjoint model.

    Attributes
    ----------
    options : OptionsLinearCurrentDrive
        Control parameters for linear current drive model.
    output : LinearCurrentDriveOutput
        Output values from linear current drive model.

    Methods
    -------
    configure_rho_grid
    """

    __slots__ = (
        "options",
        "output",
    )

    def __init__(
        self,
        options: OptionsLinearCurrentDrive,
    ):
        """
        Inits LinearCurrentDrive.

        Parameters
        ----------
        options : OptionsLinearCurrentDrive
            Control parameters for linear current drive model.
        """
        self.options = options
        self.output = LinearCurrentDriveOutput(self.options.n_rho)

    def configure_rho_grid(
        self, coordinate_coordinator: CoordinateCoordinator
    ):
        """
        Configure flux coordinate grid.

        Parameters
        ----------
        coordinate_coordinator : CoordinateCoordinator
            Coordinate coordinator object containing defined coordinate
            systems.
        """
        rho_toroidal_coordinate = coordinate_coordinator.coordinates[
            CoordinateSystem.RHO_TOROIDAL
        ]

        if self.options.rho_poloidal_grid:
            # Construct equispaced rho poloidal grid.
            self.output.rho_poloidal[:] = np.linspace(
                0.0, 1.0, self.output.rho_poloidal.size
            )

            # Calculate equivalent rho toroidal grid.
            self.output.rho_toroidal[:] = (
                rho_toroidal_coordinate.rho_spline_1_to_2(
                    self.output.rho_poloidal.reshape((-1, 1))
                )
            )

            self.output.bin_edges[:] = self.output.rho_poloidal
        else:
            # Construct equispaced rho poloidal grid.
            self.output.rho_toroidal[:] = np.linspace(
                0.0, 1.0, self.output.rho_toroidal.size
            )

            # Calculate equivalent rho poloidal grid.
            self.output.rho_poloidal[:] = (
                rho_toroidal_coordinate.rho_spline_2_to_1(
                    self.output.rho_poloidal.reshape((-1, 1))
                )
            )

            self.output.bin_edges[:] = self.output.rho_toroidal

        # Calculate centres of bins.
        self.output.bin_centres = 0.5 * (
            self.output.bin_edges[1:] + self.output.bin_edges[:-1]
        )

        # Calculate area and

    def bin_ray_data(self, ray_tracing_output: RayTracingOutput):
        """
        Collect ray data into flux surface bins.

        Parameters
        ----------
        ray_tracing_output : RayTracingOutput
            Output from ray tracing calculation.
        """
        time = ray_tracing_output.time_ns
        frac_resonance = ray_tracing_output.damping_fraction_resonance
        frac_collisional = ray_tracing_output.damping_fraction_collisional

        if self.options.rho_poloidal_grid:
            rho = ray_tracing_output.position[CoordinateSystem.RHO_POLOIDAL][
                :, 0
            ]
        else:
            rho = ray_tracing_output.position[CoordinateSystem.RHO_TOROIDAL][
                :, 0
            ]

        # Calculate damped power in each bin.
        output = self.output
        output.bin_damped_power_resonance_w.fill(0.0)
        output.bin_damped_power_collisional_w.fill(0.0)

        bin_edges = output.bin_edges
        idx_0, idx_1 = -1, -1

        for i in range(1, time.size):
            if rho[i] > output.bin_centres[-1]:
                idx_0 = bin_edges.size
                continue

            # Get intersection point of ray element with flux surfaces.
            if idx_0 == -1:
                idx_0 = bisect.bisect_left(bin_edges, rho[i - 1])

            idx_1 = bisect.bisect_left(bin_edges, rho[i])

            if idx_0 == idx_1:
                # Entire ray element in a single bin.
                output.bin_damped_power_resonance_w[idx_0] += max(
                    0.0, frac_resonance[i] - frac_resonance[i - 1]
                )
                output.cumulative_power_collisional_w[idx_0] += max(
                    0.0, frac_collisional[i] - frac_collisional[i - 1]
                )
            else:
                # Ray spans multiple bins.
                grad_resonance = abs(
                    (frac_resonance[i] - frac_resonance[i - 1])
                    / (rho[i] - rho[i - 1])
                )
                grad_collisional = abs(
                    (frac_collisional[i] - frac_collisional[i - 1])
                    / (rho[i] - rho[i - 1])
                )

                # First and last bins only have partial contribution.
                output.bin_damped_power_resonance_w[idx_0] += (
                    grad_resonance * abs(rho[i - 1] - bin_edges[idx_0])
                )
                output.bin_damped_power_resonance_w[idx_1] += (
                    grad_collisional * abs(rho[i] - bin_edges[idx_1 - 1])
                )

                for j in range(min(idx_0, idx_1) + 1, max(idx_0, idx_1)):
                    output.bin_damped_power_resonance_w[j] += (
                        grad_resonance * abs(bin_edges[j + 1] - bin_edges[j])
                    )
                    output.bin_damped_power_resonance_w[j] += (
                        grad_collisional * abs(bin_edges[j + 1] - bin_edges[j])
                    )

            # Start of next ray element same as end of this element.
            idx_0 = idx_1

        # Multiply fraction of total power damped by initial power in ray.
        initial_power = ray_tracing_output.initial_power_w
        output.bin_damped_power_resonance_w *= initial_power
        output.bin_damped_power_collisional_w *= initial_power

        # Calculate cumulative power profiles.
        output.cumulative_power_resonance_w[0] = 0.0
        output.cumulative_power_resonance_w[1:] = np.cumsum(
            output.bin_damped_power_resonance_w
        )
        output.cumulative_power_collisional_w[0] = 0.0
        output.cumulative_power_collisional_w[1:] = np.cumsum(
            output.bin_damped_power_collisional_w
        )
        output.cumulative_power_w = (
            output.cumulative_power_resonance_w
            + output.cumulative_power_collisional_w
        )

        # Calculate total damped power.
        output.total_damped_power_resonance_w = (
            output.cumulative_power_resonance_w[-1]
        )
        output.total_damped_power_collisional_w = (
            output.cumulative_power_collisional_w[-1]
        )
        output.total_damped_power_w = (
            output.total_damped_power_resonance_w
            + output.total_damped_power_collisional_w
        )

        # Calculate total power damped outside rho = 1.
        output.edge_damped_power_w = (
            ray_tracing_output.cumulative_damped_power_w[-1]
            - output.total_damped_power_w
        )

    def calculate(
        self,
        system_data: SystemData,
        ray_tracing_outputs: list[RayTracingOutput],
    ):
        """
        Run linear current drive model.

        Parameters
        ----------
        system_data : SystemData
            System data object containing description of plasma system.
        ray_tracing_outputs : list[RayTracingOutput]
            Ray tracing output for each ray.

        Raises
        ------
        ValueError
            System data is not a tokamak geometry.
        """
        # Require a tokamak geometry so various flux surface averaged
        # quantities are available.
        if type(system_data.magnetic) is not MagneticTokamak:
            raise ValueError(
                "Magnetic field topology must be 'tokamak' to use linear "
                "current drive model."
            )

        # Construct flux coordinate grids.
        self.configure_rho_grid(system_data.coordinate_coordinator)

        for ray_tracing_output in ray_tracing_outputs:
            # Calculate power density.
            self.bin_ray_data(ray_tracing_output)
