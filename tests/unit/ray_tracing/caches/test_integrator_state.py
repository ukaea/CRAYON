"""
Unit tests for ray_tracing.caches.integrator_state
"""

# Standard imports
import logging

# Third party imports
import numpy as np
import numpy.testing as nptest

# Local imports
from crayon.ray_tracing.caches.integrator_state import State, StateDt

logger = logging.getLogger(__name__)


class TestState:
    """
    Unit tests for State.
    """

    @staticmethod
    def test_properties():
        """
        Test properties of State.
        """
        state = State()

        # Test optical depth.
        state.optical_depth_internal = 1.0
        state.optical_depth_external = 2.0

        nptest.assert_allclose(state.optical_depth, 3.0)

        # Test phase.
        state.eikonal_phase_x_rad = 3.0
        state.adiabatic_phase_rad = 4.0

        nptest.assert_allclose(state.phase_rad, 7.0)

        # Test power.
        state.initial_power_w = 5.0

        nptest.assert_allclose(state.power_w, 5.0 * np.exp(-3.0))

    @staticmethod
    def test_set_auxilliaries():
        """
        Test set auxiliary parameters.
        """
        state = State()
        state.set_auxilliaries(1.0, 2.0)

        nptest.assert_allclose(state.eikonal_phase_rad, 1.0)
        nptest.assert_allclose(state.initial_power_w, 2.0)

    @staticmethod
    def test_pack_unpack():
        """
        Test packing / unpacking State into state vector.
        """
        state = State()

        # Test pack.
        state.position_cartesian[:] = (1.0, 2.0, 3.0)
        state.wavevector_cartesian[:] = (4.0, 5.0, 6.0)
        state.optical_depth_internal = 7.0
        state.damping_fraction_resonance = 8.0
        state.damping_fraction_collisional = 9.0
        state.arc_length_m = 10.0
        state.eikonal_phase_rad = 11.0
        state.adiabatic_phase_rad = 12.0
        state.magnification = 13.0
        state.focusing_tensor[:, :] = (
            (14.0, 15.0, 16.0),
            (0.0, 17.0, 18.0),
            (0.0, 0.0, 19.0),
        )
        state.pack()

        actual_value = state.state_vector
        expected_value = np.arange(1, 20, dtype=float)

        nptest.assert_allclose(actual_value, expected_value)

        # Test unpack.
        state.unpack(1.0, -np.arange(1, 20, dtype=float))

        nptest.assert_allclose(state.time_ns, 1.0)
        nptest.assert_allclose(state.position_cartesian, (-1.0, -2.0, -3.0))
        nptest.assert_allclose(state.wavevector_cartesian, (-4.0, -5.0, -6.0))
        nptest.assert_allclose(state.optical_depth_internal, -7.0)
        nptest.assert_allclose(state.damping_fraction_resonance, -8.0)
        nptest.assert_allclose(state.damping_fraction_collisional, -9.0)
        nptest.assert_allclose(state.arc_length_m, -10.0)
        nptest.assert_allclose(state.eikonal_phase_rad, -11.0)
        nptest.assert_allclose(state.adiabatic_phase_rad, -12.0)
        nptest.assert_allclose(state.magnification, -13.0)
        nptest.assert_allclose(
            state.focusing_tensor,
            (
                (-14.0, -15.0, -16.0),
                (-15.0, -17.0, -18.0),
                (-16.0, -18.0, -19.0),
            ),
        )


class TestStateDt:
    """
    Unit tests for StateDt.
    """

    @staticmethod
    def test_properties():
        """
        Test StateDt properties.
        """
        state_dt = StateDt()
        state_dt.velocity_xk[:] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        nptest.assert_allclose(state_dt.velocity_x, (1.0, 2.0, 3.0))
        nptest.assert_allclose(state_dt.velocity_k, (4.0, 5.0, 6.0))

    @staticmethod
    def test_pack():
        """
        Test packing state_dt into state vector.
        """
        state_dt = StateDt()
        state_dt.velocity_xk[:] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        state_dt.damping_rate = 7.0
        state_dt.damping_fraction_resonance_dt = 8.0
        state_dt.damping_fraction_collisional_dt = 9.0
        state_dt.velocity = 10.0
        state_dt.eikonal_phase_dt = 11.0
        state_dt.adiabatic_phase_dt = 12.0
        state_dt.magnification_dt = 13.0
        state_dt.focusing_tensor_dt[:, :] = (
            (14.0, 15.0, 16.0),
            (0.0, 17.0, 18.0),
            (0.0, 0.0, 19.0),
        )
        state_dt.pack()

        actual_value = state_dt.state_vector_dt
        expected_value = np.arange(1, 20, dtype=float)

        nptest.assert_allclose(actual_value, expected_value)
