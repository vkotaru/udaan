"""Tests for QuadrotorCsPayloadMujoco — unit + integration across cable models."""

import mujoco
import numpy as np
import pytest

from udaan.control.quadrotor_cspayload import QuadCSPayloadController
from udaan.core import defaults as D
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase, QuadrotorCsPayloadMujoco
from udaan.utils.trajectory import StepSetpoint

from .test_quadrotor_cspayload_base import (
    INPUT_PATHS,
    assert_rotor_force_path_matches,
    attitude_calls_per_step,
    fly_lissajous,
    probe_attitude_command,
)

# Cable model variants. (cable_model kwarg, descriptive name, payload tolerance [m])
# Tendon's spring-damper has a known steady-state offset, so its tolerance is looser.
CABLE_MODELS = [
    ("links", "links", 0.10),
    ("tendon", "tendon", 0.30),
]

# (payload start, payload target, description)
TRAJECTORIES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "hover at setpoint"),
    (np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), "climb 0.5m"),
    (np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), "lateral + climb"),
    (np.array([1.0, 1.0, 0.5]), np.array([0.0, 0.0, 1.0]), "diagonal + climb"),
]


class TestMujocoCsPayloadUnit:
    """Basic instantiation."""

    @pytest.mark.parametrize("cable_model", ["links", "tendon"])
    def test_instantiate(self, cable_model):
        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model=cable_model)
        assert mdl.state is not None

    def test_state_types(self):
        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model="links")
        mdl.reset()
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)
        assert isinstance(mdl.state.cable_attitude, S2)
        assert isinstance(mdl.state.cable_angular_velocity, TS2)

    @pytest.mark.parametrize("cable_model", ["bogus", "cable"])
    def test_invalid_cable_model_raises(self, cable_model):
        with pytest.raises(ValueError, match="valid choices: tendon, links"):
            QuadrotorCsPayloadMujoco(render=False, cable_model=cable_model)

    def test_the_default_cable_model_is_the_tendon(self):
        assert QuadrotorCsPayloadMujoco()._model_type.name == "TENDON"


@pytest.mark.integration
@pytest.mark.mujoco
class TestMujocoCsPayloadIntegration:
    """Closed-loop payload tracking for each cable model variant."""

    SIM_TIME = 6.0

    @pytest.mark.parametrize("cable_model,name,tol", CABLE_MODELS)
    @pytest.mark.parametrize("start,target,desc", TRAJECTORIES)
    def test_payload_converges(self, cable_model, name, tol, start, target, desc):
        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model=cable_model)
        mdl._payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=self.SIM_TIME, payload_position=start)
        pos_err = np.linalg.norm(mdl.state.payload_position - target)
        assert pos_err < tol, f"[{name}, {desc}] Payload position error {pos_err:.4f}m > {tol}m"


MJCF_CABLES = ["tendon", "links"]


@pytest.mark.mujoco
class TestMjcfMachine:
    """The MJCF is the source of the machine, and it is the analytical model's."""

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_backend_reads_the_mjcf(self, cable_model):
        mdl = QuadrotorCsPayloadMujoco(cable_model=cable_model)
        m = mdl._mjMdl.model
        assert mdl.mass == float(m.body_mass[mdl._mj_quad_index])
        assert mdl.payload_mass == float(m.body_mass[mdl._mj_payload_index])
        np.testing.assert_allclose(mdl.inertia, mdl._mjcf_inertia(mdl._mj_quad_index))

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_base_model_flies_the_mjcf_machine(self, cable_model):
        """``QuadrotorCsPayloadBase``'s defaults are the MJCF's airframe, payload and cable."""
        base = QuadrotorCsPayloadBase()
        mj = QuadrotorCsPayloadMujoco(cable_model=cable_model)
        assert base.mass == pytest.approx(mj.mass, abs=1e-12)
        np.testing.assert_allclose(base.inertia, mj.inertia, atol=1e-12)
        assert base.payload_mass == pytest.approx(mj.payload_mass, abs=1e-12)
        assert base.cable_length == pytest.approx(mj.cable_length, abs=1e-9)

    def test_the_link_chain_hangs_one_cable_length(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        mdl.reset(position=np.array([0.0, 0.0, 3.0]))
        chord = np.linalg.norm(mdl.state.payload_position - mdl.state.position)
        assert chord == pytest.approx(D.DEFAULT_M2_CABLE_LENGTH, abs=1e-9)

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_controllers_are_built_for_the_mjcf(self, cable_model):
        mdl = QuadrotorCsPayloadMujoco(cable_model=cable_model)
        ctrl = mdl.payload_controller
        assert (ctrl.mass_q, ctrl.mass_l) == (mdl.mass, mdl.payload_mass + mdl.cable_mass)
        assert ctrl.cable_length == mdl.cable_length
        np.testing.assert_array_equal(ctrl.inertia, mdl.inertia)
        np.testing.assert_array_equal(mdl.attitude_controller.inertia, mdl.inertia)

    def test_cable_mass(self):
        """The tendon is massless; the chain's links carry 10 g."""
        assert QuadrotorCsPayloadMujoco(cable_model="tendon").cable_mass == 0.0
        links = QuadrotorCsPayloadMujoco(cable_model="links")
        m = links._mjMdl.model
        chain = m.body_subtreemass[links._mj_quad_index] - m.body_mass[links._mj_quad_index]
        assert links.cable_mass == pytest.approx(chain - links.payload_mass, abs=1e-12)
        assert links.cable_mass == pytest.approx(0.010, abs=1e-12)

    def test_the_controller_carries_the_chain_mass_through_sync(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        ctrl = QuadCSPayloadController()
        ctrl.sync_from_model(mdl)
        assert ctrl.mass_l == pytest.approx(mdl.payload_mass + 0.010, abs=1e-12)

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_hover_on_the_setpoint_has_no_standing_offset(self, cable_model):
        """Starting at rest on the setpoint, the payload holds its altitude to 1 mm.

        The vertical loop has no integral action, so a weight the controller
        does not carry stands off as ``m g / k_z`` (4.3 cm for the chain's 10 g).
        """
        mdl = QuadrotorCsPayloadMujoco(cable_model=cable_model)
        target = np.array([0.0, 0.0, 1.0])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(8.0, payload_position=target.copy())
        assert abs(mdl.state.payload_position[2] - 1.0) < 0.001

    def test_cli_hover_on_the_tendon_ends_on_its_setpoint(self):
        """``udaan run quad-payload``'s run: default setpoint, start (−1, 2, 0.5), 10 s."""
        mdl = QuadrotorCsPayloadMujoco(cable_model="tendon")
        mdl.simulate(10.0, payload_position=np.array([-1.0, 2.0, 0.5]))
        # The controller's weight is the plant's: no standing altitude offset.
        assert abs(mdl.state.payload_position[2] - 1.0) < 0.005

    def test_the_inertia_tensor_is_read_in_the_body_frame(self):
        """A full tensor written through the setter (principal moments plus
        ``body_iquat``) reads back as the same tensor."""
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        R = SO3.from_two_vectors(np.array([0.3, -0.2, 1.0]), np.array([1.0, 0.4, 0.0]))
        J = np.asarray(R) @ np.diag([0.004, 0.006, 0.01]) @ np.asarray(R).T
        mdl.inertia = J
        np.testing.assert_allclose(mdl._mjcf_inertia(mdl._mj_quad_index), J, atol=1e-15)

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_actuator_ranges_are_the_model_limits(self, cable_model):
        ranges = QuadrotorCsPayloadMujoco(cable_model=cable_model)._mjMdl.model.actuator_ctrlrange
        np.testing.assert_allclose(ranges[4], [D.DEFAULT_MIN_THRUST, D.DEFAULT_MAX_THRUST])
        np.testing.assert_allclose(ranges[5:8, 0], D.DEFAULT_MIN_TORQUE)
        np.testing.assert_allclose(ranges[5:8, 1], D.DEFAULT_MAX_TORQUE)
        for i in range(4):
            np.testing.assert_allclose(
                ranges[i], [D.DEFAULT_PROP_MIN_FORCE, D.DEFAULT_PROP_MAX_FORCE]
            )

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_rotor_geometry_is_the_allocation_matrix(self, cable_model):
        mdl = QuadrotorCsPayloadMujoco(cable_model=cable_model)
        m = mdl._mjMdl.model
        for i in range(4):
            x, y, _ = m.site_pos[m.actuator_trnid[i, 0]]
            column = np.array([1.0, y, -x, m.actuator_gear[i, 5]])
            np.testing.assert_allclose(column, mdl.allocation_matrix[:, i], atol=1e-10)


@pytest.mark.mujoco
class TestMujocoRates:
    """M1's rate scheme on the MuJoCo backend."""

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_a_step_advances_exactly_dt(self, cable_model):
        mdl = QuadrotorCsPayloadMujoco(cable_model=cable_model)
        assert mdl._mjMdl.model.opt.timestep == mdl.dt / 8
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        hover = np.array([0.0, 0.0, (mdl.mass + mdl.payload_mass) * 9.81])
        for _ in range(200):
            mdl.step(hover)
        assert mdl.t == 200 * mdl.dt
        assert mdl._mjMdl.data.time == mdl.t

    @pytest.mark.parametrize("kwargs", INPUT_PATHS)
    def test_attitude_loop_runs_four_times_per_step_on_every_path(self, kwargs):
        assert (
            attitude_calls_per_step(QuadrotorCsPayloadMujoco(cable_model="tendon", **kwargs)) == 4
        )

    def test_the_attitude_zoh_keyword_is_refused(self):
        with pytest.raises(TypeError, match="attitude_rate_hz"):
            QuadrotorCsPayloadMujoco(attitude_zoh=True)

    def test_a_rate_that_does_not_divide_the_substeps_is_refused(self):
        with pytest.raises(ValueError, match="attitude_rate_hz"):
            QuadrotorCsPayloadMujoco(attitude_rate_hz=300.0)

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_every_path_flies_the_same_run(self, cable_model):
        target = np.array([0.0, 0.0, 1.0])
        final = []
        for kwargs in INPUT_PATHS:
            mdl = QuadrotorCsPayloadMujoco(cable_model=cable_model, **kwargs)
            mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
            mdl.simulate(2.0, payload_position=np.array([0.5, 0.0, 0.5]))
            final.append(mdl.state.payload_position.copy())
        np.testing.assert_allclose(final[1], final[0], atol=1e-6)
        np.testing.assert_allclose(final[2], final[0], atol=1e-6)

    @pytest.mark.parametrize("cable_model", MJCF_CABLES)
    def test_the_rotor_force_round_trip_flies_the_wrench_run_where_no_rotor_clips(
        self, cable_model
    ):
        assert_rotor_force_path_matches(
            lambda **kw: QuadrotorCsPayloadMujoco(cable_model=cable_model, **kw)
        )


@pytest.mark.mujoco
class TestLinkChainState:
    """The chain's reset and the payload state it reports."""

    def test_the_first_state_after_reset_is_the_placed_one(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        mdl.reset(payload_position=np.array([0.3, -0.2, 0.8]))
        np.testing.assert_allclose(mdl.state.payload_position, [0.3, -0.2, 0.8], atol=1e-9)
        np.testing.assert_allclose(mdl.state.position, [0.3, -0.2, 1.8], atol=1e-9)
        np.testing.assert_allclose(np.asarray(mdl.state.cable_attitude), [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(mdl.state.payload_velocity, np.zeros(3), atol=1e-12)

    def test_reset_refuses_the_payload_velocity(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        with pytest.raises(ValueError, match="cannot honour"):
            mdl.reset(payload_velocity=np.zeros(3))

    def test_reset_turns_the_chain_to_the_cable_state(self):
        """Airframe pose and rates, cable direction and rate all read back after reset."""
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        q = np.array([np.sin(0.5), 0.0, -np.cos(0.5)])
        omega = np.array([0.0, -1.5, 0.0])
        b3 = np.array([0.1, 0.05, 1.0]) / np.linalg.norm([0.1, 0.05, 1.0])
        R = SO3.from_two_vectors(b3, np.array([1.0, 0.0, 0.0]))
        v, Om = np.array([0.3, -0.2, 0.1]), np.array([0.2, 0.1, -0.3])
        mdl.reset(
            position=np.array([0.0, 0.0, 3.0]),
            velocity=v,
            orientation=R,
            angular_velocity=TSO3(Om),
            cable_attitude=S2(q),
            cable_angular_velocity=TS2(omega),
        )
        np.testing.assert_allclose(np.asarray(mdl.state.cable_attitude), q, atol=1e-12)
        np.testing.assert_allclose(np.asarray(mdl.state.cable_angular_velocity), omega, atol=1e-12)
        np.testing.assert_allclose(np.asarray(mdl.state.orientation), np.asarray(R), atol=1e-12)
        np.testing.assert_allclose(np.asarray(mdl.state.angular_velocity), Om, atol=1e-12)
        np.testing.assert_allclose(mdl.state.velocity, v, atol=1e-12)
        np.testing.assert_allclose(mdl.state.payload_velocity, v + np.cross(omega, q), atol=1e-12)

    def test_reset_refuses_two_placements(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="links")
        with pytest.raises(ValueError, match="not both"):
            mdl.reset(position=np.array([0.0, 0.0, 2.0]), payload_position=np.zeros(3))

    def test_payload_velocity_is_the_chains_in_a_vertical_climb(self):
        """Climbing straight up, the hanging chain moves rigidly with the airframe."""
        mdl = QuadrotorCsPayloadMujoco(cable_model="links", input="wrench")
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        climb = np.array([1.3 * (mdl.mass + mdl.payload_mass + 0.01) * 9.81, 0.0, 0.0, 0.0])
        worst = 0.0
        for _ in range(100):
            mdl.step(climb)
            worst = max(
                worst, float(np.linalg.norm(mdl.state.payload_velocity - mdl.state.velocity))
            )
        assert mdl.state.velocity[2] > 1.0
        assert worst < 1e-4


@pytest.mark.mujoco
class TestTendonLimit:
    def test_the_limit_time_constant_is_two_timesteps(self):
        mdl = QuadrotorCsPayloadMujoco(physics_rate_hz=800.0)
        np.testing.assert_allclose(mdl._mjMdl.model.tendon_solref_lim[0], [2 * mdl.dt / 4, 1.0])

    def test_the_cable_holds_its_length_in_a_swing(self):
        """Released 30° off vertical under hover thrust, the taut tendon stretches
        less than 0.1 mm."""
        mdl = QuadrotorCsPayloadMujoco(input="wrench")
        q = np.array([np.sin(np.pi / 6), 0.0, -np.cos(np.pi / 6)])
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]), cable_attitude=S2(q))
        hover = np.array([(mdl.mass + mdl.payload_mass) * 9.81, 0.0, 0.0, 0.0])
        stretch = 0.0
        for _ in range(300):
            mdl.step(hover)
            chord = np.linalg.norm(mdl.state.payload_position - mdl.state.position)
            stretch = max(stretch, chord - mdl.cable_length)
        assert stretch < 1e-4


@pytest.mark.mujoco
class TestCableRate:
    def test_the_cable_rate_is_the_rate_of_the_cable_direction(self):
        """``q̇ = ω × q`` matches the finite difference of ``q`` on a 0.5 m tendon."""
        mdl = QuadrotorCsPayloadMujoco(cable_model="tendon", input="wrench")
        mdl.cable_length = 0.5
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 1.0]), payload_velocity=np.array([0.4, 0.0, 0.0])
        )
        hover = np.array([(mdl.mass + mdl.payload_mass) * 9.81, 0.0, 0.0, 0.0])
        q0, dq0 = np.asarray(mdl.state.cable_attitude).copy(), mdl.state.dq().copy()
        mdl.step(hover)
        dq1 = mdl.state.dq().copy()
        fd = (np.asarray(mdl.state.cable_attitude) - q0) / mdl.dt
        np.testing.assert_allclose(fd, 0.5 * (dq0 + dq1), atol=2e-3)
        assert np.linalg.norm(dq0) == pytest.approx(0.8, rel=1e-6)


@pytest.mark.mujoco
class TestTendonReset:
    def test_reset_sets_the_cable_rate(self):
        """A cable rate passed to reset() reads back and the cable then turns at it."""
        mdl = QuadrotorCsPayloadMujoco(cable_model="tendon", input="wrench")
        q = np.array([np.sin(0.4), 0.0, -np.cos(0.4)])
        omega = np.array([0.0, 1.5, 0.0])
        v = np.array([0.3, -0.2, 0.1])
        mdl.reset(
            position=np.array([0.0, 0.0, 2.0]),
            velocity=v,
            cable_attitude=S2(q),
            cable_angular_velocity=TS2(omega),
        )
        np.testing.assert_allclose(np.asarray(mdl.state.cable_angular_velocity), omega, atol=1e-9)
        np.testing.assert_allclose(mdl.state.velocity, v, atol=1e-12)
        np.testing.assert_allclose(np.asarray(mdl.state.cable_attitude), q, atol=1e-9)

        hover = np.array([(mdl.mass + mdl.payload_mass) * 9.81, 0.0, 0.0, 0.0])
        mdl.step(hover)
        fd = (np.asarray(mdl.state.cable_attitude) - q) / mdl.dt
        np.testing.assert_allclose(fd, np.cross(omega, q), atol=0.03)

    @pytest.mark.parametrize("cable_model", ["tendon", "links"])
    def test_backends_start_from_the_analytical_models_moving_state(self, cable_model):
        """A moving reset gives the same state on the analytical model and MuJoCo."""
        q = np.array([np.sin(0.4), 0.0, -np.cos(0.4)])
        kwargs = {
            "position": np.array([0.0, 0.0, 2.0]),
            "velocity": np.array([0.3, -0.2, 0.1]),
            "cable_attitude": S2(q),
            "cable_angular_velocity": TS2(np.array([0.0, 1.5, 0.0])),
        }
        base = QuadrotorCsPayloadBase(input="wrench")
        mj = QuadrotorCsPayloadMujoco(cable_model=cable_model, input="wrench")
        base.reset(**kwargs)
        mj.reset(**kwargs)

        def velocities(m):
            return np.concatenate(
                [
                    m.state.velocity,
                    m.state.payload_velocity,
                    np.asarray(m.state.cable_angular_velocity),
                ]
            )

        np.testing.assert_allclose(velocities(mj), velocities(base), atol=1e-9)
        np.testing.assert_allclose(
            mj.state.payload_position, base.state.payload_position, atol=1e-9
        )
        for m in (base, mj):
            m.step(np.array([(m.mass + m.payload_mass) * 9.81, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(velocities(mj), velocities(base), atol=2e-3)

    def test_reset_refuses_a_payload_velocity_with_a_cable_rate(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="tendon")
        with pytest.raises(ValueError, match="not both"):
            mdl.reset(
                payload_velocity=np.zeros(3), cable_angular_velocity=TS2(np.array([0.0, 1.0, 0.0]))
            )


@pytest.mark.mujoco
class TestDivergence:
    def test_a_diverged_run_raises_instead_of_looping(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="tendon", input="wrench")
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        mdl._mjMdl.data.qvel[0] = np.nan
        with pytest.raises(RuntimeError, match="diverged"):
            mdl.step(np.array([8.0, 0.0, 0.0, 0.0]))


@pytest.mark.mujoco
class TestMujocoController:
    def test_the_feedforward_closes_the_attitude_gap(self):
        rms_ff, angle_ff = fly_lissajous(QuadrotorCsPayloadMujoco(cable_model="tendon"))
        rms_no, angle_no = fly_lissajous(
            QuadrotorCsPayloadMujoco(cable_model="tendon"),
            bind=lambda traj: lambda t: traj.get_full(t),
        )
        assert angle_ff < 0.5 < angle_no
        assert rms_ff < 0.6 * rms_no

    def test_a_drop_never_commands_an_inverted_airframe(self):
        mdl = QuadrotorCsPayloadMujoco(cable_model="tendon")
        drop = StepSetpoint(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.5]), 1.0)
        mdl.payload_controller.setpoint = drop.get
        angles, _ = probe_attitude_command(mdl)
        mdl.simulate(6.0, payload_position=np.array([0.0, 0.0, 2.0]))
        assert max(angles) < 30.0
        assert np.linalg.norm(mdl.state.payload_position - [0.0, 0.0, 0.5]) < 0.01


def test_divergence_warnings_are_the_ones_mujoco_resets_on():
    from udaan.models.quadrotor_cspayload.mujoco import _DIVERGENCE_WARNINGS

    assert mujoco.mjtWarning.mjWARN_BADQACC in _DIVERGENCE_WARNINGS
