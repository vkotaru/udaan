"""M3: the analytical model and the MuJoCo backend fly one machine.

Behavioural checks on what the two backends share: the airframe and its
inertia frame, the actuator envelope, the loop rates and clock, every input
path, the ring geometry, the reset, the cable's sign, the tension floor, the
time-scale gains and the reference feasibility flag.
"""

from __future__ import annotations

import logging
import threading

import mujoco
import numpy as np
import pytest

from udaan.control.multi_quad_cspayload.payload_controller import bound_tilt
from udaan.core.defaults import (
    DEFAULT_FORMATION_CLEARANCE,
    DEFAULT_GENERATED_QUAD_INERTIA,
    DEFAULT_GENERATED_QUAD_MASS,
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_ROTOR_SPAN,
    MULTI_MAX_COMMANDED_TILT_DEG,
    MULTI_MIN_CABLE_TENSION_FRACTION,
)
from udaan.flatness import MultiQuadCsPayload
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco
from udaan.models.multi_quad_cspayload.mujoco import body_frame_inertia
from udaan.models.quadrotor.base import _ThrustCommand
from udaan.utils.assets import xml_model_generator
from udaan.utils.formation import (
    check_formation,
    default_formation_radius,
    min_formation_radius,
    neighbour_spacing,
)
from udaan.utils.trajectory import SpiralTrajectory

pytestmark = pytest.mark.mujoco

_E3 = np.array([0.0, 0.0, 1.0])


def _hold(target):
    return lambda t: (np.asarray(target, dtype=float), np.zeros(3), np.zeros(3))


def _mjcf_body_inertia(mdl):
    m = mdl._mjMdl.model
    return np.stack(
        [body_frame_inertia(m.body_inertia[b], m.body_iquat[b]) for b in mdl._quad_body_id]
    )


# ─── The airframe and its inertia frame ──────────────────────────────────


class TestOneAirframe:
    def test_backend_reads_the_body_frame_inertia(self):
        """MuJoCo's principal moments are rotated by body_iquat into the body frame."""
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        np.testing.assert_allclose(mdl.inertias, _mjcf_body_inertia(mdl), rtol=1e-12)
        # The generated airframe's principal frame is rotated, so diag(I) of the
        # file MuJoCo loaded is not the body-frame tensor.
        raw = mujoco.MjModel.from_xml_path(mdl._mjMdl.full_path)
        b = mdl._quad_body_id[0]
        assert not np.allclose(np.diag(raw.body_inertia[b]), mdl.inertias[0], rtol=1e-2)
        np.testing.assert_allclose(
            body_frame_inertia(raw.body_inertia[b], raw.body_iquat[b]), mdl.inertias[0], rtol=1e-9
        )

    def test_inertia_setter_makes_mujoco_fly_the_assigned_tensor(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        J = np.array([[0.004, 0.0005, 0.0], [0.0005, 0.006, 0.0], [0.0, 0.0, 0.010]])
        mdl.inertias = J
        np.testing.assert_allclose(_mjcf_body_inertia(mdl), np.tile(J, (3, 1, 1)), atol=1e-12)

    def test_mujoco_rotates_as_the_believed_inertia_says(self):
        """A torque pulse on MuJoCo gives the angular acceleration J⁻¹M of the model's J."""
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3, input="wrench")
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        hover = (mdl.masses[0] + mdl.payload_mass / 3) * 9.81
        M = np.array([0.02, -0.015, 0.01])
        u = np.tile([hover, *M], (3, 1))
        mdl.step(u)
        alpha = np.asarray(mdl.state.angular_velocities[0]) / mdl.dt
        expected = np.linalg.solve(mdl.inertias[0], M)
        np.testing.assert_allclose(alpha, expected, rtol=0.02)

    def test_analytical_defaults_are_the_generated_airframe(self):
        mj = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        base = MultiQuadCsPayloadBase(num_quadrotors=3)
        np.testing.assert_allclose(base.masses, mj.masses, rtol=1e-12)
        np.testing.assert_allclose(base.inertias, mj.inertias, rtol=1e-3)
        np.testing.assert_allclose(DEFAULT_GENERATED_QUAD_INERTIA, mj.inertias[0], rtol=1e-3)
        assert pytest.approx(mj.masses[0], rel=1e-12) == DEFAULT_GENERATED_QUAD_MASS

    def test_open_loop_roll_pulse_agrees(self):
        """Same wrench, same start: the two backends rotate the same airframe."""
        start = np.array([0.0, 0.0, 1.0])
        models = [
            MultiQuadCsPayloadBase(num_quadrotors=3, input="wrench"),
            MultiQuadCsPayloadMujoco(num_quadrotors=3, input="wrench"),
        ]
        R = []
        for mdl in models:
            mdl.reset(payload_position=start)
            hover = (mdl.masses[0] + mdl.payload_mass / 3) * 9.81
            for k in range(100):
                M = np.array([0.02, 0.0, 0.0]) if k < 10 else np.zeros(3)
                mdl.step(np.tile([hover, *M], (3, 1)))
            R.append(np.asarray(mdl.state.orientations[0]))
        angle = np.degrees(np.arccos(np.clip((np.trace(R[0].T @ R[1]) - 1) / 2, -1, 1)))
        assert angle < 0.5


# ─── Actuator envelope ───────────────────────────────────────────────────


class TestActuatorEnvelope:
    def test_generated_ctrlrange_is_the_defaults_envelope(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        ranges = np.asarray(mdl._mjMdl.model.actuator_ctrlrange).reshape(3, 4, 2)
        for r in ranges:
            np.testing.assert_allclose(r[0], [DEFAULT_MIN_THRUST, DEFAULT_MAX_THRUST])
            np.testing.assert_allclose(r[1:, 0], DEFAULT_MIN_TORQUE)
            np.testing.assert_allclose(r[1:, 1], DEFAULT_MAX_TORQUE)

    def test_mujoco_and_the_model_clip_at_the_same_moment(self):
        """A roll moment past the model's clip reaches MuJoCo clipped to the same value."""
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3, input="wrench")
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        hover = (mdl.masses[0] + mdl.payload_mass / 3) * 9.81
        mdl.step(np.tile([hover, 4.5, 0.0, 0.0], (3, 1)))
        applied = np.asarray(mdl._mjMdl.data.actuator_force).reshape(3, 4)
        np.testing.assert_allclose(applied[:, 1], min(4.5, DEFAULT_MAX_TORQUE[0]))

    def test_a_different_ctrlrange_is_adopted_with_a_warning(self, caplog):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=2)
        mdl._mjMdl.model.actuator_ctrlrange[:, :] *= 0.5
        with caplog.at_level(logging.WARNING):
            mdl._read_ctrlrange()
        assert "ctrlrange" in caplog.text
        assert mdl._max_thrust == pytest.approx(DEFAULT_MAX_THRUST / 2)
        np.testing.assert_allclose(mdl._max_torque, DEFAULT_MAX_TORQUE / 2)


# ─── Rates and clock ─────────────────────────────────────────────────────


class _CountingAttitude:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def compute(self, *args, **kwargs):
        self.calls += 1
        return self.inner.compute(*args, **kwargs)


class TestRatesAndClock:
    @pytest.mark.parametrize("cls", [MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco])
    def test_attitude_loop_closes_four_times_per_step_on_both(self, cls):
        mdl = cls(num_quadrotors=2)
        spies = [_CountingAttitude(c) for c in mdl.attitude_controllers]
        mdl.attitude_controllers = spies
        mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for _ in range(10):
            mdl.step(mdl.compute_default_input())
        assert [s.calls for s in spies] == [40, 40]
        assert mdl.physics_steps == 8
        assert mdl.attitude_rate_hz == pytest.approx(800.0)

    @pytest.mark.parametrize("cls", [MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco])
    def test_simulate_ends_on_the_requested_time(self, cls):
        mdl = cls(num_quadrotors=2)
        mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
        steps = []
        step = mdl.step

        def counted(u, desired_att=None):
            steps.append(1)
            step(u, desired_att=desired_att)

        mdl.step = counted
        mdl.simulate(2.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert len(steps) == 400
        assert mdl.t == 2.0

    def test_divergence_raises_instead_of_restarting_the_clock(self):
        class Tumble:
            def compute(self, t, state, thrust_force, desired_att=None):
                return 20.0, np.array([5.0, -5.0, 2.0])

        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        mdl.attitude_controllers = [Tumble() for _ in range(3)]
        outcome = {}

        def run():
            try:
                mdl.simulate(3.0)
                outcome["result"] = "returned"
            except RuntimeError as err:
                outcome["result"] = str(err)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout=60.0)
        assert not th.is_alive(), "simulate() did not return"
        assert "diverged" in outcome["result"]


# ─── Input paths ─────────────────────────────────────────────────────────


class TestInputPaths:
    @pytest.mark.parametrize("cls", [MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco])
    def test_every_input_flies_the_same_law(self, cls):
        traj = SpiralTrajectory(
            start=np.array([0.0, 0.0, 1.0]), radius=0.5, speed=1.0, n_loops=1, advance=0.5
        )
        out = {}
        for inp in ("acceleration", "wrench", "prop_forces"):
            mdl = cls(num_quadrotors=3, input=inp)
            mdl.payload_controller.setpoint = traj.get_full
            mdl.simulate(3.0, payload_position=traj.get(0.0)[0])
            out[inp] = mdl.state.payload_position.copy()
        np.testing.assert_allclose(out["wrench"], out["acceleration"], atol=1e-9)
        # Rotor forces round-trip through the allocation, then clip per rotor.
        np.testing.assert_allclose(out["prop_forces"], out["acceleration"], atol=1e-3)


# ─── Ring geometry ───────────────────────────────────────────────────────


class TestFormationGeometry:
    @pytest.mark.parametrize("nQ", [2, 3, 4, 5, 6, 10])
    def test_default_radius_clears_the_rotor_span(self, nQ):
        r = default_formation_radius(nQ)
        assert neighbour_spacing(nQ, r) >= DEFAULT_ROTOR_SPAN + DEFAULT_FORMATION_CLEARANCE - 1e-12
        assert r >= 0.5

    def test_rotor_span_is_the_generated_airframe(self, tmp_path):
        path = tmp_path / "one.xml"
        xml_model_generator.multi_quad_cspayload(
            num_quadrotors=1, filename=str(path), verbose=False
        )
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "quad0")
        reach = 0.0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                continue
            centre = np.linalg.norm(d.geom_xpos[g][:2] - d.xpos[body][:2])
            reach = max(reach, centre + m.geom_size[g][0])
        assert 2 * reach == pytest.approx(DEFAULT_ROTOR_SPAN, abs=1e-9)

    @pytest.mark.parametrize("nQ", [2, 3, 4, 5, 6])
    def test_default_ring_has_no_contact_at_reset(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=nQ, cable_lengths=1.5)
        mdl.reset()
        assert mdl._mjMdl.data.ncon == 0

    def test_ten_vehicles_fly_on_long_cables(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=10, cable_lengths=2.0)
        mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
        contacts = 0
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for _ in range(400):
            mdl.step(mdl.compute_default_input())
            contacts += int(mdl._mjMdl.data.ncon > 0)
        assert contacts == 0
        assert np.linalg.norm(mdl.state.payload_position - [0.0, 0.0, 1.0]) < 0.01

    def test_a_crowded_ring_is_refused(self):
        with pytest.raises(ValueError, match="rotor span"):
            MultiQuadCsPayloadBase(num_quadrotors=6, formation_radius=0.5)
        with pytest.raises(ValueError, match="rotor span"):
            MultiQuadCsPayloadMujoco(num_quadrotors=6, formation_radius=0.5)
        mdl = MultiQuadCsPayloadBase(num_quadrotors=6)
        with pytest.raises(ValueError, match="rotor span"):
            mdl.formation_radius = 0.6

    def test_a_radius_reaching_the_cable_is_refused(self):
        assert min_formation_radius(10) > 1.0
        with pytest.raises(ValueError, match="strictly below every cable length"):
            MultiQuadCsPayloadBase(num_quadrotors=10)
        with pytest.raises(ValueError, match="strictly below every cable length"):
            check_formation(3, 0.5, 0.5)

    @pytest.mark.parametrize("nQ", [2, 3])
    def test_long_cables_spawn_above_the_ground(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=nQ, cable_lengths=2.0)
        mdl.reset()
        assert mdl.state.payload_position[2] > 0.5
        assert mdl._mjMdl.data.ncon == 0


# ─── Reset ───────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_places_the_formation_from_the_current_lengths(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        mdl.cable_lengths = [1.0, 1.2, 1.0]
        mdl.reset()
        xL = mdl.state.payload_position
        chords = [np.linalg.norm(xL - p) for p in mdl.state.positions]
        np.testing.assert_allclose(chords, [1.0, 1.2, 1.0], atol=1e-9)
        hover = mdl.payload_controller.tension_policy.hover_directions
        for a, b in zip(hover, mdl.state.cable_attitudes, strict=False):
            assert np.dot(a, np.asarray(b)) == pytest.approx(1.0, abs=1e-9)

    def test_hetero_lengths_hover_without_slack(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        mdl.cable_lengths = [1.0, 1.2, 1.0]
        target = np.array([0.0, 0.0, 1.1])
        mdl.payload_controller.setpoint = _hold(target)
        mdl.reset(payload_position=target)
        worst = 0.0
        for _ in range(400):
            mdl.step(mdl.compute_default_input())
            worst = max(worst, float(np.linalg.norm(mdl.state.payload_position - target)))
        assert worst < 0.02

    @pytest.mark.parametrize("cls", [MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco])
    def test_payload_velocity_moves_the_whole_formation(self, cls):
        v = np.array([0.8, -0.3, 0.2])
        mdl = cls(num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]), payload_velocity=v)
        np.testing.assert_allclose(mdl.state.payload_velocity, v, atol=1e-12)
        for vi in mdl.state.velocities:
            np.testing.assert_allclose(vi, v, atol=1e-12)
        for w in mdl.state.cable_angular_velocities:
            np.testing.assert_allclose(np.asarray(w), np.zeros(3), atol=1e-12)

    def test_a_moving_start_matches_the_analytical_model(self):
        """Started on a moving reference, both backends fly the same run."""

        class Line:
            x0 = np.array([0.0, 0.0, 1.0])
            v = np.array([1.0, 0.5, 0.2])

            def get_full(self, t):
                z = np.zeros(3)
                return self.x0 + self.v * t, self.v.copy(), z, z, z

        line = Line()
        end = []
        for cls in (MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco):
            mdl = cls(num_quadrotors=3)
            mdl.payload_controller.setpoint = line.get_full
            mdl.simulate(1.0, payload_position=line.x0, payload_velocity=line.v)
            err = float(np.linalg.norm(mdl.state.payload_position - line.get_full(mdl.t)[0]))
            assert err < 0.01
            end.append(mdl.state.payload_position.copy())
        assert np.linalg.norm(end[0] - end[1]) < 0.005


# ─── Tendon stiffness ────────────────────────────────────────────────────


class TestTendon:
    def test_limit_time_constant_is_two_timesteps(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        h = mdl._mjMdl.model.opt.timestep
        np.testing.assert_allclose(
            mdl._mjMdl.model.tendon_solref_lim, np.tile([2 * h, 1.0], (3, 1))
        )

    def test_hover_stretch_is_sub_millimetre(self):
        mdl = MultiQuadCsPayloadMujoco(num_quadrotors=3)
        mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for _ in range(200):
            mdl.step(mdl.compute_default_input())
        stretch = np.asarray(mdl._mjMdl.data.ten_length) - mdl.cable_lengths
        assert np.max(stretch) < 2e-4


# ─── The analytical cable's sign ─────────────────────────────────────────


class TestCompressiveCable:
    def test_a_pushing_cable_is_counted_and_warned_once(self, caplog):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3, input="wrench")
        flip = np.diag([1.0, -1.0, -1.0])
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]), orientations=[flip] * 3)
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                mdl.step(np.tile([10.0, 0.0, 0.0, 0.0], (3, 1)))
        assert mdl.compressive_cable_steps == 3 * mdl.physics_steps
        assert np.all(mdl.cable_tensions() < 0.0)
        assert sum("pushed" in r.getMessage() for r in caplog.records) == 1
        mdl.reset()
        assert mdl.compressive_cable_steps == 0

    def test_hover_tension_is_the_weight_share(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
        mdl.simulate(1.0, payload_position=np.array([0.0, 0.0, 1.0]))
        hover = mdl.payload_controller.tension_policy.anchors
        np.testing.assert_allclose(mdl.cable_tensions(), np.linalg.norm(hover, axis=1), rtol=1e-3)
        assert mdl.compressive_cable_steps == 0


# ─── Tension floor ───────────────────────────────────────────────────────


class TestTensionFloor:
    def test_a_downward_command_keeps_the_thrust_up_the_cable(self, monkeypatch):
        """Payload far above its setpoint: the along-cable thrust stays at T_min, up the cable.

        The parallel channel alone: the commanded-tilt bound, which acts on this state's
        commands, is taken out here; the delivered commands are checked below."""
        from udaan.control.multi_quad_cspayload import payload_controller

        bound = payload_controller.bound_tilt
        monkeypatch.setattr(
            payload_controller, "bound_tilt", lambda u, _deg, _floor: np.array(u, dtype=float)
        )
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 3.0]))
        mdl.state.payload_velocity = np.array([0.0, 0.0, 3.0])
        mdl._sync_quad_states()
        ctrl = mdl.payload_controller
        ctrl.setpoint = _hold([0.0, 0.0, 0.0])
        u = ctrl.compute(0.0, mdl.state)
        floors = ctrl.min_tensions
        anchors = np.linalg.norm(ctrl.tension_policy.anchors, axis=1)
        np.testing.assert_allclose(floors, MULTI_MIN_CABLE_TENSION_FRACTION * anchors)
        for i in range(3):
            q = np.asarray(mdl.state.cable_attitudes[i])
            assert -float(u[i] @ q) == pytest.approx(floors[i], rel=1e-12)

        # What the attitude loops get: upward, within the bound.
        monkeypatch.setattr(payload_controller, "bound_tilt", bound)
        delivered = ctrl.compute(0.0, mdl.state)
        assert np.all(delivered[:, 2] >= floors - 1e-12)
        assert np.all(_tilt_deg(delivered) <= MULTI_MAX_COMMANDED_TILT_DEG + 1e-9)


# ─── Measured cable tension ──────────────────────────────────────────────


def _hovering(cls=MultiQuadCsPayloadBase, nQ=3, tf=1.0):
    mdl = cls(num_quadrotors=nQ)
    mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
    mdl.simulate(tf, payload_position=np.array([0.0, 0.0, 1.0]))
    return mdl


class TestCableTension:
    @pytest.mark.parametrize("cls", [MultiQuadCsPayloadBase, MultiQuadCsPayloadMujoco])
    def test_the_state_carries_the_measured_hover_tension(self, cls):
        mdl = _hovering(cls)
        hover = np.linalg.norm(mdl.payload_controller.tension_policy.anchors, axis=1)
        np.testing.assert_allclose(mdl.state.cable_tensions, hover, rtol=0.01)

    def test_a_reset_leaves_the_tension_unmeasured(self):
        mdl = _hovering(MultiQuadCsPayloadBase, tf=0.1)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        assert np.all(np.isnan(mdl.state.cable_tensions))


# ─── Commanded tilt bound ────────────────────────────────────────────────


def _tilt_deg(u):
    u = np.asarray(u, dtype=float)
    return np.degrees(np.arctan2(np.hypot(u[..., 0], u[..., 1]), u[..., 2]))


def _swing_release(nQ=4, length=0.5, angle_deg=30.0, input="acceleration", **ctrl):
    """The whole ring rotated rigidly about the payload by ``angle_deg`` about y, released at rest."""
    mdl = MultiQuadCsPayloadBase(
        num_quadrotors=nQ,
        cable_lengths=length,
        formation_radius=min_formation_radius(nQ),
        input=input,
    )
    for k, v in ctrl.items():
        setattr(mdl.payload_controller, k, v)
    mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
    mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    cables = [rot @ np.asarray(q) for q in mdl.state.cable_attitudes]
    mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]), cable_attitudes=cables)
    return mdl


class TestTiltBound:
    """Each thrust vector handed to the attitude loop is at most θ_max from vertical,
    with a vertical part of at least its cable's tension floor."""

    def test_the_default_bound_is_the_defaults_constant(self):
        ctrl = MultiQuadCsPayloadBase(num_quadrotors=3).payload_controller
        assert ctrl.max_tilt_deg == MULTI_MAX_COMMANDED_TILT_DEG == 50.0

    def test_the_bounded_command_is_upright_continuous_and_keeps_its_azimuth(self):
        """Random commands, including level and downward ones: the result is within
        θ_max, points up by at least the floor, keeps its horizontal direction, leaves
        a command inside the bound alone, and moves by no more than a few times a small
        change of its input, across u_z = 0 and u_z = floor too."""
        rng = np.random.default_rng(3)
        deg, floor = MULTI_MAX_COMMANDED_TILT_DEG, 0.07
        for u in rng.normal(scale=5.0, size=(2000, 3)):
            out = bound_tilt(u, deg, floor)
            assert _tilt_deg(out) <= deg + 1e-9
            assert out[2] >= floor
            h_in, h_out = u[:2], out[:2]
            if np.linalg.norm(h_out) > 1e-12:
                np.testing.assert_allclose(
                    h_out / np.linalg.norm(h_out), h_in / np.linalg.norm(h_in), atol=1e-12
                )
            if u[2] >= floor and _tilt_deg(u) <= deg:
                np.testing.assert_array_equal(out, u)
            elif np.hypot(u[0], u[1]) > np.tan(np.radians(deg)) * max(u[2], floor):
                # Clamped: on the bound, at the raised vertical part.
                assert _tilt_deg(out) == pytest.approx(deg, abs=1e-9)
                assert out[2] == pytest.approx(max(u[2], floor), abs=1e-12)
            d = rng.normal(scale=1e-6, size=3)
            assert np.linalg.norm(bound_tilt(u + d, deg, floor) - out) <= 4.0 * np.linalg.norm(d)
        for uz in (0.0, floor):
            lo = bound_tilt([10.0, 0.0, uz - 1e-9], deg, floor)
            hi = bound_tilt([10.0, 0.0, uz + 1e-9], deg, floor)
            assert np.linalg.norm(hi - lo) < 1e-6
        z = 2.0
        edge = z * np.tan(np.radians(deg))
        lo = bound_tilt([edge - 1e-9, 0.0, z], deg, floor)
        hi = bound_tilt([edge + 1e-9, 0.0, z], deg, floor)
        assert np.linalg.norm(hi - lo) < 1e-6
        np.testing.assert_array_equal(bound_tilt([0.0, 0.0, -3.0], deg, floor), [0.0, 0.0, floor])

    @pytest.mark.parametrize("inp", ["acceleration", "wrench", "prop_forces"])
    def test_a_swing_release_never_commands_past_the_bound(self, inp):
        """nQ = 4 on 0.5 m cables, released 30° off: the unbounded cascade commands
        past 60°; with the bound no command passes 50° and no airframe 55°, on every
        input path."""
        mdl = _swing_release(input=inp)
        worst_cmd, worst_tilt = 0.0, 0.0
        for _ in range(int(1.0 / mdl.dt)):
            u = mdl.compute_default_input()
            worst_cmd = max(worst_cmd, float(np.max(_tilt_deg(u))))
            mdl.step(u if inp == "acceleration" else _ThrustCommand(u))
            worst_tilt = max(
                worst_tilt,
                *(np.degrees(np.arccos(np.asarray(R)[2, 2])) for R in mdl.state.orientations),
            )
        assert worst_cmd <= MULTI_MAX_COMMANDED_TILT_DEG + 1e-9
        assert worst_tilt < 55.0

    def test_a_payload_above_its_setpoint_and_rising_is_brought_down(self):
        """Payload 2 m above its setpoint and rising at 3 m/s: the cascade asks for a net
        downward force, which the bound turns into small upward thrusts at 50°. The
        formation stops the climb within half a metre, and no rigid cable pushes."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.payload_controller.setpoint = _hold([0.0, 0.0, 1.0])
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 3.0]), payload_velocity=np.array([0, 0, 3.0])
        )
        peak = 3.0
        while mdl.t < 4.0:
            mdl.step(mdl.compute_default_input())
            peak = max(peak, float(mdl.state.payload_position[2]))
        assert peak < 3.6
        assert mdl.compressive_cable_steps == 0
        assert np.linalg.norm(mdl.state.payload_position - [0.0, 0.0, 1.0]) < 0.1

    def test_the_bound_is_validated(self):
        ctrl = MultiQuadCsPayloadBase(num_quadrotors=3).payload_controller
        for bad in (0.0, 90.0, -5.0):
            with pytest.raises(ValueError):
                ctrl.max_tilt_deg = bad
        with pytest.raises(ValueError):
            bound_tilt([0.0, 0.0, -1.0], 50.0, 0.0)
        with pytest.raises(ValueError):
            bound_tilt([1.0, 0.0, 1.0], 95.0, 0.1)


# ─── Gains ───────────────────────────────────────────────────────────────


class TestTimeScaleGains:
    @pytest.mark.parametrize(("nQ", "length"), [(3, 2.0), (5, 2.0), (4, 0.75)])
    def test_step_settles_across_cable_lengths(self, nQ, length):
        """A 0.5 m step settles, with the gains following the cable."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ, cable_lengths=length)
        start, target = np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.0, 1.0])
        mdl.payload_controller.setpoint = _hold(target)
        mdl.reset(payload_position=start)
        tilt = 0.0
        late = []
        for _k in range(int(12.0 / mdl.dt)):
            mdl.step(mdl.compute_default_input())
            tilt = max(
                tilt, *(np.degrees(np.arccos(np.asarray(R)[2, 2])) for R in mdl.state.orientations)
            )
            if mdl.t > 9.0:
                late.append(float(np.linalg.norm(mdl.state.payload_position - target)))
        assert max(late) < 0.01
        # A 0.5 m setpoint step tilts the airframes by up to about 36° on 0.75 m
        # cables (the payload loop at the slowest cable loop / 3.5); the bound
        # sits below the cascade's 50° commanded-tilt cap, so a step that needs
        # the cap fails here.
        assert tilt < 40.0


# ─── Reference feasibility ───────────────────────────────────────────────


def _spiral(speed):
    return SpiralTrajectory(
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.6,
        speed=speed,
        n_loops=2,
        advance=2.0,
        hold_time=0.5,
    )


class TestReferenceFeasibility:
    def test_flat_map_flags_a_non_upright_sample(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        fm = MultiQuadCsPayload.from_model(mdl)
        fast, slow = _spiral(2.0), _spiral(1.5)
        bad = fm.check_reference(fast.get_jet, fast.duration)
        assert not bad.feasible and bad.intervals
        assert np.nanmin(bad.min_thrust_z) <= 0.0
        assert fm.check_reference(slow.get_jet, slow.duration).feasible

    def test_recover_marks_the_sample(self):
        from udaan.flatness import Jet, MultiQuadCsPayloadFlats, yaw_jet

        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        fm = MultiQuadCsPayload.from_model(mdl)
        traj = _spiral(2.0)
        res = fm.check_reference(traj.get_jet, traj.duration)
        bad = float(res.times[int(np.argmax(~res.upright))])
        good = float(res.times[int(np.argmax(res.upright))])
        for t, want in ((bad, False), (good, True)):
            flats = MultiQuadCsPayloadFlats(
                x_L=Jet(np.asarray(traj.get_jet(t, 6), dtype=float)), psi=[yaw_jet(None, t)] * 3
            )
            ref, _ = fm.recover(flats)
            assert ref.upright is want

    def test_simulate_drops_the_feedforward_on_those_samples(self, caplog):
        traj = _spiral(2.0)
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        fm = MultiQuadCsPayload.from_model(mdl)
        expected = fm.check_reference(traj.get_jet, traj.duration).infeasible_samples
        mdl.payload_controller.setpoint = traj.get_full
        with caplog.at_level(logging.WARNING):
            mdl.simulate(traj.duration, payload_position=traj.get(0.0)[0])
        assert mdl.feedforward_drops == pytest.approx(expected, abs=2)
        assert sum("not upright" in r.getMessage() for r in caplog.records) == 1
