"""M4's contract with the rest of udaan: one law, one rate scheme, one machine.

Each class pins one property of :class:`MultiQuadRigidbodyBase` and its MuJoCo
backend by its behaviour: what a run does, what MuJoCo integrates, what the
closed loop settles to.

* :class:`TestInputPaths` — ``simulate()`` flies the same law on the
  ACCELERATION, WRENCH and PROP_FORCES paths.
* :class:`TestRates` — 200 Hz cascade, 800 Hz attitude loop, 1600 Hz physics on
  both backends; MuJoCo's timestep is ``dt/N``.
* :class:`TestClock` — the clock counts substeps.
* :class:`TestDivergenceGuard` — a diverged MuJoCo step raises.
* :class:`TestInertiaFrame` — the airframe inertia is read and written in the
  body frame.
* :class:`TestTendonStiffness` — the tendons' limit holds the cable length.
* :class:`TestSlackCounter` — one slack counter on both backends.
* :class:`TestAirframeDefaults` — the analytical airframe is the MJCF's.
* :class:`TestCableReferenceRates` — the cable loop follows the flat map's
  cable rates.
* :class:`TestUprightReference` — samples that need an airframe that is not
  upright are flagged, dropped and counted.
* :class:`TestTensionFloor` — the default policy floors every allocated tension.
* :class:`TestFormationGeometry` — the ring clears its airframes and spawns
  above the ground at every cable length.
* :class:`TestPayloadAttitudeDesign` — the payload attitude loop is placed at
  ``ω_L = ρ ω_p`` (``RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO``) on the payload's
  inertia.
* :class:`TestGainDesign` — the time-scale gain design, in closed loop.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import scipy.optimize

from udaan.control.multi_quad_rigidbody import MultiQuadRigidbodyController, cable_reference_rates
from udaan.core.defaults import (
    CABLE_CASCADE_DAMPING,
    DEFAULT_PAYLOAD_SPAWN_HEIGHT,
    DEFAULT_ROTOR_SPAN,
    GRAVITY,
    RIGID_MIN_CABLE_TENSION_FRACTION,
    RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO,
)
from udaan.flatness import MultiQuadRigidbody, WrenchAllocation
from udaan.flatness.allocation import MinimumTensionDistribution
from udaan.manif import SO3, TSO3, rodrigues_expm
from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyBase, MultiQuadRigidbodyMujoco
from udaan.utils.assets import multi_quad_rigidbody
from udaan.utils.trajectory import HelixTrajectory, SpiralTrajectory

_E3 = np.array([0.0, 0.0, 1.0])
_MODEL_LOGGER = "udaan.models.multi_quad_rigidbody.base"
BACKENDS = [
    MultiQuadRigidbodyBase,
    pytest.param(MultiQuadRigidbodyMujoco, marks=pytest.mark.mujoco),
]


class Hold:
    """A payload position held at ``x``, with the ``get_full`` jet that unlocks the feed-forward."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def get(self, t):
        return self.x, np.zeros(3), np.zeros(3)

    def get_full(self, t):
        return self.x, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)


def yaw_rate(rate):
    """A payload attitude reference turning at ``rate`` rad/s about ``e₃``."""
    return lambda t: (rodrigues_expm(rate * t * _E3), rate * _E3, np.zeros(3))


def angle_deg(R_a, R_b) -> float:
    c = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def make(cls, **kwargs):
    if cls is MultiQuadRigidbodyMujoco:
        kwargs.setdefault("render", False)
    return cls(**kwargs)


# ─── One law on every input path ──────────────────────────────────────


@pytest.mark.parametrize("cls", BACKENDS)
class TestInputPaths:
    def test_every_input_path_flies_the_acceleration_run(self, cls):
        """``simulate()`` closes the attitude loops inside ``step()`` on every path."""
        target = np.array([0.4, -0.2, 1.3])
        finals = {}
        for path in ("acceleration", "wrench", "prop_forces"):
            mdl = make(cls, num_quadrotors=4, input=path)
            mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
            mdl.simulate(tf=1.0, payload_position=np.array([0.0, 0.0, 1.0]))
            finals[path] = (mdl.state.payload_position.copy(), mdl.t)
        x_acc, t_acc = finals["acceleration"]
        assert np.linalg.norm(x_acc - np.array([0.0, 0.0, 1.0])) > 0.05
        for path in ("wrench", "prop_forces"):
            x, t = finals[path]
            assert t == t_acc
            np.testing.assert_allclose(x, x_acc, atol=1e-9)


# ─── Rates and clock ──────────────────────────────────────────────────


@pytest.mark.parametrize("cls", BACKENDS)
class TestRates:
    def test_attitude_updates_per_step(self, cls, monkeypatch):
        """Four attitude updates per 5 ms step on both backends: 800 Hz."""
        calls = []
        mdl = make(cls, num_quadrotors=3)
        original = type(mdl._att_controllers[0]).compute

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(type(mdl._att_controllers[0]), "compute", spy)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for _ in range(10):
            mdl.step(mdl.compute_default_input())
        assert len(calls) == 10 * 4 * mdl.num_quadrotors
        assert mdl.attitude_rate_hz == pytest.approx(800.0)
        assert mdl.physics_rate_hz == pytest.approx(1600.0)

    def test_the_rate_validation_refuses_a_fractional_substep(self, cls):
        with pytest.raises(ValueError, match="physics_rate_hz"):
            make(cls, num_quadrotors=3, physics_rate_hz=1500.0)
        with pytest.raises(TypeError, match="inner_loop_steps"):
            make(cls, num_quadrotors=3, inner_loop_steps=4)


@pytest.mark.mujoco
def test_mujoco_timestep_is_dt_over_n():
    mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
    assert mdl._mjMdl.model.opt.timestep == pytest.approx(mdl.dt / 8, abs=1e-15)
    before = mdl._mjMdl.data.time
    mdl.step(mdl.compute_default_input())
    assert mdl._mjMdl.data.time - before == pytest.approx(mdl.dt, abs=1e-15)


@pytest.mark.parametrize("cls", BACKENDS)
class TestClock:
    def test_the_clock_counts_substeps(self, cls):
        """At 1 ms substeps, 2000 steps end on 10 s exactly, and simulate(10) takes 2000."""
        mdl = make(cls, num_quadrotors=3, physics_rate_hz=1000.0, attitude_rate_hz=1000.0)
        mdl.payload_controller.setpoint = lambda t: (
            np.array([0, 0, 1.0]),
            np.zeros(3),
            np.zeros(3),
        )
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for _ in range(2000):
            mdl.step(mdl.compute_default_input())
        assert mdl.t == 10.0

        steps = []
        original = mdl.step

        def counting(u, desired_att=None):
            steps.append(1)
            return original(u, desired_att=desired_att)

        mdl.step = counting
        mdl.simulate(tf=10.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert len(steps) == 2000
        assert mdl.t == 10.0


# ─── MuJoCo: divergence, inertia, tendon ──────────────────────────────


@pytest.mark.mujoco
class TestDivergenceGuard:
    def test_a_diverged_step_raises(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        mdl._mjMdl.data.qvel[:] = 1e12
        with pytest.raises(RuntimeError, match="diverged"):
            mdl.step(mdl.compute_default_input())


def _mass_matrix_rotation_block(mdl, i):
    """MuJoCo's own joint-space inertia of airframe ``i``'s rotational DOFs: its body-frame J."""
    model, data = mdl._mjMdl.model, mdl._mjMdl.data
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    for k in range(model.nv):
        e = np.zeros(model.nv)
        e[k] = 1.0
        mujoco.mj_mulM(model, data, M[k], e)  # row k of the symmetric M
    adr = int(mdl._quad_qveladr[i]) + 3
    return M[adr : adr + 3, adr : adr + 3]


@pytest.mark.mujoco
class TestInertiaFrame:
    def test_the_model_carries_the_body_frame_tensor(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for i in range(3):
            np.testing.assert_allclose(
                mdl.inertias[i], _mass_matrix_rotation_block(mdl, i), atol=1e-12
            )
        # The generated airframe is longer in y than in x and flat: J_xx < J_yy < J_zz.
        J = np.diag(mdl.inertias[0])
        assert J[0] < J[1] < J[2]

    def test_an_assigned_tensor_is_what_mujoco_flies(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
        J = np.array([[6e-3, 1e-3, 0.0], [1e-3, 7e-3, 5e-4], [0.0, 5e-4, 1.1e-2]])
        mdl.inertias = J
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(_mass_matrix_rotation_block(mdl, 1), J, atol=1e-12)


@pytest.mark.mujoco
class TestTendonStiffness:
    def test_a_hovering_cable_holds_its_length(self):
        """The limit's time constant is two timesteps: under 0.1 mm of stretch at hover."""
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=4)
        target = np.array([0.0, 0.0, 1.0])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=2.0, payload_position=target)
        data, model = mdl._mjMdl.data, mdl._mjMdl.model
        stretch = data.ten_length[:4] - model.tendon_range[:4, 1]
        assert np.all(stretch > 0.0)  # loaded
        assert np.max(stretch) < 1e-4


# ─── Slack ────────────────────────────────────────────────────────────


@pytest.mark.mujoco
class TestSlackCounter:
    def test_a_slack_tendon_is_counted_and_reset_clears_it(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
        target = np.array([0.0, 0.0, 1.0])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=0.5, payload_position=target)
        assert mdl.slack_cable_steps == 0

        # Throw the payload up into its cables.
        adr = mdl._payload_qveladr
        mdl._mjMdl.data.qvel[adr + 2] = 2.0
        for _ in range(20):
            mdl.step(mdl.compute_default_input())
        assert mdl.slack_cable_steps > 0

        mdl.reset(payload_position=target)
        assert mdl.slack_cable_steps == 0


# ─── The analytical airframe ──────────────────────────────────────────


@pytest.mark.mujoco
def test_the_analytical_airframe_is_the_mjcf_airframe():
    base = MultiQuadRigidbodyBase(num_quadrotors=3)
    mj = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
    np.testing.assert_allclose(base.masses, mj.masses, rtol=1e-9)
    np.testing.assert_allclose(base.inertias, mj.inertias, rtol=1e-3)


# ─── Cable reference rates ────────────────────────────────────────────


@pytest.mark.slow
class TestCableReferenceRates:
    def test_a_constant_yaw_rate_is_tracked_without_lag(self):
        """Every cable sweeps a cone; the cable loop has to lead it, not damp it."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        hold = Hold([0.0, 0.0, 1.5])
        mdl.payload_controller.setpoint = hold.get
        mdl.payload_controller.attitude_setpoint = yaw_rate(0.3)
        mdl.reset(payload_position=hold.x)
        errors = []
        flat_map = MultiQuadRigidbody.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )
        while mdl.t < 8.0:
            ref = mdl._reference(flat_map, hold.get_full, mdl.t)
            u = mdl.compute_default_input(cable_reference_rates(ref))
            desired = [
                (ref.orientations[i], ref.angular_velocities[i], ref.angular_accelerations[i])
                for i in range(4)
            ]
            mdl.step(u, desired_att=desired)
            R_d = mdl.payload_controller.attitude_setpoint(mdl.t)[0]
            errors.append(angle_deg(mdl.state.payload_orientation, R_d))
        steady = np.asarray(errors[-400:])
        assert steady.mean() < 2.0

    def test_simulate_feeds_the_cable_rates(self, monkeypatch):
        """The rates reach the cascade through ``simulate()`` itself."""
        seen = []
        original = MultiQuadRigidbodyBase.compute_default_input

        def spy(self, cable_reference=None):
            seen.append(cable_reference is not None and np.any(np.abs(cable_reference[0]) > 0))
            return original(self, cable_reference)

        monkeypatch.setattr(MultiQuadRigidbodyBase, "compute_default_input", spy)
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        hold = Hold([0.0, 0.0, 1.5])
        mdl.payload_controller.setpoint = hold.get
        mdl.payload_controller.attitude_setpoint = yaw_rate(0.3)
        mdl.simulate(tf=0.5, payload_position=hold.x)
        assert len(seen) == 100 and all(seen)

    def test_the_shipped_spiral_never_commands_a_downward_thrust(self, monkeypatch):
        """The default spiral is inside every limit by the flat map; the closed loop
        tracks it with every thrust vector pointing up."""
        traj = SpiralTrajectory()
        commands = []
        original = MultiQuadRigidbodyBase.step

        def spy(self, u, desired_att=None):
            commands.append(np.asarray(u, dtype=float).reshape(-1, 3)[:, 2].min())
            return original(self, u, desired_att=desired_att)

        monkeypatch.setattr(MultiQuadRigidbodyBase, "step", spy)
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.payload_controller.setpoint = traj.get
        mdl.simulate(tf=10.0, payload_position=traj.get(0.0)[0])
        assert min(commands) > 0.0
        assert np.linalg.norm(mdl.state.payload_position - traj.get(mdl.t)[0]) < 0.05


# ─── Upright reference ────────────────────────────────────────────────


@pytest.mark.slow
class TestUprightReference:
    def test_a_whirling_reference_is_flagged_before_flight(self):
        """On 2 m cables the shipped spiral needs inverted airframes (the cable's whirl)."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4, cable_lengths=2.0)
        flat_map = MultiQuadRigidbody.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )
        traj = SpiralTrajectory()
        report = flat_map.check_reference(traj.get_full, 10.0)
        assert not report.feasible
        assert report.intervals
        assert "not upright" in str(report)

        short = MultiQuadRigidbodyBase(num_quadrotors=4)
        short_map = MultiQuadRigidbody.from_model(
            short, tension_policy=short.payload_controller.tension_policy
        )
        assert short_map.check_reference(traj.get_full, 10.0).feasible

    def test_simulate_drops_and_counts_the_samples(self, caplog):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4, cable_lengths=2.0)
        traj = SpiralTrajectory()
        mdl.payload_controller.setpoint = traj.get
        with caplog.at_level(logging.WARNING, logger=_MODEL_LOGGER):
            mdl.simulate(tf=10.0, payload_position=traj.get(0.0)[0])
        assert mdl.feedforward_drops > 0
        warnings = [
            r for r in caplog.records if "without the airframe feed-forward" in r.getMessage()
        ]
        assert len(warnings) == 1


# ─── Tension floor ────────────────────────────────────────────────────


def _floor_allocation(nQ, fraction=RIGID_MIN_CABLE_TENSION_FRACTION):
    mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
    policy = mdl.payload_controller.tension_policy
    assert isinstance(policy, MinimumTensionDistribution)
    return mdl, WrenchAllocation(mdl.attachment_points, policy=policy)


class TestTensionFloor:
    @pytest.mark.parametrize("nQ", [3, 4, 10])
    def test_the_split_is_the_closest_floored_null_coordinate(self, nQ):
        """Against a general-purpose QP on random wrenches: same split, every floor held."""
        mdl, alloc = _floor_allocation(nQ)
        bias = np.asarray(alloc.policy.bias_forces)
        d = bias / np.linalg.norm(bias, axis=1, keepdims=True)
        floors = alloc.policy.min_tensions
        N = alloc.null_basis
        lam_bar = N.T @ bias.reshape(-1)
        rng = np.random.default_rng(1)
        active_seen = 0
        for _ in range(20):
            w = alloc.reference_wrench + rng.normal(scale=[4, 4, 6, 1, 1, 1])
            tau = alloc.split(w)
            np.testing.assert_allclose(alloc.grasp @ tau.reshape(-1), w, atol=1e-9)
            along = np.einsum("ij,ij->i", tau, d)
            tau0 = alloc.grasp_pinv @ w
            cons = [
                {
                    "type": "ineq",
                    "fun": (
                        lambda lam, i=i, tau0=tau0: (
                            d[i] @ (tau0 + N @ lam)[3 * i : 3 * i + 3] - floors[i]
                        )
                    ),
                }
                for i in range(nQ)
            ]
            res = scipy.optimize.minimize(
                lambda lam: 0.5 * np.sum((lam - lam_bar) ** 2),
                lam_bar,
                jac=lambda lam: lam - lam_bar,
                constraints=cons,
                method="SLSQP",
                options={"ftol": 1e-14, "maxiter": 500},
            )
            if min(c["fun"](res.x) for c in cons) < -1e-6:
                continue  # infeasible floors: the closest split is checked only where one exists
            assert np.all(along >= floors - 1e-8)
            np.testing.assert_allclose(tau.reshape(-1), tau0 + N @ res.x, atol=1e-6)
            active_seen += int(np.any(along < floors + 1e-9))
        assert active_seen > 0, "no sample engaged a floor: the test proves nothing"

    def test_the_jet_matches_a_finite_difference_where_a_floor_is_active(self):
        mdl, alloc = _floor_allocation(4)
        bias = np.asarray(alloc.policy.bias_forces)
        d = bias / np.linalg.norm(bias, axis=1, keepdims=True)
        rng = np.random.default_rng(7)
        for _ in range(200):
            w0 = alloc.reference_wrench + rng.normal(scale=[4, 4, 6, 1, 1, 1])
            along = np.einsum("ij,ij->i", alloc.split(w0), d)
            if np.any(np.isclose(along, alloc.policy.min_tensions)):
                break
        else:
            pytest.fail("no sample engaged a floor")
        w1 = np.array([0.5, -0.3, 0.8, 0.1, 0.0, -0.2])
        h = 1e-6
        fd = (alloc.split(w0 + h * w1) - alloc.split(w0 - h * w1)) / (2 * h)
        jet = alloc.allocate_jet([w0, w1])
        np.testing.assert_allclose(np.array([j[1] for j in jet]), fd, atol=1e-6)

    @pytest.mark.slow
    def test_the_closed_loop_never_commands_below_the_floor(self):
        """nQ = 10, where the hover share is smallest: a 1 m lateral step."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=10)
        ctrl = mdl.payload_controller
        floors = ctrl.tension_policy.min_tensions
        bias = np.asarray(ctrl.tension_policy.bias_forces)
        d = bias / np.linalg.norm(bias, axis=1, keepdims=True)
        target = np.array([1.0, 0.0, 1.0])
        ctrl.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        lowest = np.inf
        while mdl.t < 3.0:
            accel_plus_g, moment, _ = ctrl.desired_wrench(mdl.t, mdl.state)
            R_L = np.asarray(mdl.state.payload_orientation)
            wrench = np.concatenate([R_L.T @ (ctrl._payload_mass * accel_plus_g), moment])
            tau = ctrl.allocation.split(wrench)
            lowest = min(lowest, float(np.min(np.einsum("ij,ij->i", tau, d) - floors)))
            mdl.step(mdl.compute_default_input())
        assert lowest >= -1e-9


# ─── Formation geometry ───────────────────────────────────────────────


@pytest.mark.mujoco
class TestFormationGeometry:
    def test_the_generator_refuses_airframes_that_would_touch(self, tmp_path):
        with pytest.raises(ValueError, match="clearance"):
            multi_quad_rigidbody(
                num_quadrotors=10,
                filename=str(tmp_path / "ring.xml"),
                formation_radius=0.5,
                verbose=False,
            )

    def test_the_default_ten_ring_has_no_contact_at_reset(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=10)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        mujoco.mj_forward(mdl._mjMdl.model, mdl._mjMdl.data)
        assert mdl._mjMdl.data.ncon == 0
        pos = np.array([p[:2] for p in mdl.state.positions])
        gaps = np.linalg.norm(pos[:, None] - pos[None], axis=2)[~np.eye(10, dtype=bool)]
        assert gaps.min() >= DEFAULT_ROTOR_SPAN + 0.1 - 1e-9

    @pytest.mark.parametrize("length", [0.6, 1.0, 2.0])
    def test_the_default_spawn_follows_the_cable_length(self, length):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=4, cable_lengths=length)
        mujoco.mj_forward(mdl._mjMdl.model, mdl._mjMdl.data)
        mdl._query_latest_state()
        assert mdl.state.payload_position[2] == pytest.approx(DEFAULT_PAYLOAD_SPAWN_HEIGHT)

    def test_the_base_model_refuses_the_same_ring(self):
        with pytest.raises(ValueError, match="clearance"):
            MultiQuadRigidbodyBase(num_quadrotors=10, formation_radius=0.5)
        mdl = MultiQuadRigidbodyBase(num_quadrotors=10)
        with pytest.raises(ValueError, match="clearance"):
            mdl.formation_radius = 0.5
        assert mdl.formation_radius > 0.6


# ─── Gain design ──────────────────────────────────────────────────────


def _payload_step(axis, length=1.0, sync=False, amplitude=0.1, tf=8.0):
    """Peak time (s) and overshoot (fraction) of a payload position step on the base model."""
    mdl = MultiQuadRigidbodyBase(num_quadrotors=4, cable_lengths=1.0 if sync else length)
    if sync:
        mdl.cable_lengths = length
    start = np.array([0.0, 0.0, 1.0])
    target = start + amplitude * np.eye(3)[axis]
    mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    mdl.reset(payload_position=start)
    times, response = [], []
    while mdl.t < tf:
        mdl.step(mdl.compute_default_input())
        times.append(mdl.t)
        response.append((mdl.state.payload_position[axis] - start[axis]) / amplitude)
    k = int(np.argmax(response))
    return times[k], response[k] - 1.0, mdl


def _second_order(omega, zeta=CABLE_CASCADE_DAMPING):
    """Peak time and overshoot of a second-order step at ``ω``, ``ζ``."""
    wd = omega * np.sqrt(1.0 - zeta**2)
    return np.pi / wd, float(np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta**2)))


def _payload_attitude_omega(length):
    """``ω_L = ρ ω_p = ρ · 2√(g/ℓ)/5`` (the cable below its ``ω_a/5`` cap)."""
    return RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO * (2.0 * np.sqrt(GRAVITY / length) / 5.0)


def _payload_attitude_step(axis, length, amplitude=0.1, tf=5.0, h=1e-3):
    """Peak time (s) and overshoot of the payload attitude loop alone on ``J_L``.

    The controller's own moment drives the rigid payload,
    ``J_L Ω̇ = M − Ω × J_L Ω``, from level to a step of ``amplitude`` rad about
    body ``axis``; the cables are left out.
    """
    ctrl = MultiQuadRigidbodyController(num_quadrotors=4, cable_lengths=length)
    J = ctrl._payload_inertia
    e = np.eye(3)[axis]
    R_d = rodrigues_expm(amplitude * e)
    ctrl.attitude_setpoint = lambda t: (R_d, np.zeros(3), np.zeros(3))
    R, Omega = np.eye(3), np.zeros(3)
    times, response = [], []
    for k in range(int(round(tf / h))):
        state = SimpleNamespace(
            payload_position=np.zeros(3),
            payload_velocity=np.zeros(3),
            payload_orientation=R,
            payload_angular_velocity=Omega,
        )
        _, M, _ = ctrl.desired_wrench(k * h, state)
        Omega = Omega + h * np.linalg.solve(J, M - np.cross(Omega, J @ Omega))
        R = R @ rodrigues_expm(Omega * h)
        times.append((k + 1) * h)
        response.append(float(np.dot(vee_log(R), e)) / amplitude)
    k = int(np.argmax(response))
    return times[k], response[k] - 1.0


def vee_log(R):
    """The rotation vector of ``R`` (small angles away from π)."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * w / (2.0 * np.sin(angle))


class TestPayloadAttitudeDesign:
    @pytest.mark.parametrize("length", [1.0, 2.0])
    @pytest.mark.parametrize("axis", [0, 2])
    def test_the_payload_attitude_loop_is_placed_at_omega_L(self, axis, length):
        """k_R = J_L ω_L², k_Ω = 2ζ J_L ω_L: on the payload alone a small step
        peaks at π/(ω_L√(1−ζ²)) with ζ = 0.7's overshoot, on roll and on yaw,
        and slows with the cable as ω_p does."""
        peak, overshoot = _payload_attitude_step(axis, length)
        t_pk, os = _second_order(_payload_attitude_omega(length))
        assert peak == pytest.approx(t_pk, rel=0.05)
        assert overshoot == pytest.approx(os, abs=0.01)


@pytest.mark.slow
class TestGainDesign:
    def test_the_payload_yaw_step_settles_on_two_metre_cables(self):
        """A 0.4 rad payload yaw step at ℓ = 2 m, nQ = 4, held in place, is inside
        1° of its reference from 5 s after the step on."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4, cable_lengths=2.0)
        x = np.array([0.0, 0.0, 1.0])
        R_d = rodrigues_expm(0.4 * _E3)
        mdl.payload_controller.setpoint = lambda t: (x, np.zeros(3), np.zeros(3))
        mdl.payload_controller.attitude_setpoint = lambda t: (
            R_d if t >= 1.0 else np.eye(3),
            np.zeros(3),
            np.zeros(3),
        )
        mdl.reset(payload_position=x)
        worst = 0.0
        while mdl.t < 8.0:
            mdl.step(mdl.compute_default_input())
            if mdl.t >= 6.0:
                worst = max(worst, angle_deg(mdl.state.payload_orientation, R_d))
        assert worst < 1.0

    def test_the_payload_stays_level_on_a_fast_helix(self, monkeypatch):
        """A horizontal helix at 2 m/s on 1 m cables (nQ = 4, level payload
        reference, feed-forward on) accelerates the payload at 5 m/s² sideways;
        the payload's attitude error stays under 2° RMS from the end of the
        initial hold on."""
        traj = HelixTrajectory(start=np.array([0.0, 0.0, 1.0]))
        errors = []
        original = MultiQuadRigidbodyBase.step

        def spy(self, u, desired_att=None):
            out = original(self, u, desired_att=desired_att)
            if self.t >= 1.0:
                errors.append(angle_deg(self.state.payload_orientation, np.eye(3)))
            return out

        monkeypatch.setattr(MultiQuadRigidbodyBase, "step", spy)
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.payload_controller.setpoint = traj.get_full
        mdl.simulate(tf=traj.duration, payload_position=traj.get(0.0)[0])
        assert np.sqrt(np.mean(np.square(errors))) < 2.0

    def test_the_vertical_loop_is_placed_at_omega_z(self):
        """ω_z = ω_a/25 = 1.6 rad/s at ζ = 0.7: a 2.75 s peak, 4.6 % overshoot."""
        peak, overshoot, _ = _payload_step(axis=2)
        t_pk, os = _second_order(40.0 / 25.0)
        assert peak == pytest.approx(t_pk, rel=0.1)
        assert overshoot == pytest.approx(os, abs=0.015)

    def test_the_lateral_loop_slows_with_the_cable(self):
        """ω_p = 2√(g/ℓ)/5: the lateral step's peak time grows ≈ √2 from 1 m to 2 m."""
        peak_1, overshoot_1, _ = _payload_step(axis=0, length=1.0)
        peak_2, overshoot_2, _ = _payload_step(axis=0, length=2.0)
        t_pk, _ = _second_order(2.0 * np.sqrt(GRAVITY) / 5.0)
        assert peak_1 == pytest.approx(t_pk, rel=0.1)
        assert 1.25 < peak_2 / peak_1 < 1.5
        assert overshoot_1 < 0.08 and overshoot_2 < 0.08

    def test_a_synced_model_flies_the_freshly_built_one(self):
        _, _, built = _payload_step(axis=0, length=2.0)
        _, _, synced = _payload_step(axis=0, length=2.0, sync=True)
        np.testing.assert_allclose(
            synced.state.payload_position, built.state.payload_position, atol=1e-12
        )

    def test_the_airframe_attitude_loop_is_placed_at_omega_a(self):
        """k_R = J ω_a², k_Ω = 2ζ J ω_a on the model's own airframe: a 5° roll step
        on the rigid body peaks at π/(ω_a√(1−ζ²)) with ζ = 0.7's overshoot."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        ctrl = mdl.attitude_controllers[0]
        J = mdl.inertias[0]
        Rd = rodrigues_expm(np.radians(5.0) * np.array([1.0, 0.0, 0.0]))
        R, Omega, h = np.eye(3), np.zeros(3), 1e-4
        thrust = Rd @ _E3 * 10.0
        roll = []
        for k in range(3000):
            _, M = ctrl.compute(k * h, (SO3(R), TSO3(Omega)), thrust)
            Omega = Omega + h * np.linalg.solve(J, np.asarray(M) - np.cross(Omega, J @ Omega))
            R = R @ rodrigues_expm(Omega * h)
            roll.append(np.arctan2(R[2, 1], R[2, 2]))
        roll = np.degrees(np.asarray(roll)) / 5.0
        k = int(np.argmax(roll))
        t_pk, os = _second_order(40.0)
        assert k * h == pytest.approx(t_pk, rel=0.1)
        assert roll[k] - 1.0 == pytest.approx(os, abs=0.02)
