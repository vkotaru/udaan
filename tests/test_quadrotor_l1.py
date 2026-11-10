"""The quadrotor's two L1 adaptive controllers, flown on the MuJoCo backend."""

import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from udaan.control.quadrotor import GeometricL1AttitudeController
from udaan.control.quadrotor.geometric_l1_attitude import routh_bound
from udaan.manif import SO3

HOVER = np.array([0.0, 0.0, 2.0])


class TestAdaptationCoupling:
    """``c`` couples the attitude error into the adaptation law. Above
    ``k_Ω(k_R + Γ)/Γ`` the predictor–adaptation loop is unstable."""

    def test_bound_and_default_at_the_default_gains(self):
        """Pinned, not recomputed: ``k_Ω(k_R + Γ)/Γ`` at ``k_R = [2.4, 2.4, 1.35]``,
        ``k_Ω = [0.35, 0.35, 0.225]``, ``Γ = 100``."""
        ctrl = GeometricL1AttitudeController(inertia=np.diag([0.0053, 0.0049, 0.0098]))
        bound = routh_bound(ctrl._gains.kp, ctrl._gains.kd, ctrl._Gamma)
        np.testing.assert_allclose(bound, [0.3584, 0.3584, 0.2280375], rtol=1e-12)
        np.testing.assert_allclose(ctrl.c, [0.1792, 0.1792, 0.11401875], rtol=1e-12)

    @pytest.mark.parametrize("J", [5e-3, 1e-1])
    def test_bound_is_where_the_linearised_loop_loses_stability(self, J):
        """Roots of ``J s³ + k_Ω s² + (k_R + Γ) s + Γc/J``, the linearised
        predictor–adaptation loop of one axis: all in the left half-plane just
        inside the bound, one in the right half-plane just outside it — at two
        inertias 20× apart, because the bound does not depend on ``J``."""
        ctrl = GeometricL1AttitudeController(inertia=np.eye(3) * J)
        kR, kO, G = ctrl._gains.kp, ctrl._gains.kd, ctrl._Gamma
        bound = routh_bound(kR, kO, G)
        for axis in range(3):

            def max_real(c, axis=axis):
                return np.roots([J, kO[axis], kR[axis] + G[axis], G[axis] * c / J]).real.max()

            assert max_real(0.99 * bound[axis]) < 0.0
            assert max_real(1.01 * bound[axis]) > 0.0

    def test_default_follows_a_gain_change(self):
        ctrl = GeometricL1AttitudeController(inertia=np.eye(3) * 5e-3)
        ctrl._gains.kd = ctrl._gains.kd * 2.0
        np.testing.assert_allclose(
            ctrl.c, 0.5 * routh_bound(ctrl._gains.kp, ctrl._gains.kd, ctrl._Gamma)
        )

    def test_an_explicit_c_past_the_bound_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            GeometricL1AttitudeController(inertia=np.eye(3) * 5e-3, c=1.0)
        assert "Routh stability bound" in caplog.text

    def test_an_explicit_c_inside_the_bound_is_kept_quietly(self, caplog):
        with caplog.at_level(logging.WARNING):
            ctrl = GeometricL1AttitudeController(inertia=np.eye(3) * 5e-3, c=0.05)
        np.testing.assert_allclose(ctrl.c, 0.05)
        assert caplog.text == ""


def _fly_l1_attitude(torque=None, duration=3.0):
    from udaan.models.quadrotor import QuadrotorMujoco

    mdl = QuadrotorMujoco(render=False)
    mdl.attitude_controller = GeometricL1AttitudeController(inertia=mdl.inertia)
    mdl.position_controller.setpoint = lambda t: (HOVER, np.zeros(3), np.zeros(3))
    tilt = SO3(Rotation.from_rotvec(np.radians(5.0) * np.array([1.0, 0, 0])).as_matrix())
    mdl.reset(position=HOVER, orientation=tilt)
    if torque is not None:
        mdl._mjMdl.data.xfrc_applied[1, 3:] = torque
    sigma, moment = [], []
    att = mdl.attitude_controller
    compute = att.compute

    def recorded(*args, **kwargs):
        f, M = compute(*args, **kwargs)
        sigma.append(att.sigma_hat.copy())
        moment.append(np.array(M))
        return f, M

    att.compute = recorded
    while mdl.t < duration:
        mdl.step(mdl.position_controller.compute(mdl.t, (mdl.state.position, mdl.state.velocity)))
    late = slice(len(sigma) // 2, None)
    return np.array(sigma), np.array(moment)[late]


@pytest.mark.mujoco
class TestL1AttitudeOnMujoco:
    def test_estimate_stays_small_with_no_disturbance(self):
        """From a 5° start with no disturbance σ̂ stays small and the moment steady."""
        sigma, moment = _fly_l1_attitude()
        assert np.abs(sigma).max() < 0.05
        assert moment.std(axis=0).max() < 1e-3

    def test_estimates_a_constant_torque(self):
        sigma, moment = _fly_l1_attitude(torque=np.array([0.1, 0.0, 0.0]))
        late = sigma[len(sigma) // 2 :]
        np.testing.assert_allclose(late.mean(axis=0), [0.1, 0.0, 0.0], atol=5e-3)
        assert moment.std(axis=0).max() < 1e-3


# ─── PositionL1Controller ─────────────────────────────────────────────


def _double_integrator(controller, setpoint, mass, steps, dt=0.005, disturbance=np.zeros(3)):
    """Fly ``controller`` on an exact point-mass plant; yield ``(u, state)`` per step."""
    from udaan.core.defaults import GRAVITY

    p, v = setpoint(0.0)[0].copy(), np.zeros(3)
    g = np.array([0.0, 0.0, GRAVITY])
    for k in range(steps):
        t = k * dt
        u = controller.compute(t, (p.copy(), v.copy()))
        acc = u / mass - g + disturbance
        p, v = p + v * dt + 0.5 * acc * dt**2, v + acc * dt
        yield u, p


class TestPositionL1Law:
    MASS = 0.75

    def test_is_exactly_pd_with_nothing_to_adapt_to(self):
        """On a plant that matches its own model, the predictor is exact,
        ``σ̂ ≡ 0``, and L1 *is* PD — on a moving reference, every step."""
        from udaan.control.quadrotor import PositionL1Controller, PositionPDController
        from udaan.utils.trajectory import CrazyTrajectory

        traj = CrazyTrajectory(center=np.array([0.0, 0.0, 3.0]))
        l1 = PositionL1Controller(mass=self.MASS, setpoint=traj.get)
        pd = PositionPDController(mass=self.MASS, setpoint=traj.get)
        worst = 0.0
        for u, _ in _double_integrator(l1, traj.get, self.MASS, 2000):
            u_pd = pd.compute(l1._last_t, (l1._last_x[:3], l1._last_x[3:]))
            worst = max(worst, float(np.abs(u - u_pd).max()))
        # Equal to round-off: T = t - t_last is not bit-identical to the plant's dt.
        assert worst < 1e-12
        assert np.abs(l1.lpf_delta).max() < 1e-12

    def test_is_exactly_pd_on_a_plant_that_thrusts_along_b3(self):
        """The predictor's realised thrust ``(u·b₃)b₃``, given ``R``, matches a
        plant that can only push along a fixed tilted ``b₃``: no disturbance,
        so L1 is PD to round-off even though ``u`` is never fully delivered."""
        from scipy.spatial.transform import Rotation as Rot

        from udaan.control.quadrotor import PositionL1Controller, PositionPDController
        from udaan.core.defaults import GRAVITY

        R = Rot.from_rotvec([0.3, -0.2, 0.0]).as_matrix()
        b3 = R[:, 2]
        hover = lambda t: (HOVER, np.zeros(3), np.zeros(3))
        l1 = PositionL1Controller(mass=self.MASS, setpoint=hover)
        pd = PositionPDController(mass=self.MASS, setpoint=hover)
        p, v, dt, worst = HOVER.copy(), np.zeros(3), 0.005, 0.0
        for k in range(1000):
            t = k * dt
            u = l1.compute(t, (p.copy(), v.copy(), R))
            worst = max(worst, float(np.abs(u - pd.compute(t, (p, v))).max()))
            acc = (u @ b3) * b3 / self.MASS - np.array([0.0, 0.0, GRAVITY])
            p, v = p + v * dt + 0.5 * acc * dt**2, v + acc * dt
        assert worst < 1e-9

    @pytest.mark.parametrize(
        "limits, saturated_is_disturbance", [(None, False), ((-np.inf, np.inf), True)]
    )
    def test_thrust_saturation_is_not_read_as_a_disturbance(self, limits, saturated_is_disturbance):
        """A plant that clips collective thrust at 20 N, flown at a target far
        enough below that PD asks for more. With the predictor clipping the same
        way, L1 stays PD; told the thrust is unlimited, it reads the clipped
        part as a disturbance."""
        from udaan.control.quadrotor import PositionL1Controller, PositionPDController
        from udaan.core.defaults import DEFAULT_MAX_THRUST, GRAVITY

        target = lambda t: (HOVER + [0.0, 0.0, 6.0], np.zeros(3), np.zeros(3))
        kwargs = {} if limits is None else {"thrust_limits": limits}
        l1 = PositionL1Controller(mass=self.MASS, setpoint=target, **kwargs)
        pd = PositionPDController(mass=self.MASS, setpoint=target)
        R, p, v, dt, worst, saturated = np.eye(3), HOVER.copy(), np.zeros(3), 0.005, 0.0, 0
        for k in range(400):
            t = k * dt
            u = l1.compute(t, (p.copy(), v.copy(), R))
            worst = max(worst, float(np.abs(u - pd.compute(t, (p, v))).max()))
            f = u @ R[:, 2]
            saturated += f > DEFAULT_MAX_THRUST
            acc = np.clip(f, 0.0, DEFAULT_MAX_THRUST) * R[:, 2] / self.MASS - [0.0, 0.0, GRAVITY]
            p, v = p + v * dt + 0.5 * acc * dt**2, v + acc * dt
        assert saturated > 20  # the case actually exercises the clip
        assert (worst > 1e-3) is saturated_is_disturbance

    @pytest.mark.parametrize("dt", [0.005, 0.02])
    def test_a_constant_disturbance_is_estimated_exactly(self, dt):
        """The predictor's σ̂ settles one step late at ``aT/(e^{aT} - 1)`` of a
        constant disturbance; the matched estimate δ̂ scales that back by
        ``(e^{aT} - 1)/(aT)`` and so equals the disturbance force. Deadbeat at
        any step: no adaptation gain, so no Euler step-size limit."""
        from udaan.control.quadrotor import PositionL1Controller

        hover = lambda t: (HOVER, np.zeros(3), np.zeros(3))
        l1 = PositionL1Controller(mass=self.MASS, setpoint=hover)
        d = np.array([1.0, 0.0, -0.5])  # m/s²
        *_, (_, p) = _double_integrator(l1, hover, self.MASS, 2000, dt=dt, disturbance=d)
        aT = l1.predictor_pole * dt
        np.testing.assert_allclose(l1.sigma_hat[3:], d * aT / np.expm1(aT), rtol=1e-9)
        np.testing.assert_allclose(l1.delta, self.MASS * d, rtol=1e-9, atol=1e-12)
        assert np.linalg.norm(p - HOVER) < 1e-6  # and the PD loop is left nothing

    def test_bandwidths_are_derived_from_the_gains(self):
        from udaan.control.quadrotor import PositionL1Controller
        from udaan.control.quadrotor.position_l1 import pd_bandwidth

        l1 = PositionL1Controller(mass=self.MASS)
        wb = pd_bandwidth(l1._gains.kp, l1._gains.kd, self.MASS)
        assert wb == pytest.approx(12.896, abs=1e-3)
        assert l1.predictor_pole == pytest.approx(5 * wb)
        assert l1.lpf_cutoff == pytest.approx(np.sqrt(5) * wb)

    def test_no_python_control_dependency(self):
        import inspect

        import udaan.control.quadrotor.position_l1 as module

        assert "import control" not in inspect.getsource(module)


def _fly_position(cls, mass_scale=1.0, force=None, setpoint=None, duration=10.0):
    import warnings

    from udaan.models.quadrotor import QuadrotorMujoco

    setpoint = setpoint or (lambda t: (HOVER, np.zeros(3), np.zeros(3)))
    mdl = QuadrotorMujoco(render=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # constructing the controller must not warn
        mdl.position_controller = cls(mass=mass_scale * mdl.mass, setpoint=setpoint)
    mdl.reset(position=setpoint(0.0)[0])
    if force is not None:
        mdl._mjMdl.data.xfrc_applied[1, :3] = force
    err = []
    while mdl.t < duration:
        # The orientation rides along exactly as QuadrotorBase.simulate() passes it.
        state = (mdl.state.position, mdl.state.velocity, mdl.state.orientation)
        mdl.step(mdl.position_controller.compute(mdl.t, state))
        err.append(np.linalg.norm(mdl.state.position - setpoint(mdl.t)[0]))
    err = np.array(err)
    assert np.isfinite(err).all()
    return err


@pytest.mark.mujoco
class TestPositionL1OnMujoco:
    @pytest.mark.parametrize(
        "case", [{"force": np.array([1.0, 0.0, 0.0])}, {"mass_scale": 0.8}], ids=["force", "mass"]
    )
    def test_removes_the_steady_error_pd_is_left_with(self, case):
        """A 1 N side force leaves PD 0.244 m off and a 20 % mass under-estimate
        0.181 m low. L1's unbiased estimate removes both, to under 1 mm."""
        from udaan.control.quadrotor import PositionL1Controller, PositionPDController

        pd = _fly_position(PositionPDController, **case)
        l1 = _fly_position(PositionL1Controller, **case)
        late = slice(len(pd) // 2, None)
        assert pd[late].mean() > 0.15
        assert l1[late].mean() < 1e-3

    def test_matches_pd_at_nominal_hover(self):
        from udaan.control.quadrotor import PositionL1Controller

        assert _fly_position(PositionL1Controller).max() < 1e-3

    def test_tracks_a_moving_reference_as_well_as_pd(self):
        """With nothing to reject, L1 must cost nothing: predicting with the
        thrust the attitude delivers, it stays within 1 % of PD."""
        from udaan.control.quadrotor import PositionL1Controller, PositionPDController
        from udaan.utils.trajectory import CrazyTrajectory

        traj = CrazyTrajectory(center=np.array([0.0, 0.0, 3.0]))
        pd = _fly_position(PositionPDController, setpoint=traj.get, duration=20.0)
        l1 = _fly_position(PositionL1Controller, setpoint=traj.get, duration=20.0)
        assert np.sqrt(np.mean(l1**2)) < 1.01 * np.sqrt(np.mean(pd**2))


# ─── Repeated runs ────────────────────────────────────────────────────


@pytest.mark.mujoco
@pytest.mark.parametrize("which", ["pd", "l1_position", "l1_attitude"])
def test_a_second_simulate_repeats_the_first(which):
    """``QuadrotorBase.reset()`` resets every controller that has ``reset()``, so a
    second ``simulate()`` repeats the first exactly."""
    from udaan.control.quadrotor import PositionL1Controller
    from udaan.models.quadrotor import QuadrotorMujoco

    target = np.array([1.0, 0.0, 2.0])
    mdl = QuadrotorMujoco(render=False)
    if which == "l1_position":
        mdl.position_controller = PositionL1Controller(mass=mdl.mass)
    elif which == "l1_attitude":
        mdl.attitude_controller = GeometricL1AttitudeController(inertia=mdl.inertia)
    mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    finals = []
    for _ in range(2):
        mdl.simulate(3.0, position=HOVER)
        finals.append(mdl.state.position.copy())
    np.testing.assert_allclose(finals[1], finals[0], atol=1e-12)
    assert np.linalg.norm(finals[0] - target) < 0.01


@pytest.mark.parametrize("cls", ["position", "attitude"])
def test_a_controller_that_sees_time_run_backwards_starts_over(cls):
    """The controller's own guard, for a caller that never calls ``reset()``."""
    from udaan.control.quadrotor import PositionL1Controller

    hover = lambda t: (HOVER, np.zeros(3), np.zeros(3))
    if cls == "position":
        ctrl = PositionL1Controller(mass=0.75, setpoint=hover)
        first = [ctrl.compute(t, (HOVER + 0.1, np.zeros(3))) for t in (0.0, 0.005, 0.01)]
        again = [ctrl.compute(t, (HOVER + 0.1, np.zeros(3))) for t in (0.0, 0.005, 0.01)]
    else:
        ctrl = GeometricL1AttitudeController(inertia=np.eye(3) * 5e-3)
        tilt = SO3(Rotation.from_rotvec([0.1, 0.0, 0.0]).as_matrix())
        from udaan.manif import TSO3

        state = (tilt, TSO3(np.zeros(3)))
        thrust = np.array([0.0, 0.0, 7.0])
        first = [np.array(ctrl.compute(t, state, thrust)[1]) for t in (0.0, 0.005, 0.01)]
        again = [np.array(ctrl.compute(t, state, thrust)[1]) for t in (0.0, 0.005, 0.01)]
    np.testing.assert_allclose(again, first, atol=1e-12)


# ─── The orientation reaches the L1 predictor ─────────────────────────


def _lissajous_rms(strip_orientation):
    from udaan.control.quadrotor import PositionL1Controller
    from udaan.models.quadrotor import QuadrotorMujoco
    from udaan.utils.trajectory import CrazyTrajectory

    traj = CrazyTrajectory(center=np.array([0.0, 0.0, 3.0]))
    mdl = QuadrotorMujoco(render=False)
    # A plain closure, not the bound `traj.get`: it has no `__self__`, so
    # simulate() runs without the flatness feedforward. With the feedforward
    # on, the attitude lag is small enough that the commanded-thrust fallback
    # does not diverge (it tracks at 0.109 m) and this test could not tell.
    ctrl = PositionL1Controller(mass=mdl.mass, setpoint=lambda t: traj.get(t))
    if strip_orientation:
        compute = ctrl.compute
        ctrl.compute = lambda t, state: compute(t, tuple(state)[:2])
    mdl.position_controller = ctrl
    err = []
    step = mdl.step

    def recorded(u, desired_att=None):
        step(u, desired_att=desired_att)
        err.append(np.linalg.norm(mdl.state.position - traj.get(mdl.t)[0]))

    mdl.step = recorded
    mdl.simulate(20.0, position=traj.get(0.0)[0])
    return float(np.sqrt(np.mean(np.square(err))))


@pytest.mark.mujoco
def test_simulate_hands_the_orientation_to_the_predictor():
    """Through ``simulate()`` itself, feedforward off. Without ``R`` the
    predictor falls back to the commanded thrust and this run diverges
    (2.05 m RMS); with it the run tracks like PD (0.194 m against 0.193 m). So
    this fails if ``simulate()`` ever stops passing the orientation."""
    assert _lissajous_rms(strip_orientation=False) < 0.25
    assert _lissajous_rms(strip_orientation=True) > 1.0  # the fallback, for contrast


@pytest.mark.mujoco
def test_fleet_hands_the_orientation_to_its_position_controllers():
    from udaan.models.fleet import QuadrotorFleet

    fleet = QuadrotorFleet(num_quadrotors=2, render=False)
    seen = []
    for quad in fleet.quadrotors:
        compute = quad.position_controller.compute

        def spy(t, state, compute=compute, quad=quad):
            seen.append(len(state) == 3 and state[2] is quad.state.orientation)
            return compute(t, state)

        quad.position_controller.compute = spy
    fleet.simulate(tf=0.05)
    assert seen and all(seen)
