"""The cable cascade's time-scale gain design.

The gains are formulas in the machine's parameters, so they are checked against
those formulas, and the behaviour they are for — damping and stability at every
cable length — is checked on the closed loop itself: a numerical linearisation
of one ``step()`` at hover, and simulated hover, step and swing runs on long
cables.
"""

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from udaan.control.quadrotor import GeometricAttitudeController
from udaan.control.quadrotor_cspayload import (
    QuadCSPayloadController,
    cable_cascade_gains,
    loop_frequencies,
)
from udaan.core import defaults as D
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor import QuadrotorBase
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase

G = D.GRAVITY
ZETA = 0.7
W_A = 40.0


def _expected(mQ, mL, l, J):
    """The design written out: ω_a = 40, ω_c = min(2√(g/ℓ), ω_a/5), ω_p = ω_c/5, ω_z = ω_a/25."""
    w_c = min(2.0 * math.sqrt(G / l), W_A / 5.0)
    w_p, w_z = w_c / 5.0, W_A / 25.0
    M = mQ + mL
    return {
        "kx": [M * w_p**2, M * w_p**2, M * w_z**2],
        "kv": [2 * ZETA * M * w_p, 2 * ZETA * M * w_p, 2 * ZETA * M * w_z],
        "kq": [mQ * l * w_c**2] * 3,
        "kw": [2 * ZETA * mQ * l * w_c] * 3,
        "kR": [J[0] * W_A**2, J[1] * W_A**2, 1.35],
        "kO": [2 * ZETA * J[0] * W_A, 2 * ZETA * J[1] * W_A, 0.225],
    }


MACHINES = [
    (0.75, 0.15, 1.0, (0.0053, 0.0049, 0.0098)),
    (0.75, 0.15, 0.3, (0.0053, 0.0049, 0.0098)),  # short cable: ω_c capped at ω_a/5
    (1.2, 0.4, 2.5, (0.01, 0.012, 0.02)),
    (0.5, 0.05, 10.0, (0.003, 0.003, 0.005)),
]


class TestDesignFormulas:
    def test_design_constants(self):
        assert D.CABLE_CASCADE_ATTITUDE_BANDWIDTH == 40.0
        assert D.CABLE_CASCADE_LOOP_RATIO == 5.0
        assert D.CABLE_CASCADE_PENDULUM_FACTOR == 2.0
        assert D.CABLE_CASCADE_DAMPING == 0.7

    @pytest.mark.parametrize("mQ,mL,l,J", MACHINES)
    def test_gains_equal_the_formulas(self, mQ, mL, l, J):
        exp = _expected(mQ, mL, l, J)
        g = cable_cascade_gains(mQ, mL, l, np.diag(J))
        for got, key in (
            (g.position_kp, "kx"),
            (g.position_kd, "kv"),
            (g.cable_kp, "kq"),
            (g.cable_kd, "kw"),
            (g.attitude_kp, "kR"),
            (g.attitude_kd, "kO"),
        ):
            np.testing.assert_allclose(got, exp[key], rtol=1e-12, err_msg=key)

    @pytest.mark.parametrize("mQ,mL,l,J", MACHINES)
    def test_model_flies_the_formulas(self, mQ, mL, l, J):
        """The default controllers of a model with these parameters carry these gains."""
        mdl = QuadrotorCsPayloadBase()
        mdl.mass, mdl.payload_mass, mdl.cable_length, mdl.inertia = mQ, mL, l, J
        mdl._init_default_controllers()
        exp = _expected(mQ, mL, l, J)
        kx, kv = mdl.payload_controller.position_gains
        kq, kw = mdl.payload_controller.cable_gains
        np.testing.assert_allclose(kx, exp["kx"], rtol=1e-12)
        np.testing.assert_allclose(kv, exp["kv"], rtol=1e-12)
        np.testing.assert_allclose(kq, exp["kq"], rtol=1e-12)
        np.testing.assert_allclose(kw, exp["kw"], rtol=1e-12)
        np.testing.assert_allclose(mdl.attitude_controller._gains.kp, exp["kR"], rtol=1e-12)
        np.testing.assert_allclose(mdl.attitude_controller._gains.kd, exp["kO"], rtol=1e-12)

    def test_cable_frequency_is_capped_on_short_cables(self):
        assert loop_frequencies(0.3).cable == pytest.approx(W_A / 5.0)
        assert loop_frequencies(1.0).cable == pytest.approx(2.0 * math.sqrt(G))

    def test_sync_from_model_recomputes(self):
        mdl = QuadrotorCsPayloadBase()
        ctrl = mdl.payload_controller
        mdl.cable_length = 4.0
        ctrl.sync_from_model(mdl)
        exp = _expected(mdl.mass, mdl.payload_mass, 4.0, np.diag(mdl.inertia))
        np.testing.assert_allclose(ctrl.cable_gains[0], exp["kq"], rtol=1e-12)
        np.testing.assert_allclose(ctrl.position_gains[0], exp["kx"], rtol=1e-12)

    def test_explicit_gains_override_and_survive_sync(self):
        ctrl = QuadCSPayloadController(position_kp=[1.0, 2.0, 3.0], cable_kd=5.0)
        mdl = QuadrotorCsPayloadBase()
        mdl.cable_length = 3.0
        ctrl.sync_from_model(mdl)
        np.testing.assert_array_equal(ctrl.position_gains[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(ctrl.cable_gains[1], [5.0, 5.0, 5.0])
        exp = _expected(mdl.mass, mdl.payload_mass, 3.0, np.diag(mdl.inertia))
        np.testing.assert_allclose(ctrl.cable_gains[0], exp["kq"], rtol=1e-12)


class TestQuadrotorDefaultsUnchanged:
    """The design writes M2's own attitude-controller instance; M1 keeps its gains."""

    def test_attitude_controller_defaults(self):
        ctrl = GeometricAttitudeController(inertia=np.eye(3))
        np.testing.assert_array_equal(ctrl._gains.kp, [2.4, 2.4, 1.35])
        np.testing.assert_array_equal(ctrl._gains.kd, [0.35, 0.35, 0.225])

    def test_quadrotor_model_gains(self):
        mdl = QuadrotorBase()
        np.testing.assert_array_equal(mdl.attitude_controller._gains.kp, [2.4, 2.4, 1.35])
        np.testing.assert_array_equal(mdl.attitude_controller._gains.kd, [0.35, 0.35, 0.225])
        np.testing.assert_array_equal(D.DEFAULT_ATT_KP, [2.4, 2.4, 1.35])
        np.testing.assert_array_equal(D.DEFAULT_ATT_KD, [0.35, 0.35, 0.225])


# ─── closed loop ────────────────────────────────────────────────────────────


def _hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _closed_loop_poles(l, eps=1e-6, gains=None):
    """Continuous-equivalent poles log(λ)/dt of one hover ``step()``, by central differences.

    State in local coordinates (16): payload position and velocity, the cable
    direction as a rotation vector about −e₃ (x, y), the cable rate (x, y), the
    airframe attitude as a rotation vector, and the body rate. ``gains`` are
    keywords for a payload controller that replaces the default one.
    """
    mdl = QuadrotorCsPayloadBase()
    mdl.cable_length = l
    mdl._init_default_controllers()
    if gains is not None:
        mdl.payload_controller = QuadCSPayloadController(**gains)
        mdl.payload_controller.sync_from_model(mdl)
    base = np.array([0.0, 0.0, 1.0])
    mdl.payload_controller.setpoint = lambda t: (base.copy(), np.zeros(3), np.zeros(3))
    down = np.array([0.0, 0.0, -1.0])

    def f(x):
        mdl.reset()
        s = mdl.state
        s.payload_position = base + x[0:3]
        s.payload_velocity = x[3:6].copy()
        q = expm(_hat(np.array([x[6], x[7], 0.0]))) @ down
        om = np.array([x[8], x[9], 0.0])
        om = om - q * (q @ om)
        s.cable_attitude, s.cable_angular_velocity = S2(q), TS2(om)
        s.orientation = SO3(expm(_hat(x[10:13])))
        s.angular_velocity = TSO3(x[13:16].copy())
        s.position = s.payload_position - l * q
        s.velocity = s.payload_velocity - l * np.cross(om, q)
        mdl.step(mdl.payload_controller.compute(mdl.t, mdl.state))
        q = np.asarray(s.cable_attitude)
        ax = np.cross(down, q)
        sn = np.linalg.norm(ax)
        rv = np.zeros(3) if sn < 1e-15 else ax / sn * math.atan2(sn, float(down @ q))
        om = np.asarray(s.cable_angular_velocity)
        return np.concatenate(
            [
                np.asarray(s.payload_position) - base,
                np.asarray(s.payload_velocity),
                rv[:2],
                om[:2],
                Rotation.from_matrix(np.asarray(s.orientation)).as_rotvec(),
                np.asarray(s.angular_velocity),
            ]
        )

    n = 16
    phi = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        phi[:, i] = (f(dx) - f(-dx)) / (2 * eps)
    return np.log(np.linalg.eigvals(phi).astype(complex)) / mdl.dt


@pytest.mark.parametrize("l", [0.5, 1.0, 2.0, 10.0])
def test_closed_loop_damping(l):
    """Every closed-loop pole of the linearised step() has ζ ≥ 0.65."""
    s = _closed_loop_poles(l)
    s = s[np.abs(s) > 1e-6]
    zeta = -s.real / np.abs(s)
    assert zeta.min() >= 0.65, (
        f"ℓ = {l} m: ζ_min = {zeta.min():.3f} at |s| = {abs(s[zeta.argmin()]):.2f}"
    )


def _fly(l, kind):
    """Hover (0.1 m off), a 1 m lateral step, or a 30° swing release; returns the payload error log."""
    mdl = QuadrotorCsPayloadBase()
    mdl.cable_length = l
    mdl._init_default_controllers()
    target = np.array([0.0, 0.0, l + 1.0])
    mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    kwargs = {"payload_position": target.copy()}
    if kind == "hover":
        kwargs["payload_position"] = target + np.array([0.1, 0.0, 0.0])
    elif kind == "step":
        kwargs["payload_position"] = target - np.array([1.0, 0.0, 0.0])
    else:
        th = math.radians(30.0)
        kwargs["cable_attitude"] = S2(np.array([math.sin(th), 0.0, -math.cos(th)]))
    mdl.reset(**kwargs)
    log, tilt = [], 0.0
    while mdl.t < 10.0 * math.sqrt(l):
        mdl.step(mdl.payload_controller.compute(mdl.t, mdl.state))
        R = np.asarray(mdl.state.orientation)
        tilt = max(tilt, math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))))
        err = float(np.linalg.norm(mdl.state.payload_position - target))
        log.append((mdl.t, err))
        if not np.isfinite(err) or err > 10.0:
            break
    return np.array(log), tilt


@pytest.mark.parametrize("l", [2.0, pytest.param(10.0, marks=pytest.mark.slow)], ids=["2m", "10m"])
@pytest.mark.parametrize("kind", ["hover", "step", "swing"])
def test_long_cable_is_stable(l, kind):
    """Settles within 5 cm by 8√ℓ s (the settling budget scales as √ℓ) and holds 1 cm after."""
    log, tilt = _fly(l, kind)
    horizon = 10.0 * math.sqrt(l)
    assert log[-1, 0] >= horizon - 1e-9, f"diverged at t = {log[-1, 0]:.2f} s"
    late = log[log[:, 0] > 8.0 * math.sqrt(l), 1]
    assert late.max() < 0.05, f"payload error {late.max():.3f} m after 8√ℓ s"
    assert log[-1, 1] < 0.01
    assert tilt < 60.0


# ─── the fixed gains ────────────────────────────────────────────────────────

FIXED_GAINS = {
    "position_kp": D.DEFAULT_PAYLOAD_POS_KP,
    "position_kd": D.DEFAULT_PAYLOAD_POS_KD,
    "cable_kp": D.DEFAULT_PAYLOAD_CABLE_KP,
    "cable_kd": D.DEFAULT_PAYLOAD_CABLE_KD,
}


class TestFixedGains:
    """``DEFAULT_PAYLOAD_*``: fixed gains, flown through the controller's keywords."""

    def test_values(self):
        np.testing.assert_array_equal(D.DEFAULT_PAYLOAD_POS_KP, [4.0, 4.0, 8.0])
        np.testing.assert_array_equal(D.DEFAULT_PAYLOAD_POS_KD, [3.0, 3.0, 6.0])
        np.testing.assert_array_equal(D.DEFAULT_PAYLOAD_CABLE_KP, [24.0, 24.0, 24.0])
        np.testing.assert_array_equal(D.DEFAULT_PAYLOAD_CABLE_KD, [8.0, 8.0, 8.0])

    def test_the_keywords_keep_them_through_sync(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.cable_length = 1.5
        ctrl = QuadCSPayloadController(**FIXED_GAINS)
        ctrl.sync_from_model(mdl)
        np.testing.assert_array_equal(ctrl.position_gains[0], D.DEFAULT_PAYLOAD_POS_KP)
        np.testing.assert_array_equal(ctrl.position_gains[1], D.DEFAULT_PAYLOAD_POS_KD)
        np.testing.assert_array_equal(ctrl.cable_gains[0], D.DEFAULT_PAYLOAD_CABLE_KP)
        np.testing.assert_array_equal(ctrl.cable_gains[1], D.DEFAULT_PAYLOAD_CABLE_KD)

    def test_a_model_flies_them_to_its_setpoint(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.payload_controller = QuadCSPayloadController(**FIXED_GAINS)
        mdl.payload_controller.sync_from_model(mdl)
        target = np.array([0.0, 0.0, 1.0])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(10.0, payload_position=target + np.array([0.1, 0.0, 0.0]))
        assert np.linalg.norm(mdl.state.payload_position - target) < 0.01

    @pytest.mark.parametrize(
        "l, stable", [(0.2, False), (0.3, True), (1.0, True), (1.7, True), (1.8, False)]
    )
    def test_stable_cable_lengths(self, l, stable):
        """Stable at hover for 0.25 m ≤ ℓ ≤ 1.76 m on the default machine, unstable outside."""
        s = _closed_loop_poles(l, gains=FIXED_GAINS)
        assert (s.real.max() < 0.0) == stable, f"ℓ = {l} m: max Re(s) = {s.real.max():.4f}"
