"""Open-loop feedforward tracking tests for the multi-quad cspayload flat map.

The flat-output-derived ``(f_i, M_i)`` is fed straight into the ``nQ``-quadrotor
+ cable-suspended-point-mass-payload dynamics — no closed-loop controller
anywhere. If the flat-to-state map is exact, the trajectory the dynamics produce
*is* the flat reference, and the only residual is integrator error.

Two integrators are exercised:

* The shipped ZOH-Euler in :class:`MultiQuadCsPayloadBase` (1.25 ms inner step,
  feedforward held constant across the 5 ms outer step).
* A high-order adaptive RK (DOP853 via :func:`scipy.integrate.solve_ivp`)
  re-evaluating ``(f_i, M_i)`` continuously, with quaternion attitudes. At
  ``rtol = atol = 1e-12`` this drives integrator error to round-off, so *any*
  non-trivial residue is direct evidence of a flat-to-state formula bug.

ODE layout (``6 + 6 nQ + 7 nQ`` states)::

    y = [x_L(3), v_L(3),
         q_0(3), w_0(3), …, q_{n-1}(3), w_{n-1}(3),
         quat_0(4), Omega_0(3), …, quat_{n-1}(4), Omega_{n-1}(3)]

with ``‖q_i‖ = 1`` and ``w_i · q_i = 0`` preserved exactly by the continuous
dynamics; the constraints are re-projected at each RHS call to absorb round-off.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as _ScipyRot

from tests.conftest import TOL_INTEGRATION
from udaan.core.defaults import GRAVITY
from udaan.flatness import (
    Jet,
    LegacyFixedTensionSplit,
    MultiQuadCsPayload,
    MultiQuadCsPayloadFlats,
)
from udaan.manif import SO3
from udaan.models.multi_quad_cspayload.base import MultiQuadCsPayloadBase

TF_ZOH = 5.0
TF_ODE = 3.0
# ZOH-Euler at h = 1.25 ms over TF_ZOH s accumulates ~cm-level payload drift on a
# fully three-dimensional trajectory — five times ``TOL_PHYSICAL``, and purely
# integrator error: the DOP853 gate below runs the same trajectories with the
# same feedforward and lands at ``TOL_INTEGRATION``. Still tight enough that a
# flat-to-state formula bug fails it by orders of magnitude.
TOL_ZOH = 5e-2
_E3 = np.array([0.0, 0.0, 1.0])


# ─── Trajectory builders ──────────────────────────────────────────────


def hover_flats(t: float, nQ: int) -> MultiQuadCsPayloadFlats:
    """Stationary hover with the payload at the origin."""
    return MultiQuadCsPayloadFlats(
        x_L=Jet.zeros(order=6, dim=3),
        psi=[Jet.zeros(order=2, dim=1) for _ in range(nQ)],
    )


def circle_flats(
    t: float, nQ: int, r: float = 0.4, omega: float = 1.0, z0: float = 0.0
) -> MultiQuadCsPayloadFlats:
    """Payload circle in the XY plane; every yaw tracks the heading angle.

    Radius and rate are modest so every cable stays well clear of slack.
    """
    c, s = np.cos(omega * t), np.sin(omega * t)
    x_data = np.array(
        [
            [r * c, r * s, z0],
            [-r * omega * s, r * omega * c, 0.0],
            [-r * omega**2 * c, -r * omega**2 * s, 0.0],
            [r * omega**3 * s, -r * omega**3 * c, 0.0],
            [r * omega**4 * c, r * omega**4 * s, 0.0],
            [-r * omega**5 * s, r * omega**5 * c, 0.0],
            [-r * omega**6 * c, -r * omega**6 * s, 0.0],
        ]
    )
    psi = Jet(np.array([omega * t, omega, 0.0]))
    return MultiQuadCsPayloadFlats(x_L=Jet(x_data), psi=[psi for _ in range(nQ)])


def lissajous_flats(t: float, nQ: int) -> MultiQuadCsPayloadFlats:
    """A genuinely three-dimensional payload trajectory with per-quadrotor yaw.

    ``x_L = [a sin(w1 t), b sin(w2 t + φ), c sin(w3 t)]`` — all three axes move,
    the vertical axis included, so the cable directions sweep in 3-D and every
    derivative rung of the recovery is exercised.
    """
    amps = np.array([0.35, 0.30, 0.20])
    rates = np.array([1.1, 0.7, 0.9])
    phases = np.array([0.0, 0.6, 1.3])
    rows = []
    for k in range(7):
        # d^k/dt^k [A sin(w t + φ)] = A w^k sin(w t + φ + k π/2)
        rows.append(amps * rates**k * np.sin(rates * t + phases + k * np.pi / 2.0))
    psi = [Jet(np.array([0.3 * i + 0.4 * t, 0.4, 0.0])) for i in range(nQ)]
    return MultiQuadCsPayloadFlats(x_L=Jet(np.stack(rows)), psi=psi)


# ─── Harness ──────────────────────────────────────────────────────────


def _build(nQ: int, masses=None, cable_lengths=None, policy=None):
    """A matched (flat map, model) pair with identical physical parameters."""
    model = MultiQuadCsPayloadBase(num_quadrotors=nQ, input="wrench")
    if masses is not None:
        model.masses = masses
    if cable_lengths is not None:
        model.cable_lengths = cable_lengths
    f2s = MultiQuadCsPayload.from_model(model, tension_policy=policy)
    return f2s, model


def _seed(model: MultiQuadCsPayloadBase, ref) -> None:
    state = ref.as_state()
    model.reset(
        payload_position=np.asarray(state.payload_position).copy(),
        payload_velocity=np.asarray(state.payload_velocity).copy(),
        cable_attitudes=state.cable_attitudes,
        cable_angular_velocities=state.cable_angular_velocities,
        orientations=state.orientations,
        angular_velocities=state.angular_velocities,
    )


def _errors(sim, ref) -> dict:
    """Worst-over-quadrotors error of a simulated state vs. the flat reference.

    ``sim`` is a dict of the same shape the two integrators both produce.
    """
    nQ = len(ref.positions)
    out = {
        "payload_pos": float(np.linalg.norm(sim["payload_position"] - ref.payload_position)),
        "payload_vel": float(np.linalg.norm(sim["payload_velocity"] - ref.payload_velocity)),
    }
    cable_q = cable_w = att = body_w = 0.0
    for i in range(nQ):
        cable_q = max(
            cable_q,
            float(np.linalg.norm(sim["cable_attitudes"][i] - np.asarray(ref.cable_attitudes[i]))),
        )
        cable_w = max(
            cable_w,
            float(
                np.linalg.norm(
                    sim["cable_angular_velocities"][i] - np.asarray(ref.cable_angular_velocities[i])
                )
            ),
        )
        att = max(
            att,
            float(SO3(np.asarray(ref.orientations[i])).config_error(SO3(sim["orientations"][i]))),
        )
        body_w = max(
            body_w,
            float(
                np.linalg.norm(sim["angular_velocities"][i] - np.asarray(ref.angular_velocities[i]))
            ),
        )
    out.update(cable_q=cable_q, cable_omega=cable_w, attitude=att, body_omega=body_w)
    return out


def _model_snapshot(model: MultiQuadCsPayloadBase) -> dict:
    return {
        "payload_position": np.asarray(model.state.payload_position),
        "payload_velocity": np.asarray(model.state.payload_velocity),
        "cable_attitudes": [np.asarray(q) for q in model.state.cable_attitudes],
        "cable_angular_velocities": [np.asarray(w) for w in model.state.cable_angular_velocities],
        "orientations": [np.asarray(R) for R in model.state.orientations],
        "angular_velocities": [np.asarray(w) for w in model.state.angular_velocities],
    }


def run_zoh(flats_of_t, nQ: int, tf: float = TF_ZOH, **kwargs) -> dict:
    """Seed from the ``t = 0`` reference, march with pure feedforward."""
    f2s, model = _build(nQ, **kwargs)
    ref0, _ = f2s.recover(flats_of_t(0.0, nQ))
    _seed(model, ref0)

    while model.t < tf - 1e-9:
        _, inputs = f2s.recover(flats_of_t(model.t, nQ))
        model.step(inputs.wrench)

    ref_final, _ = f2s.recover(flats_of_t(tf, nQ))
    return _errors(_model_snapshot(model), ref_final)


# ─── High-order ODE integration (DOP853 + quaternion attitudes) ───────


def _unpack(y: np.ndarray, nQ: int):
    x_L = y[0:3]
    v_L = y[3:6]
    q, w, quat, Omega = [], [], [], []
    base = 6
    for i in range(nQ):
        qi = y[base + 6 * i : base + 6 * i + 3]
        wi = y[base + 6 * i + 3 : base + 6 * i + 6]
        qi = qi / np.linalg.norm(qi)
        wi = wi - qi * wi.dot(qi)
        q.append(qi)
        w.append(wi)
    base = 6 + 6 * nQ
    for i in range(nQ):
        qu = y[base + 7 * i : base + 7 * i + 4]
        quat.append(qu / np.linalg.norm(qu))
        Omega.append(y[base + 7 * i + 4 : base + 7 * i + 7])
    return x_L, v_L, q, w, quat, Omega


def _rot(quat_wxyz: np.ndarray) -> np.ndarray:
    return _ScipyRot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()


def _quat_dot(quat_wxyz: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """Hamilton, scalar-first: ``q̇ = ½ q ⊗ [0, Ω]``."""
    w, x, y_, z = quat_wxyz
    p_, q_, r_ = Omega
    return 0.5 * np.array(
        [
            -x * p_ - y_ * q_ - z * r_,
            w * p_ - z * q_ + y_ * r_,
            z * p_ + w * q_ - x * r_,
            -y_ * p_ + x * q_ + w * r_,
        ]
    )


def _rhs(t, y, f2s: MultiQuadCsPayload, flats_of_t):
    """Multi-quad cspayload ODE, with the feedforward re-evaluated at exactly ``t``.

    Mirrors ``MultiQuadCsPayloadBase._zoh`` in continuous time::

        (m_L I + Σ m_i q_i q_iᵀ)(a_L + g e₃) = Σ (q_iq_iᵀ u_i − m_i l_i ‖q̇_i‖² q_i)
        ω̇_i = q̂_i (a_L + g e₃)/l_i − q̂_i u_i^⊥/(m_i l_i)
    """
    nQ = f2s.num_quadrotors
    x_L, v_L, q, w, quat, Omega = _unpack(y, nQ)
    _, inputs = f2s.recover(flats_of_t(t, nQ))

    ge3 = GRAVITY * _E3
    mass_matrix = f2s.payload_mass * np.eye(3)
    rhs_vec = np.zeros(3)
    u_perp = []
    for i in range(nQ):
        m_i = float(f2s.masses[i])
        l_i = float(f2s.cable_lengths[i])
        qqT = np.outer(q[i], q[i])
        mass_matrix = mass_matrix + m_i * qqT
        u_i = inputs.thrusts[i] * (_rot(quat[i]) @ _E3)
        u_perp.append(u_i - qqT @ u_i)
        dq_i = np.cross(w[i], q[i])
        rhs_vec = rhs_vec + qqT @ u_i - m_i * l_i * float(dq_i.dot(dq_i)) * q[i]
    a_L = np.linalg.solve(mass_matrix, rhs_vec) - ge3

    out = [v_L, a_L]
    cable_block = []
    att_block = []
    for i in range(nQ):
        m_i = float(f2s.masses[i])
        l_i = float(f2s.cable_lengths[i])
        cable_block.append(np.cross(w[i], q[i]))
        cable_block.append(
            np.cross(q[i], a_L + ge3) / l_i - np.cross(q[i], u_perp[i]) / (m_i * l_i)
        )
        J = f2s.inertias[i]
        att_block.append(_quat_dot(quat[i], Omega[i]))
        att_block.append(np.linalg.solve(J, inputs.moments[i] - np.cross(Omega[i], J @ Omega[i])))
    return np.concatenate(out + cable_block + att_block)


def run_dop853(flats_of_t, nQ: int, tf: float = TF_ODE, **kwargs) -> dict:
    f2s, _ = _build(nQ, **kwargs)
    ref0, _ = f2s.recover(flats_of_t(0.0, nQ))

    y0 = [np.asarray(ref0.payload_position), np.asarray(ref0.payload_velocity)]
    for i in range(nQ):
        y0.append(np.asarray(ref0.cable_attitudes[i]))
        y0.append(np.asarray(ref0.cable_angular_velocities[i]))
    for i in range(nQ):
        xyzw = _ScipyRot.from_matrix(np.asarray(ref0.orientations[i])).as_quat()
        y0.append(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]))
        y0.append(np.asarray(ref0.angular_velocities[i]))

    sol = solve_ivp(
        _rhs,
        (0.0, tf),
        np.concatenate(y0),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        args=(f2s, flats_of_t),
    )
    assert sol.success, f"DOP853 failed: {sol.message}"

    x_L, v_L, q, w, quat, Omega = _unpack(sol.y[:, -1], nQ)
    sim = {
        "payload_position": x_L,
        "payload_velocity": v_L,
        "cable_attitudes": q,
        "cable_angular_velocities": w,
        "orientations": [_rot(qu) for qu in quat],
        "angular_velocities": Omega,
    }
    ref_final, _ = f2s.recover(flats_of_t(tf, nQ))
    return _errors(sim, ref_final)


def _assert_below(errors: dict, tol: float, label: str):
    for key, value in errors.items():
        assert value < tol, f"[{label}] {key} = {value:.3e} exceeds {tol:.0e}"


# ─── Tests: ZOH-Euler open loop ───────────────────────────────────────


class TestOpenLoopHover:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_hover_is_a_fixed_point(self, nQ):
        # Exact feedforward at a genuine equilibrium: nothing should move at all.
        e = run_zoh(hover_flats, nQ)
        _assert_below(e, 1e-10, f"hover nQ={nQ}")

    def test_hover_with_heterogeneous_parameters(self):
        e = run_zoh(hover_flats, 4, masses=[0.7, 0.9, 1.1, 1.3], cable_lengths=[0.9, 1.1, 1.3, 1.5])
        _assert_below(e, 1e-10, "hover heterogeneous")

    def test_hover_with_legacy_policy(self):
        e = run_zoh(hover_flats, 3, policy=LegacyFixedTensionSplit())
        _assert_below(e, 1e-10, "hover legacy")


class TestOpenLoopDynamic:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_circle(self, nQ):
        e = run_zoh(circle_flats, nQ)
        _assert_below(e, TOL_ZOH, f"circle nQ={nQ}")

    @pytest.mark.parametrize("nQ", [2, 3])
    def test_lissajous(self, nQ):
        e = run_zoh(lissajous_flats, nQ)
        _assert_below(e, TOL_ZOH, f"lissajous nQ={nQ}")

    def test_circle_with_heterogeneous_parameters(self):
        e = run_zoh(circle_flats, 3, masses=[0.7, 0.9, 1.2], cable_lengths=[0.9, 1.15, 1.4])
        _assert_below(e, TOL_ZOH, "circle heterogeneous")

    def test_circle_with_legacy_policy(self):
        e = run_zoh(circle_flats, 3, policy=LegacyFixedTensionSplit())
        _assert_below(e, TOL_ZOH, "circle legacy")


# ─── Tests: DOP853 — the actual correctness gate ──────────────────────


class TestHighOrderRoundTrip:
    """With integrator error at round-off, anything left is a formula bug."""

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_circle(self, nQ):
        e = run_dop853(circle_flats, nQ)
        _assert_below(e, TOL_INTEGRATION, f"circle nQ={nQ}")

    @pytest.mark.parametrize("nQ", [2, 3])
    def test_lissajous(self, nQ):
        e = run_dop853(lissajous_flats, nQ)
        _assert_below(e, TOL_INTEGRATION, f"lissajous nQ={nQ}")

    def test_heterogeneous_masses_and_cable_lengths(self):
        e = run_dop853(lissajous_flats, 3, masses=[0.7, 0.9, 1.2], cable_lengths=[0.9, 1.15, 1.4])
        _assert_below(e, TOL_INTEGRATION, "lissajous heterogeneous")

    def test_legacy_policy_also_round_trips(self):
        e = run_dop853(circle_flats, 3, policy=LegacyFixedTensionSplit())
        _assert_below(e, TOL_INTEGRATION, "circle legacy")
