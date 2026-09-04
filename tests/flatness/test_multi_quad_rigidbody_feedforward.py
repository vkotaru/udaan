"""Open-loop feedforward tracking tests for the M4 rigid-body-payload flat map.

The flat-output-derived ``(f_i, M_i)`` is fed straight into the ``nQ``-quadrotor
+ cable-suspended-**rigid-body**-payload dynamics — no closed-loop controller
anywhere. If the flat-to-state map is exact, the trajectory the dynamics produce
*is* the flat reference, and the only residual is integrator error.

This is the load-bearing gate for Step 7, and it checks one thing M3's
equivalent could not: the **payload attitude**. A rigid-body payload is
positioned *and* oriented by the cable team, so an error anywhere in the wrench
chain — the ``R_L`` jet, the frame change ``𝐓_i = -R_L τ_i``, the gyroscopic
term of ``W_m``, the ``Ω_L^{(5)}`` rung — shows up as a drifting ``R_L`` while
the position may still look perfect.

Two integrators are exercised:

* The shipped ZOH-Euler in :class:`MultiQuadRigidbodyBase` (1.25 ms inner step,
  feedforward held constant across the 5 ms outer step).
* A high-order adaptive RK (DOP853 via :func:`scipy.integrate.solve_ivp`)
  re-evaluating ``(f_i, M_i)`` continuously, with quaternion attitudes for both
  the payload and every quadrotor. At ``rtol = atol = 1e-12`` this drives
  integrator error to round-off, so *any* non-trivial residue is direct
  evidence of a flat-to-state formula bug.

ODE layout (``13 + 6 nQ + 7 nQ`` states)::

    y = [x_L(3), v_L(3), quat_L(4), Omega_L(3),
         q_0(3), w_0(3), …, q_{n-1}(3), w_{n-1}(3),
         quat_0(4), Omega_0(3), …, quat_{n-1}(4), Omega_{n-1}(3)]

with ``‖q_i‖ = 1`` and ``w_i · q_i = 0`` preserved exactly by the continuous
dynamics; the constraints are re-projected at each RHS call to absorb round-off.

Payload-attitude trajectories
-----------------------------
``R_L(t) = R_x(\\phi(t))\\, R_z(\\psi_L t)`` with ``\\phi = A \\sin(\\omega t)``
and a *constant* yaw rate. That combination is deliberate: it is a genuinely
non-fixed-axis rotation — the body-frame ``Ω_L`` direction moves — yet every
derivative of

.. math::

    \\Omega_L = \\dot\\phi\\,[\\cos\\psi_L t,\\; -\\sin\\psi_L t,\\; 0]^\\top
                + \\psi_L\\, e_3

is available in closed form by the Leibniz rule on two elementary sinusoid
jets. Using a fixed rotation axis instead would leave ``Ω_L`` parallel to a
constant vector and never exercise the frame-transport terms; taking a general
Euler-angle path would need derivatives of ``sin(A \\sin \\omega t)``, which have
no closed form and would have to be finite-differenced — destroying the very
exactness the gate is measuring.
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
    MultiQuadRigidbody,
    MultiQuadRigidbodyFlats,
    leibniz_product,
)
from udaan.manif import hat
from udaan.models.multi_quad_rigidbody.base import MultiQuadRigidbodyBase

TF_ZOH = 3.0
TF_ODE = 3.0
# ZOH-Euler at h = 1.25 ms over TF_ZOH s accumulates cm-level payload drift on a
# fully six-dimensional pose trajectory — purely integrator error: the DOP853
# gate below runs the same trajectories with the same feedforward and lands at
# TOL_INTEGRATION. Still tight enough that a flat-to-state formula bug fails it
# by orders of magnitude.
TOL_ZOH = 5e-2
_E3 = np.array([0.0, 0.0, 1.0])

# An off-ring grasp. Its ring hover trim is infeasible (residual 0.85 N), so
# building a map on it logs a warning — expected, and only about the bias no
# longer matching reset(); the reference is still an exact equilibrium, which
# test_multi_quad_rigidbody.py pins directly.
_ASYMMETRIC_ATTACHMENTS = np.array(
    [[0.55, 0.05, 0.05], [-0.15, 0.60, 0.02], [-0.62, -0.20, 0.08], [0.20, -0.55, -0.03]]
)


# ─── Trajectory builders ──────────────────────────────────────────────


def _sinusoid_jet(t: float, amp: float, rate: float, order: int, phase: float = 0.0) -> list[float]:
    """Derivative tuple of ``amp · sin(rate · t + phase)`` — exact at every order."""
    return [amp * rate**k * np.sin(rate * t + phase + k * np.pi / 2) for k in range(order + 1)]


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _level_attitude() -> tuple[np.ndarray, Jet]:
    """Payload held level and still."""
    return np.eye(3), Jet.zeros(order=5, dim=3)


def _pose_attitude(
    t: float, tilt: float = 0.15, tilt_rate: float = 1.2, yaw_rate: float = 0.6
) -> tuple[np.ndarray, Jet]:
    """``R_L = R_x(A sin ω t) R_z(ψ_L t)`` and its exact body-rate jet.

    See the module docstring for why this particular family: non-fixed axis,
    bounded tilt (so the cables stay comfortably taut), closed-form
    derivatives to any order.
    """
    order = 5
    phi = tilt * np.sin(tilt_rate * t)
    R_L = _rot_x(phi) @ _rot_z(yaw_rate * t)

    # φ̇ and its derivatives: drop the zeroth rung of the jet of φ itself.
    dphi = _sinusoid_jet(t, tilt, tilt_rate, order + 1)[1:]
    cos_psi = _sinusoid_jet(t, 1.0, yaw_rate, order, phase=np.pi / 2)
    sin_psi = _sinusoid_jet(t, 1.0, yaw_rate, order)

    omega_x = leibniz_product(dphi, cos_psi)
    omega_y = [-v for v in leibniz_product(dphi, sin_psi)]
    omega_z = [yaw_rate] + [0.0] * order
    omega = np.stack([np.array([omega_x[k], omega_y[k], omega_z[k]]) for k in range(order + 1)])
    return R_L, Jet(omega)


def _position_jet(t: float, amps=(0.35, 0.30, 0.20), rates=(1.1, 0.7, 0.9)) -> Jet:
    """Fully three-dimensional payload position with exact derivatives to pop."""
    amps = np.asarray(amps, dtype=float)
    rates = np.asarray(rates, dtype=float)
    phases = np.array([0.0, 0.6, 1.3])
    rows = [amps * rates**k * np.sin(rates * t + phases + k * np.pi / 2) for k in range(7)]
    return Jet(np.stack(rows))


def _yaws(t: float, nQ: int) -> list[Jet]:
    return [Jet(np.array([0.3 * i + 0.4 * t, 0.4, 0.0])) for i in range(nQ)]


def hover_flats(t: float, nQ: int) -> MultiQuadRigidbodyFlats:
    """Stationary level hover with the payload at the origin."""
    R_L, omega = _level_attitude()
    return MultiQuadRigidbodyFlats(
        x_L=Jet.zeros(order=6, dim=3),
        R_L=R_L,
        Omega_L=omega,
        psi=[Jet.zeros(order=2, dim=1) for _ in range(nQ)],
    )


def circle_flats(t: float, nQ: int, r: float = 0.4, omega: float = 1.0) -> MultiQuadRigidbodyFlats:
    """Payload circle in the XY plane, attitude held level."""
    c, s = np.cos(omega * t), np.sin(omega * t)
    rows = np.array(
        [
            [r * c, r * s, 0.0],
            [-r * omega * s, r * omega * c, 0.0],
            [-r * omega**2 * c, -r * omega**2 * s, 0.0],
            [r * omega**3 * s, -r * omega**3 * c, 0.0],
            [r * omega**4 * c, r * omega**4 * s, 0.0],
            [-r * omega**5 * s, r * omega**5 * c, 0.0],
            [-r * omega**6 * c, -r * omega**6 * s, 0.0],
        ]
    )
    R_L, ang = _level_attitude()
    return MultiQuadRigidbodyFlats(x_L=Jet(rows), R_L=R_L, Omega_L=ang, psi=_yaws(t, nQ))


def lissajous_flats(t: float, nQ: int) -> MultiQuadRigidbodyFlats:
    """A genuinely three-dimensional payload translation, attitude held level."""
    R_L, ang = _level_attitude()
    return MultiQuadRigidbodyFlats(x_L=_position_jet(t), R_L=R_L, Omega_L=ang, psi=_yaws(t, nQ))


def spin_flats(t: float, nQ: int, yaw_rate: float = 0.8) -> MultiQuadRigidbodyFlats:
    """Payload spinning about its own vertical axis, translation held still.

    The purest payload-attitude test: nothing moves except ``R_L``, so any
    residue is attributable to the attitude channel alone.
    """
    omega = np.zeros((6, 3))
    omega[0] = np.array([0.0, 0.0, yaw_rate])
    return MultiQuadRigidbodyFlats(
        x_L=Jet.zeros(order=6, dim=3),
        R_L=_rot_z(yaw_rate * t),
        Omega_L=Jet(omega),
        psi=[Jet.zeros(order=2, dim=1) for _ in range(nQ)],
    )


def pose_flats(t: float, nQ: int) -> MultiQuadRigidbodyFlats:
    """The full six-degree-of-freedom payload manoeuvre: translation *and* attitude."""
    R_L, ang = _pose_attitude(t)
    return MultiQuadRigidbodyFlats(x_L=_position_jet(t), R_L=R_L, Omega_L=ang, psi=_yaws(t, nQ))


# ─── Harness ──────────────────────────────────────────────────────────


def _build(nQ: int, masses=None, cable_lengths=None, attachment_points=None, policy=None):
    """A matched (flat map, model) pair with identical physical parameters."""
    model = MultiQuadRigidbodyBase(num_quadrotors=nQ, input="wrench")
    if masses is not None:
        model.masses = masses
    if cable_lengths is not None:
        model.cable_lengths = cable_lengths
    if attachment_points is not None:
        model.attachment_points = attachment_points
    return MultiQuadRigidbody.from_model(model, tension_policy=policy), model


def _seed(model: MultiQuadRigidbodyBase, ref) -> None:
    state = ref.as_state()
    model.reset(
        payload_position=np.asarray(state.payload_position).copy(),
        payload_velocity=np.asarray(state.payload_velocity).copy(),
        payload_orientation=np.asarray(state.payload_orientation),
        payload_angular_velocity=np.asarray(state.payload_angular_velocity),
        cable_attitudes=state.cable_attitudes,
        cable_angular_velocities=state.cable_angular_velocities,
        orientations=state.orientations,
        angular_velocities=state.angular_velocities,
    )


def _attitude_angle(reference, simulated) -> float:
    r"""Rotation angle ``θ = ‖log(R_d^⊤ R)‖`` between two attitudes, in radians.

    Deliberately **not** :meth:`~udaan.manif.SO3.config_error`, which is
    ``0.5 tr(I − R_d^⊤R) = 1 − cos θ ≈ θ²/2``: a quadratic metric compared
    against the same tolerance as the linear position/rate channels lets an
    attitude error four orders of magnitude larger through (``1e-6`` admits
    ``θ = 1.41e-3 rad``), and it under-reports what *is* achieved by a square
    root — it is also free to come out slightly negative at round-off, which no
    ``value < tol`` assertion would ever catch.  The payload attitude is the one
    channel this file exists to test, so it is measured in radians and held to
    the same standard as everything else.

    ``arctan2(sin θ, cos θ)`` rather than ``arccos``: at ``θ = 1e-8`` the cosine
    is ``1 − 5e-17`` and rounds to exactly ``1.0``, so ``arccos`` returns ``0``
    for the whole small-angle regime that matters here, whereas the off-diagonal
    entries carrying ``sin θ`` are accurate to relative round-off.
    """
    error = np.asarray(reference).T @ np.asarray(simulated)
    axis = np.array(
        [
            error[2, 1] - error[1, 2],
            error[0, 2] - error[2, 0],
            error[1, 0] - error[0, 1],
        ]
    )
    return float(np.arctan2(0.5 * np.linalg.norm(axis), 0.5 * (np.trace(error) - 1.0)))


def _errors(sim: dict, ref) -> dict:
    """Worst-over-quadrotors error of a simulated state vs. the flat reference.

    Every entry is a linear quantity in SI units — metres, m/s, rad, rad/s — so
    one tolerance means the same thing in all of them.
    """
    nQ = len(ref.positions)
    out = {
        "payload_pos": float(np.linalg.norm(sim["payload_position"] - ref.payload_position)),
        "payload_vel": float(np.linalg.norm(sim["payload_velocity"] - ref.payload_velocity)),
        # The one M3 could not measure: payload attitude, in radians.
        "payload_att": _attitude_angle(
            np.asarray(ref.payload_orientation), sim["payload_orientation"]
        ),
        "payload_omega": float(
            np.linalg.norm(
                sim["payload_angular_velocity"] - np.asarray(ref.payload_angular_velocity)
            )
        ),
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
        att = max(att, _attitude_angle(np.asarray(ref.orientations[i]), sim["orientations"][i]))
        body_w = max(
            body_w,
            float(
                np.linalg.norm(sim["angular_velocities"][i] - np.asarray(ref.angular_velocities[i]))
            ),
        )
    out.update(cable_q=cable_q, cable_omega=cable_w, attitude=att, body_omega=body_w)
    return out


def _model_snapshot(model: MultiQuadRigidbodyBase) -> dict:
    return {
        "payload_position": np.asarray(model.state.payload_position),
        "payload_velocity": np.asarray(model.state.payload_velocity),
        "payload_orientation": np.asarray(model.state.payload_orientation),
        "payload_angular_velocity": np.asarray(model.state.payload_angular_velocity),
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
    quat_L = y[6:10]
    quat_L = quat_L / np.linalg.norm(quat_L)
    Omega_L = y[10:13]
    q, w, quat, Omega = [], [], [], []
    base = 13
    for i in range(nQ):
        qi = y[base + 6 * i : base + 6 * i + 3]
        wi = y[base + 6 * i + 3 : base + 6 * i + 6]
        qi = qi / np.linalg.norm(qi)
        wi = wi - qi * wi.dot(qi)
        q.append(qi)
        w.append(wi)
    base = 13 + 6 * nQ
    for i in range(nQ):
        qu = y[base + 7 * i : base + 7 * i + 4]
        quat.append(qu / np.linalg.norm(qu))
        Omega.append(y[base + 7 * i + 4 : base + 7 * i + 7])
    return x_L, v_L, quat_L, Omega_L, q, w, quat, Omega


def _rot(quat_wxyz: np.ndarray) -> np.ndarray:
    return _ScipyRot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()


def _to_quat(R: np.ndarray) -> np.ndarray:
    xyzw = _ScipyRot.from_matrix(np.asarray(R)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


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


def _rhs(t, y, f2s: MultiQuadRigidbody, flats_of_t):
    """M4 ODE in continuous time, with the feedforward re-evaluated at exactly ``t``.

    Mirrors ``MultiQuadRigidbodyBase.accelerations``: assemble the 6×6 coupled
    payload system, solve once, then back-substitute the cable angular
    accelerations. Written out here rather than called through the model so
    that the gate does not compare the model against itself.
    """
    nQ = f2s.num_quadrotors
    x_L, v_L, quat_L, Omega_L, q, w, quat, Omega = _unpack(y, nQ)
    R_L = _rot(quat_L)
    _, inputs = f2s.recover(flats_of_t(t, nQ))

    ge3 = GRAVITY * _E3
    J_L = f2s.payload_inertia
    M11 = f2s.payload_mass * np.eye(3)
    M12 = np.zeros((3, 3))
    M22 = J_L.copy()
    h1 = np.zeros(3)
    h2 = -np.cross(Omega_L, J_L @ Omega_L)
    Omega_hat = hat(Omega_L)

    A, centripetal, u_perp = [], [], []
    for i in range(nQ):
        m_i = float(f2s.masses[i])
        l_i = float(f2s.cable_lengths[i])
        r_i = f2s.attachment_points[i]
        A_i = R_L @ hat(r_i)
        c_i = R_L @ (Omega_hat @ (Omega_hat @ r_i))
        A.append(A_i)
        centripetal.append(c_i)

        qqT = np.outer(q[i], q[i])
        u_i = inputs.thrusts[i] * (_rot(quat[i]) @ _E3)
        u_perp.append(u_i - qqT @ u_i)
        dq_i = np.cross(w[i], q[i])
        b_i = m_i * l_i * float(dq_i.dot(dq_i)) * q[i] - qqT @ u_i
        g_i = m_i * (qqT @ c_i) + b_i

        m_qqT = m_i * qqT
        M11 = M11 + m_qqT
        M12 = M12 - m_qqT @ A_i
        M22 = M22 + A_i.T @ m_qqT @ A_i
        h1 = h1 - g_i
        h2 = h2 + A_i.T @ g_i

    solution = np.linalg.solve(np.block([[M11, M12], [M12.T, M22]]), np.concatenate([h1, h2]))
    accel_plus_g = solution[:3]
    dOmega_L = solution[3:]

    out = [v_L, accel_plus_g - ge3, _quat_dot(quat_L, Omega_L), dOmega_L]
    for i in range(nQ):
        m_i = float(f2s.masses[i])
        l_i = float(f2s.cable_lengths[i])
        attach = accel_plus_g - A[i] @ dOmega_L + centripetal[i]
        out.append(np.cross(w[i], q[i]))
        out.append(np.cross(q[i], attach) / l_i - np.cross(q[i], u_perp[i]) / (m_i * l_i))
    for i in range(nQ):
        J = f2s.inertias[i]
        out.append(_quat_dot(quat[i], Omega[i]))
        out.append(np.linalg.solve(J, inputs.moments[i] - np.cross(Omega[i], J @ Omega[i])))
    return np.concatenate(out)


def run_dop853(flats_of_t, nQ: int, tf: float = TF_ODE, **kwargs) -> dict:
    f2s, _ = _build(nQ, **kwargs)
    ref0, _ = f2s.recover(flats_of_t(0.0, nQ))

    y0 = [
        np.asarray(ref0.payload_position),
        np.asarray(ref0.payload_velocity),
        _to_quat(np.asarray(ref0.payload_orientation)),
        np.asarray(ref0.payload_angular_velocity),
    ]
    for i in range(nQ):
        y0.append(np.asarray(ref0.cable_attitudes[i]))
        y0.append(np.asarray(ref0.cable_angular_velocities[i]))
    for i in range(nQ):
        y0.append(_to_quat(np.asarray(ref0.orientations[i])))
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

    x_L, v_L, quat_L, Omega_L, q, w, quat, Omega = _unpack(sol.y[:, -1], nQ)
    sim = {
        "payload_position": x_L,
        "payload_velocity": v_L,
        "payload_orientation": _rot(quat_L),
        "payload_angular_velocity": Omega_L,
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
        # Exact feedforward at a genuine equilibrium: nothing should move at
        # all, including the payload attitude. Works at every nQ because the
        # hover wrench lies inside range(Φ) even when the grasp is deficient.
        e = run_zoh(hover_flats, nQ)
        _assert_below(e, 1e-10, f"hover nQ={nQ}")

    def test_hover_with_heterogeneous_parameters(self):
        e = run_zoh(hover_flats, 4, masses=[0.7, 0.9, 1.1, 1.3], cable_lengths=[0.9, 1.1, 1.3, 1.5])
        _assert_below(e, 1e-10, "hover heterogeneous")

    def test_hover_with_an_asymmetric_grasp(self):
        e = run_zoh(hover_flats, 4, attachment_points=_ASYMMETRIC_ATTACHMENTS)
        _assert_below(e, 1e-10, "hover asymmetric grasp")


class TestOpenLoopDynamic:
    @pytest.mark.parametrize("nQ", [3, 4])
    def test_circle(self, nQ):
        e = run_zoh(circle_flats, nQ)
        _assert_below(e, TOL_ZOH, f"circle nQ={nQ}")

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_lissajous(self, nQ):
        e = run_zoh(lissajous_flats, nQ)
        _assert_below(e, TOL_ZOH, f"lissajous nQ={nQ}")

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_payload_spin(self, nQ):
        e = run_zoh(spin_flats, nQ)
        _assert_below(e, TOL_ZOH, f"spin nQ={nQ}")

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_full_pose_manoeuvre(self, nQ):
        e = run_zoh(pose_flats, nQ)
        _assert_below(e, TOL_ZOH, f"pose nQ={nQ}")

    def test_pose_with_heterogeneous_parameters(self):
        e = run_zoh(
            pose_flats, 4, masses=[0.7, 0.9, 1.1, 1.3], cable_lengths=[0.95, 1.05, 1.15, 1.25]
        )
        _assert_below(e, TOL_ZOH, "pose heterogeneous")


# ─── Tests: DOP853 — the actual correctness gate ──────────────────────


class TestHighOrderRoundTrip:
    """With integrator error at round-off, anything left is a formula bug."""

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_circle(self, nQ):
        e = run_dop853(circle_flats, nQ)
        _assert_below(e, TOL_INTEGRATION, f"circle nQ={nQ}")

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_lissajous(self, nQ):
        e = run_dop853(lissajous_flats, nQ)
        _assert_below(e, TOL_INTEGRATION, f"lissajous nQ={nQ}")

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_payload_spin(self, nQ):
        """Attitude channel in isolation: only ``R_L`` moves."""
        e = run_dop853(spin_flats, nQ)
        _assert_below(e, TOL_INTEGRATION, f"spin nQ={nQ}")

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_full_pose_manoeuvre(self, nQ):
        """The gate M3 has no analogue of — payload position *and* attitude."""
        e = run_dop853(pose_flats, nQ)
        _assert_below(e, TOL_INTEGRATION, f"pose nQ={nQ}")

    def test_heterogeneous_masses_and_cable_lengths(self):
        e = run_dop853(
            pose_flats, 4, masses=[0.7, 0.9, 1.1, 1.3], cable_lengths=[0.95, 1.05, 1.15, 1.25]
        )
        _assert_below(e, TOL_INTEGRATION, "pose heterogeneous")

    def test_asymmetric_grasp_geometry(self):
        """Nothing in the recovery assumes the ring — only that ``rank Φ = 6``."""
        e = run_dop853(pose_flats, 4, attachment_points=_ASYMMETRIC_ATTACHMENTS)
        _assert_below(e, TOL_INTEGRATION, "pose asymmetric grasp")

    def test_single_quadrotor_at_the_centre_of_mass(self):
        """``r = 0`` collapses M4 to M3: one cable, no moment channel."""
        e = run_dop853(lissajous_flats, 1, attachment_points=np.zeros((1, 3)))
        _assert_below(e, TOL_INTEGRATION, "lissajous nQ=1, r=0")
