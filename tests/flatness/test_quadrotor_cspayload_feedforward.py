"""Open-loop feedforward tracking tests for the cspayload flat-to-state map.

The flat-output-derived (f, M) is fed straight into the wrench-input
dynamics of the quadrotor + cable-suspended payload model — no closed-loop
controller. Tracking error after a finite horizon should be bounded by the
integrator's residual.

Two integrators are exercised:
  - The shipped ZOH-Euler in QuadrotorCsPayloadBase (h = 1.25 ms inner step,
    feedforward held constant across the 5 ms outer step).
  - A high-order adaptive RK (DOP853 via scipy.integrate.solve_ivp)
    re-evaluating (f, M) continuously and using a quaternion attitude
    parameterisation. Drives integrator error effectively to round-off, so
    any non-zero residue is direct evidence of a flat-to-state formula bug.

The 14-D cspayload state is laid out for DOP853 as
    y = [x_L (3), v_L (3), q (3), ω (3), q_wxyz (4), Ω (3)]   (19D)
with q ∈ S² and ω · q = 0 enforced by the dynamics; the quaternion is
re-normalised at every RHS call to absorb round-off.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as _ScipyRot

from udaan.core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from udaan.manif import SO3
from udaan.models.quadrotor_cspayload.base import QuadrotorCsPayloadBase
from udaan.utils.flatness import (
    Jet,
    QuadrotorCsPayload,
    QuadrotorCsPayloadFlats,
)

TF = 5.0


# ─── Trajectory builders ──────────────────────────────────────────────


def hover_flats(t: float) -> QuadrotorCsPayloadFlats:
    """Stationary hover: payload at [0, 0, 0]; quadrotor sits at [0, 0, ℓ]."""
    x_data = np.zeros((7, 3))
    psi_data = np.zeros(3)
    return QuadrotorCsPayloadFlats(x_L=Jet(x_data), psi=Jet(psi_data))


def circle_flats(
    t: float, r: float = 0.4, omega: float = 1.0, z0: float = 0.0
) -> QuadrotorCsPayloadFlats:
    """Payload circle in the XY plane at height z0; yaw tracks heading angle ωt.

    Radius and angular speed are kept modest so the cable stays well clear of
    its slack-cable singularity (T > 0) and the swing angle is moderate.
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
    psi_data = np.array([omega * t, omega, 0.0])
    return QuadrotorCsPayloadFlats(x_L=Jet(x_data), psi=Jet(psi_data))


# ─── Harness ──────────────────────────────────────────────────────────


def _f2s_default() -> QuadrotorCsPayload:
    return QuadrotorCsPayload(
        mass_q=DEFAULT_QUAD_MASS,
        mass_l=DEFAULT_PAYLOAD_MASS,
        inertia=DEFAULT_QUAD_INERTIA,
        cable_length=DEFAULT_CABLE_LENGTH,
    )


def _seed_model(model: QuadrotorCsPayloadBase, ref) -> None:
    state = ref.as_state()
    model.reset(
        payload_position=np.asarray(state.payload_position).copy(),
        payload_velocity=np.asarray(state.payload_velocity).copy(),
        cable_attitude=state.cable_attitude,
        cable_angular_velocity=state.cable_angular_velocity,
        orientation=state.orientation,
        angular_velocity=state.angular_velocity,
    )


def _final_errors(model: QuadrotorCsPayloadBase, ref_final) -> dict:
    """Errors of the model state vs. the analytic flat reference at model.t."""
    pos_L = float(np.linalg.norm(model.state.payload_position - ref_final.payload_position))
    vel_L = float(np.linalg.norm(model.state.payload_velocity - ref_final.payload_velocity))
    q_err = float(
        np.linalg.norm(
            np.asarray(model.state.cable_attitude) - np.asarray(ref_final.cable_attitude)
        )
    )
    om_cable_err = float(
        np.linalg.norm(
            np.asarray(model.state.cable_angular_velocity) - ref_final.cable_angular_velocity
        )
    )
    att_err = float(SO3(np.asarray(ref_final.orientation)).config_error(model.state.orientation))
    om_err = float(
        np.linalg.norm(
            np.asarray(model.state.angular_velocity) - np.asarray(ref_final.angular_velocity)
        )
    )
    return {
        "payload_pos": pos_L,
        "payload_vel": vel_L,
        "cable_q": q_err,
        "cable_omega": om_cable_err,
        "attitude": att_err,
        "body_omega": om_err,
    }


def _run_open_loop(flats_of_t, tf: float = TF) -> dict:
    """Seed the model from the t=0 reference, march with pure feedforward,
    return final-state errors against the flat reference at tf."""
    f2s = _f2s_default()
    model = QuadrotorCsPayloadBase(input="wrench")

    ref0, _ = f2s.recover(flats_of_t(0.0))
    _seed_model(model, ref0)

    while model.t < tf - 1e-9:
        _, inputs = f2s.recover(flats_of_t(model.t))
        u = np.array([inputs.thrust, *inputs.moment])
        model.step(u)

    ref_final, _ = f2s.recover(flats_of_t(tf))
    return _final_errors(model, ref_final)


# ─── Tests ────────────────────────────────────────────────────────────


class TestOpenLoopHover:
    def test_tracking_error_is_purely_numerical(self):
        e = _run_open_loop(hover_flats)

        # Pure-feedforward hover with exact thrust = (m_Q + m_L) g and
        # zero moment: the dynamics should stay perfectly stationary up
        # to machine epsilon.
        assert e["payload_pos"] < 1e-10, f"payload pos drift {e['payload_pos']:.2e} m"
        assert e["payload_vel"] < 1e-10, f"payload vel drift {e['payload_vel']:.2e} m/s"
        assert e["cable_q"] < 1e-10, f"cable q drift {e['cable_q']:.2e}"
        assert e["cable_omega"] < 1e-10, f"cable ω drift {e['cable_omega']:.2e} rad/s"
        assert e["attitude"] < 1e-10, f"attitude drift {e['attitude']:.2e}"
        assert e["body_omega"] < 1e-10, f"body Ω drift {e['body_omega']:.2e} rad/s"


class TestOpenLoopCircle:
    def test_tracking_error_is_purely_numerical(self):
        e = _run_open_loop(circle_flats)

        # ZOH-Euler at h = 1.25 ms over TF s. Generous bounds (~10× the
        # single-quadrotor tolerance, since the cable adds two
        # integration paths and S²/TS² subspace projection).
        assert e["payload_pos"] < 5e-2, f"payload pos {e['payload_pos']:.2e} m"
        assert e["payload_vel"] < 5e-2, f"payload vel {e['payload_vel']:.2e} m/s"
        assert e["cable_q"] < 5e-2, f"cable q {e['cable_q']:.2e}"
        assert e["cable_omega"] < 5e-2, f"cable ω {e['cable_omega']:.2e} rad/s"
        assert e["attitude"] < 5e-2, f"attitude {e['attitude']:.2e}"
        assert e["body_omega"] < 5e-2, f"body Ω {e['body_omega']:.2e} rad/s"


# ─── High-order ODE integration (DOP853 + quaternion attitude) ────────


def _cspayload_rhs(t, y, m_Q, m_L, L, J, J_inv, flats_of_t, f2s):
    """Full cspayload ODE in the 19-D layout
        y = [x_L, v_L, q, ω, q_wxyz, Ω].

    Re-evaluates the flat-to-state feedforward (f, M) at the *exact*
    current t — strictly tighter than the ZOH path. The S² / TS²
    constraints (‖q‖ = 1, ω · q = 0) are preserved exactly by the
    continuous-time dynamics; small numerical drift is absorbed by
    re-normalising q and re-projecting ω each call.
    """
    x_L = y[0:3]
    v_L = y[3:6]
    q = y[6:9]
    omega = y[9:12]
    q_wxyz = y[12:16]
    Omega = y[16:19]

    # Numerical hygiene: re-project onto S² / T_q S² / unit-quaternion.
    q = q / np.linalg.norm(q)
    omega = omega - q * (omega.dot(q))
    q_wxyz = q_wxyz / np.linalg.norm(q_wxyz)

    R = _ScipyRot.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]  # scipy uses xyzw
    ).as_matrix()

    _, inputs = f2s.recover(flats_of_t(t))
    f = inputs.thrust
    M = inputs.moment

    e3 = np.array([0.0, 0.0, 1.0])
    fRe3 = f * (R @ e3)
    g_e3 = GRAVITY * e3

    # Translational payload dynamics:
    #   (m_Q + m_L)(v̇_L + g e_3) = (q · fRe3 - m_Q L ‖ω‖²) q
    # (using ‖q̇‖² = ‖ω × q‖² = ‖ω‖² for ω ⊥ q, ‖q‖ = 1)
    omega_sq = float(omega.dot(omega))
    scalar = (q.dot(fRe3) - m_Q * L * omega_sq) / (m_Q + m_L)
    v_L_dot = -g_e3 + scalar * q

    # Cable kinematics:  q̇ = ω × q
    q_dot = np.cross(omega, q)

    # Cable angular dynamics:  m_Q L ω̇ = -q × fRe3
    omega_dot = -np.cross(q, fRe3) / (m_Q * L)

    # Quaternion kinematics:  q̇ = ½ q ⊗ [0, Ω]   (Hamilton, scalar-first)
    w, x, y_, z = q_wxyz
    p_, q_, r_ = Omega
    q_quat_dot = 0.5 * np.array(
        [
            -x * p_ - y_ * q_ - z * r_,
            w * p_ - z * q_ + y_ * r_,
            z * p_ + w * q_ - x * r_,
            -y_ * p_ + x * q_ + w * r_,
        ]
    )

    # Body-rate dynamics:  J Ω̇ = M - Ω × J Ω
    Omega_dot = J_inv @ (M - np.cross(Omega, J @ Omega))

    return np.concatenate([v_L, v_L_dot, q_dot, omega_dot, q_quat_dot, Omega_dot])


def _run_high_order(flats_of_t, tf: float = TF) -> dict:
    """Same harness as `_run_open_loop`, but integrate with adaptive DOP853
    (8th-order RK with adaptive step). Tight tolerances drive integrator
    error to round-off, isolating any flat-to-state formula error in the
    residual."""
    f2s = _f2s_default()
    m_Q = DEFAULT_QUAD_MASS
    m_L = DEFAULT_PAYLOAD_MASS
    L = DEFAULT_CABLE_LENGTH
    J = np.asarray(DEFAULT_QUAD_INERTIA, dtype=float)
    J_inv = np.linalg.inv(J)

    ref0, _ = f2s.recover(flats_of_t(0.0))
    R0 = np.asarray(ref0.orientation)
    q0_xyzw = _ScipyRot.from_matrix(R0).as_quat()
    q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])

    y0 = np.concatenate(
        [
            np.asarray(ref0.payload_position),
            np.asarray(ref0.payload_velocity),
            np.asarray(ref0.cable_attitude),
            np.asarray(ref0.cable_angular_velocity),
            q0_wxyz,
            np.asarray(ref0.angular_velocity),
        ]
    )

    sol = solve_ivp(
        _cspayload_rhs,
        (0.0, tf),
        y0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        args=(m_Q, m_L, L, J, J_inv, flats_of_t, f2s),
        dense_output=False,
    )
    assert sol.success, f"DOP853 failed: {sol.message}"

    yf = sol.y[:, -1]
    ref_final, _ = f2s.recover(flats_of_t(tf))

    # Reconstruct sim quantities for the same comparison shape as ZOH.
    q_sim = yf[6:9] / np.linalg.norm(yf[6:9])
    om_cable_sim = yf[9:12]
    om_cable_sim = om_cable_sim - q_sim * om_cable_sim.dot(q_sim)
    q_wxyz = yf[12:16] / np.linalg.norm(yf[12:16])
    R_sim = _ScipyRot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()

    payload_pos = float(np.linalg.norm(yf[0:3] - ref_final.payload_position))
    payload_vel = float(np.linalg.norm(yf[3:6] - ref_final.payload_velocity))
    cable_q = float(np.linalg.norm(q_sim - np.asarray(ref_final.cable_attitude)))
    cable_omega = float(np.linalg.norm(om_cable_sim - ref_final.cable_angular_velocity))
    attitude = float(SO3(np.asarray(ref_final.orientation)).config_error(SO3(R_sim)))
    body_omega = float(np.linalg.norm(yf[16:19] - np.asarray(ref_final.angular_velocity)))

    return {
        "payload_pos": payload_pos,
        "payload_vel": payload_vel,
        "cable_q": cable_q,
        "cable_omega": cable_omega,
        "attitude": attitude,
        "body_omega": body_omega,
    }


class TestOpenLoopCircleHighOrder:
    """The circle test, re-run with DOP853 + quaternion attitude.

    Errors should drop several orders of magnitude vs. the ZOH-Euler
    version — confirming that the residual ZOH error is integrator
    noise, not a flat-to-state formula bug.
    """

    def test_high_order_drives_error_to_roundoff(self):
        e = _run_high_order(circle_flats)
        # DOP853 at rtol=atol=1e-12 should drive payload position drift
        # to sub-µm and attitude drift to round-off. Bounds are generous
        # (~3 orders of magnitude above expected) to leave headroom.
        assert e["payload_pos"] < 1e-6, f"payload pos {e['payload_pos']:.2e} m"
        assert e["payload_vel"] < 1e-6, f"payload vel {e['payload_vel']:.2e} m/s"
        assert e["cable_q"] < 1e-7, f"cable q {e['cable_q']:.2e}"
        assert e["cable_omega"] < 1e-7, f"cable ω {e['cable_omega']:.2e} rad/s"
        assert e["attitude"] < 1e-9, f"attitude {e['attitude']:.2e}"
        assert e["body_omega"] < 1e-9, f"body Ω {e['body_omega']:.2e} rad/s"


# ─── Diagnostic — run with `pytest -s` to print the actual numbers ────


@pytest.mark.parametrize("name,builder", [("hover", hover_flats), ("circle", circle_flats)])
def test_print_errors_zoh(name, builder):
    e = _run_open_loop(builder)
    print(
        f"\n  [{name}] ZOH-Euler errors @ t={TF:.1f}s:"
        f"\n    payload pos : {e['payload_pos']:.3e} m"
        f"\n    payload vel : {e['payload_vel']:.3e} m/s"
        f"\n    cable q     : {e['cable_q']:.3e}"
        f"\n    cable ω     : {e['cable_omega']:.3e} rad/s"
        f"\n    attitude Ψ  : {e['attitude']:.3e}"
        f"\n    body Ω      : {e['body_omega']:.3e} rad/s"
    )


def test_print_errors_dop853_circle():
    e = _run_high_order(circle_flats)
    print(
        f"\n  [circle, DOP853 rtol=atol=1e-12] errors @ t={TF:.1f}s:"
        f"\n    payload pos : {e['payload_pos']:.3e} m"
        f"\n    payload vel : {e['payload_vel']:.3e} m/s"
        f"\n    cable q     : {e['cable_q']:.3e}"
        f"\n    cable ω     : {e['cable_omega']:.3e} rad/s"
        f"\n    attitude Ψ  : {e['attitude']:.3e}"
        f"\n    body Ω      : {e['body_omega']:.3e} rad/s"
    )
