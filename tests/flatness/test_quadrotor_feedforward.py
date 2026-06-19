"""Open-loop feedforward tracking tests for the quadrotor flat-to-state map.

The flat-output-derived (f, M) is fed straight into the wrench-input dynamics
without any closed-loop controller. Tracking error after 10 s should be
bounded by the integrator's residual — purely numerical, not controller-
driven.

Two integrators are exercised:
  - The shipped ZOH-Euler in QuadrotorBase (h = 1.25 ms inner step,
    feed-forward held constant across the 5 ms outer step).
  - A high-order adaptive RK (DOP853 via scipy.integrate.solve_ivp)
    re-evaluating (f, M) continuously and using a quaternion attitude
    parameterisation. Drives integrator error effectively to round-off,
    so any non-zero residue is direct evidence of a flat-to-state
    formula bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as _ScipyRot

from udaan.core.defaults import DEFAULT_QUAD_INERTIA, DEFAULT_QUAD_MASS, GRAVITY
from udaan.flatness import Jet, Quadrotor, QuadrotorFlats
from udaan.manif import SO3
from udaan.models.quadrotor import QuadrotorBase

TF = 10.0


# ─── Trajectory builders ──────────────────────────────────────────────


def hover_flats(t: float) -> QuadrotorFlats:
    """Stationary hover at [0, 0, 1] with yaw = 0."""
    x_data = np.zeros((5, 3))
    x_data[0] = np.array([0.0, 0.0, 1.0])
    psi_data = np.zeros(3)
    return QuadrotorFlats(x=Jet(x_data), psi=Jet(psi_data))


def circle_flats(t: float, r: float = 1.0, omega: float = 1.0, z0: float = 1.0) -> QuadrotorFlats:
    """Circle in the XY plane at height z0; yaw tracks heading angle ωt."""
    c, s = np.cos(omega * t), np.sin(omega * t)
    x_data = np.array(
        [
            [r * c, r * s, z0],
            [-r * omega * s, r * omega * c, 0.0],
            [-r * omega**2 * c, -r * omega**2 * s, 0.0],
            [r * omega**3 * s, -r * omega**3 * c, 0.0],
            [r * omega**4 * c, r * omega**4 * s, 0.0],
        ]
    )
    psi_data = np.array([omega * t, omega, 0.0])
    return QuadrotorFlats(x=Jet(x_data), psi=Jet(psi_data))


# ─── Harness ──────────────────────────────────────────────────────────


def _f2s_default() -> Quadrotor:
    return Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)


def _seed_model(model: QuadrotorBase, ref) -> None:
    model.reset(
        position=np.asarray(ref.position).copy(),
        velocity=np.asarray(ref.velocity).copy(),
        orientation=ref.orientation,
        angular_velocity=ref.angular_velocity,
    )


def _run_open_loop(flats_of_t, tf: float = TF) -> tuple:
    """Seed the model from the t=0 reference, march with pure feedforward,
    return final-state errors against the flat reference at tf."""
    f2s = _f2s_default()
    model = QuadrotorBase(input="wrench")

    # Seed
    ref0, _ = f2s.recover(flats_of_t(0.0))
    _seed_model(model, ref0)

    # March
    while model.t < tf - 1e-9:
        ref, inputs = f2s.recover(flats_of_t(model.t))
        u = np.array([inputs.thrust, *inputs.moment])
        model.step(u)

    # Compare final state to reference at tf
    ref_final, _ = f2s.recover(flats_of_t(tf))
    pos_err = float(np.linalg.norm(model.state.position - ref_final.position))
    vel_err = float(np.linalg.norm(model.state.velocity - ref_final.velocity))
    att_err = float(SO3(np.asarray(ref_final.orientation)).config_error(model.state.orientation))
    omega_err = float(
        np.linalg.norm(
            np.asarray(model.state.angular_velocity) - np.asarray(ref_final.angular_velocity)
        )
    )
    return pos_err, vel_err, att_err, omega_err


# ─── Tests ────────────────────────────────────────────────────────────


class TestOpenLoopHover:
    def test_tracking_error_is_purely_numerical(self):
        pos, vel, att, om = _run_open_loop(hover_flats)

        # Pure-feedforward hover with exact m·g thrust: the dynamics
        # should stay perfectly stationary up to machine epsilon.
        assert pos < 1e-10, f"position drift {pos:.2e} m"
        assert vel < 1e-10, f"velocity drift {vel:.2e} m/s"
        assert att < 1e-10, f"attitude drift {att:.2e}"
        assert om < 1e-10, f"angular-velocity drift {om:.2e} rad/s"


class TestOpenLoopCircle:
    def test_tracking_error_is_purely_numerical(self):
        pos, vel, att, om = _run_open_loop(circle_flats)

        # ZOH-Euler at h = 1.25 ms over 10 s = 8000 substeps. Local
        # truncation error O(h²) ≈ 1.5e-6, accumulated O(t·h) — sub-cm
        # position drift, sub-degree attitude drift expected. Bounds set
        # generously (~10× expected) to leave headroom for downstream
        # changes that don't break the flatness math.
        assert pos < 1e-2, f"position error {pos:.2e} m"
        assert vel < 5e-2, f"velocity error {vel:.2e} m/s"
        assert att < 1e-2, f"attitude error {att:.2e} (Morse-style)"
        assert om < 5e-2, f"angular-velocity error {om:.2e} rad/s"


# ─── High-order ODE integration (DOP853 + quaternion attitude) ────────


def _rigid_body_rhs(t, y, m, J, J_inv, flats_of_t, f2s):
    """Rigid-body ODE in state y = [x(3), v(3), q_wxyz(4), ω(3)] = 13D.

    Uses the flat-to-state map at the *exact* current t for (f, M), not
    a held value from the last outer-step grid point — so this is a
    strictly tighter test of the flatness formula than the ZOH path.
    """
    pos, vel, q_wxyz, omega = y[0:3], y[3:6], y[6:10], y[10:13]
    # Re-normalise quaternion against round-off (DOP853 tolerances handle
    # this fine in practice, but cheap insurance).
    q_wxyz = q_wxyz / np.linalg.norm(q_wxyz)
    R = _ScipyRot.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]  # scipy uses xyzw
    ).as_matrix()

    _, inputs = f2s.recover(flats_of_t(t))
    f, M = inputs.thrust, inputs.moment

    # Translational
    e3 = np.array([0.0, 0.0, 1.0])
    accel = (f / m) * (R @ e3) - GRAVITY * e3

    # Quaternion kinematics:  q̇ = ½ q ⊗ [0, ω]   (Hamilton, scalar-first)
    w, x, y_, z = q_wxyz
    p, q, r = omega
    q_dot = 0.5 * np.array(
        [
            -x * p - y_ * q - z * r,
            w * p - z * q + y_ * r,
            z * p + w * q - x * r,
            -y_ * p + x * q + w * r,
        ]
    )

    # Rotational:  J ω̇ = M − ω × J ω
    omega_dot = J_inv @ (M - np.cross(omega, J @ omega))

    return np.concatenate([vel, accel, q_dot, omega_dot])


def _run_high_order(flats_of_t, tf: float = TF) -> tuple:
    """Same harness as `_run_open_loop`, but integrate with adaptive
    DOP853 (8th-order RK with adaptive step). Tight tolerances drive
    integrator error to round-off, isolating any flat-to-state formula
    error in the residual."""
    f2s = _f2s_default()
    m = DEFAULT_QUAD_MASS
    J = np.asarray(DEFAULT_QUAD_INERTIA, dtype=float)
    J_inv = np.linalg.inv(J)

    # Seed from reference at t=0
    ref0, _ = f2s.recover(flats_of_t(0.0))
    R0 = np.asarray(ref0.orientation)
    q0_xyzw = _ScipyRot.from_matrix(R0).as_quat()
    q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])
    y0 = np.concatenate(
        [
            np.asarray(ref0.position),
            np.asarray(ref0.velocity),
            q0_wxyz,
            np.asarray(ref0.angular_velocity),
        ]
    )

    sol = solve_ivp(
        _rigid_body_rhs,
        (0.0, tf),
        y0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        args=(m, J, J_inv, flats_of_t, f2s),
        dense_output=False,
    )
    assert sol.success, f"DOP853 failed: {sol.message}"

    yf = sol.y[:, -1]
    pos_sim = yf[0:3]
    vel_sim = yf[3:6]
    q_wxyz = yf[6:10] / np.linalg.norm(yf[6:10])
    R_sim = _ScipyRot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    omega_sim = yf[10:13]

    ref_final, _ = f2s.recover(flats_of_t(tf))
    pos_err = float(np.linalg.norm(pos_sim - ref_final.position))
    vel_err = float(np.linalg.norm(vel_sim - ref_final.velocity))
    att_err = float(SO3(np.asarray(ref_final.orientation)).config_error(SO3(R_sim)))
    omega_err = float(np.linalg.norm(omega_sim - np.asarray(ref_final.angular_velocity)))
    return pos_err, vel_err, att_err, omega_err


class TestOpenLoopCircleHighOrder:
    """The circle test, re-run with DOP853 + quaternion attitude.

    Errors should drop several orders of magnitude vs. the ZOH-Euler
    version — confirming that the residual ZOH error is integrator
    noise, not a flat-to-state formula bug.
    """

    def test_high_order_drives_error_to_roundoff(self):
        pos, vel, att, om = _run_high_order(circle_flats)
        # DOP853 at rtol=atol=1e-12 should drive position drift to
        # picometer scale and attitude drift to round-off. Bounds are
        # generous (~3 orders of magnitude above expected).
        assert pos < 1e-7, f"position {pos:.2e} m"
        assert vel < 1e-7, f"velocity {vel:.2e} m/s"
        assert att < 1e-10, f"attitude {att:.2e}"
        assert om < 1e-10, f"angular vel {om:.2e} rad/s"


# ─── Diagnostic — run with `pytest -s` to print the actual numbers ────


@pytest.mark.parametrize("name,builder", [("hover", hover_flats), ("circle", circle_flats)])
def test_print_errors_zoh(name, builder):
    pos, vel, att, om = _run_open_loop(builder)
    print(
        f"\n  [{name}] ZOH-Euler errors @ t={TF:.1f}s:"
        f"\n    position    : {pos:.3e} m"
        f"\n    velocity    : {vel:.3e} m/s"
        f"\n    attitude Ψ  : {att:.3e}"
        f"\n    angular vel : {om:.3e} rad/s"
    )


def test_print_errors_dop853_circle():
    pos, vel, att, om = _run_high_order(circle_flats)
    print(
        f"\n  [circle, DOP853 rtol=atol=1e-12] errors @ t={TF:.1f}s:"
        f"\n    position    : {pos:.3e} m"
        f"\n    velocity    : {vel:.3e} m/s"
        f"\n    attitude Ψ  : {att:.3e}"
        f"\n    angular vel : {om:.3e} rad/s"
    )
