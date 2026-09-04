"""Open-loop feedforward round trip for the M5 flat-to-state map.

The flat-derived ``(f, M)`` is fed straight into the Step-9 model
(:class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`) with **no
controller at all**. If the map is exact, the model's trajectory reproduces the
flat outputs; anything beyond the integrator's own residual is a formula bug.

Two integrators, for two different jobs:

* **DOP853** (adaptive 8th-order RK, ``rtol = atol = 1e-12``) driving the
  model's own :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.accelerations`
  — the entire right-hand side of the ODE — with ``(f, M)`` re-evaluated at the
  exact current time. This drives integrator error to round-off, so the residual
  *is* the formula error. It is also the only usable integrator at larger link
  counts: Step 9 established that this system is conditionally stable under
  ZOH-Euler (``stability_growth_rate() > 1`` diverges within a few seconds, and
  the default chain crosses that at ``n = 5``). An adaptive RK has no such
  limit, so **every** ``n`` in this file is covered by the DOP853 path.

* **ZOH-Euler** — the model's shipped ``step()``, feedforward held constant
  across the 5 ms outer step, exercising the real input pipeline including
  actuator clipping. Restricted to ``n ≤ 3`` on the default chain, where
  ``stability_growth_rate()`` is 0.008 / 0.16 / 0.45 and a 3 s run stays
  comfortably bounded.

The state is laid out for DOP853 as

    y = [x_L (3), v_L (3), {q_i (3), ω_i (3)}_{i=0}^{n-1}, quat_wxyz (4), Ω (3)]

i.e. ``6n + 13`` doubles, with ``‖q_i‖ = 1``, ``ω_i · q_i = 0`` and the
quaternion re-normalised at every RHS call to absorb round-off.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as _ScipyRot

from udaan.flatness import Jet, QuadrotorFlexCable, QuadrotorFlexCableFlats
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor_flexcable.base import QuadrotorFlexCableBase

TF = 5.0
TF_ZOH = 3.0

# The DOP853 round trip measures 1e-14 … 1.6e-11: integrator round-off over 5 s.
# ``tests.conftest.TOL_INTEGRATION`` (1e-6) would leave five orders of slack —
# enough for a 1e-7 formula error to pass unnoticed — so this file carries its
# own bound, ~60x the worst residual actually observed (the link angular
# velocities at n = 2).
TOL_ROUNDTRIP = 1e-9


# ─── Trajectory builders ──────────────────────────────────────────────


def _sine_rows(t: float, order: int, amps, omegas, phases, offset) -> np.ndarray:
    """Closed-form derivative rows of ``x_j(t) = offset_j + A_j cos(ω_j t + φ_j)``.

    Every row is exact at any order, which is what makes the ``2n + 4``
    requirement reachable without numerical differentiation — at ``n = 5`` that
    is the 14th derivative.
    """
    data = np.zeros((order + 1, 3))
    for k in range(order + 1):
        for j in range(3):
            data[k, j] = (
                amps[j] * omegas[j] ** k * np.cos(omegas[j] * t + phases[j] + k * np.pi / 2)
            )
    data[0] += np.asarray(offset, dtype=float)
    return data


def hover_flats(f2s: QuadrotorFlexCable):
    """Stationary hover: the payload sits still and the chain hangs straight."""

    def builder(t: float) -> QuadrotorFlexCableFlats:
        return QuadrotorFlexCableFlats(
            x_L=Jet(np.zeros((f2s.jet_order + 1, 3))), psi=Jet(np.zeros(3))
        )

    return builder


def lissajous_flats(f2s: QuadrotorFlexCable):
    """A dynamic 3-D payload trajectory with a moving yaw.

    Amplitudes and rates are chosen so the chain stays comfortably taut
    (``‖ẍ_L + g e₃‖`` never approaches zero — the peak vertical acceleration is
    under 0.1 g) and the required thrust stays inside the model's 20 N actuator
    limit, which the ZOH path enforces.
    """

    def builder(t: float) -> QuadrotorFlexCableFlats:
        x_L = Jet(
            _sine_rows(
                t,
                f2s.jet_order,
                [0.3, 0.3, 0.12],
                [1.2, 1.2, 0.9],
                [0.0, -np.pi / 2, 0.4],
                [0.0, 0.0, 0.0],
            )
        )
        psi = Jet(
            np.array([0.2 * np.sin(0.7 * t), 0.14 * np.cos(0.7 * t), -0.098 * np.sin(0.7 * t)])
        )
        return QuadrotorFlexCableFlats(x_L=x_L, psi=psi)

    return builder


TRAJECTORIES = {"hover": hover_flats, "lissajous": lissajous_flats}


# ─── State packing ────────────────────────────────────────────────────


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    xyzw = _ScipyRot.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = quat_wxyz / np.linalg.norm(quat_wxyz)
    return _ScipyRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _pack(ref, n: int) -> np.ndarray:
    parts = [np.asarray(ref.payload_position), np.asarray(ref.payload_velocity)]
    for i in range(n):
        parts.append(np.asarray(ref.link_attitudes[i]))
        parts.append(np.asarray(ref.link_angular_velocities[i]))
    parts.append(_quat_wxyz(np.asarray(ref.orientation)))
    parts.append(np.asarray(ref.angular_velocity))
    return np.concatenate(parts)


def _unpack(y: np.ndarray, n: int):
    x_L, v_L = y[0:3], y[3:6]
    idx = 6
    directions, rates = [], []
    for _ in range(n):
        q = y[idx : idx + 3]
        q = q / np.linalg.norm(q)
        omega = y[idx + 3 : idx + 6]
        omega = omega - q * float(omega.dot(q))
        directions.append(q)
        rates.append(omega)
        idx += 6
    return x_L, v_L, directions, rates, y[idx : idx + 4], y[idx + 4 : idx + 7]


# ─── DOP853 harness ───────────────────────────────────────────────────


def _rhs(t, y, model: QuadrotorFlexCableBase, f2s: QuadrotorFlexCable, flats_of_t):
    """The model's own equations of motion, driven by the flat feedforward.

    ``accelerations()`` is documented as the whole right-hand side of the ODE,
    so writing the (re-projected) integrator state onto ``model.state`` and
    calling it is the same dynamics the shipped ``step()`` integrates — minus
    the ZOH hold and the actuator clipping.
    """
    n = model.num_links
    x_L, v_L, directions, rates, quat, Omega = _unpack(y, n)

    model.state.payload_position = x_L
    model.state.payload_velocity = v_L
    model.state.link_attitudes = [S2(q) for q in directions]
    model.state.link_angular_velocities = [TS2(w) for w in rates]
    model.state.orientation = SO3(_matrix(quat))
    model.state.angular_velocity = TSO3(Omega)

    _, inputs = f2s.recover(flats_of_t(t))
    acc = model.accelerations(inputs.thrust, inputs.moment)

    out = [v_L, acc["payload_accel"]]
    for i in range(n):
        out.append(np.cross(rates[i], directions[i]))
        out.append(acc["link_ang_accels"][i])

    # Quaternion kinematics: q̇ = ½ q ⊗ [0, Ω]  (Hamilton, scalar-first)
    q = quat / np.linalg.norm(quat)
    w, x, y_, z = q
    p, r, s = Omega
    out.append(
        0.5
        * np.array(
            [
                -x * p - y_ * r - z * s,
                w * p - z * r + y_ * s,
                z * p + w * r - x * s,
                -y_ * p + x * r + w * s,
            ]
        )
    )
    out.append(acc["quad_ang_accel"])
    return np.concatenate(out)


def _errors(x_L, v_L, directions, rates, R, Omega, ref, f2s) -> dict:
    n = f2s.num_links
    q_err = max(
        float(np.linalg.norm(directions[i] - np.asarray(ref.link_attitudes[i]))) for i in range(n)
    )
    w_err = max(
        float(np.linalg.norm(rates[i] - np.asarray(ref.link_angular_velocities[i])))
        for i in range(n)
    )
    x_Q = x_L - sum(f2s.link_lengths[i] * directions[i] for i in range(n))
    return {
        "payload_pos": float(np.linalg.norm(x_L - ref.payload_position)),
        "payload_vel": float(np.linalg.norm(v_L - ref.payload_velocity)),
        "quad_pos": float(np.linalg.norm(x_Q - ref.position)),
        "link_q": q_err,
        "link_omega": w_err,
        "attitude": abs(float(SO3(np.asarray(ref.orientation)).config_error(SO3(R)))),
        "body_omega": float(np.linalg.norm(Omega - np.asarray(ref.angular_velocity))),
    }


def run_high_order(n: int, trajectory: str, tf: float = TF, **model_kwargs) -> dict:
    """Seed from the flat reference at t = 0, integrate open loop, compare at tf."""
    model = QuadrotorFlexCableBase(num_links=n, **model_kwargs)
    f2s = QuadrotorFlexCable.from_model(model)
    flats_of_t = TRAJECTORIES[trajectory](f2s)

    ref0, _ = f2s.recover(flats_of_t(0.0))
    sol = solve_ivp(
        _rhs,
        (0.0, tf),
        _pack(ref0, n),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        args=(model, f2s, flats_of_t),
    )
    assert sol.success, f"DOP853 failed: {sol.message}"

    x_L, v_L, directions, rates, quat, Omega = _unpack(sol.y[:, -1], n)
    ref_final, _ = f2s.recover(flats_of_t(tf))
    return _errors(x_L, v_L, directions, rates, _matrix(quat), Omega, ref_final, f2s)


# ─── ZOH-Euler harness ────────────────────────────────────────────────


def run_zoh(n: int, trajectory: str, tf: float = TF_ZOH, **model_kwargs) -> dict:
    model = QuadrotorFlexCableBase(num_links=n, input="wrench", **model_kwargs)
    f2s = QuadrotorFlexCable.from_model(model)
    flats_of_t = TRAJECTORIES[trajectory](f2s)

    ref0, _ = f2s.recover(flats_of_t(0.0))
    model.reset(
        payload_position=np.asarray(ref0.payload_position).copy(),
        payload_velocity=np.asarray(ref0.payload_velocity).copy(),
        link_attitudes=[np.asarray(q).copy() for q in ref0.link_attitudes],
        link_angular_velocities=[np.asarray(w).copy() for w in ref0.link_angular_velocities],
        orientation=np.asarray(ref0.orientation).copy(),
        angular_velocity=np.asarray(ref0.angular_velocity).copy(),
    )

    while model.t < tf - 1e-9:
        _, inputs = f2s.recover(flats_of_t(model.t))
        model.step(np.array([inputs.thrust, *inputs.moment]))

    ref_final, _ = f2s.recover(flats_of_t(tf))
    directions = [np.asarray(q) for q in model.state.link_attitudes]
    rates = [np.asarray(w) for w in model.state.link_angular_velocities]
    return _errors(
        np.asarray(model.state.payload_position),
        np.asarray(model.state.payload_velocity),
        directions,
        rates,
        np.asarray(model.state.orientation),
        np.asarray(model.state.angular_velocity),
        ref_final,
        f2s,
    )


# ─── The load-bearing gate ────────────────────────────────────────────


class TestOpenLoopHighOrder:
    """DOP853 open-loop round trip — the gate on the whole map.

    Bounds sit ~4 orders of magnitude above the measured residuals, which are
    at the 1e-14 … 1e-11 level: integrator round-off over a 5 s horizon and
    nothing else.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_hover(self, n):
        e = run_high_order(n, "hover")
        for key, value in e.items():
            assert value < 1e-9, f"n={n} hover {key} = {value:.3e}"

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_lissajous(self, n):
        e = run_high_order(n, "lissajous")
        assert e["payload_pos"] < TOL_ROUNDTRIP, f"n={n} payload pos {e['payload_pos']:.3e} m"
        assert e["payload_vel"] < TOL_ROUNDTRIP, f"n={n} payload vel {e['payload_vel']:.3e}"
        assert e["quad_pos"] < TOL_ROUNDTRIP, f"n={n} quad pos {e['quad_pos']:.3e} m"
        assert e["link_q"] < TOL_ROUNDTRIP, f"n={n} link q {e['link_q']:.3e}"
        assert e["link_omega"] < TOL_ROUNDTRIP, f"n={n} link ω {e['link_omega']:.3e}"
        assert e["attitude"] < 1e-9, f"n={n} attitude {e['attitude']:.3e}"
        assert e["body_omega"] < 1e-9, f"n={n} body Ω {e['body_omega']:.3e}"

    @pytest.mark.parametrize("n", [2, 4])
    def test_heterogeneous_links(self, n):
        """Unequal link masses *and* lengths — the case a uniform chain would
        hide a per-index indexing error behind."""
        e = run_high_order(
            n,
            "lissajous",
            mass=0.85,
            link_masses=np.linspace(0.04, 0.34, n),
            link_lengths=np.linspace(0.12, 0.36, n),
        )
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n={n} heterogeneous {key} = {value:.3e}"

    def test_a_larger_chain(self):
        """``n = 7``: jet order 18. Well past the ZOH-Euler stability limit for
        this chain, which is exactly why it runs on DOP853."""
        e = run_high_order(7, "lissajous")
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n=7 {key} = {value:.3e}"


class TestOpenLoopZoh:
    """The shipped ZOH-Euler pipeline, including actuator clipping.

    Restricted to the link counts where the chain is not stiff for a 1.25 ms
    forward-Euler substep (``stability_growth_rate() ≤ 0.45``). The residual
    here is first-order integrator error, ~4 orders larger than the DOP853
    run — which is the point: it is the difference between the two that shows
    the DOP853 residual to be round-off rather than a formula defect.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_hover_is_stationary(self, n):
        e = run_zoh(n, "hover")
        for key, value in e.items():
            assert value < 1e-9, f"n={n} hover {key} = {value:.3e}"

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_lissajous(self, n):
        model = QuadrotorFlexCableBase(num_links=n)
        assert model.stability_growth_rate() < 0.5, "chain chosen outside the stable ZOH regime"
        e = run_zoh(n, "lissajous")
        for key, value in e.items():
            assert value < 5e-2, f"n={n} lissajous {key} = {value:.3e}"

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_zoh_error_shrinks_with_the_step(self, n):
        """First-order convergence: a 4x smaller step gives a 4x smaller
        residual (the measured ratio is 0.250 at every ``n`` tried).

        This is what identifies the ZOH residual as integrator error. A
        flat-map formula error would not move with the step size at all.
        """
        coarse = run_zoh(n, "lissajous", dt=1.0 / 200.0)
        fine = run_zoh(n, "lissajous", dt=1.0 / 800.0)
        assert fine["payload_pos"] < 0.27 * coarse["payload_pos"]
        assert fine["link_q"] < 0.27 * coarse["link_q"]


# ─── Diagnostics — run with `pytest -s` to print the numbers ──────────


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("trajectory", ["hover", "lissajous"])
def test_print_errors_dop853(n, trajectory):
    e = run_high_order(n, trajectory)
    body = "".join(f"\n    {key:12s}: {value:.3e}" for key, value in e.items())
    print(f"\n  [n={n}, {trajectory}, DOP853 rtol=atol=1e-12] errors @ t={TF:.1f}s:{body}")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_print_errors_zoh(n):
    e = run_zoh(n, "lissajous")
    body = "".join(f"\n    {key:12s}: {value:.3e}" for key, value in e.items())
    print(f"\n  [n={n}, lissajous, ZOH-Euler] errors @ t={TF_ZOH:.1f}s:{body}")
