"""Open-loop feedforward round trip for the M6 flat-to-state map.

The flat-derived per-vehicle ``(f_k, M_k)`` is fed straight into the Step-12
model (:class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`) with **no
controller at all**. If the map is exact, the model's trajectory reproduces the
flat outputs; anything beyond the integrator's own residual is a formula bug.

This is the gate on the whole map, and it is the only check here that reaches
the recovered moment by a route independent of how the map computed it. The map
builds ``M = J Ω̇ + Ω × J Ω``; the model inverts exactly that expression to get
``Ω̇``, so comparing the two would be tautological — the residual would be zero
for *any* ``M``. What is not tautological is whether the ``Ω̇`` the map wants is
the one that keeps ``R(t)`` pointing the thrust where the *translational*
recursion needs it: the round trip closes only if the attitude, the thrust
magnitude, the link geometry and the hose's mass matrix all agree, and the
translational half of the state is compared against flat outputs the model never
saw.

Two integrators, for two different jobs:

* **DOP853** (adaptive 8th-order RK, ``rtol = atol = 1e-12``) driving the
  model's own :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.accelerations`
  — the whole right-hand side of the ODE — with the feedforward re-evaluated at
  the exact current time. This drives integrator error to round-off, so the
  residual *is* the formula error. It is also the only integrator usable past
  the ZOH-Euler envelope, which Step 12 put at ``n ≤ 4`` at any ``nQ``.

* **ZOH-Euler** — the model's shipped ``step()`` under the ``WRENCH`` input
  type, feedforward held constant across the 5 ms outer step, exercising the
  real input pipeline including actuator clipping. Kept inside the envelope.

The state is laid out for DOP853 as

    y = [x_0 (3), v_0 (3), {q_i (3), ω_i (3)}_{i<n}, {quat_wxyz (4), Ω (3)}_{k<nQ}]

i.e. ``6 + 6n + 7 n_Q`` doubles, with ``‖q_i‖ = 1``, ``ω_i · q_i = 0`` and each
quaternion re-normalised at every RHS call to absorb round-off.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as _ScipyRot

from udaan.flatness import Jet, MultiQuadHose, MultiQuadHoseFlats
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_hose.base import MultiQuadHoseBase

TF = 5.0
TF_ZOH = 3.0

# The DOP853 round trip measures 1e-15 … 1e-11 over 5 s: integrator round-off.
# ``tests.conftest.TOL_INTEGRATION`` (1e-6) would leave five orders of slack, so
# this file carries its own bound, well above the worst residual observed.
TOL_ROUNDTRIP = 1e-8

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Flat-output builders ─────────────────────────────────────────────


def default_flat_tensions(f2s: MultiQuadHose, pull=3.0, spread=1.0) -> list[np.ndarray]:
    """Flat tension vectors that hold the hose taut and spread sideways.

    Choosing the tensions directly rather than the thrusts is the point of
    these coordinates, and it is also what keeps the *thrusts* bounded. Each
    vehicle's thrust is
    :math:`u_{a} = \\bar m_a g e_3 + \\tau_{a^-} - \\tau_a`, and between two
    consecutive vehicles :math:`\\tau` grows only by the weight of the hose
    between them — so pinning every flat tension to the same downward pull
    ``-pull e₃`` makes each interior vehicle carry its own segment and nothing
    else, however long the hose is. Spreading the horizontal component apart
    from vehicle to vehicle is what opens the catenary out instead of stacking
    the airframes vertically; it also keeps ``‖𝛕_i‖`` bounded away from zero
    where the vertical component changes sign, so every link stays taut.
    """
    return [np.array([spread * (1.0 + 0.3 * k), 0.0, -pull]) for k in range(f2s.num_quadrotors - 1)]


def _sine_rows(t: float, order: int, amps, omegas, phases, offset) -> np.ndarray:
    """Closed-form derivative rows of ``x_j(t) = offset_j + A_j cos(ω_j t + φ_j)``.

    Exact at any order, which is what makes ``2n + 4`` reachable without
    numerical differentiation — at ``n = 4`` that is the 12th derivative.
    """
    data = np.zeros((order + 1, 3))
    for k in range(order + 1):
        for j in range(3):
            data[k, j] = (
                amps[j] * omegas[j] ** k * np.cos(omegas[j] * t + phases[j] + k * np.pi / 2)
            )
    data[0] += np.asarray(offset, dtype=float)
    return data


def _constant_tension_jets(f2s: MultiQuadHose, values) -> list[Jet]:
    jets = []
    for value, order in zip(values, f2s.tension_jet_orders, strict=True):
        rows = np.zeros((order + 1, 3))
        rows[0] = value
        jets.append(Jet(rows))
    return jets


def _yaw_jets(f2s: MultiQuadHose, t: float, moving: bool) -> list[Jet]:
    if not moving:
        return [Jet(np.zeros(3)) for _ in range(f2s.num_quadrotors)]
    out = []
    for k in range(f2s.num_quadrotors):
        w = 0.5 + 0.13 * k
        a = 0.18
        out.append(
            Jet(np.array([a * np.sin(w * t), a * w * np.cos(w * t), -a * w**2 * np.sin(w * t)]))
        )
    return out


def hover_flats(f2s: MultiQuadHose):
    """A genuine static equilibrium: the root sits still, the hose hangs in the
    catenary its own flat tensions describe, every yaw is zero."""
    values = default_flat_tensions(f2s)

    def builder(t: float) -> MultiQuadHoseFlats:
        return MultiQuadHoseFlats(
            x_0=Jet(np.zeros((f2s.jet_order + 1, 3))),
            tensions=_constant_tension_jets(f2s, values),
            psi=_yaw_jets(f2s, t, moving=False),
        )

    return builder


def lissajous_flats(f2s: MultiQuadHose):
    """A dynamic 3-D root trajectory with moving yaws and *constant* tensions.

    Constant flat tensions are not a static hose: the link forces below each
    vehicle still vary, because they pick up ``\\bar m_j \\ddot x_j`` at every
    vertex, so the whole geometry breathes. Amplitudes are chosen to keep every
    link comfortably taut and every thrust inside the model's 20 N limit, which
    the ZOH path enforces.
    """
    values = default_flat_tensions(f2s)

    def builder(t: float) -> MultiQuadHoseFlats:
        return MultiQuadHoseFlats(
            x_0=Jet(
                _sine_rows(
                    t,
                    f2s.jet_order,
                    [0.25, 0.25, 0.10],
                    [1.1, 1.1, 0.8],
                    [0.0, -np.pi / 2, 0.4],
                    [0.0, 0.0, 0.0],
                )
            ),
            tensions=_constant_tension_jets(f2s, values),
            psi=_yaw_jets(f2s, t, moving=True),
        )

    return builder


def breathing_flats(f2s: MultiQuadHose):
    """Moving root *and* time-varying flat tensions — the hose changes shape.

    The tension jets are exact sinusoids about the static values, so this is
    the case in which the flat tensions carry non-zero derivatives of their
    own and the ``link_forces`` depth contract actually bites.
    """
    values = default_flat_tensions(f2s)

    def builder(t: float) -> MultiQuadHoseFlats:
        jets = []
        for k, (value, order) in enumerate(zip(values, f2s.tension_jet_orders, strict=True)):
            w = 0.6 + 0.11 * k
            amp = 0.10 * float(np.linalg.norm(value))
            rows = np.zeros((order + 1, 3))
            rows[0] = value
            for d in range(order + 1):
                rows[d, 0] += amp * w**d * np.cos(w * t + d * np.pi / 2)
            jets.append(Jet(rows))
        return MultiQuadHoseFlats(
            x_0=Jet(
                _sine_rows(
                    t,
                    f2s.jet_order,
                    [0.18, 0.18, 0.08],
                    [0.9, 0.9, 0.7],
                    [0.0, -np.pi / 2, 0.2],
                    [0.0, 0.0, 0.0],
                )
            ),
            tensions=jets,
            psi=_yaw_jets(f2s, t, moving=True),
        )

    return builder


TRAJECTORIES = {
    "hover": hover_flats,
    "lissajous": lissajous_flats,
    "breathing": breathing_flats,
}


# ─── State packing ────────────────────────────────────────────────────


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    xyzw = _ScipyRot.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = quat_wxyz / np.linalg.norm(quat_wxyz)
    return _ScipyRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _pack(ref, n: int, nQ: int) -> np.ndarray:
    parts = [np.asarray(ref.root_position), np.asarray(ref.root_velocity)]
    for i in range(n):
        parts.append(np.asarray(ref.link_attitudes[i]))
        parts.append(np.asarray(ref.link_angular_velocities[i]))
    for k in range(nQ):
        parts.append(_quat_wxyz(np.asarray(ref.orientations[k])))
        parts.append(np.asarray(ref.angular_velocities[k]))
    return np.concatenate(parts)


def _unpack(y: np.ndarray, n: int, nQ: int):
    x_0, v_0 = y[0:3], y[3:6]
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
    quats, Omegas = [], []
    for _ in range(nQ):
        quats.append(y[idx : idx + 4])
        Omegas.append(y[idx + 4 : idx + 7])
        idx += 7
    return x_0, v_0, directions, rates, quats, Omegas


def _quat_dot(quat: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """Hamilton, scalar-first: ``q̇ = ½ q ⊗ [0, Ω]``."""
    q = quat / np.linalg.norm(quat)
    w, x, y_, z = q
    p, r, s = Omega
    return 0.5 * np.array(
        [
            -x * p - y_ * r - z * s,
            w * p - z * r + y_ * s,
            z * p + w * r - x * s,
            -y_ * p + x * r + w * s,
        ]
    )


# ─── DOP853 harness ───────────────────────────────────────────────────


def _write_state(model, x_0, v_0, directions, rates, quats, Omegas) -> None:
    model.state.root_position = x_0
    model.state.root_velocity = v_0
    model.state.link_attitudes = [S2(q) for q in directions]
    model.state.link_angular_velocities = [TS2(w) for w in rates]
    model.state.orientations = [SO3(_matrix(q)) for q in quats]
    model.state.angular_velocities = [TSO3(w) for w in Omegas]


def _rhs(t, y, model: MultiQuadHoseBase, f2s: MultiQuadHose, flats_of_t):
    n, nQ = model.num_links, model.num_quadrotors
    x_0, v_0, directions, rates, quats, Omegas = _unpack(y, n, nQ)
    _write_state(model, x_0, v_0, directions, rates, quats, Omegas)

    _, inputs = f2s.recover(flats_of_t(t))
    acc = model.accelerations(inputs.thrusts, inputs.moments)

    out = [v_0, acc["root_accel"]]
    for i in range(n):
        out.append(np.cross(rates[i], directions[i]))
        out.append(acc["link_ang_accels"][i])
    for k in range(nQ):
        out.append(_quat_dot(quats[k], Omegas[k]))
        out.append(acc["quad_ang_accels"][k])
    return np.concatenate(out)


def _errors(x_0, v_0, directions, rates, Rs, Omegas, ref, f2s) -> dict:
    n, nQ = f2s.num_links, f2s.num_quadrotors
    vertices = np.zeros((n + 1, 3))
    vertices[0] = x_0
    for i in range(n):
        vertices[i + 1] = vertices[i] + f2s.link_lengths[i] * directions[i]
    return {
        "root_pos": float(np.linalg.norm(x_0 - ref.root_position)),
        "root_vel": float(np.linalg.norm(v_0 - ref.root_velocity)),
        "vertex_pos": float(np.max(np.linalg.norm(vertices - ref.vertex_positions, axis=1))),
        "link_q": max(
            float(np.linalg.norm(directions[i] - np.asarray(ref.link_attitudes[i])))
            for i in range(n)
        ),
        "link_omega": max(
            float(np.linalg.norm(rates[i] - np.asarray(ref.link_angular_velocities[i])))
            for i in range(n)
        ),
        "attitude": max(
            abs(float(SO3(np.asarray(ref.orientations[k])).config_error(SO3(Rs[k]))))
            for k in range(nQ)
        ),
        "body_omega": max(
            float(np.linalg.norm(Omegas[k] - np.asarray(ref.angular_velocities[k])))
            for k in range(nQ)
        ),
    }


def run_high_order(n, attachments, trajectory, tf=TF, **model_kwargs) -> dict:
    """Seed from the flat reference at t = 0, integrate open loop, compare at tf."""
    model = MultiQuadHoseBase(num_links=n, attachments=attachments, **model_kwargs)
    f2s = MultiQuadHose.from_model(model)
    flats_of_t = TRAJECTORIES[trajectory](f2s)

    ref0, _ = f2s.recover(flats_of_t(0.0))
    sol = solve_ivp(
        _rhs,
        (0.0, tf),
        _pack(ref0, n, model.num_quadrotors),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        args=(model, f2s, flats_of_t),
    )
    assert sol.success, f"DOP853 failed: {sol.message}"

    x_0, v_0, directions, rates, quats, Omegas = _unpack(sol.y[:, -1], n, model.num_quadrotors)
    ref_final, _ = f2s.recover(flats_of_t(tf))
    return _errors(x_0, v_0, directions, rates, [_matrix(q) for q in quats], Omegas, ref_final, f2s)


# ─── ZOH-Euler harness ────────────────────────────────────────────────


def run_zoh(n, attachments, trajectory, tf=TF_ZOH, **model_kwargs) -> dict:
    model = MultiQuadHoseBase(num_links=n, attachments=attachments, input="wrench", **model_kwargs)
    f2s = MultiQuadHose.from_model(model)
    flats_of_t = TRAJECTORIES[trajectory](f2s)

    ref0, _ = f2s.recover(flats_of_t(0.0))
    model.reset(
        root_position=np.asarray(ref0.root_position).copy(),
        root_velocity=np.asarray(ref0.root_velocity).copy(),
        link_attitudes=[np.asarray(q).copy() for q in ref0.link_attitudes],
        link_angular_velocities=[np.asarray(w).copy() for w in ref0.link_angular_velocities],
        orientations=[np.asarray(R).copy() for R in ref0.orientations],
        angular_velocities=[np.asarray(w).copy() for w in ref0.angular_velocities],
    )

    while model.t < tf - 1e-9:
        _, inputs = f2s.recover(flats_of_t(model.t))
        model.step(inputs.as_wrench())

    ref_final, _ = f2s.recover(flats_of_t(tf))
    return _errors(
        np.asarray(model.state.root_position),
        np.asarray(model.state.root_velocity),
        [np.asarray(q) for q in model.state.link_attitudes],
        [np.asarray(w) for w in model.state.link_angular_velocities],
        [np.asarray(R) for R in model.state.orientations],
        [np.asarray(w) for w in model.state.angular_velocities],
        ref_final,
        f2s,
    )


# ─── The load-bearing gate ────────────────────────────────────────────

# (n, attachments) combinations: both ends, one vehicle only, an *interior*
# vehicle (the case the legacy reference silently fails to handle), a vehicle
# at every vertex, and a hose whose root carries no vehicle at all.
CONFIGURATIONS = [
    (1, (0, 1)),
    (2, (2,)),
    (3, (0, 3)),
    (4, (0, 2, 4)),
    (4, (2, 4)),
    (4, (0, 1, 2, 3, 4)),
    (5, (0, 3, 5)),
]


class TestOpenLoopHighOrder:
    """DOP853 open-loop round trip — the gate on the whole map."""

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_hover_is_stationary(self, n, attachments):
        e = run_high_order(n, attachments, "hover")
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n={n} {attachments} hover {key} = {value:.3e}"

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_lissajous(self, n, attachments):
        e = run_high_order(n, attachments, "lissajous")
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n={n} {attachments} lissajous {key} = {value:.3e}"

    @pytest.mark.parametrize(("n", "attachments"), [(2, (0, 2)), (4, (0, 2, 4))])
    def test_breathing_tensions(self, n, attachments):
        """The flat tensions carry derivatives of their own."""
        e = run_high_order(n, attachments, "breathing")
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n={n} {attachments} breathing {key} = {value:.3e}"

    @pytest.mark.parametrize(("n", "attachments"), [(3, (0, 3)), (4, (0, 2, 4))])
    def test_heterogeneous_parameters(self, n, attachments):
        """Unequal vertex masses, link lengths, airframe masses *and* inertias —
        the case a uniform hose would hide a per-index indexing error behind."""
        nQ = len(attachments)
        inertias = np.stack(
            [
                np.diag([1.43e-2 + 3e-3 * k, 1.61e-2 - 2e-3 * k, 2.89e-2 + 1e-3 * k])
                for k in range(nQ)
            ]
        )
        e = run_high_order(
            n,
            attachments,
            "lissajous",
            vertex_masses=np.linspace(0.04, 0.19, n + 1),
            link_lengths=np.linspace(0.30, 0.62, n),
            masses=np.linspace(0.75, 0.95, nQ),
            inertias=inertias,
        )
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n={n} {attachments} heterogeneous {key} = {value:.3e}"

    def test_a_longer_hose(self):
        """``n = 8``: jet order 20. Far past the ZOH-Euler envelope Step 12 put
        at ``n ≤ 4``, which is exactly why it runs on DOP853."""
        e = run_high_order(8, (0, 4, 8), "lissajous")
        for key, value in e.items():
            assert value < TOL_ROUNDTRIP, f"n=8 {key} = {value:.3e}"


class TestOpenLoopZoh:
    """The shipped ZOH-Euler pipeline, including actuator clipping.

    Kept inside Step 12's ``n ≤ 4`` envelope. The residual here is first-order
    integrator error, orders larger than the DOP853 run — which is the point:
    the gap between the two is what shows the DOP853 residual to be round-off
    rather than a formula defect.
    """

    ZOH_CONFIGS = [(2, (0, 2)), (3, (0, 3)), (4, (0, 2, 4))]

    @pytest.mark.parametrize(("n", "attachments"), ZOH_CONFIGS)
    def test_hover_is_stationary(self, n, attachments):
        e = run_zoh(n, attachments, "hover")
        for key, value in e.items():
            assert value < 1e-9, f"n={n} {attachments} hover {key} = {value:.3e}"

    @pytest.mark.parametrize(("n", "attachments"), ZOH_CONFIGS)
    def test_lissajous(self, n, attachments):
        model = MultiQuadHoseBase(num_links=n, attachments=attachments)
        assert model.stability_growth_rate() < 1.0, "configuration outside the ZOH envelope"
        e = run_zoh(n, attachments, "lissajous")
        for key, value in e.items():
            assert value < 5e-2, f"n={n} {attachments} lissajous {key} = {value:.3e}"

    @pytest.mark.parametrize(("n", "attachments"), ZOH_CONFIGS)
    @pytest.mark.slow
    def test_zoh_error_shrinks_with_the_step(self, n, attachments):
        """First-order convergence: a 4x smaller step gives a ~4x smaller
        residual. This is what identifies the ZOH residual as integrator error —
        a flat-map formula error would not move with the step size at all."""
        coarse = run_zoh(n, attachments, "lissajous", dt=1.0 / 200.0)
        fine = run_zoh(n, attachments, "lissajous", dt=1.0 / 800.0)
        assert fine["root_pos"] < 0.30 * coarse["root_pos"]
        assert fine["link_q"] < 0.30 * coarse["link_q"]


# ─── Diagnostics — run with `pytest -s` to print the numbers ──────────


@pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
@pytest.mark.parametrize("trajectory", ["hover", "lissajous"])
def test_print_errors_dop853(n, attachments, trajectory):
    e = run_high_order(n, attachments, trajectory)
    body = "".join(f"\n    {key:12s}: {value:.3e}" for key, value in e.items())
    print(f"\n  [n={n}, att={attachments}, {trajectory}, DOP853] errors @ t={TF:.1f}s:{body}")


@pytest.mark.parametrize(("n", "attachments"), TestOpenLoopZoh.ZOH_CONFIGS)
def test_print_errors_zoh(n, attachments):
    e = run_zoh(n, attachments, "lissajous")
    body = "".join(f"\n    {key:12s}: {value:.3e}" for key, value in e.items())
    print(f"\n  [n={n}, att={attachments}, lissajous, ZOH] errors @ t={TF_ZOH:.1f}s:{body}")
