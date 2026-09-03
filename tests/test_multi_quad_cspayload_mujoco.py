"""Tests for MultiQuadCsPayloadMujoco — MJCF wiring, state sync, station keeping."""

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_TIGHT
from udaan.core.defaults import DEFAULT_POS_KP, GRAVITY
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

pytestmark = pytest.mark.mujoco

# Every count goes through the same runtime MJCF generation path; the asset is
# cached per configuration hash. See tests/test_multi_quad_cspayload_xml.py for
# the generator itself.
QUAD_COUNTS = [2, 3, 4]


class TestMujocoSetup:
    """Model loading, index resolution and parameters read back from the MJCF."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_instantiate(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        assert mdl.num_quadrotors == nQ
        assert mdl._mjMdl.model.nu == 4 * nQ
        assert len(mdl.state.positions) == nQ

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_params_read_from_mjcf(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        assert mdl.masses.shape == (nQ,)
        assert np.all(mdl.masses > 0.0)
        assert mdl.inertias.shape == (nQ, 3, 3)
        assert mdl.payload_mass > 0.0
        assert mdl.cable_lengths.shape == (nQ,)
        assert np.all(mdl.cable_lengths > 0.0)

    def test_state_types(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        mdl.reset()
        for i in range(3):
            assert isinstance(mdl.state.orientations[i], SO3)
            assert isinstance(mdl.state.angular_velocities[i], TSO3)
            assert isinstance(mdl.state.cable_attitudes[i], S2)
            assert isinstance(mdl.state.cable_angular_velocities[i], TS2)

    def test_mass_setter_writes_into_mujoco(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        mdl.masses = [1.1, 1.2, 1.3]
        for i, body_id in enumerate(mdl._quad_body_id):
            assert mdl._mjMdl.model.body_mass[body_id] == pytest.approx(mdl.masses[i])
        mdl.payload_mass = 0.42
        assert mdl._mjMdl.model.body_mass[mdl._payload_body_id] == pytest.approx(0.42)


class TestMujocoStateSync:
    """_query_latest_state pulls the right quantities out of MuJoCo."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_reset_places_payload(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        target = np.array([-1.0, 2.0, 0.5])
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        assert mdl.t == 0.0
        for i in range(nQ):
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_cable_attitudes_are_unit_and_point_at_the_load(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for i in range(nQ):
            q = np.asarray(mdl.state.cable_attitudes[i])
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=TOL_TIGHT)
            direction = mdl.state.payload_position - mdl.state.positions[i]
            np.testing.assert_allclose(q, direction / np.linalg.norm(direction), atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_cable_omega_scales_with_a_longer_cable(self, k):
        """Same relative velocity, longer cable -> proportionally smaller omega.

        This is the structural cover for the ``1/length`` factor the retired
        model dropped. It compares two *different* cable lengths rather than
        recomputing ``q × Δv / l`` alongside the implementation, so it cannot
        be satisfied by any length-independent formula regardless of what the
        packaged MJCF happens to start at.

        (An earlier ``test_cable_omega_uses_the_inverse_length_factor`` was
        deleted: its ``expected`` restated the implementation's own expression,
        and its second assertion, ``l·(x/l) == x``, is an identity. It only
        discriminated because the MJCF then started slack at ~0.83 m — which
        it no longer does, so this test moves the quad itself instead.)
        """
        payload_vel = np.array([0.0, 0.0, -1.0])
        recovered = {}
        for length in (1.0, 2.0):
            mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
            mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
            # Push quad k out to `length` along its own cable direction.
            qk = np.asarray(mdl.state.cable_attitudes[k])
            adr = mdl._quad_qposadr[k]
            mdl._mjMdl.data.qpos[adr : adr + 3] = mdl.state.payload_position - length * qk
            i = mdl._payload_qveladr
            mdl._mjMdl.data.qvel[i : i + 3] = payload_vel
            mdl._mjMdl.data.qvel[mdl._quad_qveladr[k] : mdl._quad_qveladr[k] + 3] = 0.0
            mdl._query_latest_state()
            omega = np.asarray(mdl.state.cable_angular_velocities[k])
            # ω must stay tangent to the sphere whatever the length.
            assert abs(float(np.dot(np.asarray(mdl.state.cable_attitudes[k]), omega))) < 1e-12
            recovered[length] = omega.copy()

        np.testing.assert_allclose(recovered[2.0], 0.5 * recovered[1.0], atol=TOL_ANALYTIC)
        assert np.linalg.norm(recovered[1.0]) > 1e-3


def predicted_inward_drift(
    num_quadrotors: int,
    payload_mass: float,
    cable_length: float,
    formation_radius: float,
    kp_horizontal: float,
) -> float:
    """Steady-state horizontal drift of each quadrotor under the station-keeping PDs.

    :class:`~udaan.control.quadrotor.PositionPDController` emits
    ``u = -Kp e - Kd ė + m_ctrl g e₃`` with ``m_ctrl = m_q + m_L/nQ``, and at
    equilibrium the quadrotor balances that against its cable tension ``T``
    along ``q_i`` and its own weight::

        u + T q_i - m_q g e₃ = 0

    *Vertically*, with ``q_{i,z} = -dz/l``, the balance reads
    ``-Kp_z e_z + (m_L/nQ) g - T dz/l = 0``, which ``e_z = 0`` satisfies exactly
    at ``T = (m_L/nQ) g l/dz`` — precisely the payload share the controller
    already feed-forwards. So there is **no vertical sag at all**; the whole
    steady-state error is horizontal.

    *Horizontally*, ``Kp_x e = T r_eff/l = m_L g r_eff / (nQ dz)`` where
    ``r_eff`` is the quadrotor's current horizontal distance to the payload axis
    and ``dz = sqrt(l² - r_eff²)`` its height above the payload. The ring
    therefore contracts on itself — the quadrotors drift inward by ``e``, so
    ``r_eff = r₀ - e`` and the payload sinks to keep the cable taut — and the
    drift is the fixed point of::

        e = m_L g (r₀ - e) / (nQ Kp_x sqrt(l² - (r₀ - e)²))

    which this function reaches by iteration.
    """
    e = 0.0
    for _ in range(100):
        r_eff = formation_radius - e
        dz = np.sqrt(cable_length**2 - r_eff**2)
        e = payload_mass * GRAVITY * r_eff / (num_quadrotors * dz * kp_horizontal)
    return e


@pytest.mark.integration
class TestMujocoStationKeeping:
    """Closed-loop: the default per-quadrotor PDs hold the team steady."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_formation_is_held(self, nQ):
        """The team settles on the *predicted* equilibrium, not merely near the start.

        The formation spawns taut, so cable tension acts from t=0 and the
        position PDs (no integral action) settle with a steady-state error.
        That error is entirely horizontal and its size is predictable — see
        :func:`predicted_inward_drift` — so this asserts it **two-sided**:
        drifting too far fails, and so does not drifting at all, which is what
        losing cable tension would look like.

        (The earlier one-sided ``atol=0.15`` on ``|pos - target|`` passed for
        any drift in ``[0, 0.15]``, zero included, so it could no longer detect
        a slack formation. Its comment blamed the taut spawn for a "≈0.14 m"
        droop: wrong on both counts. Re-inserting the retired slack ``zL=1.34``
        gives the *same* 0.103 m drift — both spawns relax to the same
        equilibrium, only the payload's final ``z`` differs. The number moved
        because the masses did, from a hand-written asset's 0.99 kg quad /
        0.15 kg payload to the generator's ``DEFAULT_QUAD_MASS`` /
        ``DEFAULT_PAYLOAD_MASS``.)
        """
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        start = np.array([-1.0, 2.0, 0.5])
        mdl.simulate(tf=4.0, payload_position=start)

        assert mdl.t >= 4.0
        # Derived from the model's own parameters and the packaged gains — not
        # a measured constant pasted back in.
        expected_drift = predicted_inward_drift(
            nQ,
            mdl.payload_mass,
            float(mdl.cable_lengths[0]),
            mdl._formation_radius,
            float(DEFAULT_POS_KP[0]),
        )
        assert expected_drift > 0.01

        # The payload holds its horizontal station and everything settles.
        np.testing.assert_allclose(mdl.state.payload_position[:2], start[:2], atol=1e-3)
        assert np.linalg.norm(mdl.state.payload_velocity) < 0.01
        for i in range(nQ):
            assert np.linalg.norm(mdl.state.velocities[i]) < 0.01
            # No vertical sag: the payload-share feedforward cancels it exactly.
            assert mdl.state.positions[i][2] == pytest.approx(
                mdl._formation_targets[i][2], abs=1e-3
            )
            # The ring contracts by exactly the predicted amount.
            radius = np.linalg.norm(mdl.state.positions[i][:2] - mdl.state.payload_position[:2])
            assert mdl._formation_radius - radius == pytest.approx(expected_drift, rel=0.02)
