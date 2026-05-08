"""Unit tests for :class:`udaan.utils.flatness.QuadrotorCsPayload` — the
quadrotor + cable-suspended-point-mass-payload differential-flatness map."""

from __future__ import annotations

import numpy as np
import pytest

from udaan.core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from udaan.models.quadrotor_cspayload.base import QuadrotorCsPayloadBase
from udaan.utils.flatness import (
    Jet,
    QuadrotorCsPayload,
    QuadrotorCsPayloadFlats,
)

# ─── Helpers ──────────────────────────────────────────────────────────


def _zero_flats() -> QuadrotorCsPayloadFlats:
    """Hover flats: payload position and yaw jets all zero."""
    return QuadrotorCsPayloadFlats(
        x_L=Jet.zeros(order=6, dim=3),
        psi=Jet.zeros(order=2, dim=1),
    )


def _circle_flats(
    t: float, omega: float = 0.5, radius: float = 0.4, height: float = 1.0
) -> QuadrotorCsPayloadFlats:
    """Analytic flats for a horizontal payload circle at constant altitude
    with the quadrotor yaw tracking the heading angle.

    x_L(t) = [r cos(ωt), r sin(ωt), h],   ψ(t) = ω t

    The radius is kept small so the swing angle stays well within the
    taut-cable / non-singular regime (T > 0 throughout).
    """
    ct, st = np.cos(omega * t), np.sin(omega * t)
    rows = []
    rows.append(np.array([radius * ct, radius * st, height]))
    rows.append(np.array([-radius * omega * st, radius * omega * ct, 0.0]))
    rows.append(np.array([-radius * omega**2 * ct, -radius * omega**2 * st, 0.0]))
    rows.append(np.array([radius * omega**3 * st, -radius * omega**3 * ct, 0.0]))
    rows.append(np.array([radius * omega**4 * ct, radius * omega**4 * st, 0.0]))
    rows.append(np.array([-radius * omega**5 * st, radius * omega**5 * ct, 0.0]))
    rows.append(np.array([-radius * omega**6 * ct, -radius * omega**6 * st, 0.0]))
    x_jet = Jet.from_list(rows)
    psi_jet = Jet(np.array([omega * t, omega, 0.0]))
    return QuadrotorCsPayloadFlats(x_L=x_jet, psi=psi_jet)


# ─── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_build_from_physical_params(self):
        f2s = QuadrotorCsPayload(
            mass_q=DEFAULT_QUAD_MASS,
            mass_l=DEFAULT_PAYLOAD_MASS,
            inertia=DEFAULT_QUAD_INERTIA,
            cable_length=DEFAULT_CABLE_LENGTH,
        )
        assert f2s.mass_q == pytest.approx(DEFAULT_QUAD_MASS)
        assert f2s.mass_l == pytest.approx(DEFAULT_PAYLOAD_MASS)
        assert f2s.cable_length == pytest.approx(DEFAULT_CABLE_LENGTH)
        np.testing.assert_allclose(f2s.inertia, DEFAULT_QUAD_INERTIA)

    def test_from_model_reads_model_params(self):
        model = QuadrotorCsPayloadBase()
        f2s = QuadrotorCsPayload.from_model(model)
        assert f2s.mass_q == pytest.approx(model.mass)
        assert f2s.mass_l == pytest.approx(model.payload_mass)
        assert f2s.cable_length == pytest.approx(model.cable_length)
        np.testing.assert_allclose(f2s.inertia, model.inertia)

    def test_inertia_shape_validated(self):
        with pytest.raises(ValueError, match="inertia must be 3×3"):
            QuadrotorCsPayload(inertia=np.zeros((2, 2)))

    def test_cable_length_positive(self):
        with pytest.raises(ValueError, match="cable_length must be positive"):
            QuadrotorCsPayload(cable_length=0.0)
        with pytest.raises(ValueError, match="cable_length must be positive"):
            QuadrotorCsPayload(cable_length=-1.0)

    def test_class_attribute_triple_declared(self):
        assert QuadrotorCsPayload.Flats is QuadrotorCsPayloadFlats
        assert QuadrotorCsPayload.RefState.__name__ == "QuadrotorCsPayloadRefState"
        assert QuadrotorCsPayload.Inputs.__name__ == "QuadrotorCsPayloadInputs"


# ─── QuadrotorCsPayloadFlats validation ───────────────────────────────


class TestFlatsValidation:
    def test_valid_order_flats_construct(self):
        QuadrotorCsPayloadFlats(
            x_L=Jet.zeros(order=6, dim=3),
            psi=Jet.zeros(order=2, dim=1),
        )

    def test_higher_order_flats_also_accepted(self):
        QuadrotorCsPayloadFlats(
            x_L=Jet.zeros(order=8, dim=3),
            psi=Jet.zeros(order=4, dim=1),
        )

    def test_x_wrong_dim_rejected(self):
        with pytest.raises(ValueError, match="x_L must be a 3-D jet"):
            QuadrotorCsPayloadFlats(
                x_L=Jet.zeros(order=6, dim=2),
                psi=Jet.zeros(order=2, dim=1),
            )

    def test_x_insufficient_order_rejected(self):
        with pytest.raises(ValueError, match=r"x_L.order must be >= 6"):
            QuadrotorCsPayloadFlats(
                x_L=Jet.zeros(order=5, dim=3),
                psi=Jet.zeros(order=2, dim=1),
            )

    def test_psi_not_scalar_rejected(self):
        with pytest.raises(ValueError, match="psi must be a scalar jet"):
            QuadrotorCsPayloadFlats(
                x_L=Jet.zeros(order=6, dim=3),
                psi=Jet.zeros(order=2, dim=3),
            )

    def test_psi_insufficient_order_rejected(self):
        with pytest.raises(ValueError, match=r"psi.order must be >= 2"):
            QuadrotorCsPayloadFlats(
                x_L=Jet.zeros(order=6, dim=3),
                psi=Jet.zeros(order=1, dim=1),
            )


# ─── Hover recovery ───────────────────────────────────────────────────


class TestHover:
    @pytest.fixture
    def f2s(self):
        return QuadrotorCsPayload(
            mass_q=DEFAULT_QUAD_MASS,
            mass_l=DEFAULT_PAYLOAD_MASS,
            inertia=DEFAULT_QUAD_INERTIA,
            cable_length=DEFAULT_CABLE_LENGTH,
        )

    @pytest.fixture
    def ref_inputs(self, f2s):
        return f2s.recover(_zero_flats())

    def test_hover_thrust_equals_total_weight(self, ref_inputs):
        _, inputs = ref_inputs
        expected = (DEFAULT_QUAD_MASS + DEFAULT_PAYLOAD_MASS) * GRAVITY
        assert inputs.thrust == pytest.approx(expected, abs=1e-10)

    def test_hover_tension_equals_payload_weight(self, ref_inputs):
        _, inputs = ref_inputs
        expected = DEFAULT_PAYLOAD_MASS * GRAVITY
        assert inputs.tension == pytest.approx(expected, abs=1e-10)

    def test_hover_moment_is_zero(self, ref_inputs):
        _, inputs = ref_inputs
        np.testing.assert_allclose(inputs.moment, np.zeros(3), atol=1e-10)

    def test_hover_cable_points_down(self, ref_inputs):
        ref, _ = ref_inputs
        np.testing.assert_allclose(
            np.asarray(ref.cable_attitude), np.array([0.0, 0.0, -1.0]), atol=1e-10
        )

    def test_hover_quadrotor_above_payload(self, ref_inputs):
        # x_L = 0, q = -e_3, so x_Q = x_L - ℓ q = +ℓ e_3.
        ref, _ = ref_inputs
        np.testing.assert_allclose(
            ref.position, np.array([0.0, 0.0, DEFAULT_CABLE_LENGTH]), atol=1e-10
        )

    def test_hover_orientation_is_identity(self, ref_inputs):
        ref, _ = ref_inputs
        R = np.asarray(ref.orientation)
        err = np.linalg.norm(R - np.eye(3), ord="fro")
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_hover_cable_and_body_rates_zero(self, ref_inputs):
        ref, _ = ref_inputs
        np.testing.assert_allclose(ref.cable_angular_velocity, np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(ref.cable_angular_acceleration, np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(np.asarray(ref.angular_velocity), np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(ref.angular_acceleration, np.zeros(3), atol=1e-10)


# ─── Pure yaw-rate recovery ───────────────────────────────────────────


class TestPureYawRate:
    """Hover with yaw rate — payload stays still, quadrotor spins about world e_3."""

    @pytest.fixture
    def f2s(self):
        return QuadrotorCsPayload()

    @pytest.fixture
    def ref_inputs(self, f2s):
        x_jet = Jet.zeros(order=6, dim=3)
        psi_jet = Jet(np.array([0.0, 0.5, 0.0]))
        return f2s.recover(QuadrotorCsPayloadFlats(x_L=x_jet, psi=psi_jet))

    def test_angular_velocity_matches_yaw_rate(self, ref_inputs):
        ref, _ = ref_inputs
        Omega = np.asarray(ref.angular_velocity)
        np.testing.assert_allclose(Omega[:2], np.zeros(2), atol=1e-10)
        assert Omega[2] == pytest.approx(0.5, abs=1e-10)

    def test_orientation_at_t0_is_identity(self, ref_inputs):
        ref, _ = ref_inputs
        np.testing.assert_allclose(np.asarray(ref.orientation), np.eye(3), atol=1e-10)

    def test_cable_stays_vertical_under_pure_yaw(self, ref_inputs):
        # Yawing the quadrotor about e_3 with the payload at origin should
        # leave the cable pointing straight down — no swing.
        ref, _ = ref_inputs
        np.testing.assert_allclose(
            np.asarray(ref.cable_attitude), np.array([0.0, 0.0, -1.0]), atol=1e-10
        )
        np.testing.assert_allclose(ref.cable_angular_velocity, np.zeros(3), atol=1e-10)


# ─── Yaw with non-zero second derivative ──────────────────────────────


class TestYawWithAcceleration:
    @pytest.fixture
    def f2s(self):
        return QuadrotorCsPayload()

    @pytest.fixture
    def ref_inputs(self, f2s):
        x_jet = Jet.zeros(order=6, dim=3)
        psi_jet = Jet(np.array([0.0, 0.5, 0.3]))
        return f2s.recover(QuadrotorCsPayloadFlats(x_L=x_jet, psi=psi_jet))

    def test_angular_acceleration_matches_yaw_accel(self, ref_inputs):
        ref, _ = ref_inputs
        dOmega = np.asarray(ref.angular_acceleration)
        np.testing.assert_allclose(dOmega[:2], np.zeros(2), atol=1e-10)
        assert dOmega[2] == pytest.approx(0.3, abs=1e-10)

    def test_moment_yaw_axis(self, ref_inputs):
        _, inputs = ref_inputs
        np.testing.assert_allclose(inputs.moment[:2], np.zeros(2), atol=1e-10)
        Jzz = DEFAULT_QUAD_INERTIA[2, 2]
        assert inputs.moment[2] == pytest.approx(Jzz * 0.3, abs=1e-10)


# ─── Singularities ────────────────────────────────────────────────────


class TestSingularities:
    def test_payload_free_fall_raises(self):
        f2s = QuadrotorCsPayload()
        # ẍ_L + g e_3 = 0 — payload in free fall, cable goes slack.
        x_data = np.zeros((7, 3))
        x_data[2] = np.array([0.0, 0.0, -GRAVITY])
        with pytest.raises(ValueError, match="payload is in free fall"):
            f2s.recover(
                QuadrotorCsPayloadFlats(
                    x_L=Jet(x_data),
                    psi=Jet.zeros(order=2, dim=1),
                )
            )


# ─── Cable kinematic consistency ──────────────────────────────────────


class TestCableConsistency:
    """Recovered (q, ω, q̇) should satisfy the constraints
    ‖q‖ = 1, ω · q = 0, and q̇ = ω × q at every recovered point."""

    @pytest.fixture
    def f2s(self):
        return QuadrotorCsPayload()

    @pytest.mark.parametrize("t", [0.0, 0.13, 0.79, 1.5])
    def test_q_unit_norm(self, f2s, t):
        ref, _ = f2s.recover(_circle_flats(t))
        q = np.asarray(ref.cable_attitude)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("t", [0.0, 0.13, 0.79, 1.5])
    def test_omega_perpendicular_to_q(self, f2s, t):
        ref, _ = f2s.recover(_circle_flats(t))
        q = np.asarray(ref.cable_attitude)
        omega = np.asarray(ref.cable_angular_velocity)
        assert q.dot(omega) == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("t", [0.0, 0.13, 0.79, 1.5])
    def test_quadrotor_position_matches_payload_minus_cable(self, f2s, t):
        ref, _ = f2s.recover(_circle_flats(t))
        L = f2s.cable_length
        expected = ref.payload_position - L * np.asarray(ref.cable_attitude)
        np.testing.assert_allclose(ref.position, expected, atol=1e-12)


# ─── Round-trip with the model ────────────────────────────────────────


class TestRoundTrip:
    """Recover (ref, inputs) from analytic flats, step the cspayload model
    with the feedforward wrench, and confirm tracking against the analytic
    reference at the next sample."""

    @pytest.fixture
    def f2s(self):
        return QuadrotorCsPayload()

    @pytest.fixture
    def model(self):
        return QuadrotorCsPayloadBase(input="wrench")

    def test_one_step_tracks_analytic_reference(self, f2s, model):
        flats0 = _circle_flats(0.0)
        ref0, u0 = f2s.recover(flats0)
        state0 = ref0.as_state()
        model.reset(
            payload_position=state0.payload_position,
            payload_velocity=state0.payload_velocity,
            cable_attitude=state0.cable_attitude,
            cable_angular_velocity=state0.cable_angular_velocity,
            orientation=state0.orientation,
            angular_velocity=state0.angular_velocity,
        )

        u = np.array([u0.thrust, *u0.moment])
        model.step(u)

        flats1 = _circle_flats(model.t)
        ref1, _ = f2s.recover(flats1)

        # ZOH over 5 ms, smooth trajectory: tight tracking.
        np.testing.assert_allclose(model.state.payload_position, ref1.payload_position, atol=1e-3)
        np.testing.assert_allclose(model.state.payload_velocity, ref1.payload_velocity, atol=1e-3)

        R_sim = np.asarray(model.state.orientation)
        R_ref = np.asarray(ref1.orientation)
        att_err = np.linalg.norm(R_sim - R_ref, ord="fro")
        assert att_err < 1e-2

    def test_two_steps_tracks_analytic_reference(self, f2s, model):
        flats0 = _circle_flats(0.0)
        ref0, _ = f2s.recover(flats0)
        state0 = ref0.as_state()
        model.reset(
            payload_position=state0.payload_position,
            payload_velocity=state0.payload_velocity,
            cable_attitude=state0.cable_attitude,
            cable_angular_velocity=state0.cable_angular_velocity,
            orientation=state0.orientation,
            angular_velocity=state0.angular_velocity,
        )

        for _ in range(2):
            flats_t = _circle_flats(model.t)
            _, u_t = f2s.recover(flats_t)
            model.step(np.array([u_t.thrust, *u_t.moment]))

        flats_end = _circle_flats(model.t)
        ref_end, _ = f2s.recover(flats_end)
        np.testing.assert_allclose(
            model.state.payload_position, ref_end.payload_position, atol=1e-2
        )
        np.testing.assert_allclose(
            model.state.payload_velocity, ref_end.payload_velocity, atol=1e-2
        )
