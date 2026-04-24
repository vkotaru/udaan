"""Unit tests for :class:`udaan.utils.flatness.Quadrotor` — the rigid-body
quadrotor differential-flatness map."""

from __future__ import annotations

import numpy as np
import pytest

from udaan.core.defaults import DEFAULT_QUAD_INERTIA, DEFAULT_QUAD_MASS, GRAVITY
from udaan.models.quadrotor.base import QuadrotorBase
from udaan.utils.flatness import Jet, Quadrotor, QuadrotorFlats

# ─── Helpers ──────────────────────────────────────────────────────────


def _zero_flats() -> QuadrotorFlats:
    """Hover flats: position and yaw jets all zero."""
    return QuadrotorFlats(x=Jet.zeros(order=4, dim=3), psi=Jet.zeros(order=2, dim=1))


def _circle_flats(
    t: float, omega: float = 0.5, radius: float = 1.0, height: float = 1.0
) -> QuadrotorFlats:
    """Analytic flats for a horizontal circle at constant altitude with the
    quadrotor yaw tracking the heading angle.

    x(t) = [r cos(ωt), r sin(ωt), h],   ψ(t) = ω t

    The non-zero ψ̇ and ψ̈ exercise the yaw-derivative paths in
    ``Quadrotor.recover``. (For a circle ψ̇ = ω is constant, so ψ̈ = 0; an
    additional yaw-acceleration check lives in ``TestYawWithAcceleration``.)
    """
    ct, st = np.cos(omega * t), np.sin(omega * t)
    pos = np.array([radius * ct, radius * st, height])
    vel = np.array([-radius * omega * st, radius * omega * ct, 0.0])
    acc = np.array([-radius * omega**2 * ct, -radius * omega**2 * st, 0.0])
    jerk = np.array([radius * omega**3 * st, -radius * omega**3 * ct, 0.0])
    snap = np.array([radius * omega**4 * ct, radius * omega**4 * st, 0.0])
    x_jet = Jet.from_list([pos, vel, acc, jerk, snap])
    psi_jet = Jet(np.array([omega * t, omega, 0.0]))
    return QuadrotorFlats(x=x_jet, psi=psi_jet)


# ─── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_build_from_physical_params(self):
        q = Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)
        assert q.mass == pytest.approx(DEFAULT_QUAD_MASS)
        np.testing.assert_allclose(q.inertia, DEFAULT_QUAD_INERTIA)

    def test_from_model_reads_model_params(self):
        model = QuadrotorBase()
        q = Quadrotor.from_model(model)
        assert q.mass == pytest.approx(model.mass)
        np.testing.assert_allclose(q.inertia, model.inertia)

    def test_inertia_shape_validated(self):
        with pytest.raises(ValueError, match="inertia must be 3×3"):
            Quadrotor(mass=1.0, inertia=np.zeros((2, 2)))

    def test_class_attribute_triple_declared(self):
        # Flat2State contract — all three attributes present.
        assert Quadrotor.Flats is QuadrotorFlats
        assert Quadrotor.RefState.__name__ == "QuadrotorRefState"
        assert Quadrotor.Inputs.__name__ == "QuadrotorInputs"


# ─── QuadrotorFlats validation ────────────────────────────────────────


class TestFlatsValidation:
    def test_valid_order_flats_construct(self):
        x = Jet.zeros(order=4, dim=3)
        psi = Jet.zeros(order=2, dim=1)
        # Should not raise.
        QuadrotorFlats(x=x, psi=psi)

    def test_higher_order_flats_also_accepted(self):
        x = Jet.zeros(order=6, dim=3)
        psi = Jet.zeros(order=3, dim=1)
        QuadrotorFlats(x=x, psi=psi)

    def test_x_wrong_dim_rejected(self):
        with pytest.raises(ValueError, match="x must be a 3-D jet"):
            QuadrotorFlats(x=Jet.zeros(order=4, dim=2), psi=Jet.zeros(order=2, dim=1))

    def test_x_insufficient_order_rejected(self):
        with pytest.raises(ValueError, match=r"x.order must be >= 4"):
            QuadrotorFlats(x=Jet.zeros(order=3, dim=3), psi=Jet.zeros(order=2, dim=1))

    def test_psi_not_scalar_rejected(self):
        with pytest.raises(ValueError, match="psi must be a scalar jet"):
            QuadrotorFlats(x=Jet.zeros(order=4, dim=3), psi=Jet.zeros(order=2, dim=3))

    def test_psi_insufficient_order_rejected(self):
        with pytest.raises(ValueError, match=r"psi.order must be >= 2"):
            QuadrotorFlats(x=Jet.zeros(order=4, dim=3), psi=Jet.zeros(order=1, dim=1))


# ─── Hover recovery ───────────────────────────────────────────────────


class TestHover:
    @pytest.fixture
    def quad(self):
        return Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)

    @pytest.fixture
    def ref_inputs(self, quad):
        return quad.recover(_zero_flats())

    def test_hover_thrust_equals_weight(self, ref_inputs):
        _, inputs = ref_inputs
        assert inputs.thrust == pytest.approx(DEFAULT_QUAD_MASS * GRAVITY, abs=1e-10)

    def test_hover_moment_is_zero(self, ref_inputs):
        _, inputs = ref_inputs
        np.testing.assert_allclose(inputs.moment, np.zeros(3), atol=1e-10)

    def test_hover_orientation_is_identity(self, ref_inputs):
        ref, _ = ref_inputs
        R = np.asarray(ref.orientation)
        err = np.linalg.norm(R - np.eye(3), ord="fro")
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_hover_angular_velocity_and_accel_zero(self, ref_inputs):
        ref, _ = ref_inputs
        np.testing.assert_allclose(np.asarray(ref.angular_velocity), np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(ref.angular_acceleration, np.zeros(3), atol=1e-10)


# ─── Constant-acceleration recovery ───────────────────────────────────


class TestConstantAccel:
    @pytest.fixture
    def quad(self):
        return Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)

    @pytest.fixture
    def ref_inputs(self, quad):
        x_jet = Jet.from_list(
            [np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)]
        )
        psi_jet = Jet.zeros(order=2, dim=1)
        return quad.recover(QuadrotorFlats(x=x_jet, psi=psi_jet))

    def test_thrust_matches_m_sqrt_g2_plus_ax2(self, ref_inputs):
        _, inputs = ref_inputs
        expected = DEFAULT_QUAD_MASS * np.sqrt(GRAVITY**2 + 1.0**2)
        assert inputs.thrust == pytest.approx(expected, abs=1e-10)
        # And it should be strictly larger than hover thrust.
        assert inputs.thrust > DEFAULT_QUAD_MASS * GRAVITY

    def test_body_z_tilted_forward(self, ref_inputs):
        ref, _ = ref_inputs
        R = np.asarray(ref.orientation)
        b3 = R[:, 2]
        assert b3[0] > 0.0, "b3 should tilt in +x when accel is +x"
        assert b3[2] > 0.0, "b3 should still point mostly +z"

    def test_static_attitude_has_zero_angular_velocity(self, ref_inputs):
        ref, _ = ref_inputs
        np.testing.assert_allclose(np.asarray(ref.angular_velocity), np.zeros(3), atol=1e-10)


# ─── Pure yaw-rate recovery ───────────────────────────────────────────


class TestPureYawRate:
    @pytest.fixture
    def quad(self):
        return Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)

    @pytest.fixture
    def ref_inputs(self, quad):
        x_jet = Jet.zeros(order=4, dim=3)
        psi_jet = Jet(np.array([0.0, 0.5, 0.0]))
        return quad.recover(QuadrotorFlats(x=x_jet, psi=psi_jet))

    def test_angular_velocity_matches_yaw_rate(self, ref_inputs):
        ref, _ = ref_inputs
        Omega = np.asarray(ref.angular_velocity)
        np.testing.assert_allclose(Omega[:2], np.zeros(2), atol=1e-10)
        assert Omega[2] == pytest.approx(0.5, abs=1e-10)

    def test_orientation_stays_identity_at_t0(self, ref_inputs):
        # ψ = 0 at t=0 even though ψ̇ = 0.5 — instantaneous R should be I.
        ref, _ = ref_inputs
        R = np.asarray(ref.orientation)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


# ─── Yaw with non-zero second derivative ──────────────────────────────


class TestYawWithAcceleration:
    """Pure rotation about the world yaw axis with a non-zero ψ̈ — exercises
    the dΩ recovery path that consumes the yaw acceleration."""

    @pytest.fixture
    def quad(self):
        return Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)

    @pytest.fixture
    def ref_inputs(self, quad):
        x_jet = Jet.zeros(order=4, dim=3)
        # ψ = 0, ψ̇ = 0.5, ψ̈ = 0.3 — all three derivatives non-trivial.
        psi_jet = Jet(np.array([0.0, 0.5, 0.3]))
        return quad.recover(QuadrotorFlats(x=x_jet, psi=psi_jet))

    def test_angular_velocity_yaw_axis(self, ref_inputs):
        ref, _ = ref_inputs
        Omega = np.asarray(ref.angular_velocity)
        np.testing.assert_allclose(Omega[:2], np.zeros(2), atol=1e-10)
        assert Omega[2] == pytest.approx(0.5, abs=1e-10)

    def test_angular_acceleration_yaw_axis(self, ref_inputs):
        ref, _ = ref_inputs
        dOmega = np.asarray(ref.angular_acceleration)
        # With zero translational accel and pure-yaw rotation, dΩ should be
        # entirely along world e3 and equal to ψ̈ at t = 0 (R = I).
        np.testing.assert_allclose(dOmega[:2], np.zeros(2), atol=1e-10)
        assert dOmega[2] == pytest.approx(0.3, abs=1e-10)

    def test_moment_yaw_axis(self, ref_inputs):
        _, inputs = ref_inputs
        # M = J dΩ + Ω × J Ω. At hover with pure yaw motion both terms lie
        # along e3 (Ω × J Ω is zero when Ω is along a principal axis and the
        # inertia is diagonal). Off-axis components must vanish.
        np.testing.assert_allclose(inputs.moment[:2], np.zeros(2), atol=1e-10)
        # The e3 component is J_zz · ψ̈ (no Coriolis contribution here).
        Jzz = DEFAULT_QUAD_INERTIA[2, 2]
        assert inputs.moment[2] == pytest.approx(Jzz * 0.3, abs=1e-10)


# ─── Integrator round-trip ────────────────────────────────────────────


class TestRoundTrip:
    """Recover (ref, inputs) from an analytic circular flat output, step a
    ``QuadrotorBase`` with the feedforward wrench, and confirm the integrated
    state tracks the analytic reference at the next sample."""

    @pytest.fixture
    def quad(self):
        return Quadrotor(mass=DEFAULT_QUAD_MASS, inertia=DEFAULT_QUAD_INERTIA)

    @pytest.fixture
    def model(self):
        return QuadrotorBase(input="wrench")

    def test_one_step_tracks_analytic_reference(self, quad, model):
        omega = 0.5
        radius = 1.0
        height = 1.0

        # Recover at t=0.
        flats0 = _circle_flats(0.0, omega=omega, radius=radius, height=height)
        ref0, u0 = quad.recover(flats0)

        # Seed the model with the reference state at t=0.
        state0 = ref0.as_state()
        model.reset(
            position=state0.position,
            velocity=state0.velocity,
            orientation=state0.orientation,
            angular_velocity=state0.angular_velocity,
        )

        # Feedforward wrench input.
        u = np.array([u0.thrust, *u0.moment])

        # One outer step (dt = 5 ms, 4 inner substeps at h = 1.25 ms).
        model.step(u)
        t_next = model.t

        # Analytic flats at the next sample, and the reference state there.
        flats1 = _circle_flats(t_next, omega=omega, radius=radius, height=height)
        ref1, _ = quad.recover(flats1)

        # Position and velocity should track the analytic reference tightly
        # — ZOH over 5 ms, smooth trajectory.
        np.testing.assert_allclose(model.state.position, ref1.position, atol=1e-3)
        np.testing.assert_allclose(model.state.velocity, ref1.velocity, atol=1e-3)

        # Attitude error (Frobenius) — smooth reference, short horizon.
        R_sim = np.asarray(model.state.orientation)
        R_ref = np.asarray(ref1.orientation)
        att_err = np.linalg.norm(R_sim - R_ref, ord="fro")
        assert att_err < 1e-2

    def test_two_steps_tracks_analytic_reference(self, quad, model):
        omega = 0.5
        radius = 1.0
        height = 1.0

        flats0 = _circle_flats(0.0, omega=omega, radius=radius, height=height)
        ref0, _ = quad.recover(flats0)
        model.reset(
            position=ref0.position,
            velocity=ref0.velocity,
            orientation=ref0.orientation,
            angular_velocity=ref0.angular_velocity,
        )

        # Two outer steps, re-recovering the feedforward input each outer tick.
        for _ in range(2):
            flats_t = _circle_flats(model.t, omega=omega, radius=radius, height=height)
            _, u_t = quad.recover(flats_t)
            model.step(np.array([u_t.thrust, *u_t.moment]))

        flats_end = _circle_flats(model.t, omega=omega, radius=radius, height=height)
        ref_end, _ = quad.recover(flats_end)

        np.testing.assert_allclose(model.state.position, ref_end.position, atol=1e-2)
        np.testing.assert_allclose(model.state.velocity, ref_end.velocity, atol=1e-2)
