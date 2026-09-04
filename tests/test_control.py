"""Tests for udaan.control — controller instantiation and output dimensions."""

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT
from udaan.control import Gains, PDController
from udaan.control.quadrotor import (
    DirectPropellerForceController,
    GeometricAttitudeController,
    PositionPDController,
)
from udaan.control.quadrotor.geometric_attitude import _swung_heading
from udaan.control.quadrotor_cspayload import QuadCSPayloadController
from udaan.core.defaults import MIN_ATTITUDE_HINT_CONDITIONING
from udaan.manif import SO3, TSO3, rodrigues_expm


class TestGains:
    def test_default(self):
        g = Gains()
        np.testing.assert_allclose(g.kp, np.zeros(3))
        np.testing.assert_allclose(g.kd, np.zeros(3))
        np.testing.assert_allclose(g.ki, np.zeros(3))

    def test_custom(self):
        g = Gains(kp=np.ones(3), kd=2 * np.ones(3))
        np.testing.assert_allclose(g.kp, np.ones(3))
        np.testing.assert_allclose(g.kd, 2 * np.ones(3))


class TestPDController:
    def test_compute_output_shape(self):
        ctrl = PDController(kp=np.ones(3), kd=np.ones(3))
        t = 0.0
        state = (np.array([1.0, 0.0, 0.0]), np.zeros(3))
        u = ctrl.compute(t, state)
        assert u.shape == (3,)

    def test_at_setpoint_gives_zero(self):
        sp = np.array([0.0, 0.0, 1.0])
        ctrl = PDController(kp=np.ones(3), kd=np.ones(3))
        u = ctrl.compute(0.0, (sp, np.zeros(3)))
        np.testing.assert_allclose(u, np.zeros(3), atol=1e-15)


class TestPositionPDController:
    def test_output_shape(self):
        ctrl = PositionPDController(mass=1.0)
        u = ctrl.compute(0.0, (np.zeros(3), np.zeros(3)))
        assert u.shape == (3,)


class TestGeometricAttitudeController:
    def test_output_shape(self):
        from udaan.manif import SO3, TSO3

        inertia = np.diag([0.01, 0.01, 0.02])
        ctrl = GeometricAttitudeController(inertia=inertia)
        R = SO3()
        Omega = TSO3()
        thrust_force = np.array([0.0, 0.0, 9.81])
        f, M = ctrl.compute(0.0, (R, Omega), thrust_force)
        assert np.isscalar(f) or f.shape == ()
        assert M.shape == (3,)

    def test_hover_gives_gravity_thrust(self):
        from udaan.manif import SO3, TSO3

        mass = 1.0
        inertia = np.diag([0.01, 0.01, 0.02])
        ctrl = GeometricAttitudeController(mass=mass, inertia=inertia)
        thrust_force = np.array([0.0, 0.0, 9.81])
        f, M = ctrl.compute(0.0, (SO3(), TSO3()), thrust_force)
        assert f > 0


class TestDirectPropellerForceController:
    def test_output_shape(self):
        from udaan.manif import SO3, TSO3

        ctrl = DirectPropellerForceController(mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]))
        state = (np.zeros(3), np.zeros(3), SO3(), TSO3())
        u = ctrl.compute(0.0, state)
        assert u.shape == (4,)


class TestQuadCSPayloadController:
    def test_output_shape(self):
        ctrl = QuadCSPayloadController()

        class MockState:
            payload_position = np.array([0.0, 0.0, 0.0])
            payload_velocity = np.zeros(3)
            cable_attitude = np.array([0.0, 0.0, -1.0])

            def dq(self):
                return np.zeros(3)

        u = ctrl.compute(0.0, MockState())
        assert u.shape == (3,)


class TestReferenceAttitudeComposition:
    """``desired_att``'s ``Rd`` contributes its heading, and only its heading.

    The channel used to be inert: ``Rd`` was recomputed from the thrust
    direction and the supplied one discarded, so passing ``R_i = I`` gave a
    bit-identical run. Substituting it *wholesale* is not the fix — that
    overwrites ``b₃`` and severs the position loop, taking brisk-spiral
    tracking RMS from 0.170 m to 4.39 m (M1), 0.197 m to 4.06 m (M3) and
    0.306 m to 4.00 m (M4).

    The composition splits the two commands by degree of freedom: ``b₃`` stays
    with the thrust command, and the reference resolves the heading about it.
    """

    @staticmethod
    def _controller():
        return GeometricAttitudeController(inertia=np.diag([0.01, 0.01, 0.02]))

    def test_the_thrust_direction_still_sets_b3(self):
        """Position feedback's path into the attitude command is untouched."""
        ctrl = self._controller()
        thrust = np.array([1.0, 2.0, 9.0])
        reference = SO3(rodrigues_expm(np.array([0.0, 0.0, 1.1])))
        Rd = ctrl._cmd_accel_to_cmd_att(thrust, reference=reference)
        np.testing.assert_allclose(
            np.asarray(Rd)[:, 2], thrust / np.linalg.norm(thrust), atol=TOL_ANALYTIC
        )

    def test_the_reference_sets_the_heading(self):
        """A yawed reference yaws the command; the inert version could not."""
        ctrl = self._controller()
        thrust = np.array([0.0, 0.0, 9.0])
        yawed = SO3(rodrigues_expm(np.array([0.0, 0.0, 0.9])))
        with_reference = np.asarray(ctrl._cmd_accel_to_cmd_att(thrust, reference=yawed))
        without = np.asarray(ctrl._cmd_accel_to_cmd_att(thrust))
        np.testing.assert_allclose(with_reference[:, 0], np.asarray(yawed)[:, 0], atol=TOL_ANALYTIC)
        np.testing.assert_allclose(without[:, 0], np.array([1.0, 0.0, 0.0]), atol=TOL_ANALYTIC)
        assert np.linalg.norm(with_reference - without) > 0.5

    def test_the_channel_is_not_inert(self):
        """Guard the regression directly: identity must not reproduce the reference."""
        ctrl = self._controller()
        state = (SO3(), TSO3())
        thrust = np.array([0.3, -0.2, 9.0])
        yawed = SO3(rodrigues_expm(np.array([0.0, 0.0, 0.9])))
        rates = (TSO3(np.array([0.1, 0.0, 0.2])), np.zeros(3))
        _, torque_ref = ctrl.compute(0.0, state, thrust, (yawed, *rates))
        _, torque_identity = ctrl.compute(0.0, state, thrust, (SO3(), *rates))
        assert np.linalg.norm(torque_ref - torque_identity) > 1e-3, (
            "passing identity reproduced the reference run — the channel is inert again"
        )

    def test_no_reference_is_heading_hold_at_zero_yaw(self):
        """The fallback is unchanged, so nothing without feedforward moved."""
        ctrl = self._controller()
        thrust = np.array([0.3, -0.2, 9.0])
        state = (SO3(), TSO3())
        f_a, M_a = ctrl.compute(0.0, state, thrust)
        f_b, M_b = ctrl.compute(0.0, state, thrust, None)
        assert f_a == f_b
        np.testing.assert_allclose(M_a, M_b, atol=TOL_EXACT)


class TestHeadingHintConditioning:
    r"""The heading hint stays usable as ``b₃`` swings onto the reference's ``b₁``.

    :meth:`~udaan.manif.SO3.from_two_vectors` divides by ``‖b₃ × hint‖``. The
    fixed world-``x`` hint the composition replaced was benign; the
    reference's own ``b₁`` tilts with the reference and can line up with the
    commanded thrust direction, which is a genuine singularity of the
    construction and *not* caught by ``from_two_vectors``'s own ``1e-10``
    guard.

    Measured on an M1 spiral at 8 m/s, radius 2 m, ψ̇ = 3 rad/s: unguarded, the
    conditioning reached 0.0065 and the commanded attitude jumped 152° in one
    5 ms step, straight into ``eR``; the world-``x`` hint's worst on the same
    flown commands was 31°. Guarded, the same run floors conditioning at 0.31
    and the worst jump at 31.2°.
    """

    #: The measured worst conditioning of the unguarded hint.
    MEASURED_DEGENERATE = 0.00593

    @staticmethod
    def _reference(tilt=0.9, yaw=0.7):
        """A tilted, yawed reference attitude — the flat map's usual output."""
        return np.asarray(
            SO3(
                rodrigues_expm(np.array([0.0, 0.0, yaw]))
                @ rodrigues_expm(np.array([0.0, tilt, 0.0]))
            )
        )

    @staticmethod
    def _sweep_through(reference, offset):
        """``b₃`` at angle ``offset`` from the reference's ``b₁``, in its ``b₁b₂`` plane.

        ``offset = 0`` is the singularity itself; ``sin(offset)`` is the
        conditioning of the raw hint.
        """
        b1, b2 = reference[:, 0], reference[:, 1]
        return np.cos(offset) * b1 + np.sin(offset) * b2

    def test_the_degenerate_case_is_reachable_at_all(self):
        """Sanity: the sweep really does drive the raw hint to the measured value."""
        reference = self._reference()
        b3 = self._sweep_through(reference, np.arcsin(self.MEASURED_DEGENERATE))
        raw = float(np.linalg.norm(np.cross(b3, reference[:, 0])))
        assert raw == pytest.approx(self.MEASURED_DEGENERATE, rel=1e-6)

    def test_conditioning_is_floored_at_the_threshold(self):
        """Whatever the raw hint does, the hint actually used stays usable."""
        reference = self._reference()
        for offset in np.linspace(-0.6, 0.6, 241):
            b3 = self._sweep_through(reference, offset)
            hint = GeometricAttitudeController._heading_hint(b3, reference)
            conditioning = float(np.linalg.norm(np.cross(b3, hint)))
            assert conditioning >= MIN_ATTITUDE_HINT_CONDITIONING - TOL_ANALYTIC, (
                f"conditioning {conditioning:.4g} at offset {offset:.4f}"
            )

    def test_no_large_single_step_jump_through_the_degenerate_case(self):
        """The command stays continuous across the singularity.

        The sweep walks ``b₃`` straight through the reference's ``b₁`` in steps
        of 1 mrad — finer than any 5 ms step of the flown run. Unguarded, the
        projected hint reverses as ``b₃`` crosses, so the commanded attitude
        turns 147° between two neighbouring samples. Guarded, the worst step is
        0.36°, because the blend has gone all the way over to the swung hint by
        the time the raw one has no heading left in it.
        """
        ctrl = GeometricAttitudeController(inertia=np.diag([0.01, 0.01, 0.02]))
        reference = self._reference()
        offsets = np.arange(-0.05, 0.05, 1e-3)
        guarded = [
            np.asarray(
                ctrl._cmd_accel_to_cmd_att(9.0 * self._sweep_through(reference, o), SO3(reference))
            )
            for o in offsets
        ]
        unguarded = [
            np.asarray(SO3.from_two_vectors(self._sweep_through(reference, o), reference[:, 0]))
            for o in offsets
        ]
        assert _max_step_angle(unguarded) > 90.0, "the sweep no longer exercises the singularity"
        assert _max_step_angle(guarded) < 5.0, (
            f"guarded command jumped {_max_step_angle(guarded):.1f}° in one step"
        )

    def test_a_well_conditioned_hint_is_passed_through_untouched(self):
        """The guard is inert where it should be — no change away from the cliff."""
        reference = self._reference()
        b3 = np.array([0.0, 0.0, 1.0])
        assert np.linalg.norm(np.cross(b3, reference[:, 0])) > MIN_ATTITUDE_HINT_CONDITIONING
        np.testing.assert_allclose(
            GeometricAttitudeController._heading_hint(b3, reference),
            reference[:, 0],
            atol=TOL_EXACT,
        )

    def test_the_blend_is_continuous_at_the_threshold(self):
        """Weight 1 at the threshold, so switching on costs nothing."""
        reference = self._reference()
        offset = np.arcsin(MIN_ATTITUDE_HINT_CONDITIONING)
        inside = GeometricAttitudeController._heading_hint(
            self._sweep_through(reference, offset - 1e-9), reference
        )
        outside = GeometricAttitudeController._heading_hint(
            self._sweep_through(reference, offset + 1e-9), reference
        )
        # `outside` is the raw hint; `inside` is the blend at weight ~1. They
        # differ only by the projection onto the plane orthogonal to b3.
        b3 = self._sweep_through(reference, offset)
        projected = outside - float(b3 @ outside) * b3
        np.testing.assert_allclose(
            inside / np.linalg.norm(inside),
            projected / np.linalg.norm(projected),
            atol=1e-6,
        )

    def test_the_swung_hint_is_orthogonal_to_the_thrust_direction(self):
        """That orthogonality is what makes it the well-conditioned end of the blend."""
        reference = self._reference()
        for offset in np.linspace(-0.6, 0.6, 61):
            b3 = self._sweep_through(reference, offset)
            swung = _swung_heading(b3, reference)
            assert abs(float(b3 @ swung)) < TOL_ANALYTIC
            assert float(np.linalg.norm(swung)) == pytest.approx(1.0, abs=TOL_ANALYTIC)

    def test_an_inverted_thrust_command_has_no_minimal_rotation_to_use(self):
        """``b₃`` opposite the reference's ``b₃`` is where the swing is undefined.

        It is not reachable from :meth:`_heading_hint` — a ``b₃`` antiparallel
        to the reference's is *perpendicular* to the reference's ``b₁``, i.e.
        perfectly conditioned, so the guard never engages there. The branch
        exists so a caller handing in a reference that is not orthonormal gets
        a fallback rather than a division by ``1 + cos = 0``.
        """
        reference = self._reference()
        assert _swung_heading(-reference[:, 2], reference) is None


def _max_step_angle(rotations):
    """Largest angle, in degrees, between neighbouring frames of a sequence."""
    worst = 0.0
    for a, b in zip(rotations, rotations[1:], strict=False):
        cos = (np.trace(a.T @ b) - 1.0) / 2.0
        worst = max(worst, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return worst
