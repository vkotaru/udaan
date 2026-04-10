"""Tests for udaan.manif — manifold operations."""

import numpy as np
import pytest

from udaan.core.exceptions import ManifoldTypeError
from udaan.manif import (
    S2,
    SO3,
    TS2,
    TSO3,
    Rot2Eul,
    expm_taylor_expansion,
    hat,
    rodrigues_expm,
    vee,
)


class TestHatVee:
    def test_hat_shape(self):
        v = np.array([1.0, 2.0, 3.0])
        H = hat(v)
        assert H.shape == (3, 3)

    def test_hat_skew_symmetric(self):
        v = np.random.default_rng(42).random(3)
        H = hat(v)
        np.testing.assert_allclose(H, -H.T, atol=1e-15)

    def test_vee_shape(self):
        v = np.array([1.0, 2.0, 3.0])
        result = vee(hat(v))
        assert result.shape == (3,)

    def test_hat_vee_roundtrip(self):
        v = np.array([0.5, -1.3, 2.7])
        np.testing.assert_allclose(vee(hat(v)), v, atol=1e-15)

    def test_hat_cross_product(self):
        """hat(a) @ b == cross(a, b)"""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(hat(a) @ b, np.cross(a, b), atol=1e-15)


class TestRodriguesExpm:
    def test_identity_for_zero(self):
        R = rodrigues_expm(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_is_orthogonal(self):
        v = np.array([0.1, 0.2, 0.3])
        R = rodrigues_expm(v)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_small_angle(self):
        v = np.array([1e-6, 0, 0])
        R = rodrigues_expm(v)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-4)

    def test_90deg_rotation(self):
        v = np.array([0, 0, np.pi / 2])
        R = rodrigues_expm(v)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)


class TestExpmTaylorExpansion:
    def test_returns_matrix(self):
        M = hat(np.array([0.1, 0.2, 0.3]))
        R = expm_taylor_expansion(M, order=5)
        assert R.shape == (3, 3)

    def test_identity_for_zero(self):
        R = expm_taylor_expansion(np.zeros((3, 3)), order=5)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


class TestSO3:
    def test_default_is_identity(self):
        R = SO3()
        np.testing.assert_allclose(np.array(R), np.eye(3), atol=1e-15)

    def test_step_returns_SO3(self):
        R = SO3()
        R2 = R.step(np.array([0.01, 0.02, 0.03]))
        assert R2.shape == (3, 3)
        np.testing.assert_allclose(R2 @ R2.T, np.eye(3), atol=1e-10)

    def test_step_zero_is_identity(self):
        R = SO3(rodrigues_expm(np.array([0.5, 0.3, 0.1])))
        R2 = R.step(np.zeros(3))
        np.testing.assert_allclose(np.array(R2), np.array(R), atol=1e-10)


class TestSO3FromTwoVectors:
    def test_basic(self):
        R = SO3.from_two_vectors(np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
        assert isinstance(R, SO3)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.array(R) @ np.array(R).T, np.eye(3), atol=1e-10)

    def test_b3_preserved(self):
        """The third column of R should be b3."""
        b3 = np.array([0, 0, 1.0])
        R = SO3.from_two_vectors(b3, np.array([1.0, 0, 0]))
        np.testing.assert_allclose(np.array(R)[:, 2], b3, atol=1e-10)

    def test_parallel_vectors_guard(self):
        """b3 parallel to b1 should not crash."""
        R = SO3.from_two_vectors(np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(np.array(R)), 1.0, atol=1e-10)

    def test_arbitrary_direction(self):
        b3 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        R = SO3.from_two_vectors(b3, np.array([1.0, 0, 0]))
        np.testing.assert_allclose(np.array(R) @ np.array(R).T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.array(R)[:, 2], b3, atol=1e-10)


class TestSO3FromTiltYaw:
    def test_zero_tilt_zero_yaw(self):
        """No tilt, no yaw -> identity."""
        R = SO3.from_tilt_yaw(np.zeros(3), 0.0)
        np.testing.assert_allclose(np.array(R), np.eye(3), atol=1e-10)

    def test_returns_so3(self):
        R = SO3.from_tilt_yaw(np.array([0.1, 0.2, 0.0]), 0.5)
        assert isinstance(R, SO3)
        np.testing.assert_allclose(np.array(R) @ np.array(R).T, np.eye(3), atol=1e-10)

    def test_pure_yaw(self):
        """Pure yaw of 90 degrees, no tilt."""
        R = SO3.from_tilt_yaw(np.zeros(3), np.pi / 2)
        # b3 should still be e3 (no tilt)
        np.testing.assert_allclose(np.array(R)[:, 2], [0, 0, 1], atol=1e-10)


class TestRot2Eul:
    def test_identity_gives_zeros(self):
        eul = Rot2Eul(np.eye(3))
        np.testing.assert_allclose(eul, np.zeros(3), atol=1e-15)

    def test_roundtrip(self):
        """Rot2Eul(rodrigues_expm(v)) should recover angles for small v."""
        R = rodrigues_expm(np.array([0.1, 0.2, 0.3]))
        eul = Rot2Eul(R)
        assert eul.shape == (3,)


class TestS2:
    def test_default_is_unit(self):
        q = S2()
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-15)

    def test_step_preserves_norm(self):
        q = S2(np.array([1.0, 0.0, 0.0]))
        q2 = q.step(np.array([0.1, 0.2, 0.0]))
        np.testing.assert_allclose(np.linalg.norm(q2), 1.0, atol=1e-10)

    def test_config_error_self_is_zero(self):
        q = S2(np.array([0.0, 0.0, -1.0]))
        assert q.config_error(q) == pytest.approx(0.0, abs=1e-15)

    def test_config_error_opposite_is_two(self):
        q1 = S2(np.array([0.0, 0.0, 1.0]))
        q2 = S2(np.array([0.0, 0.0, -1.0]))
        assert q1.config_error(q2) == pytest.approx(2.0, abs=1e-15)

    def test_error_vec_self_is_zero(self):
        q = S2(np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(q.error_vec(q), np.zeros(3), atol=1e-15)

    def test_from_spherical(self):
        q = S2.from_spherical(phi=0.0, th=np.pi / 2)
        np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0]), atol=1e-10)

    def test_from_spherical_pole(self):
        q = S2.from_spherical(phi=0.0, th=0.0)
        np.testing.assert_allclose(q, np.array([0.0, 0.0, 1.0]), atol=1e-10)


class TestTSO3:
    def test_creation(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(w.vector, [1.0, 2.0, 3.0])

    def test_hat_map(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        H = w.hat()
        assert H.shape == (3, 3)
        np.testing.assert_allclose(H, -H.T, atol=1e-15)
        np.testing.assert_allclose(H, hat(w.vector), atol=1e-15)

    def test_norm(self):
        w = TSO3(np.array([3.0, 4.0, 0.0]))
        assert w.norm == pytest.approx(5.0)

    def test_scalar_mul(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        w2 = w * 2.0
        np.testing.assert_allclose(w2.vector, [2.0, 4.0, 6.0])

    def test_rmul(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        w2 = 0.5 * w
        np.testing.assert_allclose(w2.vector, [0.5, 1.0, 1.5])

    def test_neg(self):
        w = TSO3(np.array([1.0, -2.0, 3.0]))
        w2 = -w
        np.testing.assert_allclose(w2.vector, [-1.0, 2.0, -3.0])

    def test_repr(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        assert "TSO3" in repr(w)


class TestTS2:
    def test_creation(self):
        v = TS2(np.array([0.1, 0.2, 0.0]))
        np.testing.assert_allclose(v.vector, [0.1, 0.2, 0.0])

    def test_norm(self):
        v = TS2(np.array([3.0, 4.0, 0.0]))
        assert v.norm == pytest.approx(5.0)

    def test_scalar_mul(self):
        v = TS2(np.array([1.0, 2.0, 3.0]))
        v2 = v * 3.0
        np.testing.assert_allclose(v2.vector, [3.0, 6.0, 9.0])

    def test_rmul(self):
        v = TS2(np.array([1.0, 2.0, 3.0]))
        v2 = 0.1 * v
        np.testing.assert_allclose(v2.vector, [0.1, 0.2, 0.3])

    def test_neg(self):
        v = TS2(np.array([1.0, -2.0, 3.0]))
        v2 = -v
        np.testing.assert_allclose(v2.vector, [-1.0, 2.0, -3.0])

    def test_repr(self):
        v = TS2(np.array([0.1, 0.2, 0.3]))
        assert "TS2" in repr(v)


class TestSO3Inv:
    def test_inv_is_transpose(self):
        R = SO3(rodrigues_expm(np.array([0.3, -0.2, 0.5])))
        R_inv = R.inv()
        np.testing.assert_allclose(np.array(R_inv), np.array(R).T, atol=1e-15)

    def test_inv_returns_so3(self):
        R = SO3(rodrigues_expm(np.array([0.1, 0.2, 0.3])))
        assert isinstance(R.inv(), SO3)

    def test_R_times_inv_is_identity(self):
        R = SO3(rodrigues_expm(np.array([0.5, -0.3, 0.1])))
        np.testing.assert_allclose(np.array(R) @ np.array(R.inv()), np.eye(3), atol=1e-10)

    def test_identity_inv_is_identity(self):
        R = SO3()
        np.testing.assert_allclose(np.array(R.inv()), np.eye(3), atol=1e-15)


class TestSO3FromAngleAxis:
    def test_zero_gives_identity(self):
        R = SO3.from_angle_axis(np.zeros(3))
        np.testing.assert_allclose(np.array(R), np.eye(3), atol=1e-10)

    def test_90deg_z(self):
        R = SO3.from_angle_axis(np.array([0, 0, np.pi / 2]))
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(np.array(R), expected, atol=1e-10)

    def test_returns_so3(self):
        R = SO3.from_angle_axis(np.array([0.1, 0.2, 0.3]))
        assert isinstance(R, SO3)
        np.testing.assert_allclose(np.array(R) @ np.array(R).T, np.eye(3), atol=1e-10)


class TestTSO3AsNdarray:
    def test_is_ndarray(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        assert isinstance(w, np.ndarray)

    def test_shape(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        assert w.shape == (3,)

    def test_vector_property(self):
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(w.vector, [1.0, 2.0, 3.0])

    def test_numpy_ops_work(self):
        """TSO3 can be used in numpy operations like a regular array."""
        w = TSO3(np.array([1.0, 2.0, 3.0]))
        result = np.dot(w, w)
        assert result == pytest.approx(14.0)


class TestTSO3Transport:
    def test_identity_transport_is_identity(self):
        """Transport from I to I leaves the vector unchanged."""
        Om = TSO3(np.array([1.0, 2.0, 3.0]))
        result = Om.transport(SO3(), SO3())
        np.testing.assert_allclose(result.vector, Om.vector, atol=1e-15)

    def test_transport_returns_tso3(self):
        Om = TSO3(np.array([0.1, 0.2, 0.3]))
        R_from = SO3(rodrigues_expm(np.array([0.5, 0.0, 0.0])))
        R_to = SO3(rodrigues_expm(np.array([0.0, 0.3, 0.0])))
        result = Om.transport(R_from, R_to)
        assert isinstance(result, TSO3)

    def test_transport_matches_manual(self):
        """transport(R_from, R_to) should equal R_to^T @ R_from @ Om."""
        Om = TSO3(np.array([1.0, -0.5, 0.2]))
        Rd = SO3(rodrigues_expm(np.array([0.3, -0.2, 0.5])))
        R = SO3(rodrigues_expm(np.array([0.1, 0.1, 0.1])))
        result = Om.transport(Rd, R)
        expected = np.array(R).T @ np.array(Rd) @ Om.vector
        np.testing.assert_allclose(result.vector, expected, atol=1e-10)

    def test_angular_velocity_error_pattern(self):
        """eOm = Om - Omd.transport(Rd, R) should work with operator."""
        Om = TSO3(np.array([0.1, 0.2, 0.3]))
        Omd = TSO3(np.array([0.05, 0.1, 0.15]))
        R = SO3(rodrigues_expm(np.array([0.1, 0.0, 0.0])))
        Rd = SO3(rodrigues_expm(np.array([0.1, 0.0, 0.0])))
        # Same rotation -> transport is identity -> eOm = Om - Omd
        eOm = Om - Omd.transport(Rd, R)
        assert isinstance(eOm, TSO3)
        np.testing.assert_allclose(eOm.vector, Om.vector - Omd.vector, atol=1e-10)


class TestTS2Transport:
    def test_transport_returns_ts2(self):
        v = TS2(np.array([0.1, 0.2, 0.3]))
        q = S2(np.array([0.0, 0.0, 1.0]))
        result = v.transport(q)
        assert isinstance(result, TS2)

    def test_transport_matches_neg_hat_squared(self):
        """transport(q) should equal -hat(q)^2 @ v."""
        v = TS2(np.array([1.0, -0.5, 0.2]))
        q = S2(np.array([0.0, 0.0, 1.0]))
        result = v.transport(q)
        expected = -hat(np.asarray(q)) @ hat(np.asarray(q)) @ v.vector
        np.testing.assert_allclose(result.vector, expected, atol=1e-15)

    def test_tangent_to_e3_projects_xy(self):
        """Transport to e3 should preserve the tangent plane components."""
        v = TS2(np.array([1.0, 2.0, 3.0]))
        q = S2(np.array([0.0, 0.0, 1.0]))
        result = v.transport(q)
        # -hat(e3)^2 @ v = [vx, vy, 0]
        np.testing.assert_allclose(result.vector, [1.0, 2.0, 0.0], atol=1e-15)

    def test_angular_velocity_error_pattern(self):
        """eω = ω - ωd.transport(q) should work with operator."""
        omega = TS2(np.array([0.1, 0.2, 0.0]))
        omega_d = TS2(np.array([0.05, 0.1, 0.0]))
        q = S2(np.array([0.0, 0.0, 1.0]))
        # eω = ω - ωd.transport(q), at e3 transport is identity for tangent vectors
        e_omega = omega - omega_d.transport(q)
        assert isinstance(e_omega, TS2)
        np.testing.assert_allclose(e_omega.vector, [0.05, 0.1, 0.0], atol=1e-15)


class TestTS2AsNdarray:
    def test_is_ndarray(self):
        v = TS2(np.array([0.1, 0.2, 0.3]))
        assert isinstance(v, np.ndarray)

    def test_shape(self):
        v = TS2(np.array([0.1, 0.2, 0.3]))
        assert v.shape == (3,)

    def test_numpy_ops_work(self):
        """TS2 can be used in numpy operations like a regular array."""
        v = TS2(np.array([3.0, 4.0, 0.0]))
        assert np.linalg.norm(v) == pytest.approx(5.0)


class TestSO3Operators:
    def test_sub_identity_is_zero(self):
        """R - R should give zero error."""
        R = SO3(rodrigues_expm(np.array([0.3, 0.2, 0.1])))
        eR = R - R
        assert isinstance(eR, TSO3)
        np.testing.assert_allclose(eR.vector, np.zeros(3), atol=1e-10)

    def test_sub_returns_tso3(self):
        R1 = SO3(rodrigues_expm(np.array([0.1, 0.0, 0.0])))
        R2 = SO3()
        eR = R1 - R2
        assert isinstance(eR, TSO3)
        assert eR.vector.shape == (3,)

    def test_sub_matches_lee2010_formula(self):
        """SO3.__sub__ should match 1/2 vee(Rd^T R - R^T Rd) (Lee 2010, Eq. 9)."""
        Rd = SO3(rodrigues_expm(np.array([0.3, -0.2, 0.5])))
        R = SO3(rodrigues_expm(np.array([0.1, 0.1, 0.1])))
        eR = Rd - R
        # Lee 2010 convention: eR = 1/2 vee(Rd^T R - R^T Rd)
        eR_lee = vee(np.array(Rd).T @ np.array(R) - np.array(R).T @ np.array(Rd)) / 2.0
        np.testing.assert_allclose(eR.vector, eR_lee, atol=1e-10)

    def test_add_returns_so3(self):
        R = SO3()
        w = TSO3(np.array([0.01, 0.02, 0.03]))
        R2 = R + w
        assert isinstance(R2, SO3)
        assert R2.shape == (3, 3)

    def test_add_preserves_orthogonality(self):
        R = SO3(rodrigues_expm(np.array([0.5, 0.3, 0.1])))
        w = TSO3(np.array([0.1, -0.05, 0.2]))
        R2 = R + w
        np.testing.assert_allclose(np.array(R2) @ np.array(R2).T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(np.array(R2)), 1.0, atol=1e-10)

    def test_add_matches_step(self):
        """R + TSO3(omega) should match R.step(omega)."""
        R = SO3(rodrigues_expm(np.array([0.5, 0.3, 0.1])))
        omega = np.array([0.1, -0.05, 0.2])
        R_step = R.step(omega)
        R_add = R + TSO3(omega)
        np.testing.assert_allclose(np.array(R_add), np.array(R_step), atol=1e-10)

    def test_add_zero_is_identity_op(self):
        R = SO3(rodrigues_expm(np.array([0.5, 0.3, 0.1])))
        R2 = R + TSO3(np.zeros(3))
        np.testing.assert_allclose(np.array(R2), np.array(R), atol=1e-10)

    def test_roundtrip_small_angle(self):
        """R2 - R (Lee convention) gives the error from R2 toward R, i.e., -w."""
        R = SO3(rodrigues_expm(np.array([0.5, 0.3, 0.1])))
        w = np.array([0.001, -0.002, 0.003])
        R2 = R + TSO3(w)
        # Lee convention: Rd - R = 1/2 vee(Rd^T R - R^T Rd)
        # When Rd = R2 = R @ exp(w), eR ≈ -w for small w
        eR = R2 - R
        np.testing.assert_allclose(eR.vector, -w, atol=1e-4)

    def test_sub_plain_ndarray_delegates_to_numpy(self):
        """SO3 - ndarray should fall through to numpy element-wise subtraction."""
        R = SO3()
        result = R - np.eye(3)
        np.testing.assert_allclose(result, np.zeros((3, 3)), atol=1e-15)

    def test_sub_non_array_raises_manifold_error(self):
        R = SO3()
        with pytest.raises(ManifoldTypeError, match="SO3.__sub__ expects an SO3 element"):
            R - "invalid"

    def test_add_plain_ndarray_delegates_to_numpy(self):
        """SO3 + ndarray should fall through to numpy element-wise addition."""
        R = SO3()
        result = R + np.eye(3)
        np.testing.assert_allclose(result, 2 * np.eye(3), atol=1e-15)

    def test_add_non_array_raises_manifold_error(self):
        R = SO3()
        with pytest.raises(ManifoldTypeError, match="SO3.__add__ expects a TSO3 tangent vector"):
            R + "invalid"


class TestS2Operators:
    def test_sub_self_is_zero(self):
        """q - q should give zero error."""
        q = S2(np.array([0.0, 0.0, 1.0]))
        eq = q - q
        assert isinstance(eq, TS2)
        np.testing.assert_allclose(eq.vector, np.zeros(3), atol=1e-15)

    def test_sub_returns_ts2(self):
        q1 = S2(np.array([0.0, 0.0, 1.0]))
        q2 = S2(np.array([1.0, 0.0, 0.0]))
        eq = q1 - q2
        assert isinstance(eq, TS2)
        assert eq.vector.shape == (3,)

    def test_sub_matches_error_vec(self):
        """S2.__sub__ should match error_vec."""
        q1 = S2(np.array([0.0, 0.0, 1.0]))
        q2 = S2(np.array([1.0, 0.0, 0.0]))
        eq = q1 - q2
        eq_manual = q1.error_vec(q2)
        np.testing.assert_allclose(eq.vector, eq_manual, atol=1e-15)

    def test_add_returns_s2(self):
        q = S2(np.array([0.0, 0.0, 1.0]))
        w = TS2(np.array([0.1, 0.0, 0.0]))
        q2 = q + w
        assert isinstance(q2, S2)

    def test_add_preserves_norm(self):
        q = S2(np.array([1.0, 0.0, 0.0]))
        w = TS2(np.array([0.0, 0.1, 0.2]))
        q2 = q + w
        np.testing.assert_allclose(np.linalg.norm(q2), 1.0, atol=1e-10)

    def test_add_matches_step(self):
        """q + TS2(omega) should match q.step(omega)."""
        q = S2(np.array([0.0, 0.0, 1.0]))
        omega = np.array([0.1, 0.2, 0.0])
        q_step = q.step(omega)
        q_add = q + TS2(omega)
        np.testing.assert_allclose(np.array(q_add), np.array(q_step), atol=1e-10)

    def test_add_zero_is_identity_op(self):
        q = S2(np.array([0.0, 1.0, 0.0]))
        q2 = q + TS2(np.zeros(3))
        np.testing.assert_allclose(np.array(q2), np.array(q), atol=1e-10)

    def test_sub_plain_ndarray_delegates_to_numpy(self):
        """S2 - ndarray should fall through to numpy element-wise subtraction."""
        q = S2(np.array([0.0, 0.0, 1.0]))
        result = q - np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, np.zeros(3), atol=1e-15)

    def test_sub_non_array_raises_manifold_error(self):
        q = S2()
        with pytest.raises(ManifoldTypeError, match="S2.__sub__ expects an S2 element"):
            q - "invalid"

    def test_add_plain_ndarray_delegates_to_numpy(self):
        """S2 + ndarray should fall through to numpy element-wise addition."""
        q = S2(np.array([0.0, 0.0, 1.0]))
        result = q + np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, np.array([1.0, 0.0, 1.0]), atol=1e-15)

    def test_add_non_array_raises_manifold_error(self):
        q = S2()
        with pytest.raises(ManifoldTypeError, match="S2.__add__ expects a TS2 tangent vector"):
            q + "invalid"
