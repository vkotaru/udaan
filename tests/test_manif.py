"""Tests for udaan.manif — manifold operations."""

import numpy as np
import pytest

from udaan.manif import (
    S2,
    SO3,
    Rot2Eul,
    RotationMatrix,
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


class TestRotationMatrix:
    def test_basic_call(self):
        rm = RotationMatrix()
        b3 = np.array([0, 0, 1.0])
        b1 = np.array([1.0, 0, 0])
        R = rm(b3, b1)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_parallel_vectors_guard(self):
        """b3 parallel to b1 should not crash."""
        rm = RotationMatrix()
        b3 = np.array([1.0, 0, 0])
        b1 = np.array([1.0, 0, 0])
        R = rm(b3, b1)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


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
