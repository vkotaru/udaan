"""Unit tests for :class:`udaan.utils.flatness.Jet`."""

from __future__ import annotations

import numpy as np
import pytest

from udaan.utils.flatness import Jet

# ─── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_1d_input_promoted_to_scalar_jet(self):
        j = Jet(np.array([0.0, 1.0, 2.0]))
        assert j.data.shape == (3, 1)
        assert j.is_scalar
        assert j.dim == 1
        assert j.order == 2

    def test_2d_input_vector_jet(self):
        arr = np.stack([np.zeros(3), np.ones(3), 2 * np.ones(3)])
        j = Jet(arr)
        assert j.data.shape == (3, 3)
        assert not j.is_scalar
        assert j.dim == 3
        assert j.order == 2

    def test_list_input_coerced(self):
        j = Jet([[0.0, 0.0], [1.0, 0.0]])
        assert j.data.shape == (2, 2)
        assert j.order == 1
        assert j.dim == 2

    def test_dtype_coerced_to_float(self):
        j = Jet(np.array([[1, 2, 3], [4, 5, 6]], dtype=int))
        assert j.data.dtype == np.float64

    def test_default_is_single_zero_scalar(self):
        j = Jet()
        assert j.data.shape == (1, 1)
        assert j.order == 0
        assert j.dim == 1
        assert j[0] == 0.0

    def test_3d_input_rejected(self):
        with pytest.raises(ValueError, match="1-D .* or 2-D"):
            Jet(np.zeros((2, 2, 2)))

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="at least one row"):
            Jet(np.zeros((0, 3)))


# ─── Factory constructors ─────────────────────────────────────────────


class TestFactories:
    def test_from_list_scalar(self):
        j = Jet.from_list([0.5, 1.0, 2.0])
        assert j.is_scalar
        assert j.order == 2
        assert j[0] == pytest.approx(0.5)
        assert j[2] == pytest.approx(2.0)

    def test_from_list_vector(self):
        pos, vel, acc = np.array([0, 0, 1]), np.array([1, 0, 0]), np.zeros(3)
        j = Jet.from_list([pos, vel, acc])
        assert j.dim == 3
        assert j.order == 2
        np.testing.assert_allclose(j[1], [1, 0, 0])

    def test_zeros_scalar(self):
        j = Jet.zeros(order=4)
        assert j.data.shape == (5, 1)
        assert j.is_scalar
        np.testing.assert_array_equal(j.data, 0)

    def test_zeros_vector(self):
        j = Jet.zeros(order=4, dim=3)
        assert j.data.shape == (5, 3)
        np.testing.assert_array_equal(j.data, 0)


# ─── Indexing & iteration ─────────────────────────────────────────────


class TestIndexing:
    def test_scalar_indexing_returns_float(self):
        j = Jet(np.array([0.1, 0.2, 0.3]))
        for k in range(3):
            v = j[k]
            assert isinstance(v, float)
            assert v == pytest.approx(0.1 * (k + 1))

    def test_vector_indexing_returns_1d_array(self):
        j = Jet(np.stack([np.arange(3), np.arange(3, 6), np.arange(6, 9)]).astype(float))
        for k in range(3):
            v = j[k]
            assert isinstance(v, np.ndarray)
            assert v.shape == (3,)
        np.testing.assert_allclose(j[2], [6, 7, 8])

    def test_out_of_range_raises(self):
        j = Jet(np.zeros((3, 2)))
        with pytest.raises(IndexError):
            _ = j[3]
        with pytest.raises(IndexError):
            _ = j[-1]

    def test_iteration_yields_all_derivatives(self):
        j = Jet(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        seen = list(j)
        assert len(seen) == 3
        np.testing.assert_allclose(seen[0], [1.0, 2.0])
        np.testing.assert_allclose(seen[2], [5.0, 6.0])

    def test_len_equals_order_plus_one(self):
        j = Jet.zeros(order=4, dim=3)
        assert len(j) == 5


# ─── Named accessors ──────────────────────────────────────────────────


class TestNamedAccessors:
    def test_accessors_alias_numerical_indexing(self):
        # Build an order-6 jet so every accessor resolves
        data = np.stack([k * np.array([1.0, 2.0]) for k in range(7)])
        j = Jet(data)
        np.testing.assert_array_equal(j.value, j[0])
        np.testing.assert_array_equal(j.velocity, j[1])
        np.testing.assert_array_equal(j.acceleration, j[2])
        np.testing.assert_array_equal(j.jerk, j[3])
        np.testing.assert_array_equal(j.snap, j[4])
        np.testing.assert_array_equal(j.crackle, j[5])
        np.testing.assert_array_equal(j.pop, j[6])

    def test_accessor_missing_order_raises(self):
        j = Jet.zeros(order=2, dim=3)
        _ = j.value
        _ = j.velocity
        _ = j.acceleration
        with pytest.raises(AttributeError, match="no jerk"):
            _ = j.jerk
        with pytest.raises(AttributeError, match="no snap"):
            _ = j.snap
        with pytest.raises(AttributeError, match="no crackle"):
            _ = j.crackle
        with pytest.raises(AttributeError, match="no pop"):
            _ = j.pop

    def test_order_zero_has_value_only(self):
        j = Jet(np.array([3.14]))
        assert j.order == 0
        assert j.value == pytest.approx(3.14)
        with pytest.raises(AttributeError, match="no velocity"):
            _ = j.velocity


# ─── numpy interop ────────────────────────────────────────────────────


class TestNumpyInterop:
    def test_array_cast_returns_underlying_data(self):
        j = Jet(np.stack([np.ones(3), 2 * np.ones(3)]))
        arr = np.asarray(j)
        assert arr.shape == (2, 3)
        assert np.shares_memory(arr, j.data)

    def test_array_cast_with_dtype_copies(self):
        j = Jet(np.array([[1.0, 2.0], [3.0, 4.0]]))
        arr = np.asarray(j, dtype=np.float32)
        assert arr.dtype == np.float32
        assert not np.shares_memory(arr, j.data)

    def test_arithmetic_via_array_interface(self):
        # Confirm numpy operators unwrap the jet via __array__.
        j = Jet(np.array([[1.0, 2.0], [3.0, 4.0]]))
        arr = np.asarray(j) * 2.0
        np.testing.assert_allclose(arr, [[2.0, 4.0], [6.0, 8.0]])


# ─── Transformations ──────────────────────────────────────────────────


class TestTransformations:
    def test_truncate_reduces_order(self):
        j = Jet.zeros(order=4, dim=3)
        t = j.truncate(2)
        assert t.order == 2
        assert t.dim == 3

    def test_truncate_to_same_order_is_identity_copy(self):
        j = Jet(np.array([[0.0, 1.0], [2.0, 3.0]]))
        t = j.truncate(1)
        np.testing.assert_array_equal(t.data, j.data)

    def test_truncate_invalid_raises(self):
        j = Jet.zeros(order=2)
        with pytest.raises(ValueError):
            j.truncate(5)
        with pytest.raises(ValueError):
            j.truncate(-1)

    def test_differentiate_shifts_derivatives(self):
        # Given J(y) = (y, ẏ, ÿ), then J(ẏ) = (ẏ, ÿ) — one row shorter.
        j = Jet(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        d = j.differentiate()
        assert d.order == 1
        np.testing.assert_array_equal(d[0], [1.0, 0.0])
        np.testing.assert_array_equal(d[1], [0.0, 1.0])

    def test_differentiate_order_zero_raises(self):
        j = Jet(np.array([42.0]))
        with pytest.raises(ValueError, match="order-0"):
            j.differentiate()


# ─── Display ──────────────────────────────────────────────────────────


class TestRepr:
    def test_scalar_repr_shows_values(self):
        j = Jet(np.array([0.0, 1.5, 2.0]))
        r = repr(j)
        assert "order=2" in r
        assert "dim=1" in r
        assert "1.5" in r

    def test_vector_repr_shows_row_arrays(self):
        j = Jet(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        r = repr(j)
        assert "order=1" in r
        assert "dim=3" in r
