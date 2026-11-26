"""The smooth step every circle-type reference ramps with (``udaan.utils.trajectory.ramp``)."""

import numpy as np
import pytest

from udaan.utils.trajectory.ramp import (
    DEFAULT_RAMP_SMOOTHNESS,
    ramp_phase_jet,
    smooth_step_derivatives,
    smooth_step_integral,
    step_jet,
)

POINTS = (0.02, 0.1, 0.27, 0.5, 0.63, 0.9, 0.98)
ORDERS = (1, 5, 10, 21, 41)


def _richardson(fn, u, h):
    """Central difference of ``fn`` at ``u``, Richardson-extrapolated: error O(h⁴)."""

    def d(step):
        return (fn(u + step) - fn(u - step)) / (2.0 * step)

    return (4.0 * d(0.5 * h) - d(h)) / 3.0


def _peaks(m, order):
    grid = np.linspace(0.005, 0.995, 199)
    return np.abs(np.array([smooth_step_derivatives(v, order, m) for v in grid])).max(axis=0)


class TestDerivatives:
    @pytest.mark.parametrize("m", ORDERS)
    @pytest.mark.parametrize("u", POINTS)
    def test_each_row_is_the_derivative_of_the_row_below(self, m, u):
        order = min(m + 3, 24)
        scale = np.maximum(_peaks(m, order + 1), 1e-300)
        jet = smooth_step_derivatives(u, order, m)
        h = 1e-3 * min(u, 1.0 - u)
        for k in range(1, order + 1):
            fd = _richardson(lambda v, k=k: smooth_step_derivatives(v, k - 1, m)[k - 1], u, h)
            assert abs(jet[k] - fd) <= 1e-6 * scale[k]

    def test_order_five_is_the_degree_eleven_smoothstep(self):
        poly = np.polynomial.Polynomial([0, 0, 0, 0, 0, 0, 462, -1980, 3465, -3080, 1386, -252])
        for u in POINTS:
            expected, p = [], poly
            for _ in range(13):
                expected.append(p(u))
                p = p.deriv()
            np.testing.assert_allclose(
                smooth_step_derivatives(u, 12, 5), expected, atol=1e-9 * 2e10
            )

    @pytest.mark.parametrize("m", ORDERS)
    def test_continuous_through_order_m_at_both_ends(self, m):
        """Rows 1…m vanish at both ends and row m + 1 does not: the step is C^m exactly.

        Within ``ε`` of an end row ``r ≤ m`` is ``O(ε^{m+1-r})`` times row ``m + 1``.
        """
        eps = 1e-12
        for u in (eps, 1.0 - eps):
            jet = smooth_step_derivatives(u, m + 1, m)
            assert jet[m + 1] != 0.0
            assert np.all(np.abs(jet[1 : m + 1]) <= 1e-9 * abs(jet[m + 1]))

    @pytest.mark.parametrize("m", ORDERS)
    @pytest.mark.parametrize("u", POINTS)
    def test_symmetry(self, m, u):
        """``S(u) + S(1 - u) = 1``: odd derivatives agree, even ones (k ≥ 2) change sign."""
        order = min(m + 1, 20)
        a = smooth_step_derivatives(u, order, m)
        b = smooth_step_derivatives(1.0 - u, order, m)
        assert a[0] + b[0] == pytest.approx(1.0, abs=1e-13)
        sign = np.where(np.arange(order + 1) % 2 == 1, 1.0, -1.0)
        np.testing.assert_allclose(a[1:], (sign * b)[1:], atol=1e-11 * max(1.0, np.abs(a).max()))

    def test_constant_outside_the_step(self):
        np.testing.assert_array_equal(smooth_step_derivatives(0.0, 5, 3), np.zeros(6))
        np.testing.assert_array_equal(smooth_step_derivatives(1.0, 5, 3), [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(smooth_step_derivatives(-0.5, 3, 3), np.zeros(4))

    def test_step_jet_is_in_time(self):
        u, T = 0.37, 4.0
        np.testing.assert_allclose(
            step_jet(u * T, T, 6, 4),
            smooth_step_derivatives(u, 6, 4) / T ** np.arange(7),
            rtol=1e-14,
        )

    def test_refuses_a_negative_or_fractional_order(self):
        for bad in (-1, 2.5):
            with pytest.raises(ValueError, match="smoothness"):
                smooth_step_derivatives(0.3, 2, bad)


class TestPhase:
    @pytest.mark.parametrize("m", ORDERS)
    def test_covers_half_its_full_rate_phase(self, m):
        assert smooth_step_integral(1.0, m) == pytest.approx(0.5, abs=1e-15)
        assert smooth_step_integral(2.5, m) == pytest.approx(2.0, abs=1e-15)
        assert smooth_step_integral(-1.0, m) == 0.0

    @pytest.mark.parametrize("m", ORDERS)
    @pytest.mark.parametrize("u", POINTS)
    def test_its_derivative_is_the_step(self, m, u):
        fd = _richardson(lambda v: smooth_step_integral(v, m), u, 1e-3 * min(u, 1.0 - u))
        assert fd == pytest.approx(smooth_step_derivatives(u, 0, m)[0], abs=1e-10)

    def test_ramp_phase_jet(self):
        w, T, k = 1.7, 3.0, 9
        for t in (-1.0, 0.4, 1.5, 2.9, 3.0, 7.0):
            jet = ramp_phase_jet(t, w, T, 8, k)
            assert jet[0] == pytest.approx(w * T * smooth_step_integral(t / T, k - 1), abs=1e-14)
            np.testing.assert_allclose(jet[1:], w * step_jet(t, T, 7, k - 1), rtol=1e-15)
        assert ramp_phase_jet(7.0, w, T, 0, k)[0] == pytest.approx(w * (7.0 - 1.5), abs=1e-14)
        assert DEFAULT_RAMP_SMOOTHNESS == 6
