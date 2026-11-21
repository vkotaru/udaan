"""M7 / M8 flat map: M6's map with the anchor as a vertex-0 vehicle, and its references."""

from fractions import Fraction
from math import comb, factorial

import numpy as np
import pytest

from udaan.flatness.tethered_hose import (
    TetherBlend,
    TetherCircle,
    TetheredHose,
    TetherHold,
    circle_rate_bound,
    smoothstep_jet,
)
from udaan.models.tethered_hose import TetheredHoseBase, TetheredQuadrotorBase
from udaan.utils.hose import tethered_vehicle_positions


def _exact_smoothstep(u: Fraction, m: int, order: int, T: Fraction) -> list[float]:
    """``I_u(m + 1, m + 1)`` and its time derivatives in exact rational arithmetic."""
    beta = Fraction(factorial(m) ** 2, factorial(2 * m + 1))
    coefficients = {
        m + i + 1: Fraction(comb(m, i) * (-1) ** i, m + i + 1) / beta for i in range(m + 1)
    }
    out = []
    for r in range(order + 1):
        value = sum(
            (
                c * Fraction(factorial(p), factorial(p - r)) * u ** (p - r)
                for p, c in coefficients.items()
                if p >= r
            ),
            Fraction(0),
        )
        out.append(float(value / T**r))
    return out


class TestSmoothstep:
    @pytest.mark.parametrize("m", [10, 22, 42])
    @pytest.mark.parametrize(
        "u", [Fraction(1, 50), Fraction(3, 10), Fraction(1, 2), Fraction(97, 100)]
    )
    def test_matches_exact_arithmetic(self, m, u):
        exact = np.array(_exact_smoothstep(u, m, m, Fraction(6)))
        approx = smoothstep_jet(float(u) * 6.0, 6.0, m, m)
        scale = np.abs(exact).max()
        assert np.abs(approx - exact).max() < 1e-9 * scale

    def test_is_constant_outside_the_step(self):
        np.testing.assert_array_equal(smoothstep_jet(-1.0, 6.0, 5, 5), np.zeros(6))
        np.testing.assert_array_equal(smoothstep_jet(7.0, 6.0, 5, 5), [1, 0, 0, 0, 0, 0])


class TestMap:
    @pytest.mark.parametrize("cls", [TetheredQuadrotorBase, TetheredHoseBase])
    def test_a_constant_flat_output_is_the_trim(self, cls):
        m = cls()
        fm = TetheredHose.from_model(m)
        top = np.array([0.5, 0.0, 0.8]) * m.hose_length
        trim = fm.trim(top)
        ref, inputs = fm.recover(fm.constant_flats(trim.flat_tensions))
        np.testing.assert_allclose(
            np.array(ref.positions),
            tethered_vehicle_positions(top, m.num_quadrotors, m.anchor),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            inputs.thrusts, np.linalg.norm(trim.thrust_vectors, axis=1), atol=1e-12
        )
        np.testing.assert_allclose(inputs.anchor_force, trim.anchor_force, atol=1e-12)
        np.testing.assert_allclose(ref.root_position, m.anchor)

    def test_jet_orders(self):
        fm = TetheredHose(num_links=6, attachments=(2, 4, 6), hose_length=3.0, hose_mass=0.75)
        assert fm.jet_order == 14
        assert fm.tension_jet_orders == (14, 10, 6)

    def test_a_vehicle_at_the_anchor_is_refused(self):
        with pytest.raises(ValueError, match="anchor"):
            TetheredHose(num_links=4, attachments=(0, 4))


class TestReferences:
    @pytest.mark.slow
    def test_the_whirl_feed_forward_flies_the_circle_open_loop(self):
        """The recovered wrenches alone keep the vehicle on the steady whirl."""
        m = TetheredQuadrotorBase(dt=0.001, inner_loop_steps=10)
        circle = TetherCircle(m.hose_controller.flat_map, [1.0, 0.0, 1.6], rate=1.0, ramp_time=6.0)
        ref, _ = circle.recover(6.0)
        m.reset(
            link_attitudes=ref.link_attitudes,
            link_angular_velocities=ref.link_angular_velocities,
            orientations=ref.orientations,
            angular_velocities=ref.angular_velocities,
        )
        t, worst = 6.0, 0.0
        while t < 8.0 - 1e-9:
            m.step(circle.recover(t)[1].as_wrench())
            t += m.dt
            worst = max(
                worst,
                float(np.linalg.norm(m.state.positions[-1] - circle.recover(t)[0].positions[-1])),
            )
        assert worst < 1e-4

    def test_the_circle_starts_at_rest_and_ends_on_the_whirl(self):
        fm = TetheredHose()
        circle = TetherCircle(fm, [1.0, 0.0, 1.6], rate=1.5, ramp_time=6.0)
        start, _ = circle.recover(0.0)
        assert np.abs(np.array(start.velocities)).max() < 1e-12
        np.testing.assert_allclose(start.positions[-1], [1.0, 0.0, 1.6], atol=1e-12)
        for t in (6.0, 9.3):
            ref, _ = circle.recover(t)
            phi = circle.azimuth(t)
            np.testing.assert_allclose(
                ref.positions[-1], [np.cos(phi), np.sin(phi), 1.6], atol=1e-9
            )
            assert np.linalg.norm(ref.velocities[-1]) == pytest.approx(1.5, rel=1e-9)

    def test_every_jet_row_is_the_derivative_of_the_one_before(self):
        fm = TetheredHose()
        circle = TetherCircle(fm, [1.0, 0.0, 1.6], rate=1.5, ramp_time=6.0)
        h = 1e-5
        for t in (1.3, 4.7):
            jets = circle.tension_jets(t)
            plus, minus = circle.tension_jets(t + h), circle.tension_jets(t - h)
            for k in range(len(jets[0]) - 1):
                numeric = (plus[0][k] - minus[0][k]) / (2 * h)
                assert np.abs(numeric - jets[0][k + 1]).max() < 1e-4 * max(
                    1.0, np.abs(jets[0][k + 1]).max()
                )

    def test_a_blend_ends_at_the_new_trim(self):
        fm = TetheredHose()
        blend = TetherBlend(fm, [1.0, 0.0, 1.6], [0.8, 0.0, 1.75], 6.0, start_time=1.0)
        np.testing.assert_allclose(blend.get(0.5)[0], [1.0, 0.0, 1.6], atol=1e-12)
        end, _ = blend.recover(7.5)
        np.testing.assert_allclose(end.positions[-1], [0.8, 0.0, 1.75], atol=1e-12)
        assert np.abs(np.array(end.velocities)).max() < 1e-12

    def test_a_hold_is_static(self):
        hold = TetherHold(TetheredHose(), [1.0, 0.0, 1.6])
        pos, vel, acc = hold.get(3.0)
        np.testing.assert_allclose(pos, [1.0, 0.0, 1.6], atol=1e-12)
        assert np.abs(vel).max() < 1e-12 and np.abs(acc).max() < 1e-12

    def test_the_rate_bound(self):
        fm = TetheredHose()
        assert circle_rate_bound(fm, [1.0, 0.0, 1.6]) == pytest.approx(0.5 * np.sqrt(9.81 / 0.5))

    def test_the_rate_bound_leaves_one_newton_of_thrust(self):
        """Where thrust binds, the top vehicle's steady circle asks for 95 % of 20 N, not all of it."""
        heavy = TetheredHose(
            num_links=10, attachments=(2, 4, 6, 8, 10), hose_length=20.0, hose_mass=5.0
        )
        rate = circle_rate_bound(heavy, [10.0, 0.0, 16.0])
        peak = np.linalg.norm(
            heavy.trim([10.0, 0.0, 16.0], whirl_rate=rate).thrust_vectors, axis=1
        ).max()
        assert rate < 0.5 * np.sqrt(9.81 / 2.0)
        assert peak == pytest.approx(19.0, abs=1e-6)
