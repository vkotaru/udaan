"""Reference feasibility: does a reference keep the airframe upright?"""

import logging

import numpy as np
import pytest

from udaan.flatness import (
    Jet,
    QuadrotorCsPayload,
    QuadrotorCsPayloadFlats,
    infeasible_intervals,
    is_upright,
    scan_reference,
)
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase
from udaan.utils.trajectory import SpiralTrajectory


def _spiral(speed):
    return SpiralTrajectory(start=np.array([0.0, 0.0, 1.5]), speed=speed)


class TestHelpers:
    def test_is_upright(self):
        tilt = lambda a: np.array(  # noqa: E731
            [[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]
        )
        assert is_upright(np.eye(3))
        assert is_upright(tilt(np.radians(89.0)))
        assert not is_upright(tilt(np.radians(91.0)))
        assert not is_upright([np.eye(3), tilt(np.pi)])
        assert not is_upright(None)

    def test_intervals(self):
        t = np.arange(10) * 0.5
        ok = [True, False, False, True, True, False, True, True, True, False]
        assert infeasible_intervals(t, ok) == [(0.5, 1.5), (2.5, 3.0), (4.5, 5.0)]
        assert infeasible_intervals(t, [True] * 10) == []

    def test_scan_counts_refused_samples(self):
        rep = scan_reference(lambda t: None if 0.1 <= t < 0.2 else np.eye(3), 0.5, dt=0.05)
        assert rep.infeasible_samples == 2
        assert rep.intervals == [pytest.approx((0.1, 0.2))]


class TestSpiral:
    """The 2 m/s spiral's reference turns the airframe over at the top of its loops; 1.5 m/s does not."""

    def test_flat_map_flags_inverted_samples(self):
        fm = QuadrotorCsPayload()
        tr = _spiral(2.0)
        flags = []
        for t in np.arange(3.0, 3.6, 0.005):
            ref, _ = fm.recover(
                QuadrotorCsPayloadFlats(x_L=Jet(tr.get_jet(t, 6)), psi=Jet(np.zeros(3)))
            )
            flags.append((ref.upright, float(np.asarray(ref.orientation)[2, 2])))
        assert any(not up for up, _ in flags)
        assert all(up == (z > 0.0) for up, z in flags)

    def test_two_metres_per_second_is_infeasible(self):
        rep = QuadrotorCsPayload().check_reference(_spiral(2.0).get_jet, 10.0)
        assert not rep.feasible
        assert len(rep.intervals) == 2
        (a0, a1), (b0, b1) = rep.intervals
        assert 3.0 < a0 < a1 < 3.5 and 5.5 < b0 < b1 < 6.0

    def test_one_and_a_half_metres_per_second_is_feasible(self):
        rep = QuadrotorCsPayload().check_reference(_spiral(1.5).get_jet, 12.0)
        assert rep.feasible
        assert rep.intervals == []


class TestSimulate:
    def test_drops_and_warns_once(self, caplog):
        tr = _spiral(2.0)
        expected = QuadrotorCsPayload().check_reference(tr.get_jet, 10.0)
        mdl = QuadrotorCsPayloadBase()
        mdl.payload_controller.setpoint = tr.get_full
        with caplog.at_level(logging.WARNING, logger="udaan.models.quadrotor_cspayload.base"):
            mdl.simulate(10.0, payload_position=np.array([0.0, 0.0, 1.5]))
        warnings = [r for r in caplog.records if "not upright" in r.getMessage()]
        assert len(warnings) == 1
        assert "[3.2" in warnings[0].getMessage()
        assert mdl.feedforward_drops == expected.infeasible_samples

    def test_feasible_run_is_silent(self, caplog):
        mdl = QuadrotorCsPayloadBase()
        mdl.payload_controller.setpoint = _spiral(1.5).get_full
        with caplog.at_level(logging.WARNING, logger="udaan.models.quadrotor_cspayload.base"):
            mdl.simulate(4.0, payload_position=np.array([0.0, 0.0, 1.5]))
        assert mdl.feedforward_drops == 0
        assert not [r for r in caplog.records if "not upright" in r.getMessage()]

    def test_counter_resets_per_run(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.payload_controller.setpoint = _spiral(2.0).get_full
        mdl.simulate(4.0, payload_position=np.array([0.0, 0.0, 1.5]))
        first = mdl.feedforward_drops
        assert first > 0
        mdl.payload_controller.setpoint = _spiral(1.5).get_full
        mdl.simulate(1.0, payload_position=np.array([0.0, 0.0, 1.5]))
        assert mdl.feedforward_drops == 0
