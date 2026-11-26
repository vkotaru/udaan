"""``udaan run fleet`` and ``udaan run cspayload-fleet``: lanes and gains.

Every vehicle of a side-by-side fleet spawns at its own x-offset, so each one
is given the shared target translated into that lane; otherwise the vehicles
converge on one point. The cable-gain sweep scales each agent's own gains.
"""

import numpy as np
import pytest
from typer.testing import CliRunner

from udaan.cli import app
from udaan.cli.demos import CSPAYLOAD_DEMOS, DEMOS
from udaan.models.mujoco import QuadrotorCsPayloadFleet, QuadrotorFleet

pytestmark = pytest.mark.mujoco


def _fleet_for(demo):
    params = DEMOS[demo]["config"]()
    fleet = QuadrotorFleet(
        num_quadrotors=params["num_quadrotors"],
        render=False,
        disturbances=params["disturbances"],
    )
    DEMOS[demo]["configure"](fleet)
    return fleet


@pytest.mark.parametrize("demo", ["l1-comparison", "gain-sweep"])
def test_fleet_demo_targets_lie_in_each_vehicles_lane(demo):
    fleet = _fleet_for(demo)
    targets = np.array([q.position_controller.setpoint(0.0)[0] for q in fleet.quadrotors])
    np.testing.assert_allclose(targets[:, 0], fleet._offsets)
    np.testing.assert_allclose(targets[:, 1], 0.0)


@pytest.mark.integration
def test_l1_comparison_vehicles_stay_apart():
    fleet = _fleet_for("l1-comparison")
    fleet.simulate(tf=4.0, position=np.array([0.0, 0.0, 1.0]))
    pos = np.array([q.state.position for q in fleet.quadrotors])
    gaps = [np.linalg.norm(pos[i] - pos[j]) for i in range(len(pos)) for j in range(i)]
    assert min(gaps) > 1.0


def _recorded_fleet(monkeypatch, cls):
    seen = {}

    def record(self, tf, **kwargs):
        seen["fleet"] = self
        seen["kwargs"] = kwargs

    monkeypatch.setattr(cls, "simulate", record)
    return seen


def test_fleet_default_run_holds_each_vehicle_in_its_lane(monkeypatch):
    seen = _recorded_fleet(monkeypatch, QuadrotorFleet)
    result = CliRunner().invoke(app, ["run", "fleet", "--no-render", "-n", "3", "-t", "1"])
    assert result.exit_code == 0, result.output
    fleet = seen["fleet"]
    targets = np.array([q.position_controller.setpoint(0.0)[0] for q in fleet.quadrotors])
    np.testing.assert_allclose(targets, [[dx, 0.0, 2.0] for dx in fleet._offsets])


def test_cspayload_fleet_default_run_starts_and_ends_in_each_lane(monkeypatch):
    seen = _recorded_fleet(monkeypatch, QuadrotorCsPayloadFleet)
    result = CliRunner().invoke(
        app, ["run", "cspayload-fleet", "--no-render", "-n", "3", "-t", "1"]
    )
    assert result.exit_code == 0, result.output
    fleet = seen["fleet"]
    targets = np.array([fleet[i]._payload_controller.setpoint(0.0)[0] for i in range(fleet.nQ)])
    starts = np.asarray(seen["kwargs"]["payload_positions"])
    np.testing.assert_allclose(targets, [[dx, 0.0, 1.5] for dx in fleet._offsets])
    assert np.all(np.abs(starts[:, 0] - fleet._offsets) <= 0.5)


def test_cspayload_gain_sweep_scales_each_agents_own_cable_gains():
    nominal = QuadrotorCsPayloadFleet(num_agents=4, render=False)
    design = [nominal[i]._payload_controller.cable_gains for i in range(4)]
    fleet = QuadrotorCsPayloadFleet(num_agents=4, render=False)
    CSPAYLOAD_DEMOS["gain-sweep"]["configure"](fleet)
    for i, scale in enumerate([0.5, 1.0, 1.5, 2.0]):
        kp, kd = fleet[i]._payload_controller.cable_gains
        np.testing.assert_allclose(kp, scale * design[i][0])
        np.testing.assert_allclose(kd, scale * design[i][1])


def test_cspayload_fleet_agents_carry_the_mjcf_machine():
    """Each agent's model and controller use the MJCF airframe and the hanging mass."""
    import mujoco

    fleet = QuadrotorCsPayloadFleet(num_agents=2, render=False)
    m = fleet._mjMdl.model
    quad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "agent0_quad")
    hanging = sum(
        m.body_mass[b]
        for b in range(m.nbody)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith("agent0_cbl")
    )
    assert fleet[0].mass == pytest.approx(m.body_mass[quad])
    assert fleet[0].payload_mass == pytest.approx(hanging)


@pytest.mark.integration
def test_cspayload_fleet_payloads_reach_their_targets():
    fleet = QuadrotorCsPayloadFleet(num_agents=2, render=False)
    starts = CSPAYLOAD_DEMOS["same-gains"]["configure"](fleet)
    fleet.simulate(tf=8.0, payload_positions=starts)
    for i in range(fleet.nQ):
        target = fleet[i]._payload_controller.setpoint(fleet.t)[0]
        assert np.linalg.norm(fleet[i].state.payload_position - target) < 0.03


def test_fleet_gain_sweep_scales_the_default_position_gains():
    from udaan.core.defaults import DEFAULT_POS_KD, DEFAULT_POS_KP

    fleet = _fleet_for("gain-sweep")
    for q, scale in zip(fleet.quadrotors, [0.25, 0.5, 1.0, 2.0], strict=True):
        np.testing.assert_allclose(q.position_controller._gains.kp, scale * DEFAULT_POS_KP)
        np.testing.assert_allclose(q.position_controller._gains.kd, scale * DEFAULT_POS_KD)


def test_cspayload_gain_sweep_survives_a_controller_resync():
    fleet = QuadrotorCsPayloadFleet(num_agents=4, render=False)
    CSPAYLOAD_DEMOS["gain-sweep"]["configure"](fleet)
    before = [fleet[i]._payload_controller.cable_gains[0] for i in range(4)]
    for i in range(4):
        fleet[i]._payload_controller.sync_from_model(fleet[i])
        np.testing.assert_allclose(fleet[i]._payload_controller.cable_gains[0], before[i])
