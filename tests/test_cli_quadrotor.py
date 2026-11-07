"""``udaan run quadrotor``: where each trajectory is flown.

The default start ``x0 = (1, 1, 0)`` sits on the ground, so every trajectory
flown *at* ``x0`` must place its reference clear of the floor and start the
vehicle on it. ``simulate`` is replaced by a recorder, so these tests read the
reference the CLI bound and never fly it.
"""

import numpy as np
import pytest
from typer.testing import CliRunner

from udaan.cli import app
from udaan.models.quadrotor import QuadrotorBase


@pytest.fixture
def recorded(monkeypatch):
    seen = {}

    def record(self, tf, **kwargs):
        seen["setpoint"] = self.position_controller.setpoint
        seen["tf"] = tf
        seen["start"] = np.asarray(kwargs["position"], dtype=float)

    monkeypatch.setattr(QuadrotorBase, "simulate", record)
    return seen


def _run(traj):
    result = CliRunner().invoke(
        app, ["run", "quadrotor", "-m", "base", "--no-render", "--traj", traj, "-t", "6"]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("traj", ["circle", "spiral", "lissajous", "flip"])
def test_reference_stays_clear_of_the_ground(recorded, traj):
    _run(traj)
    heights = [recorded["setpoint"](t)[0][2] for t in np.linspace(0.0, recorded["tf"], 400)]
    assert min(heights) > 0.5


@pytest.mark.parametrize("traj", ["circle", "spiral", "lissajous", "flip"])
def test_vehicle_starts_on_its_reference(recorded, traj):
    _run(traj)
    np.testing.assert_allclose(recorded["start"], recorded["setpoint"](0.0)[0], atol=1e-9)


@pytest.mark.parametrize("traj", ["step", "figure8"])
def test_step_and_figure8_stay_clear_of_the_ground(recorded, traj):
    _run(traj)
    heights = [recorded["setpoint"](t)[0][2] for t in np.linspace(0.0, recorded["tf"], 400)]
    assert min(heights) > 0.5


@pytest.mark.parametrize("traj", ["step", "mission", "figure8"])
def test_step_mission_figure8_start_on_the_vehicle(recorded, traj):
    _run(traj)
    np.testing.assert_allclose(recorded["start"], recorded["setpoint"](0.0)[0], atol=1e-9)


def test_mission_takes_off_from_and_lands_on_the_start(recorded):
    _run("mission")
    np.testing.assert_allclose(
        recorded["setpoint"](recorded["tf"])[0], recorded["start"], atol=1e-9
    )
    assert max(recorded["setpoint"](t)[0][2] for t in np.linspace(0, recorded["tf"], 400)) > 2.0


def test_figure8_binds_its_yaw(monkeypatch):
    seen = {}

    def record(self, tf, **kwargs):
        seen["yaw"] = self.yaw_setpoint

    monkeypatch.setattr(QuadrotorBase, "simulate", record)
    _run("figure8")
    assert (
        seen["yaw"] is not None and seen["yaw"].__self__.__class__.__name__ == "Figure8Trajectory"
    )


@pytest.mark.parametrize("traj", ["step", "mission", "figure8"])
def test_step_mission_figure8_fly_end_to_end(traj):
    """Smoke test: the real simulate() on the base model, headless."""
    result = CliRunner().invoke(
        app, ["run", "quadrotor", "-m", "base", "--no-render", "--traj", traj, "-t", "4"]
    )
    assert result.exit_code == 0, result.output
    assert f"base, {traj}" in result.output


def test_unknown_trajectory_lists_the_choices():
    result = CliRunner().invoke(
        app, ["run", "quadrotor", "-m", "base", "--no-render", "--traj", "nope"]
    )
    assert result.exit_code == 1
    for name in ("step", "mission", "figure8"):
        assert name in result.output
