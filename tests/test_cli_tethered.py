"""``udaan run tethered-quad`` (M7) and ``udaan run tethered-hose`` (M8), headless."""

import pytest
from typer.testing import CliRunner

from udaan.cli import app


@pytest.mark.parametrize("command", ["tethered-quad", "tethered-hose"])
@pytest.mark.parametrize("trajectory", ["hold", "circle", "climb"])
def test_flies_on_the_analytical_model(command, trajectory):
    result = CliRunner().invoke(
        app, ["run", command, "-m", "base", "--no-render", "-t", "0.5", "--traj", trajectory]
    )
    assert result.exit_code == 0, result.output
    assert "top vehicle error" in result.output


@pytest.mark.mujoco
def test_flies_on_mujoco():
    result = CliRunner().invoke(
        app, ["run", "tethered-hose", "--no-render", "-t", "0.5", "-q", "2"]
    )
    assert result.exit_code == 0, result.output


def test_an_unknown_trajectory_exits_1():
    result = CliRunner().invoke(app, ["run", "tethered-quad", "-m", "base", "--traj", "loop"])
    assert result.exit_code == 1
    assert "Unknown trajectory" in result.output
