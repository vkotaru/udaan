"""``udaan run multi-quad`` and ``multi-quad-rigid`` fly M3 and M4 on MuJoCo.

They are aliases of ``multi-quad-payload`` and ``multi-quad-rigidbody``: same
options, same model, same start, and the MCP catalog lists each under both ids.
"""

import numpy as np
import pytest
from typer.testing import CliRunner

from udaan.cli import app
from udaan.mcp.catalog import ALIASES, SYSTEMS
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco
from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyMujoco

pytestmark = pytest.mark.mujoco


def _recorded(monkeypatch, cls):
    seen = {}

    def record(self, tf, **kwargs):
        seen["model"] = self
        seen["tf"] = tf
        seen["kwargs"] = kwargs

    monkeypatch.setattr(cls, "simulate", record)
    return seen


@pytest.mark.parametrize(
    ("command", "cls", "num_quads"),
    [
        ("multi-quad", MultiQuadCsPayloadMujoco, 4),
        ("multi-quad-rigid", MultiQuadRigidbodyMujoco, 3),
    ],
)
def test_the_alias_flies_the_mujoco_model(monkeypatch, command, cls, num_quads):
    seen = _recorded(monkeypatch, cls)
    result = CliRunner().invoke(
        app,
        ["run", command, "--no-render", "-n", str(num_quads), "-t", "1", "-p", "0.5,0,1"],
    )
    assert result.exit_code == 0, result.output
    model = seen["model"]
    assert type(model) is cls
    assert model.num_quadrotors == num_quads
    assert seen["tf"] == 1.0
    np.testing.assert_allclose(seen["kwargs"]["payload_position"], [0.5, 0.0, 1.0])


@pytest.mark.parametrize(("command", "default"), [("multi-quad", 3), ("multi-quad-rigid", 4)])
def test_the_alias_keeps_the_default_quadrotor_count(monkeypatch, command, default):
    cls = MultiQuadCsPayloadMujoco if command == "multi-quad" else MultiQuadRigidbodyMujoco
    seen = _recorded(monkeypatch, cls)
    result = CliRunner().invoke(app, ["run", command, "--no-render", "-t", "1"])
    assert result.exit_code == 0, result.output
    assert seen["model"].num_quadrotors == default


def test_generate_xml_multi_quad_writes_the_m3_asset(tmp_path):
    import mujoco

    alias, target = tmp_path / "alias.xml", tmp_path / "m3.xml"
    for command, path in (("multi-quad", alias), ("multi-quad-payload", target)):
        result = CliRunner().invoke(app, ["generate-xml", command, "-n", "2", "-o", str(path)])
        assert result.exit_code == 0, result.output
    assert alias.read_text() == target.read_text()
    assert mujoco.MjModel.from_xml_path(str(alias)).nbody > 1


@pytest.mark.parametrize(("alias", "target"), list(ALIASES.items()))
def test_the_catalog_lists_the_alias_as_its_target(alias, target):
    spec, original = SYSTEMS[alias], SYSTEMS[target]
    assert spec.builder == original.builder
    assert spec.params == original.params
    assert spec.label == original.label
