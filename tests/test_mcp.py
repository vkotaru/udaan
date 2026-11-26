"""The MCP server: catalog coverage, headless runs per system, validation, client round-trips.

Skipped without the ``mcp`` extra (``pip install "udaan[mcp]"``).
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import sys
import time

import numpy as np
import pytest

pytest.importorskip("mcp.server.mcpserver")

import anyio  # noqa: E402  (installed with mcp)
from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from udaan.cli import scenarios  # noqa: E402
from udaan.cli.generate import generate_app  # noqa: E402
from udaan.cli.run import run_app  # noqa: E402
from udaan.mcp import catalog, runner  # noqa: E402
from udaan.mcp.catalog import SYSTEMS, ValidationError  # noqa: E402
from udaan.mcp.runner import RunError  # noqa: E402
from udaan.mcp.server import build_server  # noqa: E402

#: Builder keywords the server sets itself rather than taking as ``params``.
_SERVER_SET = {"backend", "trajectory", "duration", "render", "verbose"}

_SIMULATED = [s for s in SYSTEMS.values() if s.builder is not None]


def _command_names(app) -> set[str]:
    return {c.name for c in app.registered_commands}


# ─── Catalog ─────────────────────────────────────────────────────────


class TestCatalog:
    def test_every_run_command_is_a_system(self):
        assert _command_names(run_app) <= set(SYSTEMS)

    def test_every_generate_command_is_a_system(self):
        assert _command_names(generate_app) <= set(SYSTEMS)

    def test_labels_cover_m1_to_m8(self):
        labels = {s.label for s in SYSTEMS.values()}
        assert {f"M{k}" for k in range(1, 9)} <= labels

    @pytest.mark.parametrize("spec", _SIMULATED, ids=lambda s: s.id)
    def test_parameters_match_the_builder(self, spec):
        builder = getattr(scenarios, spec.builder)
        taken = set(inspect.signature(builder).parameters) - _SERVER_SET
        assert {p.name for p in spec.params} == taken

    @pytest.mark.parametrize("spec", _SIMULATED, ids=lambda s: s.id)
    def test_defaults_match_the_builder(self, spec):
        params = inspect.signature(getattr(scenarios, spec.builder)).parameters
        trajectory = params.get("trajectory")
        if trajectory is None:
            assert len(spec.trajectories) == 1
        else:
            assert spec.trajectories[0] == trajectory.default
        for p in spec.params:
            if p.default is not None:
                assert params[p.name].default == p.default, p.name

    def test_no_system_offers_vfx(self):
        assert all("vfx" not in s.backends for s in SYSTEMS.values())

    def test_descriptions_are_json(self):
        json.dumps([s.describe() for s in SYSTEMS.values()])


# ─── Validation ──────────────────────────────────────────────────────


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"system": "nope"}, "Unknown system"),
            ({"system": "quadrotor", "backend": "vfx"}, "vfx is never offered"),
            ({"system": "fleet", "backend": "base"}, "runs on: mujoco"),
            ({"system": "quadrotor", "trajectory": "loop"}, "flies:"),
            ({"system": "quadrotor", "duration": 0.0}, "duration must be > 0"),
            ({"system": "quadrotor", "duration": 1e4}, "duration must be <="),
            ({"system": "quadrotor", "duration": float("nan")}, "finite"),
            ({"system": "quadrotor", "duration": "5"}, "must be a number"),
            ({"system": "quad-flexcable", "params": {"num_links": 99}}, "num_links must be <="),
            ({"system": "hose", "params": {"num_links": 0}}, "num_links must be >="),
            ({"system": "multi-quad-payload", "params": {"num_quadrotors": 50}}, "<="),
            ({"system": "quadrotor", "params": {"mass": 1.0}}, "no parameter 'mass'"),
            ({"system": "quad-payload", "params": {"cable_model": "links"}}, "only to backend"),
            ({"system": "quadrotor", "params": {"position": [0, 0]}}, r"\[x, y, z\]"),
            ({"system": "quadrotor", "params": {"position": [0, 0, 1e6]}}, "100 m"),
            ({"system": "hose", "params": {"attachments": "0,4"}}, "list of integers"),
            ({"system": "fleet", "params": {"demo": "nope"}}, "must be one of"),
            ({"system": "cspayload-fleet", "params": {"same_start": 1}}, "true or false"),
            ({"system": "comparison"}, "MJCF only"),
        ],
    )
    def test_refused(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            catalog.validate(**kwargs)

    def test_defaults_fill_in(self):
        spec, kwargs = catalog.validate("quadrotor")
        assert kwargs == {"backend": "base", "trajectory": "hover", "duration": 10.0}
        _, kwargs = catalog.validate("fleet")
        assert kwargs == {"duration": 10.0}
        assert spec.default_backend == "base"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"timeout_s": 0.0}, "timeout_s"),
            ({"output_path": "/tmp/series.txt"}, r"\.csv or \.json"),
            ({"output_path": "/no/such/dir/series.csv"}, "does not exist"),
        ],
    )
    def test_simulate_refuses_before_running(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            runner.simulate("quadrotor", duration=0.1, **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fps": 0}, "fps"),
            ({"width": 10_000}, "width"),
            ({"output_path": "/tmp/clip.avi"}, r"\.mp4 or \.gif"),
            ({"output_path": "/tmp/clip.gif", "duration": 10.0, "fps": 30}, "GIF is capped"),
            ({"output_path": "/tmp/clip.gif", "width": 1280, "height": 720}, "GIF is at most"),
            ({"width": 641}, "even width and height"),
        ],
    )
    def test_render_video_refuses_before_running(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            runner.render_video("quadrotor", **kwargs)

    def test_a_mujoco_only_geometry_parameter_is_refused_on_base(self):
        with pytest.raises(ValidationError, match="only to backend mujoco"):
            catalog.validate("multi-quad-rigidbody", params={"attachment_radius": 1.0})

    @pytest.mark.parametrize(
        ("system", "params"),
        [
            ("quadrotor", {"position": [0, 0, 1]}),
            ("quad-payload", {"cable_model": "tendon", "cable_length": 2.0}),
            ("tethered-quad", {"standoff": 0.5}),
            ("comparison", {"num_links": 3}),
        ],
    )
    def test_generate_xml_refuses_what_does_not_shape_the_file(self, system, params):
        with pytest.raises(ValidationError):
            runner.generate_xml(system, params=params)

    def test_a_model_refusal_comes_back_as_a_validation_error(self):
        # A pattern without a vehicle on the last vertex is refused by the LQR scenario.
        with pytest.raises(ValidationError, match="no vehicle at the last vertex"):
            runner.simulate("hose", duration=0.1, params={"attachments": [0, 2]})


# ─── Runs ────────────────────────────────────────────────────────────


def _cases():
    for spec in _SIMULATED:
        if "base" in spec.backends:
            yield pytest.param(spec.id, "base", id=f"{spec.id}-base")
        else:
            yield pytest.param(spec.id, "mujoco", id=f"{spec.id}-mujoco", marks=pytest.mark.mujoco)


@pytest.mark.parametrize(("system", "backend"), list(_cases()))
def test_simulate_every_system(system, backend):
    m = runner.simulate(system, backend=backend, duration=0.3)
    assert m["completed"], m
    assert m["simulated_time"] == pytest.approx(0.3, abs=0.02)
    assert np.isfinite(m["max_tilt_deg"])
    assert m["tracking_error_m"]["max"] >= m["tracking_error_m"]["rms"] >= 0.0
    assert m["wall_time_s"] > 0.0


@pytest.mark.mujoco
@pytest.mark.parametrize("system", ["quadrotor", "multi-quad-payload", "tethered-hose"])
def test_simulate_on_mujoco(system):
    m = runner.simulate(system, backend="mujoco", duration=0.3)
    assert m["completed"], m
    assert m["label"].startswith("mujoco")


def test_cable_tension_is_reported_where_the_model_has_it():
    m = runner.simulate("multi-quad-payload", duration=0.3)
    assert m["min_cable_tension_n"] > 0.0
    assert "min_cable_tension_n" not in runner.simulate("quadrotor", duration=0.1)


def test_trajectory_and_params_reach_the_model():
    m = runner.simulate(
        "quad-flexcable", duration=0.2, trajectory="lissajous", params={"num_links": 2}
    )
    assert m["label"] == "base, 2 links, lissajous"
    assert m["feedforward_drops"] == 0


def test_timeseries_is_written(tmp_path):
    path = tmp_path / "run.csv"
    m = runner.simulate("quadrotor", duration=0.1, output_path=str(path))
    assert m["timeseries_path"] == str(path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == m["samples"] == 20
    assert {"t", "x0", "ref_x0", "error", "tilt_deg"} <= set(rows[0])


def test_timeseries_json(tmp_path):
    path = tmp_path / "run.json"
    runner.simulate("multi-quad-payload", duration=0.05, output_path=str(path))
    rows = json.loads(path.read_text())
    assert "min_tension" in rows[0]


def test_wall_clock_budget_stops_a_long_run():
    start = time.monotonic()
    m = runner.simulate("tethered-hose", duration=60.0, timeout_s=1.0)
    assert time.monotonic() - start < 30.0
    assert not m["completed"]
    assert m["stop_reason"] == "wall_budget"
    assert "wall-clock budget" in m["stopped"]


def test_a_child_past_the_parent_deadline_is_killed(monkeypatch):
    # The child's own deadline (600 s away) is far past the parent's (1 s), so only the kill
    # stops it.
    monkeypatch.setattr(runner, "KILL_GRACE", 0.0)
    _, kwargs = catalog.validate("tethered-hose", duration=60.0)
    payload = {"system": "tethered-hose", "kwargs": kwargs, "series_path": None}
    start = time.monotonic()
    with pytest.raises(RunError, match="did not finish within 1 s"):
        runner._in_child("simulate", {**payload, "deadline": time.time() + 600.0}, timeout=1.0)
    assert time.monotonic() - start < 15.0


@pytest.mark.mujoco
def test_generate_xml_copies_a_loadable_asset(tmp_path):
    import mujoco

    out = tmp_path / "hose.xml"
    r = runner.generate_xml("hose", params={"num_links": 3}, output_path=str(out))
    assert r["path"] == str(out)
    assert (tmp_path / "scene.xml").exists()
    mujoco.MjModel.from_xml_path(str(out))


@pytest.mark.mujoco
def test_generate_xml_text_and_packaged_assets():
    r = runner.generate_xml("quadrotor", include_text=True)
    assert r["xml"].lstrip().startswith("<")
    assert runner.generate_xml("comparison")["path"].endswith("quadrotor_comparison.xml")


class TestCheckReference:
    def test_the_helix_is_feasible(self):
        r = runner.check_reference("quad-payload", trajectory="spiral")
        assert r["feasible"]
        assert r["infeasible_intervals_s"] == []

    def test_a_constant_setpoint_is_refused(self):
        with pytest.raises(ValidationError, match="constant setpoint"):
            runner.check_reference("quad-payload", trajectory="hover")

    def test_systems_without_a_check_are_refused(self):
        with pytest.raises(ValidationError, match="available for: quad-payload"):
            runner.check_reference("quadrotor")


def _rendering_required() -> bool:
    """Whether a missing offscreen backend is a failure rather than a skip.

    The CI test job installs OSMesa and sets ``MUJOCO_GL=osmesa``, so on CI
    (``CI`` set) or wherever OSMesa is asked for explicitly, a video test that
    cannot render reports a broken setup. Elsewhere it skips.
    """
    return bool(os.environ.get("CI")) or os.environ.get("MUJOCO_GL", "").lower() == "osmesa"


@pytest.fixture(scope="module")
def offscreen_rendering():
    """Skip unless MuJoCo can render offscreen here; fail instead where it must."""
    error = runner.check_offscreen_rendering()
    if error is None:
        return
    if _rendering_required():
        pytest.fail(f"offscreen rendering is required here but unavailable: {error}")
    pytest.skip(f"offscreen rendering unavailable on this machine: {error}")


@pytest.mark.mujoco
def test_render_video(tmp_path, offscreen_rendering):
    path = tmp_path / "clip.gif"
    r = runner.render_video("quadrotor", duration=0.5, fps=10, output_path=str(path))
    assert r["frames"] == 6  # one per 0.1 s, from the first step at 5 ms to 0.5 s
    assert path.stat().st_size > 0
    assert r["gl"] in ("egl", "osmesa", "cgl")  # never glfw, which opens a window
    if os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
        assert r["gl"] == "osmesa"


class TestOffscreenUnavailable:
    """An unusable OpenGL backend is reported with the remedy on any machine.

    ``MUJOCO_GL=bogus`` makes ``import mujoco`` raise in the child, as a
    missing libEGL does on a headless CI runner.
    """

    @pytest.fixture
    def bogus_gl(self, monkeypatch):
        monkeypatch.setitem(runner.OFFSCREEN_GL, sys.platform, ("bogus",))
        monkeypatch.setattr(runner, "_gl_checked", {})

    def test_the_check_reports_the_backend_and_the_cause(self, bogus_gl):
        error = runner.check_offscreen_rendering()
        assert error is not None
        assert "Offscreen rendering is unavailable (MUJOCO_GL=bogus)" in error
        assert "invalid value for environment variable MUJOCO_GL" in error
        assert "libegl1" in error

    def test_render_video_refuses_before_running(self, bogus_gl, tmp_path):
        with pytest.raises(RunError, match=r"Offscreen rendering is unavailable \(MUJOCO_GL=bogus"):
            runner.render_video("quadrotor", duration=0.1, output_path=str(tmp_path / "a.mp4"))

    def test_a_video_child_that_dies_on_import_gets_the_same_message(self, bogus_gl, monkeypatch):
        # The check passing but the child failing is the CI case: the child died while
        # importing MuJoCo and used to report only its exit code.
        monkeypatch.setattr(runner, "check_offscreen_rendering", lambda gl=None: None)
        with pytest.raises(RunError, match="invalid value for environment variable MUJOCO_GL"):
            runner.render_video("quadrotor", duration=0.1)


def test_a_child_that_dies_on_import_forwards_its_traceback():
    _, kwargs = catalog.validate("quadrotor", duration=0.1)
    payload = {"system": "quadrotor", "kwargs": kwargs, "deadline": None, "series_path": None}
    with pytest.raises(RunError, match=r"exited with code 1 .*: RuntimeError: invalid value"):
        runner._in_child("simulate", payload, timeout=60.0, env={"MUJOCO_GL": "bogus"})


# ─── MCP round-trips ─────────────────────────────────────────────────

_TOOLS = {
    "list_systems",
    "describe_system",
    "simulate",
    "render_video",
    "generate_xml",
    "check_reference",
}


def test_in_memory_client_round_trip():
    async def main():
        async with Client(build_server()) as client:
            tools = await client.list_tools()
            listed = await client.call_tool("list_systems", {})
            ok = await client.call_tool("simulate", {"system": "quadrotor", "duration": 0.2})
            bad = await client.call_tool("simulate", {"system": "quadrotor", "backend": "vfx"})
            return tools, listed, ok, bad

    tools, listed, ok, bad = anyio.run(main)
    assert {t.name for t in tools.tools} == _TOOLS
    assert {s["id"] for s in listed.structured_content["systems"]} == set(SYSTEMS)
    assert not ok.is_error
    assert ok.structured_content["completed"]
    assert bad.is_error
    assert "vfx is never offered" in bad.content[0].text


def test_stdio_round_trip():
    params = StdioServerParameters(command=sys.executable, args=["-m", "udaan.mcp"])

    async def main():
        async with Client(params) as client:
            described = await client.call_tool("describe_system", {"system": "hose"})
            run = await client.call_tool("simulate", {"system": "hose", "duration": 0.2})
            return described, run

    described, run = anyio.run(main)
    assert described.structured_content["label"] == "M6"
    assert run.structured_content["completed"]


def test_main_without_the_sdk_names_the_extra(monkeypatch, capsys):
    import udaan.mcp

    monkeypatch.setattr(udaan.mcp, "MCP_AVAILABLE", False)
    with pytest.raises(SystemExit) as exc:
        udaan.mcp.main()
    assert exc.value.code == 1
    assert 'pip install "udaan[mcp]"' in capsys.readouterr().err
