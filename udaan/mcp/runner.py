"""Headless runs behind the MCP tools: simulate, record a video, write an MJCF, check a reference.

Every run executes in a fresh child Python process, so that a run past its
wall-clock budget can be killed outright, a crash cannot take the server down,
and nothing the model prints reaches the server's stdout, which is the MCP
stdio channel. The child reads its job from stdin and writes its result to a
private copy of its stdout; its own stdout and stderr go to a pipe the parent
reads, so a child that dies, even while importing, is reported with the last
line of its traceback. At most :data:`MAX_CONCURRENT_RUNS` children run at once.

Nothing here opens a window: every model is built with ``render=False``, and
video frames come from ``mujoco.Renderer`` on an offscreen OpenGL context (see
:data:`OFFSCREEN_GL`). Before a video run the parent checks, once per backend,
that such a context can be made, and refuses the call with the remedy if not.
The ``vfx`` backend is never reachable.

This module does not need the ``mcp`` SDK. A call that its tool's client
cancels is not stopped: its child runs until it finishes or its budget ends.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .catalog import MAX_DURATION, ValidationError, get_system, validate

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from udaan.cli.scenarios import Scenario

#: Default wall-clock budget of one run, in s.
DEFAULT_TIMEOUT = 120.0
#: Largest wall-clock budget a call may ask for, in s.
MAX_TIMEOUT = 1800.0
#: Runs allowed at once; further calls wait for a slot (up to their own timeout).
MAX_CONCURRENT_RUNS = 2
#: Time a child has past its budget to return its partial result before it is killed, in s.
KILL_GRACE = 30.0
#: Largest video frame, in pixels.
MAX_VIDEO_SIZE = (1920, 1080)
#: Largest video frame rate, in frames per s.
MAX_FPS = 60
#: GIF frames are held in memory until written, so a GIF is capped at this many
#: frames and at :data:`MAX_GIF_SIZE`.
MAX_GIF_FRAMES = 300
#: Largest GIF frame, in pixels.
MAX_GIF_SIZE = (640, 480)
#: Offscreen OpenGL backends a video may use, per platform (``MUJOCO_GL``); the
#: first is the default. ``glfw``, MuJoCo's own default, opens a window.
OFFSCREEN_GL = {"linux": ("egl", "osmesa"), "darwin": ("cgl",)}
#: Characters of MJCF text ``generate_xml`` returns inline at most.
MAX_XML_TEXT = 200_000
#: Wall time the offscreen-rendering check may take, in s.
GL_PROBE_TIMEOUT = 60.0

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)
#: Marks the start of the pickled result on the child's result stream.
_RESULT_MARK = b"\n--udaan-mcp-result--\n"
#: Characters of a dead child's output kept for the error message.
_TAIL = 4000

#: Offscreen-rendering check results by backend: ``None`` (usable) or the error.
_gl_checked: dict[str, str | None] = {}
_gl_lock = threading.Lock()

#: Run by the offscreen-rendering check: import MuJoCo and draw one frame.
_GL_PROBE = """\
import mujoco
model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
renderer = mujoco.Renderer(model, height=16, width=16)
renderer.update_scene(mujoco.MjData(model))
renderer.render()
"""


class RunError(RuntimeError):
    """A run that could not complete; the message says why and what to change."""


def _last_line(text: str) -> str:
    """The last non-empty line of ``text`` (a traceback's exception line)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "no output"


def gl_unavailable_message(gl: str, detail: str) -> str:
    """The error for an offscreen OpenGL backend that cannot be used, with the remedy."""
    return (
        f"Offscreen rendering is unavailable (MUJOCO_GL={gl}): {detail}. Set MUJOCO_GL=egl "
        "or MUJOCO_GL=osmesa in the server's environment and install that library "
        "(Debian/Ubuntu: libegl1 for EGL, libosmesa6 for OSMesa); on macOS, CGL needs "
        "no install."
    )


def offscreen_backend() -> str:
    """The ``MUJOCO_GL`` a video run uses: the environment's, if offscreen, else the default.

    Raises:
        RunError: on a platform with no offscreen backend.
    """
    choices = OFFSCREEN_GL.get(sys.platform)
    if choices is None:
        raise RunError(f"render_video supports Linux (EGL, OSMesa) and macOS, not {sys.platform}.")
    gl = os.environ.get("MUJOCO_GL", "").lower()
    return gl if gl in choices else choices[0]


def check_offscreen_rendering(gl: str | None = None) -> str | None:
    """``None`` if MuJoCo can render offscreen with ``gl``, else the error with its remedy.

    Draws one frame in a separate Python process (MuJoCo picks its OpenGL
    backend at import), once per backend for the life of this process.
    ``gl`` defaults to :func:`offscreen_backend`.
    """
    gl = offscreen_backend() if gl is None else gl
    with _gl_lock:
        if gl in _gl_checked:
            return _gl_checked[gl]
        env = {**os.environ, "MUJOCO_GL": gl}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _GL_PROBE],
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=GL_PROBE_TIMEOUT,
            )
            error = (
                None
                if proc.returncode == 0
                else gl_unavailable_message(gl, _last_line(proc.stderr + proc.stdout))
            )
        except subprocess.TimeoutExpired:
            error = gl_unavailable_message(
                gl, f"drawing one frame did not finish within {GL_PROBE_TIMEOUT:g} s"
            )
        _gl_checked[gl] = error
        return error


# ─── Parent side: the child process ──────────────────────────────────


def _check_timeout(timeout: float | None) -> float:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValidationError(f"timeout_s must be a number, got {timeout!r}.")
    if not 1.0 <= timeout <= MAX_TIMEOUT:
        raise ValidationError(f"timeout_s must lie in [1, {MAX_TIMEOUT:g}] s, got {timeout!r}.")
    return float(timeout)


def _in_child(
    job: str, payload: dict[str, Any], timeout: float, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run ``job`` in a child process and return its result, or raise :class:`RunError`.

    ``timeout`` s is the call's budget, waiting for a free slot included. A
    payload with ``deadline=None`` gets what is left of it after the wait, as
    an absolute ``time.time()``, so the child can stop at the deadline and
    return partial results; a child still running :data:`KILL_GRACE` s later
    is killed. ``env`` is added to the child's environment, which is where
    ``MUJOCO_GL`` has to be: MuJoCo reads it once, at import.
    """
    start = time.monotonic()
    if not _slots.acquire(timeout=timeout):
        raise RunError(
            f"{MAX_CONCURRENT_RUNS} runs are already in progress and none finished within "
            f"{timeout:g} s; retry later."
        )
    try:
        remaining = max(1.0, timeout - (time.monotonic() - start))
        if "deadline" in payload and payload["deadline"] is None:
            payload = {**payload, "deadline": time.time() + remaining}
        proc = subprocess.Popen(
            [sys.executable, "-c", "from udaan.mcp.runner import _child_main; _child_main()"],
            env={**os.environ, **(env or {})},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = proc.communicate(
                pickle.dumps((job, payload)), timeout=remaining + KILL_GRACE
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RunError(
                f"The run did not finish within {timeout:g} s of wall time and was "
                "stopped. Shorten the duration or raise timeout_s."
            ) from None
    finally:
        _slots.release()

    log = err.decode(errors="replace")
    if log.strip():
        sys.stderr.write(log[-_TAIL:])
    head, mark, body = out.rpartition(_RESULT_MARK)
    if not mark:
        detail = _last_line(log + head.decode(errors="replace"))
        if job == "video" and env and "MUJOCO_GL" in env:
            raise RunError(gl_unavailable_message(env["MUJOCO_GL"], detail))
        raise RunError(
            f"The simulation process exited with code {proc.returncode} before returning a "
            f"result: {detail}"
        )
    result: dict[str, Any] = pickle.loads(body)
    if "error" in result:
        kind = result.get("kind")
        if kind == "validation":
            raise ValidationError(result["error"])
        raise RunError(result["error"])
    return result


def _child_main() -> None:
    """Child entry point: read ``(job, payload)`` from stdin, write the result dict back."""
    # Anything a model or a library prints goes to stderr; the result goes to a
    # private copy of the original stdout, after a marker.
    result_stream = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    job, payload = pickle.load(sys.stdin.buffer)
    try:
        result = _JOBS[job](**payload)
    except ValidationError as err:
        result = {"error": str(err), "kind": "validation"}
    except RunError as err:
        result = {"error": str(err)}
    except Exception as err:
        from udaan.cli.scenarios import ScenarioError

        if isinstance(err, ScenarioError):
            result = {"error": str(err), "kind": "validation"}
        else:
            tb = traceback.format_exc(limit=6)
            result = {"error": f"{type(err).__name__}: {err}\n{tb}"}
    result_stream.write(_RESULT_MARK + pickle.dumps(result))
    result_stream.close()


# ─── Child side: building and flying a scenario ──────────────────────


class _Stop(Exception):
    """Raised from the step hook to end a run early (divergence or wall-clock budget)."""


class _WarningLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


def _build(system: str, kwargs: dict[str, Any]) -> Scenario:
    from udaan.cli import scenarios

    spec = get_system(system)
    assert spec.builder is not None
    builder: Callable[..., Scenario] = getattr(scenarios, spec.builder)
    return builder(render=False, **kwargs)


def _rotations(model: Any) -> np.ndarray:
    """``(k, 3, 3)`` attitudes of every airframe in ``model``."""
    import numpy as np

    state: Any = getattr(model, "state", None)
    if hasattr(model, "quadrotors"):  # QuadrotorFleet
        Rs = [q.state.orientation for q in model.quadrotors]
    elif hasattr(model, "agents"):  # QuadrotorCsPayloadFleet
        Rs = [a.state.orientation for a in model.agents]
    elif hasattr(state, "orientations"):  # M3, M4, M6-M8
        Rs = list(state.orientations)
    elif hasattr(state, "orientation"):  # M1, M2, M5
        Rs = [state.orientation]
    else:  # the MuJoCo-only multi-quad models
        Rs = [q.rotation for q in state.quads]
    return np.array([np.asarray(R, dtype=float) for R in Rs])


def _cable_tensions(model: Any) -> np.ndarray | None:
    """M3's per-cable scalar tensions in N, or ``None`` for a model that has none."""
    from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase

    if isinstance(model, MultiQuadCsPayloadBase):
        return model.cable_tensions()
    return None


#: Counters a model may expose, reported when present.
_COUNTERS = ("feedforward_drops", "slack_cable_steps", "compressive_cable_steps")


class _Recorder:
    """Samples a scenario after every outer step: errors, tilt, tensions, optional frames."""

    def __init__(self, scn: Scenario, deadline: float, keep_series: bool) -> None:
        self.scn = scn
        self.deadline = deadline
        self.keep_series = keep_series
        self.rows: list[dict[str, Any]] = []
        self.errors: list[float] = []
        self.max_tilt = 0.0
        self.min_tension = math.inf
        #: ``"diverged"``, ``"wall_budget"`` or ``"model_error"`` once a run stops early.
        self.stop_reason: str | None = None
        self.stop_detail: str | None = None
        self.frame_hook: Callable[[float], None] | None = None

    def sample(self) -> None:
        import numpy as np

        mdl = self.scn.model
        t = float(mdl.t)
        x = np.asarray(self.scn.tracked(), dtype=float)
        R = _rotations(mdl)
        tensions = _cable_tensions(mdl)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(R))):
            self.stop_reason = "diverged"
            self.stop_detail = f"state became non-finite at t = {t:.3f} s"
            raise _Stop
        tilt = float(np.degrees(np.max(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))))
        self.max_tilt = max(self.max_tilt, tilt)
        err = math.nan
        ref = None
        if self.scn.reference is not None:
            ref = np.asarray(self.scn.reference(t), dtype=float)
            err = float(np.max(np.linalg.norm(x - ref, axis=-1)))
            self.errors.append(err)
        min_t = math.nan
        if tensions is not None and len(tensions):
            min_t = float(np.min(tensions))
            self.min_tension = min(self.min_tension, min_t)
        if self.keep_series:
            row: dict[str, Any] = {"t": t}
            for i, p in enumerate(x):
                row.update({f"x{i}": p[0], f"y{i}": p[1], f"z{i}": p[2]})
            if ref is not None:
                for i, p in enumerate(ref):
                    row.update({f"ref_x{i}": p[0], f"ref_y{i}": p[1], f"ref_z{i}": p[2]})
            row["error"] = err
            row["tilt_deg"] = tilt
            if tensions is not None:
                row["min_tension"] = min_t
            self.rows.append(row)
        if self.frame_hook is not None:
            self.frame_hook(t)
        if time.time() > self.deadline:
            self.stop_reason = "wall_budget"
            self.stop_detail = f"wall-clock budget reached at t = {t:.3f} s"
            raise _Stop

    def attach(self) -> None:
        """Sample after each ``model.step`` call, which every ``simulate`` loop makes once per step."""
        mdl = self.scn.model
        step = mdl.step

        def sampled_step(*args: Any, **kwargs: Any) -> Any:
            out = step(*args, **kwargs)
            self.sample()
            return out

        mdl.step = sampled_step


def _fly(
    system: str,
    kwargs: dict[str, Any],
    deadline: float,
    keep_series: bool,
    frame_hook_factory: Callable[[Scenario], Callable[[float], None]] | None = None,
) -> tuple[Scenario, _Recorder, dict[str, Any]]:
    """Build, fly and summarise one scenario (child side)."""
    import numpy as np

    handler = _WarningLog()
    root = logging.getLogger("udaan")
    root.addHandler(handler)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_start = time.perf_counter()
        scn = _build(system, kwargs)
        rec = _Recorder(scn, deadline, keep_series)
        if frame_hook_factory is not None:
            rec.frame_hook = frame_hook_factory(scn)
        rec.attach()
        if scn.tf > MAX_DURATION:
            raise ValidationError(
                f"This trajectory sets its own duration, {scn.tf:g} s, past the "
                f"{MAX_DURATION:g} s limit."
            )
        sim_start = time.perf_counter()
        completed = True
        try:
            scn.run()
        except _Stop:
            completed = False
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as err:
            # The models raise these for a state they refuse to continue from
            # (a slack cable, a flat-map refusal, a failed solve); any other
            # exception is a crash and reaches the client as an error.
            completed = False
            rec.stop_reason = "model_error"
            rec.stop_detail = f"{type(err).__name__}: {err}"
        sim_wall = time.perf_counter() - sim_start
    root.removeHandler(handler)

    msgs = list(dict.fromkeys(handler.messages + [str(w.message) for w in caught]))
    errs = np.asarray(rec.errors, dtype=float)
    metrics: dict[str, Any] = {
        "system": system,
        "label": scn.label,
        "tracked": get_system(system).tracked,
        "duration_s": scn.tf,
        "simulated_time": float(scn.model.t),
        "completed": completed,
        "samples": len(rec.errors) if scn.reference is not None else None,
        "max_tilt_deg": rec.max_tilt,
        "wall_time_s": sim_wall,
        "build_time_s": sim_start - build_start,
    }
    if not completed:
        metrics["stop_reason"] = rec.stop_reason
        metrics["stopped"] = rec.stop_detail
    if errs.size:
        metrics["tracking_error_m"] = {
            "rms": float(np.sqrt(np.mean(errs**2))),
            "max": float(np.max(errs)),
            "final": float(errs[-1]),
        }
    if math.isfinite(rec.min_tension):
        metrics["min_cable_tension_n"] = rec.min_tension
    for name in _COUNTERS:
        value = getattr(scn.model, name, None)
        if isinstance(value, int):
            metrics[name] = value
    metrics["warnings"] = msgs[:20]
    if scn.notes:
        metrics["notes"] = scn.notes
    return scn, rec, metrics


def _write_series(rows: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() == ".json":
        import json

        path.write_text(json.dumps(rows, allow_nan=True))
        return
    import csv

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["t"])
        writer.writeheader()
        writer.writerows(rows)


def _simulate_job(
    system: str, kwargs: dict[str, Any], deadline: float, series_path: str | None
) -> dict[str, Any]:
    _, rec, metrics = _fly(system, kwargs, deadline, keep_series=series_path is not None)
    if series_path is not None:
        _write_series(rec.rows, Path(series_path))
        metrics["timeseries_path"] = series_path
    return metrics


def _video_job(
    system: str,
    kwargs: dict[str, Any],
    deadline: float,
    path: str,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    import mujoco

    frames: list[np.ndarray] = []
    state: dict[str, Any] = {"next": 0.0, "count": 0, "truncated": False}
    gif = path.lower().endswith(".gif")
    writer: Any = None

    def factory(scn: Scenario) -> Callable[[float], None]:
        nonlocal writer
        mj = getattr(scn.model, "_mjMdl", None)
        if mj is None:
            raise RunError(f"System '{system}' has no MuJoCo scene to record.")
        try:
            renderer = mujoco.Renderer(mj.model, height=height, width=width)
        except Exception as err:  # MuJoCo and PyOpenGL raise several types here
            raise RunError(
                gl_unavailable_message(
                    os.environ.get("MUJOCO_GL", "?"), f"{type(err).__name__}: {err}"
                )
            ) from err
        # The interactive viewer's camera: tracking body 1 (the first vehicle).
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = max(mj.model.stat.extent * 1.5, 1.5)
        cam.elevation = -20
        cam.azimuth = 135
        if not gif:
            import imageio.v2 as imageio

            writer = imageio.get_writer(path, fps=fps, macro_block_size=1)

        def capture(t: float) -> None:
            if t + 1e-9 < state["next"]:
                return
            if gif and len(frames) >= MAX_GIF_FRAMES:
                state["truncated"] = True
                return
            state["next"] += 1.0 / fps
            renderer.update_scene(mj.data, camera=cam)
            frame = renderer.render().copy()
            state["count"] += 1
            if gif:
                frames.append(frame)
            else:
                writer.append_data(frame)

        return capture

    try:
        _, _, metrics = _fly(
            system, kwargs, deadline, keep_series=False, frame_hook_factory=factory
        )
    finally:
        if writer is not None:
            writer.close()
    if gif:
        import imageio.v3 as iio

        iio.imwrite(path, frames, duration=1000 / fps, loop=0)
    metrics.update(
        {
            "gl": os.environ.get("MUJOCO_GL"),
            "video_path": path,
            "frames": state["count"],
            "truncated": state["truncated"],
            "fps": fps,
            "size": [width, height],
        }
    )
    return metrics


def _xml_job(system: str, kwargs: dict[str, Any], output_path: str | None, text: bool) -> dict:
    from udaan.utils.assets.cache import ensure_scene_beside

    spec = get_system(system)
    if spec.builder is None:  # comparison: a packaged MJCF with no closed loop
        from udaan.utils.assets.cache import PACKAGED_MJCF_DIR

        source = str(PACKAGED_MJCF_DIR / "quadrotor_comparison.xml")
        label = "packaged"
    else:
        scn = _build(system, kwargs)
        source = scn.model._mjMdl.full_path
        label = scn.label
    out: dict[str, Any] = {"system": system, "label": label, "source_path": source}
    path = source
    if output_path is not None:
        shutil.copyfile(source, output_path)
        ensure_scene_beside(output_path)
        path = output_path
    out["path"] = path
    if text:
        body = Path(path).read_text()
        out["truncated"] = len(body) > MAX_XML_TEXT
        out["xml"] = body[:MAX_XML_TEXT]
    return out


def _reference_job(system: str, kwargs: dict[str, Any], dt: float) -> dict[str, Any]:
    from udaan.flatness import QuadrotorCsPayload

    scn = _build(system, kwargs)
    setpoint = scn.model.payload_controller.setpoint
    jet = getattr(getattr(setpoint, "__self__", None), "get_jet", None)
    if jet is None:
        raise ValidationError(
            "This trajectory is a constant setpoint with no derivative jet; only "
            "'circle' and 'spiral' can be checked."
        )
    report = QuadrotorCsPayload.from_model(scn.model).check_reference(jet, scn.tf, dt=dt)
    import numpy as np

    finite = report.min_thrust_z[np.isfinite(report.min_thrust_z)]
    return {
        "system": system,
        "label": scn.label,
        "feasible": report.feasible,
        "samples": int(report.times.size),
        "infeasible_samples": report.infeasible_samples,
        "infeasible_intervals_s": [list(iv) for iv in report.intervals],
        "min_thrust_z": float(finite.min()) if finite.size else None,
        "summary": str(report),
    }


_JOBS: dict[str, Callable[..., dict[str, Any]]] = {
    "simulate": _simulate_job,
    "video": _video_job,
    "xml": _xml_job,
    "reference": _reference_job,
}


# ─── Public entry points (parent side) ───────────────────────────────


def _output_path(path: str | None, suffixes: tuple[str, ...], what: str) -> str | None:
    """Absolute ``path``, after checking its suffix and that its directory exists."""
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.suffix.lower() not in suffixes:
        raise ValidationError(f"{what} must end in {' or '.join(suffixes)}, got {path!r}.")
    p = p.resolve()
    if not p.parent.is_dir():
        raise ValidationError(f"Directory {str(p.parent)!r} does not exist.")
    if p.is_dir():
        raise ValidationError(f"{str(p)!r} is a directory.")
    return str(p)


def _temp_path(suffix: str) -> str:
    fd, name = tempfile.mkstemp(prefix="udaan-", suffix=suffix)
    os.close(fd)
    return name


def _in_child_with_temp(
    temp: str | None, job: str, payload: dict[str, Any], timeout: float, **kw: Any
) -> dict[str, Any]:
    """:func:`_in_child`, deleting ``temp`` (a temporary output file) if the run fails."""
    try:
        return _in_child(job, payload, timeout, **kw)
    except Exception:
        if temp is not None:
            with contextlib.suppress(OSError):
                Path(temp).unlink()
        raise


def simulate(
    system: str,
    backend: str | None = None,
    trajectory: str | None = None,
    duration: float | None = None,
    params: dict[str, Any] | None = None,
    save_timeseries: bool = False,
    output_path: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Fly one system headless and return summary metrics.

    The time series (per outer step: tracked and reference positions, error,
    largest tilt, smallest cable tension) is written to ``output_path``
    (``.csv`` or ``.json``) when one is given, or to a temporary ``.csv`` when
    ``save_timeseries`` is set.

    Raises:
        ValidationError: for an argument the server refuses.
        RunError: when the run cannot complete (wall-clock timeout, a crash).
    """
    _, kwargs = validate(system, backend, trajectory, duration, params)
    timeout = _check_timeout(timeout_s)
    series = _output_path(output_path, (".csv", ".json"), "output_path")
    temp = None
    if series is None and save_timeseries:
        series = temp = _temp_path(".csv")
    payload = {
        "system": system,
        "kwargs": kwargs,
        "deadline": None,
        "series_path": series,
    }
    return _in_child_with_temp(temp, "simulate", payload, timeout)


def render_video(
    system: str,
    trajectory: str | None = None,
    duration: float | None = None,
    params: dict[str, Any] | None = None,
    output_path: str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Fly one system on MuJoCo and record it offscreen to an MP4 or GIF.

    The file goes to ``output_path`` (``.mp4`` or ``.gif``), or to a temporary
    ``.mp4``. The camera tracks the first body, as the interactive viewer's
    does. An MP4 needs an even width and height; a GIF is at most
    :data:`MAX_GIF_SIZE` and :data:`MAX_GIF_FRAMES` frames, and a trajectory
    that sets its own length past that is cut short (``truncated``). Returns
    the simulation metrics plus ``video_path``, ``frames``, ``truncated``,
    ``fps``, ``size`` and ``gl`` (the offscreen backend).
    """
    _, kwargs = validate(system, "mujoco", trajectory, duration, params)
    timeout = _check_timeout(timeout_s)
    for name, value, low, high in (
        ("width", width, 16, MAX_VIDEO_SIZE[0]),
        ("height", height, 16, MAX_VIDEO_SIZE[1]),
        ("fps", fps, 1, MAX_FPS),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValidationError(f"{name} must be an integer in [{low}, {high}], got {value!r}.")
    path = _output_path(output_path, (".mp4", ".gif"), "output_path")
    if path is not None and path.lower().endswith(".gif"):
        if width > MAX_GIF_SIZE[0] or height > MAX_GIF_SIZE[1]:
            raise ValidationError(
                f"A GIF is at most {MAX_GIF_SIZE[0]} x {MAX_GIF_SIZE[1]}; record an .mp4 for more."
            )
        # One frame per 1/fps from the first step to the end, both included.
        if math.floor(kwargs["duration"] * fps) + 1 > MAX_GIF_FRAMES:
            raise ValidationError(
                f"A GIF is capped at {MAX_GIF_FRAMES} frames; {kwargs['duration']:g} s at {fps} "
                "fps is more. Lower fps or duration, or record an .mp4."
            )
    elif width % 2 or height % 2:
        raise ValidationError(f"An MP4 needs an even width and height, got {width} x {height}.")
    gl = offscreen_backend()
    unavailable = check_offscreen_rendering(gl)
    if unavailable is not None:
        raise RunError(unavailable)
    temp = None
    if path is None:
        path = temp = _temp_path(".mp4")
    payload = {
        "system": system,
        "kwargs": kwargs,
        "deadline": None,
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
    }
    return _in_child_with_temp(temp, "video", payload, timeout, env={"MUJOCO_GL": gl})


def generate_xml(
    system: str,
    params: dict[str, Any] | None = None,
    output_path: str | None = None,
    include_text: bool = False,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """The MJCF a system's MuJoCo backend loads for ``params``.

    Builds the MuJoCo model exactly as ``simulate`` would (which generates and
    caches the asset when it is generated from parameters) and returns its
    path; with ``output_path`` it copies the file there, with ``scene.xml``
    beside it so it loads. ``include_text`` also returns the XML itself. A
    parameter that does not shape the file (a start position, a reference
    knob, a value set on the loaded model) is refused.
    """
    spec = get_system(system)
    kwargs: dict[str, Any] = {}
    if spec.builder is not None:
        no_mjcf = {p.name for p in spec.params if not p.mjcf}
        refused = sorted(no_mjcf & set(params or {}))
        if refused:
            raise ValidationError(
                f"Parameter(s) {', '.join(refused)} of '{system}' do not shape its MJCF."
            )
        _, kwargs = validate(system, "mujoco", None, None, params)
    elif params:
        raise ValidationError(f"System '{system}' takes no parameters.")
    timeout = _check_timeout(timeout_s)
    out = _output_path(output_path, (".xml",), "output_path")
    payload = {"system": system, "kwargs": kwargs, "output_path": out, "text": include_text}
    return _in_child("xml", payload, timeout)


def check_reference(
    system: str,
    trajectory: str | None = None,
    duration: float | None = None,
    params: dict[str, Any] | None = None,
    dt: float = 0.005,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Pre-flight check that a reference keeps every airframe upright (no flight).

    Runs the system's flat map along the trajectory the scenario binds and
    reports the intervals where the airframe would need ``b3 . e3`` at or
    below the upright threshold. Available for systems whose
    ``describe_system`` lists ``check_reference``.
    """
    spec = get_system(system)
    if not spec.check_reference:
        names = [s.id for s in _systems_with_check()]
        raise ValidationError(
            f"System '{system}' has no reference check; available for: {', '.join(names)}."
        )
    _, kwargs = validate(system, "base", trajectory, duration, params)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not 1e-4 <= dt <= 0.1:
        raise ValidationError(f"dt must lie in [1e-4, 0.1] s, got {dt!r}.")
    timeout = _check_timeout(timeout_s)
    payload = {"system": system, "kwargs": kwargs, "dt": float(dt)}
    return _in_child("reference", payload, timeout)


def _systems_with_check() -> list[Any]:
    from .catalog import SYSTEMS

    return [s for s in SYSTEMS.values() if s.check_reference]
