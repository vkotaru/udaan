"""The ``udaan`` MCP server: list, describe, simulate, record, generate MJCF, check a reference.

Built on the official MCP Python SDK (``mcp`` 2.x, the ``MCPServer`` API).
Every tool is a thin wrapper over :mod:`udaan.mcp.catalog` and
:mod:`udaan.mcp.runner`; a refused argument or a failed run comes back to the
client as a tool error carrying the reason.

Run it over stdio with the ``udaan-mcp`` console script, or ``python -m udaan.mcp``.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .catalog import MAX_DURATION, SYSTEMS, ValidationError, get_system
from .runner import DEFAULT_TIMEOUT, RunError
from .runner import check_reference as _check_reference
from .runner import generate_xml as _generate_xml
from .runner import render_video as _render_video
from .runner import simulate as _simulate

INSTRUCTIONS = f"""\
udaan simulates quadrotors and cable-suspended payloads. Call list_systems for the
system ids, then simulate to fly one headless and read back tracking error, tilt and
cable tension. Every run is bounded: duration <= {MAX_DURATION:g} s of simulated time,
and a wall-clock timeout (timeout_s, default {DEFAULT_TIMEOUT:g} s). backend="base" is
the analytical model (fast, no MuJoCo); "mujoco" is the physics backend. render_video
records an MP4/GIF offscreen (EGL or OSMesa on Linux, CGL on macOS). generate_xml
returns the MJCF a MuJoCo run loads. Cancelling a call does not stop its run.
"""


def _call(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run ``fn``, reporting a refused argument or a failed run as a tool error."""
    try:
        result: dict[str, Any] = fn(*args, **kwargs)
        return result
    except (ValidationError, RunError) as err:
        raise ToolError(str(err)) from err


def build_server() -> MCPServer:
    """The server with every tool registered."""
    from udaan import __version__

    server = MCPServer(name="udaan", version=__version__, instructions=INSTRUCTIONS)

    # The catalog tools are async so that they answer on the event loop even
    # while every worker thread is waiting on a run.
    @server.tool()
    async def list_systems() -> dict[str, Any]:
        """Every system: id, label, description, backends, trajectories, parameters."""
        return {
            "systems": [s.describe(full=False) for s in SYSTEMS.values()],
            "limits": {"max_duration_s": MAX_DURATION},
        }

    @server.tool()
    async def describe_system(system: str) -> dict[str, Any]:
        """One system's full description, including notes on its trajectories."""
        try:
            return get_system(system).describe(full=True)
        except ValidationError as err:
            raise ToolError(str(err)) from err

    @server.tool()
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
        """Fly a system headless and return summary metrics.

        Returns tracking_error_m (rms/max/final, in m, against the tracked point's
        reference), max_tilt_deg, min_cable_tension_n where the model reports
        tensions, counters such as feedforward_drops, warnings, and wall_time_s;
        stop_reason (diverged, wall_budget, model_error) when completed is false.
        backend defaults to "base" where it exists. duration (simulated s)
        defaults to the system's own; flip, spiral, mission and figure8 set their
        own length, reported as duration_s. params takes the system's
        parameters from describe_system. Set save_timeseries (temporary CSV) or
        output_path (.csv or .json) for the per-step series.
        """
        return _call(
            _simulate,
            system,
            backend=backend,
            trajectory=trajectory,
            duration=duration,
            params=params,
            save_timeseries=save_timeseries,
            output_path=output_path,
            timeout_s=timeout_s,
        )

    @server.tool()
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
        """Fly a system on MuJoCo and record it offscreen (EGL/OSMesa) to MP4 or GIF.

        Returns video_path (output_path, .mp4 or .gif, or a temporary .mp4), frames
        and the same metrics as simulate. Fails with a message if no offscreen
        OpenGL is available.
        """
        return _call(
            _render_video,
            system,
            trajectory=trajectory,
            duration=duration,
            params=params,
            output_path=output_path,
            width=width,
            height=height,
            fps=fps,
            timeout_s=timeout_s,
        )

    @server.tool()
    def generate_xml(
        system: str,
        params: dict[str, Any] | None = None,
        output_path: str | None = None,
        include_text: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """The MJCF the system's MuJoCo backend loads for these params.

        Returns its path (copied to output_path, .xml, when given) and, with
        include_text, the XML itself. Start positions and reference knobs are
        refused because they do not change the file.
        """
        return _call(
            _generate_xml,
            system,
            params=params,
            output_path=output_path,
            include_text=include_text,
            timeout_s=timeout_s,
        )

    @server.tool()
    def check_reference(
        system: str,
        trajectory: str | None = None,
        duration: float | None = None,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Pre-flight: does the trajectory keep the airframe upright? No flight.

        Runs the flat map along the reference and returns feasible, the
        infeasible intervals and the smallest b3.e3. Available for quad-payload
        'circle' and 'spiral'.
        """
        return _call(
            _check_reference,
            system,
            trajectory=trajectory,
            duration=duration,
            params=params,
            timeout_s=timeout_s,
        )

    return server


def main() -> None:
    """Console entry point: serve over stdio."""
    build_server().run("stdio")
