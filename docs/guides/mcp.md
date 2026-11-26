# Use with Claude (MCP server and skill)

udaan ships an [MCP](https://modelcontextprotocol.io) server, `udaan-mcp`, that
lets Claude or any other MCP client fly every udaan system headless and read the
results back as JSON. It also ships an Agent Skill, `skills/udaan/SKILL.md`, that
teaches Claude when and how to use the server.

The server and `udaan run` build their models through the same scenario
builders ({py:mod}`udaan.cli.scenarios`), so a tool call and a command with the
same backend, trajectory and parameters fly the same model from the same start.
The default backends differ: the server uses `base` where it exists, the CLI
`mujoco`.

## Install and register

```bash
pip install "udaan[mcp]"
```

The `mcp` extra installs the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(2.x) and the `udaan-mcp` console script, which serves over stdio.
`python -m udaan.mcp` starts the same server; it prints nothing and waits for a
client on stdin (Ctrl-C to quit). To check it end to end, register it as below
and run `claude mcp list`, or use the MCP Inspector
(`npx @modelcontextprotocol/inspector udaan-mcp`).

Register it with Claude Code:

```bash
claude mcp add udaan -- udaan-mcp
```

Start a new Claude Code session to load the tools (`mcp__udaan__simulate` and
so on). The command must be on the PATH Claude Code sees; after a virtualenv
install, register its absolute path, `/path/to/venv/bin/udaan-mcp`. With uv and
no prior install: `claude mcp add udaan -- uvx --from "udaan[mcp]" udaan-mcp`.

Other clients take the command as JSON, for example a project `.mcp.json` or
Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "udaan": { "command": "udaan-mcp" }
  }
}
```

To run the server without installing the package first, use
`"command": "uvx", "args": ["--from", "udaan[mcp]", "udaan-mcp"]`.

### As a Claude Code plugin

The repository is also a Claude Code plugin marketplace. The plugin bundles the
skill and registers the server. Its server runs `udaan-mcp` from your PATH, so
install `udaan[mcp]` into an environment on that PATH first. Use either the
plugin or `claude mcp add`, not both.

```text
/plugin marketplace add vkotaru/udaan
/plugin install udaan@udaan
```

To use only the skill, copy `skills/udaan/` into `~/.claude/skills/` (or
`.claude/skills/` in a project).

## Tools

| Tool | Returns |
|---|---|
| `list_systems()` | every system: id, label, description, backends, trajectories, parameters with defaults and bounds |
| `describe_system(system)` | one system, with notes on its trajectories |
| `simulate(system, backend, trajectory, duration, params, save_timeseries, output_path, timeout_s)` | summary metrics of a headless run (below) |
| `render_video(system, trajectory, duration, params, output_path, width, height, fps, timeout_s)` | an MP4 or GIF recorded offscreen on MuJoCo, plus the metrics |
| `generate_xml(system, params, output_path, include_text)` | the MJCF the MuJoCo backend loads for those parameters |
| `check_reference(system, trajectory, duration, params)` | for `quad-payload` with `circle` or `spiral`, without flying: `feasible`, `infeasible_intervals_s`, `min_thrust_z` (smallest vertical component of the thrust axis, 1 = upright) and `summary` |

`params` is an object keyed by the parameter names `describe_system` lists, for
example `{"num_links": 5, "attachments": [0, 2, 5]}` for the hose.
`backend` defaults to `"base"`, the analytical model, wherever one exists.
`duration` (default 10 s) is simulated time; `flip`, `spiral`, `mission` and
`figure8` set their own length and ignore it, so read `duration_s` in the result.

`simulate` returns:

| Key | Meaning |
|---|---|
| `tracking_error_m` | `rms`, `max` and `final` distance, in m, between the tracked point and its reference, taken as the worst point at each step on multi-vehicle systems |
| `max_tilt_deg` | largest angle between any airframe's thrust axis and vertical |
| `min_cable_tension_n` | smallest cable tension, in N; reported on M3 only |
| `feedforward_drops`, `slack_cable_steps`, `compressive_cable_steps` | the model's own counters, where it has them |
| `duration_s`, `simulated_time` | the duration flown (set by the trajectory where it has its own length) and the model clock at the end |
| `label`, `tracked`, `notes` | the run's description, the point the error is measured on, and remarks such as a chosen circle rate |
| `completed`, `stop_reason`, `stopped` | whether the run reached its duration; if not, `diverged`, `wall_budget` or `model_error`, with a detail line |
| `samples`, `build_time_s` | the steps scored, and the wall time spent constructing the model |
| `warnings` | warnings the model logged during the run |
| `wall_time_s` | wall time of the flight, without model construction |
| `timeseries_path` | per-step CSV or JSON, when `output_path` or `save_timeseries` (a temporary `.csv`) is given; columns `t`, `x{i}`/`y{i}`/`z{i}` and `ref_x{i}`/… per tracked point, `error`, `tilt_deg`, and `min_tension` on M3 |

`render_video` adds `video_path`, `frames`, `fps`, `size` and `gl` (the offscreen
backend used). `generate_xml` returns `path`, `source_path` (the cached or
packaged file) and, with `include_text`, `xml` and `truncated`.

## Systems

| id | Label | Backends | Trajectories | Tracked point |
|---|---|---|---|---|
| `quadrotor` | M1 | base, mujoco | hover, step, mission, flip, spiral, circle, lissajous, figure8 | airframe |
| `quad-payload` | M2 | base, mujoco (`cable_model` tendon or links) | hover, circle, spiral | payload |
| `multi-quad-payload`, `multi-quad` | M3 | base, mujoco | hover | payload |
| `multi-quad-rigidbody`, `multi-quad-rigid` | M4 | base, mujoco | hover | payload |
| `quad-flexcable` | M5 | base, mujoco | hover, lissajous | payload |
| `hose` | M6 | base, mujoco; `controller` lqr or station on either | hover, lissajous | hose root, or each vehicle with `station` |
| `tethered-quad` | M7 | base, mujoco | hold, circle, climb | top vehicle |
| `tethered-hose` | M8 | base, mujoco | hold, circle, climb | top vehicle |
| `fleet` | | mujoco (`demo` l1-comparison or gain-sweep) | hover | each quadrotor |
| `cspayload-fleet` | | mujoco (`demo` same-gains or gain-sweep) | hover | each payload |
| `comparison` | | MJCF only (`generate_xml`) | | |

M1 to M8 are udaan's model numbers. The hose root is vertex 0, the first point
of the hose. `station` gives each vehicle its own station-keeping PD instead of
one LQR for the whole hose. On the systems that track several points, each
step's error is the worst point's.

## Limits and isolation

Every run executes in its own child process, so a run cannot block the server
or take it down:

- `duration` is simulated time, at most {py:data}`~udaan.mcp.catalog.MAX_DURATION`
  (60 s). Link counts are bounded by {py:data}`~udaan.mcp.catalog.MAX_LINKS`
  and vehicle counts by {py:data}`~udaan.mcp.catalog.MAX_VEHICLES`. Every
  parameter has a range, which `describe_system` lists.
- `timeout_s` (default 120 s, at most 1800 s) is a wall-clock budget, model
  construction included. When it
  runs out, the run stops and returns its partial metrics with
  `completed: false`. A child that still has not returned 30 s later is killed,
  and the call fails with an error instead of partial metrics. Time spent
  waiting for a free run slot counts against the same budget.
- `output_path` must be in an existing directory; a relative path resolves
  against the server's working directory, so pass an absolute one. A GIF holds
  at most {py:data}`~udaan.mcp.runner.MAX_GIF_FRAMES` (300) frames of at most
  {py:data}`~udaan.mcp.runner.MAX_GIF_SIZE` (640 x 480); a trajectory that sets
  its own length past that is cut short and reported `truncated`. An MP4 needs
  an even width and height.
- `generate_xml` refuses parameters that do not change the file: start
  positions, reference knobs such as the tether `standoff`, and values set on
  the loaded model such as the quad-payload `cable_length`.
- Cancelling a call from the client does not stop its run: the child runs to
  the end of its budget and keeps its slot until then.
- At most {py:data}`~udaan.mcp.runner.MAX_CONCURRENT_RUNS` runs execute at once.
  Further calls wait for a free slot.
- A run's output is captured by the server, so model output cannot corrupt
  the stdio channel. A run that dies, even while importing, is reported with
  the last line of its traceback.

Nothing opens a window: every model is built with `render=False`, and
`render_video` draws with `mujoco.Renderer` on an offscreen context.
On Linux that context is EGL, or OSMesa when the server's environment sets
`MUJOCO_GL=osmesa`. On macOS it is CGL. Before its first video the server draws
one test frame with that backend; if that fails, every `render_video` call fails
at once with the cause and the packages to install. Other platforms are refused. The VPython
(`vfx`) backend is never offered, because it blocks on a browser scene.

## From Python

The tools are also plain functions in {py:mod}`udaan.mcp.runner`, which need no
MCP SDK:

```python
from udaan.mcp import runner

m = runner.simulate("quad-payload", trajectory="circle", duration=9.0)
print(m["completed"], m["tracking_error_m"], m["max_tilt_deg"])
if runner.check_offscreen_rendering() is None:
    runner.render_video("hose", duration=5.0, output_path="/tmp/hose.mp4")
```
