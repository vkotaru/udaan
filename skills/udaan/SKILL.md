---
name: udaan
description: Simulate quadrotors and cable-suspended payloads in the udaan Python package, through its udaan-mcp MCP server or the udaan CLI (MuJoCo or analytical models). Use when asked to simulate, compare or record a video of a quadrotor, a slung or multi-drone payload, a flexible cable or hose, or a tethered drone in udaan; to report simulated tracking error, tilt or cable tension; to export the MJCF of a udaan system; to check a udaan quad-payload reference before simulating it; or to install or register udaan-mcp. Not for real-vehicle flight or MuJoCo models outside udaan.
---

# udaan: aerial-robotics simulation

udaan simulates quadrotors and the loads they carry. It has two ways in:

- **The MCP server** `udaan-mcp`. Prefer it: every run is bounded, headless and
  returns its numbers as JSON.
- **The CLI** `udaan`, as a fallback when the server is not configured.

## Install and register

If tools named `list_systems` and `simulate` from the server `udaan` are
available (in Claude Code: `mcp__udaan__simulate`), the server is set up: use
them. Otherwise, with the user's agreement:

```bash
pip install "udaan[mcp]"            # the package, the udaan-mcp script and the MCP SDK
claude mcp add udaan -- udaan-mcp   # register the server with Claude Code (stdio)
```

- `udaan-mcp` must be on the PATH of the client that launches it. After a
  virtualenv install, register the absolute path instead:
  `claude mcp add udaan -- /path/to/venv/bin/udaan-mcp`.
- A newly registered server appears only in a new session. Tell the user to
  restart, and use the CLI fallback below until then.
- With uv and no prior install: `claude mcp add udaan -- uvx --from "udaan[mcp]" udaan-mcp`.
- Register either this server or the udaan plugin, not both.

Other clients take the same stdio command as JSON, for example in `.mcp.json`
or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "udaan": { "command": "udaan-mcp" }
  }
}
```

With uv: `{"command": "uvx", "args": ["--from", "udaan[mcp]", "udaan-mcp"]}`.

## Tools

| Tool | Use it to |
|---|---|
| `list_systems()` | get every system id, its backends, trajectories and parameters with their defaults and bounds |
| `describe_system(system)` | read one system's details, including notes on its trajectories |
| `simulate(system, backend, trajectory, duration, params, save_timeseries, output_path, timeout_s)` | fly a system headless; returns `tracking_error_m` (rms/max/final), `max_tilt_deg`, `min_cable_tension_n` (M3 only), counters, `warnings`, `wall_time_s` |
| `render_video(system, trajectory, duration, params, output_path, width, height, fps, timeout_s)` | record an MP4 or GIF on MuJoCo, offscreen; returns `video_path` and the metrics |
| `generate_xml(system, params, output_path, include_text)` | get the MJCF file a MuJoCo run loads for those params |
| `check_reference(system, trajectory, duration, params)` | test, without flying, whether `quad-payload` `circle` or `spiral` keeps the airframe upright; returns `feasible`, `infeasible_intervals_s`, `min_thrust_z` (smallest vertical component of the thrust axis, 1 is upright) and `summary` |

A first call: `simulate(system="quad-payload", trajectory="circle", duration=8)`,
then read `completed`, `tracking_error_m.rms` and `max_tilt_deg`.

- `params` is an object keyed by the parameter names `describe_system` lists,
  for example `{"num_links": 5, "cable_length": 1.2}` on `quad-flexcable`.
- `backend` defaults to `"base"` (analytical, fast) where the system has one.
  Use `"mujoco"` for contact and full physics. `render_video` always flies MuJoCo.
- `duration` (default 10) is simulated seconds, at most 60. `flip`, `spiral`,
  `mission` and `figure8` set their own length and ignore it: always report the
  result's `duration_s`.
- `output_path` must be an absolute path in an existing directory: `.csv` or
  `.json` for `simulate`, `.mp4` or `.gif` for `render_video`, `.xml` for
  `generate_xml`. Without it, files go to a temporary path that the result
  returns. A GIF holds at most 300 frames; use `.mp4` for longer clips.

## Systems

| id | Label | What it is | Backends | Trajectories |
|---|---|---|---|---|
| `quadrotor` | M1 | one quadrotor, geometric SE(3) control | base, mujoco | hover, step, mission, flip, spiral, circle, lissajous, figure8 |
| `quad-payload` | M2 | quadrotor + point-mass payload on one cable; `cable_model` `tendon` or `links` on MuJoCo | base, mujoco | hover, circle, spiral |
| `multi-quad-payload` | M3 | nQ quadrotors, one point-mass payload | base, mujoco | hover |
| `multi-quad-rigidbody` | M4 | nQ quadrotors, one rigid-body payload | base, mujoco | hover |
| `quad-flexcable` | M5 | quadrotor + payload on an n-link cable, LQR | base, mujoco | hover, lissajous |
| `hose` | M6 | quadrotors in series on one n-link hose; `controller` `lqr` or `station` | base, mujoco | hover, lissajous |
| `tethered-quad` | M7 | one quadrotor tethered to a ground anchor | base, mujoco | hold, circle, climb |
| `tethered-hose` | M8 | quadrotors on a hose pinned to a ground anchor | base, mujoco | hold, circle, climb |
| `multi-quad` | | quadrotors station-keeping a point-mass payload (PD each) | mujoco | hover |
| `multi-quad-rigid` | | four quadrotors station-keeping a rigid-body payload | mujoco | hover |
| `fleet` | | independent quadrotors side by side; `demo` `l1-comparison` or `gain-sweep` | mujoco | hover |
| `cspayload-fleet` | | quad-payload agents side by side; `demo` `same-gains` or `gain-sweep` | mujoco | hover |
| `comparison` | | plant + reference quadrotor scene, MJCF only | `generate_xml` only | none |

Labels M1 to M8 are udaan's model numbers; nQ is the number of quadrotors
(`num_quadrotors`). The tracking error is measured on the system's tracked
point: the airframe on M1; the payload on M2 to M5; on M6 the hose root (vertex
0, the first point of the hose) with `controller` `lqr`, and every vehicle with
`station`; the top vehicle on M7 and M8. On `multi-quad`, `multi-quad-rigid`,
`fleet`, `cspayload-fleet` and M6 `station`, each step's error is the worst
vehicle's (or payload's). A `hover` on M5 and M6 is a step away from the spawn,
so a large `max` error with a small `final` error is expected.

## Reading results

- `completed: false` means the run ended early; `stop_reason` is `diverged`,
  `wall_budget` (`timeout_s`, default 120 s) or `model_error`, and `stopped`
  gives the detail. Report it; do not treat partial metrics as a full run. A run that
  crashes, or overruns `timeout_s` by 30 s, comes back as a tool error instead.
- Report `duration_s` and `simulated_time` with the metrics.
- `warnings` holds the model's own warnings. On M5, for example, the analytical
  model warns that a long chain is numerically stiff; switch to `backend="mujoco"`.
- A refused argument comes back as a tool error that names the accepted values.
  Fix the call rather than retrying it unchanged.

## Headless pitfalls

- The server never opens a window. Every `udaan run` command defaults to
  MuJoCo with a viewer window, so on the CLI always pass `--no-render` (or
  `-m base` where the command has `-m`). On the CLI, `multi-quad-payload` and
  `multi-quad-rigidbody` have no `-m` and run MuJoCo only, although the server
  also offers `base` for them.
- Never use the VPython backend (`vfx`). It blocks on a browser scene even with
  `--no-render`. The server and the CLI both refuse it.
- `render_video` renders offscreen: EGL on Linux (or OSMesa, if the server's
  environment sets `MUJOCO_GL=osmesa`), CGL on macOS; other platforms are
  refused. If no offscreen context can be made it fails with a message naming
  the packages; it does not hang. `comparison` cannot be recorded. The CLI's `-r out.mp4` records
  through a viewer window, so it needs a display.

## CLI fallback

```bash
udaan run quadrotor -m base --traj spiral -t 10
udaan run quad-payload -m mujoco -c links --traj circle -t 8 --no-render
udaan run multi-quad-payload -n 4 -t 5 --no-render
udaan run quad-flexcable -m base -n 4 --traj lissajous -t 10
udaan run hose -m base -n 4 -a 0,2,4 -t 10
udaan run tethered-quad -m base --traj circle -t 12
udaan run fleet --demo l1-comparison -t 10 --no-render
udaan generate-xml hose -n 6 -o hose.xml
udaan run <command> --help                    # every option
```

The CLI prints a one-line summary, and the final error for M5, M7, M8 and M6
with `--controller lqr`. It does not
print the RMS or tilt metrics. For those, call the Python function behind the
tool, which returns the same dictionary:

```bash
python -c "from udaan.mcp import runner; import json; print(json.dumps(runner.simulate('quadrotor', trajectory='circle'), indent=1))"
```
