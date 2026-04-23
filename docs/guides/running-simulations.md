# Running simulations

This page summarises the CLI and scripted entry points. Every command accepts
`--help` for the full option list.

## Model capabilities

Each model ships with several backends and input modes. The table below shows
what each supports — use it to pick the right entry point for your task.

| Model | `base` (dynamics only) | `vfx` (VPython) | `mujoco` (MuJoCo) | Fleet |
|---|:---:|:---:|:---:|:---:|
| {py:class}`~udaan.models.quadrotor.QuadrotorBase` | ✓ | ✓ | ✓ | ✓ (via `fleet`) |
| {py:class}`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase` | ✓ | ✓ | ✓ (tendon / links / cable) | ✓ (via `cspayload-fleet`) |
| {py:class}`~udaan.models.mujoco.MultiQuadrotorCSPointmass` | — | — | ✓ | — |
| {py:class}`~udaan.models.mujoco.MultiQuadRigidbody` | — | — | ✓ | — |

See {doc}`controllers` for the controllers shipped with each model and how to swap them.

### Input and force types

`QuadrotorBase` and `QuadrotorCsPayloadBase` accept several input
repackagings, chosen at construction time via the `input=...` kwarg:

| Input type | Controller produces | Integrator consumes | Use when |
|---|---|---|---|
| `acceleration` (default) | 3-vec desired thrust force | wrench via geometric attitude controller | high-level trajectory tracking |
| `wrench` | 4-vec `[f, M_x, M_y, M_z]` | wrench directly | custom attitude laws, SysID |
| `prop_forces` | 4-vec per-rotor forces | allocated wrench | motor-level experiments |

### Cable models (payload only)

| `cable_model` | MuJoCo backend | Captures slack? | Notes |
|---|---|:---:|---|
| `tendon` | spatial tendon constraint | — | fast, less realistic under slack |
| `links` (default) | rigid N-link chain | ✓ | most stable, recommended default |
| `cable` | MuJoCo composite cable | ✓ | experimental; see {doc}`../theory/dynamics/quadrotor-cspayload` caveats |

## Quadrotor

```bash
udaan run quadrotor                              # MuJoCo viewer, hover
udaan run quadrotor -m base                      # pure dynamics, no rendering
udaan run quadrotor --traj spiral -p 0,0,2       # helical spiral trajectory
udaan run quadrotor --traj lissajous -p 0,0,2    # 3D Lissajous
```

## Quadrotor with cable-suspended payload

```bash
udaan run quad-payload -c tendon        # spatial-tendon cable model
udaan run quad-payload -c links         # N-link rigid cable model
```

The controller used is derived in {doc}`../theory/controllers/quadrotor-payload`; the
underlying dynamics are covered in {doc}`../theory/dynamics/quadrotor-cspayload`.

## Side-by-side fleets

Two fleet commands exist for comparing controllers or gains:

```bash
udaan run fleet --demo l1-comparison           # N quadrotors, L1 vs PD
udaan run cspayload-fleet --demo gain-sweep    # N quad+payload, gain sweep
```

## Scripted (Python)

```python
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco

mdl = QuadrotorCsPayloadMujoco(render=True, cable_model="links")
mdl._payload_controller.setpoint = lambda t: ([0, 0, 1], [0, 0, 0], [0, 0, 0])
mdl.simulate(tf=8.0, payload_position=[1, 1, 0.5])
```
