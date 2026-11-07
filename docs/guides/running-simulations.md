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
| {py:class}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase` | ✓ | — | ✓ | — |
| {py:class}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase` | ✓ | — | ✓ | — |

The `vfx` (VPython) backends are importable as library classes but are **not**
offered by the CLI's `--model`: driven headlessly they block inside VPython and
never return, `--no-render` included. See {doc}`controllers` for the controllers
shipped with each model and how to swap them.

### Input and force types

`QuadrotorBase` and `QuadrotorCsPayloadBase` accept several input
repackagings, chosen at construction time via the `input=...` kwarg:

| Input type | Controller produces | Integrator consumes | Use when |
|---|---|---|---|
| `acceleration` (default) | 3-vec desired thrust force | wrench via geometric attitude controller | high-level trajectory tracking |
| `wrench` | 4-vec `[f, M_x, M_y, M_z]` | wrench directly | custom attitude laws, SysID |
| `prop_forces` | 4-vec per-rotor forces | allocated wrench | motor-level experiments |

On `QuadrotorBase`, `simulate()` flies all three paths with the same position
controller, the same attitude controller and the same flatness feedforward, and
every path closes the attitude loop inside `step()` at the same rate on both
backends; only how its wrench reaches the integrator differs.

### Loop rates (M1 quadrotor)

M1 runs two loops over its physics, each at its own rate, all three constructor
keywords:

| loop | computes | rate | keyword (default) |
|---|---|---|---|
| position (outer) | the thrust vector, whose direction is the desired `b₃` and so the desired attitude | once per `step()` | `dt` (5 ms, 200 Hz) |
| attitude (inner) | thrust and torques from that thrust vector and the attitude reference | `k` times per `step()` | `attitude_rate_hz` (800 Hz) |
| physics | integrates the rigid body | `N` substeps per `step()` | `physics_rate_hz` (1600 Hz) |

The defaults are 200 / 800 / 1600 Hz. The position loop and the feedforward reference are held across the
attitude updates; each attitude update is held across `N/k` physics substeps.

```python
QuadrotorMujoco(attitude_rate_hz=400)  # 200, 400, 800 or 1600 Hz at the defaults
QuadrotorMujoco(attitude_rate_hz=None)  # once per step, 200 Hz
QuadrotorBase(dt=0.01, physics_rate_hz=1000, attitude_rate_hz=500)  # 100 / 500 / 1000 Hz
```

`attitude_rate_hz` must be `k/dt` for a whole `k` that divides the `N` physics
substeps per step (eight 0.625 ms substeps by default); anything else raises
`ValueError`. The clock counts whole substeps, so `model.t` after `k` steps is
`k · dt` to the last bit at every valid rate.

### Physics rate (M1 quadrotor)

The physics integrates at `physics_rate_hz` (default 1600 Hz) on both backends:
each `step()` takes `N = physics_rate_hz · dt` substeps of `h = dt / N` — ZOH
substeps on `QuadrotorBase`, MuJoCo steps on `QuadrotorMujoco`, which writes `h`
into the MuJoCo model's timestep after loading `quadrotor_mj.xml` (the MJCF's own
timestep is overwritten). Nothing is rounded, so a step always advances
exactly `dt`.

```python
QuadrotorMujoco(physics_rate_hz=800)  # N = 4 substeps of 1.25 ms
QuadrotorBase(physics_rate_hz=1000, attitude_rate_hz=1000)  # N = 5 substeps of 1 ms
```

The rate must lie in 500–2000 Hz and make `N` a whole number, otherwise
`ValueError` lists the valid rates for that `dt`: at the default 5 ms, 600, 800,
1000, …, 2000 Hz. 500 Hz needs a `dt` that is a multiple of 2 ms.
`attitude_rate_hz` is then validated against the resulting `N`, the default 800 Hz
included: it divides `N = 4` and `8` (800 and 1600 Hz at 5 ms) but not
`N = 5`, so `physics_rate_hz=1000` alone raises and names the default; pass an
`attitude_rate_hz` that divides it (200 or 1000 Hz, or `None`).

### Rotor layout and limits (M1 quadrotor)

The allocation matrix is built from constructor keywords: `arm_length` (0.2 m),
`rotor_angles` (π/4, 3π/4, 5π/4, 7π/4), `spin_directions` (−1, 1, −1, 1),
`force_constant` and `torque_constant` (`k_F`, `k_M`; only `c = k_M/k_F`
enters), and the per-rotor limits `prop_min_force` / `prop_max_force` (0, 10 N),
each validated. `QuadrotorBase(arm_length=0.25)` is a different airframe; on
`QuadrotorMujoco` the MJCF is the source of all of them (rotor sites, yaw-gear
signs and size, `ctrlrange`), and a passed value that disagrees with it raises
`ValueError`. `DirectPropellerForceController.from_model(mdl)` allocates through
the model's own matrix.
So a trajectory bound to `position_controller.setpoint` (and a `yaw_setpoint`)
is flown on every path.

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
udaan run quadrotor --traj flip                  # vertical loop
udaan run quadrotor --traj lissajous -p 0,0,2    # 3D Lissajous
udaan run quadrotor --traj step                  # true setpoint step of (1, 1, 0.5) m at t = 2 s
udaan run quadrotor --traj mission               # take off, five waypoints, land
udaan run quadrotor --traj figure8               # figure-8 with the heading swinging ±90°
```

`step` is a true step (`StepSetpoint`): no smoothing, so no feedforward. `mission`
(`WaypointTrajectory`) takes off from `-p` (default `1,1,0`, on the ground),
visits five waypoints with a stop at each, and lands back where it started;
each leg is a 7th-order rest-to-rest polynomial, continuous through jerk.
`figure8` (`Figure8Trajectory`) flies two lemniscates about `-p` (default
`1,1,2`) with the heading `ψ = 90°·sin θ` bound as the model's `yaw_setpoint`;
it is continuous through snap. `mission`, `figure8`, `spiral` and `flip` set the
run's duration themselves.

The flip does not land where it took off. At the defaults it ends 5.66 m above
and 0.62 m along x from its start. Its climb and recovery each end at the
height their own boundary velocities and accelerations imply, so that ż never
reverses, and that height is not the one they started from. During the loop
itself the attitude feedforward is the rotation about the fixed pitch axis that
the trajectory supplies (`FlipTrajectory.reference_attitude`), not the generic
flat map, whose world-frame heading convention is singular on a pitch loop. See
{py:class}`~udaan.utils.trajectory.FlipTrajectory`.

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
