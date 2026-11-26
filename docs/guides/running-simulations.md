# Running simulations

This page summarises the CLI and scripted entry points. Every command accepts
`--help` for the full option list.

## Model capabilities

Each model ships with several backends and input modes. The table below shows
what each supports — use it to pick the right entry point for your task.

| | Model | `base` (dynamics only) | `vfx` (VPython) | `mujoco` (MuJoCo) | CLI |
|---|---|:---:|:---:|:---:|---|
| M1 | {py:class}`~udaan.models.quadrotor.QuadrotorBase` | ✓ | ✓ (library only) | ✓ | `quadrotor` |
| M2 | {py:class}`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase` | ✓ | ✓ (library only) | ✓ (tendon / links) | `quad-payload` |
| M3 | {py:class}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase` | ✓ | — | ✓ | `multi-quad-payload` (MuJoCo) |
| M4 | {py:class}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase` | ✓ | — | ✓ | `multi-quad-rigidbody` (MuJoCo) |
| M5 | {py:class}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase` | ✓ | — | ✓ | `quad-flexcable` |
| M6 | {py:class}`~udaan.models.multi_quad_hose.MultiQuadHoseBase` | ✓ | — | ✓ | `hose` |
| M7 | {py:class}`~udaan.models.tethered_hose.TetheredQuadrotorBase` | ✓ | — | ✓ | `tethered-quad` |
| M8 | {py:class}`~udaan.models.tethered_hose.TetheredHoseBase` | ✓ | — | ✓ | `tethered-hose` |

Each MuJoCo backend is the `...Mujoco` class beside its `...Base` in the same
package. The VPython backends (`QuadrotorVfx`, `QuadrotorCsPayloadVfx`) are
importable but not offered on the CLI, because VPython blocks while it opens
its browser scene.

Comparison and earlier models, all MuJoCo:

| Model | What it runs | CLI |
|---|---|---|
| {py:class}`~udaan.models.mujoco.QuadrotorFleet` | `N` M1 quadrotors side by side, each with its own controllers and optional unmodelled mass | `fleet` |
| {py:class}`~udaan.models.mujoco.QuadrotorCsPayloadFleet` | `N` M2 quadrotor-payload agents side by side on link-chain cables | `cspayload-fleet` |
| {py:class}`~udaan.models.mujoco.QuadrotorComparison` | a plant and a reference quadrotor in one scene, stepped by the caller | — |
| {py:class}`~udaan.models.mujoco.MultiQuadrotorCSPointmass` | the earlier `nQ`-quadrotor point-mass payload model | `multi-quad` |
| {py:class}`~udaan.models.mujoco.MultiQuadRigidbody` | the earlier four-quadrotor rigid-body payload model | `multi-quad-rigid` |

{py:mod}`udaan.models.base` holds analytical point-mass models
(`FloatingPointmass`, `S2Pendulum`, `PointmassSuspendedPayload`,
`MultiPointmassSuspendedPayload`) that are stepped by the caller and have no
`simulate()`.

Short flags are per command: `-n` is the number of quadrotors on `multi-quad-*`
and `fleet` but the number of links on `quad-flexcable`, `hose` and
`tethered-*`; `-c` is the cable model on `quad-payload` but the controller on
`hose`; `-a` is the attachment vertices on `hose` but the attachment radius on
`multi-quad-rigidbody`. `udaan run <command> --help` lists them.

See {doc}`controllers` for the controllers shipped with each model and how to swap them.

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
udaan run quad-payload -c tendon                    # spatial-tendon cable model, hover
udaan run quad-payload -c links                     # 10-link chain cable model, hover
udaan run quad-payload --traj circle                # payload on a 1 m circle at 1 m/s
udaan run quad-payload --traj spiral                # payload on a horizontal helix
udaan run quad-payload --traj spiral --cable-length 2 -m base
```

`--traj` takes `hover` (the default: the payload starts at `-p`, default
`-1,2,0.5`, and flies to the controller's own setpoint), `circle` and `spiral`.
`circle` (`CircularTraj`) is a 1 m circle in the horizontal plane centred on
`-p`, starting from rest and reaching 1 m/s over a 2 s smooth ramp; `spiral` is a {py:class}`~udaan.utils.trajectory.HelixTrajectory`
starting at `-p`: a 0.8 m horizontal circle at 2 m/s, two loops, climbing 1 m,
with smooth ramps in and out, and it sets the run's duration itself. Both
default `-p` to `0,0,1.5` and start the payload on its reference; both carry
jerk and snap, so the cable and airframe feed-forwards engage.

`--cable-length` sets the cable length in metres (default: the model's own,
1 m) and rebuilds the default controllers for it. The link chain's length is
fixed by its MJCF, so `-c links` refuses any other value.

The controller used is derived in {doc}`../theory/controllers/quadrotor-payload`; the
underlying dynamics are covered in {doc}`../theory/dynamics/quadrotor-cspayload`.

## Several quadrotors carrying one payload

```bash
udaan run multi-quad-payload -n 3 --no-render       # M3: three quadrotors, point-mass payload
udaan run multi-quad-payload -n 4 -l 1.5 --radius 0.8
udaan run multi-quad-rigidbody -n 4 --no-render     # M4: four quadrotors, rigid-body payload
udaan generate-xml multi-quad-rigidbody -n 6 -o rigid.xml
```

In M3 each quadrotor hangs the shared point-mass payload from its own taut
cable, in a ring of radius `--radius`. The payload controller computes the net
force on the payload and splits it into cable tensions with a
`TensionSplitPolicy`; every cable and airframe then runs its own loop. In M4 the
payload is a rigid body with cables at a ring of attachment points, and the
controller tracks its position and attitude by allocating a payload wrench to
the cables; at least three quadrotors are needed to command a non-level
attitude. Both commands run the MuJoCo backend and generate their MJCF from the
options; the analytical models are
{py:class}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase` and
{py:class}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase`. The
dynamics are in {doc}`../theory/dynamics/multi-quad-cspointmass` and
{doc}`../theory/dynamics/multi-quad-rigidbody`, the M3 controller in
{doc}`../theory/controllers/multi-quad-payload`, and the flat maps in
{doc}`../theory/differential-flatness/multi-quad-cspayload` and
{doc}`../theory/differential-flatness/multi-quad-rigidbody`.

## Quadrotor with a payload on a flexible cable

```bash
udaan run quad-flexcable -m base -t 10              # analytical model, 3-link cable
udaan run quad-flexcable -n 5 --no-render           # MuJoCo, 5-link cable
udaan run quad-flexcable --traj lissajous --no-render
```

The payload is regulated through the whole link chain by an infinite-horizon
LQR, with the flat map's reference state and wrench as feed-forward when the
trajectory supplies derivatives to order `2n + 4` (`get_jet`). The controller
is derived in {doc}`../theory/controllers/quadrotor-flexcable`, the dynamics
in {doc}`../theory/dynamics/quadrotor-flexcable` and the flat map in
{doc}`../theory/differential-flatness/quadrotor-flexcable`.

## Several quadrotors carrying one flexible hose

```bash
udaan run hose -m base -t 10                        # analytical model, 4 links, a vehicle at each end
udaan run hose -n 4 -a 0,2,4 --no-render            # MuJoCo, three vehicles
udaan run hose --traj lissajous --no-render
udaan generate-xml hose -n 4 -a 0,2,4 -o hose.xml
```

The hose is a chain of `n` rigid links with a quadrotor bolted to chosen
vertices (`-a`, default both ends). The whole hose is regulated by one
infinite-horizon LQR about the static trim at the root setpoint, with the flat
map's reference state and wrenches as feed-forward when the trajectory supplies
derivatives to order `2n + 4` (`get_jet`). The LQR needs a vehicle at the last
vertex; `--controller station` flies any pattern with per-vehicle PDs instead.
The controller is derived in {doc}`../theory/controllers/multi-quad-hose`, the
dynamics in {doc}`../theory/dynamics/multi-quad-hose` and the flat map in
{doc}`../theory/differential-flatness/multi-quad-hose`.

## Quadrotors on a hose pinned to a ground anchor

```bash
udaan run tethered-quad -m base -t 10                 # M7: one quadrotor, 2 m tether, 4 links
udaan run tethered-quad --traj circle --no-render     # circle about the anchor, from rest
udaan run tethered-hose -q 3 --traj climb --no-render # M8: three quadrotors on a 3 m hose
udaan generate-xml tethered-hose -n 6 -a 2,4,6 -o tethered.xml
```

M7 is one quadrotor at the top of an `n`-link tether whose bottom end is pinned
to an anchor on the ground; M8 is several quadrotors in series on one such hose,
placed evenly on the line from the anchor to the top vehicle. Both are M6 with
the root vertex held still, flown by M6's hose LQR re-linearised at the pinned
trim and carried round the anchor by its rotational symmetry. A trajectory is
planned in the anchor-link force (the flat output), so the vehicle paths are
outputs: `hold` is the static trim at `--standoff` and `--height` (fractions of
the hose length), `circle` turns the formation about the anchor after a 6 s ramp
from rest, and `climb` blends to a higher, closer station over 6 s. Contacts are
off, and `simulate()` warns when a reference takes the hose below the ground. The
dynamics are in {doc}`../theory/dynamics/tethered-hose`, the controller in
{doc}`../theory/controllers/tethered-hose` and the flat map in
{doc}`../theory/differential-flatness/tethered-hose`.

## Side-by-side fleets

Two fleet commands compare controllers or gains, every vehicle in its own lane
along x:

```bash
udaan run fleet --demo l1-comparison           # PD (clean), PD (disturbed), L1 (disturbed)
udaan run fleet --demo gain-sweep              # four PD gain scales
udaan run fleet -n 4 --trail                   # four quadrotors, default controllers
udaan run cspayload-fleet --demo same-gains    # two quadrotor-payload agents, default gains
udaan run cspayload-fleet --demo gain-sweep    # cable gains × 0.5, 1, 1.5, 2
```

In `l1-comparison` two of the three quadrotors carry an unmodelled 0.25 kg mass
off their centre; the PD one drifts and the L1 one holds its setpoint.

## Input paths, loop rates and rotor layout

These settings are constructor keywords; the defaults suit every command above.

### Input and force types

`QuadrotorBase`, `QuadrotorCsPayloadBase`, `MultiQuadCsPayloadBase` and
`MultiQuadRigidbodyBase` accept several input repackagings, chosen at construction time via the `input=...` kwarg:

| Input type | Controller produces | Integrator consumes | Use when |
|---|---|---|---|
| `acceleration` (default) | 3-vec desired thrust force | wrench via geometric attitude controller | high-level trajectory tracking |
| `wrench` | 4-vec `[f, M_x, M_y, M_z]` | wrench directly | custom attitude laws, SysID |
| `prop_forces` | 4-vec per-rotor forces | allocated wrench | motor-level experiments |

On `QuadrotorBase`, `simulate()` flies all three paths with the same position
controller, the same attitude controller and the same flatness feedforward, and
every path closes the attitude loop inside `step()` at the same rate on both
backends; only how its wrench reaches the integrator differs. So a trajectory
bound to `position_controller.setpoint` (and a `yaw_setpoint`) is flown on every
path.

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

### Cable models (payload only)

| `cable_model` | MuJoCo backend | Captures slack? | Notes |
|---|---|:---:|---|
| `tendon` (default) | massless spatial tendon with a length limit | ✓ | holds its length; carries no force when slack |
| `links` | 10-link chain of ball joints | ✓ | carries the cable's own 10 g |

## Scripted (Python)

A setpoint is any callable `t -> (position, velocity, acceleration)`, each a
3-vector. A trajectory object whose `get` is bound also exposes higher
derivatives (`get_full`, `get_jet`), which turns on the flatness feed-forward:

```python
import numpy as np

from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco
from udaan.utils.trajectory import CircularTraj

mdl = QuadrotorCsPayloadMujoco(render=True, cable_model="links")

# A fixed target ...
mdl.payload_controller.setpoint = lambda t: (np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3))
mdl.simulate(tf=8.0, payload_position=[1, 1, 0.5])

# ... or a trajectory with feed-forward: a 1 m radius circle from rest.
circle = CircularTraj(center=np.array([0.0, 0.0, 1.5]), radius=1.0, speed=1.0, ramp_time=2.0)
mdl.payload_controller.setpoint = circle.get_full
mdl.simulate(tf=12.0, payload_position=circle.get(0.0)[0])
```
