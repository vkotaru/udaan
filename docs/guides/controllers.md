# Controllers

Each model in `udaan` ships with a default controller stack. You can swap or
extend any controller in-place on a live model instance — the base classes
expose `position_controller`, `attitude_controller`, and (for payload models)
`payload_controller` properties.

## Quadrotor

| Controller | Class | Role | Notes |
|---|---|---|---|
| Geometric SE(3) attitude | {py:class}`~udaan.control.quadrotor.GeometricAttitudeController` | inner attitude loop | default; see {doc}`../theory/controllers/quadrotor-se3` |
| Geometric L1-adaptive attitude | {py:class}`~udaan.control.quadrotor.GeometricL1AttitudeController` | inner attitude loop with disturbance estimator | robust to mass/inertia error |
| PD position | {py:class}`~udaan.control.quadrotor.PositionPDController` | outer position loop | default; tracks $x^d(t)$ |
| L1-adaptive position | {py:class}`~udaan.control.quadrotor.PositionL1Controller` | outer position loop with disturbance estimator | mass/wind compensation |
| Direct propeller-force | {py:class}`~udaan.control.quadrotor.DirectPropellerForceController` | motor-level passthrough | SysID / open-loop experiments |

## Quadrotor with cable-suspended payload

| Controller | Class | Role | Notes |
|---|---|---|---|
| Cascaded geometric payload | {py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController` | payload position → cable attitude → quad attitude | default; see {doc}`../theory/controllers/quadrotor-payload` |

## Quadrotor with a flexible ($n$-link) cable

| Controller | Class | Role | Notes |
|---|---|---|---|
| Payload-tracking LQR | {py:class}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController` | full-state regulation of $(x_Q, R, \{q_i\})$, feedforwarded by the flat map | default for M5; see {doc}`../theory/controllers/quadrotor-flexcable` |

The odd one out, in three ways. It is **not a cascade** — a serial chain
has no per-link body to command, so the published law is an LQR on the
variational error and the gain comes from a Riccati solve rather than a
gain pair. It emits a **wrench** `[f, M]` rather than a thrust vector, so
`QuadrotorFlexCableBase` defaults to `input="wrench"` and there is no
attitude loop underneath it. And its feedforward needs the payload
position through its $2n+4$-th derivative, so a trajectory bound to
`setpoint` should implement `get_jet(t, order)`; one that stops at
`get_full`'s snap is accepted, zero-padded, and warned about once.

The station-keeping quadrotor position PD is still there as a baseline,
reached through `compute_station_keeping_input()`. It emits a thrust
*vector*, so it runs on the `ACCELERATION` pipeline: construct the model
with `input="acceleration"` and `simulate()` drives that loop instead of
the payload LQR (and, since the loop does not regulate the payload, without
the flat-map feedforward).

## Multi-quadrotor cooperative transport

| Controller | Class | Role | Notes |
|---|---|---|---|
| Point-mass payload cascade | {py:class}`~udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController` | payload position → tension split → per-cable attitude → per-quad attitude | default for M3; the split is a pluggable {py:class}`~udaan.flatness.TensionSplitPolicy` |
| Rigid-body payload cascade | {py:class}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController` | payload **pose** → grasp-map wrench split → per-cable attitude → per-quad attitude | default for M4; needs $n_Q \ge 3$ non-collinear attachments to hold an attitude — see {doc}`../theory/dynamics/multi-quad-rigidbody` |

The rigid-body cascade carries a second reference alongside `setpoint`:
`attitude_setpoint(t) -> (R_d, Ω_d, Ω̇_d)` for the payload's attitude,
which is independent of its position in a way it cannot be for a point
mass. Binding a trajectory's `get_full` to `setpoint` switches on the M4
flatness feedforward, exactly as it does for M3; the reference is
allocated through the cascade's own
{py:attr}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.tension_policy`,
so the closed loop and the feedforward cannot split the wrench
differently.

Both cascades cache their physical parameters (masses, cable lengths,
and for M4 the whole grasp) when they are built. Assigning a parameter
on the model afterwards re-synchronises them, so
`mdl.attachment_points = ...` and passing `attachment_points=` to the
constructor give the same closed loop.

## Swapping a controller

All controllers follow a `compute(t, state) -> input` contract, so swapping
is a single assignment:

```python
from udaan.models.quadrotor import QuadrotorMujoco
from udaan.control.quadrotor import PositionL1Controller, GeometricL1AttitudeController

mdl = QuadrotorMujoco(render=True)
mdl.position_controller = PositionL1Controller(mass=mdl.mass, setpoint=my_traj)
mdl.attitude_controller = GeometricL1AttitudeController(
    mass=mdl.mass, inertia=mdl.inertia,
)
mdl.simulate(tf=10.0, position=[1.0, 1.0, 0.0])
```

## Tuning gains

Each controller exposes its gains either through constructor kwargs or
through a `._gains` attribute. For the payload controller:

```python
import numpy as np

mdl._payload_controller._gain_pos.kp = np.array([4.0, 4.0, 8.0])
mdl._payload_controller._gain_pos.kd = np.array([3.0, 3.0, 6.0])
mdl._payload_controller._gain_cable.kp = np.array([24.0, 24.0, 24.0])
mdl._payload_controller._gain_cable.kd = np.array([8.0, 8.0, 8.0])
```

The defaults ship with the values shown above — verified stable over the
benchmark trajectories used in the test suite.

## Writing your own controller

Subclass `Controller` (or one of the existing PD/L1 bases) and implement
`compute(self, t, state) -> np.ndarray`. Any instance can then be assigned
to the model's controller property. See
{py:class}`~udaan.control.quadrotor.PositionPDController` for the simplest
template.
