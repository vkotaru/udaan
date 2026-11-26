# Controllers

Each model in `udaan` ships with a default controller stack. You can swap or
extend any controller in-place on a live model instance — the base classes
expose `position_controller`, `attitude_controller`, and (for payload models)
`payload_controller` properties.

## Quadrotor (M1)

| Controller | Class | Role | Notes |
|---|---|---|---|
| Geometric SE(3) attitude | {py:class}`~udaan.control.quadrotor.GeometricAttitudeController` | inner attitude loop | default; see {doc}`../theory/controllers/quadrotor-se3` |
| Geometric L1-adaptive attitude | {py:class}`~udaan.control.quadrotor.GeometricL1AttitudeController` | inner attitude loop with disturbance estimator | robust to mass/inertia error |
| PD position | {py:class}`~udaan.control.quadrotor.PositionPDController` | outer position loop | default; tracks $x^d(t)$ |
| L1-adaptive position | {py:class}`~udaan.control.quadrotor.PositionL1Controller` | PD plus piecewise-constant L1 disturbance estimate | mass error, constant forces; see {doc}`../theory/controllers/geometric-l1` for its bandwidth caveat |
| Direct propeller-force | {py:class}`~udaan.control.quadrotor.DirectPropellerForceController` | motor-level passthrough | SysID / open-loop experiments |

## Quadrotor with cable-suspended payload (M2)

| Controller | Class | Role | Notes |
|---|---|---|---|
| Cascaded geometric payload | {py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController` | payload position → cable attitude → quad attitude | default; gains from the time-scale design in {py:mod}`udaan.control.quadrotor_cspayload.gains`; see {doc}`../theory/controllers/quadrotor-payload` |

## Several quadrotors, one payload (M3, M4)

| Controller | Class | Role | Notes |
|---|---|---|---|
| Cooperative point-mass cascade | {py:class}`~udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController` | payload position → tension split → each cable → each airframe | M3 default; the split is a `TensionSplitPolicy`; see {doc}`../theory/controllers/multi-quad-payload` |
| Rigid-body payload cascade | {py:class}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController` | payload pose → wrench allocation → each cable → each airframe | M4 default; `attitude_setpoint` sets the payload attitude; derived with the dynamics, in {doc}`../theory/dynamics/multi-quad-rigidbody` |

Both are read and replaced through `payload_controller`; each airframe's
attitude loop is the M1 geometric controller.

## Flexible cable, hose and tether (M5–M8)

| Controller | Class | Role | Notes |
|---|---|---|---|
| Chain-aware LQR | {py:class}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController` | payload position through the `n`-link cable | M5 default, on `payload_controller`; see {doc}`../theory/controllers/quadrotor-flexcable` |
| Hose LQR | {py:class}`~udaan.control.multi_quad_hose.MultiQuadHoseController` | the whole hose and every vehicle about the static trim | M6 default, on `hose_controller`; see {doc}`../theory/controllers/multi-quad-hose` |
| Pinned-hose LQR | {py:class}`~udaan.control.tethered_hose.TetheredHoseController` | the tether and its vehicles, carried round the anchor | M7 and M8 default, on `hose_controller`; see {doc}`../theory/controllers/tethered-hose` |

These systems have one actuated body per vehicle on a serial chain, so a single
LQR on the whole chain replaces the geometric cascade. Each emits a wrench, and
takes the flat map's reference state and wrench as feed-forward when the
trajectory exposes `get_jet`.

## Swapping a controller

All controllers follow a `compute(t, state) -> input` contract, so swapping
is a single assignment. A setpoint is a callable `t -> (position, velocity,
acceleration)`:

```python
import numpy as np

from udaan.control.quadrotor import GeometricL1AttitudeController, PositionL1Controller
from udaan.models.quadrotor import QuadrotorMujoco

def my_traj(t):
    return np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3)

mdl = QuadrotorMujoco(render=True)
mdl.position_controller = PositionL1Controller(mass=mdl.mass, setpoint=my_traj)
mdl.attitude_controller = GeometricL1AttitudeController(
    mass=mdl.mass, inertia=mdl.inertia,
)
mdl.simulate(tf=10.0, position=[1.0, 1.0, 0.0])
```

## Tuning gains

Gains are constructor keywords. The M1 position loop takes `kp` and `kd`; the
M2 payload controller takes `position_kp`, `position_kd`, `cable_kp` and
`cable_kd`, and any it is not given come from the time-scale design for the
model's machine:

```python
import numpy as np

from udaan.control.quadrotor_cspayload import QuadCSPayloadController
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase

mdl = QuadrotorCsPayloadBase()
controller = QuadCSPayloadController(cable_kp=np.array([30.0, 30.0, 30.0]))
controller.sync_from_model(mdl)  # the model's masses, cable length and inertia
mdl.payload_controller = controller
print(mdl.payload_controller.position_gains, mdl.payload_controller.cable_gains)
```

The LQR controllers (M5–M8) take their state and input weights as keywords
(`q_link`, `q_attitude`, `r_thrust`, `r_moment`, …) and solve the gain for the
model's machine.

## Writing your own controller

Subclass `Controller` (or one of the existing PD/L1 bases) and implement
`compute(self, t, state) -> np.ndarray`. Any instance can then be assigned
to the model's controller property. See
{py:class}`~udaan.control.quadrotor.PositionPDController` for the simplest
template.
