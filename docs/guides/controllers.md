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
