# Getting started

## Install

```bash
pip install udaan
```

For development, include the optional extras:

```bash
git clone https://github.com/vkotaru/udaan.git
cd udaan
pip install -e ".[all]"
```

## Your first simulation

```python
from udaan.models.quadrotor import QuadrotorMujoco

mdl = QuadrotorMujoco(render=True)
mdl.simulate(tf=10.0, position=[1.0, 1.0, 0.0])
```

This spawns a single quadrotor in a MuJoCo viewer at `[1, 1, 0]` and flies it
to its default setpoint, a hover at `[0, 0, 1]`. The default controller is the geometric
SE(3) controller described in {doc}`../theory/controllers/quadrotor-se3`.

## Next steps

- {doc}`running-simulations` — the eight systems, the CLI and scripted runs.
- {doc}`controllers` — the controller each system ships with, and how to swap it.
- {doc}`../theory/dynamics/quadrotor` — equations of motion behind the model.
