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

This spawns a single quadrotor in a MuJoCo viewer and commands it to hover at
the origin starting from `[1, 1, 0]`. The default controller is the geometric
SE(3) controller described in {doc}`../theory/controllers/quadrotor-se3`.

## Next steps

- {doc}`running-simulations` — CLI reference and scripted runs.
- {doc}`../theory/dynamics/quadrotor` — equations of motion behind the model.
