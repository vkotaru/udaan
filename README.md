<p align="center">
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/logo.png" width="360" alt="udaan — aerial robotics framework"/>
</p>

<p align="center">
  <strong>MuJoCo-backed models, geometric controllers and differential-flatness maps for quadrotors carrying cable-suspended payloads.</strong>
</p>

<p align="center">
  <a href="https://github.com/vkotaru/udaan/actions"><img src="https://img.shields.io/github/actions/workflow/status/vkotaru/udaan/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://github.com/vkotaru/udaan/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square" alt="License"></a>
  <a href="https://pypi.org/project/udaan/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://mujoco.org"><img src="https://img.shields.io/badge/MuJoCo-3.0%2B-76B900?style=flat-square" alt="MuJoCo"></a>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/quadrotor.gif" width="250" alt="Quadrotor"/>
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/quad_payload_tendon.gif" width="250" alt="Quad + Payload (tendon)"/>
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/quad_payload_links.gif" width="250" alt="Quad + Payload (links)"/>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/multi_quad_pointmass.gif" width="250" alt="Multi-Quad Pointmass"/>
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/multi_quad_rigid.gif" width="250" alt="Multi-Quad Rigidbody"/>
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/fleet_l1.gif" width="250" alt="Fleet L1 Comparison"/>
</p>

---

> Developed as part of the thesis: _Dynamics and Control for Collaborative Aerial Manipulation_ ([Kotaru, 2022](https://github.com/vkotaru)).
> Original research code: [vkotaru/floating_models](https://github.com/vkotaru/floating_models). This package is the cleaned-up public release, refactored with [Claude](https://claude.ai).

&#x1F681; Quadrotor dynamics &nbsp;&bull;&nbsp; &#x1F39B; Geometric control on SE(3)/SO(3) &nbsp;&bull;&nbsp; &#x26D3; Cable-suspended payloads &nbsp;&bull;&nbsp; &#x1F91D; Multi-quad cooperative transport &nbsp;&bull;&nbsp; &#x1F4D0; Differential flatness &nbsp;&bull;&nbsp; &#x1F579; MuJoCo simulation &nbsp;&bull;&nbsp; &#x1F5A5; CLI interface

## Systems

Six systems, each with analytical dynamics, a MuJoCo backend, a controller and a
differential-flatness flat-to-state map.

| System | Package | CLI | Description | Flat output |
|---|---|---|---|---|
| Quadrotor | `models.quadrotor` | `quadrotor` | Single rigid-body quadrotor | `(x_Q, psi)` |
| Quadrotor + payload | `models.quadrotor_cspayload` | `quad-payload` | One quadrotor, cable-suspended point-mass payload | `(x_L, psi)` |
| Multi-quad + payload | `models.multi_quad_cspayload` | `multi-quad` | `nQ` quadrotors sharing a point-mass payload | `(x_L, {psi_i}, cable forces)` |
| Multi-quad + rigid body | `models.multi_quad_rigidbody` | `multi-quad-rigid` | `nQ` quadrotors sharing a rigid-body payload | `(x_L, R_L, {psi_i}, lambda)` |
| Quadrotor + flexible cable | `models.quadrotor_flexcable` | `quad-flexcable` | One quadrotor, `n`-link flexible cable | `(x_L, psi)` |
| Multi-quad hose | `models.multi_quad_hose` | `hose` | `nQ` quadrotors in series along one `n`-link hose | `(x_0, {tau_a}, {psi_k})` |

## Installation

```bash
pip install udaan
```

MuJoCo is included as a core dependency. Install all extras (dev, docs, RL):

```bash
pip install udaan[all]
```

For development:

```bash
git clone https://github.com/vkotaru/udaan.git
cd udaan
pip install -e ".[all]"
```

## Quick Start

### CLI

```bash
# Quadrotor with geometric SE(3) control
udaan run quadrotor                                    # MuJoCo (default)
udaan run quadrotor -m base                            # pure dynamics (no viz)

# Trajectory tracking
udaan run quadrotor --traj hover -p 1,1,0              # hover (default)
udaan run quadrotor --traj spiral -p 0,0,2             # helical spiral
udaan run quadrotor --traj lissajous -p 0,0,2          # 3D Lissajous
udaan run quadrotor --traj circle -p 0,0,1             # circular

# Cable-suspended payload
udaan run quad-payload -t 10 -c tendon                 # spatial tendon cable
udaan run quad-payload -t 10 -c links                  # rigid link chain
udaan run quad-payload -t 10 -c cable                  # composite cable (experimental)

# Flexible (n-link) cable-suspended payload
udaan run quad-flexcable -n 3 -t 10                    # payload LQR, hover setpoint
udaan run quad-flexcable -n 3 -t 20 --traj lissajous   # + differential-flatness feedforward
udaan run quad-flexcable -n 2 -t 10 -m base            # pure dynamics (no viz)

# Multi-quadrotor cooperative transport
udaan run multi-quad -n 3 -t 10                        # N-quad pointmass payload
udaan run multi-quad-rigid -n 4 -t 10                  # N-quad rigid-body payload

# Several quadrotors carrying one flexible hose
udaan run hose -n 4 -t 10                              # hose LQR, root step setpoint
udaan run hose -n 4 -a 0,2,4 --traj lissajous          # 3 quads + flatness feedforward
udaan run hose -n 4 -t 10 -c station -m base           # per-vehicle station keeping

# Fleet: compare controllers side-by-side
udaan run fleet --demo l1-comparison                   # L1 adaptive vs PD (L1 is experimental)
udaan run fleet --demo gain-sweep                      # PD gain comparison
udaan run fleet -n 4 --trail                           # 4 quads with trails

# Cable-payload fleet: compare payload controllers / gains side-by-side
udaan run cspayload-fleet --demo same-gains            # 2 agents, default gains
udaan run cspayload-fleet --demo gain-sweep            # cable kp/kd × [0.5, 1, 1.5, 2]
udaan run cspayload-fleet --demo gain-sweep --same-start  # overlap start, gain-driven divergence
udaan run cspayload-fleet -n 4                         # 4 agents, default gains

# Recording
udaan run quadrotor -t 5 -r out.gif                    # save to GIF
udaan run quadrotor --traj spiral -r spiral.mp4        # save to MP4
```

### Python

```python
from udaan.models.quadrotor import QuadrotorBase, QuadrotorMujoco

# Pure dynamics (no rendering)
mdl = QuadrotorBase()
mdl.simulate(tf=10, position=[1.0, 1.0, 0.0])

# MuJoCo with visualization
mdl = QuadrotorMujoco(render=True)
mdl.simulate(tf=10, position=[1.0, 1.0, 0.0])
```

Every system follows the same shape — `base.py` holds the pure analytical
dynamics and `mujoco.py` subclasses it to delegate to the physics backend:

```python
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase

mdl = MultiQuadCsPayloadBase(num_quadrotors=4)
mdl.cable_lengths = [1.0, 1.2, 1.0, 1.2]
mdl.simulate(tf=5.0, payload_position=[0.0, 0.0, 1.0])
```

### Differential flatness

Each system has a flat-to-state map that recovers the full reference state and
the feedforward inputs from the flat output and its time derivatives. Flat
outputs are carried as `Jet` bundles — an `(order+1, dim)` array whose row *k* is
the *k*-th derivative, with no factorial scaling. Note `udaan.flatness` is not
re-exported from the top-level package; import it explicitly.

```python
import numpy as np
from udaan.flatness import Jet, Quadrotor

flat_map = Quadrotor(mass=0.9, inertia=np.diag([0.0023, 0.0023, 0.004]))

# x through snap (order 4), psi through psi-ddot (order 2)
x = Jet(np.zeros((5, 3)))
psi = Jet(np.zeros((3, 1)))

ref, inputs = flat_map.recover(Quadrotor.Flats(x=x, psi=psi))
print(ref.position, ref.orientation, inputs.thrust, inputs.moment)
```

## Documentation

[udaan.readthedocs.io](https://udaan.readthedocs.io) — guides, the full API
reference, and theory pages deriving the dynamics, controllers and
differential-flatness maps for each of the six systems.

## Examples

Runnable, headless examples for every system live in
[`examples/`](https://github.com/vkotaru/udaan/tree/main/examples).
To run the whole set and check that each one passes:

```bash
scripts/run_examples.sh
```

## Citation

If you use this package in academic work, please cite it. Metadata is in
[`CITATION.cff`](https://github.com/vkotaru/udaan/blob/main/CITATION.cff);
GitHub's *Cite this repository* button will render it as BibTeX or APA.

## License

BSD 3-Clause License. See [LICENSE](https://github.com/vkotaru/udaan/blob/main/LICENSE) for details.
