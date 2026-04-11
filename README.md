<p align="center">
  <img src="https://raw.githubusercontent.com/vkotaru/udaan/main/.media/logo.png" width="360" alt="udaan — aerial robotics framework"/>
</p>

<p align="center">
  <strong>A Python MuJoCo-based models and controllers for quadcopter cable-suspended payload systems.</strong>
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

&#x1F681; Quadrotor dynamics &nbsp;&bull;&nbsp; &#x1F39B; Geometric control on SE(3)/SO(3) &nbsp;&bull;&nbsp; &#x26D3; Cable-suspended payloads &nbsp;&bull;&nbsp; &#x1F91D; Multi-quad cooperative transport &nbsp;&bull;&nbsp; &#x1F579; MuJoCo simulation &nbsp;&bull;&nbsp; &#x1F5A5; CLI interface

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
udaan run quadrotor -m vfx                             # VPython visualization

# Trajectory tracking
udaan run quadrotor --traj hover -p 1,1,0              # hover (default)
udaan run quadrotor --traj spiral -p 0,0,2             # helical spiral
udaan run quadrotor --traj lissajous -p 0,0,2          # 3D Lissajous
udaan run quadrotor --traj circle -p 0,0,1             # circular

# Cable-suspended payload
udaan run quad-payload -t 10 -m tendon                 # tendon model
udaan run quad-payload -t 10 -m links                  # rigid links

# Multi-quadrotor cooperative transport
udaan run multi-quad -n 3 -t 10                        # N-quad pointmass payload
udaan run multi-quad-rigid -t 10                       # rigid-body payload

# Fleet: compare controllers side-by-side
udaan run fleet --demo l1-comparison                   # L1 adaptive vs PD
udaan run fleet --demo gain-sweep                      # PD gain comparison
udaan run fleet -n 4 --trail                           # 4 quads with trails

# Recording
udaan run quadrotor -t 5 -r out.gif                    # save to GIF
udaan run quadrotor --traj spiral -r spiral.mp4        # save to MP4
```

### Python

```python
from udaan.models.quadrotor import QuadrotorBase, QuadrotorMujoco

# Pure dynamics (no rendering)
mdl = QuadrotorBase()
mdl.simulate(tf=10, position=[1., 1., 0.])

# MuJoCo with visualization
mdl = QuadrotorMujoco(render=True)
mdl.simulate(tf=10, position=[1., 1., 0.])
```

## Documentation

Full API reference, controller roadmap, and tutorials coming soon at [udaan.readthedocs.io](https://udaan.readthedocs.io).

## License

BSD 3-Clause License. See [LICENSE](https://github.com/vkotaru/udaan/blob/main/LICENSE) for details.
