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
udaan run quadrotor -t 10                             # quadrotor with geometric control
udaan run quad-payload -t 10 -m tendon                # quadrotor + cable-suspended payload
udaan run multi-quad -n 3 -t 10                       # multi-quadrotor cooperative payload
udaan run multi-quad-rigid -t 10                      # multi-quadrotor rigid-body payload
udaan run fleet --demo l1-comparison -t 10            # L1 vs PD controller comparison
udaan run fleet --demo gain-sweep -t 10               # PD gain tuning comparison
udaan run quadrotor -t 5 -r out.gif                   # record to GIF
```

### Python

```python
import udaan as U
import numpy as np

mdl = U.models.mujoco.Quadrotor(render=True)
mdl.simulate(tf=10, position=np.array([1., 1., 0.]))
```

## Documentation

Full API reference, controller roadmap, and tutorials coming soon at [udaan.readthedocs.io](https://udaan.readthedocs.io).

## License

BSD 3-Clause License. See [LICENSE](https://github.com/vkotaru/udaan/blob/main/LICENSE) for details.
