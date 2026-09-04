"""Multi-agent MuJoCo models: fleets of independent vehicles, and comparison rigs.

These are *models*, not backend plumbing — the distinction that this package
exists to make. Each one loads a single MJCF holding several vehicles and drives
them from a list of per-agent :mod:`udaan.models.quadrotor` /
:mod:`udaan.models.quadrotor_cspayload` state-and-controller objects:

* :class:`QuadrotorFleet` — ``n`` independent quadrotors, one colour and one
  controller each, for side-by-side controller comparison.
* :class:`QuadrotorCsPayloadFleet` — the same, with a cable-suspended payload
  per agent.
* :class:`QuadrotorComparison` — a plant and a reference quadrotor in one scene.

They differ from the per-system backends in :mod:`udaan.models.quadrotor` and
friends in that they have no analytical counterpart: there is no
``FleetBase`` to agree with, and no flatness map. They share the small
:class:`~udaan.models.fleet._base.BaseModel` rather than a per-system
``base.py``.

The MuJoCo plumbing they sit on is :mod:`udaan.models.backend`.
"""

from ._base import BaseModel as BaseModel
from .quadrotor_comparison import QuadrotorComparison as QuadrotorComparison
from .quadrotor_cspayload_fleet import QuadrotorCsPayloadFleet as QuadrotorCsPayloadFleet
from .quadrotor_fleet import QuadrotorFleet as QuadrotorFleet

__all__ = [
    "BaseModel",
    "QuadrotorComparison",
    "QuadrotorCsPayloadFleet",
    "QuadrotorFleet",
]
