"""Compatibility re-exports under the ``udaan.models.mujoco`` name.

The names exported here are owned by two modules:

* :mod:`udaan.models.backend` — ``MujocoModel`` and the viewer.
* :mod:`udaan.models.fleet` — ``QuadrotorFleet``, ``QuadrotorCsPayloadFleet``,
  ``QuadrotorComparison``.

plus the ``Quadrotor`` alias for :class:`~udaan.models.quadrotor.QuadrotorMujoco`.
New code should import from the module that owns the name.

The three submodule paths ``udaan.models.mujoco.quadrotor_fleet``,
``udaan.models.mujoco.quadrotor_cspayload_fleet`` and
``udaan.models.mujoco.quadrotor_comparison`` also resolve. This is a module, not
a package, so ``from udaan.models.mujoco.quadrotor_fleet import QuadrotorFleet``
would raise ``ModuleNotFoundError: 'udaan.models.mujoco' is not a package``
unless the dotted names are registered by hand. The :data:`sys.modules` entries
below do that, pointing at the ``udaan.models.fleet`` module objects, so the
dotted paths resolve and ``sys.modules`` never holds two copies of the
same module. The import machinery consults ``sys.modules`` for ``a.b.c`` right
after importing the parent ``a.b``, which is this module — so registering here
is early enough.
"""

import sys

from . import fleet as _fleet
from .backend import MujocoModel as MujocoModel
from .fleet import QuadrotorComparison as QuadrotorComparison
from .fleet import QuadrotorCsPayloadFleet as QuadrotorCsPayloadFleet
from .fleet import QuadrotorFleet as QuadrotorFleet
from .quadrotor.mujoco import QuadrotorMujoco as Quadrotor

__all__ = [
    "MujocoModel",
    "Quadrotor",
    "QuadrotorComparison",
    "QuadrotorCsPayloadFleet",
    "QuadrotorFleet",
]

#: Submodule name under this module -> the module that owns it. Aliased into
#: :data:`sys.modules` so ``import udaan.models.mujoco.quadrotor_fleet`` and
#: ``from ... import ...`` against it both resolve.
_SUBMODULE_ALIASES = {
    "quadrotor_comparison": _fleet.quadrotor_comparison,
    "quadrotor_cspayload_fleet": _fleet.quadrotor_cspayload_fleet,
    "quadrotor_fleet": _fleet.quadrotor_fleet,
}

for _name, _module in _SUBMODULE_ALIASES.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)
    globals().setdefault(_name, _module)
del _name, _module
