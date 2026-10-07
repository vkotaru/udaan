"""Compatibility re-exports for the old ``udaan.models.mujoco`` package.

This name used to be a package holding two unrelated kinds of thing: the MuJoCo
*infrastructure* (``MujocoModel``, the GLFW viewer) that every per-system
backend imports, and the multi-agent fleet *models*. Each has a home of its own
now:

* :mod:`udaan.models.backend` — ``MujocoModel`` and the viewer.
* :mod:`udaan.models.fleet` — ``QuadrotorFleet``, ``QuadrotorCsPayloadFleet``,
  ``QuadrotorComparison``.

Everything that was importable from ``udaan.models.mujoco`` still is, including
the ``Quadrotor`` alias for
:class:`~udaan.models.quadrotor.QuadrotorMujoco`. New code should import from
the module that actually owns the name.

That "everything" includes the three *submodule paths* the old package
published — ``udaan.models.mujoco.quadrotor_fleet``,
``udaan.models.mujoco.quadrotor_cspayload_fleet`` and
``udaan.models.mujoco.quadrotor_comparison``. This is a module now, not a
package, so ``from udaan.models.mujoco.quadrotor_fleet import QuadrotorFleet``
would raise ``ModuleNotFoundError: 'udaan.models.mujoco' is not a package``
unless the old dotted names are registered by hand. :data:`sys.modules` entries
below do exactly that, pointing at the ``udaan.models.fleet`` module objects, so
the old paths keep working and ``sys.modules`` never holds two copies of the
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

#: Old dotted path -> the module that owns it now. Aliased into
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
