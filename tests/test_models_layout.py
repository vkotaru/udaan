"""The old ``udaan.models.mujoco`` import paths still resolve, to the same objects.

``udaan.models.mujoco`` used to be a *package* holding the MuJoCo plumbing and
the multi-agent models together. Those now live in :mod:`udaan.models.backend`
and :mod:`udaan.models.fleet`, and the old name is a compatibility *module*.

A module cannot have submodules, so the three dotted paths the old package
published —

    udaan.models.mujoco.quadrotor_fleet
    udaan.models.mujoco.quadrotor_cspayload_fleet
    udaan.models.mujoco.quadrotor_comparison

— would raise ``ModuleNotFoundError: 'udaan.models.mujoco' is not a package``
unless they are aliased into ``sys.modules``. They were public: the docs shipped
an autodoc stub for each. These tests pin both halves of the promise the
CHANGELOG makes — the paths resolve, and they resolve to the *same* objects the
new homes hold, so ``isinstance`` and ``is`` keep working across the two names.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.mujoco

SUBMODULES = [
    ("quadrotor_fleet", "QuadrotorFleet"),
    ("quadrotor_cspayload_fleet", "QuadrotorCsPayloadFleet"),
    ("quadrotor_comparison", "QuadrotorComparison"),
]


@pytest.mark.parametrize("submodule,name", SUBMODULES)
def test_the_old_submodule_path_imports(submodule, name):
    module = importlib.import_module(f"udaan.models.mujoco.{submodule}")
    assert hasattr(module, name)


@pytest.mark.parametrize("submodule,name", SUBMODULES)
def test_it_is_the_same_module_object_as_the_new_home(submodule, name):
    """An alias, not a second copy — two copies would break ``isinstance``."""
    old = importlib.import_module(f"udaan.models.mujoco.{submodule}")
    new = importlib.import_module(f"udaan.models.fleet.{submodule}")
    assert old is new
    assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize(
    "name",
    [
        "MujocoModel",
        "Quadrotor",
        "QuadrotorComparison",
        "QuadrotorCsPayloadFleet",
        "QuadrotorFleet",
    ],
)
def test_every_top_level_name_survived(name):
    import udaan.models.mujoco as shim

    assert name in shim.__all__
    assert getattr(shim, name) is not None


def test_the_canonical_objects_are_the_same_ones():
    import udaan.models.backend as backend
    import udaan.models.fleet as fleet
    import udaan.models.mujoco as shim
    from udaan.models.quadrotor.mujoco import QuadrotorMujoco

    assert shim.MujocoModel is backend.MujocoModel
    assert shim.QuadrotorFleet is fleet.QuadrotorFleet
    assert shim.QuadrotorCsPayloadFleet is fleet.QuadrotorCsPayloadFleet
    assert shim.QuadrotorComparison is fleet.QuadrotorComparison
    assert shim.Quadrotor is QuadrotorMujoco


def test_a_from_import_against_the_old_path_works():
    """The form the retired docs taught, verbatim."""
    from udaan.models.fleet import QuadrotorFleet as Canonical
    from udaan.models.mujoco.quadrotor_fleet import QuadrotorFleet

    assert QuadrotorFleet is Canonical
