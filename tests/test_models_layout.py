"""The ``udaan.models.mujoco`` compatibility import paths resolve, to the same objects.

The MuJoCo plumbing and the multi-agent models live in
:mod:`udaan.models.backend` and :mod:`udaan.models.fleet`;
``udaan.models.mujoco`` is a compatibility *module* re-exporting them.

A module cannot have submodules, so the three dotted paths —

    udaan.models.mujoco.quadrotor_fleet
    udaan.models.mujoco.quadrotor_cspayload_fleet
    udaan.models.mujoco.quadrotor_comparison

— would raise ``ModuleNotFoundError: 'udaan.models.mujoco' is not a package``
unless they are aliased into ``sys.modules``. These tests pin both halves of
the compatibility contract — the paths resolve, and they resolve to the *same*
objects :mod:`udaan.models.fleet` holds, so ``isinstance`` and ``is`` work
across the two names.
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


def test_a_from_import_against_a_compatibility_submodule_path_works():
    """``from udaan.models.mujoco.<submodule> import ...`` resolves to the canonical class."""
    from udaan.models.fleet import QuadrotorFleet as Canonical
    from udaan.models.mujoco.quadrotor_fleet import QuadrotorFleet

    assert QuadrotorFleet is Canonical
