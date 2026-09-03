"""Shared fixtures and tolerance constants for the ``udaan`` test suite.

Importing the constants
-----------------------

``tests/`` is a package (it has an ``__init__.py``), so the usual bare
``from conftest import TOL_ANALYTIC`` raises ``ModuleNotFoundError`` at
collection time. Use the package path::

    from tests.conftest import TOL_ANALYTIC

Fixtures need no import — pytest injects them by name.

Tolerance ladder
----------------

Pick the loosest rung that still fails on a real regression. The values are
calibrated against how this suite actually behaves; the counts below are how
many existing assertions already sit on each rung.

===================  ======  ==========================================================
``TOL_EXACT``        1e-15   Algebraic identities exact up to float round-off: hat/vee
                             round-trips, ``SO3`` orthonormality, ``Jet`` row indexing.
                             (22 sites)
``TOL_TIGHT``        1e-12   Short closed-form chains — Leibniz products, derivative
                             normalisation, single-step manifold retractions. (9 sites)
``TOL_ANALYTIC``     1e-10   The workhorse rung: flat-to-state recovery against an
                             independently derived analytic expression, and anything
                             routed through ``sin``/``cos`` or a matrix exponential.
                             (67 sites)
``TOL_INTEGRATION``  1e-6    Residual after O(1e3) integration steps — the ZOH-Euler
                             loop in the base models, or DOP853 via ``solve_ivp``.
``TOL_PHYSICAL``     1e-2    Convergence of a physical quantity over a full simulation:
                             tracking error, hover thrust, settled attitude. (8 sites)
===================  ======  ==========================================================

Scope
-----

Deliberately minimal. Per-system model fixtures are **not** defined here: the
existing M1/M2 tests already build their own under different names (``quad``,
``f2s``, ``model``), so a differently-named duplicate would be dead code, and
the systems added after this point need their own set-up anyway. Add a fixture
here when a second test file genuinely needs the same one.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Tolerance ladder ──────────────────────────────────────────────────
TOL_EXACT = 1e-15
TOL_TIGHT = 1e-12
TOL_ANALYTIC = 1e-10
TOL_INTEGRATION = 1e-6
TOL_PHYSICAL = 1e-2


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so every failure is reproducible."""
    return np.random.default_rng(42)
