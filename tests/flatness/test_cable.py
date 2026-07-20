"""Tests for the shared cable-kinematics primitives (:mod:`udaan.flatness.cable`).

Covers the general :func:`tension_direction_jet` (on a force jet that is *not* of
the point-mass form, to exercise the generalisation), the point-mass Newton
force jet, and the equivalence of the ``cable_direction_jet`` convenience
composition with the earlier inlined behaviour.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from udaan.flatness import (
    cable_direction_jet,
    normalize_derivatives,
    point_mass_cable_force,
    tension_direction_jet,
)
from udaan.flatness import quadrotor_cspayload as qcp


def test_tension_direction_jet_general_force():
    """Decompose an arbitrary cable-force jet ``F = T q`` — one whose tension
    *and* direction both vary, i.e. NOT the point-mass ``-m(a+ge3)`` form — and
    check the defining identities hold at every order: ``F = T·q`` and
    ``‖q‖ ≡ 1``.  This is the 'tension-as-flat-output' regime."""
    # Build a force jet directly (as an allocation would hand us), order K=4.
    rng = np.random.default_rng(7)
    K = 4
    force = [rng.standard_normal(3) for _ in range(K + 1)]
    force[0] += 5.0  # keep ‖F‖ well away from zero

    T, q = tension_direction_jet(force)

    # T[0] = ‖F‖ ≥ 0
    assert T[0] == pytest.approx(float(np.linalg.norm(force[0])))
    assert T[0] > 0

    # F^(k) == sum_j C(k,j) T^(j) q^(k-j)   (Leibniz reconstruction of F = T q)
    for k in range(K + 1):
        recon = sum(comb(k, j) * T[j] * q[k - j] for j in range(k + 1))
        np.testing.assert_allclose(recon, force[k], rtol=0, atol=1e-9)

    # ‖q‖^2 ≡ 1  ->  its derivatives vanish beyond order 0
    qq = [
        sum(comb(k, j) * float(np.dot(q[j], q[k - j])) for j in range(k + 1)) for k in range(K + 1)
    ]
    assert qq[0] == pytest.approx(1.0)
    for k in range(1, K + 1):
        assert qq[k] == pytest.approx(0.0, abs=1e-9)


def test_point_mass_cable_force_newton():
    """F = -m_L (a_L + g e3) at order 0; higher orders = -m_L a_L^(k)."""
    m, g = 0.5, 9.81
    a = [np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.0, -1.0]), np.array([0.0, 2.0, 0.0])]
    force = point_mass_cable_force(a, m, g)
    np.testing.assert_allclose(force[0], -m * (a[0] + g * np.array([0, 0, 1.0])))
    np.testing.assert_allclose(force[1], -m * a[1])
    np.testing.assert_allclose(force[2], -m * a[2])


def test_cable_direction_jet_is_composition():
    """cable_direction_jet == tension_direction_jet ∘ point_mass_cable_force,
    and matches a direct normalize_derivatives of the Newton force (the earlier
    inlined behaviour) exactly."""
    m, g = 0.8, 9.81
    a = [np.array([0.2, 0.1, -0.4]), np.array([0.5, -0.3, 0.1]), np.array([-0.2, 0.0, 0.7])]

    T1, q1 = cable_direction_jet(a, m, g)
    T2, q2 = tension_direction_jet(point_mass_cable_force(a, m, g))
    # legacy path: normalize the hand-built Newton force tuple
    tvec = [-m * (a[0] + g * np.array([0, 0, 1.0]))] + [-m * a[k] for k in range(1, len(a))]
    T3, q3 = normalize_derivatives(tvec)

    np.testing.assert_allclose(T1, T2)
    np.testing.assert_allclose(T1, T3)
    for k in range(len(a)):
        np.testing.assert_allclose(q1[k], q2[k])
        np.testing.assert_allclose(q1[k], q3[k])


def test_reexport_identity():
    """The name re-exported from quadrotor_cspayload is the same object, so the
    controller's existing import keeps working."""
    assert qcp.cable_direction_jet is cable_direction_jet


def test_slack_cable_raises():
    with pytest.raises(ValueError):
        tension_direction_jet([np.zeros(3), np.ones(3)])
    # point-mass payload in free fall: a_L = -g e3  ->  F = 0
    with pytest.raises(ValueError):
        cable_direction_jet([np.array([0.0, 0.0, -9.81]), np.zeros(3)], 1.0, 9.81)
