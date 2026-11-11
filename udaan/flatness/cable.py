"""Cable-kinematics jets shared across cable-suspended-load flatness maps.

A taut, massless cable transmits a force ``F = T q`` where ``T ≥ 0`` is the
scalar tension and ``q`` is the unit direction the cable pulls along (this
codebase's convention: ``q`` points from the quadrotor to the load).  Flatness
maps for cable-suspended systems repeatedly need the time-derivative tuples
("jets") of ``T`` and ``q`` given the cable-force jet.  That single operation is
:func:`tension_direction_jet`, built on the shared Leibniz engine
(:func:`udaan.flatness.derivatives.normalize_derivatives`).

The *force* jet itself is model-specific:

* a single point-mass payload — this repo's cable-suspended-payload system, and
  each independent link of a multi-quad point-mass system — obeys Newton's law
  ``F = -m_L (a_L + g e3)``: :func:`point_mass_cable_force`;
* a rigid-body payload carried by several cables is over-actuated, so the
  per-cable force comes from a wrench→tension allocation (a separate primitive)
  in which the tension is a *free / flat* quantity; the resulting force jet
  decomposes with the very same :func:`tension_direction_jet`.

Because the input is the force vector rather than the load acceleration,
:func:`tension_direction_jet` serves both regimes — it does not assume the
tension is determined by the load dynamics.
"""

from __future__ import annotations

import numpy as np

from ..core.defaults import GRAVITY
from .derivatives import normalize_derivatives

__all__ = ["tension_direction_jet", "point_mass_cable_force", "cable_direction_jet"]

_E3 = np.array([0.0, 0.0, 1.0])


def tension_direction_jet(
    cable_force: list[np.ndarray],
) -> tuple[list[float], list[np.ndarray]]:
    """Tension-magnitude and unit-direction jets of a taut cable.

    Given the derivative tuple ``cable_force = [F, Ḟ, …, F^(K)]`` of the cable
    force vector ``F = T q`` (scalar tension ``T ≥ 0``, unit pull direction
    ``q``), return ``(T, q)`` where ``T[k] = (d/dt)^k T`` (scalar) and
    ``q[k] = (d/dt)^k q`` (vector) for ``k = 0 … K``.

    This is the general "tension→direction jet": it takes the cable *force* as
    input, so it applies whether the force is fixed by the load dynamics (single
    payload) or chosen by a wrench→tension allocation (rigid-body / multi-cable,
    where the tension is a flat output).  It is a thin cable-semantics wrapper
    over :func:`~udaan.flatness.derivatives.normalize_derivatives`.

    Raises
    ------
    ValueError
        On a slack cable (``‖F‖ ≈ 0``): the pull direction is undefined.
    """
    return normalize_derivatives(cable_force)


def point_mass_cable_force(
    payload_accel: list[np.ndarray], mass_l: float, gravity: float = GRAVITY
) -> list[np.ndarray]:
    """Cable-force jet holding up a point-mass load, from its acceleration jet.

    Newton's law on a point mass ``m_L`` gives the cable force
    ``F = -m_L (a_L + g e3)`` (the cable pulls the load, ``q`` points
    quadrotor→load).  Given ``payload_accel = [a_L, ȧ_L, …, a_L^(K)]`` return
    the force derivative tuple ``[F, Ḟ, …, F^(K)]`` (gravity is constant, so it
    contributes only to the zeroth term).
    """
    force = [-mass_l * (payload_accel[0] + gravity * _E3)]
    for k in range(1, len(payload_accel)):
        force.append(-mass_l * payload_accel[k])
    return force


def cable_direction_jet(
    payload_accel: list[np.ndarray], mass_l: float, gravity: float = GRAVITY
) -> tuple[list[float], list[np.ndarray]]:
    """Tension and unit-direction jets for a point-mass cable-suspended payload.

    Convenience composition of :func:`point_mass_cable_force` and
    :func:`tension_direction_jet` — the single-payload specialisation used by the
    quadrotor / cable-suspended-payload map and controller.  See those functions
    for the convention and the free-fall (slack-cable) :class:`ValueError`.
    """
    return tension_direction_jet(point_mass_cable_force(payload_accel, mass_l, gravity))
