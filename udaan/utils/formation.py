r"""Ring-formation geometry for quadrotors sharing one cable-suspended payload.

The quadrotors sit evenly on a horizontal ring of radius ``r`` above the
payload, so neighbours are ``2 r sin(π/n_Q)`` apart, centre to centre. Two
airframes whose rotor discs reach :data:`~udaan.core.defaults.DEFAULT_ROTOR_SPAN`
tip to tip need at least that spacing to clear each other, plus a margin
(:data:`~udaan.core.defaults.DEFAULT_FORMATION_CLEARANCE`) because a tilted
airframe's discs move towards its neighbour. A ring is therefore admissible
when::

    2 r sin(π/n_Q) >= span + clearance    and    r < l_i  for every cable i

The second condition is the taut-cable triangle ``r² + dz² = l²``: at
``r >= l`` the cable cannot reach the payload's axis. A lone quadrotor hangs
straight above the payload and has neither constraint.

These helpers are shared by the model (which refuses an inadmissible ring at
construction) and the MJCF generator (which refuses it before writing a file
MuJoCo would fly with airframes inside each other).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.defaults import (
    DEFAULT_FORMATION_CLEARANCE,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_PAYLOAD_SPAWN_HEIGHT,
    DEFAULT_ROTOR_SPAN,
)

__all__ = [
    "check_formation",
    "default_formation_radius",
    "default_spawn_height",
    "min_formation_radius",
    "neighbour_spacing",
]


def neighbour_spacing(num_quadrotors: int, radius: float) -> float:
    """Centre-to-centre distance between neighbours on the ring, ``2 r sin(π/n_Q)``.

    ``inf`` for a single quadrotor, which has no neighbour.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")
    if num_quadrotors == 1:
        return math.inf
    return 2.0 * float(radius) * math.sin(math.pi / num_quadrotors)


def min_formation_radius(
    num_quadrotors: int,
    rotor_span: float = DEFAULT_ROTOR_SPAN,
    clearance: float = DEFAULT_FORMATION_CLEARANCE,
) -> float:
    """Smallest ring radius at which neighbours clear each other.

    ``(rotor_span + clearance) / (2 sin(π/n_Q))``, so that the neighbour spacing
    is the rotor span plus the clearance. ``0`` for a single quadrotor.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")
    if num_quadrotors == 1:
        return 0.0
    return (float(rotor_span) + float(clearance)) / (2.0 * math.sin(math.pi / num_quadrotors))


def default_formation_radius(
    num_quadrotors: int,
    rotor_span: float = DEFAULT_ROTOR_SPAN,
    clearance: float = DEFAULT_FORMATION_CLEARANCE,
) -> float:
    """Default ring radius: ``max(DEFAULT_FORMATION_RADIUS, min_formation_radius(n_Q))``.

    The fixed radius for small rings, grown with ``n_Q`` once neighbours would
    otherwise be closer than the rotor span plus the clearance.
    """
    return max(
        float(DEFAULT_FORMATION_RADIUS),
        min_formation_radius(num_quadrotors, rotor_span, clearance),
    )


def check_formation(
    num_quadrotors: int,
    radius: float,
    cable_lengths,
    rotor_span: float = DEFAULT_ROTOR_SPAN,
    clearance: float = DEFAULT_FORMATION_CLEARANCE,
) -> None:
    """Refuse a ring the airframes cannot fly on.

    Args:
        num_quadrotors: number of quadrotors on the ring.
        radius: ring radius, in metres.
        cable_lengths: a scalar or one length per cable, in metres.
        rotor_span: tip-to-tip span of one airframe, in metres.
        clearance: margin between neighbouring rotor spans, in metres.

    Raises:
        ValueError: if neighbours are closer than ``rotor_span + clearance``, or
            the radius reaches any cable length. Both messages name the bound.
    """
    if num_quadrotors == 1:
        return
    spacing = neighbour_spacing(num_quadrotors, radius)
    need = float(rotor_span) + float(clearance)
    if spacing < need - 1e-12:
        r_min = min_formation_radius(num_quadrotors, rotor_span, clearance)
        raise ValueError(
            f"formation radius {radius:g} m puts neighbouring quadrotors {spacing:.3f} m apart "
            f"at num_quadrotors = {num_quadrotors}; they need the rotor span {rotor_span:g} m "
            f"plus a {clearance:g} m clearance ({need:g} m), i.e. a radius of at least "
            f"{r_min:.3f} m"
        )
    lengths = np.atleast_1d(np.asarray(cable_lengths, dtype=float))
    if np.any(lengths <= radius):
        raise ValueError(
            f"formation radius {radius:g} m must be strictly below every cable length "
            f"{lengths.tolist()}: a taut cable from the ring to the payload axis has no "
            "solution otherwise. Lengthen the cables, or fly fewer quadrotors (the radius "
            f"the clearance needs at num_quadrotors = {num_quadrotors} is "
            f"{min_formation_radius(num_quadrotors, rotor_span, clearance):.3f} m)"
        )


def default_spawn_height(
    cable_length: float,
    radius: float,
    payload_height: float = DEFAULT_PAYLOAD_SPAWN_HEIGHT,
) -> float:
    """Ring height that spawns the payload at ``payload_height`` on taut cables.

    ``payload_height + sqrt(l² − r²)``: the payload's height is fixed and the
    ring sits one taut-cable drop above it, so a longer cable raises the ring
    rather than lowering the payload into the ground.

    Raises:
        ValueError: if ``cable_length <= radius``.
    """
    if cable_length <= radius:
        raise ValueError(
            f"cable_length ({cable_length}) must be strictly greater than the formation "
            f"radius ({radius})"
        )
    return float(payload_height) + math.sqrt(float(cable_length) ** 2 - float(radius) ** 2)
