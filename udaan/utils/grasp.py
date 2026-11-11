r"""Ring-formation geometry for quadrotors that carry one payload on cables.

``q_i \in S^2`` points **from quadrotor** ``i`` **to the payload**, as in the
rest of the package.
"""

from __future__ import annotations

import numpy as np

from ..core.defaults import DEFAULT_FORMATION_RADIUS
from ..manif import S2

__all__ = ["formation_cable_attitudes"]


def formation_cable_attitudes(
    num_quadrotors: int,
    cable_lengths: np.ndarray,
    radius: float = DEFAULT_FORMATION_RADIUS,
) -> list[S2]:
    """Cable directions for a ring formation of ``num_quadrotors`` quadrotors.

    The quadrotors sit on a horizontal circle of ``radius`` centred above the
    payload; each cable is taut, so quadrotor ``i`` is lifted by
    ``sqrt(l_i² - radius²)``.  Returns the unit vectors pointing *from* each
    quadrotor *to* the payload.

    Pure geometry, used by the model, its MJCF generator and the flat map's
    hover formation; it sits in :mod:`udaan.utils` so that
    :mod:`udaan.flatness` and :mod:`udaan.control` can use it below
    :mod:`udaan.models` in the package layering.
    :mod:`udaan.models.multi_quad_cspayload` re-exports it unchanged.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")
    if num_quadrotors == 1:
        # A lone quadrotor hovers directly above the payload — no ring, so the
        # radius (and its taut-cable constraint) does not apply.
        return [S2(np.array([0.0, 0.0, -1.0]))]

    lengths = np.asarray(cable_lengths, dtype=float)
    if np.any(lengths <= radius):
        raise ValueError(
            f"formation radius {radius} must be strictly below every cable length {lengths}"
        )

    angles = np.linspace(0.0, 2 * np.pi, num_quadrotors, endpoint=False)
    attitudes: list[S2] = []
    for i, theta in enumerate(angles):
        dz = np.sqrt(lengths[i] ** 2 - radius**2)
        offset = np.array([radius * np.cos(theta), radius * np.sin(theta), dz])
        attitudes.append(S2(-offset / lengths[i]))
    return attitudes
