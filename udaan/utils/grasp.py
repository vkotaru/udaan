r"""Grasp geometry for a cable-suspended **rigid-body** payload (M4).

A point-mass payload only cares about the *sum* of the cable forces; a rigid
body also cares about where each one is applied.  The three helpers here are
what that difference amounts to, and they are shared by the model
(:mod:`udaan.models.multi_quad_rigidbody`), the controller
(:mod:`udaan.control.multi_quad_rigidbody`) and the tests, so that all three
agree on one definition of the geometry rather than three.

Conventions follow the rest of the package: ``r_i`` is the attachment point in
the **payload body frame**, and ``q_i \in S^2`` points **from quadrotor** ``i``
**to the load**.
"""

from __future__ import annotations

import numpy as np

from ..core.defaults import (
    DEFAULT_ATTACHMENT_HEIGHT,
    DEFAULT_ATTACHMENT_RADIUS,
    DEFAULT_FORMATION_RADIUS,
)
from ..manif import S2
from ..manif.utils import hat

__all__ = [
    "formation_cable_attitudes",
    "grasp_map",
    "hover_wrench_map",
    "ring_attachment_points",
]


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

    Lives here rather than in the M3 model it was written for because the M4
    controller needs the same ring to bias the null space of :func:`grasp_map`,
    and :mod:`udaan.control` sits *below* :mod:`udaan.models` in the package
    layering.  It is pure geometry, so it belongs on the utility side of that
    line; :mod:`udaan.models.multi_quad_cspayload` re-exports it unchanged.
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


def ring_attachment_points(
    num_quadrotors: int,
    radius: float = DEFAULT_ATTACHMENT_RADIUS,
    height: float = DEFAULT_ATTACHMENT_HEIGHT,
) -> np.ndarray:
    """Body-frame cable attachment points on a horizontal ring.

    Attachment ``i`` sits at azimuth ``2πi/nQ`` on a circle of ``radius``,
    lifted ``height`` above the payload centre of mass.  The azimuths are the
    ones :func:`formation_cable_attitudes`
    uses, so pairing the two puts quadrotor ``i`` radially outboard of its own
    attachment point — which is what makes the resulting hover trim exert zero
    net moment on the load.

    A single quadrotor gets ``(0, 0, height)``: the attachment has to lie on
    the payload's vertical axis, or one vertical cable could not balance the
    moment on its own.

    Returns:
        ``(nQ, 3)`` array of body-frame attachment points.
    """
    if num_quadrotors < 1:
        raise ValueError(f"num_quadrotors must be >= 1, got {num_quadrotors}")
    if num_quadrotors == 1:
        return np.array([[0.0, 0.0, float(height)]])

    angles = np.linspace(0.0, 2 * np.pi, num_quadrotors, endpoint=False)
    return np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(num_quadrotors, float(height))]
    )


def grasp_map(attachment_points) -> np.ndarray:
    r"""The ``(6, 3 nQ)`` grasp matrix ``Φ`` of a set of attachment points.

    ``Φ`` maps a stack of **body-frame** cable forces applied *to the payload*
    to the resultant body-frame wrench on it:

    .. math::

        \Phi \begin{bmatrix} \tau_1 \\ \vdots \\ \tau_{n_Q} \end{bmatrix}
        = \begin{bmatrix} \sum_i \tau_i \\ \sum_i r_i \times \tau_i \end{bmatrix} .

    Its rank is the number of wrench directions the grasp can produce **from
    arbitrary force vectors** ``τ_i``, so it is the object that says whether a
    geometry can hold an attitude at all: a rank below 6 leaves some moment (or
    force) direction unactuated, and rank 6 is necessary for full SE(3)
    authority.

    It is not sufficient, and the gap matters.  A taut cable can only pull
    along its own direction, so what the team can actually apply *at a given
    instant* is not ``Φ τ`` over all ``τ`` but ``G t`` over scalar tensions
    ``t_i ≥ 0`` along the current ``q_i`` — the ``(6, nQ)`` map
    :func:`hover_wrench_map`.  At the default ring that map has **rank 3 for
    every** ``nQ ≥ 3``: the instantaneous wrench authority is 3-dimensional, and
    the missing SE(3) directions only become available as the cascade reorients
    the cables, one attitude-loop time constant later.  That lag is visible in
    closed loop — the default ring settles with an ~11° steady-state yaw-rate
    lag — and it is a property of the mechanism, not of the allocation.

    Both statements are *static* ones about allocation; neither bears on
    whether the dynamics are well posed, which they always are (see
    :meth:`udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.mass_matrix`).
    """
    points = np.asarray(attachment_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"attachment_points must be (nQ, 3); got {points.shape}")
    return np.hstack([np.vstack([np.eye(3), hat(r)]) for r in points])


def hover_wrench_map(cable_attitudes, attachment_points, payload_orientation=None) -> np.ndarray:
    r"""The ``(6, nQ)`` map from scalar cable tensions to the payload wrench.

    Column ``i`` is ``[q_i ; r_i × (R_L^T q_i)]``, so with ``t`` the vector of
    tension magnitudes of ``𝐓_i = t_i q_i`` (the force on quadrotor ``i``) the
    static hover conditions read

    .. math::

        G\, t = \begin{bmatrix} -m_L g e_3 \\ 0 \end{bmatrix} .

    Solving that in least squares and checking the residual is how a trim is
    built *from first principles* rather than read off a model.  A non-zero
    residual means the grasp geometry cannot hold a static hover at all — the
    extra constraint a rigid-body payload imposes and a point mass does not.
    """
    q = np.asarray(cable_attitudes, dtype=float)
    r = np.asarray(attachment_points, dtype=float)
    R_L = np.eye(3) if payload_orientation is None else np.asarray(payload_orientation, dtype=float)
    if q.shape != r.shape:
        raise ValueError(f"cable_attitudes {q.shape} and attachment_points {r.shape} disagree")
    return np.column_stack(
        [np.concatenate([q[i], np.cross(r[i], R_L.T @ q[i])]) for i in range(q.shape[0])]
    )
