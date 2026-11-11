r"""Ring-formation and grasp geometry for quadrotors that carry one payload on cables.

``q_i \in S^2`` points **from quadrotor** ``i`` **to the payload**, as in the
rest of the package.

For a cable-suspended **rigid-body** payload (M4), ``r_i`` is the attachment
point in the **payload body frame**. A point-mass payload only cares about the
*sum* of the cable forces; a rigid body also cares about where each one is
applied. :func:`ring_attachment_points`, :func:`grasp_map` and
:func:`hover_wrench_map` are what that difference amounts to, and they are
shared by the model (:mod:`udaan.models.multi_quad_rigidbody`), the controller
(:mod:`udaan.control.multi_quad_rigidbody`) and the tests, so that all three
agree on one definition of the geometry.
"""

from __future__ import annotations

import numpy as np

from ..core.defaults import (
    DEFAULT_ATTACHMENT_HEIGHT,
    DEFAULT_ATTACHMENT_RADIUS,
    DEFAULT_FORMATION_CLEARANCE,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_PAYLOAD_SPAWN_HEIGHT,
    DEFAULT_ROTOR_SPAN,
)
from ..manif import S2
from ..manif.utils import hat
from .formation import default_spawn_height, min_formation_radius

__all__ = [
    "formation_cable_attitudes",
    "grasp_map",
    "check_rigid_formation",
    "hover_wrench_map",
    "rigid_formation_radius",
    "rigid_spawn_height",
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
    the missing SE(3) directions are produced by reorienting the cables.  A
    payload that rotates or translates therefore needs its cables to move with
    it — at a constant yaw rate every cable sweeps a cone — and the cascade
    commands that motion through the reference cable rates of the flat map
    (:class:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController`,
    "Flatness feedforward").  The steady-state tracking of a smooth reference is
    set by that feed-forward and the controller's gains, not by this rank.

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


def rigid_formation_radius(
    num_quadrotors: int,
    attachment_radius: float = DEFAULT_ATTACHMENT_RADIUS,
    rotor_span: float = DEFAULT_ROTOR_SPAN,
    clearance: float = DEFAULT_FORMATION_CLEARANCE,
) -> float:
    """Default formation radius of a rigid-payload ring.

    Each airframe sits ``formation_radius`` radially outboard of its own
    attachment point, so the airframes ring the payload at
    ``attachment_radius + formation_radius``. The default is
    :data:`~udaan.core.defaults.DEFAULT_FORMATION_RADIUS`, grown once that ring
    would put neighbours closer than ``rotor_span + clearance``
    (:func:`udaan.utils.formation.min_formation_radius`)::

        max(DEFAULT_FORMATION_RADIUS, min_formation_radius(n_Q) − attachment_radius)
    """
    ring = min_formation_radius(num_quadrotors, rotor_span, clearance)
    return max(float(DEFAULT_FORMATION_RADIUS), ring - float(attachment_radius))


def check_rigid_formation(
    attachment_points,
    formation_radius: float,
    cable_lengths,
    rotor_span: float = DEFAULT_ROTOR_SPAN,
    clearance: float = DEFAULT_FORMATION_CLEARANCE,
) -> None:
    """Refuse a rigid-payload ring the airframes cannot fly on.

    Airframe ``i`` sits ``formation_radius`` outboard of attachment point
    ``i`` on the azimuth ``2πi/n_Q`` (:func:`formation_cable_attitudes`). On
    the attachment ring of radius ``a`` the airframes therefore ring the
    payload at ``a + formation_radius``, and the rule of
    :func:`udaan.utils.formation.check_formation` applies to that ring:
    neighbours at least ``rotor_span + clearance`` apart (for attachment points
    off a ring, the smallest horizontal distance between any two airframes).
    The cable condition is on the formation radius itself, which must stay
    strictly below every cable length.

    Raises:
        ValueError: naming the spacing and the radius it needs, or the cable
            length the formation radius reaches.
    """
    points = np.asarray(attachment_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"attachment_points must be (nQ, 3); got {points.shape}")
    nQ = int(points.shape[0])
    if nQ == 1:
        return
    radius = float(formation_radius)
    radii = np.linalg.norm(points[:, :2], axis=1)
    angles = np.linspace(0.0, 2 * np.pi, nQ, endpoint=False)
    airframes = points[:, :2] + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    gaps = np.linalg.norm(airframes[:, None, :] - airframes[None, :, :], axis=2)
    spacing = float(np.min(gaps[~np.eye(nQ, dtype=bool)]))
    need = float(rotor_span) + float(clearance)
    if spacing < need - 1e-12:
        ring = ""
        if np.allclose(radii, radii[0], rtol=0.0, atol=1e-12):
            r_min = min_formation_radius(nQ, rotor_span, clearance) - float(radii[0])
            ring = f", i.e. a formation radius of at least {r_min:.3f} m"
        raise ValueError(
            f"formation radius {radius:g} m puts neighbouring quadrotors {spacing:.3f} m "
            f"apart at num_quadrotors = {nQ}; they need the rotor span {rotor_span:g} m "
            f"plus a {clearance:g} m clearance ({need:g} m){ring}"
        )
    lengths = np.atleast_1d(np.asarray(cable_lengths, dtype=float))
    if np.any(lengths <= radius):
        r_clear = max(0.0, min_formation_radius(nQ, rotor_span, clearance) - float(np.max(radii)))
        raise ValueError(
            f"formation radius {radius:g} m must be strictly below every cable length "
            f"{lengths.tolist()}: a taut cable reaching {radius:g} m outboard of its "
            "attachment point has no solution otherwise. Lengthen the cables, or pass a "
            "smaller formation_radius (the clearance between airframes needs at least "
            f"{r_clear:.3f} m at num_quadrotors = {nQ})"
        )


def rigid_spawn_height(
    cable_length: float,
    formation_radius: float,
    attachment_height: float = DEFAULT_ATTACHMENT_HEIGHT,
    payload_height: float = DEFAULT_PAYLOAD_SPAWN_HEIGHT,
) -> float:
    """Airframe ring height that spawns the payload's centre at ``payload_height``.

    The cables attach ``attachment_height`` above the payload's centre, and
    each airframe sits one taut-cable drop ``sqrt(l² − r²)`` above its
    attachment point (:func:`udaan.utils.formation.default_spawn_height`), so a
    longer cable raises the ring rather than lowering the payload into the
    ground.
    """
    return default_spawn_height(
        cable_length, formation_radius, float(payload_height) + float(attachment_height)
    )
