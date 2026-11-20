r"""Spawn geometry for a flexible hose carried by several quadrotors (M6).

M3 and M4 spawn their formations from a right triangle — a ring of radius
``r`` and a taut cable of length ``l`` fix the payload height at
``\sqrt{l^2 - r^2}``, and the case ``l \le r`` is refused rather than clamped.
M6's analogue is a *general* triangle, and this module is where it lives so
that the analytical model
(:mod:`udaan.models.multi_quad_hose`), the MJCF generator
(:func:`udaan.utils.assets.xml_model_generator.multi_quad_hose`) and the tests
all read the same definition instead of three.

The geometry
------------
The hose is a chain of ``n`` links with lengths ``l_i > 0`` joined at ``n + 1``
vertices; ``attachments`` lists the vertices carrying a quadrotor, strictly
increasing.  Between two consecutive attachments ``a < b`` the hose has the
links ``a … b-1``, of total length ``L = \sum_{i=a}^{b-1} l_i``.  Put the two
quadrotors on a horizontal line a distance ``d = \sigma L`` apart, with
``\sigma`` the ``sag_ratio``, and hang the segment between them as a **single
V**: the first ``p`` links descend at a common angle ``\alpha`` below the
horizontal, the remaining ``k - p`` ascend at ``\beta``.

That V is a triangle with sides ``L_A`` (the descending group), ``L_B`` (the
ascending group) and base ``d``, so both angles are closed-form law-of-cosines
expressions,

.. math::

    \cos\alpha = \frac{L_A^2 + d^2 - L_B^2}{2 L_A d}, \qquad
    \cos\beta  = \frac{L_B^2 + d^2 - L_A^2}{2 L_B d},

and the two constraints hold identically: the horizontal extents sum to
``(L_A^2 + d^2 - L_B^2 + L_B^2 + d^2 - L_A^2)/(2d) = d``, and the vertical
extents ``L_A \sin\alpha`` and ``L_B \sin\beta`` are both the triangle's
height, so the segment ends where it started.  **Every link is exactly taut**;
the sag is in the fold, not in any link's length.

Feasibility is the triangle inequality, and it is refused rather than clamped:

* ``d < L_A + L_B = L`` — automatic for ``\sigma < 1``.
* ``d > |L_A - L_B|`` — the V cannot close past the point where the two groups
  fold flat onto each other.  The split ``p`` is chosen to minimise
  ``|L_A - L_B|``, so for uniform links this is ``\sigma > 1/k`` at odd ``k``
  and vacuous at even ``k``; it can still bite for very unequal link lengths.

Two degenerate cases are handled rather than refused:

* **A one-link segment** cannot fold at all: a rigid link of length ``l``
  between two attachments *is* the separation, so it is laid horizontally and
  the ``sag_ratio`` is ignored for that segment.
* **Free ends.**  Links above the first attachment and below the last one
  carry no second quadrotor, so they simply hang from the nearest one:
  ``q_i = +e_3`` above (each vertex hanging below the next) and ``q_i = -e_3``
  below.  This is their static equilibrium, and it is the only part of the
  spawn that is one.

The spawn as a whole is **not** an equilibrium — a hose held level by two
drones sags the moment gravity is switched on.  It is the taut, collision-free
configuration both backends start from; the equilibrium is
:meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.hover_trim`, which hangs
the whole hose vertically and therefore stacks the airframes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.defaults import DEFAULT_HOSE_SAG_RATIO, GRAVITY

__all__ = [
    "TetheredTrim",
    "hose_spawn_directions",
    "hose_vertex_positions",
    "tethered_trim",
    "tethered_vehicle_positions",
    "validate_attachments",
]

_E3 = np.array([0.0, 0.0, 1.0])

#: Below this, a length or a span is treated as zero rather than divided by.
_MIN_SPAN = 1e-12


def validate_attachments(attachments, num_links: int) -> tuple[int, ...]:
    """Normalise and check the vertex indices carrying a quadrotor.

    Args:
        attachments: iterable of vertex indices in ``[0, num_links]``.
        num_links: ``n``; there are ``n + 1`` vertices.

    Returns:
        The indices as a tuple of ``int``.

    Raises:
        ValueError: if the sequence is empty, contains a duplicate, is not
            strictly increasing, or leaves ``[0, num_links]``.

    Strict increase is required rather than sorted-on-input for a reason the
    MATLAB reference implementation illustrates: it accepts an arbitrary ``setI`` but
    assigns thrusts to vertices by walking the chain *in vertex order*, so an
    unsorted ``setI`` silently pairs quadrotor ``k``'s thrust and inertia with
    a different vertex than the one its own ``ind`` records.  Refusing the
    input rules that mis-pairing out.
    """
    indices = [int(v) for v in attachments]
    if not indices:
        raise ValueError("attachments must name at least one vertex")
    if any(b <= a for a, b in zip(indices, indices[1:], strict=False)):
        raise ValueError(
            "attachments must be strictly increasing (each vertex carries at most one "
            f"quadrotor, and thrusts are matched to vertices in chain order); got {indices}"
        )
    if indices[0] < 0 or indices[-1] > num_links:
        raise ValueError(
            f"attachments must lie in [0, {num_links}] for a {num_links}-link hose; got {indices}"
        )
    return tuple(indices)


def _balanced_split(lengths: np.ndarray) -> int:
    """Index ``p`` splitting ``lengths`` into two groups of most equal total.

    ``1 <= p <= len(lengths) - 1``, so both groups are non-empty.
    """
    total = float(lengths.sum())
    cumulative = np.cumsum(lengths)[:-1]
    return int(np.argmin(np.abs(2.0 * cumulative - total))) + 1


def _segment_directions(lengths: np.ndarray, span: float) -> np.ndarray:
    """Unit directions folding ``lengths`` into a V of horizontal extent ``span``.

    The V lies in the ``xz`` plane and runs in ``+x``; both endpoints are at the
    same height.  Returns ``(k, 3)``.
    """
    k = lengths.shape[0]
    if k == 1:
        # A rigid link is its own separation; there is nothing to fold.
        return np.array([[1.0, 0.0, 0.0]])

    p = _balanced_split(lengths)
    length_a = float(lengths[:p].sum())
    length_b = float(lengths[p:].sum())
    total = length_a + length_b
    if span > total:
        raise ValueError(
            f"a hose segment of length {total:.6g} m cannot span {span:.6g} m: the links "
            "would have to stretch. Reduce sag_ratio, or lengthen the segment"
        )
    if span <= abs(length_a - length_b) + _MIN_SPAN:
        raise ValueError(
            f"a hose segment of length {total:.6g} m cannot fold down to a {span:.6g} m span: "
            f"its two halves are {length_a:.6g} m and {length_b:.6g} m, and folded flat they "
            f"already reach {abs(length_a - length_b):.6g} m. Raise sag_ratio, or make the "
            "link lengths more even"
        )

    cos_a = float(np.clip((length_a**2 + span**2 - length_b**2) / (2.0 * length_a * span), -1, 1))
    cos_b = float(np.clip((length_b**2 + span**2 - length_a**2) / (2.0 * length_b * span), -1, 1))
    # The triangle inequality above keeps both inside [-1, 1]; the clip is
    # against round-off at ``sag_ratio = 1``, where the triangle degenerates to
    # a straight segment and both cosines land exactly on 1.
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a**2)))
    sin_b = float(np.sqrt(max(0.0, 1.0 - cos_b**2)))

    directions = np.zeros((k, 3))
    directions[:p] = np.array([cos_a, 0.0, -sin_a])
    directions[p:] = np.array([cos_b, 0.0, sin_b])
    return directions


def hose_spawn_directions(
    link_lengths,
    attachments,
    sag_ratio: float = DEFAULT_HOSE_SAG_RATIO,
) -> np.ndarray:
    """Link directions ``q_i`` of the taut spawn configuration, shape ``(n, 3)``.

    Args:
        link_lengths: ``(n,)`` strictly positive link lengths.
        attachments: strictly increasing vertex indices carrying a quadrotor.
        sag_ratio: horizontal span of each interior segment as a fraction of
            that segment's own length, in ``(0, 1)``.

    Returns:
        ``(n, 3)`` unit vectors; ``q_i`` points from vertex ``i`` to vertex
        ``i + 1``.

    Raises:
        ValueError: for a non-positive length, an out-of-range ``sag_ratio``,
            an invalid attachment list, or a segment whose triangle has no
            solution (see the module docstring).
    """
    lengths = np.asarray(link_lengths, dtype=float).reshape(-1)
    if lengths.size < 1:
        raise ValueError("a hose needs at least one link")
    if not np.all(lengths > 0.0):
        raise ValueError(f"link_lengths must be strictly positive; got {lengths.tolist()}")
    if not 0.0 < float(sag_ratio) <= 1.0:
        raise ValueError(f"sag_ratio must lie in (0, 1]; got {sag_ratio}")

    n = lengths.shape[0]
    indices = validate_attachments(attachments, n)

    directions = np.zeros((n, 3))
    # Free end above the first attachment: each vertex hangs below the next, so
    # the direction from vertex i to vertex i+1 points *up*.
    directions[: indices[0]] = _E3
    # Free end below the last attachment: straight down.
    directions[indices[-1] :] = -_E3

    for first, second in zip(indices, indices[1:], strict=False):
        segment = lengths[first:second]
        span = float(sag_ratio) * float(segment.sum())
        directions[first:second] = _segment_directions(segment, span)

    return directions


def hose_vertex_positions(root_position, link_lengths, directions) -> np.ndarray:
    """Vertex positions of a hose, shape ``(n + 1, 3)``.

    ``x_0`` is ``root_position`` and ``x_{i+1} = x_i + l_i q_i`` — the chain
    recursion the whole model is built on, in one place so the generator, the
    model and the tests cannot disagree about it.
    """
    lengths = np.asarray(link_lengths, dtype=float).reshape(-1)
    q = np.asarray(directions, dtype=float).reshape(lengths.shape[0], 3)
    out = np.zeros((lengths.shape[0] + 1, 3))
    out[0] = np.asarray(root_position, dtype=float)
    for i in range(lengths.shape[0]):
        out[i + 1] = out[i] + lengths[i] * q[i]
    return out


# ─── A hose pinned to a ground anchor (M7 / M8) ───────────────────────

#: Residual, in metres, below which a pinned-pinned segment solve is accepted.
_SEGMENT_TOL = 1e-9

#: Starting tension scales, as multiples of a segment's own weight, tried in
#: turn by :func:`tethered_trim`'s root solve.
_SEGMENT_TENSION_STARTS = (1.0, 0.1, 10.0, 100.0, 0.01)


@dataclass
class TetheredTrim:
    """A steady state of a hose pinned at vertex 0, from :func:`tethered_trim`.

    ``tensions[i]`` is ``𝛕_i``, the force link ``i`` exerts on its lower vertex
    ``i``; it points along ``q_i``, up the hose.  ``flat_tensions`` is the
    tethered flat output's force part: ``𝛕_0`` (the anchor link) followed by
    ``𝛕_a`` at every vehicle vertex but the top one.
    """

    #: ``(n, 3)`` link directions ``q_i``, vertex ``i`` to vertex ``i + 1``.
    directions: np.ndarray
    #: ``(n, 3)`` link force vectors ``𝛕_i``.
    tensions: np.ndarray
    #: ``(n + 1, 3)`` vertex positions; row 0 is the anchor.
    vertex_positions: np.ndarray
    #: ``(nQ, 3)`` thrust vectors ``f_k R_k e_3``.
    thrust_vectors: np.ndarray
    #: ``(nQ, 3)`` ``[𝛕_0, 𝛕_{a_1}, …, 𝛕_{a_{nQ-1}}]``.
    flat_tensions: np.ndarray
    #: ``(3,)`` force the anchor exerts on the hose.
    anchor_force: np.ndarray

    @property
    def ground_clearance(self) -> float:
        """Height of the lowest vertex but the anchor, in metres; the ground is ``z = 0``.

        Negative where the hose would pass below the ground plane.
        """
        return float(np.min(self.vertex_positions[1:, 2]))


def tethered_vehicle_positions(top_position, num_quadrotors: int, anchor) -> np.ndarray:
    """Vehicle positions evenly spaced on the line from ``anchor`` to ``top_position``.

    Vehicle ``k`` (``k = 0 … nQ − 1``) sits at ``anchor + (k + 1)/nQ ·
    (top − anchor)``, so the last one is at ``top_position``.  Returns
    ``(nQ, 3)``.
    """
    top = np.asarray(top_position, dtype=float).reshape(3)
    base = np.asarray(anchor, dtype=float).reshape(3)
    fractions = (np.arange(int(num_quadrotors)) + 1.0) / float(num_quadrotors)
    return np.asarray(base[None, :] + fractions[:, None] * (top - base)[None, :])


def _sweep_segment(tau_start, start, lengths, interior_masses, accel, gravity):
    """Walk one segment up from its lower held vertex.

    Returns the link forces, the vertex positions (both ends included) and the
    force of the segment's last link, or ``None`` if a link goes slack.
    """
    e3 = np.array([0.0, 0.0, gravity])
    tau = np.asarray(tau_start, dtype=float)
    x = np.asarray(start, dtype=float)
    forces = []
    points = [x]
    for i, length in enumerate(lengths):
        norm = float(np.linalg.norm(tau))
        if norm <= 0.0:
            return None
        forces.append(tau)
        x = x + length * tau / norm
        points.append(x)
        if i < len(lengths) - 1:
            tau = tau + interior_masses[i] * (accel(x) + e3)
    return np.array(forces), np.array(points)


def _solve_segment(start, end, lengths, interior_masses, accel, gravity):
    """``𝛕`` on a segment's first link that carries it from ``start`` to ``end``."""
    from scipy.optimize import fsolve

    chord = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    total = float(np.sum(lengths))
    distance = float(np.linalg.norm(chord))
    if distance >= total:
        raise ValueError(
            f"two held points {distance:.6g} m apart cannot be joined by {total:.6g} m of hose"
        )
    if distance <= 0.0:
        raise ValueError("two held points coincide; the hose between them has no taut state")

    def residual(tau):
        swept = _sweep_segment(tau, start, lengths, interior_masses, accel, gravity)
        if swept is None:
            return np.full(3, total)
        return swept[1][-1] - np.asarray(end, dtype=float)

    weight = gravity * float(np.sum(interior_masses)) + gravity * 1e-3
    direction = chord / distance
    for scale in _SEGMENT_TENSION_STARTS:
        guess = scale * weight * direction - 0.5 * gravity * float(
            np.sum(interior_masses)
        ) * np.array([0.0, 0.0, 1.0])
        tau, _, ier, _ = fsolve(residual, guess, full_output=True, xtol=1e-13)
        if ier == 1 and float(np.max(np.abs(residual(tau)))) < _SEGMENT_TOL * max(1.0, total):
            return np.asarray(tau, dtype=float)
    raise ValueError(
        f"no taut state joins {np.asarray(start).tolist()} to {np.asarray(end).tolist()} "
        f"with {len(lengths)} link(s) of total length {total:.6g} m"
    )


def tethered_trim(
    link_lengths,
    vertex_total_masses,
    attachments,
    positions,
    *,
    anchor,
    gravity: float = GRAVITY,
    whirl_rate: float = 0.0,
) -> TetheredTrim:
    r"""The steady state of a hose pinned at vertex 0 with its vehicles held at ``positions``.

    Each segment between two held points (the anchor and vehicle 1, vehicle 1
    and vehicle 2, …) is a pinned-pinned discrete catenary: the force on its
    first link is the root of a three-equation solve, and the sweep up the
    segment is M6's vertex balance

    .. math::

        𝛕_j = 𝛕_{j-1} + \bar m_j (a_j + g e_3), \qquad
        q_j = 𝛕_j / \|𝛕_j\| ,

    at every bare vertex.  At a vehicle vertex ``v`` the thrust is
    ``u_v = \bar m_v(a_v + g e_3) + 𝛕_{v-1} - 𝛕_v`` with ``𝛕_n = 0``, and the
    anchor's force on the hose is ``\bar m_0 g e_3 - 𝛕_0``.

    With ``whirl_rate`` ``Ω ≠ 0`` the state is a rigid rotation about the
    vertical axis through the anchor: every vertex carries the centripetal
    acceleration ``a_j = -Ω^2 (x_j - x_{anchor})_{xy}``, and the result is
    the configuration at the instant it passes through ``positions``.

    Args:
        link_lengths: ``(n,)``.
        vertex_total_masses: ``(n + 1,)`` hose plus airframe mass at each
            vertex; entry 0 is carried by the anchor.
        attachments: strictly increasing vehicle vertices, all ``>= 1``, the
            last ``== n``.
        positions: ``(nQ, 3)`` where each vehicle is held.
        anchor: ``(3,)`` position of vertex 0.
        gravity: ``g``.
        whirl_rate: ``Ω``, in rad/s.

    Raises:
        ValueError: for an attachment pattern the pinned hose does not allow,
            two held points the hose between them cannot reach, or a segment
            with no taut solution.
    """
    lengths = np.asarray(link_lengths, dtype=float).reshape(-1)
    n = lengths.shape[0]
    mbar = np.asarray(vertex_total_masses, dtype=float).reshape(n + 1)
    indices = validate_attachments(attachments, n)
    if indices[0] < 1 or indices[-1] != n:
        raise ValueError(
            f"a hose pinned at vertex 0 needs its vehicles in [1, {n}] with one at vertex {n}; "
            f"got {indices}"
        )
    held = np.asarray(positions, dtype=float).reshape(len(indices), 3)
    base = np.asarray(anchor, dtype=float).reshape(3)
    rate_sq = float(whirl_rate) ** 2

    def accel(x):
        offset = np.asarray(x, dtype=float) - base
        return -rate_sq * np.array([offset[0], offset[1], 0.0])

    tensions = np.zeros((n, 3))
    points = np.zeros((n + 1, 3))
    points[0] = base
    flat = np.zeros((len(indices), 3))
    starts = [0, *indices[:-1]]
    ends = list(indices)
    anchors = [base, *held[:-1]]
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        tau = _solve_segment(anchors[k], held[k], lengths[a:b], mbar[a + 1 : b], accel, gravity)
        swept = _sweep_segment(tau, anchors[k], lengths[a:b], mbar[a + 1 : b], accel, gravity)
        assert swept is not None  # the solve rejects a slack sweep
        tensions[a:b] = swept[0]
        points[a + 1 : b + 1] = swept[1][1:]
        flat[k] = tau

    e3 = np.array([0.0, 0.0, gravity])
    thrusts = np.zeros((len(indices), 3))
    for k, v in enumerate(indices):
        above = tensions[v] if v < n else np.zeros(3)
        thrusts[k] = mbar[v] * (accel(points[v]) + e3) + tensions[v - 1] - above
    directions = tensions / np.linalg.norm(tensions, axis=1)[:, None]
    return TetheredTrim(
        directions=directions,
        tensions=tensions,
        vertex_positions=points,
        thrust_vectors=thrusts,
        flat_tensions=flat,
        anchor_force=mbar[0] * e3 - tensions[0],
    )
