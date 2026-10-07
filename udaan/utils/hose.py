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

import numpy as np

from ..core.defaults import DEFAULT_HOSE_SAG_RATIO

__all__ = [
    "hose_spawn_directions",
    "hose_vertex_positions",
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
    legacy MATLAB reference demonstrates: it accepts an arbitrary ``setI`` but
    assigns thrusts to vertices by walking the chain *in vertex order*, so an
    unsorted ``setI`` silently pairs quadrotor ``k``'s thrust and inertia with
    a different vertex than the one its own ``ind`` records.  Refusing the
    input is the cheapest way to make that class of bug impossible.
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
