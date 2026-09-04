r"""Cable-chain cascade — tension and position recursion along a serial chain.

The last of the cross-cutting flatness primitives.  A *flexible* cable is
modelled as a serial chain of ``n`` rigid links, each carrying a point mass
lumped at its own **distal** end (the same lumping
:mod:`udaan.models.quadrotor_flexcable` integrates).  Flatness of such a system
rests on one observation: the chain is a *statically determinate* structure
hanging off its tip.  Knowing the tip's acceleration fixes the force in the
last link; that force fixes the last link's direction, hence the position of
the node above it; that node's acceleration fixes the next force up; and so on.
Nothing is integrated and nothing is solved — the whole chain is recovered by a
single sweep from the tip towards the top.

:func:`cable_chain_cascade` is that sweep.  Every per-link step is the *existing*
primitive :func:`~udaan.flatness.cable.tension_direction_jet`; the only new
content here is the recursion that threads them together, together with the
bookkeeping of how much derivative depth survives each rung.

Geometry and indexing
---------------------
Links are indexed ``i = 0 … n-1`` **from the top down**, matching
:attr:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableState.link_attitudes`:
link ``0`` hangs off whatever holds the chain up (a quadrotor, in M5 and M6) and
link ``n-1`` carries the tip.  Nodes are indexed ``0 … n``, node ``0`` being the
top attachment and node ``i+1`` the distal end of link ``i``:

.. math::

    x_{i+1} = x_i + \ell_i q_i, \qquad
    x_n = x_{\text{tip}}, \qquad
    x_0 = x_{\text{top}} .

Mass ``m_i`` sits at node ``i+1``, so ``masses[-1]`` is the payload (plus
whatever share of the cable's own mass the last link carries).

Sign convention
---------------
``q_i`` points **from the top of the chain towards the tip** — downwards at
rest, ``q_i = -e_3`` — which is this codebase's convention throughout
(:mod:`udaan.flatness.cable`, :mod:`udaan.models.quadrotor_cspayload`,
:mod:`udaan.models.multi_quad_cspayload`, and M5's own state).

The recursion is carried on the **cable-force jet**

.. math::
    :label: chain-force

    F_i \;=\; -\sum_{k \ge i} m_k \bigl(\ddot x_{k+1} + g e_3\bigr)
              \;+\; \sum_{k \ge i} u_k ,

the force the sub-chain ``{i … n-1}`` exerts on the body immediately above link
``i`` (``u_k`` is any external force applied at node ``k+1`` — zero for a plain
cable; see `External forces`_).  With that sign,

.. math::

    q_i = \frac{F_i}{\|F_i\|}, \qquad T_i = \|F_i\| \ \ge 0 ,

i.e. exactly the same normalisation
:func:`~udaan.flatness.cable.point_mass_cable_force` /
:func:`~udaan.flatness.cable.tension_direction_jet` already performs, and at
``n = 1`` :eq:`chain-force` *is* ``F = -m_L(\ddot x_L + g e_3)`` term for term.

The legacy MATLAB reference
(``phd_research/quadrotor-flexible-cables-payload/Single-Quad-Flex-Cable/
get_desired_traj.m``, L38-121) uses the **same** direction — it accumulates
``B_i = +\sum_{k \ge i} m_k(\ddot x_{k+1} + g e_3)`` and then writes
``q_i = -B_i / \|B_i\|``, so at rest ``B_i = +m g e_3`` and ``q_i = -e_3``,
downwards, exactly as here.  What differs is only *where* the negation sits:
``B_i = -F_i``, so the reference carries the minus sign in every direction
formula while this module folds it into the force once and normalises ``F_i``
unchanged.  (Its link *indexing* does run the other way — index ``1`` is the
link at the quadrotor and index ``n`` the one at the payload — which is the
reverse of the top-down order used here.)

The hanging branch
------------------
``T_i = \|F_i\|`` is a norm, so it is positive again the instant a trajectory
*overshoots* the point where the link unloads: the slack check alone refuses to
land on ``F_i = 0`` but never to cross it, and past the crossing ``q_i`` has
flipped and the node below link ``i`` sits *above* the node above it.  On a
sampled trajectory the exact crossing is measure-zero, so without a second test
a dive through ``\ddot x_L \cdot e_3 = -g`` would silently return the flipped
reference at most of its sample times.

The cascade therefore also requires each link to stay on the **hanging branch**
— ``F_i \cdot (-e_3) \ge 0``, i.e. ``q_i`` no more than 90° from straight down.
This is the chain's version of M3's ``\mathbf{T}_i \cdot \bar q_i > 0`` sign
test (:mod:`udaan.flatness.multi_quad_cspayload`) with the hover direction
``-e_3``, and like it, it is deliberately conservative: it also rejects a link
that has swung past horizontal without ever unloading.  A rigid link is
perfectly well posed there — see
:meth:`~udaan.flatness.QuadrotorFlexCable.recover` — but a *cable* is not, and
these are cable maps.

Derivative depth
----------------
This is the primitive's one genuinely new accounting problem, and it is the
reason M5's jet order grows with the link count where M2-M4's does not.

Each rung costs **two** derivatives.  Going up from node ``i+1`` to link ``i``:
the force :eq:`chain-force` needs ``\ddot x_{i+1}``, so ``F_i`` — and hence
``q_i`` — is two orders shallower than the node position below it, while the
node position above, ``x_i = x_{i+1} - \ell_i q_i``, inherits ``q_i``'s order.
With the tip jet supplied at order ``K``:

======================  ================
Quantity                Order available
======================  ================
``x_n`` (tip)           ``K``
``F_{n-1}, q_{n-1}``    ``K - 2``
``x_{n-1}``             ``K - 2``
``F_{n-2}, q_{n-2}``    ``K - 4``
...                     ...
``F_0, q_0, x_0``       ``K - 2n``
======================  ================

A consumer that needs the top node through its **snap** (every thrust-vector
map does: ``\ddot x_0`` for the thrust, two more derivatives for the moment)
therefore needs ``K - 2n = 4``, i.e.

.. math::
    :label: chain-order

    K = 2n + 4 .

:func:`chain_jet_order` returns it.  At ``n = 1`` that is 6, exactly M2's
requirement; at ``n = 5`` it is 14.  The cascade truncates as it climbs rather
than demanding uniform depth, so a caller that hands it a deeper jet gets the
same answer — the extra rows are simply never read.

External forces
---------------
``node_forces`` injects an external force jet at a node.  A plain cable has
none, and M5 passes nothing.  M6 — the same chain with quadrotors attached at
chosen vertices — passes each attached vehicle's thrust-vector jet
``f_j R_j e_3`` at that vertex (and folds the vehicle's mass into ``masses``),
which is the whole of what the chain part of that map needs.  Those thrust jets
are *flat* quantities there, exactly as the internal cable forces are in M3 and
the null-space coordinate is in M4, so they arrive as data rather than as
unknowns and the cascade stays a single sweep.

Two contract facts M6 has to live with:

* **Nothing can be applied at node 0.**  ``node_forces`` has one entry per
  *link*, and entry ``i`` lands at link ``i``'s distal end — nodes ``1 … n``.
  Node ``0`` is the top attachment, whose force balance belongs to whatever
  body holds the chain up, not to the chain.
* **The caller owns the jet depth.**  An external-force jet is truncated to
  whatever it supplies, and that truncation propagates up the rest of the
  chain, so a short jet quietly costs derivative depth at the top.  The
  cascade rejects a jet shallower than the rung it enters carries — link
  ``i``'s force order ``K - 2(n - i)`` — which is exactly the order a vehicle
  at node ``i+1`` gets for free: that node's position comes out at
  ``K - 2(n - 1 - i)``, and its thrust vector costs the usual two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

from ..core.defaults import GRAVITY
from .cable import point_mass_cable_force, tension_direction_jet

__all__ = ["CableChain", "cable_chain_cascade", "chain_jet_order"]

_E3 = np.array([0.0, 0.0, 1.0])


def chain_jet_order(num_links: int) -> int:
    """Tip-position jet order an ``n``-link chain needs, ``2n + 4``.

    The depth at which the **top node** of the chain comes out through its snap
    — the requirement of any consumer that turns ``x_0`` into a thrust vector
    and then into a moment.  See the "Derivative depth" section of :mod:`udaan.flatness.chain`
    and :eq:`chain-order`.
    """
    if num_links < 1:
        raise ValueError(f"num_links must be >= 1, got {num_links}")
    return 2 * num_links + 4


@dataclass
class CableChain:
    """Result of a :func:`cable_chain_cascade` sweep.

    Every entry is a *derivative tuple* (a list whose ``k``-th element is the
    ``k``-th time derivative), truncated to whatever depth survived the
    recursion — deeper near the tip, shallower near the top, as tabulated
    under "Derivative depth" in :mod:`udaan.flatness.chain`.

    Attributes:
        node_positions: ``n + 1`` position jets.  ``node_positions[0]`` is the
            top attachment (available through snap at the documented input
            order) and ``node_positions[n]`` is the tip, returned as supplied.
        directions: ``n`` unit-direction jets, ``directions[i]`` being link
            ``i``'s ``q_i`` — pointing top→tip.
        tensions: ``n`` scalar jets, ``tensions[i][0]`` being the tension
            ``T_i = \\|F_i\\| \\ge 0`` carried by link ``i``.  A norm, so it is
            a one-sided margin only: it falls to zero at the slack boundary but
            grows again past it.  The two-sided refusal is the hanging-branch
            check — see "The hanging branch" in :mod:`udaan.flatness.chain`.
        forces: ``n`` cable-force jets, ``forces[i]`` being ``F_i`` of
            :eq:`chain-force` — the force link ``i`` exerts on the body
            immediately above it.
    """

    node_positions: list[list[np.ndarray]]
    directions: list[list[np.ndarray]]
    tensions: list[list[float]]
    forces: list[list[np.ndarray]]

    @property
    def num_links(self) -> int:
        """Number of links in the chain."""
        return len(self.directions)

    @property
    def top_position(self) -> list[np.ndarray]:
        """Position jet of node ``0`` — what holds the chain up."""
        return self.node_positions[0]

    @property
    def top_force(self) -> list[np.ndarray]:
        """``F_0``: the force the chain applies to the body at node ``0``.

        A body of mass ``m`` at that node with thrust vector ``u`` obeys
        ``m(\\ddot x_0 + g e_3) = u + F_0``, so the thrust vector a flatness map
        must produce is ``u = m(\\ddot x_0 + g e_3) - F_0``.
        """
        return self.forces[0]

    def angular_velocity(self, i: int) -> np.ndarray:
        """Link ``i``'s angular velocity ``\\omega_i = q_i \\times \\dot q_i``.

        The unique element of ``T_{q_i}S^2`` with ``\\dot q_i = \\omega_i \\times
        q_i``; ``q_i \\cdot \\omega_i = 0`` by construction.
        """
        q = self.directions[i]
        return np.cross(q[0], q[1])

    def angular_acceleration(self, i: int) -> np.ndarray:
        """Link ``i``'s angular acceleration ``\\dot\\omega_i = q_i \\times \\ddot q_i``.

        Differentiating ``\\omega_i = q_i \\times \\dot q_i`` gives
        ``\\dot\\omega_i = q_i \\times \\ddot q_i``, the ``\\dot q_i \\times \\dot
        q_i`` term vanishing.
        """
        q = self.directions[i]
        return np.cross(q[0], q[2])


def cable_chain_cascade(
    tip_position: Sequence[np.ndarray],
    masses: Sequence[float] | np.ndarray,
    lengths: Sequence[float] | np.ndarray,
    *,
    gravity: float = GRAVITY,
    node_forces: Sequence[Sequence[np.ndarray] | None] | None = None,
) -> CableChain:
    r"""Recurse position and tension up a serial cable chain, tip → top.

    Sweeps :eq:`chain-force` from the tip upwards, decomposing each link's
    force jet with :func:`~udaan.flatness.cable.tension_direction_jet` and
    propagating the node positions with ``x_i = x_{i+1} - \ell_i q_i``.  This is
    the whole chain half of the M5 flat map and (with ``node_forces``) of M6's.

    Args:
        tip_position: derivative tuple of the tip position, ``[x, ẋ, ẍ, …]``.
            A :class:`~udaan.flatness.Jet` works directly (it iterates over its
            rows).  Must reach at least :func:`chain_jet_order` for the top node
            to come out through its snap; extra rows are ignored.
        masses: ``n`` point masses, ``masses[i]`` lumped at the distal end of
            link ``i``.  Must be strictly positive.
        lengths: ``n`` link lengths, strictly positive.
        gravity: gravitational acceleration.
        node_forces: optional per-link external-force jets.  ``node_forces[i]``
            is applied at the distal end of link ``i`` (the node carrying
            ``masses[i]``); ``None`` entries and a ``None`` argument mean no
            external force.  A supplied jet must be a derivative tuple of
            3-vectors reaching order ``K - 2(n - i)``, the order link ``i``'s
            own force jet carries — pad a constant force with zero rows.
            Nothing can be applied at node ``0``.  See "External forces" in
            :mod:`udaan.flatness.chain`.

    Returns:
        A :class:`CableChain`.

    Raises:
        ValueError: on a malformed argument, on a tip or ``node_forces`` jet
            shallower than the depth accounting requires, when some link goes
            **slack** — its force jet vanishes and its direction is undefined —
            or when a link leaves the **hanging branch** (``q_i · e₃ > 0``,
            i.e. the trajectory has crossed the slack boundary rather than
            stopping at it).  The message names the offending link.
    """
    m = np.asarray(masses, dtype=float)
    length = np.asarray(lengths, dtype=float)
    if m.ndim != 1 or length.ndim != 1:
        raise ValueError("masses and lengths must be one-dimensional")
    n = m.size
    if n < 1:
        raise ValueError("cable_chain_cascade needs at least one link")
    if length.size != n:
        raise ValueError(f"masses and lengths must have equal length; got {n} and {length.size}")
    if not np.all(m > 0.0):
        raise ValueError(f"masses must be strictly positive; got {m.tolist()}")
    if not np.all(length > 0.0):
        raise ValueError(f"lengths must be strictly positive; got {length.tolist()}")

    tip = [np.asarray(row, dtype=float) for row in tip_position]
    required = chain_jet_order(n)
    if len(tip) < required + 1:
        raise ValueError(
            f"cable_chain_cascade: an {n}-link chain needs the tip position through its "
            f"{required}-th derivative (2n + 4); got order {len(tip) - 1}."
        )
    if any(row.shape != (3,) for row in tip):
        raise ValueError("tip_position rows must be 3-vectors")
    # Extra depth is harmless but never read — truncate so the recursion's
    # bookkeeping is exactly the documented one.
    tip = tip[: required + 1]

    external_jets: list[list[np.ndarray] | None] = [None] * n
    if node_forces is not None:
        if len(node_forces) != n:
            raise ValueError(f"node_forces must have {n} entries; got {len(node_forces)}")
        for i, entry in enumerate(node_forces):
            if entry is None:
                continue
            rows = [np.asarray(row, dtype=float) for row in entry]
            if any(row.shape != (3,) for row in rows):
                raise ValueError(f"node_forces[{i}] rows must be 3-vectors")
            # An external jet truncates the whole chain above it, so a short one
            # would silently cost derivative depth at the top and surface as an
            # IndexError there.  Demand the depth the rung itself carries.
            needed = required - 2 * (n - i)
            if len(rows) < needed + 1:
                raise ValueError(
                    f"cable_chain_cascade: node_forces[{i}] must reach the "
                    f"{needed}-th derivative (K - 2(n - i) with K = {required}); "
                    f"got order {len(rows) - 1}.  Pad a constant force with zero rows."
                )
            external_jets[i] = rows

    node_positions: list[list[np.ndarray]] = [[] for _ in range(n + 1)]
    node_positions[n] = tip
    directions: list[list[np.ndarray]] = [[] for _ in range(n)]
    tensions: list[list[float]] = [[] for _ in range(n)]
    forces: list[list[np.ndarray]] = [[] for _ in range(n)]

    # ``below`` is F_{i+1}, the force jet arriving from the sub-chain already
    # swept; ``None`` on the first (tip) rung, where there is nothing below.
    below: list[np.ndarray] | None = None

    for i in range(n - 1, -1, -1):
        x_node = node_positions[i + 1]  # position jet of the node link i carries
        accel = x_node[2:]  # acceleration jet — two orders shallower

        # F_i = -m_i (ẍ_{i+1} + g e_3) + u_i + F_{i+1}, truncated to the
        # shallowest contribution (which is always the mass term).
        force = point_mass_cable_force(accel, float(m[i]), gravity)
        external = external_jets[i]
        if external is not None:
            force = _add_jets(force, external)
        if below is not None:
            force = _add_jets(force, below)

        try:
            tension, direction = tension_direction_jet(force)
        except ValueError as exc:  # slack link — ‖F_i‖ ≈ 0, q_i undefined
            raise ValueError(
                f"cable_chain_cascade: link {i} of {n} is slack "
                f"(‖F_{i}‖ ≈ {float(np.linalg.norm(force[0])):.3e} N) — its direction "
                f"is undefined, so the taut-chain flat map does not apply here."
            ) from exc

        # ‖F_i‖ turns positive again the moment the trajectory overshoots the
        # slack point, so the norm alone only refuses to *land* on the boundary,
        # never to cross it.  Requiring the link to stay on the hanging branch
        # is what makes the refusal two-sided — see "The hanging branch" above.
        if float(np.dot(direction[0], _E3)) > 0.0:
            raise ValueError(
                f"cable_chain_cascade: link {i} of {n} has left the hanging branch "
                f"(q_{i}·e₃ = {float(np.dot(direction[0], _E3)):+.3f} > 0, so the node "
                f"below it sits above the node above it).  A rigid link is well posed "
                f"there but a cable is not: the trajectory has crossed ‖F_{i}‖ = 0."
            )

        # x_i = x_{i+1} - ℓ_i q_i, at whatever depth both sides support.
        depth = min(len(x_node), len(direction))
        node_positions[i] = [x_node[k] - length[i] * direction[k] for k in range(depth)]

        directions[i] = direction
        tensions[i] = tension
        forces[i] = force
        below = force

    return CableChain(
        node_positions=node_positions,
        directions=directions,
        tensions=tensions,
        forces=forces,
    )


def _add_jets(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Sum two derivative tuples, truncating to the shallower of the two."""
    depth = min(len(a), len(b))
    return [a[k] + b[k] for k in range(depth)]
