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
none, and M5 passes nothing.  A vehicle bolted to node ``i+1`` contributes its
thrust-vector jet ``f R e_3`` there (and folds its mass into ``masses``).

Two contract facts:

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

.. _chain-prescribed-forces:

Prescribed link forces — the second consumer's generalisation
-------------------------------------------------------------
``node_forces`` is the right hook only when the applied force is *known*.  M6
(:mod:`udaan.flatness.multi_quad_hose`) is the second consumer of this
primitive and it is not in that position: a hose's vehicles are interior to the
chain and their thrusts are what the map is solving for.  What M6 knows instead
is the **link force itself** at each vehicle — the tensions are its flat
outputs — so it needs the dual hook, and that is ``link_forces``:

* ``node_forces[i] = u``  →  ``F_i := (\text{accumulated}) + u``, force out.
* ``link_forces[i] = F``  →  ``F_i := F``, and the applied force
  ``u_i = F - (\text{accumulated})`` comes back out.

Both write the same rung of the same recursion from opposite ends, so the two
are mutually exclusive per link, carry the same depth requirement, and land in
the same place: :attr:`CableChain.applied_forces` reports ``u_i`` either way.
A prescribed force cuts nothing out of the sweep — the accumulation below it is
exactly what ``u_i`` is measured against.

The hanging branch is not universal
-----------------------------------
The second generalisation is a *removal*.  The hanging-branch test below is
sound for a chain suspended from its top, which is every configuration M5 can
be in.  It is wrong for a hose held up at **both** ends: there the links near
the far vehicle genuinely point upwards, and the test refuses the
configuration outright.  ``require_hanging_branch=False`` turns it off.

Precisely which configurations it refuses is worth stating, because it is
narrower than "every catenary": what the test rejects is a chain whose *own*
sweep runs uphill.  A hose with a single vehicle — at the last vertex, the
only place M6 allows one — hangs off that vehicle exactly as M5's cable does,
and passes the test unchanged (measured: ``(n, attachments)`` of ``(2, (2,))``,
``(3, (3,))`` and ``(4, (4,))`` are all accepted with the check on).  It is
``nQ ≥ 2`` — a vehicle at some interior or root vertex pulling part of the
chain up towards it — that the test refuses.

**What turning it off costs.**  Not nothing.  ``T_i = \|F_i\|`` is a norm on
every rung the recursion *derives*, prescribed or not, so those rungs flip
through zero exactly as `The hanging branch`_ describes, and with the test off
there is nothing to catch them.  A prescribed rung is genuinely immune — the
caller hands over the force vector, sign and all — but a chain of ``n`` links
with ``p`` prescribed rungs still has ``n - p`` derived ones that can fold.
``min_tension`` is the partial replacement: the fold always happens where a
derived ``\|F_i\|`` passes through zero, so refusing a derived rung below a
floor refuses the *neighbourhood* of the fold.  It is a margin, not a branch
test — a sample grid coarse enough to step clean over that neighbourhood can
still land past a fold undetected — and it is the caller's job to set the floor
to a meaningful fraction of the tension it plans to hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
            ``T_i = \\|F_i\\| \\ge 0`` carried by link ``i``.  A norm on every
            rung the recursion derives, so it is a one-sided margin only: it
            falls to zero at the slack boundary but grows again past it.  The
            two-sided refusal is the hanging-branch check, and where that
            cannot be used ``min_tension`` refuses the boundary's
            neighbourhood instead — see "The hanging branch" and "The hanging
            branch is not universal" in :mod:`udaan.flatness.chain`.  On a rung
            whose force the caller *prescribed* the margin is two-sided
            already: the force vector is given, not inferred from a norm.
        forces: ``n`` cable-force jets, ``forces[i]`` being ``F_i`` of
            :eq:`chain-force` — the force link ``i`` exerts on the body
            immediately above it.
        applied_forces: ``n`` external-force jets, ``applied_forces[i]`` being
            the force ``u_i`` applied at the distal end of link ``i`` (node
            ``i+1``).  It is the ``node_forces`` entry where one was supplied,
            the *recovered* ``F_i - (\\text{accumulated})`` where a
            ``link_forces`` entry was, and zeros where neither — so a consumer
            that prescribes link forces reads its vehicles' thrust vectors
            straight out of here.  See :ref:`chain-prescribed-forces`.
    """

    node_positions: list[list[np.ndarray]]
    directions: list[list[np.ndarray]]
    tensions: list[list[float]]
    forces: list[list[np.ndarray]]
    applied_forces: list[list[np.ndarray]] = field(default_factory=list)

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
    link_forces: Sequence[Sequence[np.ndarray] | None] | None = None,
    require_hanging_branch: bool = True,
    min_tension: float = 0.0,
) -> CableChain:
    r"""Recurse position and tension up a serial cable chain, tip → top.

    Sweeps :eq:`chain-force` from the tip upwards, decomposing each link's
    force jet with :func:`~udaan.flatness.cable.tension_direction_jet` and
    propagating the node positions with ``x_i = x_{i+1} - \ell_i q_i``.  This is
    the whole chain half of the M5 flat map, and (with ``link_forces`` and
    ``require_hanging_branch=False``) of M6's.

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
        link_forces: optional per-link **prescribed** force jets — the dual of
            ``node_forces``.  ``link_forces[i]`` fixes ``F_i`` outright instead
            of accumulating it, and the external force that makes the rung
            balance is recovered into
            :attr:`CableChain.applied_forces`\\ ``[i]``.  Same depth
            requirement, and mutually exclusive with ``node_forces[i]``.  See
            :ref:`chain-prescribed-forces`.
        require_hanging_branch: refuse a link with ``q_i · e₃ > 0``.  On for a
            chain that hangs from its top; a caller whose chain is held up at
            both ends (a hose with two or more vehicles) must turn it off,
            because its far links point upwards legitimately.
        min_tension: floor, in newtons, on the tension of every rung the
            recursion **derives** — those with no ``link_forces`` entry.  Zero
            (the default) disables it.  It is the partial stand-in for the
            branch test when that has to be turned off: a derived rung folds
            exactly where its ``\|F_i\|`` passes through zero, so a floor
            refuses the neighbourhood of the fold.  It is a margin, not a
            branch test — see "The hanging branch is not universal" in
            :mod:`udaan.flatness.chain` for what it does and does not catch.
            Rungs whose force was *prescribed* are exempt: their sign is the
            caller's, not a norm's.

    Returns:
        A :class:`CableChain`.

    Raises:
        ValueError: on a malformed argument, on a tip, ``node_forces`` or
            ``link_forces`` jet shallower than the depth accounting requires,
            when some link goes **slack** — its force jet vanishes and its
            direction is undefined — when a *derived* link's tension falls
            below ``min_tension`` — or, with ``require_hanging_branch``, when
            a link leaves the **hanging branch** (``q_i · e₃ > 0``, i.e. the
            trajectory has crossed the slack boundary rather than stopping at
            it).  The message names the offending link.
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

    external_jets = _parse_force_jets(node_forces, "node_forces", n, required)
    prescribed_jets = _parse_force_jets(link_forces, "link_forces", n, required)
    both = [i for i in range(n) if external_jets[i] is not None and prescribed_jets[i] is not None]
    if both:
        raise ValueError(
            f"cable_chain_cascade: link {both[0]} has both a node_forces and a link_forces "
            "entry.  They write the same rung from opposite ends — supply one or the other."
        )

    node_positions: list[list[np.ndarray]] = [[] for _ in range(n + 1)]
    node_positions[n] = tip
    directions: list[list[np.ndarray]] = [[] for _ in range(n)]
    tensions: list[list[float]] = [[] for _ in range(n)]
    forces: list[list[np.ndarray]] = [[] for _ in range(n)]
    applied: list[list[np.ndarray]] = [[] for _ in range(n)]

    # ``below`` is F_{i+1}, the force jet arriving from the sub-chain already
    # swept; ``None`` on the first (tip) rung, where there is nothing below.
    below: list[np.ndarray] | None = None

    for i in range(n - 1, -1, -1):
        x_node = node_positions[i + 1]  # position jet of the node link i carries
        accel = x_node[2:]  # acceleration jet — two orders shallower

        # The part of F_i the recursion itself supplies:
        #   -m_i (ẍ_{i+1} + g e_3) + F_{i+1},
        # truncated to the shallowest contribution (always the mass term).
        base = point_mass_cable_force(accel, float(m[i]), gravity)
        if below is not None:
            base = _add_jets(base, below)

        # F_i = base + u_i.  Either side of that identity may be the known one:
        # ``node_forces`` gives u_i and the force comes out; ``link_forces``
        # gives F_i and the force *applied* comes out.  See
        # :ref:`chain-prescribed-forces`.
        prescribed = prescribed_jets[i]
        if prescribed is not None:
            force = prescribed
            depth = min(len(force), len(base))
            external = [force[k] - base[k] for k in range(depth)]
        else:
            external = external_jets[i] or [np.zeros(3) for _ in base]
            force = _add_jets(base, external)
            external = external[: len(force)]

        try:
            tension, direction = tension_direction_jet(force)
        except ValueError as exc:  # slack link — ‖F_i‖ ≈ 0, q_i undefined
            raise ValueError(
                f"cable_chain_cascade: link {i} of {n} is slack "
                f"(‖F_{i}‖ ≈ {float(np.linalg.norm(force[0])):.3e} N) — its direction "
                f"is undefined, so the taut-chain flat map does not apply here."
            ) from exc

        # A derived rung's force is a sum that can cancel, so its tension walks
        # to zero and back out the far side with the direction flipped.  The
        # floor refuses the neighbourhood of that crossing.  A prescribed rung
        # is exempt: the caller gave the vector, so its sign is not in doubt.
        if min_tension > 0.0 and prescribed is None and float(tension[0]) < min_tension:
            raise ValueError(
                f"cable_chain_cascade: link {i} of {n} is below the tension floor "
                f"(T_{i} = {float(tension[0]):.3e} N < min_tension = {min_tension:.3e} N).  "
                f"Its force is *derived*, so ‖F_{i}‖ is a norm: past the crossing the link "
                f"folds to the other side with the tension rising again and nothing to say "
                f"so.  Hold more tension through link {i}, or lower min_tension to accept "
                f"the risk."
            )

        # ‖F_i‖ turns positive again the moment the trajectory overshoots the
        # slack point, so the norm alone only refuses to *land* on the boundary,
        # never to cross it.  Requiring the link to stay on the hanging branch
        # is what makes the refusal two-sided — see "The hanging branch" above.
        if require_hanging_branch and float(np.dot(direction[0], _E3)) > 0.0:
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
        applied[i] = external
        below = force

    return CableChain(
        node_positions=node_positions,
        directions=directions,
        tensions=tensions,
        forces=forces,
        applied_forces=applied,
    )


def _parse_force_jets(
    entries: Sequence[Sequence[np.ndarray] | None] | None,
    name: str,
    n: int,
    required: int,
) -> list[list[np.ndarray] | None]:
    """Validate a per-link force-jet argument and return it row-wise.

    Shared by ``node_forces`` and ``link_forces``: both attach one 3-vector jet
    to link ``i``'s rung and both are subject to the same depth rule, so the
    two must not be allowed to drift apart in their checking.
    """
    parsed: list[list[np.ndarray] | None] = [None] * n
    if entries is None:
        return parsed
    if len(entries) != n:
        raise ValueError(f"{name} must have {n} entries; got {len(entries)}")
    for i, entry in enumerate(entries):
        if entry is None:
            continue
        rows = [np.asarray(row, dtype=float) for row in entry]
        if any(row.shape != (3,) for row in rows):
            raise ValueError(f"{name}[{i}] rows must be 3-vectors")
        # A force jet truncates the whole chain above it, so a short one would
        # silently cost derivative depth at the top and surface as an
        # IndexError there.  Demand the depth the rung itself carries.
        needed = required - 2 * (n - i)
        if len(rows) < needed + 1:
            raise ValueError(
                f"cable_chain_cascade: {name}[{i}] must reach the "
                f"{needed}-th derivative (K - 2(n - i) with K = {required}); "
                f"got order {len(rows) - 1}.  Pad a constant force with zero rows."
            )
        # Extra depth is truncated, exactly as the tip jet is, so the rung's
        # bookkeeping is the documented one whichever side supplied the force.
        parsed[i] = rows[: needed + 1]
    return parsed


def _add_jets(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Sum two derivative tuples, truncating to the shallower of the two."""
    depth = min(len(a), len(b))
    return [a[k] + b[k] for k in range(depth)]
