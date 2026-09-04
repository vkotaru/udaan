r"""Differential flatness for ``nQ`` quadrotors carrying one flexible hose — M6.

Flat output::

    y = (x_0, {𝛕_a : a ∈ attachments, a ≠ n}, {ψ_k}_{k=1..nQ})

where ``x_0 ∈ R³`` is the **root vertex** position, ``𝛕_a ∈ R³`` is the *vector*
tension in the link hanging off the vehicle at vertex ``a``, and ``ψ_k`` is
vehicle ``k``'s yaw.  Counting: ``3 + 3(nQ − 1) + nQ = 4 nQ = dim u``, the
dimension-matching condition.  The dynamics being inverted are
:class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`.

Why tensions and not positions
------------------------------
Every earlier map in this package takes *positions* as its flat output — the
payload's in M2/M3/M5, the payload pose in M4 — and the vehicles fall out of
them.  That works because the load is a single body whose force balance closes
the recursion.  A hose has ``n + 1`` bodies in series and a vehicle may be
bolted to any of them, so prescribing the root trajectory closes only the root
vertex's balance; every vehicle vertex below it has *two* unknowns (the thrust,
and the force in the link continuing past it) and one equation.

Handing the link force over as a flat output is what removes the shortfall, and
it is the same move M3 makes with its internal cable forces and M4 with the
grasp null space: the free coordinate is paid for out of the flat-output
budget.  The book-keeping is exact — ``3(nQ − 1)`` free force vectors is
precisely the ``4nQ − 3 − nQ`` the count leaves over.  The choice is also the
*useful* one physically: it is the tension, not the vehicle position, that has
to stay positive for a hose to be a hose, so a planner that works in these
coordinates can enforce tautness by construction.

The price is that the vehicles' positions are no longer prescribed.  They come
out of the recursion, and steering them to particular points is an inverse
problem the legacy reference solves numerically (``optimize_for_des_state``);
it is a planner's job, not this map's.

**A vehicle must sit at the last vertex.**  Vertex ``n`` has no link below it,
so ``𝛕_n = 0`` is not a free choice — the vertex's force balance is
``m̄_n(\ddot x_n + g e_3) = u_n + (\text{force from link } n-1)``, and with no
``u_n`` to absorb it that is three equations the flat output must satisfy
rather than three quantities it determines.  The system is then **not flat**
in these coordinates, and :class:`MultiQuadHose` refuses to be constructed.
The legacy reference has the same guard (``flat2state.m`` L107,
``error('Flatness Error: This system is not differentially-flat')``) and its
examples all force the last vertex into the attachment set for exactly this
reason.

The recursion
-------------
Vertices ``j = 0 … n``; link ``i`` runs from vertex ``i`` to vertex ``i+1``,
so ``q_i`` points *down the chain* and ``x_{i+1} = x_i + \ell_i q_i`` —
:mod:`udaan.models.multi_quad_hose`'s convention exactly.  Write ``𝛕_i`` for
the force link ``i`` exerts on its **proximal** vertex ``i``; then
``q_i = 𝛕_i/\|𝛕_i\|`` and vertex ``j``'s balance is

.. math::
    :label: m6f-vertex

    m̄_j (\ddot x_j + g e_3) \;=\; u_j \;-\; 𝛕_{j-1} \;+\; 𝛕_j ,
    \qquad 𝛕_{-1} := 0, \quad 𝛕_n := 0 ,

with ``u_j`` the thrust vector at vertex ``j`` (zero where there is no
vehicle).  Read as a top-down sweep it is a *definition* at every vertex, never
a solve:

======================================  ==============================================
Vertex ``j``                            :eq:`m6f-vertex` gives
======================================  ==============================================
no vehicle, ``j < n``                   ``𝛕_j = m̄_j(\ddot x_j + g e_3) + 𝛕_{j-1}``
vehicle, ``j < n`` (``𝛕_j`` is flat)    ``u_j = m̄_j(\ddot x_j + g e_3) + 𝛕_{j-1} − 𝛕_j``
vehicle at ``j = n``                    ``u_n = m̄_n(\ddot x_n + g e_3) + 𝛕_{n-1}``
no vehicle at ``j = n``                 *nothing* — the system is not flat
======================================  ==============================================

and after each rung ``x_{j+1} = x_j + \ell_j q_j``.  This is the reference's
recursion term for term, with the two sign conventions reconciled: its ``T_i``
is the force in link ``i`` under 1-based link numbering, i.e. this module's
``𝛕_{i-1}``.

Which sweep, and why it runs backwards
--------------------------------------
:func:`~udaan.flatness.chain.cable_chain_cascade` — the M5 primitive — is
*already* this recursion.  Its ``F_i`` is the force link ``i`` exerts on the
body above it, its ``q_i = F_i/\|F_i\|`` points top→tip, and its rung

.. math::

    F_i \;=\; -m_i(\ddot x_{i+1} + g e_3) \;+\; u_i \;+\; F_{i+1}

is :eq:`m6f-vertex` rearranged.  What differs is only *which end drives it*:
the cascade sweeps tip→top from a prescribed tip position, and M6 prescribes
its **root**.  The two are the same sweep read in opposite index order, so this
module reverses the chain — cascade node ``c`` is hose vertex ``n − c`` — and
calls the primitive once.  Under that relabelling:

* the hose's **root** ``x_0`` is the cascade's **tip**, so ``x_0``'s jet order
  is the one :func:`~udaan.flatness.chain.chain_jet_order` constrains;
* the hose's **last vertex** — the one that must carry a vehicle — is the
  cascade's node ``0``, "whatever holds the chain up", and its thrust is
  recovered from :attr:`~udaan.flatness.chain.CableChain.top_force` by the
  relation that attribute already documents.  **The not-flat guard is
  literally the cascade's own contract**: node ``0`` is where the chain hands
  its load to a body, and a hose with no vehicle there has no such body;
* a flat tension ``𝛕_a`` is a **prescribed link force** at cascade rung
  ``n − 1 − a``, and the vehicle's thrust comes back as that rung's
  :attr:`~udaan.flatness.chain.CableChain.applied_forces` entry.  Signs flip
  with the reversal: ``F_{n-1-i} = -𝛕_i`` and ``q^{\text{cascade}}_{n-1-i} =
  -q_i``, because the cascade's directions point the other way along the same
  links.

Two things the second consumer needed from the primitive, and one it did not:

* ``link_forces`` — **new**.  ``node_forces`` prescribes the *applied* force at
  a node, which is the right hook when the vehicle's thrust is known.  M6's
  vehicles are interior and their thrusts are unknowns; what it knows is the
  link force.  See :ref:`chain-prescribed-forces`.
* ``require_hanging_branch=False`` at ``nQ ≥ 2`` — **new**, and the one place
  M6 gives something up.  A hose strung between two or more vehicles has links
  that legitimately point upwards, and the test refuses such a configuration
  outright, so it has to go.  At ``nQ = 1`` it does *not*: a hose with its one
  vehicle at the last vertex hangs off that vehicle exactly as M5's cable
  does, so the map leaves the test **on** there and keeps M5's full
  protection.  See `What is and is not guarded`_.
* The depth accounting, the per-rung normalisation, the position propagation
  and the slack refusal needed **nothing** — they are used unchanged.

What is and is not guarded
--------------------------
Only the links hanging directly off a vehicle have their force *prescribed*.
The other ``n − (nQ − 1)`` are **derived** —
``𝛕_j = m̄_j(\ddot x_j + g e_3) + 𝛕_{j-1}`` — and their directions come out of
a norm, ``q_j = 𝛕_j/\|𝛕_j\|``, exactly as M5's do.  A derived tension that
passes through zero therefore folds its link to the other side and comes back
out with the tension rising again, and neither the slack refusal (which only
declines to *land* on zero) nor the prescribed tensions upstream of it say
anything about that.  Two guards, and one gap:

``nQ = 1``
    Fully guarded.  There are no prescribed rungs, M6 *is* M5, and
    ``require_hanging_branch`` is left on: a root dive through
    ``\ddot x_0 · e_3 = -g`` is refused rather than silently folded.

``nQ ≥ 2``, prescribed rungs
    Two-sided by construction — and only here.  ``𝛕_a`` is handed over as a
    vector, so its direction is the caller's choice and its magnitude walks to
    zero without any branch to flip onto.

``nQ ≥ 2``, derived rungs
    **Guarded only by a margin.**  No per-rung direction test exists: a
    derived link may legitimately point down while a prescribed one points up,
    so there is no fixed half-space to demand.  What the map does instead is
    what M3 and M4 do — a ``min_tension`` floor, applied to the derived rungs
    (:attr:`MultiQuadHose.min_tension`).  The fold always happens where a
    derived tension crosses zero, so the floor refuses the *neighbourhood* of
    the fold; a sample grid coarse enough to step clean over that
    neighbourhood can still land past a fold undetected.  Set the floor to a
    meaningful fraction of the tension being planned, not to the slack
    default.

Derivative orders required::

    x_0  through its (2n + 4)-th derivative
    𝛕_a  through its (2(n − a) + 2)-th derivative
    ψ_k  through ψ̈_k

The ``2n + 4`` is the same law M5 obeys and for the same reason — each rung
spends two derivatives — but the *binding* consumer is different.  In M5 it is
the single airframe at the top of the chain; here it is the mandatory vehicle
at the last vertex, whose position comes out at ``K − 2n`` and whose moment
costs the usual four.  :func:`~udaan.flatness.chain.chain_jet_order` returns
it, and :attr:`MultiQuadHose.tension_jet_orders` returns the per-tension
depths, which are exactly what a prescribed force at that rung must carry.

Reference:
    P. Kotaru and K. Sreenath, "Multiple Quadrotors Carrying a Flexible Hose:
    Dynamics, Differential Flatness and Control", IFAC World Congress, 2020.
    The recursion above was re-derived rather than transcribed; see
    ``docs/theory/differential-flatness/multi-quad-hose.md`` for what that
    turned up in the reference's own ``flat2state.m``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.defaults import (
    DEFAULT_HOSE_LENGTH,
    DEFAULT_HOSE_MASS,
    DEFAULT_NUM_HOSE_LINKS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ..core.types import Vec3
from ..manif import S2, SO3, TS2, TSO3
from ..utils.hose import validate_attachments
from .base import Flat2State
from .chain import cable_chain_cascade, chain_jet_order
from .jet import Jet
from .multi_quad_cspayload import DEFAULT_MIN_TENSION
from .quadrotor import _attitude_from_thrust_vector

_E3 = np.array([0.0, 0.0, 1.0])

__all__ = [
    "HoseCascade",
    "MultiQuadHose",
    "MultiQuadHoseFlats",
    "MultiQuadHoseInputs",
    "MultiQuadHoseRefState",
    "NotFlatError",
]


class NotFlatError(ValueError):
    """The attachment pattern makes the hose non-flat in these coordinates.

    Raised when no vehicle sits at the last vertex.  It is a *structural*
    refusal — a statement about the mechanism, not about the trajectory being
    planned — so it is a distinct type from the slack-link and free-fall
    :class:`ValueError`\\ s a caller may legitimately absorb by dropping the
    feedforward for one step.  No trajectory rescues it; only moving a vehicle
    does.
    """


def _zeros3() -> Vec3:
    return np.zeros(3)


# ─── Cascade result, in hose indexing ─────────────────────────────────


@dataclass
class HoseCascade:
    """Geometry, link forces and thrusts of the hose, in **hose** indexing.

    :func:`~udaan.flatness.chain.cable_chain_cascade` returns its answer in the
    cascade's own tip→top order; this is the same answer relabelled to the
    hose's vertex/link numbering, with the direction and force signs flipped
    back.  Every entry is a derivative tuple, truncated to whatever depth
    survived the recursion — deepest at the root, shallowest at the last
    vertex, two orders per rung.

    Attributes:
        vertex_positions: ``n + 1`` position jets, ``vertex_positions[j]``
            being vertex ``j``.  ``[0]`` is the root, returned as supplied;
            ``[n]`` is the last vertex, available through its snap at the
            documented input order.
        directions: ``n`` unit-direction jets; ``directions[i]`` is link
            ``i``'s ``q_i``, pointing vertex ``i`` → vertex ``i+1``.
        tensions: ``n`` scalar jets; ``tensions[i][0] = \\|𝛕_i\\| \\ge 0``.
        forces: ``n`` link-force jets; ``forces[i]`` is ``𝛕_i``, the force link
            ``i`` exerts on its proximal vertex ``i``.
        vertex_thrusts: ``n + 1`` entries; the thrust-vector jet ``u_j`` at
            each vertex carrying a vehicle and ``None`` at every other vertex.
    """

    vertex_positions: list[list[np.ndarray]]
    directions: list[list[np.ndarray]]
    tensions: list[list[float]]
    forces: list[list[np.ndarray]]
    vertex_thrusts: list[list[np.ndarray] | None]

    @property
    def num_links(self) -> int:
        """Number of links in the hose."""
        return len(self.directions)

    def angular_velocity(self, i: int) -> np.ndarray:
        """Link ``i``'s angular velocity ``ω_i = q_i × q̇_i``."""
        q = self.directions[i]
        return np.cross(q[0], q[1])

    def angular_acceleration(self, i: int) -> np.ndarray:
        """Link ``i``'s angular acceleration ``ω̇_i = q_i × q̈_i``."""
        q = self.directions[i]
        return np.cross(q[0], q[2])


# ─── Flat output / reference state / inputs ───────────────────────────


@dataclass
class MultiQuadHoseFlats:
    """Flat output of M6 at one instant.

    Attributes:
        x_0:  dim-3 :class:`Jet` of the **root vertex** position.  The order it
              must reach depends on the link count —
              :func:`~udaan.flatness.chain.chain_jet_order` gives ``2n + 4`` —
              so the binding check lives in :meth:`MultiQuadHose.recover`,
              which knows ``n``.  What is checked here is the
              ``n``-independent floor: dim 3 and order ≥ 6.
        tensions: one dim-3 :class:`Jet` per vehicle **not** at the last
              vertex, in vehicle order — ``𝛕_a``, the vector force in the link
              hanging off that vehicle.  There are ``nQ − 1`` of them and their
              required orders are :attr:`MultiQuadHose.tension_jet_orders`.
        psi:  ``nQ`` scalar :class:`Jet`\\ s of the per-vehicle yaw, order ≥ 2.
    """

    x_0: Jet
    tensions: list[Jet] = field(default_factory=list)
    psi: list[Jet] = field(default_factory=list)

    def __post_init__(self):
        if self.x_0.dim != 3:
            raise ValueError(f"x_0 must be a 3-D jet; got dim={self.x_0.dim}")
        if self.x_0.order < 6:
            raise ValueError(
                f"x_0.order must be >= 6 (the n = 1 floor; an n-link hose needs "
                f"2n + 4); got {self.x_0.order}"
            )
        for k, tension in enumerate(self.tensions):
            if tension.dim != 3:
                raise ValueError(f"tensions[{k}] must be a 3-D jet; got dim={tension.dim}")
        for k, psi in enumerate(self.psi):
            if not psi.is_scalar:
                raise ValueError(f"psi[{k}] must be a scalar jet; got dim={psi.dim}")
            if psi.order < 2:
                raise ValueError(f"psi[{k}].order must be >= 2; got {psi.order}")


@dataclass
class MultiQuadHoseRefState:
    """Recovered kinematic reference state for ``nQ`` quadrotors on one hose.

    Field naming mirrors
    :class:`~udaan.models.multi_quad_hose.MultiQuadHoseState` where the two
    overlap; the ``*_acceleration`` fields and the full vertex kinematics are
    extras the flat map produces for free.  Per-link lists are ordered along
    the hose (root → last vertex) and per-vehicle lists in vehicle order.
    """

    # Root vertex — the model's integrated translational state.
    root_position: Vec3 = field(default_factory=_zeros3)
    root_velocity: Vec3 = field(default_factory=_zeros3)
    root_acceleration: Vec3 = field(default_factory=_zeros3)

    # Per-link direction / rate / rate-derivative.
    link_attitudes: list[Vec3] = field(default_factory=list)
    link_angular_velocities: list[Vec3] = field(default_factory=list)
    link_angular_accelerations: list[Vec3] = field(default_factory=list)

    # Every vertex, ``(n + 1, 3)`` each — row ``j`` is vertex ``j``.
    vertex_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    vertex_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Per-vehicle kinematics.
    positions: list[Vec3] = field(default_factory=list)
    velocities: list[Vec3] = field(default_factory=list)
    orientations: list[SO3] = field(default_factory=list)
    angular_velocities: list[TSO3] = field(default_factory=list)
    angular_accelerations: list[Vec3] = field(default_factory=list)

    def as_state(self, attachments=None):
        """Project onto the model's ``MultiQuadHoseState`` shape.

        Drops the acceleration extras and the vertex kinematics, and wraps the
        per-link entries as :class:`~udaan.manif.S2` / :class:`~udaan.manif.TS2`.
        ``attachments`` defaults to the pattern implied by nothing at all — pass
        the map's own :attr:`MultiQuadHose.attachments` so the state agrees.
        """
        from ..models.multi_quad_hose.base import MultiQuadHoseState

        n = len(self.link_attitudes)
        state = MultiQuadHoseState(num_links=n, attachments=attachments)
        state.root_position = np.asarray(self.root_position, dtype=float)
        state.root_velocity = np.asarray(self.root_velocity, dtype=float)
        state.link_attitudes = [S2(np.asarray(q)) for q in self.link_attitudes]
        state.link_angular_velocities = [TS2(np.asarray(w)) for w in self.link_angular_velocities]
        state.orientations = [SO3(np.asarray(R)) for R in self.orientations]
        state.angular_velocities = [TSO3(np.asarray(w)) for w in self.angular_velocities]
        state.positions = [np.asarray(p, dtype=float) for p in self.positions]
        state.velocities = [np.asarray(v, dtype=float) for v in self.velocities]
        return state


@dataclass
class MultiQuadHoseInputs:
    """Feedforward inputs recovered from the flat output.

    Attributes:
        thrusts: ``(nQ,)`` collective thrust magnitudes, in vehicle order.
        moments: ``(nQ, 3)`` body-frame moments.
        tensions: ``(n,)`` link tension magnitudes ``\\|𝛕_i\\| \\ge 0``.  Not
            an actuator input — the hose's internal forces — but the slack
            diagnostic: ``tensions[i] → 0`` is the boundary at which link ``i``
            goes limp and the taut-hose map ceases to apply.  It is two-sided
            only on the links whose force is a **flat output** (those hanging
            off a vehicle other than the last): there the vector is given, not
            inferred from a norm, so there is no branch to flip onto.  On every
            *derived* link it is the same one-sided norm M5 reports, and what
            stands between a trajectory and a folded hose is the
            :attr:`MultiQuadHose.min_tension` floor — see :ref:`m6-guards`.
    """

    thrusts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    moments: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tensions: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_wrench(self) -> np.ndarray:
        """``(nQ, 4)`` stack of ``[thrust, Mx, My, Mz]`` — a ``step()`` input."""
        return np.column_stack([self.thrusts, self.moments])


# ─── The map ──────────────────────────────────────────────────────────


class MultiQuadHose(Flat2State):
    """Differential-flatness recovery for ``nQ`` quadrotors on one flexible hose.

    Parameters mirror
    :class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`: ``vertex_masses``
    is the hose's own lumped mass at each of the ``n + 1`` vertices and
    ``masses`` is the per-vehicle airframe mass; what enters the recursion is
    their sum at each vertex.  The ``hose_mass`` / ``hose_length`` convenience
    spreads one number evenly, exactly as the model does, so
    :meth:`from_model` and a hand-built map agree.

    ``min_tension`` is the floor under every link whose force the recursion
    derives, and at ``nQ ≥ 2`` it is the *only* thing standing between a
    trajectory and a silently folded link; :ref:`m6-guards` says what it does
    and does not catch.

    Raises:
        NotFlatError: if no vehicle sits at the last vertex — see "Why tensions
            and not positions" in the module docstring.
    """

    Flats = MultiQuadHoseFlats
    RefState = MultiQuadHoseRefState
    Inputs = MultiQuadHoseInputs

    def __init__(
        self,
        *,
        num_links: int = DEFAULT_NUM_HOSE_LINKS,
        attachments=None,
        vertex_masses=None,
        link_lengths=None,
        hose_length: float = DEFAULT_HOSE_LENGTH,
        hose_mass: float = DEFAULT_HOSE_MASS,
        masses=None,
        inertias=None,
        gravity: float = GRAVITY,
        min_tension: float = DEFAULT_MIN_TENSION,
    ):
        self.num_links = int(num_links)
        if self.num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {self.num_links}")
        n = self.num_links

        self.attachments = validate_attachments((0, n) if attachments is None else attachments, n)
        if self.attachments[-1] != n:
            raise NotFlatError(
                f"a hose with no vehicle at its last vertex is not differentially flat in "
                f"these coordinates: vertex {n} has no link below it, so its force balance "
                f"is three equations on the flat output rather than three quantities it "
                f"determines.  Got attachments {self.attachments} for a {n}-link hose; add "
                f"vertex {n}."
            )
        nQ = len(self.attachments)
        self.num_quadrotors = nQ

        if link_lengths is None:
            lengths = np.full(n, float(hose_length) / n)
        else:
            lengths = self._broadcast(link_lengths, "link_lengths", n)
        if vertex_masses is None:
            hose = np.full(n + 1, float(hose_mass) / (n + 1))
        else:
            hose = self._broadcast(vertex_masses, "vertex_masses", n + 1)
        quad_masses = (
            np.full(nQ, DEFAULT_QUAD_MASS)
            if masses is None
            else self._broadcast(masses, "masses", nQ)
        )

        if not np.all(lengths > 0.0):
            raise ValueError(f"link_lengths must be strictly positive; got {lengths.tolist()}")
        if np.any(hose < 0.0):
            raise ValueError(f"vertex_masses must be non-negative; got {hose.tolist()}")
        if not np.all(quad_masses > 0.0):
            raise ValueError(f"masses must be strictly positive; got {quad_masses.tolist()}")

        self.link_lengths = lengths
        self.vertex_masses = hose
        self.masses = quad_masses

        #: ``m̄_j`` — hose mass plus airframe mass at each of the ``n + 1``
        #: vertices.  Every entry must be strictly positive for the same reason
        #: the model requires it: a massless vertex has no force balance to
        #: propagate the recursion through.
        total = hose.copy()
        for k, vertex in enumerate(self.attachments):
            total[vertex] += quad_masses[k]
        if not np.all(total > 0.0):
            raise ValueError(
                "every vertex needs strictly positive total mass (hose + airframe); got "
                f"{total.tolist()} for attachments {self.attachments}"
            )
        self.vertex_total_masses = total

        if inertias is None:
            self.inertias = np.tile(np.asarray(DEFAULT_QUAD_INERTIA, dtype=float), (nQ, 1, 1))
        else:
            arr = np.asarray(inertias, dtype=float)
            if arr.shape == (3, 3):
                arr = np.tile(arr, (nQ, 1, 1))
            if arr.shape != (nQ, 3, 3):
                raise ValueError(f"inertias must be (3, 3) or ({nQ}, 3, 3); got {arr.shape}")
            self.inertias = arr

        self.gravity = float(gravity)

        #: Floor, in newtons, under the tension of every link whose force the
        #: recursion **derives**.  The only thing standing between a
        #: multi-vehicle hose and a silently folded link — see
        #: :ref:`m6-guards` for exactly what it does and does not catch.  The
        #: default is M3/M4's slack floor, which detects the crossing itself
        #: and little more; a planner should raise it to a fraction of the
        #: tension it means to hold.
        self.min_tension = float(min_tension)
        if self.min_tension < 0.0:
            raise ValueError(f"min_tension must be non-negative; got {self.min_tension}")

        #: Vehicle indices whose flat output includes a tension — every vehicle
        #: but the one at the last vertex, in vehicle order.  ``ft_ind`` in the
        #: reference.
        self.tension_vehicles = tuple(range(nQ - 1))

    @staticmethod
    def _broadcast(value, name: str, size: int) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(size, float(arr))
        if arr.shape != (size,):
            raise ValueError(f"{name} must be a scalar or have {size} entries; got {arr.shape}")
        return arr.astype(float)

    # ─── Derived parameters ────────────────────────────────────────

    @property
    def num_vertices(self) -> int:
        """``n + 1``."""
        return self.num_links + 1

    @property
    def jet_order(self) -> int:
        """Root-position jet order this map requires: ``2n + 4``.

        Grows with the link count for M5's reason — each rung of the recursion
        spends two derivatives — but the binding consumer is the mandatory
        vehicle at the last vertex, whose position emerges at ``K − 2n`` and
        whose moment needs four more.  See
        :func:`~udaan.flatness.chain.chain_jet_order`.
        """
        return chain_jet_order(self.num_links)

    @property
    def tension_jet_orders(self) -> tuple[int, ...]:
        """Order each flat tension jet must reach, in flat-tension order.

        ``𝛕_a`` enters the recursion at the rung carrying force order
        ``K − 2(a + 1)``, so a shallower jet would quietly cost derivative
        depth at every vertex below it.  This is
        :func:`~udaan.flatness.chain.cable_chain_cascade`'s own
        ``K − 2(n − i)`` rule read in hose indices.
        """
        required = self.jet_order
        return tuple(required - 2 * (self.attachments[k] + 1) for k in self.tension_vehicles)

    @property
    def hose_length(self) -> float:
        """Total taut hose length ``Σ_i ℓ_i``."""
        return float(np.sum(self.link_lengths))

    @property
    def total_mass(self) -> float:
        """Hose + every airframe, in kg."""
        return float(self.vertex_total_masses.sum())

    @classmethod
    def from_model(cls, model) -> MultiQuadHose:
        """Build from a
        :class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`-like model."""
        return cls(
            num_links=model.num_links,
            attachments=model.attachments,
            vertex_masses=np.asarray(model.vertex_masses, dtype=float).copy(),
            link_lengths=np.asarray(model.link_lengths, dtype=float).copy(),
            masses=np.asarray(model.masses, dtype=float).copy(),
            inertias=np.asarray(model.inertias, dtype=float).copy(),
        )

    # ─── Recovery ──────────────────────────────────────────────────

    def _check(self, flats: MultiQuadHoseFlats) -> None:
        n = self.num_links
        required = self.jet_order
        if flats.x_0.order < required:
            raise ValueError(
                f"MultiQuadHose: an {n}-link hose needs x_0 through its {required}-th "
                f"derivative (2n + 4); got order {flats.x_0.order}."
            )
        expected = len(self.tension_vehicles)
        if len(flats.tensions) != expected:
            raise ValueError(
                f"MultiQuadHose: expected {expected} flat tension jets (one per vehicle "
                f"except the one at vertex {n}); got {len(flats.tensions)}."
            )
        for k, (tension, order) in enumerate(
            zip(flats.tensions, self.tension_jet_orders, strict=True)
        ):
            if tension.order < order:
                raise ValueError(
                    f"MultiQuadHose: tensions[{k}] (vehicle at vertex "
                    f"{self.attachments[self.tension_vehicles[k]]}) must reach its "
                    f"{order}-th derivative; got order {tension.order}.  Pad a constant "
                    f"tension with zero rows."
                )
        if len(flats.psi) != self.num_quadrotors:
            raise ValueError(
                f"MultiQuadHose: expected {self.num_quadrotors} yaw jets; got {len(flats.psi)}."
            )

    def cascade(self, flats: MultiQuadHoseFlats) -> HoseCascade:
        """Run only the geometry/force sweep, returning a :class:`HoseCascade`.

        Exposed for the same reason M5 exposes its chain: the recursion is
        worth inspecting and testing on its own — link directions, tensions,
        every vertex position and every vehicle's thrust vector, at every
        derivative the sweep produced, before the attitude machinery collapses
        them to a state and a wrench.

        Raises:
            ValueError: if a flat jet is shallower than the depth accounting
                requires, if some link goes **slack** (``\\|𝛕_i\\| ≈ 0``, its
                direction undefined), if a *derived* link's tension falls below
                :attr:`min_tension`, or — at ``nQ = 1``, where the test is
                still on — if a link leaves the hanging branch.  See
                :func:`~udaan.flatness.chain.cable_chain_cascade` and
                :ref:`m6-guards`.
        """
        self._check(flats)
        n = self.num_links
        required = self.jet_order
        mbar = self.vertex_total_masses

        # Reverse into cascade indexing: cascade node c is hose vertex n - c,
        # so cascade rung k is hose link n - 1 - k and carries the mass of hose
        # vertex n - 1 - k.  See "Which sweep, and why it runs backwards".
        masses = mbar[n - 1 :: -1]
        lengths = self.link_lengths[::-1]

        # A flat tension 𝛕_a is the prescribed force at cascade rung n - 1 - a,
        # negated: the cascade's F points up the chain where 𝛕 points down.
        prescribed: list[list[np.ndarray] | None] = [None] * n
        for k, jet in zip(self.tension_vehicles, flats.tensions, strict=True):
            vertex = self.attachments[k]
            prescribed[n - 1 - vertex] = [-np.asarray(jet[d], dtype=float) for d in range(len(jet))]

        try:
            chain = cable_chain_cascade(
                [np.asarray(flats.x_0[d], dtype=float) for d in range(required + 1)],
                masses,
                lengths,
                gravity=self.gravity,
                link_forces=prescribed,
                # A hose held up at both ends has links pointing upwards on
                # purpose, so the branch test has to go — but only then.  With
                # one vehicle there are no prescribed rungs, the hose hangs off
                # its last vertex exactly as M5's cable does, and keeping the
                # test on costs nothing and catches every fold.  See
                # :ref:`m6-guards`.
                require_hanging_branch=(len(self.tension_vehicles) == 0),
                # What partly replaces it at nQ ≥ 2.
                min_tension=self.min_tension,
            )
        except ValueError as exc:  # cascade indices are reversed — say so
            raise ValueError(
                f"{exc}  (This is the reversed sweep: cascade link k is hose link {n} - 1 - k.)"
            ) from exc

        # Un-reverse.  Directions and forces flip sign with the index order;
        # tensions are norms and do not.
        vertex_positions = [chain.node_positions[n - j] for j in range(n + 1)]
        directions = [[-row for row in chain.directions[n - 1 - i]] for i in range(n)]
        forces = [[-row for row in chain.forces[n - 1 - i]] for i in range(n)]
        tensions = [chain.tensions[n - 1 - i] for i in range(n)]

        # Thrusts.  Every vehicle but the last reads its own rung's recovered
        # applied force; the last vertex is the cascade's node 0, whose thrust
        # is `top_force`'s documented relation u = m(ẍ + g e₃) − F_0.
        thrusts: list[list[np.ndarray] | None] = [None] * (n + 1)
        for k in self.tension_vehicles:
            vertex = self.attachments[k]
            thrusts[vertex] = chain.applied_forces[n - 1 - vertex]
        x_last = vertex_positions[n]
        top = chain.top_force
        depth = min(len(x_last) - 2, len(top))
        last = [mbar[n] * (x_last[2] + self.gravity * _E3) - top[0]]
        last += [mbar[n] * x_last[d + 2] - top[d] for d in range(1, depth)]
        thrusts[n] = last

        return HoseCascade(
            vertex_positions=vertex_positions,
            directions=directions,
            tensions=tensions,
            forces=forces,
            vertex_thrusts=thrusts,
        )

    def recover(
        self, flats: MultiQuadHoseFlats
    ) -> tuple[MultiQuadHoseRefState, MultiQuadHoseInputs]:
        n = self.num_links
        nQ = self.num_quadrotors

        # ── Step 1: the sweep, root → last vertex ──────────────────────
        chain = self.cascade(flats)

        # ── Step 2: per-vehicle attitude from its thrust-vector jet ────
        thrusts = np.zeros(nQ)
        moments = np.zeros((nQ, 3))
        orientations: list[SO3] = []
        rates: list[TSO3] = []
        rate_derivs: list[Vec3] = []
        for k, vertex in enumerate(self.attachments):
            u = chain.vertex_thrusts[vertex]
            assert u is not None  # every attachment gets one, by construction
            if float(np.linalg.norm(u[0])) < 1e-6:
                raise ValueError(
                    f"MultiQuadHose.recover: vehicle {k} (vertex {vertex}) has a vanishing "
                    f"thrust vector (‖f R e₃‖ ≈ {float(np.linalg.norm(u[0])):.3e} N), so its "
                    f"attitude is undefined — it is in free fall.  Shift load onto it with "
                    f"the flat tensions."
                )
            psi = flats.psi[k]
            f, R, Omega, dOmega, moment = _attitude_from_thrust_vector(
                u[0],
                u[1],
                u[2],
                float(psi.value),
                float(psi.velocity),
                float(psi.acceleration),
                self.inertias[k],
            )
            thrusts[k] = f
            moments[k] = moment
            orientations.append(SO3(R))
            rates.append(TSO3(Omega))
            rate_derivs.append(dOmega)

        # ── Step 3: package ────────────────────────────────────────────
        vertex_positions = np.stack([chain.vertex_positions[j][0] for j in range(n + 1)])
        vertex_velocities = np.stack([chain.vertex_positions[j][1] for j in range(n + 1)])
        ref = MultiQuadHoseRefState(
            root_position=np.asarray(flats.x_0[0], dtype=float),
            root_velocity=np.asarray(flats.x_0[1], dtype=float),
            root_acceleration=np.asarray(flats.x_0[2], dtype=float),
            link_attitudes=[chain.directions[i][0] for i in range(n)],
            link_angular_velocities=[chain.angular_velocity(i) for i in range(n)],
            link_angular_accelerations=[chain.angular_acceleration(i) for i in range(n)],
            vertex_positions=vertex_positions,
            vertex_velocities=vertex_velocities,
            positions=[vertex_positions[v] for v in self.attachments],
            velocities=[vertex_velocities[v] for v in self.attachments],
            orientations=orientations,
            angular_velocities=rates,
            angular_accelerations=rate_derivs,
        )
        inputs = MultiQuadHoseInputs(
            thrusts=thrusts,
            moments=moments,
            tensions=np.array([chain.tensions[i][0] for i in range(n)]),
        )
        return ref, inputs
