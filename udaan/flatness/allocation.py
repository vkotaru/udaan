r"""Wrench → per-cable tension allocation for a cable-suspended **rigid-body**
payload (primitive D).

A point-mass payload constrains only the *sum* of the cable forces, so M3
resolves the shortfall with a :class:`~udaan.flatness.TensionSplitPolicy`.  A
rigid body also constrains where each force is applied, so the balance is a
**wrench** balance,

.. math::
    :label: alloc-balance

    \Phi\, \tau = W,
    \qquad
    \Phi = \begin{bmatrix} I & \cdots & I \\
                           \hat r_1 & \cdots & \hat r_{n_Q}\end{bmatrix}
         \in \mathbb{R}^{6 \times 3 n_Q},
    \qquad
    W = \begin{bmatrix} W_f \\ W_m \end{bmatrix},

with ``τ_i`` the body-frame force cable ``i`` applies **to the load** at the
body-frame attachment point ``r_i``.  ``Φ`` is
:func:`udaan.utils.grasp.grasp_map`; it is built from the ``r_i`` alone, which
are *constants* of the mechanism, so ``Φ``, ``Φ⁺`` and any basis of ``ker Φ``
are constant matrices — a fact this module leans on heavily when it
differentiates :eq:`alloc-balance`.

Six equations, ``3 n_Q`` unknowns: for ``n_Q ≥ 3`` non-collinear attachments
``rank Φ = 6`` and the solution set is a ``3 n_Q − 6`` dimensional affine space.
Picking a point in it is a design choice — exactly the role M3's tension-split
policy plays — and it is made here by a :class:`TensionDistributionPolicy`.

The chosen parameterisation
---------------------------
Write ``τ̄`` for the policy's **bias forces** — a full stack of ``n_Q``
body-frame forces, in newtons, that the policy would like the cables to carry —
and ``W̄ := Φ τ̄`` for the wrench they produce.  The allocation is

.. math::
    :label: alloc-split

    \tau(W) \;=\; \Phi^{+} W \;+\; \bigl(I - \Phi^{+}\Phi\bigr)\, \bar\tau
    \;=\; \bar\tau \;+\; \Phi^{+}\,(W - \bar W),

the second equality holding exactly for *any* ``Φ`` (substitute ``W̄ = Φ τ̄`` and
use ``Φ⁺ Φ τ̄`` twice).  Three consequences, all load-bearing:

* **Exact at the bias point.**  ``τ(W̄) = τ̄`` identically, not approximately.
  With the ring policy that makes the flat hover reference *equal* to the
  simulation model's ``reset()`` trim.
* **Affine in the wrench**, so tautness has the same closed-form certificate
  M3's affine split has — :meth:`WrenchAllocation.wrench_margin`.
* **Differentiating is free.**  ``Φ⁺`` and the null-space term are constant, so

  .. math::
      :label: alloc-derivs

      \tau^{(k)} = \Phi^{+} W^{(k)}, \qquad k \ge 1 .

  This is the step the references take silently; here it is *true by
  construction*, because the null-space coordinate is chosen constant.  A
  time-varying policy would have to supply its own derivative tuple, and
  :meth:`WrenchAllocation.allocate_jet` re-checks :eq:`alloc-balance` at every
  order either way.

Why not the published ``LAMBDA``
--------------------------------
``phd_research/differential_flatness/matlab/@Flat2State/multiQuadRigidLoad.m``
writes :eq:`alloc-split` as ``Φ⁺ W + N·LAMBDA`` with a hard-coded

.. code-block:: matlab

    LAMBDA = [(1/4)*mL*g*vec_dot(RotY(pi/6)*e3, e1);
              (1/4)*mL*g*vec_dot(RotY(pi/6)*e3, e2);
              (1/4)*mL*g*vec_dot(RotX(pi/6)*e3, e1);
              (1/4)*mL*g*RotY(-pi/6)*e3];

— three scalars stacked on a 3-vector, whose meaning depends entirely on which
columns the (nowhere-constructed, see below) ``N`` happens to have.  It is four
numbers that give non-negative tensions in hover for one particular ``r``, and
it carries no statement about any other geometry or any other ``n_Q``.
Parameterising the null space by ``τ̄`` instead keeps every coordinate in
newtons and every one of them physically meaningful: it *is* the cable-force
stack the policy wants, projected onto what the grasp cannot see.  The
equivalent ``LAMBDA`` is recovered as :attr:`WrenchAllocation.null_coordinates`
(``N^\top \bar\tau`` in the module's own orthonormal basis) if a reader wants
the number.

What the references get wrong
-----------------------------
* ``multiQuadRigidLoad.m`` (lines 20–114) uses ``params.Phi_pinv`` and
  ``params.N`` but **never constructs them** — the block that would (lines
  52–69) is commented out, so the function raises on any input.
* Where ``N`` *is* built
  (``quadrotor-flexible-cables-payload/Multiple-Quad-Rigid-Load/
  main_multiquad_rigid_flexible.m``, lines 27–52) it is hard-coded for exactly
  ``n_Q = 4`` from the six attachment-pair directions
  ``u_{ij} = (r_j - r_i)/\|r_j - r_i\|``, under the author's own
  ``(FIX THIS - write code to automate ...)`` comment.  Those columns *are* in
  ``ker Φ`` — ``Φ[u_{ij}; -u_{ij}] = [0;\, (r_i - r_j) \times u_{ij}] = 0`` —
  but there is no argument that six of them are independent, and the recipe
  produces ``\binom{n_Q}{2}`` columns for a kernel of dimension ``3n_Q - 6``, so
  it is neither a basis nor even the right *count* except at ``n_Q = 4``.
  :func:`null_space_basis` replaces it with an SVD basis that is correct for
  every ``n_Q`` and every geometry.
* ``multiQuadRigidLoad.m`` also **mixes frames**: its ``W`` stacks a *world*
  frame force ``m_L(\ddot x_L + g e_3)`` on a *body* frame moment
  ``J_L \dot\Omega_L + \Omega_L \times J_L \Omega_L``, then feeds the resulting
  ``τ_i`` straight into a world-frame Newton equation without rotating it.  The
  two agree only at ``R_L = I``.  The C++ sibling
  (``qrotor_ros/.../floating_rigidbody_control.hpp``) gets this right
  (``R.transpose() * force`` in, ``R * τ_i`` out); this module follows the C++.
* That same C++ builds ``Φ`` and ``Φ⁺`` generically at runtime — the good part —
  but is **minimum-norm only**: no null-space term, no bias, and no tautness
  guard, so nothing stops a ``τ_i`` from pointing the wrong way down a cable
  that can only pull.  :meth:`WrenchAllocation.allocate` refuses instead.

Instantaneous authority
-----------------------
``Φ`` maps arbitrary force *vectors*.  What a taut cable can actually apply at a
frozen instant is a scalar tension along its current ``q_i``, i.e. the ``(6,
n_Q)`` map :func:`udaan.utils.grasp.hover_wrench_map`, whose rank at the default
ring is **3 for every** ``n_Q ≥ 3``.  Nothing here claims otherwise.  The
distinction is that a *flatness* map is not choosing a wrench at a frozen
instant: it plans the whole trajectory, and ``q_i := 𝐓_i/\|𝐓_i\|`` is *derived*
from the allocated force, so the reference it returns is one in which the cables
have already reoriented.  The lag that the rank-3 caveat describes is a
closed-loop cost paid by a feedback allocator, not a claim this module makes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.defaults import (
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_RIGID_PAYLOAD_MASS,
    GRAVITY,
)
from ..utils.grasp import (
    formation_cable_attitudes,
    grasp_map,
    hover_wrench_map,
    ring_attachment_points,
)
from ..utils.logging import get_logger

# The same physical tautness floor M3 uses; it is defined there because that is
# where it was first needed, and there is no reason for a second constant.
from .multi_quad_cspayload import DEFAULT_MIN_TENSION

__all__ = [
    "FixedTensionDistribution",
    "RingTensionDistribution",
    "TensionDistributionPolicy",
    "WrenchAllocation",
    "WrenchAllocationError",
    "null_space_basis",
]

_logger = get_logger(__name__)

_E3 = np.array([0.0, 0.0, 1.0])

#: A grasp needs six independent wrench directions to hold an SE(3) pose.
_FULL_WRENCH_RANK = 6

#: Relative tolerance on ``‖Φ τ − W‖`` before the commanded wrench counts as
#: outside the range of the grasp.
_RANGE_TOL = 1e-9

#: **The** singular-value cutoff of this module, relative to ``σ_max``.  One
#: constant, used in exactly two places that must not be allowed to disagree:
#: :func:`null_space_basis` (which fixes :attr:`WrenchAllocation.rank`,
#: :attr:`~WrenchAllocation.is_fully_actuated` and the
#: :class:`WrenchAllocationError` text) and the ``rcond`` handed to
#: :func:`numpy.linalg.pinv` (which fixes the rank ``Φ⁺`` *actually* inverts).
#: NumPy's own default ``rcond = 1e-15 · max(m, n)`` is two decades tighter, and
#: with the two rules in force at once a grasp with
#: ``σ_min/σ_max ∈ [1.2e-14, 1e-12]`` was reported rank-deficient while ``Φ⁺``
#: happily inverted the direction the report says is missing.
_RANK_RTOL = 1e-12

#: ``cond(Φ) = σ_max/σ_min`` above which a full-rank grasp is flagged as
#: near-degenerate.  Every sane geometry is orders below it — the default ring
#: is 2.8 at any ``n_Q``, a 1 cm ring is 85, and 6000 random ``n_Q ∈ {3,4,5}``
#: geometries peak at 167 — so this fires only on a grasp that is numerically,
#: not physically, full rank.
_CONDITION_WARN = 1e6


class WrenchAllocationError(ValueError):
    """The commanded wrench cannot be produced by this grasp.

    Raised when ``W`` (at any derivative order) lies outside ``range Φ``, so the
    least-squares ``Φ⁺ W`` silently drops a component and the recovered
    reference would *not* solve the payload's equations of motion.  Like
    :class:`~udaan.flatness.TensionSplitContractError` this is a hard failure
    rather than an aggressive-trajectory refusal: no closed-loop fallback makes
    an unachievable wrench achievable, and the caller has to change the
    trajectory or the geometry.

    The common cause is structural: at ``n_Q ≤ 2`` (or with collinear
    attachments) ``rank Φ < 6``, so the payload's attitude is not steerable and
    only wrenches whose moment lies in the range of the grasp can be commanded.
    """


def null_space_basis(matrix, rtol: float = _RANK_RTOL) -> np.ndarray:
    r"""Orthonormal basis for ``ker(matrix)``, as columns.

    Computed from the singular-value decomposition ``A = U Σ V^\top``: the right
    singular vectors whose singular values vanish span the kernel exactly, and
    they come out orthonormal.  This is the standard construction
    :footcite:`golub2013matrix` and it is what makes the result correct for
    **any** ``n_Q`` and **any** attachment geometry, including the degenerate
    ones — no pattern of hand-picked difference vectors is involved, and the
    dimension is discovered (``n - rank``) rather than assumed.

    Args:
        matrix: ``(m, n)`` array.
        rtol: singular values below ``rtol · σ_max`` count as zero; defaults to
            :data:`_RANK_RTOL`, the one cutoff this module uses everywhere
            (:class:`WrenchAllocation` hands the same number to
            :func:`numpy.linalg.pinv` as ``rcond``, so the rank it *reports* and
            the rank ``Φ⁺`` *inverts* cannot disagree).  It is deliberately
            tighter than :func:`numpy.linalg.matrix_rank` (which scales with the
            matrix size): a grasp that is only *nearly* rank-deficient should be
            reported as full rank and its ill conditioning left visible — as a
            ``cond(Φ)`` warning — not silently absorbed into the kernel.

    Returns:
        ``(n, n - rank)`` array whose columns are an orthonormal basis of the
        kernel; a ``(n, 0)`` array when the kernel is trivial.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"null_space_basis expects a 2-D array; got shape {arr.shape}")
    _, singular, vh = np.linalg.svd(arr, full_matrices=True)
    if singular.size == 0:
        return np.eye(arr.shape[1])
    rank = int(np.sum(singular > rtol * float(singular[0])))
    return np.asarray(vh[rank:].T)


# ─── Tension-distribution policies ────────────────────────────────────


class TensionDistributionPolicy(ABC):
    """Strategy that picks a point in the null space of the grasp map.

    The wrench balance :eq:`alloc-balance` leaves ``3 n_Q − rank Φ`` dimensions
    of the cable-force stack undetermined.  A policy resolves them by naming the
    force stack ``τ̄`` it *would* like the cables to carry; the allocator keeps
    the part of that wish the grasp cannot see, ``(I − Φ⁺Φ) τ̄``, and lets the
    commanded wrench fix the rest (:eq:`alloc-split`).

    Everything about the policy is constant in time, which is what makes
    :eq:`alloc-derivs` exact rather than an assumption.  A time-varying policy
    is a legitimate extension — the null-space coordinate is a genuine flat
    output, free to move — but it would have to supply its own derivative
    tuple, and no shipped policy does.

    Only ``num_quadrotors`` and ``bias_forces`` are required: the reference
    wrench, the hover cable directions and the tautness margin are all derived
    from ``τ̄`` and the geometry by :class:`WrenchAllocation`, so a policy cannot
    state one of them inconsistently with the others.
    """

    @property
    @abstractmethod
    def num_quadrotors(self) -> int:
        """Number of cables this policy distributes across."""

    @property
    @abstractmethod
    def bias_forces(self) -> np.ndarray:
        """``(nQ, 3)`` body-frame forces ``τ̄_i`` applied *to the load*.

        Newtons throughout — unlike the published ``LAMBDA``, every component
        of this parameterisation has the same physical unit.  Each ``τ̄_i`` must
        be non-zero: it fixes cable ``i``'s hover direction
        (``q̄_i = -τ̄_i/‖τ̄_i‖``, the force pulls the *quadrotor* the other way)
        and the sign test that keeps the recovery on the taut side of slack.
        """


class FixedTensionDistribution(TensionDistributionPolicy):
    """A policy given by an explicit constant bias-force stack.

    The general member of the family — :class:`RingTensionDistribution` is this
    with the bias solved from a hover trim.  Useful for reproducing a published
    split, or for biasing a formation towards a geometry the ring does not
    describe.

    The bias is *not* required to be a feasible hover trim.  If it is not — if
    ``W̄ = Φ τ̄`` is not the wrench that holds the payload at rest — the flat
    reference at rest is simply not this bias, and part of the tautness margin
    of :meth:`WrenchAllocation.wrench_margin` is already spent standing still,
    exactly as M3's :class:`~udaan.flatness.AffineTensionSplit` documents for
    anchors that do not sum to ``-m_L g e₃``.

    Args:
        bias_forces: ``(nQ, 3)`` body-frame forces on the load.
    """

    def __init__(self, bias_forces):
        arr = np.asarray(bias_forces, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"bias_forces must have shape (nQ, 3); got {arr.shape}")
        if arr.shape[0] < 1:
            raise ValueError("bias_forces must hold at least one cable force")
        if not np.all(np.isfinite(arr)):
            raise ValueError("bias_forces must be finite")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(
                "every bias force must be non-zero — a zero bias leaves cable "
                "direction and slack sign undefined at the reference condition"
            )
        self._bias = arr

    @property
    def num_quadrotors(self) -> int:
        return int(self._bias.shape[0])

    @property
    def bias_forces(self) -> np.ndarray:
        return self._bias


class RingTensionDistribution(FixedTensionDistribution):
    r"""Default policy — the bias is the model's own ring hover trim.

    The quadrotors sit on a horizontal circle of ``formation_radius`` centred
    **above their own attachment points** (:func:`formation_cable_attitudes`
    paired with :func:`ring_attachment_points`), which is the formation
    :meth:`udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.reset`
    places them in.  The scalar tensions of that formation solve the static trim

    .. math::

        G\, t = \begin{bmatrix} -m_L g e_3 \\ 0 \end{bmatrix},
        \qquad
        G = \bigl[\,q_i \;;\; r_i \times q_i\,\bigr]_{i=1..n_Q}

    (:func:`udaan.utils.grasp.hover_wrench_map`) in least squares, and the bias
    is the corresponding force *on the load*, ``τ̄_i = -t_i q̄_i``.

    Two things follow, and they are the reason this is the default rather than
    a min-norm solve:

    * ``Φ τ̄ = [m_L g e_3;\, 0] = W̄`` is exactly the hover wrench, so
      :eq:`alloc-split` returns ``τ̄`` *identically* at hover and the flat
      reference at rest **is** the model's ``reset()`` configuration — cable
      directions, quadrotor positions and thrusts all agree to round-off.
    * The formation is spread.  Minimising ``Σ_i ‖τ_i‖²`` subject to
      :eq:`alloc-balance` alone (i.e. ``τ̄ = 0``) makes every cable force
      near-parallel and collapses the team onto one point above the payload —
      the same degeneracy M3's :class:`~udaan.flatness.RingTensionSplit`
      documents.  Geometry has to enter the objective somewhere, and here the
      ring radius is that knob.

    A non-zero :attr:`hover_residual` says one narrow thing: the *ring*
    formation's cable directions ``q̄_i`` cannot produce the hover wrench with
    scalar tensions alone, so the bias solved from that trim is not the hover
    bias and ``W̄ = Φ τ̄ ≠ [m_L g e_3;\, 0]``.  It is **not** a statement about
    ``Φ``: whenever ``rank Φ = 6`` the grasp produces the hover wrench trivially,
    and the flat reference at rest is still an exact equilibrium of the model.
    What is lost is only the second bullet's coincidence — the reference no
    longer matches the model's ``reset()`` formation.  It is reported, not
    silently absorbed.

    Args:
        num_quadrotors: ``nQ ≥ 1``.
        payload_mass: ``m_L``.
        cable_lengths: scalar (broadcast) or ``(nQ,)`` cable lengths.
        attachment_points: ``(nQ, 3)`` body-frame attachment points; defaults to
            :func:`~udaan.utils.grasp.ring_attachment_points`.
        formation_radius: radius of the hover ring; must be strictly below every
            cable length for ``nQ ≥ 2``.
        gravity: ``g``.
    """

    def __init__(
        self,
        num_quadrotors: int,
        payload_mass: float = DEFAULT_RIGID_PAYLOAD_MASS,
        cable_lengths=1.0,
        attachment_points=None,
        formation_radius: float = DEFAULT_FORMATION_RADIUS,
        gravity: float = GRAVITY,
    ):
        nQ = int(num_quadrotors)
        if nQ < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {nQ}")
        if payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {payload_mass}")

        lengths = np.asarray(cable_lengths, dtype=float)
        lengths = np.full(nQ, float(lengths)) if lengths.ndim == 0 else lengths.reshape(-1)
        if lengths.shape[0] != nQ:
            raise ValueError(f"cable_lengths must have {nQ} entries; got {lengths.shape[0]}")
        if np.any(lengths <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {lengths.tolist()}")

        points = (
            ring_attachment_points(nQ)
            if attachment_points is None
            else np.asarray(attachment_points, dtype=float)
        )
        if points.shape != (nQ, 3):
            raise ValueError(f"attachment_points must be ({nQ}, 3); got {points.shape}")

        hover_q = np.array(
            [np.asarray(q) for q in formation_cable_attitudes(nQ, lengths, formation_radius)],
            dtype=float,
        )
        target = np.concatenate([-float(payload_mass) * float(gravity) * _E3, np.zeros(3)])
        wrench_map = hover_wrench_map(hover_q, points)
        tensions, *_ = np.linalg.lstsq(wrench_map, target, rcond=None)
        residual = float(np.linalg.norm(wrench_map @ tensions - target))
        if residual > _RANGE_TOL * max(1.0, float(np.linalg.norm(target))):
            _logger.warning(
                "ring hover trim is infeasible for this geometry: least-squares "
                "residual ‖G t - W_hover‖ = %.4g N. The ring formation's cable "
                "directions cannot hold the payload weight with scalar tensions "
                "alone, so the bias is not a hover trim and the flat reference at "
                "rest no longer coincides with the model's reset() formation. The "
                "reference is still an exact equilibrium whenever rank(Phi) = 6 — "
                "only the coincidence with reset() is lost.",
                residual,
            )

        super().__init__(-tensions[:, None] * hover_q)
        self._hover_tensions = np.asarray(tensions, dtype=float)
        self._hover_residual = residual
        self._hover_directions = hover_q
        self.formation_radius = float(formation_radius)

    @property
    def hover_tensions(self) -> np.ndarray:
        """``(nQ,)`` scalar cable tensions of the trim the bias was solved from."""
        return self._hover_tensions

    @property
    def hover_residual(self) -> float:
        """Least-squares residual of the static trim; non-zero means no trim exists."""
        return self._hover_residual


# ─── The allocator ────────────────────────────────────────────────────


class WrenchAllocation:
    """Wrench → per-cable force allocation for a rigid-body payload.

    Owns the geometry (``Φ``, ``Φ⁺``, a basis of ``ker Φ``) and one
    :class:`TensionDistributionPolicy`, and turns a payload-wrench *jet* into
    per-cable body-frame force *jets* by :eq:`alloc-split` and
    :eq:`alloc-derivs`.

    Args:
        attachment_points: ``(nQ, 3)`` body-frame attachment points ``r_i``.
        policy: the null-space choice; defaults to
            :class:`RingTensionDistribution` built from the arguments below.
        payload_mass: ``m_L``, only used to build the default policy.
        cable_lengths: only used to build the default policy.
        formation_radius: only used to build the default policy.
        gravity: ``g``, only used to build the default policy.
        min_tension: cables below this magnitude are reported as slack.
    """

    def __init__(
        self,
        attachment_points,
        *,
        policy: TensionDistributionPolicy | None = None,
        payload_mass: float = DEFAULT_RIGID_PAYLOAD_MASS,
        cable_lengths=1.0,
        formation_radius: float = DEFAULT_FORMATION_RADIUS,
        gravity: float = GRAVITY,
        min_tension: float = DEFAULT_MIN_TENSION,
    ):
        points = np.asarray(attachment_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"attachment_points must be (nQ, 3); got {points.shape}")
        self._attachment_points: np.ndarray = points.copy()
        nQ = int(points.shape[0])
        self._num_quadrotors = nQ
        self.min_tension = float(min_tension)

        self._grasp = grasp_map(self._attachment_points)
        # One cutoff for all three: the kernel basis, the reported rank, and the
        # rank Φ⁺ actually inverts.  numpy's pinv default would use its own,
        # two decades looser, and the two can then disagree — see _RANK_RTOL.
        self._grasp_pinv = np.linalg.pinv(self._grasp, rcond=_RANK_RTOL)
        self._null_basis = null_space_basis(self._grasp)  # same _RANK_RTOL, by default
        self._rank = 3 * nQ - int(self._null_basis.shape[1])
        self._condition_number = self._grasp_condition_number()

        if policy is None:
            policy = RingTensionDistribution(
                num_quadrotors=nQ,
                payload_mass=payload_mass,
                cable_lengths=cable_lengths,
                attachment_points=self._attachment_points,
                formation_radius=formation_radius,
                gravity=gravity,
            )
        if policy.num_quadrotors != nQ:
            raise ValueError(
                f"policy is built for {policy.num_quadrotors} quadrotors, but this "
                f"allocation has {nQ} attachment points"
            )
        self.policy = policy

        bias = np.asarray(policy.bias_forces, dtype=float).reshape(-1)
        self._bias = bias
        self._null_term = bias - self._grasp_pinv @ (self._grasp @ bias)
        self._reference_wrench = np.asarray(self._grasp @ bias)

        if self._rank < _FULL_WRENCH_RANK:
            _logger.warning(
                "grasp map of %d attachment point(s) has rank %d < 6: %d wrench "
                "direction(s) cannot be produced at all. The payload attitude is "
                "not a flat output of this geometry — only wrenches inside range(Φ) "
                "can be allocated, and anything else raises WrenchAllocationError. "
                "Full SE(3) authority needs nQ >= 3 non-collinear attachments.",
                nQ,
                self._rank,
                _FULL_WRENCH_RANK - self._rank,
            )
        elif self._condition_number > _CONDITION_WARN:
            _logger.warning(
                "grasp map is numerically full rank but near-degenerate: "
                "cond(Φ) = σ_max/σ_min = %.3g. Φ⁺ amplifies a wrench command by up "
                "to 1/σ_min = %.3g N per unit wrench, so allocate() can return a "
                "physically absurd cable-force stack whose residual ‖Φτ - W‖ still "
                "passes the range check. Move the attachment points off "
                "(near-)collinear before trusting the recovered reference.",
                self._condition_number,
                self._condition_number / float(np.linalg.norm(self._grasp, ord=2)),
            )

    def _grasp_condition_number(self) -> float:
        r"""``σ_max/σ_min`` of ``Φ`` over *all* ``min(6, 3 n_Q)`` singular values.

        Not just the retained ones: a grasp whose smallest singular value has
        been discarded as noise is unbounded-ly ill-conditioned as a map onto
        ``\mathbb{R}^6``, which is exactly what the rank warning above already
        says.  ``inf`` for an exactly rank-deficient ``Φ``.
        """
        singular = np.linalg.svd(self._grasp, compute_uv=False)
        if singular.size == 0 or float(singular[-1]) <= 0.0:
            return float("inf")
        return float(singular[0] / singular[-1])

    # ─── Geometry ──────────────────────────────────────────────────

    @classmethod
    def from_model(cls, model, policy: TensionDistributionPolicy | None = None):
        """Build from a
        :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase`-like
        model, so the allocation and the plant share one grasp geometry.

        With the default ring policy and *ring* attachment points the hover trim
        is exact, and the allocation then reproduces the model's ``reset()``
        formation at hover.  Off the ring the trim is only least-squares
        (:attr:`RingTensionDistribution.hover_residual` is non-zero and says so),
        and the flat reference at hover is a different — still exact —
        equilibrium.
        """
        return cls(
            model.attachment_points,
            policy=policy,
            payload_mass=model.payload_mass,
            cable_lengths=model.cable_lengths,
            formation_radius=model.formation_radius,
        )

    @property
    def num_quadrotors(self) -> int:
        return self._num_quadrotors

    @property
    def attachment_points(self) -> np.ndarray:
        """``(nQ, 3)`` body-frame attachment points ``r_i``."""
        return self._attachment_points

    @property
    def grasp(self) -> np.ndarray:
        """The ``(6, 3 nQ)`` grasp matrix ``Φ``."""
        return self._grasp

    @property
    def grasp_pinv(self) -> np.ndarray:
        """``Φ⁺``, the Moore–Penrose pseudo-inverse."""
        return self._grasp_pinv

    @property
    def null_basis(self) -> np.ndarray:
        """``(3 nQ, 3 nQ − rank Φ)`` orthonormal basis of ``ker Φ``.

        Generated by :func:`null_space_basis`, so it is a genuine basis at every
        ``n_Q`` and every geometry — see the module docstring for what the
        published ``n_Q = 4`` hand-built ``N`` gets wrong.
        """
        return self._null_basis

    @property
    def rank(self) -> int:
        """``rank Φ`` — how many of the six wrench directions the grasp spans."""
        return self._rank

    @property
    def is_fully_actuated(self) -> bool:
        """``True`` when ``rank Φ = 6`` and the whole payload wrench is available."""
        return self._rank >= _FULL_WRENCH_RANK

    @property
    def condition_number(self) -> float:
        """``cond(Φ) = σ_max/σ_min`` — how hard ``Φ⁺`` amplifies a wrench.

        :attr:`rank` is a yes/no answer at the cutoff :data:`_RANK_RTOL`; this
        is the continuous quantity behind it, and it is the number to look at
        when a geometry is *nearly* degenerate.  A full-rank grasp above
        :data:`_CONDITION_WARN` is flagged at construction, because
        :meth:`allocate` will otherwise return an enormous but formally
        in-range cable-force stack.
        """
        return self._condition_number

    @property
    def null_coordinates(self) -> np.ndarray:
        """``λ = N^⊤ τ̄`` — the policy's null-space coordinates in this basis.

        The homogeneous-units replacement for the published ``LAMBDA``: every
        entry is a force component (N) in the orthonormal basis
        :attr:`null_basis`, and ``N λ`` is exactly the null-space term of
        :eq:`alloc-split`.
        """
        return np.asarray(self._null_basis.T @ self._bias)

    @property
    def bias_forces(self) -> np.ndarray:
        """``(nQ, 3)`` policy bias ``τ̄_i``, body-frame forces on the load."""
        return self._bias.reshape(self._num_quadrotors, 3)

    @property
    def reference_wrench(self) -> np.ndarray:
        """``W̄ = Φ τ̄`` — the wrench at which the allocation returns ``τ̄`` exactly."""
        return self._reference_wrench

    @property
    def hover_directions(self) -> np.ndarray:
        """``(nQ, 3)`` unit cable directions ``q̄_i`` implied by the bias.

        The force on the *quadrotor* is minus the force on the load, so
        ``q̄_i = -τ̄_i/‖τ̄_i‖`` — pointing from quadrotor to payload, this
        package's convention everywhere.
        """
        bias = self.bias_forces
        return np.asarray(-bias / np.linalg.norm(bias, axis=1, keepdims=True))

    # ─── Tautness certificate ──────────────────────────────────────

    def wrench_margin(self, min_tension: float | None = None) -> float:
        r"""Largest ``‖W − W̄‖`` for which every cable is provably taut.

        Because :eq:`alloc-split` is affine, ``τ_i - \bar\tau_i = \Phi^{+}_i
        (W - \bar W)`` exactly, where ``Φ⁺_i`` is the ``(3, 6)`` block of rows
        ``3i \dots 3i+2``.  Hence

        .. math::

            \|\tau_i\| \;\ge\; \|\bar\tau_i\| - \sigma_{\max}(\Phi^{+}_i)\,
                               \|W - \bar W\|,

        so every cable holds ``‖τ_i‖ > T_min`` as long as

        .. math::

            \|W - \bar W\| \;<\;
            \min_i \frac{\|\bar\tau_i\| - T_{\min}}{\sigma_{\max}(\Phi^{+}_i)}
            \;=:\; \mathrm{margin}.

        Inside that ball the sign test of :meth:`check_taut` cannot fire either,
        by the same inequality.  The bound is *sufficient*, not tight — one
        triangle inequality and one operator norm — so :meth:`allocate` still
        checks the true ``τ_i`` at every sample.  ``inf`` is returned when no
        cable's force can move at all (``Φ⁺`` block identically zero), which
        happens only for a degenerate grasp.
        """
        floor = self.min_tension if min_tension is None else float(min_tension)
        bias = self.bias_forces
        bound = np.inf
        for i in range(self._num_quadrotors):
            block = self._grasp_pinv[3 * i : 3 * i + 3, :]
            gain = float(np.linalg.norm(block, ord=2))
            if gain <= 0.0:
                continue  # this cable's force never moves with the wrench
            bound = min(bound, (float(np.linalg.norm(bias[i])) - floor) / gain)
        return float(max(bound, 0.0)) if np.isfinite(bound) else float("inf")

    def check_taut(self, tau: np.ndarray) -> None:
        """Raise unless every cable force is taut and on the hover side of slack.

        Two tests, for the reason M3 documents at length: ``‖τ_i‖`` is a norm,
        positive on *both* sides of the slack point, so a magnitude floor
        refuses only to *land* on the boundary and never to cross it.  Requiring
        ``τ_i · τ̄_i > 0`` in addition makes the refusal two-sided.  It is
        conservative — it also rejects a cable that has swung more than 90° from
        its reference direction without ever going slack.

        Args:
            tau: ``(nQ, 3)`` allocated body-frame cable forces.

        Raises:
            ValueError: naming the offending cable and the certified margin.
        """
        bias = self.bias_forces
        for i in range(self._num_quadrotors):
            tension = float(np.linalg.norm(tau[i]))
            if tension < self.min_tension:
                raise ValueError(
                    f"WrenchAllocation: cable {i} would go slack "
                    f"(T = {tension:.3e} N < min_tension = {self.min_tension:.3e} N). "
                    "The requested payload wrench is too aggressive for this "
                    "tension-distribution policy: keep ‖W - W̄‖ below "
                    f"{self.wrench_margin():.3f} N, enlarge the formation radius, "
                    "or supply a different policy."
                )
            if float(np.dot(tau[i], bias[i])) <= 0.0:
                raise ValueError(
                    f"WrenchAllocation: cable {i} has crossed the slack boundary — "
                    "the allocated force points away from its reference force "
                    f"τ̄_{i} = {bias[i].tolist()}, so quadrotor {i} would sit on the "
                    "far side of the payload. Keep ‖W - W̄‖ below "
                    f"{self.wrench_margin():.3f} N, enlarge the formation radius, "
                    "or supply a different policy."
                )

    # ─── Allocation ────────────────────────────────────────────────

    def residual(self, tau: np.ndarray, wrench: np.ndarray) -> float:
        """``‖Φ τ − W‖`` — how much of the commanded wrench the grasp dropped."""
        return float(np.linalg.norm(self._grasp @ np.asarray(tau).reshape(-1) - wrench))

    def allocate(self, wrench, *, check_taut: bool = True) -> np.ndarray:
        """Allocate one payload wrench into ``(nQ, 3)`` body-frame cable forces.

        Implements :eq:`alloc-split`.  The result is verified against
        :eq:`alloc-balance` before it is returned: ``Φ⁺`` answers *every*
        command, achievable or not, and an unachievable one would otherwise
        propagate into a reference that does not solve the equations of motion.

        Args:
            wrench: ``(6,)`` body-frame payload wrench ``[W_f; W_m]``.
            check_taut: also enforce :meth:`check_taut`.

        Raises:
            WrenchAllocationError: the wrench is outside ``range Φ``.
            ValueError: a cable would be slack or inverted (``check_taut``).
        """
        w = np.asarray(wrench, dtype=float).reshape(-1)
        if w.shape[0] != 6:
            raise ValueError(f"wrench must be a 6-vector [force; moment]; got {w.shape}")
        tau = self._grasp_pinv @ w + self._null_term
        self._check_range(tau, w, order=0)
        tau = tau.reshape(self._num_quadrotors, 3)
        if check_taut:
            self.check_taut(tau)
        return np.asarray(tau)

    def allocate_jet(self, wrench: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Allocate a wrench *jet* into per-cable body-frame force jets.

        Args:
            wrench: ``[W, Ẇ, …, W^(K)]``, each a ``(6,)`` body-frame wrench
                derivative.

        Returns:
            A length-``nQ`` list; entry ``i`` is ``[τ_i, τ̇_i, …, τ_i^(K)]``.

        The zeroth order carries the null-space term; every higher order is
        ``Φ⁺ W^(k)`` because ``Φ`` and the policy are constant
        (:eq:`alloc-derivs`).  That is exactly the step the MATLAB reference
        takes without stating it; here it is a theorem about the shipped
        policies rather than an assumption, and the balance is re-checked at
        every order regardless.
        """
        if not wrench:
            raise ValueError("wrench jet must hold at least the zeroth derivative")
        nQ = self._num_quadrotors
        stacks = [self.allocate(wrench[0]).reshape(-1)]
        for k in range(1, len(wrench)):
            w_k = np.asarray(wrench[k], dtype=float).reshape(-1)
            if w_k.shape[0] != 6:
                raise ValueError(f"wrench[{k}] must be a 6-vector; got {w_k.shape}")
            tau_k = self._grasp_pinv @ w_k
            self._check_range(tau_k, w_k, order=k)
            stacks.append(tau_k)
        return [[stack[3 * i : 3 * i + 3] for stack in stacks] for i in range(nQ)]

    def _check_range(self, tau: np.ndarray, wrench: np.ndarray, order: int) -> None:
        dropped = self.residual(tau, wrench)
        if dropped <= _RANGE_TOL * max(1.0, float(np.linalg.norm(wrench))):
            return
        raise WrenchAllocationError(
            f"payload wrench at derivative order {order} lies outside the range "
            f"of the grasp map (rank {self._rank} of 6): ‖Φτ - W‖ = {dropped:.4g} "
            f"would be dropped by the least-squares allocation, so the recovered "
            f"reference would not solve the payload's equations of motion. "
            f"W^({order}) = {np.asarray(wrench).tolist()}. With {self._num_quadrotors} "
            "cable(s) this geometry cannot produce every wrench; nQ >= 3 "
            "non-collinear attachments are needed for full SE(3) authority."
        )
