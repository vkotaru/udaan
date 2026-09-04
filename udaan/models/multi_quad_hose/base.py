r"""MultiQuadHoseBase — pure dynamics and state, no rendering.

Several rigid-body quadrotors joined in series by **one flexible hose**,
integrated with ZOH (zero-order hold) Euler.  This is M6, the last new system
in the package and the largest: M5's chain crossed with M3/M4's multi-vehicle
allocation.

.. warning::

   **This module is a derivation, not a port.**  The legacy reference
   ``phd_research/multiple-quadrotor-flexible-hose/matlab/models/
   @MultipleQuadrotorFlexibleHose`` was re-derived from the Lagrangian rather
   than transcribed.  Unlike the earlier steps of this port, the re-derivation
   found **nothing wrong with the equations of motion** — see "The reference"
   at the bottom for what it did find, all of it outside ``get_dyn_dq``.

The system
----------
One hose, not one hose per vehicle.  The hose is a chain of ``n`` rigid links
joined by ideal spherical joints at ``n + 1`` vertices; each vertex carries a
lumped point mass, and an **indicator over the vertices** says which of them a
quadrotor is bolted to.  A vehicle is therefore a property of a vertex, not a
segment of the chain, and ``n`` and ``nQ`` are independent.

.. math::

    Q = \mathbb{R}^3 \times (S^2)^{n} \times SO(3)^{n_Q}

Ordering and conventions
------------------------
* **Vertex 0 is the root and link ``i`` runs from vertex ``i`` to vertex
  ``i+1``**, so ``q_i`` points *down the chain*:

  .. math::

      x_{i+1} = x_i + \ell_i q_i, \qquad
      x_j = x_0 + \sum_{i<j} \ell_i q_i, \qquad
      \dot q_i = \omega_i \times q_i .

  That is the reference's own recursion (``computeVertices``) and the same
  proximal-to-distal sense :mod:`udaan.flatness.cable`,
  :mod:`udaan.models.quadrotor_cspayload` and
  :mod:`udaan.models.quadrotor_flexcable` use.
* **The root ``(x_0, v_0)`` is integrated and every other vertex is derived**
  from it and the link directions.  The reference integrates the same pair.
  M5 makes the opposite choice (it integrates the payload, the *distal* end,
  and derives the airframe) because it has exactly one airframe and it is at
  the proximal end; here neither end is distinguished, so the root is.
* Quadrotor ``k`` sits at vertex ``\mathrm{attachments}[k]``, and
  ``attachments`` is **strictly increasing** — see
  :func:`udaan.utils.hose.validate_attachments` for why that is enforced
  rather than sorted on input.

State size
----------
``6 + 6n + 12 n_Q`` stored floats: ``x_0, v_0`` (6), ``q_i, \omega_i`` for
``n`` links (``6n``), and ``R_k, \Omega_k`` for ``n_Q`` vehicles
(``12 n_Q``, the attitude as a 3x3 matrix).  This matches the reference's own
``nx = nvert*6 + nQ*12`` exactly.  In *manifold* dimensions it is
``6 + 4n + 6 n_Q``.

The layout differs from the reference's flat vector, which is
``[x_0; q; v_0; \dot q; R; \Omega]`` — configuration first, then all the
velocities, then all the attitudes, then all the body rates.  Here the state
is a dataclass of manifold-typed fields (M1-M5's convention), so there is no
flat vector to order; ``x_0`` and ``v_0`` are named fields and the per-link and
per-vehicle entries are lists indexed by link and by *vehicle* (not by vertex).

Derivation
----------
Write ``\bar m_j = m_j + m_{Q_j}`` for the total mass at vertex ``j`` (the
lumped hose mass plus the airframe's, zero where there is no airframe), and

.. math::
    :label: m6-coefficients

    S_i = \sum_{j > i} \bar m_j, \qquad
    M_{00} = \sum_{j} \bar m_j, \qquad
    M_{0i} = M_{i0} = S_i\,\ell_i, \qquad
    M_{ij} = S_{\max(i,j)}\,\ell_i \ell_j .

``S_i`` is the mass hanging strictly below link ``i``.  These are not
postulated: substituting the chain recursion into
``T = \tfrac12 \sum_j \bar m_j \|\dot x_j\|^2`` and collecting terms gives

.. math::

    T = \tfrac12 M_{00}\|\dot x_0\|^2
        + \sum_k M_{0k}\,\dot x_0 \cdot \dot q_k
        + \tfrac12 \sum_{k,j} M_{kj}\,\dot q_k \cdot \dot q_j
        + \tfrac12 \sum_k \Omega_k^\top J_k \Omega_k ,
    \qquad
    V = g\,e_3 \cdot \Bigl( M_{00} x_0 + \sum_k M_{0k} q_k \Bigr),

with :eq:`m6-coefficients` read straight off.  **This is M5's Lagrangian**, with
``M_{00}``'s ``m_Q`` term generalised to ``\bar m_0``; the systems differ only
in where the external forces are applied.

**Generalised forces.**  With ``F_j = f_j R_j e_3`` the thrust at vertex ``j``
(zero where there is no vehicle), ``\delta x_j = \delta x_0 + \sum_{i<j} \ell_i
\delta q_i`` gives

.. math::
    :label: m6-forces

    \delta W = \Bigl(\sum_j F_j\Bigr)\cdot \delta x_0
             + \sum_i \ell_i\,\Bigl(\underbrace{\sum_{j>i} F_j}_{=:\,G_i}\Bigr)
               \cdot \delta q_i ,

so the generalised force conjugate to ``q_i`` is ``\ell_i G_i``, the net
external force on **everything below link** ``i``.  That is the whole content
of the reference's cumulative sum ``sum(uk(:,i+1:end),2)``, and it is this
model's analogue of M5's tension recursion: tension propagation is *implicit*,
carried by ``G_i`` and projected perpendicular to ``q_i``, never assembled as a
scalar per link.

**Euler-Lagrange.**  The ``x_0`` equation is ordinary; for each ``q_i`` the
constrained variation ``\delta q_i = \xi_i \times q_i`` reduces stationarity to
``\hat q_i[\,\cdot\,] = 0``.  With ``\ddot q_j = \dot\omega_j \times q_j - c_j
q_j``, ``c_j := \|\dot q_j\|^2``, and ``\alpha := \ddot x_0 + g e_3`` — gravity
then cancels identically from every link row, since ``\ell_i S_i = M_{0i}`` —
the system is

.. math::
    :label: m6-system

    \begin{bmatrix}
      M_{00} I       & -M_{01}\hat q_1          & \cdots & -M_{0n}\hat q_n \\
      M_{01}\hat q_1 & M_{11} I                 & \cdots & -M_{1n}\hat q_1 \hat q_n \\
      \vdots         &                          & \ddots & \vdots \\
      M_{0n}\hat q_n & -M_{n1}\hat q_n \hat q_1 & \cdots & M_{nn} I
    \end{bmatrix}
    \begin{bmatrix} \alpha \\ \dot\omega_1 \\ \vdots \\ \dot\omega_n \end{bmatrix}
    =
    \begin{bmatrix}
      \sum_j F_j + \sum_k M_{0k} c_k q_k \\
      \sum_{j \ne 1} M_{1j} c_j\, \hat q_1 q_j + \ell_1 \hat q_1 G_1 \\
      \vdots \\
      \sum_{j \ne n} M_{nj} c_j\, \hat q_n q_j + \ell_n \hat q_n G_n
    \end{bmatrix},

a symmetric ``(3 + 3n) \times (3 + 3n)`` solve, followed by
``\ddot x_0 = \alpha - g e_3`` and, per vehicle,
``\dot R_k = R_k \hat\Omega_k``, ``J_k \dot\Omega_k = M_k - \Omega_k \times J_k
\Omega_k``.  **The matrix is M5's exactly, given the same** ``\bar m``: the
attachment pattern reaches it only by deciding which vertices carry an ``m_Q``,
and two patterns that leave ``\bar m`` alone leave ``M`` alone.  It is *not*
independent of the pattern — the pattern is most of what sets ``\bar m``, and
at ``n = 4`` on the shipped link geometry ``\|M\|_F`` runs 3.17 for ``(2,)``,
4.89 for ``(0, 4)``, 6.84 for ``(0, 2, 4)`` and 10.75 for ``(0, 1, 2, 3, 4)``.
What the pattern does *not* enter is the structure: the block sparsity, the
``M_{ii} I`` diagonal and the ``-M_{ij}\hat q_i \hat q_j`` off-diagonals are
M5's whatever the vehicles are bolted to, and the vehicles' *inputs* enter only
through ``\sum_j F_j`` and ``G_i`` on the right-hand side.  Setting
``n_Q = 1`` with the vehicle at vertex 0 makes every ``G_i = 0`` and recovers
:eq:`m5-system` term for term; the tests assert that numerically against
:class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`.

**How much that reduction is worth.**  Less than it looks.  There is no shared
code in the import sense — M5 and M6 have separate ``_assemble`` methods in
separate modules and neither imports the other — but the two bodies are
near-verbatim structural clones: the same loop over link rows, the same
``M_{ii} I`` gauge substitution, the same ``c_i = \|\dot q_i\|^2``, the same
dense solve.  A derivation error common to both would reproduce itself
faithfully.  The reduction is a strong *consistency* check on the port and it
would catch an indexing or sign slip introduced here, but it is not independent
validation.  What validates these equations is the constrained-DAE
re-derivation — a different unknown vector, Lagrange multipliers on
``\|q_i\| = 1``, a KKT solve, nothing read off the coefficients above — which
agrees to 2e-12 across ``n \in \{1, \dots, 8\}``, every attachment pattern and
heterogeneous parameters.  (M5 grades its own ``n = 1`` reduction to M2 as its
strongest check precisely because M2 *was* verified some other way; that
qualifier is what fails here, since M5 was verified by this same lineage of
re-derivation rather than independently of it.)

Does M5's ``M_{ii} I`` structure carry over?
--------------------------------------------
**Yes, unchanged, and for the same reason.**  The Lagrangian diagonal block is
``-M_{ii}\hat q_i \hat q_i = M_{ii}(I - q_i q_i^\top)``, rank 2:
``\dot\omega_i`` along ``q_i`` is the gauge freedom of the ``S^2``
parametrisation and appears nowhere in the dynamics.  Replacing it by
``M_{ii} I`` adds ``M_{ii} q_i q_i^\top`` and *nothing else*, because every
other term in link row ``i`` — including the new ``\ell_i \hat q_i G_i`` — is a
cross product with ``q_i`` and so has no ``q_i`` component.  The substitution
therefore appends the exact tangency condition ``\dot\omega_i \cdot q_i = 0``
rather than approximating anything, and turns an ``n``-fold singular matrix
into a positive definite one.  The reference makes the same substitution, in
``\ddot q`` coordinates: its ``-M_{i+1,j+1}`` diagonal is algebraically
identical to ``\hat q_i^2 M_{ii} \ddot q_i`` once ``q_i \cdot \ddot q_i =
-c_i`` is used, and its own ``computeMassMatrix_om`` writes the ``\dot\omega``
form above verbatim.

**Invertibility.**  With the Lagrangian diagonal, :eq:`m6-system` is the
Hessian of ``T`` in ``(\dot x_0, \dot q_1, \dots)`` pulled back through
``\dot q_i = -\hat q_i \omega_i``: positive semi-definite, with kernel exactly
the gauge directions **provided every** ``\bar m_j > 0``.  Adding
``\sum_i M_{ii} q_i q_i^\top`` is strictly positive there, so the matrix is
positive definite for *every* link configuration.  A vertex with
``\bar m_j = 0`` breaks this and is refused: the two links meeting at a
massless vertex can then hinge freely without moving any mass, an extra
three-dimensional kernel that no gauge fix removes.  The *hose* mass at a
vertex may be zero — an airframe there supplies ``\bar m_j`` on its own, which
is exactly the configuration the M5 reduction needs.

Does M4's elimination apply?  Does M5's answer carry over?
-----------------------------------------------------------
**No, and yes, respectively — and multiple vehicles do not change it.**  In
M2/M3/M4 each cable joins the payload directly to one quadrotor, so
``\dot\omega_i`` appears only in link row ``i`` and in the payload row, and
projecting the *quadrotor* equation onto ``q_i`` yields the scalar tension as
an affine function of the payload accelerations — which collapses the system to
``3 \times 3`` (M3) or ``6 \times 6`` (M4) at any number of vehicles.

Here the links are a serial chain: link row ``i`` couples to link row ``j``
through ``-M_{ij}\hat q_i \hat q_j`` for every ``j \ne i``.  Projecting link row
``i`` onto ``q_i`` annihilates **every** term — each one is a
``\hat q_i(\cdot)``, the new force term included — so it returns the gauge
condition and nothing about ``\alpha``.  **No per-link projection decouples a
serial chain**, exactly as M5 found; attaching more vehicles to more vertices
changes only the right-hand side, so it cannot change the answer.  The dense
``(3 + 3n)`` solve is the natural direct form.

That is a statement about this family of eliminations, not a lower bound: a
serial chain is what articulated-body / recursive Newton-Euler algorithms solve
in ``O(n)``.  Not taken here, for M5's reason — at the link counts the
ZOH-Euler stability limit permits, the dense solve is not the bottleneck.

Static trim
-----------
Set every rate and acceleration to zero.  Introduce the **tension vector**
``\tau_i`` of link ``i``, defined by ``G_i = S_i g e_3 + \tau_i``.  Link row
``i`` becomes ``\hat q_i(M_{0i} g e_3 - \ell_i G_i) = 0``, i.e.
``\hat q_i \tau_i = 0``: **each link points along its own tension**.  Vertex
``j``'s force balance is then, telescoping ``F_j = G_{j-1} - G_j`` with
``\tau_{-1} = \tau_n = 0``,

.. math::
    :label: m6-trim

    F_j = \bar m_j\, g\, e_3 + \tau_{j-1} - \tau_j ,
    \qquad
    q_i = \tau_i / \|\tau_i\| ,
    \qquad
    \sum_j F_j = M_{00}\, g\, e_3 ,

and an equilibrium is exactly a choice of ``\{\tau_i\}`` that makes ``F_j = 0``
at every vertex *without* a vehicle.  This is the discrete catenary, it is
closed form, and :meth:`MultiQuadHoseBase.hover_trim` propagates it.  The
default choice puts the whole system weight on the topmost vehicle, which gives
``\tau_i = -S_i g e_3``, every link vertical, and reduces to M5's
``f = M_{00} g``, ``R = I``, ``M = 0`` when there is one vehicle at vertex 0.

Fed open loop this trim holds *exactly* — root drift 0.0 over a second — but
that number is conditional on the actuator, not just on the algebra.  The
default split asks one airframe for the whole system weight, which is 22.6 N
at ``n_Q = 2`` and 49.1 N at ``n_Q = 5`` against a shipped ``_max_thrust`` of
20 N, so on shipped limits :meth:`step` clips the very command that would have
held and the root falls 0.56 m (``n_Q = 2``) to 2.9 m (``n_Q = 5``) in the same
second.  :meth:`hover_trim` warns when the split exceeds the limit, and the
test that reports 0.0 raises ``_max_thrust`` first, deliberately: the claim
under test is about the equations of motion.  A trim you intend to *fly* wants
a split the airframes can deliver, and
:meth:`~udaan.control.multi_quad_hose.MultiQuadHoseController.equilibrium_reference`
is the one the M6 LQR regulates against: a discrete **catenary**, with a
horizontal pull that keeps every link taut and spreads the weight over the
vehicles instead of stacking them.  It is not a second trim routine — it comes
out of the flat map at an all-zero-derivative flat output — which is why it
lives on the controller rather than here.

The reference solves the same conditions numerically in
``matlab/utils/optimize_for_des_state.m``, minimising total thrust subject to
:eq:`m6-trim` plus a chain-closure constraint pinning the vehicles to
prescribed positions.  That is the useful general form — a *setpoint solver*.
It is not reimplemented here: :meth:`hover_trim` covers the closed-form branch
the tests need, and a constrained optimiser is a controller-side tool rather
than part of the plant.

Numerical stability
-------------------
Inherited from M5 and, at equal link count, slightly *better*.  Linearising
about the free-hanging equilibrium gives modes solving
``\mathrm{diag}(0, g M_{0k})\phi = \lambda M \phi``
(:meth:`MultiQuadHoseBase.natural_frequencies`), and forward Euler on an
undamped oscillator of frequency ``\omega`` multiplies amplitudes by
``\sqrt{1 + (\omega h)^2}`` per substep — ``e^{\omega^2 h/2}`` per second.
:meth:`MultiQuadHoseBase.stability_growth_rate` returns that exponent and the
constructor warns above 1.0, the threshold M5 calibrated against measurement.

The envelope is a function of ``(n, n_Q)`` and of the mass distribution, and
``n`` dominates.  On the defaults, at ``n_Q = 2`` (a vehicle at each end), the
exponent runs

===== ===== ===== ===== ===== ===== ======
``n``  1     2     4     6     8     16
===== ===== ===== ===== ===== ===== ======
rate  0.006 0.09  0.49  1.13  2.02  8.22
===== ===== ===== ===== ===== ===== ======

That row alone would put the usable region at ``n \le 5``, the limit M5 reports
for the same integrator on the same kind of chain.  **It is one row, and it is
the wrong one to generalise from.**  ``n_Q`` matters much less than ``n`` but
it is not negligible, and it is not monotonic: what the fastest mode responds
to is the mass *contrast* between neighbouring vertices, not the vehicle count.
Bolting a vehicle to every vertex evens the chain out — ``(0, 1, 2, 3, 4)`` at
``n = 4`` scores 0.13 — while clustering vehicles at one end makes the contrast
as bad as it gets.  Enumerating *every* pattern on the shipped parameters:

===== ========== ================== =========================
``n`` patterns   over the 1.0 warn  worst
===== ========== ================== =========================
3     15         0                  0.42 at ``(2, 3)``
4     31         0                  1.00 at ``(2, 3, 4)``
5     63         23                 1.94 at ``(2, 3, 4, 5)``
6     127        101                3.34 at ``(2, 3, 4, 5, 6)``
===== ========== ================== =========================

so the honest envelope at any ``n_Q`` is ``n \le 4``.  At ``n = 5``, 23 of the
63 patterns cross the threshold — ``(2, 3, 4, 5)`` at 1.94, ``(3, 4, 5)`` at
1.83, ``(0, 3, 4, 5)`` at 1.69 — even though the endpoints-only ``(0, 5)`` in
the table above sits at a comfortable 0.78.  Above ``n = 4`` the limit is
therefore *pattern-dependent*, and ``n = 4`` itself is only just inside it:
``(2, 3, 4)`` reaches 0.997.  The shipped default, ``n = 4`` with vehicles at
``(0, 4)``, scores 0.49 and is not close to trouble.  A configuration this
model warns about is one to run on the MuJoCo backend, which integrates the
same continuous dynamics with its own solver.

The reference
-------------
``@MultipleQuadrotorFlexibleHose`` is the best-written repository in the
collection — properly object-oriented, generic in both ``n`` and ``n_Q``, and
written with loops rather than hand-wired index arithmetic — and its
``get_dyn_dq`` (L274-332) survived the re-derivation **intact**: the mass
coefficients, the cumulative force sum, the gauge substitution and the sign of
every term agree with :eq:`m6-system`.  ``computeMassMatrix_om`` (L256-272)
independently agrees with the ``\dot\omega`` form used here.  Four things
outside that routine did not survive, none of which affects it:

* ``get_dyn_dq_damped`` (L334-404) **does not damp**.  It adds
  ``-c\,\omega_k = -c\,(q_k \times \dot q_k)`` to ``\ddot q_k`` *after* the
  mass-matrix solve.  That vector is orthogonal to ``\dot q_k``, so its power
  ``\dot q_k \cdot (-c\,q_k \times \dot q_k)`` is identically zero: the term is
  gyroscopic and precesses each link instead of dissipating.  Proper link
  damping is ``-c\,\omega_k`` added to ``\dot\omega_k`` (or ``-c\,\dot q_k`` to
  ``\ddot q_k``), as a generalised force inside the solve.  **It is dead code,
  and no published result depends on it**: ``parse_inputs`` only reaches it
  when ``'dynamics'`` is ``'damped'``, all five ``matlab/examples`` pass
  ``'default'`` or ``'mex'``, and the codegen path in ``generateMexFiles``
  sets ``damp_coeff = 0``.  Recorded as a latent bug in an unused branch, not
  as an error in a result.  No damped variant is ported here.
* ``parse_inputs`` accepts an arbitrary ``setI`` but ``get_dyn_dq`` assigns
  thrusts to vertices by walking the chain **in vertex order**
  (``j = j + 1`` inside the ``for i = 1:nvert`` loop), so an unsorted ``setI``
  pairs vehicle ``k``'s thrust with a different vertex than its own
  ``quad(k).ind`` records.  With homogeneous vehicles it is invisible.  Here
  :func:`udaan.utils.hose.validate_attachments` refuses the input outright.
* ``parse_inputs`` does not range-check ``setI`` either: an index past
  ``nvert`` silently grows ``obj.mQ`` and the failure surfaces much later, as a
  length mismatch in ``mbar = mQ + m``.
* The alternative link-row right-hand side commented out at L301 (and again at
  L365) sums ``mbar(i:end)`` and ``uk(:,i:end)`` where the live line sums from
  ``i+1``.  It is off by one — it charges link ``i`` with its own proximal
  vertex's mass and force — and would be wrong if it were ever enabled.  The
  live line is right.
* ``optimize_for_des_state`` (L48-52) — unlike the three above, **this one is
  on the published path**: ``computeSetpoint`` calls it, and
  ``sim_1a_SETPOINT_nQ2`` / ``sim_1b_SETPOINT_nQ3`` — the paper's Sim 1.1 and
  1.2 — call ``computeSetpoint``.  Its ``cost_fun`` is commented "sum of the
  square of the tension norms" but sums the squared *thrusts*; minimising
  thrust is the stated intent of the method, so it is the comment that is
  wrong, not the objective.  More consequential: the solver leaves the tension
  magnitudes unbounded in sign, so nothing stops it returning a setpoint with
  links in compression — a hose cannot push.  Both are worth knowing when
  reading those figures.

Dynamics reference:
Prasanth Kotaru and Koushil Sreenath. "Multiple Quadrotors Carrying a Flexible
Hose: Dynamics, Differential Flatness and Control." IFAC World Congress, 2020.
The equations above were re-derived from the Lagrangian rather than transcribed
from the accompanying MATLAB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...control.multi_quad_hose import MultiQuadHoseController
from ...control.quadrotor import GeometricAttitudeController, PositionPDController
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_HOSE_LENGTH,
    DEFAULT_HOSE_MASS,
    DEFAULT_HOSE_SAG_RATIO,
    DEFAULT_NUM_HOSE_LINKS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...flatness.multi_quad_hose import NotFlatError
from ...manif import S2, SO3, TS2, TSO3, hat
from ...utils.hose import hose_spawn_directions, validate_attachments
from ...utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = [
    "MultiQuadHoseBase",
    "MultiQuadHoseState",
]

#: Below this mass (kg) a vertex is warned about as unphysical. The dynamics
#: stay well posed — the guard that matters is strict positivity of the *total*
#: vertex mass — but a vertex this light is almost certainly a unit error.
_MIN_PHYSICAL_MASS = 1e-9

#: A ``reset()`` link direction shorter than this cannot be normalised onto
#: ``S^2`` and is rejected rather than repaired.
_MIN_DIRECTION_NORM = 1e-12

#: Per-second amplitude growth exponent above which the constructor warns that
#: the hose is too stiff for the ZOH-Euler integrator. M5's threshold, kept
#: because it is the same integrator on the same kind of chain: over a taut
#: multi-second run a chain at 0.45 stays bounded, one at 1.46 winds up, and
#: one at 3.4 overflows. See :meth:`MultiQuadHoseBase.stability_growth_rate`.
_STABILITY_GROWTH_WARN = 1.0


@dataclass
class MultiQuadHoseState:
    """State of ``nQ`` quadrotors joined in series by one ``n``-link hose.

    Per-**link** entries are lists of length ``num_links`` indexed by link, and
    per-**vehicle** entries are lists of length ``num_quadrotors`` indexed by
    vehicle — *not* by vertex.  ``attachments[k]`` is the vertex vehicle ``k``
    sits on.

    ``root_position``/``root_velocity`` are vertex 0's and are the integrated
    translational state; every other vertex, the quadrotor positions included,
    is derived from them through the chain recursion.  The MuJoCo backend
    overwrites the derived entries with the measured body states.
    """

    num_links: int = DEFAULT_NUM_HOSE_LINKS
    #: Vertex index of each vehicle. ``None`` means one at each end, which is
    #: the only default that stays valid at every ``num_links``.
    attachments: tuple[int, ...] | None = None

    # Vertex 0 — the integrated translational state.
    root_position: Vec3 = field(default_factory=lambda: np.zeros(3))
    root_velocity: Vec3 = field(default_factory=lambda: np.zeros(3))

    # Per-link direction and angular velocity.
    link_attitudes: list[S2] = field(default_factory=list)
    link_angular_velocities: list[TS2] = field(default_factory=list)

    # Per-vehicle attitude and body rate.
    orientations: list[SO3] = field(default_factory=list)
    angular_velocities: list[TSO3] = field(default_factory=list)

    # Per-vehicle position and velocity, derived from the chain.
    positions: list[Vec3] = field(default_factory=list)
    velocities: list[Vec3] = field(default_factory=list)

    def __post_init__(self):
        if self.num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {self.num_links}")
        if self.attachments is None:
            self.attachments = (0, self.num_links)
        self.attachments = validate_attachments(self.attachments, self.num_links)
        if not self.link_attitudes:
            self.reset()

    @property
    def num_quadrotors(self) -> int:
        assert self.attachments is not None  # settled by __post_init__
        return len(self.attachments)

    def reset(self):
        """Reset every field: root, every link *and* every vehicle entry."""
        n = self.num_links
        nQ = self.num_quadrotors
        self.root_position = np.zeros(3)
        self.root_velocity = np.zeros(3)
        self.link_attitudes = [S2(np.array([0.0, 0.0, -1.0])) for _ in range(n)]
        self.link_angular_velocities = [TS2() for _ in range(n)]
        self.orientations = [SO3() for _ in range(nQ)]
        self.angular_velocities = [TSO3() for _ in range(nQ)]
        self.positions = [np.zeros(3) for _ in range(nQ)]
        self.velocities = [np.zeros(3) for _ in range(nQ)]

    def dq(self, i: int) -> np.ndarray:
        """Link direction rate ``q̇_i = ω_i × q_i`` of link ``i``."""
        return np.cross(
            np.asarray(self.link_angular_velocities[i]),
            np.asarray(self.link_attitudes[i]),
        )


class MultiQuadHoseBase:
    """``nQ`` quadrotors carrying one ``n``-link flexible hose — pure dynamics.

    The quadrotors are rigid bodies: the per-vehicle input is a thrust
    magnitude and a body moment (or a thrust vector, or propeller forces — see
    :meth:`_repackage_input`).  Link lengths and vertex masses are per-index and
    accept a scalar (broadcast) or a full-length sequence; so are the vehicle
    masses and inertias.

    Mass bookkeeping.  ``vertex_masses[j]`` is the hose's own lumped mass at
    vertex ``j`` and ``masses[k]`` is vehicle ``k``'s; what enters the dynamics
    is their sum at each vertex, :attr:`vertex_total_masses`.  The constructor
    offers ``hose_mass`` as a convenience that spreads one number evenly over
    the ``n + 1`` vertices, but ``vertex_masses`` is the parameter and passing
    it directly overrides that.

    Usage::

        mdl = MultiQuadHoseBase(num_links=6, attachments=(0, 3, 6))
        mdl.link_lengths = 0.4
        mdl.reset(root_position=np.array([0.0, 0.0, 2.0]))
        mdl.simulate(tf=2.0)

    Keep an eye on :meth:`stability_growth_rate` when raising ``num_links``:
    the ZOH-Euler loop is only conditionally stable and the constructor warns
    when a configuration leaves the usable region.
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self._num_links = int(kwargs.get("num_links", DEFAULT_NUM_HOSE_LINKS))
        if self._num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {self._num_links}")
        n = self._num_links
        self._attachments = validate_attachments(kwargs.get("attachments", (0, n)), n)
        nQ = len(self._attachments)
        #: Structural, and fixed for the life of the model: the attachment
        #: pattern shapes the state, the controllers and (in the MuJoCo
        #: backend) the MJCF, so it is not a settable parameter.
        self._num_quadrotors = nQ

        # The parameter setters push their new values into the controllers, and
        # the constructor runs several of them *before* those controllers exist.
        self._controllers_ready = False

        self.state = MultiQuadHoseState(num_links=n, attachments=self._attachments)
        self.t = 0.0
        self.verbose = kwargs.get("verbose", 0)

        # Hose parameters
        total_length = float(kwargs.get("hose_length", DEFAULT_HOSE_LENGTH))
        if total_length <= 0.0:
            raise ValueError(f"hose_length must be positive, got {total_length}")
        self._link_lengths = np.full(n, total_length / n)
        hose_mass = float(kwargs.get("hose_mass", DEFAULT_HOSE_MASS))
        self._vertex_masses = np.full(n + 1, hose_mass / (n + 1))

        # Vehicle parameters
        self._masses = np.full(nQ, DEFAULT_QUAD_MASS)
        self._inertias = np.tile(DEFAULT_QUAD_INERTIA, (nQ, 1, 1))
        self._inertia_invs = np.linalg.inv(self._inertias)

        if "link_lengths" in kwargs:
            self.link_lengths = kwargs["link_lengths"]
        if "vertex_masses" in kwargs:
            self.vertex_masses = kwargs["vertex_masses"]
        if "masses" in kwargs:
            self.masses = kwargs["masses"]
        if "inertias" in kwargs:
            self.inertias = kwargs["inertias"]
        self._check_positive(self._link_lengths, "link_lengths")
        self._check_vertex_masses()

        self._sag_ratio = float(kwargs.get("sag_ratio", DEFAULT_HOSE_SAG_RATIO))

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits (per vehicle)
        self._min_thrust = 0.0
        self._max_thrust = 20.0
        self._min_torque = np.array([-5.0, -5.0, -2.0])
        self._max_torque = np.array([5.0, 5.0, 2.0])
        self._prop_min_force = 0.0
        self._prop_max_force = 10.0

        # Integration: 5 ms outer loop, 4 inner substeps at 1.25 ms
        self.dt = kwargs.get("dt", 1.0 / 200.0)
        self._inner_loop_steps = kwargs.get("inner_loop_steps", 4)
        self.h = self.dt / self._inner_loop_steps

        # Input/force type.  M6's default cascade emits a thrust *vector* per
        # vehicle, as M3's and M4's do, hence the ACCELERATION default.  An
        # unrecognised value is refused rather than resolved to the default:
        # the three pipelines take three incompatible input shapes, so a typo
        # that falls through does not mistune the loop, it runs a different one.
        self._input_type = InputType.ACCELERATION
        self._force_type = ForceType.WRENCH
        _INPUT_TYPES = {
            "acceleration": InputType.ACCELERATION,
            "wrench": InputType.WRENCH,
            "prop_forces": InputType.PROP_FORCES,
        }
        if "input" in kwargs:
            try:
                self._input_type = _INPUT_TYPES[kwargs["input"]]
            except (KeyError, TypeError):
                raise ValueError(
                    f"unknown input type {kwargs['input']!r}; "
                    f"expected one of {sorted(_INPUT_TYPES)}"
                ) from None
        if "force" in kwargs:
            if kwargs["force"] == "prop_forces":
                self._force_type = ForceType.PROP_FORCES
            elif kwargs["force"] != "wrench":
                raise ValueError(
                    f"unknown force type {kwargs['force']!r}; "
                    "expected one of ['prop_forces', 'wrench']"
                )

        # Allocation matrix
        self._compute_allocation_matrix()

        # Per-vehicle station-keeping targets, refreshed by reset()
        self._station_targets = [np.zeros(3) for _ in range(nQ)]

        #: Steps on which the flatness feedforward was refused; see
        #: :attr:`feedforward_drops`.
        self._feedforward_drops = 0

        # Default controllers
        self._init_default_controllers()

        self._warn_if_stiff()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        """Build the default loops — two of them, one per input pipeline.

        The **station-keeping baseline** drives the ``ACCELERATION`` pipeline
        (this model's default): one position PD per vehicle holding the vertex
        :meth:`reset` left it at, feeding a geometric attitude loop.  It is the
        direct analogue of M3's
        :meth:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.compute_formation_input`
        and of M5's
        :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.compute_station_keeping_input`,
        and each PD carries its own airframe plus an equal share of the hose as
        feedforward, so the team as a whole feeds forward the system weight.

        The **hose LQR** (:attr:`hose_controller`) drives the ``WRENCH``
        pipeline and is the loop that regulates the hose itself.  It needs a
        vehicle at the last vertex — that is what makes the system flat in the
        coordinates its reference is built in — so on a pattern without one it
        is simply absent, and ``input="wrench"`` then has nothing to drive.
        """
        nQ = self._num_quadrotors
        self._att_controllers = [
            GeometricAttitudeController(inertia=self._inertias[i]) for i in range(nQ)
        ]
        self._pos_controllers = [
            PositionPDController(mass=self._share_of_system_mass(i)) for i in range(nQ)
        ]
        for i in range(nQ):
            self._pos_controllers[i].setpoint = self._station_setpoint(i)
        try:
            self._hose_controller = MultiQuadHoseController.from_model(self)
        except NotFlatError as err:
            # A legal hose for the *plant*, not for the flat map the LQR's
            # reference comes out of. Refusing at construction would make the
            # analytical model narrower than the dynamics it implements.
            self._hose_controller = None
            _logger.debug("no hose LQR for attachments %s: %s", self._attachments, err)
        self._controllers_ready = True

    def _share_of_system_mass(self, index: int) -> float:
        """Vehicle ``index``'s own mass plus an equal share of the hose's."""
        return float(self._masses[index] + self._vertex_masses.sum() / self._num_quadrotors)

    def _sync_default_controllers(self):
        """Push the current physical parameters into the default controllers.

        Every controller caches its parameters at construction, so a setter
        that writes only ``self._…`` leaves the closed loop driving a different
        machine than the one being integrated — the defect M4's review found
        and the reason every parameter setter below ends here.

        The sync is duck-typed, as M3's, M4's and M5's are: a controller
        exposing ``mass``/``inertia`` is kept in step whether or not this class
        built it, one exposing a ``sync_from_model`` hook takes the parameters
        itself, and one exposing neither is left entirely alone.
        """
        if not self._controllers_ready:
            return
        for i, controller in enumerate(self._att_controllers):
            sync = getattr(controller, "sync_from_model", None)
            if sync is not None:
                sync(self)
            elif hasattr(controller, "inertia"):
                controller.inertia = self._inertias[i]
        for i, pos_controller in enumerate(self._pos_controllers):
            sync = getattr(pos_controller, "sync_from_model", None)
            if sync is not None:
                sync(self)
            elif hasattr(pos_controller, "mass"):
                pos_controller.mass = self._share_of_system_mass(i)
        sync = getattr(self._hose_controller, "sync_from_model", None)
        if sync is not None:
            sync(self)

    def _station_setpoint(self, i: int):
        """Station-keeping setpoint callable for vehicle ``i``."""
        return lambda t: (self._station_targets[i], np.zeros(3), np.zeros(3))

    @property
    def position_controllers(self) -> list:
        """Per-vehicle station-keeping position controllers."""
        return self._pos_controllers

    @position_controllers.setter
    def position_controllers(self, controllers: list):
        self._check_length(controllers, "position_controllers", self._num_quadrotors)
        self._pos_controllers = list(controllers)

    @property
    def attitude_controllers(self) -> list:
        """Per-vehicle attitude controllers."""
        return self._att_controllers

    @attitude_controllers.setter
    def attitude_controllers(self, controllers: list):
        self._check_length(controllers, "attitude_controllers", self._num_quadrotors)
        self._att_controllers = list(controllers)

    @property
    def hose_controller(self):
        """The M6 LQR — what :meth:`simulate` drives under ``InputType.WRENCH``.

        ``None`` when no vehicle sits at the last vertex: the reference the LQR
        regulates against comes out of
        :class:`udaan.flatness.MultiQuadHose`, which such a pattern is not flat
        for.  The plant is fine; it is the *reference* that does not exist.

        Bind a trajectory to its ``setpoint`` — the **root vertex** reference,
        ``t -> (pos, vel, acc)`` — to fly one::

            mdl.hose_controller.setpoint = traj.get
        """
        return self._hose_controller

    @hose_controller.setter
    def hose_controller(self, controller):
        self._hose_controller = controller

    # ─── Parameters ────────────────────────────────────────────────

    @staticmethod
    def _check_positive(values, name: str):
        arr = np.asarray(values, dtype=float)
        if not np.all(arr > 0.0):
            raise ValueError(f"{name} must be strictly positive; got {arr.tolist()}")

    @staticmethod
    def _check_length(value, name: str, expected: int):
        if len(value) != expected:
            raise ValueError(f"{name} must have {expected} entries, got {len(value)}")

    def _check_vertex_masses(self):
        """Refuse a hose whose mass distribution makes the solve singular.

        Individual ``vertex_masses`` entries may be zero — an airframe bolted
        to that vertex supplies the mass on its own, and that is exactly the
        configuration the M5 reduction uses — but the *total* at every vertex
        must be strictly positive.  See "Invertibility" in the module
        docstring for what a genuinely massless vertex does to
        :eq:`m6-system`.
        """
        if np.any(np.asarray(self._vertex_masses) < 0.0):
            raise ValueError(
                f"vertex_masses must be non-negative; got {self._vertex_masses.tolist()}"
            )
        self._check_positive(self._masses, "masses")
        total = self.vertex_total_masses
        if not np.all(total > 0.0):
            raise ValueError(
                "every vertex needs strictly positive total mass (hose + airframe), or the "
                "links meeting there hinge freely and the mass matrix is singular; got "
                f"{total.tolist()} for attachments {self._attachments}"
            )
        if np.any(total < _MIN_PHYSICAL_MASS):
            _logger.warning(
                "a vertex carries less than %g kg total (%s); the dynamics stay well posed "
                "but this is almost certainly a typo or a unit error",
                _MIN_PHYSICAL_MASS,
                total.tolist(),
            )

    def _broadcast(self, value, name: str, size: int) -> np.ndarray:
        """Scalar -> ``(size,)`` array, or validate a per-index sequence."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(size, float(arr))
        if arr.shape != (size,):
            raise ValueError(f"{name} must be a scalar or have {size} entries; got {arr.shape}")
        return arr.astype(float)

    @property
    def num_links(self) -> int:
        """``n``, the number of rigid links the hose is discretised into."""
        return self._num_links

    @property
    def num_vertices(self) -> int:
        """``n + 1``, the number of joints the hose's mass is lumped at."""
        return self._num_links + 1

    @property
    def num_quadrotors(self) -> int:
        """``nQ``, the number of vehicles bolted to the hose."""
        return self._num_quadrotors

    @property
    def attachments(self) -> tuple[int, ...]:
        """Vertex index of each vehicle, strictly increasing."""
        return self._attachments

    @property
    def link_lengths(self) -> np.ndarray:
        """Per-link lengths, shape ``(n,)``."""
        return self._link_lengths

    @link_lengths.setter
    def link_lengths(self, value):
        arr = self._broadcast(value, "link_lengths", self._num_links)
        self._check_positive(arr, "link_lengths")
        self._link_lengths = arr
        self._sync_default_controllers()

    @property
    def vertex_masses(self) -> np.ndarray:
        """Hose mass lumped at each vertex, shape ``(n + 1,)``.

        The airframe masses are *not* included; see
        :attr:`vertex_total_masses`.
        """
        return self._vertex_masses

    @vertex_masses.setter
    def vertex_masses(self, value):
        arr = self._broadcast(value, "vertex_masses", self.num_vertices)
        previous = self._vertex_masses
        self._vertex_masses = arr
        try:
            self._check_vertex_masses()
        except ValueError:
            self._vertex_masses = previous
            raise
        self._sync_default_controllers()

    @property
    def vertex_total_masses(self) -> np.ndarray:
        """``m̄_j = m_j + m_{Q_j}``, shape ``(n + 1,)`` — what the dynamics see."""
        total = np.array(self._vertex_masses, dtype=float)
        for k, vertex in enumerate(self._attachments):
            total[vertex] += self._masses[k]
        return total

    @property
    def masses(self) -> np.ndarray:
        """Per-vehicle airframe masses, shape ``(nQ,)``."""
        return self._masses

    @masses.setter
    def masses(self, value):
        arr = self._broadcast(value, "masses", self._num_quadrotors)
        previous = self._masses
        self._masses = arr
        try:
            self._check_vertex_masses()
        except ValueError:
            self._masses = previous
            raise
        self._sync_default_controllers()

    @property
    def inertias(self) -> np.ndarray:
        """Per-vehicle inertia tensors, shape ``(nQ, 3, 3)``.

        Accepts a single ``(3,)`` diagonal or ``(3, 3)`` tensor, broadcast to
        every vehicle, or a per-index ``(nQ, 3, 3)`` stack.  A ``(nQ, 3)`` stack
        of diagonals is deliberately *not* accepted: at ``nQ == 3`` it has the
        same shape as a single tensor, and symmetry — required anyway — is what
        disambiguates the two.  Build the stack with :func:`numpy.diag`.
        """
        return self._inertias

    @inertias.setter
    def inertias(self, value):
        arr = np.asarray(value, dtype=float)
        nQ = self._num_quadrotors
        if arr.shape == (3,):
            arr = np.tile(np.diag(arr), (nQ, 1, 1))
        elif arr.shape == (3, 3):
            arr = np.tile(arr, (nQ, 1, 1))
        elif arr.shape != (nQ, 3, 3):
            raise ValueError(f"inertias must be (3,), (3, 3) or (nQ, 3, 3); got {arr.shape}")
        for i in range(nQ):
            self._check_inertia_tensor(arr[i], f"inertias[{i}]")
        self._inertias = arr
        self._inertia_invs = np.linalg.inv(arr)
        self._sync_default_controllers()

    @staticmethod
    def _check_inertia_tensor(tensor: np.ndarray, name: str):
        """Reject anything that is not a symmetric positive-definite tensor."""
        scale = max(1.0, float(np.max(np.abs(tensor))))
        if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1e-12 * scale):
            raise ValueError(
                f"{name} must be a symmetric tensor; got {tensor.tolist()}. "
                "An (nQ, 3) stack of diagonals is not accepted — build the "
                "(nQ, 3, 3) stack with numpy.diag."
            )
        if not np.all(np.linalg.eigvalsh(tensor) > 0.0):
            raise ValueError(f"{name} must be positive definite; got {tensor.tolist()}")

    @property
    def inertia_invs(self) -> np.ndarray:
        return self._inertia_invs

    @property
    def sag_ratio(self) -> float:
        """Horizontal span of each spawn segment, as a fraction of its length."""
        return self._sag_ratio

    @sag_ratio.setter
    def sag_ratio(self, value: float):
        ratio = float(value)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"sag_ratio must lie in (0, 1]; got {ratio}")
        self._sag_ratio = ratio

    @property
    def hose_length(self) -> float:
        """Total (taut) hose length ``Σ_i l_i``, in metres."""
        return float(np.sum(self._link_lengths))

    @property
    def total_mass(self) -> float:
        """Hose + every airframe, in kg — the ``M₀₀`` of the derivation."""
        return float(self._vertex_masses.sum() + self._masses.sum())

    # ─── Geometry ──────────────────────────────────────────────────

    def vertex_positions(self) -> np.ndarray:
        """World positions of the ``n + 1`` vertices, shape ``(n + 1, 3)``."""
        out = np.zeros((self.num_vertices, 3))
        out[0] = np.asarray(self.state.root_position, dtype=float)
        for i in range(self._num_links):
            out[i + 1] = out[i] + self._link_lengths[i] * np.asarray(self.state.link_attitudes[i])
        return out

    def vertex_velocities(self) -> np.ndarray:
        """World velocities of the ``n + 1`` vertices, shape ``(n + 1, 3)``."""
        out = np.zeros((self.num_vertices, 3))
        out[0] = np.asarray(self.state.root_velocity, dtype=float)
        for i in range(self._num_links):
            out[i + 1] = out[i] + self._link_lengths[i] * self.state.dq(i)
        return out

    def spawn_link_attitudes(self) -> list[S2]:
        """The taut spawn directions :meth:`reset` falls back on.

        Delegates to :func:`udaan.utils.hose.hose_spawn_directions`, which the
        MJCF generator reads too — so both backends spawn the same shape.
        """
        directions = hose_spawn_directions(self._link_lengths, self._attachments, self._sag_ratio)
        return [S2(directions[i]) for i in range(self._num_links)]

    # ─── Small-oscillation analysis ────────────────────────────────

    def natural_frequencies(self) -> np.ndarray:
        r"""Small-oscillation frequencies about the free-hanging equilibrium, rad/s.

        M5's linearisation, coefficient for coefficient.  Writing
        ``q_i = -e_3 + u_i`` with ``u_i`` horizontal, ``\dot q_i \approx \dot
        u_i`` and ``e_3 \cdot q_i \approx -1 + \tfrac12\|u_i\|^2``, so per
        horizontal axis

        .. math::

            T \approx \tfrac12
              \begin{bmatrix} \dot x_0 \\ \dot u \end{bmatrix}^\top
              \begin{bmatrix} M_{00} & M_{0j} \\ M_{0i} & M_{ij} \end{bmatrix}
              \begin{bmatrix} \dot x_0 \\ \dot u \end{bmatrix},
            \qquad
            V \approx \text{const} + \tfrac12 g \sum_k M_{0k} \|u_k\|^2 ,

        and the modes solve ``\mathrm{diag}(0, g M_{0k})\phi = \lambda M \phi``.
        The zero eigenvalue is the free horizontal translation of the whole
        system; the remaining ``n`` are the hose's pendulum modes.

        This is the branch where the *whole* system weight is carried at the
        topmost vertex, so every link is at its maximum tension.  Thrust at any
        lower vertex unloads the links above it and slows those modes down, so
        the number returned is a **conservative** bound on any trim of the same
        hose — which is what a stability diagnostic should be.

        Returns:
            ``(n + 1,)`` frequencies in ascending order, the first of which is
            the rigid-translation zero.
        """
        from scipy.linalg import eigh

        n = self._num_links
        M00, M0i, Mij = self._mass_coefficients()
        mass = np.zeros((n + 1, n + 1))
        mass[0, 0] = M00
        mass[0, 1:] = M0i
        mass[1:, 0] = M0i
        mass[1:, 1:] = Mij
        stiffness = np.zeros((n + 1, n + 1))
        stiffness[1:, 1:] = np.diag(self._g * M0i)
        eigenvalues = np.clip(eigh(stiffness, mass, eigvals_only=True), 0.0, None)
        return np.asarray(np.sqrt(eigenvalues))

    def stability_growth_rate(self) -> float:
        r"""Amplitude growth rate of the fastest mode under ZOH Euler, in 1/s.

        Forward Euler applied to an undamped oscillator of frequency ``ω``
        multiplies its amplitude by ``\sqrt{1 + (\omega h)^2}`` every substep,
        so over one second it grows by ``e^{\omega^2 h / 2}``.  This is that
        exponent for the fastest mode :meth:`natural_frequencies` finds — the
        single number that says whether a given ``(n, nQ, masses, dt)`` is
        usable.

        Below ~0.5 the growth is invisible next to the physics; above ~2 a
        few-second taut run diverges.  It is not a defect of the model: the
        continuous dynamics are fine and the MuJoCo backend integrates them
        happily.  Shrink ``dt``, coarsen the hose, or make the vertices heavier.
        """
        return float(self.natural_frequencies().max() ** 2 * self.h / 2.0)

    def _warn_if_stiff(self):
        rate = self.stability_growth_rate()
        if rate > _STABILITY_GROWTH_WARN:
            _logger.warning(
                "this hose is stiff for the ZOH-Euler integrator: its fastest "
                "small-oscillation mode is %.1f rad/s, which at h = %.3g s grows "
                "amplitudes by e^%.2f per second. Expect a taut run to diverge "
                "within a few seconds. Reduce dt (currently %.4g s), reduce "
                "num_links (currently %d), or increase the vertex masses "
                "(currently %s kg).",
                float(self.natural_frequencies().max()),
                self.h,
                rate,
                self.dt,
                self._num_links,
                np.array2string(self.vertex_total_masses, precision=4),
            )

    # ─── Dynamics ──────────────────────────────────────────────────

    def _mass_coefficients(self) -> tuple[float, np.ndarray, np.ndarray]:
        """``(M₀₀, M₀ᵢ, Mᵢⱼ)`` of :eq:`m6-coefficients`."""
        mbar = self.vertex_total_masses
        length = self._link_lengths
        # tail[j] = Σ_{k ≥ j} m̄_k, so S_i = tail[i + 1] is the mass strictly
        # below link i.
        tail = np.cumsum(mbar[::-1])[::-1]
        S = tail[1:]
        M00 = float(mbar.sum())
        M0i = S * length
        n = self._num_links
        Mij = S[np.maximum.outer(np.arange(n), np.arange(n))] * np.outer(length, length)
        return M00, M0i, Mij

    def vertex_forces(self, thrusts) -> np.ndarray:
        """External force applied at each vertex, shape ``(n + 1, 3)``.

        ``F_j = f_k R_k e₃`` at the vertex vehicle ``k`` sits on, zero
        elsewhere.  Gravity is *not* included: it is folded into ``α`` and
        cancels from every link row (see the derivation).
        """
        forces = np.zeros((self.num_vertices, 3))
        thrust_array = np.asarray(thrusts, dtype=float).reshape(-1)
        for k, vertex in enumerate(self._attachments):
            forces[vertex] = thrust_array[k] * (self.state.orientations[k] @ self._e3)
        return forces

    def _assemble(self, thrusts) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the ``(3+3n)`` system :eq:`m6-system` at the current state.

        Returns ``(matrix, rhs)`` in the unknowns ``[α; ω̇_0; …; ω̇_{n-1}]``
        with ``α = a_0 + g e₃``.
        """
        n = self._num_links
        M00, M0i, Mij = self._mass_coefficients()
        q = [np.asarray(self.state.link_attitudes[i], dtype=float) for i in range(n)]
        q_hat = [hat(q[i]) for i in range(n)]
        # c_i = ‖q̇_i‖², not ‖ω_i‖² — the two agree on the tangent manifold and
        # ZOH-Euler lets q·ω drift off it at O(h). M2-M5 all use ‖q̇_i‖², and
        # using it here is what makes the M5 reduction exact rather than close.
        c = np.array([float(np.dot(self.state.dq(i), self.state.dq(i))) for i in range(n)])

        forces = self.vertex_forces(thrusts)
        # G_i = Σ_{j > i} F_j — the net external force below link i, the
        # reference's ``sum(uk(:, i+1:end), 2)``.
        below = np.cumsum(forces[::-1], axis=0)[::-1]

        size = 3 * (n + 1)
        matrix = np.zeros((size, size))
        rhs = np.zeros(size)

        matrix[0:3, 0:3] = M00 * np.eye(3)
        rhs[0:3] = forces.sum(axis=0)
        for i in range(n):
            rhs[0:3] += M0i[i] * c[i] * q[i]

        for i in range(n):
            r = 3 * (i + 1)
            matrix[0:3, r : r + 3] = -M0i[i] * q_hat[i]
            matrix[r : r + 3, 0:3] = M0i[i] * q_hat[i]
            # Diagonal: M_ii I, not M_ii (I - q qᵀ). Not an approximation — it
            # appends the exact gauge condition ω̇_i · q_i = 0 and makes the
            # matrix positive definite. See the module docstring.
            matrix[r : r + 3, r : r + 3] = Mij[i, i] * np.eye(3)
            rhs[r : r + 3] = self._link_lengths[i] * (q_hat[i] @ below[i + 1])
            for j in range(n):
                if j == i:
                    continue
                cj = 3 * (j + 1)
                matrix[r : r + 3, cj : cj + 3] = -Mij[i, j] * (q_hat[i] @ q_hat[j])
                rhs[r : r + 3] += Mij[i, j] * c[j] * (q_hat[i] @ q[j])

        return matrix, rhs

    def mass_matrix(self) -> np.ndarray:
        r"""The ``(3+3n) × (3+3n)`` coupled mass matrix at the current state.

        Positive definite for every link configuration whenever every vertex
        carries positive total mass — see "Invertibility" in the module
        docstring.  Exposed so that claim can be tested directly on degenerate
        geometries rather than inferred from a run that happens not to blow up.

        Independent of the inputs, and — *given the same*
        :attr:`vertex_total_masses` — independent of the attachment pattern
        too.  Both are worth asserting, and the second is not the stronger
        claim it sounds like: the pattern is exactly what decides which
        vertices carry an ``m_Q``, so two patterns with different ``\bar m``
        give different matrices.  At ``n = 4`` on the shipped link geometry
        ``\|M\|_F`` runs 3.17 for ``(2,)``, 4.89 for ``(0, 4)``, 6.84 for
        ``(0, 2, 4)`` and 10.75 for ``(0, 1, 2, 3, 4)``.  What is pattern-blind
        is the map ``\bar m \mapsto M``.
        """
        return self._assemble(np.zeros(self._num_quadrotors))[0]

    def accelerations(self, thrusts=None, torques=None) -> dict:
        """Every acceleration of the system at the current state.

        Solves :eq:`m6-system` once and recurses the vertex accelerations
        forward.  This is the whole right-hand side of the ODE; :meth:`_zoh` is
        an integrator wrapped around it, and the tests read residuals off it
        rather than re-deriving the algebra.

        Args:
            thrusts: ``(nQ,)`` thrust magnitudes; default zero.
            torques: ``(nQ, 3)`` body moments; default zero.

        Returns:
            A dict with keys ``root_accel`` (``ẍ_0``), ``link_ang_accels``
            (``(n, 3)`` of ``ω̇_i``), ``link_accels`` (``(n, 3)`` of ``q̈_i``),
            ``vertex_accels`` (``(n + 1, 3)``) and ``quad_ang_accels``
            (``(nQ, 3)`` of ``Ω̇_k``).
        """
        n = self._num_links
        nQ = self._num_quadrotors
        thrusts = np.zeros(nQ) if thrusts is None else np.asarray(thrusts, dtype=float).reshape(-1)
        torques = (
            np.zeros((nQ, 3))
            if torques is None
            else np.asarray(torques, dtype=float).reshape(nQ, 3)
        )

        matrix, rhs = self._assemble(thrusts)
        solution = np.linalg.solve(matrix, rhs)
        root_accel = solution[0:3] - self._ge3
        link_ang_accels = solution[3:].reshape(n, 3)

        # q̈_i = ω̇_i × q_i − ‖q̇_i‖² q_i, then ẍ_{i+1} = ẍ_i + l_i q̈_i.
        link_accels = np.zeros((n, 3))
        vertex_accels = np.zeros((n + 1, 3))
        vertex_accels[0] = root_accel
        for i in range(n):
            q_i = np.asarray(self.state.link_attitudes[i], dtype=float)
            dq_i = self.state.dq(i)
            link_accels[i] = np.cross(link_ang_accels[i], q_i) - np.dot(dq_i, dq_i) * q_i
            vertex_accels[i + 1] = vertex_accels[i] + self._link_lengths[i] * link_accels[i]

        quad_ang_accels = np.zeros((nQ, 3))
        for k in range(nQ):
            Omega = np.asarray(self.state.angular_velocities[k], dtype=float)
            quad_ang_accels[k] = self._inertia_invs[k] @ (
                torques[k] - np.cross(Omega, self._inertias[k] @ Omega)
            )

        return {
            "root_accel": root_accel,
            "link_ang_accels": link_ang_accels,
            "link_accels": link_accels,
            "vertex_accels": vertex_accels,
            "quad_ang_accels": quad_ang_accels,
        }

    def _zoh(self, thrusts: np.ndarray, torques: np.ndarray, h: float):
        """Zero-order hold Euler integration for one substep.

        The update order — root translation, then link attitudes, then the
        vehicle attitudes, then the derived vehicle poses — is M3's and M5's,
        term for term.
        """
        n = self._num_links
        omega = [np.asarray(self.state.link_angular_velocities[i]).copy() for i in range(n)]

        acc = self.accelerations(thrusts, torques)
        root_accel = acc["root_accel"]

        # root (vertex 0) translation
        self.state.root_position = (
            self.state.root_position + self.state.root_velocity * h + 0.5 * root_accel * h**2
        )
        self.state.root_velocity = self.state.root_velocity + root_accel * h

        # link attitude dynamics
        for i in range(n):
            self.state.link_attitudes[i] = self.state.link_attitudes[i].step(omega[i] * h)
            self.state.link_angular_velocities[i] += TS2(acc["link_ang_accels"][i] * h)

        # vehicle attitude dynamics
        for k in range(self._num_quadrotors):
            Omega = np.asarray(self.state.angular_velocities[k]).copy()
            self.state.orientations[k] = self.state.orientations[k].step(Omega * h)
            self.state.angular_velocities[k] += TSO3(acc["quad_ang_accels"][k] * h)

        # vehicle positions & velocities derived from the chain
        self._sync_quad_states()

    def _sync_quad_states(self):
        """Recompute vehicle positions/velocities from the chain."""
        positions = self.vertex_positions()
        velocities = self.vertex_velocities()
        for k, vertex in enumerate(self._attachments):
            self.state.positions[k] = positions[vertex]
            self.state.velocities[k] = velocities[vertex]

    # ─── Static trim ───────────────────────────────────────────────

    def hover_trim(self, thrust_split=None) -> dict:
        r"""A static equilibrium of the whole system, in closed form.

        Propagates :eq:`m6-trim`.  Given the per-vehicle thrust *vectors*
        ``F_k`` (which must sum to ``M_{00} g e_3``), the link tensions follow
        from the telescoping recursion ``\tau_i = \tau_{i-1} + \bar m_i g e_3 -
        F_i`` with ``\tau_{-1} = 0``, each link points along its own tension,
        and the balance closes automatically at the tip.

        Args:
            thrust_split: ``(nQ,)`` thrust magnitudes, all vertical, summing to
                the system weight; or ``(nQ, 3)`` thrust vectors summing to
                ``M₀₀ g e₃``.  ``None`` puts the entire weight on the **topmost**
                vehicle, which is what reduces to M5's ``f = M₀₀ g`` at
                ``nQ = 1``.

                That default is the only split guaranteed to leave every link
                in tension, and the reason is physics rather than convenience:
                in a *vertical* trim the tension in link ``i`` is the cumulative
                thrust above it minus the cumulative weight above it, so
                spreading the thrust drives that difference through zero
                somewhere.  Passing it through zero is fine — the hose simply
                folds, the links below the crossing point pointing back up, and
                an equal split gives exactly that — but landing *on* zero
                leaves a slack link with no defined direction, which is what
                the error below reports.  A split that hits it (one vehicle per
                vertex, each carrying its own weight, say) has no vertical trim
                at all; give the vehicles horizontal thrust components and the
                hose becomes a catenary with tension everywhere.

                The default also asks one vehicle for the whole system weight,
                which can exceed a single airframe's thrust limit; a warning
                says so, and :meth:`step` would clip it.

        Returns:
            A dict with ``link_attitudes`` (``(n, 3)``), ``tensions``
            (``(n, 3)``), ``thrusts`` (``(nQ,)``), ``orientations``
            (``(nQ, 3, 3)``) and ``wrench`` (``(nQ, 4)``, moments zero).

        Raises:
            ValueError: if the split does not carry the system weight, if a
                thrust is negative, or if some link ends up with zero tension —
                the link is then slack and its direction is undefined, which is
                a statement about the requested split, not about the model.
                With the default split, tension in link ``i`` is ``g S_i > 0``
                and this cannot happen.
        """
        n = self._num_links
        nQ = self._num_quadrotors
        mbar = self.vertex_total_masses
        weight = float(mbar.sum()) * self._g

        if thrust_split is None:
            forces = np.zeros((nQ, 3))
            forces[0] = weight * self._e3
        else:
            arr = np.asarray(thrust_split, dtype=float)
            if arr.shape == (nQ,):
                if np.any(arr < 0.0):
                    raise ValueError(f"thrust magnitudes must be non-negative; got {arr.tolist()}")
                forces = np.outer(arr, self._e3)
            elif arr.shape == (nQ, 3):
                forces = arr
            else:
                raise ValueError(f"thrust_split must be (nQ,) or (nQ, 3); got {arr.shape}")
            residual = forces.sum(axis=0) - weight * self._e3
            if float(np.linalg.norm(residual)) > 1e-9 * max(1.0, weight):
                raise ValueError(
                    "a static trim needs the thrusts to carry exactly the system weight "
                    f"{weight:.6g} N along e3; the split leaves a residual of {residual.tolist()}"
                )

        vertex_forces = np.zeros((n + 1, 3))
        for k, vertex in enumerate(self._attachments):
            vertex_forces[vertex] = forces[k]

        tensions = np.zeros((n, 3))
        tau = np.zeros(3)
        for i in range(n):
            tau = tau + mbar[i] * self._g * self._e3 - vertex_forces[i]
            tensions[i] = tau
        closure = tau + mbar[n] * self._g * self._e3 - vertex_forces[n]
        # Guaranteed by the weight check above; assert it rather than trust it.
        if float(np.linalg.norm(closure)) > 1e-9 * max(1.0, weight):
            raise ValueError(f"the chain does not close: residual tension {closure.tolist()}")

        norms = np.linalg.norm(tensions, axis=1)
        if np.any(norms <= 1e-12 * max(1.0, weight)):
            raise ValueError(
                "this split leaves at least one link with zero tension, so its direction is "
                f"undefined; tension norms {norms.tolist()}"
            )
        directions = tensions / norms[:, None]

        thrusts = np.linalg.norm(forces, axis=1)
        orientations = np.zeros((nQ, 3, 3))
        for k in range(nQ):
            if thrusts[k] <= 1e-12 * max(1.0, weight):
                # A vehicle carrying no load has no attitude requirement.
                orientations[k] = np.eye(3)
            else:
                orientations[k] = np.asarray(
                    SO3.from_two_vectors(forces[k] / thrusts[k], np.array([1.0, 0.0, 0.0]))
                )

        if np.any(thrusts > self._max_thrust):
            _logger.warning(
                "the trim asks for thrusts %s N but the actuator limit is %.4g N; step() "
                "would clip them and the equilibrium would not hold. Spread the load with "
                "thrust_split, or lighten the system.",
                np.array2string(thrusts, precision=3),
                self._max_thrust,
            )

        wrench = np.zeros((nQ, 4))
        wrench[:, 0] = thrusts
        return {
            "link_attitudes": directions,
            "tensions": tensions,
            "thrusts": thrusts,
            "orientations": orientations,
            "wrench": wrench,
        }

    def hover_input(self, thrust_split=None) -> np.ndarray:
        """The open-loop static-trim wrench stack, shape ``(nQ, 4)``.

        Shorthand for ``hover_trim(...)["wrench"]``; step it under the
        ``WRENCH`` input type from the state :meth:`hover_trim` describes.
        """
        return np.asarray(self.hover_trim(thrust_split)["wrench"])

    # ─── Allocation ────────────────────────────────────────────────

    def _compute_allocation_matrix(self):
        kf = DEFAULT_FORCE_CONSTANT
        km = DEFAULT_TORQUE_CONSTANT
        c = km / kf
        arm = DEFAULT_ARM_LENGTH
        ang = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        d = [-1.0, 1.0, -1.0, 1.0]

        self._alloc = np.zeros((4, 4))
        for i in range(4):
            self._alloc[0, i] = 1.0
            self._alloc[1, i] = arm * np.sin(ang[i])
            self._alloc[2, i] = -arm * np.cos(ang[i])
            self._alloc[3, i] = c * d[i]
        self._alloc_inv = np.linalg.pinv(self._alloc)

    def _wrench_to_propforces(self, wrench: np.ndarray) -> np.ndarray:
        """``(nQ, 4)`` wrenches -> ``(nQ, 4)`` propeller forces."""
        return np.asarray(wrench @ self._alloc_inv.T)

    def _propforces_to_wrench(self, prop_forces: np.ndarray) -> np.ndarray:
        """``(nQ, 4)`` propeller forces -> ``(nQ, 4)`` wrenches."""
        return np.asarray(prop_forces @ self._alloc.T)

    # ─── Input repackaging ─────────────────────────────────────────

    def _repackage_input(self, u, desired_att=None) -> np.ndarray:
        """Convert user input to a ``(nQ, 4)`` stack of ``[thrust, Mx, My, Mz]``.

        Args:
            u: ``(nQ, 3)`` thrust force vectors (``ACCELERATION``), or
                ``(nQ, 4)`` wrenches / propeller forces. Flat ``(3*nQ,)`` and
                ``(4*nQ,)`` inputs are reshaped.
            desired_att: optional per-vehicle ``(Rd, Omegad, dOmegad)``
                feedforward, one entry per vehicle.
        """
        nQ = self._num_quadrotors
        arr = np.asarray(u, dtype=float)

        if self._input_type == InputType.ACCELERATION:
            arr = arr.reshape(nQ, 3)
            wrench = np.zeros((nQ, 4))
            for i in range(nQ):
                f, M = self._att_controllers[i].compute(
                    self.t,
                    (self.state.orientations[i], self.state.angular_velocities[i]),
                    arr[i],
                    desired_att=None if desired_att is None else desired_att[i],
                )
                wrench[i] = np.array([f, *M])
        elif self._input_type == InputType.PROP_FORCES:
            wrench = self._propforces_to_wrench(arr.reshape(nQ, 4))
        else:
            wrench = arr.reshape(nQ, 4).copy()

        # Clip to actuator limits
        wrench[:, 0] = np.clip(wrench[:, 0], self._min_thrust, self._max_thrust)
        wrench[:, 1:] = np.clip(wrench[:, 1:], self._min_torque, self._max_torque)

        # Convert to prop forces if needed
        if self._force_type == ForceType.PROP_FORCES:
            props = self._wrench_to_propforces(wrench)
            props = np.clip(props, self._prop_min_force, self._prop_max_force)
            wrench = self._propforces_to_wrench(props)

        return wrench

    # ─── Step / Reset / Simulate ───────────────────────────────────

    def step(self, u, desired_att=None):
        """One outer step: repackage input, then run inner substeps."""
        wrench = self._repackage_input(u, desired_att=desired_att)
        thrusts = wrench[:, 0]
        torques = wrench[:, 1:]
        for _ in range(self._inner_loop_steps):
            self._zoh(thrusts, torques, self.h)
            self.t += self.h

    def _check_per_index(self, values, name: str, expected: int) -> list:
        """Validate a per-index ``reset()`` override and return it as a list.

        Shared with the MuJoCo backend so both raise the same message.
        """
        entries: list = list(values)
        if len(entries) != expected:
            raise ValueError(f"{name} must have {expected} entries, got {len(entries)}")
        if name == "link_attitudes":
            for i, v in enumerate(entries):
                if float(np.linalg.norm(np.asarray(v, dtype=float))) < _MIN_DIRECTION_NORM:
                    raise ValueError(
                        f"{name}[{i}] has (near-)zero norm and does not define a direction"
                    )
        return entries

    def reset(self, **kwargs):
        """Reset time and state, then apply any state overrides.

        Recognised kwargs: ``root_position``, ``root_velocity``,
        ``link_attitudes``, ``link_angular_velocities``, ``orientations``,
        ``angular_velocities``.  Per-link entries must have ``num_links``
        entries and per-vehicle ones ``num_quadrotors``.

        When ``link_attitudes`` is omitted the hose is placed in the taut spawn
        shape of :func:`udaan.utils.hose.hose_spawn_directions` — the same one
        the MJCF encodes, so the two backends start from the same geometry.

        **Off-manifold per-link input is repaired, not rejected.**  Each
        ``link_attitudes`` entry is normalised onto ``S^2`` and each
        ``link_angular_velocities`` entry is projected onto the tangent plane of
        its (already normalised) direction.  This is what the MuJoCo backend has
        to do anyway — a ball joint cannot represent a stretched link or a spin
        about the link axis — so doing it here is what makes the two backends
        land on the *same* state from the same kwargs.  Repair rather than
        rejection because the spin component is genuine gauge freedom the
        dynamics already factor out.  A zero-length direction *is* rejected:
        there is nothing to normalise.
        """
        self.t = 0.0
        self.state.reset()
        self.state.link_attitudes = self.spawn_link_attitudes()

        if "link_attitudes" in kwargs:
            values = self._check_per_index(
                kwargs["link_attitudes"], "link_attitudes", self._num_links
            )
            # Rewrap every entry: the integrator mutates the tangent wrappers in
            # place, so a caller's object must not be aliased. S2 does not
            # normalise on construction (repo-wide), so do it here.
            self.state.link_attitudes = [
                S2(np.asarray(v, dtype=float) / np.linalg.norm(np.asarray(v, dtype=float)))
                for v in values
            ]
        if "link_angular_velocities" in kwargs:
            values = self._check_per_index(
                kwargs["link_angular_velocities"], "link_angular_velocities", self._num_links
            )
            rates = []
            for i, v in enumerate(values):
                q_i = np.asarray(self.state.link_attitudes[i], dtype=float)
                omega_i = np.asarray(v, dtype=float)
                rates.append(TS2(omega_i - float(np.dot(omega_i, q_i)) * q_i))
            self.state.link_angular_velocities = rates

        for key, wrap in (("orientations", SO3), ("angular_velocities", TSO3)):
            if key in kwargs:
                values = self._check_per_index(kwargs[key], key, self._num_quadrotors)
                setattr(self.state, key, [wrap(np.asarray(v)) for v in values])

        for key in ("root_position", "root_velocity"):
            if key in kwargs:
                # Copy: a caller-supplied array must never be aliased into the
                # state (the M2 bug that silently mutated shared constants).
                setattr(self.state, key, np.array(kwargs[key], dtype=float))

        self._sync_quad_states()
        self._station_targets = [p.copy() for p in self.state.positions]

    def compute_default_input(self) -> np.ndarray:
        """Per-vehicle thrust force vectors from the station-keeping PDs.

        Every vehicle holds the vertex :meth:`reset` left it at and lets the
        hose swing between them.  Returns ``(nQ, 3)``, suitable for
        :meth:`step` under the default ``ACCELERATION`` input type.
        """
        return np.array(
            [
                self._pos_controllers[i].compute(
                    self.t, (self.state.positions[i], self.state.velocities[i])
                )
                for i in range(self._num_quadrotors)
            ]
        )

    def compute_hose_input(self, reference=None) -> np.ndarray:
        """The ``(nQ, 4)`` wrench stack from the M6 LQR.

        Args:
            reference: optional ``(RefState, Inputs)`` pair from
                :class:`udaan.flatness.MultiQuadHose`.  When ``None`` the
                controller regulates against the static trim at its setpoint,
                which is the feedback-only law.

        Suitable for :meth:`step` under the ``WRENCH`` input type.

        Raises:
            RuntimeError: if there is no hose controller — see
                :attr:`hose_controller`.
        """
        if self._hose_controller is None:
            raise RuntimeError(
                f"this hose has no vehicle at its last vertex (attachments "
                f"{self._attachments} on {self._num_links} links), so it is not "
                "differentially flat in the coordinates the LQR's reference is built "
                f"in and no hose controller was created. Add vertex {self._num_links} "
                "to the attachment set, or drive the ACCELERATION pipeline's "
                "station-keeping loop instead."
            )
        if reference is None:
            return np.asarray(self._hose_controller.compute(self.t, self.state))
        return np.asarray(self._hose_controller.compute(self.t, self.state, reference))

    # ─── Flatness feedforward ──────────────────────────────────────

    def _flatness_feedforward_fn(self):
        """``(jet_fn, exact)`` for the bound root trajectory, or ``(None, False)``.

        The hose controller's ``setpoint`` is sniffed for a bound method, and
        its owner for a way to differentiate the trajectory deeply enough.  As
        in M5, "deeply enough" is not a constant: the flat map needs the root
        position through its ``2n + 4``-th derivative
        (:func:`udaan.flatness.chain.chain_jet_order`), so ``get_full``'s five
        rows — position through snap — stop being sufficient at ``n = 1``.

        Two protocols, in order of preference:

        * ``get_jet(t, order) -> (order + 1, 3)``, which supplies every
          derivative in closed form.  This is the only one that feeds the map
          what it actually asks for, and it is what
          :class:`~udaan.utils.trajectory.CrazyTrajectory` implements.
        * ``get_full(t) -> (pos, vel, acc, jerk, snap)``, zero-padded past snap.
          :meth:`simulate` warns once when it has to fall back on this.
        """
        setpoint_fn = getattr(self._hose_controller, "setpoint", None)
        owner = getattr(setpoint_fn, "__self__", None)
        if owner is None:
            return None, False
        get_jet = getattr(owner, "get_jet", None)
        if get_jet is not None:
            return get_jet, True
        get_full = getattr(owner, "get_full", None)
        if get_full is None:
            return None, False

        def zero_padded(t, order):
            rows = [np.asarray(row, dtype=float) for row in get_full(t)]
            rows.extend(np.zeros(3) for _ in range(order + 1 - len(rows)))
            return np.array(rows[: order + 1])

        return zero_padded, False

    @property
    def feedforward_drops(self) -> int:
        """How many steps have dropped the flatness feedforward.

        Reset by :meth:`simulate`.  A non-zero count means the reference left
        the flat map's certificate on those steps — a derived link tension fell
        through :attr:`~udaan.flatness.MultiQuadHose.min_tension`, a link went
        slack, or a vehicle was in free fall — and the step ran against the
        static trim instead.  The first drop is logged as a warning, the rest at
        debug level.
        """
        return self._feedforward_drops

    def _flat_reference(self, flat_map, controller, jet_fn, t):
        """``(RefState, Inputs)`` recovered from the flat output at ``t``, or ``None``.

        ``None`` is a legitimate outcome, not an error: the map refuses any
        sample whose hose is not taut, and the right response is to fall back on
        the trim reference for that step rather than to abort.  It is counted in
        :attr:`feedforward_drops`.

        A malformed jet — wrong shape, wrong dimension — is a programming error
        and is left to propagate.
        """
        flats = controller.build_flats(np.asarray(jet_fn(t, flat_map.jet_order), dtype=float))
        try:
            return flat_map.recover(flats)
        except ValueError as err:
            self._feedforward_drops += 1
            if self._feedforward_drops == 1:
                _logger.warning(
                    "flatness feedforward dropped at t=%.4f (%s); this step ran "
                    "against the static-trim reference. Further drops are logged at "
                    "debug level; see the feedforward_drops counter for the total.",
                    t,
                    err,
                )
            else:
                _logger.debug("flatness feedforward unavailable at t=%.4f: %s", t, err)
            return None

    def simulate(self, tf, **kwargs):
        """Run closed-loop simulation to time ``tf``.

        Which loop runs is decided by the input type, because the three
        pipelines take incompatible input shapes:

        * ``ACCELERATION`` (the default) drives the station-keeping cascade of
          :meth:`compute_default_input` — one position PD per vehicle emitting
          a thrust *vector*, which is what that pipeline's geometric attitude
          loop consumes.
        * ``WRENCH`` drives the M6 LQR (:attr:`hose_controller`), which emits
          the body moments itself, so no attitude loop is in the path.

        The two pipelines differ in what "no setpoint" means, and it is worth
        knowing before reading a run.  The ``ACCELERATION`` cascade station-keeps
        wherever :meth:`reset` left each vehicle.  The LQR inherits
        :class:`~udaan.control.PDController`'s default setpoint, which is the
        **fixed point** ``(0, 0, 1)`` — so ``simulate(tf=...,
        root_position=[0, 0, 3])`` under ``WRENCH`` with nothing bound flies the
        root *down* to ``z = 1`` rather than holding it.  Bind
        ``mdl.hose_controller.setpoint`` to say otherwise.

        Under ``WRENCH``, binding a trajectory to
        ``mdl.hose_controller.setpoint`` also picks up the **flatness
        feedforward**: :class:`udaan.flatness.MultiQuadHose` recovers the whole
        reference state *and* the reference wrenches, and both are handed to the
        controller — the reference the error is measured against, and the wrench
        the correction is added to.

        The flat map is taken from the controller's **own**
        :attr:`~udaan.control.multi_quad_hose.MultiQuadHoseController.flat_map`,
        never rebuilt from this model.  The gain was solved at the controller's
        own masses and lengths; a reference recovered from different ones is the
        reference for a different machine, and the mismatch shows up as a
        steady-state offset with no other symptom.  A controller that does not
        publish a flat map therefore gets no feedforward at all.
        """
        self.reset(**kwargs)
        self._feedforward_drops = 0

        lqr = self._input_type == InputType.WRENCH
        flat_map = None
        jet_fn = None
        if lqr:
            if self._hose_controller is None:
                # Same condition compute_hose_input refuses on; say it once here
                # rather than once per step.
                self.compute_hose_input()
            jet_fn, exact = self._flatness_feedforward_fn()
            if jet_fn is not None:
                flat_map = getattr(self._hose_controller, "flat_map", None)
                if flat_map is None:
                    _logger.warning(
                        "%s exposes no flat_map, so the reference this loop is "
                        "regulating against is unknown; running feedback-only rather "
                        "than feeding forward a reference built from the model's own "
                        "parameters, which need not be the controller's.",
                        type(self._hose_controller).__name__,
                    )
                elif not exact:
                    _logger.warning(
                        "the bound trajectory supplies derivatives only through snap, "
                        "but a %d-link hose needs the root position through its %d-th "
                        "derivative (2n + 4). Orders 5-%d are being taken as zero: the "
                        "recovered positions and link directions are unaffected, the "
                        "recovered rates and moment feedforward are not. A trajectory "
                        "exposing get_jet(t, order) supplies them exactly.",
                        self._num_links,
                        flat_map.jet_order,
                        flat_map.jet_order,
                    )

        start_t = time.time_ns()
        while self.t < tf:
            if lqr:
                reference = (
                    None
                    if flat_map is None
                    else self._flat_reference(flat_map, self._hose_controller, jet_fn, self.t)
                )
                u = self.compute_hose_input(reference=reference)
            else:
                u = self.compute_default_input()
            if self.verbose >= 2:
                self._log_state(u)
            self.step(u)

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _log_state(self, u):
        wrench = self._repackage_input(u)
        _logger.debug(
            "t=%.4f x0=%s q0=%s qn=%s thrust=%s",
            self.t,
            np.array2string(np.asarray(self.state.root_position), precision=4),
            np.array2string(np.asarray(self.state.link_attitudes[0]), precision=4),
            np.array2string(np.asarray(self.state.link_attitudes[-1]), precision=4),
            np.array2string(wrench[:, 0], precision=4),
        )
