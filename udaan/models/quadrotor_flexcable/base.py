r"""QuadrotorFlexCableBase — pure dynamics and state, no rendering.

One rigid-body quadrotor carrying a payload through a **flexible cable modelled
as a serial chain of ``n`` rigid links**, integrated with ZOH (zero-order hold)
Euler.

This is M5.  It is M2 (:mod:`udaan.models.quadrotor_cspayload`) with the single
taut cable replaced by a chain: at ``n = 1`` the two are the *same* equations,
and ``tests/test_quadrotor_flexcable_base.py`` asserts that to round-off.

.. warning::

   **This module is a derivation, not a port.**  The legacy reference
   ``phd_research/quadrotor-flexible-cables-payload/Single-Quad-Flex-Cable/
   main_quad_flexible.m::odefun_singleQuadFlex`` (L226-354) was re-derived from
   the Lagrangian rather than transcribed — three earlier steps of this port
   found real defects in that repository.  This particular routine survived the
   re-derivation intact; the two deliberate departures from it are recorded
   under "Departures from the reference" below.

Configuration
-------------
The configuration manifold is

.. math::

    Q = \mathbb{R}^3 \times SO(3) \times (S^2)^{n},

and the integrated state is

.. math::

    (x_L,\; v_L,\; R,\; \Omega,\; \{q_i, \omega_i\}_{i=1}^{n}),

which is ``18 + 6n`` stored floats (``R`` is carried as a 3x3 matrix) or
``12 + 6n`` manifold dimensions.  The ``12 + 6n`` figure quoted in
``RESEARCH_MAP.md`` is the *error/variational* state an LQR linearisation uses,
not this one.

* Payload: position ``x_L``, velocity ``v_L``.  The payload is the point mass
  at the **distal end of the last link**.
* Link ``i``: direction ``q_i \in S^2``, angular velocity
  ``\omega_i \in T_{q_i}S^2`` with ``\dot q_i = \omega_i \times q_i``, length
  ``l_i``, and a point mass ``m_i`` lumped at its own **distal** end.
* Quadrotor: attitude ``R``, body rate ``\Omega``, mass ``m_Q``, inertia ``J``,
  thrust vector ``u = f R e_3`` and body moment ``M``.

Ordering and sign conventions
-----------------------------
Two choices, both made for internal consistency with M1-M4 rather than for
fidelity to the reference:

* **Link 0 is at the quadrotor, link n-1 at the payload**, and ``q_i`` points
  from its own proximal joint to its own distal joint — i.e. *from the
  quadrotor towards the load*, the same direction :mod:`udaan.flatness.cable`,
  :mod:`udaan.models.quadrotor_cspayload` and
  :mod:`udaan.models.multi_quad_cspayload` use.  The vertex positions are

  .. math::

      x_i = x_Q + \sum_{k \le i} l_k q_k, \qquad
      x_L = x_n = x_Q + \sum_{k=1}^{n} l_k q_k .

  This is also the reference's sign: its first block row carries
  ``-M_{0i}\hat q_i`` and its gravity term ``-M_{0i} g \hat q_i e_3``, both of
  which flip if ``q_i`` is taken the other way.

* **The payload is integrated and the quadrotor is derived**:

  .. math::

      x_Q = x_L - \sum_i l_i q_i, \qquad v_Q = v_L - \sum_i l_i \dot q_i .

  The reference integrates ``(x_L, v_L)`` too, and so do M2, M3 and M4 — so
  here the reference and the codebase agree and there is nothing to trade off.
  Note the *solve* below still produces ``a_Q`` first; ``a_L`` is recursed
  forward from it, exactly as the reference does at its L322-325.

Derivation
----------
Write ``m_Q`` for the quadrotor mass and lump the chain's masses as

.. math::
    :label: m5-coefficients

    M_{00} = m_Q + \sum_{k=1}^{n} m_k, \qquad
    M_{0i} = M_{i0} = \Bigl(\sum_{k \ge i} m_k\Bigr) l_i, \qquad
    M_{ij} = \Bigl(\sum_{k \ge \max(i,j)} m_k\Bigr) l_i l_j .

These are not postulated — they *are* the kinetic energy.  Substituting
``x_i = x_Q + \sum_{k\le i} l_k q_k`` into
``T = \tfrac12 m_Q \|\dot x_Q\|^2 + \tfrac12 \sum_i m_i \|\dot x_i\|^2`` and
collecting terms gives

.. math::

    T = \tfrac12 M_{00}\|\dot x_Q\|^2
        + \sum_k M_{0k}\, \dot x_Q \cdot \dot q_k
        + \tfrac12 \sum_{k,j} M_{kj}\, \dot q_k \cdot \dot q_j
        + \tfrac12 \Omega^\top J \Omega ,

with :eq:`m5-coefficients` read straight off, and

.. math::

    V = g\, e_3 \cdot \Bigl( M_{00} x_Q + \sum_k M_{0k} q_k \Bigr).

**Euler-Lagrange on** :math:`\mathbb{R}^3 \times (S^2)^n`.  The ``x_Q``
equation is ordinary; for each ``q_i`` the constrained variation
``\delta q_i = \xi_i \times q_i`` with ``\xi_i \perp q_i`` reduces the
stationarity condition to

.. math::

    \hat q_i \Bigl[ \tfrac{d}{dt}\tfrac{\partial L}{\partial \dot q_i}
                    - \tfrac{\partial L}{\partial q_i} \Bigr] = 0 .

Using ``\ddot q_j = \dot\omega_j \times q_j - c_j q_j`` with
``c_j := \|\dot q_j\|^2`` (see "Departures" below) and
``\alpha := \ddot x_Q + g e_3`` — gravity then cancels identically from every
link row — the system is

.. math::
    :label: m5-system

    \begin{bmatrix}
      M_{00} I     & -M_{01}\hat q_1        & \cdots & -M_{0n}\hat q_n \\
      M_{01}\hat q_1 & M_{11} I             & \cdots & -M_{1n}\hat q_1 \hat q_n \\
      \vdots       &                        & \ddots & \vdots \\
      M_{0n}\hat q_n & -M_{n1}\hat q_n \hat q_1 & \cdots & M_{nn} I
    \end{bmatrix}
    \begin{bmatrix} \alpha \\ \dot\omega_1 \\ \vdots \\ \dot\omega_n \end{bmatrix}
    =
    \begin{bmatrix}
      f R e_3 + \sum_k M_{0k} c_k q_k \\
      \sum_{j \ne 1} M_{1j} c_j\, \hat q_1 q_j \\
      \vdots \\
      \sum_{j \ne n} M_{nj} c_j\, \hat q_n q_j
    \end{bmatrix}.

The matrix is symmetric (``\hat q^\top = -\hat q`` flips both off-diagonal
blocks onto each other) and ``(3 + 3n) \times (3 + 3n)`` — structurally the
same shape as M3's ``3 + 3 n_Q`` and M4's ``6 + 3 n_Q``.  Then

.. math::

    \ddot x_Q = \alpha - g e_3, \qquad
    \ddot x_L = \ddot x_Q + \sum_i l_i \bigl(\dot\omega_i \times q_i - c_i q_i\bigr),

and the quadrotor attitude is unchanged from M1:
``\dot R = R \hat\Omega``, ``J \dot\Omega = M - \Omega \times J\Omega``.

**Why the diagonal block is** ``M_{ii} I`` **and not** ``M_{ii} P_i^\perp``.
The Lagrangian block is ``-M_{ii}\hat q_i \hat q_i = M_{ii}(I - q_i q_i^\top)``,
which is rank 2: ``\dot\omega_i`` along ``q_i`` does not appear in the dynamics
at all (it is the gauge freedom of the ``S^2`` parametrisation).  Replacing it
by ``M_{ii} I`` adds ``M_{ii} q_i q_i^\top`` to that block and *nothing else*
— the ``q_i`` component of link row ``i`` is otherwise identically zero, since
every one of its terms is a cross product with ``q_i``.  So the substitution is
not an approximation: it appends the gauge-fixing equation
``\dot\omega_i \cdot q_i = 0`` (the exact tangency condition, because
``\tfrac{d}{dt}(q_i\cdot\omega_i) = \dot\omega_i \cdot q_i``) and turns an
``n``-fold singular matrix into a positive definite one.  The reference makes
the same substitution.

**Invertibility.**  With the Lagrangian diagonal the matrix is the Hessian of
``T`` in the coordinates ``(\dot x_Q, \dot q_1, \dots)`` pulled back through
``\dot q_i = -\hat q_i \omega_i``, hence positive semi-definite with kernel
exactly ``\{(0, \dots, \omega_i \parallel q_i, \dots)\}``; adding
``\sum_i M_{ii} q_i q_i^\top`` is strictly positive on that kernel.  So
:eq:`m5-system` is **positive definite whenever** ``m_Q > 0`` and every
``m_i > 0``, for *every* link configuration — all links parallel, ``n = 1``,
arbitrarily unequal masses and lengths.  The geometry never makes it singular.

Does M4's parallel/perpendicular elimination apply?
---------------------------------------------------
**No, and the reason is structural.**  In M2/M3/M4 each cable joins the payload
directly to one quadrotor, so ``\dot\omega_i`` appears only in link row ``i``
and in the payload row.  Projecting the *quadrotor* equation onto ``q_i`` then
annihilates ``\dot\omega_i`` and yields the scalar tension ``t_i`` as an affine
function of the payload accelerations alone, which is what collapses the system
to ``3 \times 3`` (M3) or ``6 \times 6`` (M4) at any number of vehicles.

Here the links are a *serial chain*: the constraint force in joint ``i`` is
transmitted through every downstream link, and link row ``i`` couples to link
row ``j`` through ``-M_{ij}\hat q_i \hat q_j`` for every ``j \ne i``.
Projecting link row ``i`` onto ``q_i`` annihilates **every** term, not just
``\dot\omega_i`` — each one is a ``\hat q_i(\cdot)`` — so it returns the gauge
condition ``\dot\omega_i \cdot q_i = 0`` and no information whatsoever about
``\alpha``.  Equivalently: before the ``M_{ii} I`` gauge fix the system has
rank exactly ``3 + 2n``, each link row losing exactly one equation to its own
gauge direction, and the lost equation carries nothing about ``\alpha``.  **No
per-link projection decouples the chain**, so M4's elimination has no analogue
here and the dense ``(3 + 3n)`` solve is the natural direct form.

That is a statement about *this* family of eliminations, not a lower bound on
the problem.  A serial chain is exactly the structure the standard articulated-
body / recursive Newton-Euler algorithms exploit, and they solve it in ``O(n)``
instead of the ``O(n^3)`` a dense factorisation costs.  Not taken here: at the
link counts the ZOH-Euler stability limit permits (``n \lesssim 5``, see
"Numerical stability") the dense solve is not the bottleneck, and one assembly
path that reads like the Lagrangian is worth more than a faster one that does
not.

One reduction *is* available and is deliberately not taken: the ``(0,0)`` block
is ``M_{00} I``, a scalar multiple of the identity, so ``\alpha`` can be
eliminated in closed form to leave a ``3n \times 3n`` system in ``\dot\omega``
with blocks ``M_{ij}' = (M_{0i}M_{0j}/M_{00} - M_{ij})\hat q_i \hat q_j``
off-diagonal.  That saves three unknowns out of ``3 + 3n`` — 50% at ``n = 1``
and nothing asymptotically — in exchange for a second assembly path that would
have to be kept in agreement with this one.  Not worth it; the single solve
stays.

Departures from the reference
-----------------------------
Two, both deliberate:

* **The centripetal coefficient is** ``c_i = \|\dot q_i\|^2 = \|\omega_i \times
  q_i\|^2``, where the reference writes ``\|\omega_i\|^2``.  Both are the same
  number on the tangent manifold and neither is the exact second derivative off
  it.  Differentiating ``\dot q_i = \omega_i \times q_i`` twice gives

  .. math::

      \ddot q_i = \dot\omega_i \times q_i
                  + (\omega_i\cdot q_i)\,\omega_i - \|\omega_i\|^2 q_i ,

  whereas :eq:`m5-system` is assembled from
  ``\ddot q_i = \dot\omega_i \times q_i - c_i q_i``.  The difference is

  .. math::

      \ddot q_i^{\text{exact}} - \ddot q_i^{\text{used}}
        = (\omega_i\cdot q_i)\bigl[\omega_i - (\omega_i\cdot q_i) q_i\bigr] ,

  which vanishes identically when ``\omega_i \perp q_i``.  **So the form used is
  exact on the tangent manifold ``T_{q_i}S^2`` and wrong off it at first order
  in the tangency residual** ``\omega_i \cdot q_i``.

  Note what the frequently-cited identity ``q_i \cdot \ddot q_i = -\|\dot
  q_i\|^2`` does and does not establish: the dropped term above is orthogonal
  to ``q_i`` (dot it with ``q_i`` and the two pieces cancel), so the identity
  pins only the **radial** component of ``\ddot q_i`` and is blind to the whole
  discrepancy.  It justifies the *coefficient* ``c_i = \|\dot q_i\|^2`` given
  the ``- c_i q_i`` ansatz; it does not justify the ansatz.

  This is a modelling choice, not an oversight.  The tangency residual is a
  pure integration artefact — the continuous dynamics conserve
  ``q_i \cdot \omega_i = 0`` exactly (the ``M_{ii} I`` gauge row makes
  ``\dot\omega_i \cdot q_i = 0`` an *equation of the solve*), and ZOH-Euler
  lets it drift at ``O(h)`` — so the dropped term is ``O(h)``, the same order
  as the integrator's own error and the same trade M2, M3 and M4 make.  At the
  worst tangency drift measured in
  ``tests/test_quadrotor_flexcable_base.py::test_link_rates_stay_tangent``
  (``q_i\cdot\omega_i \approx 0.21`` at ``n = 3``, ``dt = 5`` ms) the dropped
  term is ~0.2 rad/s² per link, and it falls with ``h`` like everything else.
  Choosing ``\|\dot q_i\|^2`` over the reference's ``\|\omega_i\|^2`` is what
  keeps the ``n = 1`` reduction to M2 exact rather than merely close, since M2,
  M3 and M4 all use ``\|\dot q_i\|^2`` with the same ansatz.
* **The gravity term is folded into** ``\alpha``.  Algebraically identical
  (the ``-M_{0i} g \hat q_i e_3`` on the reference's link-row right-hand side
  cancels against the ``g e_3`` inside ``\alpha`` exactly), but it makes the
  cancellation visible instead of relying on two sign errors not happening.

Numerical stability
-------------------
M5 is the first system in this codebase whose *discretisation* has a stability
limit worth stating, and it is a property of the chain rather than a defect of
the algebra.  Linearising about the hanging equilibrium (see
:meth:`QuadrotorFlexCableBase.natural_frequencies`) gives modes solving
``\mathrm{diag}(0, g M_{0k})\phi = \lambda M \phi``; their top frequency grows
with the number of links roughly like ``n^{3/2}`` and with the payload-to-link
mass ratio like ``\sqrt{m_L/m_i}``.  At the shared 1.25 ms substep, a 1 m cable
of 10 g discretised into five links already carries a 135 rad/s mode.

Forward Euler on an undamped oscillator multiplies amplitudes by
``\sqrt{1 + (\omega h)^2}`` per substep, i.e. ``e^{\omega^2 h / 2}`` per second
of simulation, so a taut chain past that point winds up and eventually
overflows.  :meth:`QuadrotorFlexCableBase.stability_growth_rate` returns that
exponent and the constructor warns above 1.0 — calibrated against measurement:
0.45 stays bounded over 3 s, 1.46 winds to ~20 rad/s, 3.4 reaches ``nan``.

Four things follow, and none of them is a bug:

* The defaults (:data:`~udaan.core.defaults.DEFAULT_NUM_CABLE_LINKS` = 3,
  :data:`~udaan.core.defaults.DEFAULT_CABLE_MASS` = 0.1 kg) sit at 0.45, with
  roughly a factor of two of margin.  Both constants are *new with M5*: neither
  existed before it, so no pre-existing model's physics moved when they were
  added.
* **The warning is a function of the whole configuration, not of the cable
  mass alone, and it fires at large ``n`` even on the defaults.**  Holding the
  default 0.1 kg cable and raising only ``num_links`` gives 0.008 (n=1), 0.45
  (n=3), 1.46 (n=5) and 6.39 (n=10) — so ``QuadrotorFlexCableBase(num_links=10)``
  warns straight out of the constructor.  That is the intended behaviour, not a
  miscalibration: 6.39 is well past the measured transition, a chain that stiff
  really does wind up under ZOH-Euler, and the remedy is the MuJoCo backend (or
  a smaller ``dt``), not a quieter threshold.
* A **slack** chain has no limit at all.  Under zero thrust the whole system is
  in free fall, the links carry no tension and every internal frequency is
  zero — which is why the energy and momentum checks below run at any ``n``
  while the closed-loop ones do not.
* The MuJoCo backend has no such restriction: it integrates the same
  continuous dynamics with its own solver.  A configuration this model warns
  about is one to run there.

Static trim
-----------
Setting every derivative and rate to zero in :eq:`m5-system` gives, from the
link rows, ``\hat q_i e_3 = 0``, i.e. every link vertical; the hanging branch
is ``q_i = -e_3``.  The quadrotor row then reads

.. math::

    f R e_3 = M_{00} g e_3
    \quad\Longrightarrow\quad
    R = I, \qquad f = \Bigl(m_Q + \sum_k m_k\Bigr) g, \qquad M = 0 ,

i.e. the thrust carries the whole system weight and the moment is zero.  This
is derived here from the equations of motion, but ``hover_trim`` in
``tests/test_quadrotor_flexcable_base.py`` re-derives it from Newton's laws
without reading the model at all.

Validation
----------
Six independent checks, all exercised by
``tests/test_quadrotor_flexcable_base.py``:

1. **Reduction to M2 at ``n = 1``** — the strongest of them, and the only one
   comparing against code verified some other way.
2. **Hover trim** from first principles, applied open loop.
3. **Energy** under zero input, converging at ``O(h)``.
4. **Linear and angular momentum** under gravity alone.
5. **Conditioning** of :meth:`QuadrotorFlexCableBase.mass_matrix` on degenerate
   geometries.
6. **Invariants**: ``\|q_i\| = 1`` preserved; ``q_i \cdot \omega_i`` drifting at
   ``O(h)``, the same rate M2/M3/M4 do and for the same reason.

Dynamics reference:
Farhad A. Goodarzi, Daewon Lee and Taeyoung Lee. "Geometric Control of a
Quadrotor UAV Transporting a Payload Connected via Flexible Cable."
American Control Conference, 2014.
http://www.seas.gwu.edu/~tylee/paper/ACC14.1.pdf — the paper the legacy
``main_quad_flexible.m`` names in its own header. The equations above were
re-derived from the Lagrangian rather than transcribed from either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...control.quadrotor import (
    DirectPropellerForceController,
    GeometricAttitudeController,
    PositionPDController,
)
from ...control.quadrotor_flexcable import QuadrotorFlexCableController
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_MASS,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PROP_MAX_FORCE,
    DEFAULT_PROP_MIN_FORCE,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...core.validation import check_positive
from ...flatness.jet import Jet
from ...flatness.quadrotor_flexcable import QuadrotorFlexCableFlats
from ...manif import S2, SO3, TS2, TSO3, hat, zoh_step
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs

_logger = get_logger(__name__)

__all__ = [
    "QuadrotorFlexCableBase",
    "QuadrotorFlexCableState",
]

#: Below this mass (kg) a body is warned about as unphysical.  The dynamics
#: stay well posed — the guard that matters is strict positivity — but a link
#: this light is almost certainly a typo or a unit error.
_MIN_PHYSICAL_MASS = 1e-9

#: A ``reset()`` link direction shorter than this cannot be normalised onto
#: ``S^2`` and is rejected instead of being repaired.
_MIN_DIRECTION_NORM = 1e-12

#: Per-second amplitude growth exponent above which the constructor warns that
#: the chain is too stiff for the ZOH-Euler integrator. Calibrated against
#: measurement, not theory: over a 3 s taut run at ``dt = 5 ms`` a chain at
#: 0.45 stays bounded (peak ``‖ω_i‖`` ~ 0.4 rad/s), one at 1.46 winds up to
#: ~20 rad/s, and one at 3.4 overflows to ``nan``.  See
#: :meth:`QuadrotorFlexCableBase.stability_growth_rate`.
_STABILITY_GROWTH_WARN = 1.0


@dataclass
class QuadrotorFlexCableState:
    """State of one quadrotor carrying a payload on an ``n``-link cable.

    Per-link entries are lists of length ``num_links``, ordered **from the
    quadrotor down to the payload**: ``link_attitudes[0]`` is the link hinged
    at the airframe and ``link_attitudes[-1]`` the one the payload hangs from.

    ``position``/``velocity`` (the quadrotor's) are *derived* from the payload
    and the link directions in the analytical model — the links are rigid and
    inextensible — via ``x_Q = x_L - Σ l_i q_i``; the MuJoCo backend overwrites
    them with the measured body state.
    """

    num_links: int = DEFAULT_NUM_CABLE_LINKS

    # Payload position and velocity — the integrated translational state.
    payload_position: Vec3 = field(default_factory=lambda: np.zeros(3))
    payload_velocity: Vec3 = field(default_factory=lambda: np.zeros(3))

    # Quadrotor position and velocity (derived from payload + link chain)
    position: Vec3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vec3 = field(default_factory=lambda: np.zeros(3))

    # Quadrotor orientation and body angular velocity
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)

    # Per-link direction and angular velocity
    link_attitudes: list[S2] = field(default_factory=list)
    link_angular_velocities: list[TS2] = field(default_factory=list)

    def __post_init__(self):
        if self.num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {self.num_links}")
        if not self.link_attitudes:
            self.reset()

    def reset(self):
        """Reset every field: payload, quadrotor *and* every link entry."""
        n = self.num_links
        self.payload_position = np.zeros(3)
        self.payload_velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = SO3()
        self.angular_velocity = TSO3()
        self.link_attitudes = [S2(np.array([0.0, 0.0, -1.0])) for _ in range(n)]
        self.link_angular_velocities = [TS2() for _ in range(n)]

    def dq(self, i: int) -> np.ndarray:
        """Link direction rate ``q̇_i = ω_i × q_i`` of link ``i``."""
        return np.cross(
            np.asarray(self.link_angular_velocities[i]),
            np.asarray(self.link_attitudes[i]),
        )


#: The initial-condition overrides :meth:`QuadrotorFlexCableBase.reset` acts on.
#: Anything else is rejected rather than silently discarded — see
#: :mod:`udaan.models._kwargs`.
_SUPPORTED_RESET_KWARGS = frozenset(
    {
        "position",
        "velocity",
        "payload_position",
        "payload_velocity",
        "orientation",
        "angular_velocity",
        "link_attitudes",
        "link_angular_velocities",
    }
)


class QuadrotorFlexCableBase:
    """One quadrotor, an ``n``-link flexible cable, one payload — pure dynamics.

    The quadrotor is a rigid body: the input is a thrust magnitude and a body
    moment (or an acceleration, or propeller forces — see
    :meth:`_repackage_input`), not a free force vector.  Link masses and
    lengths are per-index and accept a scalar (broadcast to every link) or a
    length-``num_links`` sequence.

    Mass bookkeeping.  ``link_masses[i]`` is the point mass lumped at the
    **distal** end of link ``i``, so ``link_masses[-1]`` carries the payload.
    The constructor offers ``payload_mass``/``cable_mass`` as a convenience
    that builds that array — ``cable_mass`` spread evenly over the links and
    ``payload_mass`` added to the last — but ``link_masses`` is the parameter,
    and passing it directly overrides both.

    Usage::

        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.link_lengths = [0.3, 0.2, 0.1]
        # simulate() drives the payload LQR, which regulates the *payload*
        # onto its setpoint — so say where that is, or the run flies to the
        # default [0, 0, 1] regardless of where it started.
        target = np.array([0.0, 0.0, 1.5])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=5.0, payload_position=np.array([0.0, 0.0, 1.0]))

    Keep an eye on :meth:`stability_growth_rate` when raising ``num_links``:
    the ZOH-Euler loop is only conditionally stable, and the default chain
    leaves the stable region between ``n = 3`` (0.45) and ``n = 5`` (1.46).
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self._num_links = int(kwargs.get("num_links", DEFAULT_NUM_CABLE_LINKS))
        if self._num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {self._num_links}")
        n = self._num_links

        # The parameter setters push their new values into the controllers, and
        # the constructor runs several of them *before* those controllers exist.
        self._controllers_ready = False

        #: Steps on which the flatness feedforward was refused; see
        #: :attr:`feedforward_drops`.
        self._feedforward_drops = 0

        self.state = QuadrotorFlexCableState(num_links=n)
        self.t = 0.0
        self.verbose = kwargs.get("verbose", 0)

        # Quadrotor parameters
        self._mass = float(kwargs.get("mass", DEFAULT_QUAD_MASS))
        self._check_positive(self._mass, "mass")
        self._inertia = DEFAULT_QUAD_INERTIA.copy()
        self._inertia_inv = np.linalg.inv(self._inertia)
        if "inertia" in kwargs:
            self.inertia = kwargs["inertia"]

        # Chain parameters
        total_length = float(kwargs.get("cable_length", DEFAULT_CABLE_LENGTH))
        self._link_lengths = np.full(n, total_length / n)
        cable_mass = float(kwargs.get("cable_mass", DEFAULT_CABLE_MASS))
        payload_mass = float(kwargs.get("payload_mass", DEFAULT_PAYLOAD_MASS))
        link_masses = np.full(n, cable_mass / n)
        link_masses[-1] += payload_mass
        self._link_masses = link_masses

        if "link_lengths" in kwargs:
            self.link_lengths = kwargs["link_lengths"]
        if "link_masses" in kwargs:
            self.link_masses = kwargs["link_masses"]
        self._check_positive(self._link_lengths, "link_lengths")
        self._check_positive(self._link_masses, "link_masses")

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits
        self._min_thrust = DEFAULT_MIN_THRUST
        self._max_thrust = DEFAULT_MAX_THRUST
        self._min_torque = DEFAULT_MIN_TORQUE.copy()
        self._max_torque = DEFAULT_MAX_TORQUE.copy()
        self._prop_min_force = DEFAULT_PROP_MIN_FORCE
        self._prop_max_force = DEFAULT_PROP_MAX_FORCE

        # Integration: 5ms outer loop, 4 inner substeps at 1.25ms
        self.dt = kwargs.get("dt", 1.0 / 200.0)
        self._inner_loop_steps = kwargs.get("inner_loop_steps", 4)
        self.h = self.dt / self._inner_loop_steps

        # Input/force type.  M5's default cascade is an LQR on the quadrotor's
        # own inputs, so it emits a wrench and not the thrust *vector* M1-M4's
        # cascades emit — hence the WRENCH default, unlike every model before
        # it.  ``input="acceleration"`` selects the thrust-vector pipeline,
        # which is what :meth:`compute_station_keeping_input` produces.
        #
        # An unrecognised value is refused rather than silently resolved to
        # WRENCH: the three pipelines take three incompatible input *shapes*, so
        # a typo that falls through to the default does not mistune the loop, it
        # runs a different one.
        self._input_type = InputType.WRENCH
        self._force_type = ForceType.WRENCH
        _INPUT_TYPES = {
            "wrench": InputType.WRENCH,
            "prop_forces": InputType.PROP_FORCES,
            "acceleration": InputType.ACCELERATION,
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

        # Default controllers
        self._init_default_controllers()

        self._warn_if_stiff()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        """Build the default cascade.

        The loop :meth:`simulate` drives is
        :class:`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController`,
        an LQR that regulates the *payload* onto its trajectory through the
        whole chain and emits a wrench directly — no attitude loop underneath
        it.  Its gain is solved from this model's own masses, lengths and
        inertia, so the two always describe the same machine (see
        :meth:`_sync_default_controllers`).

        The quadrotor position PD and the geometric attitude loop are still
        built, but they are now the *station-keeping baseline*: a loop that
        holds the airframe where :meth:`reset` left it and lets the chain swing
        beneath, reached through :meth:`compute_station_keeping_input`.  It is
        the analogue of M3's
        :meth:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.compute_formation_input`.
        """
        self._att_controller = GeometricAttitudeController(inertia=self._inertia)
        self._pos_controller = PositionPDController(mass=self.total_mass)
        self._prop_controller = DirectPropellerForceController(
            mass=self._mass, inertia=self._inertia
        )
        self._payload_controller = QuadrotorFlexCableController.from_model(self)
        self._controllers_ready = True

    def _sync_default_controllers(self):
        """Push the current physical parameters into the default controllers.

        Every controller caches its parameters at construction, so a setter
        that writes only ``self._…`` leaves the closed loop driving a different
        machine than the one being integrated — the failure M3's
        ``_sync_default_controllers`` documents at length.

        The sync is duck-typed, as M3's and M4's are: a controller exposing
        ``mass``/``inertia`` is kept in step whether or not this class built
        it.  One that exposes a ``sync_from_model`` hook takes the parameters
        itself and is not written to otherwise; one that exposes neither is
        left entirely alone.
        """
        if not self._controllers_ready:
            return
        for controller, attribute, value in (
            (self._att_controller, "inertia", self._inertia),
            (self._pos_controller, "mass", self.total_mass),
            (self._payload_controller, "mass", self._mass),
        ):
            sync = getattr(controller, "sync_from_model", None)
            if sync is not None:
                sync(self)
            elif hasattr(controller, attribute):
                setattr(controller, attribute, value)

    @property
    def payload_controller(self):
        """The payload-tracking LQR :meth:`simulate` drives.

        Pluggable at runtime, like every controller here. A substitute is
        expected to implement ``compute(t, state[, reference]) -> (4,)`` wrench
        and to carry a ``setpoint(t)``; one that also publishes a ``flat_map``
        gets the flatness feedforward, and one that does not runs feedback-only
        rather than against a reference built from parameters that may not be
        its own.
        """
        return self._payload_controller

    @payload_controller.setter
    def payload_controller(self, controller):
        self._payload_controller = controller

    @property
    def position_controller(self):
        """The station-keeping quadrotor position PD.

        Not what :meth:`simulate` drives — see :attr:`payload_controller` and
        :meth:`compute_station_keeping_input`.
        """
        return self._pos_controller

    @position_controller.setter
    def position_controller(self, controller):
        self._pos_controller = controller

    @property
    def attitude_controller(self):
        return self._att_controller

    @attitude_controller.setter
    def attitude_controller(self, controller):
        self._att_controller = controller

    # ─── Properties ────────────────────────────────────────────────

    @staticmethod
    def _check_positive(values, name: str):
        """Delegates to :func:`udaan.core.validation.check_positive`, shared with M2-M6."""
        check_positive(values, name, warn_below=_MIN_PHYSICAL_MASS if "mass" in name else None)

    def _broadcast(self, value, name: str) -> np.ndarray:
        """Scalar -> ``(n,)`` array, or validate a per-index sequence."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_links, float(arr))
        if arr.shape != (self._num_links,):
            raise ValueError(
                f"{name} must be a scalar or have {self._num_links} entries; got {arr.shape}"
            )
        return arr.astype(float)

    @property
    def num_links(self) -> int:
        """Number of rigid links the cable is discretised into."""
        return self._num_links

    @property
    def mass(self) -> float:
        """Quadrotor mass, in kg."""
        return self._mass

    @mass.setter
    def mass(self, value):
        self._check_positive(float(value), "mass")
        self._mass = float(value)
        self._sync_default_controllers()

    @property
    def inertia(self) -> np.ndarray:
        return np.asarray(self._inertia)

    @inertia.setter
    def inertia(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = np.diag(arr)
        if arr.shape != (3, 3):
            raise ValueError(f"inertia must be (3,) or (3, 3); got {arr.shape}")
        self._inertia = arr
        self._inertia_inv = np.linalg.inv(arr)
        self._sync_default_controllers()

    @property
    def inertia_inv(self) -> np.ndarray:
        return self._inertia_inv

    @property
    def link_masses(self) -> np.ndarray:
        """Point mass lumped at the **distal** end of each link, shape ``(n,)``.

        ``link_masses[-1]`` is therefore the payload's mass plus whatever share
        of the cable's own mass the last link contributes.
        """
        return self._link_masses

    @link_masses.setter
    def link_masses(self, value):
        arr = self._broadcast(value, "link_masses")
        self._check_positive(arr, "link_masses")
        self._link_masses = arr
        self._sync_default_controllers()

    @property
    def link_lengths(self) -> np.ndarray:
        """Per-link lengths, shape ``(n,)``."""
        return self._link_lengths

    @link_lengths.setter
    def link_lengths(self, value):
        arr = self._broadcast(value, "link_lengths")
        self._check_positive(arr, "link_lengths")
        self._link_lengths = arr
        self._sync_default_controllers()

    @property
    def cable_length(self) -> float:
        """Total (taut) cable length ``Σ_i l_i``, in metres."""
        return float(np.sum(self._link_lengths))

    @property
    def total_mass(self) -> float:
        """Quadrotor + chain + payload, in kg — the ``M₀₀`` of the derivation."""
        return float(self._mass + np.sum(self._link_masses))

    # ─── Geometry ──────────────────────────────────────────────────

    def link_positions(self) -> np.ndarray:
        """World positions of the ``n`` lumped masses, shape ``(n, 3)``.

        Entry ``i`` is the distal end of link ``i``, i.e.
        ``x_Q + Σ_{k ≤ i} l_k q_k``; the last row is the payload position.
        """
        x = np.asarray(self.state.position, dtype=float)
        out = np.zeros((self._num_links, 3))
        for i in range(self._num_links):
            x = x + self._link_lengths[i] * np.asarray(self.state.link_attitudes[i])
            out[i] = x
        return out

    def link_velocities(self) -> np.ndarray:
        """World velocities of the ``n`` lumped masses, shape ``(n, 3)``."""
        v = np.asarray(self.state.velocity, dtype=float)
        out = np.zeros((self._num_links, 3))
        for i in range(self._num_links):
            v = v + self._link_lengths[i] * self.state.dq(i)
            out[i] = v
        return out

    # ─── Small-oscillation analysis ────────────────────────────────

    def natural_frequencies(self) -> np.ndarray:
        r"""Small-oscillation frequencies about the hanging equilibrium, in rad/s.

        Linearising :eq:`m5-system` about ``q_i = -e_3`` with the airframe free
        to translate: write ``q_i = -e_3 + u_i`` with ``u_i`` horizontal, so
        ``\dot q_i \approx \dot u_i`` and, from ``\|q_i\| = 1``,
        ``e_3 \cdot q_i \approx -1 + \tfrac12\|u_i\|^2``.  The quadratic parts
        of the energies are then, per horizontal axis,

        .. math::

            T \approx \tfrac12
              \begin{bmatrix} \dot x_Q \\ \dot u \end{bmatrix}^\top
              \begin{bmatrix} M_{00} & M_{0j} \\ M_{0i} & M_{ij} \end{bmatrix}
              \begin{bmatrix} \dot x_Q \\ \dot u \end{bmatrix},
            \qquad
            V \approx \text{const} + \tfrac12 g \sum_k M_{0k} \|u_k\|^2 ,

        so the modes solve the generalised eigenproblem
        ``\mathrm{diag}(0, g M_{0k})\,\phi = \lambda\, M \phi``.  The zero
        eigenvalue is the free horizontal translation of the whole system;
        the remaining ``n`` are the chain's pendulum modes.

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
        so over one second of simulation it grows by ``e^{\omega^2 h / 2}``.
        This returns that exponent for the fastest mode
        :meth:`natural_frequencies` finds — the single number that says whether
        a given ``(num_links, link_masses, dt)`` combination is usable.

        Below ~0.5 the growth is invisible next to the physics; above ~2 a
        few-second run diverges.  It is *not* a bug in the model: the
        continuous dynamics are fine and the MuJoCo backend integrates them
        happily.  Shrink ``dt``, coarsen the chain, or make the links heavier.
        """
        return float(self.natural_frequencies().max() ** 2 * self.h / 2.0)

    def _warn_if_stiff(self):
        rate = self.stability_growth_rate()
        if rate > _STABILITY_GROWTH_WARN:
            _logger.warning(
                "this chain is stiff for the ZOH-Euler integrator: its fastest "
                "small-oscillation mode is %.1f rad/s, which at h = %.3g s grows "
                "amplitudes by e^%.2f per second. Expect a taut-cable run to "
                "diverge within a few seconds. Reduce dt (currently %.4g s), "
                "reduce num_links (currently %d), or increase the link masses "
                "(currently %s kg).",
                float(self.natural_frequencies().max()),
                self.h,
                rate,
                self.dt,
                self._num_links,
                np.array2string(self._link_masses, precision=4),
            )

    # ─── Dynamics ──────────────────────────────────────────────────

    def _mass_coefficients(self) -> tuple[float, np.ndarray, np.ndarray]:
        """``(M₀₀, M₀ᵢ, Mᵢⱼ)`` of :eq:`m5-coefficients`."""
        m = self._link_masses
        length = self._link_lengths
        # tail[i] = Σ_{k ≥ i} m_k, computed as a reversed cumulative sum.
        tail = np.cumsum(m[::-1])[::-1]
        M00 = float(self._mass + m.sum())
        M0i = tail * length
        Mij = tail[np.maximum.outer(np.arange(self._num_links), np.arange(self._num_links))] * (
            np.outer(length, length)
        )
        return M00, M0i, Mij

    def _assemble(self, thrust: float) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the ``(3+3n)`` system :eq:`m5-system` at the current state.

        Returns ``(matrix, rhs)`` in the unknowns
        ``[α; ω̇_1; …; ω̇_n]`` with ``α = a_Q + g e₃``.
        """
        n = self._num_links
        M00, M0i, Mij = self._mass_coefficients()
        q = [np.asarray(self.state.link_attitudes[i], dtype=float) for i in range(n)]
        q_hat = [hat(q[i]) for i in range(n)]
        # c_i = ‖q̇_i‖², not ‖ω_i‖² — see "Departures from the reference".
        c = np.array([float(np.dot(self.state.dq(i), self.state.dq(i))) for i in range(n)])

        size = 3 * (n + 1)
        matrix = np.zeros((size, size))
        rhs = np.zeros(size)

        matrix[0:3, 0:3] = M00 * np.eye(3)
        rhs[0:3] = thrust * (self.state.orientation @ self._e3)
        for i in range(n):
            rhs[0:3] += M0i[i] * c[i] * q[i]

        for i in range(n):
            r = 3 * (i + 1)
            matrix[0:3, r : r + 3] = -M0i[i] * q_hat[i]
            matrix[r : r + 3, 0:3] = M0i[i] * q_hat[i]
            # Diagonal: M_ii I, not M_ii (I - q q^T). Not an approximation —
            # it appends the exact gauge condition ω̇_i · q_i = 0 and makes the
            # matrix positive definite. See the module docstring.
            matrix[r : r + 3, r : r + 3] = Mij[i, i] * np.eye(3)
            for j in range(n):
                if j == i:
                    continue
                cj = 3 * (j + 1)
                matrix[r : r + 3, cj : cj + 3] = -Mij[i, j] * (q_hat[i] @ q_hat[j])
                rhs[r : r + 3] += Mij[i, j] * c[j] * (q_hat[i] @ q[j])

        return matrix, rhs

    def mass_matrix(self) -> np.ndarray:
        """The ``(3+3n) × (3+3n)`` coupled mass matrix at the current state.

        Positive definite for every link configuration whenever the quadrotor
        and link masses are positive — see "Invertibility" in the module
        docstring.  Exposed so that claim can be tested directly on degenerate
        geometries rather than inferred from a simulation that happens not to
        blow up.
        """
        return self._assemble(0.0)[0]

    def accelerations(self, thrust: float = 0.0, torque=None) -> dict:
        """Every acceleration of the system at the current state.

        Solves :eq:`m5-system` once and recurses the payload acceleration
        forward.  This is the whole right-hand side of the ODE; :meth:`_zoh` is
        nothing but an integrator wrapped around it, and the tests read
        residuals off it rather than re-deriving the algebra.

        Args:
            thrust: thrust magnitude ``f``.
            torque: body moment ``M``, ``(3,)``; default zero.

        Returns:
            A dict with keys ``quad_accel`` (``ẍ_Q``), ``payload_accel``
            (``ẍ_L``), ``link_ang_accels`` (``(n, 3)`` of ``ω̇_i``),
            ``link_accels`` (``(n, 3)`` of ``q̈_i``) and ``quad_ang_accel``
            (``Ω̇``).
        """
        n = self._num_links
        torque = np.zeros(3) if torque is None else np.asarray(torque, dtype=float)

        matrix, rhs = self._assemble(float(thrust))
        solution = np.linalg.solve(matrix, rhs)
        quad_accel = solution[0:3] - self._ge3
        link_ang_accels = solution[3:].reshape(n, 3)

        # q̈_i = ω̇_i × q_i − ‖q̇_i‖² q_i, then ẍ_L = ẍ_Q + Σ l_i q̈_i.
        link_accels = np.zeros((n, 3))
        payload_accel = quad_accel.copy()
        for i in range(n):
            q_i = np.asarray(self.state.link_attitudes[i], dtype=float)
            dq_i = self.state.dq(i)
            link_accels[i] = np.cross(link_ang_accels[i], q_i) - np.dot(dq_i, dq_i) * q_i
            payload_accel = payload_accel + self._link_lengths[i] * link_accels[i]

        Omega = np.asarray(self.state.angular_velocity, dtype=float)
        quad_ang_accel = self._inertia_inv @ (torque - np.cross(Omega, self._inertia @ Omega))

        return {
            "quad_accel": quad_accel,
            "payload_accel": payload_accel,
            "link_ang_accels": link_ang_accels,
            "link_accels": link_accels,
            "quad_ang_accel": quad_ang_accel,
        }

    def _zoh(self, thrust: float, torque: np.ndarray, h: float):
        """Zero-order hold Euler integration for one substep.

        The update order — payload translation, then link attitudes, then the
        quadrotor attitude, then the derived quadrotor pose — is M2's, term for
        term, which is what makes the ``n = 1`` reduction hold step by step and
        not merely in the limit.
        """
        n = self._num_links
        omega = [np.asarray(self.state.link_angular_velocities[i]).copy() for i in range(n)]

        acc = self.accelerations(thrust, torque)
        payload_accel = acc["payload_accel"]

        # payload position dynamics
        self.state.payload_position = (
            self.state.payload_position
            + self.state.payload_velocity * h
            + 0.5 * payload_accel * h**2
        )
        self.state.payload_velocity = self.state.payload_velocity + payload_accel * h

        # link attitude dynamics
        for i in range(n):
            (
                self.state.link_attitudes[i],
                self.state.link_angular_velocities[i],
            ) = zoh_step(
                self.state.link_attitudes[i],
                self.state.link_angular_velocities[i],
                acc["link_ang_accels"][i],
                h,
            )

        # quadrotor attitude dynamics
        Omega = np.asarray(self.state.angular_velocity).copy()
        self.state.orientation = self.state.orientation.step(Omega * h)
        self.state.angular_velocity += TSO3(acc["quad_ang_accel"] * h)

        # quadrotor position & velocity derived from payload + chain
        self._sync_quad_state()

    def _sync_quad_state(self):
        """Recompute the quadrotor pose from the payload and the link chain."""
        offset = np.zeros(3)
        rate = np.zeros(3)
        for i in range(self._num_links):
            offset = offset + self._link_lengths[i] * np.asarray(self.state.link_attitudes[i])
            rate = rate + self._link_lengths[i] * self.state.dq(i)
        self.state.position = self.state.payload_position - offset
        self.state.velocity = self.state.payload_velocity - rate

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

    def _wrench_to_propforces(self, wrench):
        return self._alloc_inv @ wrench

    def _propforces_to_wrench(self, prop_forces):
        return self._alloc @ prop_forces

    # ─── Input repackaging ─────────────────────────────────────────

    def _repackage_input(self, u, desired_att=None) -> np.ndarray:
        """Convert user input to a wrench ``[thrust, Mx, My, Mz]``."""
        if self._input_type == InputType.ACCELERATION:
            f, M = self._att_controller.compute(
                self.t,
                (self.state.orientation, self.state.angular_velocity),
                u,
                desired_att=desired_att,
            )
            wrench = np.array([f, *M])
        elif self._input_type == InputType.PROP_FORCES:
            wrench = self._propforces_to_wrench(np.asarray(u, dtype=float))
        else:
            wrench = np.asarray(u, dtype=float).copy()

        # Clip to actuator limits
        wrench[0] = np.clip(wrench[0], self._min_thrust, self._max_thrust)
        wrench[1:] = np.clip(wrench[1:], self._min_torque, self._max_torque)

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
        thrust = float(wrench[0])
        torque = wrench[1:]
        for _ in range(self._inner_loop_steps):
            self._zoh(thrust, torque, self.h)
            self.t += self.h

    def _check_per_link(self, values, name: str) -> list:
        """Validate a per-link ``reset()`` override and return it as a list.

        Shared with the MuJoCo backend so both raise the same message on a
        wrong-length sequence.
        """
        entries: list = list(values)
        if len(entries) != self._num_links:
            raise ValueError(f"{name} must have {self._num_links} entries, got {len(entries)}")
        if name == "link_attitudes":
            for i, v in enumerate(entries):
                if float(np.linalg.norm(np.asarray(v, dtype=float))) < _MIN_DIRECTION_NORM:
                    raise ValueError(
                        f"{name}[{i}] has (near-)zero norm and does not define a direction"
                    )
        return entries

    def reset(self, **kwargs):
        """Reset time and state, then apply any state overrides.

        Recognised kwargs: ``position``, ``velocity``, ``payload_position``,
        ``payload_velocity``, ``orientation``, ``angular_velocity``,
        ``link_attitudes``, ``link_angular_velocities``.  The per-link entries
        must be sequences of length ``num_links``.

        The quadrotor and the payload are tied by the chain, so only one of
        them is independent: give ``position`` *or* ``payload_position`` and
        the other is derived from the link directions (matching M2).  Giving
        both lets ``payload_position`` win, since it is what the integrator
        advances.

        **Off-manifold per-link input is repaired, not rejected.**  Each
        ``link_attitudes`` entry is normalised onto ``S^2`` and each
        ``link_angular_velocities`` entry is projected onto the tangent plane
        of its (already normalised) direction, ``omega_i -= (omega_i . q_i)
        q_i``.  This is exactly what
        :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableMujoco.reset`
        has to do anyway — a MuJoCo ball joint has no way to represent a
        stretched link or a spin about the link axis — so doing it here is what
        makes the two backends land on the *same* state from the same kwargs.
        Repair rather than rejection because the spin component is genuine
        gauge freedom the dynamics already factor out (see "Why the diagonal
        block is ``M_ii I``" above), and a caller who hand-builds a direction
        from an unnormalised vector is asking for the direction, not for an
        error.  An *unrecognised keyword*, by contrast, is rejected: a stale
        name must not silently run a different initial condition.

        Raises:
            ValueError: for any keyword outside the list above.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        self.t = 0.0
        self.state.reset()

        if "link_attitudes" in kwargs:
            values = self._check_per_link(kwargs["link_attitudes"], "link_attitudes")
            # Rewrap every entry: the integrator mutates the tangent wrappers
            # in place, so a caller's object must not be aliased.  S2 does not
            # normalise on construction (repo-wide), so do it here.
            self.state.link_attitudes = [
                S2(np.asarray(v, dtype=float) / np.linalg.norm(np.asarray(v, dtype=float)))
                for v in values
            ]
        if "link_angular_velocities" in kwargs:
            values = self._check_per_link(
                kwargs["link_angular_velocities"], "link_angular_velocities"
            )
            rates = []
            for i, v in enumerate(values):
                q_i = np.asarray(self.state.link_attitudes[i], dtype=float)
                omega_i = np.asarray(v, dtype=float)
                rates.append(TS2(omega_i - float(np.dot(omega_i, q_i)) * q_i))
            self.state.link_angular_velocities = rates

        if "orientation" in kwargs:
            self.state.orientation = SO3(np.asarray(kwargs["orientation"]))
        if "angular_velocity" in kwargs:
            self.state.angular_velocity = TSO3(np.asarray(kwargs["angular_velocity"]))

        for key in ("payload_position", "payload_velocity", "position", "velocity"):
            if key in kwargs:
                # Copy: a caller-supplied array must never be aliased into the
                # state (the M2 bug that silently mutated shared EVAL_STARTS).
                setattr(self.state, key, np.array(kwargs[key], dtype=float))

        offset = np.zeros(3)
        rate = np.zeros(3)
        for i in range(self._num_links):
            offset = offset + self._link_lengths[i] * np.asarray(self.state.link_attitudes[i])
            rate = rate + self._link_lengths[i] * self.state.dq(i)

        if "payload_position" not in kwargs and "position" in kwargs:
            self.state.payload_position = self.state.position + offset
        if "payload_velocity" not in kwargs and "velocity" in kwargs:
            self.state.payload_velocity = self.state.velocity + rate

        self._sync_quad_state()

    def compute_default_input(self, reference=None) -> np.ndarray:
        """The wrench ``[f, M]`` from the M5 payload LQR — what :meth:`simulate` drives.

        Args:
            reference: optional ``(RefState, Inputs)`` pair from
                :class:`udaan.flatness.QuadrotorFlexCable`. When ``None`` the
                controller regulates against the hanging equilibrium at its
                setpoint, which is the feedback-only law.

        Returns a ``(4,)`` array suitable for :meth:`step` under the ``WRENCH``
        input type, which is this model's default.
        """
        if reference is None:
            return np.asarray(self._payload_controller.compute(self.t, self.state))
        return np.asarray(self._payload_controller.compute(self.t, self.state, reference))

    def compute_station_keeping_input(self) -> np.ndarray:
        """Thrust force vector from the station-keeping quadrotor position PD.

        The pre-payload-controller baseline, and the analogue of M3's
        :meth:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.compute_formation_input`:
        it holds the *airframe* where :meth:`reset` left it and lets the chain
        swing beneath, carrying the whole system weight as feedforward. Kept as
        a comparison point; under the default ``WRENCH`` input type
        :meth:`simulate` drives :meth:`compute_default_input` instead.

        Returns a ``(3,)`` array, so stepping it needs the ``ACCELERATION``
        input type — construct the model with ``input="acceleration"``, which
        is also what makes :meth:`simulate` drive *this* loop rather than the
        payload LQR.
        """
        return np.asarray(
            self._pos_controller.compute(self.t, (self.state.position, self.state.velocity))
        )

    def hover_input(self) -> np.ndarray:
        """The open-loop static-trim wrench ``[f, 0, 0, 0]``.

        Straight from the "Static trim" section of the module docstring:
        ``f = (m_Q + Σ_k m_k) g`` and zero moment, valid at ``R = I`` with
        every link hanging along ``-e₃``.
        """
        return np.array([self.total_mass * self._g, 0.0, 0.0, 0.0])

    # ─── Flatness feedforward ──────────────────────────────────────

    def _flatness_feedforward_fn(self):
        """``(jet_fn, exact)`` for the bound payload trajectory, or ``(None, False)``.

        The payload controller's ``setpoint`` is sniffed for a bound method, and
        its owner for a way to differentiate the trajectory deeply enough. M5 is
        the first system where "deeply enough" is not a constant: the flat map
        needs the payload position through its ``2n + 4``-th derivative (see
        :func:`udaan.flatness.chain.chain_jet_order`), so ``get_full``'s five
        rows — position through snap — stop being sufficient at ``n = 1``.

        Two protocols, in order of preference:

        * ``get_jet(t, order) -> (order + 1, 3)``, which supplies every
          derivative in closed form. This is the only one that feeds the map
          what it actually asks for, and it is what
          :class:`~udaan.utils.trajectory.CrazyTrajectory` implements.
        * ``get_full(t) -> (pos, vel, acc, jerk, snap)``, zero-padded past snap.
          The recovered position and link geometry are unaffected — the top
          node's *acceleration* only needs the tip through order ``2n + 2`` — but
          the recovered ``Ω``, ``Ω̇`` and ``M_d`` are built from its jerk and
          snap, so they degrade. :meth:`simulate` warns once when it has to fall
          back on this.
        """
        setpoint_fn = getattr(self._payload_controller, "setpoint", None)
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

        Reset by :meth:`simulate`. A non-zero count means the reference left the
        flat map's taut-hanging-chain certificate on those steps — a link went
        slack, swung past horizontal, or the airframe was in free fall — and the
        step ran against the hanging-equilibrium reference instead. The first
        drop is logged as a warning, the rest at debug level.
        """
        return self._feedforward_drops

    def _flat_reference(self, flat_map, jet_fn, t):
        """``(RefState, Inputs)`` recovered from the flat output at ``t``, or ``None``.

        ``None`` is a legitimate outcome, not an error: the map refuses any
        sample whose chain is not taut and hanging, and the right response is to
        fall back on the feedback-only reference for that step rather than to
        abort. It is counted in :attr:`feedforward_drops`.

        A malformed jet — wrong shape, wrong dimension — is a programming error
        and is left to propagate.
        """
        flats = QuadrotorFlexCableFlats(
            x_L=Jet(np.asarray(jet_fn(t, flat_map.jet_order), dtype=float)),
            psi=Jet(np.zeros(3)),
        )
        try:
            return flat_map.recover(flats)
        except ValueError as err:
            self._feedforward_drops += 1
            if self._feedforward_drops == 1:
                _logger.warning(
                    "flatness feedforward dropped at t=%.4f (%s); this step ran "
                    "against the hanging-equilibrium reference. Further drops are "
                    "logged at debug level; see the feedforward_drops counter for "
                    "the total.",
                    t,
                    err,
                )
            else:
                _logger.debug("flatness feedforward unavailable at t=%.4f: %s", t, err)
            return None

    def simulate(self, tf, **kwargs):
        """Run closed-loop simulation to time tf.

        Drives :attr:`payload_controller` — the M5 LQR — under the ``WRENCH``
        input type. When its ``setpoint`` is a trajectory's bound method that
        can be differentiated to the depth the chain needs, the run also picks
        up the flatness feedforward:
        :class:`udaan.flatness.QuadrotorFlexCable` recovers the whole reference
        state *and* the reference wrench ``(f_d, M_d)``, and both are handed to
        the controller. Binding such a trajectory is the whole switch.

        This is the M3/M4 mechanism with one deliberate difference. Those
        systems feed the recovered attitude and rates in as ``desired_att``,
        because their payload cascade emits a thrust *vector* and the geometric
        attitude loop is what consumes the feedforward. An LQR emits the body
        moment itself, so there is no attitude loop in the path and the
        feedforward goes where it belongs: into the reference the error state is
        measured against, and into the wrench the correction is added to.

        The flat map is taken from the payload controller's **own**
        :attr:`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.flat_map`,
        never rebuilt from this model. The controller's gain was solved at its
        own masses and lengths; a reference recovered from different ones is the
        reference for a different machine, and the mismatch shows up as a
        steady-state offset with no other symptom. A controller that does not
        publish a flat map therefore gets no feedforward at all.

        The two non-default input types drive their own loop instead, because
        neither of them can consume a wrench:
        ``input="prop_forces"`` runs
        :attr:`~udaan.control.DirectPropellerForceController`, and
        ``input="acceleration"`` runs the station-keeping quadrotor position PD
        of :meth:`compute_station_keeping_input`, whose thrust *vector* is what
        the ``ACCELERATION`` pipeline expects. Neither picks up the flatness
        feedforward — it is a payload reference, and neither of those loops
        regulates the payload.
        """
        self.reset(**kwargs)
        self._feedforward_drops = 0

        jet_fn, exact = self._flatness_feedforward_fn()
        flat_map = None
        if jet_fn is not None and self._input_type == InputType.WRENCH:
            flat_map = getattr(self._payload_controller, "flat_map", None)
            if flat_map is None:
                _logger.warning(
                    "%s exposes no flat_map, so the reference this loop is "
                    "regulating against is unknown; running feedback-only rather "
                    "than feeding forward a reference built from the model's own "
                    "parameters, which need not be the controller's.",
                    type(self._payload_controller).__name__,
                )
            elif not exact and flat_map.jet_order > 4:
                _logger.warning(
                    "the bound trajectory supplies derivatives only through snap, "
                    "but a %d-link chain needs the payload position through its "
                    "%d-th derivative (2n + 4). Orders 5-%d are being taken as "
                    "zero: the recovered positions and link directions are "
                    "unaffected, the recovered Omega/Omega-dot and moment "
                    "feedforward are not. A trajectory exposing "
                    "get_jet(t, order) supplies them exactly.",
                    self._num_links,
                    flat_map.jet_order,
                    flat_map.jet_order,
                )

        start_t = time.time_ns()
        while self.t < tf:
            if self._input_type == InputType.PROP_FORCES:
                u = self._prop_controller.compute(
                    self.t,
                    (
                        self.state.position,
                        self.state.velocity,
                        self.state.orientation,
                        self.state.angular_velocity,
                    ),
                )
            elif self._input_type == InputType.ACCELERATION:
                # The ACCELERATION pipeline hands its input to the geometric
                # attitude controller as a *thrust vector*; the payload LQR
                # emits a wrench, and feeding one to the other is a crash, not
                # a mistuning.  The loop that belongs on this pipeline is the
                # station-keeping PD.
                u = self.compute_station_keeping_input()
            else:
                reference = (
                    None if flat_map is None else self._flat_reference(flat_map, jet_fn, self.t)
                )
                u = self.compute_default_input(reference=reference)

            if self.verbose >= 2:
                self._log_state(u)

            self.step(u)

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _log_state(self, u):
        wrench = self._repackage_input(u)
        _logger.debug(
            "t=%.4f xQ=%s xL=%s q0=%s qn=%s f=%.4f M=%s",
            self.t,
            np.array2string(self.state.position, precision=4),
            np.array2string(self.state.payload_position, precision=4),
            np.array2string(np.asarray(self.state.link_attitudes[0]), precision=4),
            np.array2string(np.asarray(self.state.link_attitudes[-1]), precision=4),
            wrench[0],
            np.array2string(wrench[1:], precision=4),
        )
