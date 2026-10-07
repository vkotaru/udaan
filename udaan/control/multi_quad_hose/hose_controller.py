r"""LQR for ``nQ`` quadrotors carrying one flexible ``n``-link hose (M6).

The last controller in the package, and the second — after M5 — that is not a
geometric cascade.  The reason is M5's and adding vehicles does not change it:
a vehicle is bolted to a *vertex*, the links on either side of it still move
only through the chain, and there is no ``q_i^d`` any actuator can be told to
chase.  The published law is a linear-quadratic regulator on the variational
linearisation, and that is what is implemented.

.. math::

    (f_k, M_k) = (f_k^d, M_k^d) + [K s]_k ,
    \qquad
    s = \bigl[e_{x_0},\; e_{v_0},\; \{\xi_i\},\; \{\delta\omega_i\},\;
              \{\eta_k\},\; \{\delta\Omega_k\}\bigr] ,

with ``(f^d, M^d)`` and the whole reference state supplied by
:class:`udaan.flatness.MultiQuadHose`, or — when no trajectory is bound — by
the static trim at the setpoint (:meth:`~MultiQuadHoseController.equilibrium_reference`).

What is regulated
-----------------
The **root vertex**, plus the shape of the hose.  M6's flat output is the root
position, one *vector tension* per vehicle other than the one at the last
vertex, and a yaw each; the ``setpoint`` supplies the first, this controller's
:attr:`~MultiQuadHoseController.trim_tensions` the second, and zero the third.
Every vehicle position, attitude, thrust and moment is then an *output* of the
flat map, not a command.  Steering a particular vehicle to a particular point
is an inverse problem — the reference solves it numerically in
``optimize_for_des_state`` — and belongs to a planner.

Error state
-----------
Six blocks, in the conventions M2-M5 already use:

.. math::

    e_{x_0} &= x_0 - x_0^d, \qquad e_{v_0} = v_0 - v_0^d, \\
    \xi_i &= q_i^d \times q_i, \qquad
    \delta\omega_i = \omega_i + q_i \times (q_i \times \omega_i^d), \\
    \eta_k &= \tfrac12 \operatorname{vee}(R_k^{d\top} R_k - R_k^\top R_k^d),
    \qquad
    \delta\Omega_k = \Omega_k - R_k^\top R_k^d \Omega_k^d .

The translational block is the *root's*, which is the row of the equations of
motion the collected thrusts act on.  Every other vertex is driven to its
reference through the ``\xi_i``, since ``x_j = x_0 + \sum_{i<j}\ell_i q_i``
makes the link directions the rest of the configuration.

The block *ordering* differs from the reference MATLAB's
``[e_x; \xi; e_v; \delta\omega; \eta; \delta\Omega]`` (``error.m`` L47).  The
grouping used here is the one the reference's own cost matrix assumes — see
"Errata" below.

The equilibrium reference
-------------------------
With no trajectory bound the loop regulates against a **static trim at the
setpoint**, produced by the same flat map at an all-zero-derivative flat
output.  That is exact, not approximate: a trim translating at constant
velocity is a genuine solution, so the feedback-only reference costs nothing
in consistency.

Which trim matters.  A hose held between two vehicles has no *vertical*
equilibrium with every link taut — spreading the thrust drives some tension
through zero and folds the hose there, which is why
:meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.hover_trim` puts the
whole weight on one vehicle.  The trim here is the discrete catenary instead:
every inter-vehicle segment gets the same horizontal tension ``H`` and the
tension at the top of segment ``k`` carries minus half that segment's interior
weight, so the segment hangs symmetrically and comes back level.  Then every
tension in the segment satisfies ``\|𝛕_i\| \ge H > 0`` — **no link can be
slack, whatever the masses** — and the horizontal components cancel at every
interior vehicle, which hovers level carrying only weight.  Only the outermost
two pull, with ``\pm H``.  ``H`` itself is chosen so the trim spans
``sag_ratio`` of the hose length between the outermost vehicles, which puts the
equilibrium close to the taut spawn shape both backends reset to;
:attr:`~MultiQuadHoseController.trim_tensions` is the result, and is settable.

One family of patterns has no sag to solve for: where every inter-vehicle
segment is a *single* rigid link — ``attachments = (0, 1, …, n)`` — the hose is
straight and level whatever the pull, so ``sag_ratio`` is ignored and ``H``
falls back on a pre-tension of the hose's own weight
(:meth:`~MultiQuadHoseController._trim_pretension`).

Two decisions, and why
----------------------

**1. Infinite horizon, about that trim.**  The reference
(``matlab/controllers/control_lqr.m`` with ``solveContRicattiEqn.m``)
integrates the Riccati equation backwards with ``ode45`` over ``[t_f, t_0]``
along one specific trajectory, caches ``P(t)`` in a ``.mat`` and interpolates
it at run time.  That needs the whole reference in advance and produces a
schedule tied to one manoeuvre; a controller here has to work against a
``setpoint`` callable it has never seen.  So: one
:func:`~scipy.linalg.solve_continuous_are` at the trim.

The price is measurable, and it is measured across a *range* of reference
aggressiveness rather than on one gentle case — the mistake M5's write-up made.
Against ``CrazyTrajectory`` references of rising amplitude and frequency, at
``n = 3`` with vehicles at ``(0, 3)``:

==================  =============  =========  ============  ============  ==============
reference (ax, f1)  max link tilt  A drift    ZOH, fixed K  DOP853        ZOH, scheduled
==================  =============  =========  ============  ============  ==============
0.15, 0.15          0.9°           1.6 %      0.05 mm       0.02 mm       < 0.1 mm
0.40, 0.30          10.8°          18 %       0.67 mm       0.25 mm       0.6 mm
0.70, 0.41          43.8°          59 %       6.5 mm        1.1 mm        3.0 mm
0.80, 0.45          73.6°          305 %      NaN           0.36 m        0.76 m
==================  =============  =========  ============  ============  ==============

("A drift" is ``\|A - A_0\|/\|A_0\|`` against the trim linearisation ``A_0``.
6 s run started on the reference — the whole state, not just the root — with
the root-position RMS taken over the last 3 s and the flatness feedforward on
throughout.  "ZOH" is udaan's shipped 5 ms zero-order-hold loop over four Euler
substeps; "DOP853" holds the same 5 ms command but integrates the plant
accurately between updates, and runs the last row without actuator clipping —
see below.  "scheduled" re-solves the Riccati equation at every control step,
through the ZOH loop.)

Two readings of that table, and only the first is about the gain.

*The gain.*  The linearisation moves a great deal and the closed loop tolerates
most of it: with the feedforward carrying the reference, a gain solved once at
the trim holds sub-millimetre through 18 % drift and a millimetre or so through
59 %.  Nothing warns when a reference leaves that range, either — the flat map
does **not** refuse the aggressive cases, since at ``nQ >= 2`` the
hanging-branch test is off by construction (a hose held at both ends has links
pointing upwards on purpose).

*The integrator.*  The last two rows are mostly **not** a statement about the
gain's domain, and an earlier version of this table read them as one.  About
80 % of row 3's 6.5 mm is Euler truncation: the same loop with DOP853 between
control updates comes in at 1.1 mm.  And row 4 does not diverge at all under an
accurate integrator — the NaN is the ZOH loop blowing up, not the closed loop.
Unclipped it settles to 0.36 m, which is *better* than the 0.76 m the scheduled
column reports, and that comparison was itself measured through the same
blowing-up integrator, so "scheduling does not rescue it" was never a
conclusion the measurement could support.  Actuator saturation is in the loop
by then as well: the first clip is at ``t = 0.99 s``, before the error passes
5 cm at ``t = 1.38 s``, and clipping the DOP853 run turns 0.36 m into 6.1 m.
What row 4 shows is a reference that a 200 Hz zero-order-hold loop on 20 N
airframes cannot fly; how much of the remaining 0.36 m is the constant gain is
not something this table separates.

Gain scheduling is two lines away — ``ctrl.set_gain_from(*ctrl.linearize(ref,
ff), ref.link_attitudes)`` — and deliberately not automatic: a Riccati solve
costs milliseconds against a 5 ms control period (8 ms at ``n = 4, nQ = 2`` and
53 ms at ``nQ = 5`` on the machine these were taken on — a machine-dependent
number, but the ratio to the control period is the point), it buys nothing on a
gentle reference, and where it does help it roughly halves the error rather
than changing the picture.

*What the loop actually delivers.*  Root-position RMS over the second half of a
6 s run on a gentle Lissajous (``ax = 0.3``, ``f1 = 0.25``), both columns out of
:meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.simulate`:

=====  ===============  =================  ==============  ===========
``n``  attachments      step, final error  no feedforward  feedforward
=====  ===============  =================  ==============  ===========
2      (0, 2)           0.70 mm            269 mm          0.30 mm
3      (0, 3)           1.04 mm            302 mm          0.33 mm
4      (0, 4)           0.67 mm            308 mm          0.34 mm
4      (0, 2, 4)        0.98 mm            305 mm          0.34 mm
4      (0, 1, 2, 3, 4)  0.31 mm            335 mm          0.39 mm
3      (1, 3)           0.62 mm            391 mm          0.42 mm
2      (2,)             0.42 mm            532 mm          0.56 mm
=====  ===============  =================  ==============  ===========

"Started on the reference" means the **whole** state — root position and
velocity, every link direction and rate, every attitude and body rate — taken
from the flat map at ``t = 0``, which is what
:meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.reset` accepts and what
the tests do.  It is not a detail: starting with only ``root_position`` on the
reference and the hose in its spawn shape — the natural reading, and what the
CLI does — puts ``n = 3, (0, 3)`` at **32 mm** rather than 0.33 mm over the same
window, and it reaches 0.33 mm only once the transient has run out, after about
10 s.

The step column is a 0.44 m step on the root from the spawn shape, scored on
the final error after 8 s.  The middle column is the honest feedback-only law —
the loop regulating against the *static trim* at a moving setpoint — so most of
that 0.3 m is the lag of a ~0.8 rad/s loop behind a 1.6 rad/s reference.

**2. What is and is not controllable.**  Two statements, both measured.

*Structurally.*  The embedded error has ``6 + 6n + 6nQ`` entries but the system
has ``6 + 4n + 6nQ`` degrees of freedom: ``2n`` coordinates — the components of
``\xi_i`` and ``\delta\omega_i`` along ``q_i^d`` — are the ``S^2`` gauge, and
no input reaches them, so the embedded pair has no stabilising Riccati
solution.  **M5's structural finding transfers unchanged**, with a shorter
argument: the mass matrix has the gauge vectors as eigenvectors
(``M e = M_ii e`` for the ``e`` whose only nonzero block is ``q_i``, because
every off-diagonal block in column ``i`` carries ``\hat q_i`` as its *right*
factor and therefore annihilates ``q_i``), it is symmetric, so the gauge
component of a solved link row is the gauge component of the right-hand side
over ``M_ii`` — and every column of the input Jacobian in link row ``i`` is a
``\hat q_i(\cdot)``.  So ``B``'s gauge rows vanish identically at *every*
reference, and the Riccati solve runs on the tangent-reduced pair
(:meth:`~MultiQuadHoseController.tangent_projection`).

How the embedded failure *shows up* is a fact about this plant rather than a
correction to M5.  M5's own claim holds for M5:
:func:`~scipy.linalg.solve_continuous_are` raises on M5's embedded pair at
every ``n`` from 1 to 8, with identity weights and with M5's own.  On M6's
embedded pair it does not: over the seven configurations tested here, with
identity weights, it raises on five and on two — ``(0, 2)`` and ``(0, 4)`` —
returns a ``P`` of norm ``1e9`` and ``1e12``, Riccati residual 2 and 51, whose
closed loop still has its eigenvalues on the imaginary axis to ``1e-14``.
*Which* configurations fall on which side is a property of an extremely
ill-conditioned pencil and is not stable across solver versions.  The reduction
is needed either way — but a test that only watched for the exception would
pass for the wrong reason, which is why the test asserts the weaker, true
statement instead.  The same solver behaviour, off the gauge subspace, is why
the reduced solve is *checked* rather than trusted; see
:func:`~udaan.control.solve_lqr_gain`.

``A``'s gauge rows vanish **at any static trim**, which is stronger than M5's
statement about a hanging equilibrium: at a trim the link row's own ``\xi_i``
block is ``-\hat v_i \hat q_i`` with ``v_i \parallel q_i`` by the trim
condition itself, and ``-\lambda\hat q\hat q = \lambda(I - qq^\top)`` has a
zero ``q`` row.  Measured: ``\|(I - WW^\top)AW\|/\|A\|`` is ``\le 1e-16`` at
the trim; off it the reference's aggressiveness sets the leak, so the figure
is only meaningful with the trajectory named.  Over a 6 s window it runs
``1.8e-3`` to ``3.3e-3`` on the gentle Lissajous (``ax = 0.3``, ``f1 = 0.25``),
``4.4e-3`` to ``1.1e-2`` on the moderate one (0.4, 0.3), and as far as ``5e-1``
on the aggressive one (0.8, 0.45).

*Practically.*  The PBH margin ``min_λ σ_min[A_r - λI  B_r]`` at the trim:

=====  =================  ======  ==========================
``n``  attachments        PBH     slowest closed-loop pole
=====  =================  ======  ==========================
1      (0, 1)             0.39    -0.98 ± 1.68j
2      (0, 2)             0.48    -0.82 ± 0.54j
4      (0, 4)             0.12    -0.60 ± 10.1j
4      (0, 2, 4)          0.37    -0.75 ± 0.50j
4      (0, 1, 2, 3, 4)    0.61    -0.80 ± 0.68j
4      (4,)               0.085   -0.58 ± 13.7j
=====  =================  ======  ==========================

The last row is M5's configuration — one vehicle at the top of a four-link
chain — and it reproduces M5's collapse (0.026 at ``n = 3`` there).
**Spreading the vehicles along the hose is what recovers the authority**, and
the table orders by *where* they are rather than how many: ``(0, 4)`` at 0.12
is much weaker than ``(0, 2, 4)`` at 0.37, a four-link span with nothing in the
middle having a poorly driven interior, and a vehicle at *every* vertex is the
strongest of all at 0.61.  Rows 1 and 5 are the two patterns whose every
segment is a single link, so they stand on the pre-tension trim rather than a
catenary; before that fallback was reachable they were measured on a degenerate
~1e-9 N trim and were not this plant's numbers at all.

Which mode is slowest is pattern-dependent.  At ``(0, 2, 4)`` it is the
root-position mode (51 % root position, 42 % root velocity, 7 % hose); at
``(0, 4)`` it is a hose-bending mode coupled to the airframes' body rates
(53 % link rate, 46 % body rate, 0 % root) at 10.1 rad/s and 5.9 % damping.

*Weights.*  Uniform ``Q`` scaling says nothing — for an LQR it is the same as
scaling ``R^{-1}``, so it slides along the cheap-control locus, and ×1000 duly
moves the ``(0, 2, 4)`` slowest pole from -0.753 to -1.000.  Targeted weights
do move it, and not all the way one would guess:

======================  =========================
weight ×50              slowest pole, ``(0,2,4)``
======================  =========================
default                 -0.753 ± 0.499j
``q_root_position``     -1.003
``q_link``              -0.780 ± 0.495j
``q_attitude``          -0.572 ± 0.451j
``q_root_velocity``     -0.141
``q_link_rate``         -0.142
``q_angular_rate``      -0.163
======================  =========================

Every *rate* weight makes the slowest pole worse.  That is **not** "the
opposite of M5", which is what an earlier reading of this table said:
``q_link_rate`` ×50 damps the hose's bending modes here exactly as it does
there.  At ``(0, 4)`` the bending pole goes from -0.598 ± 10.15j (5.9 %
damping) to -2.11 ± 9.66j (21 %); at ``(0, 2, 4)`` from -1.19 ± 5.84j (20 %) to
-1.88 ± 4.46j (39 %); at ``(4,)`` from -0.583 ± 13.7j (4.3 %) to -2.62 ± 13.2j
(20 %).

What differs is *which mode is slowest*.  In M5 the bending mode is the
slowest one, so damping it moves the summary statistic.  In M6's spread
patterns the root translation is already the slowest, so all the rate
reweighting buys is a slower translational pole: -0.1414, which is
``sqrt(0.5 / 25) = sqrt(q_root_position / 50 q_root_velocity)``, the generic
heavily-rate-weighted double-integrator root.  M5 lands on exactly the same
-0.1414 for ``q_velocity`` ×50.  And M6 at ``(4,)`` — the M5-like
configuration, one vehicle at the top of the chain — behaves like M5 outright:
``q_link_rate`` ×50 leaves its slowest pole at -0.517 ± 0.712j, against M5's
-0.514 ± 0.763j.

Weighting the positions helps, and ``q_root_position`` and ``q_link`` together
reach -1.155, so there is no structural floor of M5's kind here; this plant is
simply better actuated.

The linearisation as implemented
--------------------------------
:meth:`~MultiQuadHoseController.linearize` is the *variational* linearisation:
it treats ``\delta\omega_i`` and ``\delta\Omega_k`` as if the reference rates
added (``\omega_i = \omega_i^d + \delta\omega_i``), which is the published
derivation, while the error state transports them to the current
configuration.  Against finite differences of the true nonlinear error
dynamics the two agree to ``1e-9`` **at any static trim**, where
``\omega^d = \Omega^d = 0`` kills the difference exactly, and differ by
``1e-2`` relative at a moderate moving reference.  The shipped gain rests on
the exact case; a scheduled gain does not, and says so.

Errata in the reference
-----------------------
Three, none of them in ``get_dyn_dq``, which the model's re-derivation found
intact:

* ``example1.m`` L68-72 builds ``Q = blkdiag(0.5 I_6, 0.75 I_{6n}, I_{3nQ},
  0.75 I_{3nQ})``, whose first two blocks group ``[x, v]`` and then the hose —
  but the error vector it multiplies is ``[e_x; \xi; e_v; \delta\omega; \eta;
  \delta\Omega]`` (``error.m`` L47), which interleaves them.  Six diagonal
  entries land in the wrong block: the first link's ``\xi`` is weighted 0.5
  rather than 0.75, and ``e_v`` 0.75 rather than 0.5.  Small in effect and
  clearly not the intent; :mod:`udaan.core.defaults` carries the intent.
* ``R = 0.2 I_{4nQ}`` weights a thrust and a moment equally though their
  columns of ``B`` are orders of magnitude apart on the shipped airframe — the
  thrust columns scale as ``1/M_{00}`` and the moment columns as ``J^{-1}``.
  The largest moment column is ``\|J^{-1}\| = 435`` in every configuration,
  while the thrust columns run 0.435 to 2.83 across the ones tabulated above,
  so the disparity is a factor of **150 to 1000** depending on the pattern
  rather than one number.  The resulting -842 rad/s attitude pole sits at 121°
  of hold lag at
  udaan's 200 Hz and the loop never settles; weighting the moments at 20 moves
  it to -84 rad/s (12°).  Unlike M5, where the same change cost up to 13 % of
  the chain's slowest real part, here it is nearly free: -0.6022 to -0.5982 at
  ``(0, 4)``, unmoved at ``(0, 2)`` and ``(0, 2, 4)``.  See
  :data:`~udaan.core.defaults.DEFAULT_HOSE_LQR_R_MOMENT`.
* ``optimize_for_des_state`` — the setpoint solver, and unlike the two above
  **on the published path** (``computeSetpoint`` calls it, and the paper's
  Sim 1.1/1.2 call ``computeSetpoint``).  Its ``cost_fun`` is commented "sum of
  the square of the tension norms" but sums the squared *thrusts*; minimising
  thrust is the stated intent, so it is the comment that is wrong.  More
  consequentially it leaves the tension magnitudes unbounded in sign, so
  nothing stops it returning a setpoint with links in compression — a hose
  cannot push.  The catenary trim here cannot: its tensions are bounded below
  by ``H > 0`` by construction.

References:
    P. Kotaru and K. Sreenath, "Multiple Quadrotors Carrying a Flexible Hose:
    Dynamics, Differential Flatness and Control", IFAC World Congress, 2020.
"""

from __future__ import annotations

import numpy as np

from ...control import PDController, solve_lqr_gain
from ...core.defaults import (
    DEFAULT_HOSE_LENGTH,
    DEFAULT_HOSE_LQR_Q_ANGULAR_RATE,
    DEFAULT_HOSE_LQR_Q_ATTITUDE,
    DEFAULT_HOSE_LQR_Q_LINK,
    DEFAULT_HOSE_LQR_Q_LINK_RATE,
    DEFAULT_HOSE_LQR_Q_ROOT_POSITION,
    DEFAULT_HOSE_LQR_Q_ROOT_VELOCITY,
    DEFAULT_HOSE_LQR_R_MOMENT,
    DEFAULT_HOSE_LQR_R_THRUST,
    DEFAULT_HOSE_MASS,
    DEFAULT_HOSE_SAG_RATIO,
    DEFAULT_NUM_HOSE_LINKS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ...flatness.jet import Jet
from ...flatness.multi_quad_cspayload import DEFAULT_MIN_TENSION
from ...flatness.multi_quad_hose import (
    MultiQuadHose,
    MultiQuadHoseFlats,
    MultiQuadHoseInputs,
    MultiQuadHoseRefState,
)
from ...utils import hat, vee
from ...utils.hose import validate_attachments

__all__ = ["MultiQuadHoseController"]

_E1 = np.array([1.0, 0.0, 0.0])
_E3 = np.array([0.0, 0.0, 1.0])

#: Cosine below which a link direction is treated as too close to the tangent
#: basis' seed axis for that seed to be used.  Purely a numerical guard on
#: :func:`_tangent_basis`; any value comfortably inside ``(0, 1)`` works.
_TANGENT_SEED_CUTOFF = 0.9

#: Bisection bracket for the trim's horizontal tension: the search starts at
#: ``M₀₀ g`` and doubles, then halves, this many times each before giving up
#: and falling back on :meth:`MultiQuadHoseController._trim_pretension`.
_TRIM_BRACKET_DOUBLINGS = 40

#: Relative tolerance of that bisection, as a fraction of the target span.
_TRIM_SPAN_TOL = 1e-9


def _tangent_basis(q: np.ndarray) -> np.ndarray:
    """A ``(3, 2)`` orthonormal basis of ``q``'s tangent plane on ``S^2``.

    Only the *span* matters — it is used to project the gauge directions out of
    the Riccati solve — so any consistent choice does.
    """
    seed = (
        np.array([1.0, 0.0, 0.0]) if abs(q[0]) < _TANGENT_SEED_CUTOFF else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(q, seed)
    u = u / np.linalg.norm(u)
    return np.stack([u, np.cross(q, u)], axis=1)


class MultiQuadHoseController(PDController):
    """Infinite-horizon LQR steering an ``n``-link hose carried by ``nQ`` quadrotors.

    ``compute(t, state[, reference])`` returns an ``(nQ, 4)`` stack of wrenches
    ``[f_k, M_kx, M_ky, M_kz]`` — a
    :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.step` input under
    ``InputType.WRENCH``.

    Args:
        num_links: ``n >= 1``.
        attachments: vertex indices carrying a vehicle, strictly increasing.
            The **last vertex must be in the set** or the hose is not flat in
            these coordinates and :class:`~udaan.flatness.NotFlatError` is
            raised.
        masses, inertias: per-vehicle airframe mass and inertia; scalars
            broadcast.
        vertex_masses: the hose's own lumped mass at each of the ``n + 1``
            vertices; defaults to ``hose_mass`` spread evenly.
        link_lengths: per-link lengths; defaults to ``hose_length / n``.
        gravity: ``g``.
        q_root_position: diagonal weight on the root-position error block.
        q_root_velocity: diagonal weight on the root-velocity error block.
        q_link: diagonal weight on the per-link direction errors.
        q_link_rate: diagonal weight on the per-link rate errors.
        q_attitude: diagonal weight on the per-vehicle attitude errors.
        q_angular_rate: diagonal weight on the per-vehicle body-rate errors.
        r_thrust, r_moment: input weights on each ``\\delta f_k`` and on each
            component of each ``\\delta M_k``.
        sag_ratio: horizontal span of the vehicle-to-vehicle span of the trim,
            as a fraction of the hose length between the outermost vehicles.
        horizontal_tension: bypasses ``sag_ratio`` and pins the trim's
            horizontal tension directly, in newtons.
        min_tension: floor under every *derived* link tension, passed to the
            flat map.
        setpoint: forwarded to :class:`~udaan.control.PDController`; the
            **root vertex** reference, ``t -> (pos, vel, acc, ...)``.  Worth
            passing, because the base class' default is a *fixed point* at
            ``(0, 0, 1)`` rather than "hold where you are".  So
            ``MultiQuadHoseBase(input="wrench").simulate(tf=..,
            root_position=[0, 0, 3])`` with nothing bound flies the root down
            to ``z = 1`` — the opposite of the ``ACCELERATION`` pipeline, whose
            per-vehicle PDs station-keep wherever ``reset()`` left them.

    Raises:
        ValueError: on any malformed parameter, and on ``kp``/``kd``/``ki``,
            which this controller has no PD to tune.
        udaan.flatness.NotFlatError: if no vehicle sits at the last vertex.
    """

    def __init__(
        self,
        *,
        num_links: int = DEFAULT_NUM_HOSE_LINKS,
        attachments=None,
        masses=None,
        inertias=None,
        vertex_masses=None,
        link_lengths=None,
        hose_length: float = DEFAULT_HOSE_LENGTH,
        hose_mass: float = DEFAULT_HOSE_MASS,
        gravity: float = GRAVITY,
        q_root_position: float = DEFAULT_HOSE_LQR_Q_ROOT_POSITION,
        q_root_velocity: float = DEFAULT_HOSE_LQR_Q_ROOT_VELOCITY,
        q_link: float = DEFAULT_HOSE_LQR_Q_LINK,
        q_link_rate: float = DEFAULT_HOSE_LQR_Q_LINK_RATE,
        q_attitude: float = DEFAULT_HOSE_LQR_Q_ATTITUDE,
        q_angular_rate: float = DEFAULT_HOSE_LQR_Q_ANGULAR_RATE,
        r_thrust: float = DEFAULT_HOSE_LQR_R_THRUST,
        r_moment: float = DEFAULT_HOSE_LQR_R_MOMENT,
        sag_ratio: float = DEFAULT_HOSE_SAG_RATIO,
        horizontal_tension: float | None = None,
        min_tension: float = DEFAULT_MIN_TENSION,
        **kwargs,
    ):
        # ``PDController`` would silently park these in ``self._gains``, which
        # nothing here reads — the exact class of bug M3, M4 and M5 document
        # having been bitten by.  There is no PD in this loop; say so.
        for name in ("kp", "kd", "ki"):
            if name in kwargs:
                raise ValueError(
                    f"{type(self).__name__} is an LQR, not a PD controller, so "
                    f"'{name}' has nothing to tune. Use the q_* / r_* cost weights "
                    f"(see udaan.core.defaults.DEFAULT_HOSE_LQR_*)."
                )
        super().__init__(**kwargs)

        n = int(num_links)
        if n < 1:
            raise ValueError(f"num_links must be >= 1, got {n}")
        self._num_links = n
        self._attachments = validate_attachments((0, n) if attachments is None else attachments, n)
        self._num_quadrotors = len(self._attachments)

        self._weights = self._cost_weights(
            q_root_position=q_root_position,
            q_root_velocity=q_root_velocity,
            q_link=q_link,
            q_link_rate=q_link_rate,
            q_attitude=q_attitude,
            q_angular_rate=q_angular_rate,
            r_thrust=r_thrust,
            r_moment=r_moment,
        )

        # ``Controller.__init__`` bound _g/_ge3/_gravity at the package default;
        # rebind all three together so a caller-supplied gravity cannot leave
        # them disagreeing.
        self._g = float(gravity)
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._gravity = np.array([0.0, 0.0, -self._g])

        ratio = float(sag_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"sag_ratio must lie in (0, 1); got {ratio}")
        self._sag_ratio = ratio
        if horizontal_tension is not None and not float(horizontal_tension) > 0.0:
            raise ValueError(
                f"horizontal_tension must be strictly positive; got {horizontal_tension}. "
                "A hose held between two vehicles with no horizontal pull has no taut trim."
            )
        self._horizontal_tension = None if horizontal_tension is None else float(horizontal_tension)

        self._min_tension = float(min_tension)
        if self._min_tension < 0.0:
            raise ValueError(f"min_tension must be non-negative; got {self._min_tension}")

        if link_lengths is None:
            link_lengths = np.full(n, float(hose_length) / n)
        if vertex_masses is None:
            vertex_masses = np.full(n + 1, float(hose_mass) / (n + 1))
        if masses is None:
            masses = np.full(self._num_quadrotors, DEFAULT_QUAD_MASS)
        if inertias is None:
            inertias = np.diag(np.asarray(DEFAULT_QUAD_INERTIA, dtype=float))

        self._set_parameters(
            masses=masses,
            inertias=inertias,
            vertex_masses=vertex_masses,
            link_lengths=link_lengths,
        )

    # ─── Parameter plumbing ────────────────────────────────────────

    @staticmethod
    def _cost_weights(**kwargs) -> dict[str, float]:
        """Validate the eight cost weights; every one must be strictly positive.

        A zero weight on a block is not merely soft, it is a singular ``R`` (for
        the input weights) or a ``Q`` that no longer detects part of the state,
        and the Riccati solution stops being the one this class documents.
        """
        out = {}
        for name, value in kwargs.items():
            weight = float(value)
            if not weight > 0.0:
                raise ValueError(f"{name} must be strictly positive; got {weight}")
            out[name] = weight
        return out

    def _broadcast(self, value, name: str, size: int) -> np.ndarray:
        arr = np.array(value, dtype=float)
        if arr.ndim == 0:
            return np.full(size, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != size:
            raise ValueError(f"{name} must be a scalar or have {size} entries; got {arr.shape[0]}")
        # ``np.array`` copies: a caller's array must never be aliased in.
        return arr

    def _set_parameters(self, *, masses, inertias, vertex_masses, link_lengths) -> None:
        """Validate the physical parameters and rebuild everything derived.

        The flat map, the trim, the linearisation and the gain are all cached,
        so they have to be rebuilt together; going through one routine is what
        keeps a re-synced controller identical to a freshly constructed one.
        """
        n = self._num_links
        nQ = self._num_quadrotors

        self._masses = self._broadcast(masses, "masses", nQ)
        if np.any(self._masses <= 0.0):
            raise ValueError(f"masses must be strictly positive; got {self._masses.tolist()}")

        tensors = np.array(inertias, dtype=float)
        if tensors.shape == (3,):
            tensors = np.tile(np.diag(tensors), (nQ, 1, 1))
        elif tensors.shape == (3, 3):
            tensors = np.tile(tensors, (nQ, 1, 1))
        elif tensors.shape != (nQ, 3, 3):
            raise ValueError(f"inertias must be (3,), (3, 3) or ({nQ}, 3, 3); got {tensors.shape}")
        self._inertias = tensors
        self._inertia_invs = np.linalg.inv(tensors)

        self._vertex_masses = self._broadcast(vertex_masses, "vertex_masses", n + 1)
        if np.any(self._vertex_masses < 0.0):
            raise ValueError(
                f"vertex_masses must be non-negative; got {self._vertex_masses.tolist()}"
            )
        self._link_lengths = self._broadcast(link_lengths, "link_lengths", n)
        if np.any(self._link_lengths <= 0.0):
            raise ValueError(
                f"link_lengths must be strictly positive; got {self._link_lengths.tolist()}"
            )

        total: np.ndarray = self._vertex_masses.copy()
        for k, vertex in enumerate(self._attachments):
            total[vertex] += self._masses[k]
        if not np.all(total > 0.0):
            raise ValueError(
                "every vertex needs strictly positive total mass (hose + airframe), or the "
                "links meeting there hinge freely and the mass matrix is singular; got "
                f"{total.tolist()} for attachments {self._attachments}"
            )
        self._vertex_total_masses = total

        # One flat map, built from *these* parameters.  ``simulate`` reads it off
        # this attribute rather than building its own from the model, so the
        # feedforward reference cannot describe a different machine from the one
        # the gain was solved for — M4's tension_policy lesson, transposed.
        # This is also where NotFlatError comes from: it is a statement about
        # the attachment pattern, and it belongs to the map.
        self._flat_map = MultiQuadHose(
            num_links=n,
            attachments=self._attachments,
            vertex_masses=self._vertex_masses.copy(),
            link_lengths=self._link_lengths.copy(),
            masses=self._masses.copy(),
            inertias=self._inertias.copy(),
            gravity=self._g,
            min_tension=self._min_tension,
        )
        self._trim_tensions = self._solve_trim_tensions()
        # The gain is *invalidated*, not re-solved: solving here would make
        # every ``mdl.masses = ...`` pay for a Riccati factorisation, and would
        # make a legitimate assignment on the *plant* fail whenever the
        # resulting linearisation happens to be too ill-conditioned.  Deferring
        # means such a machine can be built and integrated, and only refuses
        # when something actually asks this controller to fly it.
        self._invalidate_gain()

    def _invalidate_gain(self) -> None:
        """Drop the cached gain, so the next use re-solves it at the new parameters."""
        self._gain: np.ndarray | None = None
        self._closed_loop_poles: np.ndarray | None = None
        self._controllability_margin: float | None = None

    def _ensure_gain(self) -> np.ndarray:
        """The current gain, solving the Riccati equation if it is stale."""
        if self._gain is None:
            self._solve_gain()
        assert self._gain is not None  # _solve_gain either sets it or raises
        return self._gain

    def sync_from_model(self, model) -> None:
        """Re-read every physical parameter off ``model``, keeping the weights.

        The gain is solved from the masses, lengths and inertias, so
        ``mdl.vertex_masses = ...`` after the fact would otherwise leave this
        controller regulating a machine that no longer exists — and, worse,
        feeding forward a reference from a flat map built on the old one.  The
        model calls this from its own parameter setters.

        **An assigned** :attr:`trim_tensions` **does not survive this**: the
        trim is re-derived from the new masses and lengths like everything else
        derived.  That is the right default — a hand-set trim was chosen for
        the old machine, and silently carrying it onto a new one is the bug
        this method exists to prevent — but it does mean a caller who set the
        trim deliberately has to set it again after the model's parameters
        move.  Capture it first if that matters::

            saved = ctrl.trim_tensions.copy()
            mdl.vertex_masses = ...      # calls sync_from_model
            ctrl.trim_tensions = saved

        Raises:
            ValueError: if ``model`` has a different ``num_links`` or a
                different attachment pattern.  Every cached shape is sized by
                the two, and a controller for a different hose is a different
                controller.
        """
        if int(model.num_links) != self._num_links:
            raise ValueError(
                f"cannot sync a {self._num_links}-link controller from a "
                f"{model.num_links}-link model; rebuild it with from_model()"
            )
        if tuple(model.attachments) != self._attachments:
            raise ValueError(
                f"cannot sync a controller for attachments {self._attachments} from a model "
                f"with {tuple(model.attachments)}; rebuild it with from_model()"
            )
        self._set_parameters(
            masses=model.masses,
            inertias=model.inertias,
            vertex_masses=model.vertex_masses,
            link_lengths=model.link_lengths,
        )

    @classmethod
    def from_model(cls, model, **kwargs) -> MultiQuadHoseController:
        """Build from a
        :class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`-like model, so
        the controller and the plant agree on every parameter."""
        return cls(
            num_links=model.num_links,
            attachments=model.attachments,
            masses=np.asarray(model.masses, dtype=float).copy(),
            inertias=np.asarray(model.inertias, dtype=float).copy(),
            vertex_masses=np.asarray(model.vertex_masses, dtype=float).copy(),
            link_lengths=np.asarray(model.link_lengths, dtype=float).copy(),
            **kwargs,
        )

    # ─── Properties ────────────────────────────────────────────────

    @property
    def num_links(self) -> int:
        """``n``, the number of rigid links the hose is discretised into."""
        return self._num_links

    @property
    def num_vertices(self) -> int:
        """``n + 1``."""
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
    def masses(self) -> np.ndarray:
        """Per-vehicle airframe masses, ``(nQ,)``."""
        return self._masses

    @property
    def inertias(self) -> np.ndarray:
        """Per-vehicle inertia tensors, ``(nQ, 3, 3)``."""
        return self._inertias

    @property
    def vertex_masses(self) -> np.ndarray:
        """Hose mass lumped at each vertex, ``(n + 1,)`` — airframes excluded."""
        return self._vertex_masses

    @property
    def vertex_total_masses(self) -> np.ndarray:
        """``m̄_j = m_j + m_{Q_j}``, ``(n + 1,)`` — what the dynamics see."""
        return self._vertex_total_masses

    @property
    def link_lengths(self) -> np.ndarray:
        """Per-link lengths, ``(n,)``."""
        return self._link_lengths

    @property
    def total_mass(self) -> float:
        """``M₀₀`` — hose plus every airframe, the mass the team carries at hover."""
        return float(self._vertex_total_masses.sum())

    @property
    def cost_weights(self) -> dict[str, float]:
        """The eight LQR cost weights, as supplied. Read-only snapshot."""
        return dict(self._weights)

    @property
    def state_dim(self) -> int:
        """``6 + 6n + 6nQ`` — the size of the embedded error :meth:`error_state` returns."""
        return 6 + 6 * self._num_links + 6 * self._num_quadrotors

    @property
    def input_dim(self) -> int:
        """``4nQ`` — one thrust and three moment components per vehicle."""
        return 4 * self._num_quadrotors

    @property
    def trim_tensions(self) -> np.ndarray:
        """``(nQ - 1, 3)`` flat tensions defining the equilibrium reference.

        ``trim_tensions[k]`` is ``𝛕`` in the link hanging off vehicle ``k``,
        in the same order the flat map's
        :attr:`~udaan.flatness.MultiQuadHose.tension_vehicles` expects.  See
        :meth:`equilibrium_reference` for how the default is built.
        """
        return self._trim_tensions

    @trim_tensions.setter
    def trim_tensions(self, value):
        arr = np.asarray(value, dtype=float)
        expected = (self._num_quadrotors - 1, 3)
        if arr.shape != expected:
            raise ValueError(f"trim_tensions must have shape {expected}; got {arr.shape}")
        self._trim_tensions = arr.copy()
        self._invalidate_gain()

    @property
    def gain(self) -> np.ndarray:
        """The LQR gain ``K``, shape ``(4nQ, 6 + 6n + 6nQ)``.

        ``δu = K s``, added to the feedforward wrench.  Its columns on the
        ``2n`` gauge directions are exactly zero.  Solved lazily, on first use,
        from the parameters current at that moment.
        """
        return self._ensure_gain()

    @property
    def closed_loop_poles(self) -> np.ndarray:
        """Eigenvalues of the *reduced* closed loop ``A_r + B_r K_r``.

        Sorted by real part, most positive first, so ``[0]`` is the mode that
        limits the response.
        """
        self._ensure_gain()
        assert self._closed_loop_poles is not None
        return self._closed_loop_poles

    @property
    def controllability_margin(self) -> float:
        """``min_λ σ_min[A_r - λI  B_r]`` over the open-loop spectrum.

        The PBH test on the tangent-reduced pair: strictly positive means
        controllable.
        """
        self._ensure_gain()
        assert self._controllability_margin is not None
        return self._controllability_margin

    @property
    def flat_map(self) -> MultiQuadHose:
        """The :class:`udaan.flatness.MultiQuadHose` built from *these* parameters.

        Published so that
        :meth:`udaan.models.multi_quad_hose.MultiQuadHoseBase.simulate` can
        build the feedforward at this controller's own configuration.  A
        controller that does not expose it gets no feedforward at all rather
        than one built from the model's parameters, which may be a different
        machine.
        """
        return self._flat_map

    @property
    def jet_order(self) -> int:
        """``2n + 4`` — root-position jet depth the flat feedforward needs."""
        return self._flat_map.jet_order

    # ─── The equilibrium trim ──────────────────────────────────────

    def _segment_interior_weights(self) -> np.ndarray:
        """``(nQ - 1,)`` weights hanging between consecutive vehicles, in newtons.

        Entry ``k`` is ``g Σ m̄_j`` over the vertices **strictly** between
        vehicle ``k`` and vehicle ``k + 1``; zero for a one-link segment.
        """
        mbar = self._vertex_total_masses
        return np.array(
            [
                self._g * float(mbar[a + 1 : b].sum())
                for a, b in zip(self._attachments, self._attachments[1:], strict=False)
            ]
        )

    def _trim_span(self, horizontal: float) -> float:
        """Horizontal extent of the trim's inter-vehicle links at this ``H``."""
        mbar = self._vertex_total_masses
        weights = self._segment_interior_weights()
        span = 0.0
        for k, (a, b) in enumerate(zip(self._attachments, self._attachments[1:], strict=False)):
            tau = horizontal * _E1 - 0.5 * weights[k] * _E3
            for i in range(a, b):
                if i > a:
                    tau = tau + mbar[i] * self._g * _E3
                norm = float(np.linalg.norm(tau))
                # ``norm`` is zero only at ``H = 0`` on a segment with no
                # interior weight, which the bisection never returns; treating
                # the term as zero there keeps the span monotone.
                span += 0.0 if norm == 0.0 else self._link_lengths[i] * horizontal / norm
        return span

    def _solve_trim_tensions(self) -> np.ndarray:
        """The default flat tensions of :meth:`equilibrium_reference`.

        Every inter-vehicle segment is given the **same** horizontal tension
        ``H``, and the vertical component at the top of segment ``k`` is minus
        half that segment's interior weight, so the segment hangs symmetrically
        and comes back level.  Two consequences worth having:

        * every derived tension in the segment has ``‖𝛕‖ ≥ H > 0``, so no link
          can be slack or folded whatever the masses are;
        * the horizontal components cancel at every *interior* vehicle, which
          therefore hovers level and carries only weight.  Only the two
          outermost vehicles pull, with ``±H``.

        ``H`` itself is chosen so the trim spans ``sag_ratio`` of the hose
        length between the outermost vehicles, which puts the equilibrium close
        to the taut spawn shape
        (:func:`udaan.utils.hose.hose_spawn_directions`) the model resets to.
        The span is increasing in ``H``, so a bisection finds it — but only
        where the equation has a root at all.  A segment spanned by a *single*
        rigid link contributes its whole length whatever ``H`` is, so the span
        is bounded below by the total length of the single-link segments, and
        when ``sag_ratio`` asks for less than that no positive ``H`` answers.
        A pre-tension is used there instead and ``sag_ratio`` is ignored: with
        every segment one link long — ``attachments = (0, 1, …, n)``, say — the
        hose is straight and level at *any* pull, so the shape is fixed and
        only the pull is free.  See :meth:`_trim_pretension` and
        :meth:`_bisect_horizontal`.
        """
        nQ = self._num_quadrotors
        if nQ == 1:
            return np.zeros((0, 3))

        horizontal = self._horizontal_tension
        if horizontal is None:
            span_target = self._sag_ratio * float(
                self._link_lengths[self._attachments[0] : self._attachments[-1]].sum()
            )
            horizontal = self._bisect_horizontal(span_target)

        weights = self._segment_interior_weights()
        return np.stack([horizontal * _E1 - 0.5 * w * _E3 for w in weights])

    def _trim_pretension(self) -> float:
        """A strictly positive ``H`` for a trim whose *shape* ``H`` cannot change.

        Where every inter-vehicle segment is a single rigid link the hose is
        straight and level at any pull, so the sag criterion says nothing and
        the pull is a free pre-tension.  The hose's **own weight** is used: it
        is the scale of the tensions the sag search returns in every other
        pattern, and — unlike ``M₀₀ g`` — it does not grow with the number of
        vehicles bolted on, which at ``nQ = 5`` would ask the outermost two for
        50 N of thrust against a 20 N airframe.  Pin it with
        ``horizontal_tension=`` if the value matters.
        """
        hose_weight = self._g * float(self._vertex_masses.sum())
        # A massless hose is legal — only the vertex *totals* have to be
        # positive — and would leave nothing to scale by.
        if hose_weight > 0.0:
            return hose_weight
        return self._g * float(self._vertex_total_masses.max())

    def _bisect_horizontal(self, span_target: float) -> float:
        """Smallest ``H > 0`` whose trim spans ``span_target``.

        Both ends of the bracket have to be *found*, not assumed, and
        :meth:`_trim_pretension` is the answer when either search runs out.
        The upper one is the obvious case — no finite pull straightens a hose
        past its own length.  The lower one is the one that bites: the span
        does not go to zero with ``H``, it goes to the total length of the
        segments that are a single rigid link, since such a link spans its own
        length at any tension.  Taking ``H = 0`` as a valid lower bracket
        instead would hand back the ``high → 0`` limit of the bisection — a
        trim of ~1e-9 N that the flat map then rejects as slack, which is
        exactly what an ``attachments = (0, 1, …, n)`` hose used to produce.
        """
        high = self.total_mass * self._g
        for _ in range(_TRIM_BRACKET_DOUBLINGS):
            if self._trim_span(high) > span_target:
                break
            high *= 2.0
        else:
            # Taut past its own length: unreachable at any tension.
            return self._trim_pretension()
        low = high
        for _ in range(_TRIM_BRACKET_DOUBLINGS):
            low *= 0.5
            if self._trim_span(low) <= span_target:
                break
        else:
            # ``span_target`` is under the infimum — the single-link segments
            # alone already over-span it — so there is nothing to solve for.
            return self._trim_pretension()
        tolerance = _TRIM_SPAN_TOL * span_target
        while high - low > tolerance:
            mid = 0.5 * (low + high)
            if self._trim_span(mid) > span_target:
                high = mid
            else:
                low = mid
        return high

    # ─── Reference and error ───────────────────────────────────────

    def build_flats(self, root_jet) -> MultiQuadHoseFlats:
        """The flat output for a given **root-position jet**.

        The trajectory a caller binds supplies only ``x_0``; the rest of M6's
        flat output — one force vector per vehicle other than the last, plus
        the yaws — comes from this controller's own :attr:`trim_tensions` and a
        zero yaw.  Holding the internal forces constant while the root moves is
        a legitimate flat output, not an approximation: the derived tensions,
        the link geometry and every thrust adapt to the motion through the
        recursion.

        Args:
            root_jet: ``(K + 1, 3)`` array, row ``k`` the ``k``-th derivative of
                the root position, with ``K >= jet_order``.
        """
        rows = np.asarray(root_jet, dtype=float)
        tension_orders = self._flat_map.tension_jet_orders
        tensions = []
        for k, order in enumerate(tension_orders):
            data = np.zeros((order + 1, 3))
            data[0] = self._trim_tensions[k]
            tensions.append(Jet(data))
        return MultiQuadHoseFlats(
            x_0=Jet(rows),
            tensions=tensions,
            psi=[Jet(np.zeros(3)) for _ in range(self._num_quadrotors)],
        )

    def equilibrium_reference(self, t: float) -> tuple[MultiQuadHoseRefState, MultiQuadHoseInputs]:
        """The static trim at the setpoint — the feedback-only reference.

        The root sits at ``setpoint(t)``'s position and moves at its velocity;
        every higher derivative is taken as zero, which makes this an *exact*
        solution of the equations of motion (a trim translating uniformly), and
        the whole configuration — link directions, vehicle positions,
        attitudes, thrusts — comes out of the same flat map the feedforward
        path uses, at :attr:`trim_tensions`.

        The setpoint's *acceleration* is deliberately not used: an accelerating
        reference is not a trim, and pretending otherwise would smuggle in half
        a feedforward and make the with/without comparison meaningless.
        Accelerating references are what :attr:`flat_map` is for.

        Raises:
            ValueError: if the flat map refuses the trim as slack.  Every
                derived tension is bounded below by the horizontal pull ``H``,
                so at the default :attr:`trim_tensions` this needs an ``H``
                under ``min_tension`` — which the sag search cannot produce at
                the default ``sag_ratio``, but a ``sag_ratio`` asking for
                barely more than the single-link segments already span can.
                An assigned :attr:`trim_tensions` has no such guarantee.
        """
        setpoint = self.setpoint(t)
        rows = np.zeros((self.jet_order + 1, 3))
        rows[0] = np.asarray(setpoint[0], dtype=float)
        rows[1] = np.asarray(setpoint[1], dtype=float)
        return self._flat_map.recover(self.build_flats(rows))

    def error_state(self, state, reference: MultiQuadHoseRefState) -> np.ndarray:
        r"""The ``(6 + 6n + 6nQ,)`` variational error.

        Six blocks, in this order:

        .. math::

            e_{x_0} &= x_0 - x_0^d, \qquad e_{v_0} = v_0 - v_0^d, \\
            \xi_i &= q_i^d \times q_i, \qquad
            \delta\omega_i = \omega_i + q_i \times (q_i \times \omega_i^d), \\
            \eta_k &= \tfrac12 \operatorname{vee}(R_k^{d\top} R_k
                      - R_k^\top R_k^d), \qquad
            \delta\Omega_k = \Omega_k - R_k^\top R_k^d \Omega_k^d .

        Args:
            state: a
                :class:`~udaan.models.multi_quad_hose.MultiQuadHoseState`.
            reference: the matching
                :class:`~udaan.flatness.MultiQuadHoseRefState`.

        Raises:
            ValueError: if the state or the reference describes a different
                hose.
        """
        n = self._num_links
        nQ = self._num_quadrotors
        if len(state.link_attitudes) != n:
            raise ValueError(
                f"state has {len(state.link_attitudes)} links, but this controller has {n}"
            )
        if len(reference.link_attitudes) != n:
            raise ValueError(
                f"reference has {len(reference.link_attitudes)} links, but this controller has {n}"
            )
        if len(state.orientations) != nQ:
            raise ValueError(
                f"state has {len(state.orientations)} vehicles, but this controller has {nQ}"
            )
        if len(reference.orientations) != nQ:
            raise ValueError(
                f"reference has {len(reference.orientations)} vehicles, but this "
                f"controller has {nQ}"
            )

        error = np.zeros(self.state_dim)
        error[0:3] = np.asarray(state.root_position, dtype=float) - np.asarray(
            reference.root_position, dtype=float
        )
        error[3:6] = np.asarray(state.root_velocity, dtype=float) - np.asarray(
            reference.root_velocity, dtype=float
        )
        for i in range(n):
            q = np.asarray(state.link_attitudes[i], dtype=float)
            omega = np.asarray(state.link_angular_velocities[i], dtype=float)
            q_d = np.asarray(reference.link_attitudes[i], dtype=float)
            omega_d = np.asarray(reference.link_angular_velocities[i], dtype=float)
            error[6 + 3 * i : 9 + 3 * i] = np.cross(q_d, q)
            error[6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i] = omega + np.cross(q, np.cross(q, omega_d))
        base = 6 + 6 * n
        for k in range(nQ):
            R = np.asarray(state.orientations[k], dtype=float)
            Omega = np.asarray(state.angular_velocities[k], dtype=float)
            R_d = np.asarray(reference.orientations[k], dtype=float)
            Omega_d = np.asarray(reference.angular_velocities[k], dtype=float)
            error[base + 3 * k : base + 3 * k + 3] = 0.5 * vee(R_d.T @ R - R.T @ R_d)
            error[base + 3 * nQ + 3 * k : base + 3 * nQ + 3 * k + 3] = Omega - R.T @ R_d @ Omega_d
        return error

    # ─── Linearisation and gain ────────────────────────────────────

    def _mass_coefficients(self) -> tuple[float, np.ndarray, np.ndarray]:
        """``(M₀₀, M₀ᵢ, Mᵢⱼ)`` of the M6 derivation, from *this* controller's parameters.

        Deliberately recomputed rather than read off a model: the whole point of
        :meth:`sync_from_model` is that this object knows its own machine.
        """
        mbar = self._vertex_total_masses
        length = self._link_lengths
        tail = np.cumsum(mbar[::-1])[::-1]
        S = tail[1:]
        M00 = float(mbar.sum())
        M0i = S * length
        n = self._num_links
        Mij = S[np.maximum.outer(np.arange(n), np.arange(n))] * np.outer(length, length)
        return M00, M0i, Mij

    def linearize(
        self,
        reference: MultiQuadHoseRefState | None = None,
        inputs: MultiQuadHoseInputs | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Variational linearisation ``(A, B)``, ``ṡ = A s + B δu``.

        ``A`` is ``(6 + 6n + 6nQ)``-square and ``B`` is
        ``(6 + 6n + 6nQ) x 4nQ``, in the block order of :meth:`error_state`;
        the input is stacked thrusts first, then the moments,
        ``[δf_0 … δf_{nQ-1}, δM_0, … δM_{nQ-1}]``.

        With no arguments the reference is :meth:`equilibrium_reference` at
        ``t = 0``, which is the operating point the shipped gain is solved at.

        Args:
            reference: a :class:`~udaan.flatness.MultiQuadHoseRefState`.
            inputs: the matching :class:`~udaan.flatness.MultiQuadHoseInputs`,
                read for the reference thrusts.

        Raises:
            ValueError: if only one of the two is supplied, or if either
                describes a different hose.
        """
        if (reference is None) != (inputs is None):
            raise ValueError(
                "linearize() needs the reference state and its inputs together: the "
                "link rows depend on the reference thrusts. Pass both, or neither for "
                "the equilibrium trim."
            )
        if reference is None or inputs is None:
            reference, inputs = self.equilibrium_reference(0.0)

        n = self._num_links
        nQ = self._num_quadrotors
        if len(reference.link_attitudes) != n:
            raise ValueError(
                f"reference describes {len(reference.link_attitudes)} links, but this "
                f"controller has {n}"
            )
        if len(inputs.thrusts) != nQ:
            raise ValueError(
                f"inputs describe {len(inputs.thrusts)} vehicles, but this controller has {nQ}"
            )

        eye = np.eye(3)
        M00, M0i, Mij = self._mass_coefficients()
        lengths = self._link_lengths

        qd = np.asarray(reference.link_attitudes, dtype=float).reshape(n, 3)
        wd = np.asarray(reference.link_angular_velocities, dtype=float).reshape(n, 3)
        dwd = np.asarray(reference.link_angular_accelerations, dtype=float).reshape(n, 3)
        R_d = [np.asarray(R, dtype=float) for R in reference.orientations]
        Omega_d = [np.asarray(w, dtype=float) for w in reference.angular_velocities]
        thrust_d = np.asarray(inputs.thrusts, dtype=float).reshape(nQ)
        # α = ẍ_0 + g e₃ — gravity folded in, as the model's `_assemble` does.
        alpha_d = np.asarray(reference.root_acceleration, dtype=float) + self._ge3

        qd_hat = [hat(qd[i]) for i in range(n)]
        speed_sq = np.array([float(np.dot(wd[i], wd[i])) for i in range(n)])

        # Reference thrust vectors, per vertex, and the cumulative force below
        # each link — the ``G_i`` of the model's derivation.
        vertex_force = np.zeros((n + 1, 3))
        for k, vertex in enumerate(self._attachments):
            vertex_force[vertex] = thrust_d[k] * (R_d[k] @ _E3)
        below = np.cumsum(vertex_force[::-1], axis=0)[::-1]

        dim = self.state_dim
        xi0 = 6
        rate0 = 6 + 3 * n
        eta0 = 6 + 6 * n
        omega0 = eta0 + 3 * nQ

        # ── E: the mass matrix of the equations of motion, at the reference ──
        mass = np.zeros((3 + 3 * n, 3 + 3 * n))
        mass[0:3, 0:3] = M00 * eye
        for i in range(n):
            r = 3 * (i + 1)
            mass[0:3, r : r + 3] = -M0i[i] * qd_hat[i]
            mass[r : r + 3, 0:3] = M0i[i] * qd_hat[i]
            for j in range(n):
                c = 3 * (j + 1)
                mass[r : r + 3, c : c + 3] = (
                    Mij[i, j] * eye if i == j else -Mij[i, j] * (qd_hat[i] @ qd_hat[j])
                )

        # ── F: the state Jacobian of the same right-hand side ──
        state_jac = np.zeros((3 + 3 * n, dim))
        for i in range(n):
            link_col = xi0 + 3 * i
            rate_col = rate0 + 3 * i
            state_jac[0:3, link_col : link_col + 3] = (
                M0i[i] * (hat(dwd[i]) - speed_sq[i] * eye) @ qd_hat[i]
            )
            state_jac[0:3, rate_col : rate_col + 3] = 2.0 * M0i[i] * np.outer(qd[i], wd[i])
        for k in range(nQ):
            col = eta0 + 3 * k
            state_jac[0:3, col : col + 3] = -thrust_d[k] * R_d[k] @ hat(_E3)

        for i in range(n):
            r = 3 * (i + 1)
            block = -M0i[i] * hat(alpha_d) + lengths[i] * hat(below[i + 1])
            for k in range(n):
                if k == i:
                    continue
                block = block + Mij[i, k] * (hat(qd_hat[k] @ dwd[k]) + speed_sq[k] * qd_hat[k])
            state_jac[r : r + 3, xi0 + 3 * i : xi0 + 3 * i + 3] = block @ qd_hat[i]
            for j in range(n):
                if j == i:
                    continue
                link_col = xi0 + 3 * j
                rate_col = rate0 + 3 * j
                state_jac[r : r + 3, link_col : link_col + 3] = (
                    Mij[i, j] * qd_hat[i] @ (hat(dwd[j]) - speed_sq[j] * eye) @ qd_hat[j]
                )
                state_jac[r : r + 3, rate_col : rate_col + 3] = (
                    2.0 * Mij[i, j] * qd_hat[i] @ np.outer(qd[j], wd[j])
                )
            # Only the vehicles strictly below link ``i`` push on it.
            for k, vertex in enumerate(self._attachments):
                if vertex <= i:
                    continue
                col = eta0 + 3 * k
                state_jac[r : r + 3, col : col + 3] = (
                    -lengths[i] * thrust_d[k] * qd_hat[i] @ R_d[k] @ hat(_E3)
                )

        # ── G: the input Jacobian of the same right-hand side ──
        input_jac = np.zeros((3 + 3 * n, self.input_dim))
        for k, vertex in enumerate(self._attachments):
            input_jac[0:3, k] = R_d[k] @ _E3
            for i in range(n):
                if vertex <= i:
                    continue
                r = 3 * (i + 1)
                input_jac[r : r + 3, k] = lengths[i] * (qd_hat[i] @ R_d[k] @ _E3)

        solved_state = np.linalg.solve(mass, state_jac)
        solved_input = np.linalg.solve(mass, input_jac)

        A = np.zeros((dim, dim))
        A[0:3, 3:6] = eye
        A[3:6, :] = solved_state[0:3, :]
        for i in range(n):
            r = xi0 + 3 * i
            A[r : r + 3, xi0 + 3 * i : xi0 + 3 * i + 3] = np.outer(qd[i], qd[i]) @ hat(wd[i])
            A[r : r + 3, rate0 + 3 * i : rate0 + 3 * i + 3] = eye - np.outer(qd[i], qd[i])
        A[rate0 : rate0 + 3 * n, :] = solved_state[3:, :]
        for k in range(nQ):
            r = eta0 + 3 * k
            A[r : r + 3, r : r + 3] = -hat(Omega_d[k])
            A[r : r + 3, omega0 + 3 * k : omega0 + 3 * k + 3] = eye
            s = omega0 + 3 * k
            A[s : s + 3, s : s + 3] = self._inertia_invs[k] @ (
                hat(self._inertias[k] @ Omega_d[k]) - hat(Omega_d[k]) @ self._inertias[k]
            )

        B = np.zeros((dim, self.input_dim))
        B[3:6, :] = solved_input[0:3, :]
        B[rate0 : rate0 + 3 * n, :] = solved_input[3:, :]
        for k in range(nQ):
            s = omega0 + 3 * k
            B[s : s + 3, nQ + 3 * k : nQ + 3 * k + 3] = self._inertia_invs[k]
        return A, B

    def tangent_projection(self, link_directions=None) -> np.ndarray:
        r"""``W``, the ``(6 + 6n + 6nQ) x (6 + 4n + 6nQ)`` non-gauge basis.

        The six root coordinates and the ``6nQ`` rigid-body ones pass through
        untouched; each link contributes the two directions of
        ``q_i^{d\perp}`` and drops the one along ``q_i^d``.  ``B``'s rows on the
        dropped directions vanish identically at *every* reference, so
        ``B_r = WᵀB`` never loses anything; ``A_r = WᵀAW`` is lossless too at
        any **static** trim, which is where the shipped gain is solved.
        """
        n = self._num_links
        nQ = self._num_quadrotors
        if link_directions is None:
            link_directions = np.tile(-_E3, (n, 1))
        directions = np.asarray(link_directions, dtype=float).reshape(n, 3)
        basis = np.zeros((self.state_dim, 6 + 4 * n + 6 * nQ))
        basis[0:6, 0:6] = np.eye(6)
        for i in range(n):
            tangent = _tangent_basis(directions[i])
            basis[6 + 3 * i : 9 + 3 * i, 6 + 2 * i : 8 + 2 * i] = tangent
            basis[
                6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i,
                6 + 2 * n + 2 * i : 8 + 2 * n + 2 * i,
            ] = tangent
        basis[6 + 6 * n :, 6 + 4 * n :] = np.eye(6 * nQ)
        return basis

    def cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Q_r, R)`` — the reduced state cost and the input cost.

        ``Q_r`` is ``(6 + 4n + 6nQ)``-square: the six block weights of
        :attr:`cost_weights`, with the link blocks carrying two entries per
        link rather than three, since the third is the gauge direction the
        projection removes.
        """
        n = self._num_links
        nQ = self._num_quadrotors
        w = self._weights
        diagonal = np.concatenate(
            [
                np.full(3, w["q_root_position"]),
                np.full(3, w["q_root_velocity"]),
                np.full(2 * n, w["q_link"]),
                np.full(2 * n, w["q_link_rate"]),
                np.full(3 * nQ, w["q_attitude"]),
                np.full(3 * nQ, w["q_angular_rate"]),
            ]
        )
        inputs = np.concatenate([np.full(nQ, w["r_thrust"]), np.full(3 * nQ, w["r_moment"])])
        return np.diag(diagonal), np.diag(inputs)

    def _solve_gain(self) -> None:
        """Solve the algebraic Riccati equation at the equilibrium trim."""
        reference, inputs = self.equilibrium_reference(0.0)
        A, B = self.linearize(reference, inputs)
        self.set_gain_from(A, B, reference.link_attitudes)

    def set_gain_from(self, A: np.ndarray, B: np.ndarray, link_directions=None) -> None:
        r"""Re-solve the Riccati equation at a given linearisation and adopt the gain.

        The hook a caller schedules gains through::

            ref, ff = ...
            ctrl.set_gain_from(*ctrl.linearize(ref, ff), ref.link_attitudes)

        The solve runs on the tangent-reduced pair
        (:meth:`tangent_projection`) and the resulting ``K_r`` is pushed back
        out as ``K = K_r Wᵀ``, so the returned gain never acts on a gauge
        direction.

        **The reduction is exact at a static trim and approximate off one.**
        See "Gauge directions" in the module docstring.

        Raises:
            ValueError: if the pair is not stabilisable on the reduced
                subspace, or if the Riccati solver returns a solution that does
                not survive :func:`~udaan.control.solve_lqr_gain`'s residual
                and stability checks — which on this plant is not hypothetical,
                a hose with a wide enough spread of vertex masses and link
                lengths producing exactly that.
        """
        basis = self.tangent_projection(link_directions)
        A_r = basis.T @ A @ basis
        B_r = basis.T @ B
        Q_r, R = self.cost_matrices()
        gain_r, poles = solve_lqr_gain(
            A_r,
            B_r,
            Q_r,
            R,
            describe=(
                f"this {self._num_links}-link hose at attachments {self._attachments} "
                f"(vertex masses {self._vertex_total_masses.tolist()}, lengths "
                f"{self._link_lengths.tolist()})"
            ),
        )
        self._gain = gain_r @ basis.T
        self._closed_loop_poles = poles
        self._controllability_margin = self._pbh_margin(A_r, B_r)

    @staticmethod
    def _pbh_margin(A_r: np.ndarray, B_r: np.ndarray) -> float:
        """``min_λ σ_min[A_r - λI  B_r]`` over the open-loop spectrum."""
        size = A_r.shape[0]
        margin = np.inf
        for eigenvalue in np.linalg.eigvals(A_r):
            stacked = np.hstack([A_r - eigenvalue * np.eye(size), B_r.astype(complex)])
            margin = min(margin, float(np.linalg.svd(stacked, compute_uv=False).min()))
        return margin

    # ─── Control ───────────────────────────────────────────────────

    def compute(self, *args) -> np.ndarray:
        """The ``(nQ, 4)`` wrench stack ``[f_k, M_kx, M_ky, M_kz]``.

        Args:
            args: ``(t, state)``, or ``(t, state, reference)`` where
                ``reference`` is the ``(RefState, Inputs)`` pair recovered by
                :attr:`flat_map`.  Without it the static trim at the setpoint
                is used.
        """
        t = args[0]
        state = args[1]
        reference = args[2] if len(args) > 2 and args[2] is not None else None
        if reference is None:
            ref, feedforward = self.equilibrium_reference(t)
        else:
            ref, feedforward = reference

        nQ = self._num_quadrotors
        correction = self._ensure_gain() @ self.error_state(state, ref)
        wrench = np.zeros((nQ, 4))
        wrench[:, 0] = np.asarray(feedforward.thrusts, dtype=float) + correction[:nQ]
        wrench[:, 1:] = np.asarray(feedforward.moments, dtype=float) + correction[nQ:].reshape(
            nQ, 3
        )
        return wrench
