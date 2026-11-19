(control-multi-quad-hose)=
# Steering a flexible hose with several quadrotors

{py:class}`udaan.control.multi_quad_hose.MultiQuadHoseController` regulates
$n_Q$ quadrotors bolted at chosen vertices of **one** flexible hose,
discretised into $n$ rigid links. It closes the loop on the dynamics of
{doc}`../dynamics/multi-quad-hose` and is fed forward by the flat map of
{doc}`../differential-flatness/multi-quad-hose` {footcite}`kotaru2020hose`.

Like {doc}`quadrotor-flexcable` and for the same structural reason, it is
**not** a geometric cascade: a serial chain has no per-link body to command.
Adding vehicles does not restore one — a vehicle is bolted to a *vertex*, and
the links on either side of it still move only through the chain — so the
published law for this system, and this implementation, is a linear-quadratic
regulator on the variational linearisation.

## What the loop regulates

The reference is not a set of vehicle positions. M6's flat output is

$$
y \;=\; \bigl(x_0,\; \{\boldsymbol\tau_a\}_{a \in \mathcal{A},\, a \ne n},\;
        \{\psi_k\}_{k=1}^{n_Q}\bigr),
$$

the **root vertex** position, one vector tension per vehicle other than the
one at the last vertex, and a yaw each — see
{doc}`../differential-flatness/multi-quad-hose`. The controller's `setpoint`
supplies $x_0$; the tensions come from the controller's own
{py:attr}`~udaan.control.multi_quad_hose.MultiQuadHoseController.trim_tensions`
and the yaws are zero. Everything else — every link direction, every vehicle
position and attitude, every thrust and moment — is recovered by the map.

A consequence worth stating: **the vehicle positions are outputs, not
commands.** Steering a particular vehicle to a particular point is an inverse
problem; the reference implementation solves it numerically in
`optimize_for_des_state`, and it is a planner's job rather than this loop's.

(m6-trim)=
## The equilibrium reference

With no trajectory bound, the loop regulates against a **static trim** at the
setpoint, and that trim is produced by the same flat map, evaluated at a flat
output whose derivatives are all zero. This is exact rather than approximate:
a trim translating at constant velocity is a genuine solution of the equations
of motion, so the feedback-only reference costs the closed loop nothing in
consistency.

Which trim? A hose held between two vehicles has no *vertical* equilibrium
with every link taut — spreading the thrust drives some link's tension through
zero and folds the hose there
({py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.hover_trim`
documents why). The trim used here is the discrete catenary instead. Every
inter-vehicle segment is given the same horizontal tension $H$, and the
tension at the top of segment $k$ carries minus half that segment's interior
weight $W_k$:

$$
\boldsymbol\tau_{a_k} \;=\; H\,e_1 \;-\; \tfrac12 W_k\, e_3 ,
\qquad
\boldsymbol\tau_j \;=\; \boldsymbol\tau_{j-1} + \bar m_j\, g\, e_3
\ \ \text{on the interior vertices},
$$

so the segment hangs symmetrically and returns to its starting height. Two
properties follow, and both are what make this the right default:

* every tension in the segment satisfies $\|\boldsymbol\tau_i\| \ge H > 0$, so
  **no link can be slack or folded**, whatever the mass distribution;
* the horizontal components cancel at every *interior* vehicle, which
  therefore hovers level and carries only weight. Only the two outermost
  vehicles pull, with $\pm H$.

$H$ itself is chosen so the trim spans `sag_ratio` of the hose length between
the outermost vehicles, which places the equilibrium close to the taut spawn
shape both backends reset to. The span increases with $H$, so a bisection finds
it — **where the equation has a root at all**, which is not everywhere.

A segment spanned by a single rigid link contributes its whole length at *any*
tension, so the span is bounded below by the total length of the single-link
segments, and a `sag_ratio` under 1 asks for less than that whenever every
segment is one link long — `attachments = (0, 1, ..., n)`. The hose is then
straight and level whatever the pull, and $H$ is a free pre-tension rather than
something the sag criterion determines. It is set to the **hose's own weight**:
the scale of the tensions the search returns in every other pattern, and
independent of how many vehicles are bolted on, unlike $M_{00}g$, which at
$n_Q = 5$ would ask the outermost two for 50 N against a 20 N airframe. Pass
`horizontal_tension=` to pin it.

The search does not treat $H = 0$ as a valid lower bracket: where the span
has no root, a bisection bracketed only from above converges toward a
zero-tension trim, a slack hose that the flat map rejects. The single-link
patterns are detected and given the pre-tension above instead.

## Error state

Six blocks, in the conventions M2–M5 already use:

$$
s = \begin{bmatrix}
e_{x_0} \\ e_{v_0} \\ \xi \\ \delta\omega \\ \eta \\ \delta\Omega
\end{bmatrix},
\qquad
\begin{aligned}
e_{x_0} &= x_0 - x_0^d, &  e_{v_0} &= v_0 - v_0^d, \\
\xi_i &= q_i^d \times q_i, &
\delta\omega_i &= \omega_i + q_i \times (q_i \times \omega_i^d), \\
\eta_k &= \tfrac12 \operatorname{vee}\bigl(R_k^{d\top} R_k
          - R_k^\top R_k^d\bigr), &
\delta\Omega_k &= \Omega_k - R_k^\top R_k^d\, \Omega_k^d .
\end{aligned}
$$

The translational block is the **root vertex's**, which is the row of the
equations of motion the collected thrusts act on and the one the flat output
prescribes. Every other vertex — the vehicles included — is driven to its
reference through the $\xi_i$, since the chain recursion
$x_{j} = x_0 + \sum_{i<j}\ell_i q_i$ makes the link directions the rest of the
configuration.

The block *ordering* here differs from the reference MATLAB's
$[e_x;\ \xi;\ e_v;\ \delta\omega;\ \eta;\ \delta\Omega]$ (`error.m` L47). The
grouping used above is the one its own cost matrix assumes; see
{ref}`m6-errata`.

## Variational linearisation

Write the equations of motion as $\mathcal{M}(q)\,z = b(q, \omega, u)$ with
$z = [\alpha;\ \dot\omega_1;\ \dots;\ \dot\omega_n]$ and
$\alpha = \ddot x_0 + g e_3$ — the form
{py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.accelerations`
solves. Perturbing about a reference, with $\delta q_i = -\hat q_i^d \xi_i$
and $\delta(f_k R_k e_3) = R_k^d e_3\,\delta f_k
- f_k^d R_k^d \hat e_3\, \eta_k$, gives

$$
\mathcal{M}(q^d)
\begin{bmatrix} \delta \dot v_0 \\ \delta\dot\omega \end{bmatrix}
= F\,s + G\,\delta u ,
$$

whose rows are, for the $\alpha$ equation,

$$
\frac{\partial}{\partial \xi_i} = M_{0i}\bigl(\hat{\dot\omega}_i^d
   - \|\omega_i^d\|^2 I\bigr)\hat q_i^d, \qquad
\frac{\partial}{\partial \delta\omega_i} = 2 M_{0i}\, q_i^d \omega_i^{d\top},
\qquad
\frac{\partial}{\partial \eta_k} = -f_k^d R_k^d \hat e_3 ,
$$

and for link row $i$,

$$
\frac{\partial}{\partial \xi_i} =
  \Bigl[-M_{0i}\hat\alpha^d + \ell_i \hat G_i
  + \sum_{k \ne i} M_{ik}\bigl(\widehat{q_k^d \times \dot\omega_k^d}
    + \|\omega_k^d\|^2 \hat q_k^d\bigr)\Bigr]\hat q_i^d ,
$$

$$
\frac{\partial}{\partial \xi_j} = M_{ij}\hat q_i^d
   \bigl(\hat{\dot\omega}_j^d - \|\omega_j^d\|^2 I\bigr)\hat q_j^d,
\qquad
\frac{\partial}{\partial \delta\omega_j} =
   2 M_{ij}\hat q_i^d q_j^d \omega_j^{d\top}
\qquad (j \ne i),
$$

$$
\frac{\partial}{\partial \eta_k} = -\ell_i f_k^d\,\hat q_i^d R_k^d \hat e_3
\quad\text{for every vehicle at a vertex } > i ,
\qquad
\frac{\partial}{\partial \delta f_k} = \ell_i\, \hat q_i^d R_k^d e_3
\quad\text{likewise} ,
$$

with $G_i = \sum_{j>i} f_j^d R_j^d e_3$ the net thrust below link $i$. The
remaining rows are kinematic:
$\dot\xi_i = q_i^d q_i^{d\top}\hat\omega_i^d\,\xi_i
+ (I - q_i^d q_i^{d\top})\delta\omega_i$,
$\dot\eta_k = -\hat\Omega_k^d \eta_k + \delta\Omega_k$ and
$\delta\dot\Omega_k = J_k^{-1}\bigl(\widehat{J_k \Omega_k^d}
- \hat\Omega_k^d J_k\bigr)\delta\Omega_k + J_k^{-1}\delta M_k$.

This is the reference's `get_variation_lin_dyn` block for block, with the two
conventions reconciled: the reference carries gravity explicitly in the link
rows and this implementation folds it into $\alpha$, which is the same
statement because $\ell_i S_i = M_{0i}$.

### It is the *variational* linearisation, and that is not the same as the Jacobian of the error dynamics

Both $\delta\omega_i$ and $\delta\Omega_k$ above are the reference rate
**transported to the current configuration**, not the reference rate itself.
The blocks written above are what you get by treating them as if they were —
$\omega_i = \omega_i^d + \delta\omega_i$, $\Omega_k = \Omega_k^d +
\delta\Omega_k$ — which is the published derivation and the one every
geometric-control paper in this lineage uses.

Against finite differences of the true nonlinear error dynamics, driven by
{py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.accelerations`, the
two agree to finite-difference precision **at any static trim**, where
$\omega^d = \Omega^d = 0$ kills the difference exactly. At a moving reference
they differ by $O(\|\omega^d\|, \|\Omega^d\|)$. Since the shipped gain is solved at the trim, the gain rests on the
exact case; a scheduled gain at a moving reference does not, and the leak is
of the same order as the gauge leak below.

## Gauge directions and the tangent reduction

The embedded error has $6 + 6n + 6n_Q$ entries but the system has
$6 + 4n + 6n_Q$ degrees of freedom. Each $\xi_i$ is a cross product with
$q_i^d$ and each $\delta\omega_i$ is tangent to $q_i$, so $2n$ of the
coordinates — the components along $q_i^d$ — are the $S^2$ gauge, exactly as
$\dot\omega_i \cdot q_i$ is in the dynamics. Handing the embedded pair to
{py:func}`scipy.linalg.solve_continuous_are` asks it to stabilise $2n$
integrators no input reaches, and no stabilising solution exists. This is the
same structural property as M5's ({doc}`quadrotor-flexcable`).

How the solver reports it depends on the plant. On M5's embedded pair
{py:func}`scipy.linalg.solve_continuous_are` raises. On M6's it may instead
return a $P$ of enormous norm whose closed loop still has eigenvalues on the
imaginary axis; which configurations do which is a property of an extremely
ill-conditioned pencil and depends on the solver version. The tests therefore
assert the version-independent statement — no gain from the embedded pair is
strictly stabilising — rather than watching for the exception.

The same solver behaviour is why the *reduced* solve is checked rather than
trusted — see {py:func}`udaan.control.solve_lqr_gain`, which measures the
Riccati residual against $\|Q\|$ and confirms the closed loop is strictly
stable before adopting a gain. `solve_continuous_are` does not refuse every
pair it cannot usefully solve, and a hose with a wide enough spread of vertex
masses and link lengths — all of it legal — is exactly what gets a silently
useless $P$ back.

The mass matrix has the gauge vectors as *eigenvectors*: for
$e$ with block $i$ equal to $q_i$ and every other block zero,
$\mathcal{M}e = M_{ii} e$, because every off-diagonal block in column $i$
carries $\hat q_i$ as its **right** factor and so annihilates $q_i$. Since
$\mathcal{M}$ is symmetric,

$$
q_i^\top \bigl[\mathcal{M}^{-1} X\bigr]_i
   = e^\top \mathcal{M}^{-1} X
   = \tfrac{1}{M_{ii}}\, q_i^\top X_i ,
$$

so the gauge component of the solved link row is the gauge component of the
*right-hand side*, divided by $M_{ii}$. Every column of $G$ in link row $i$ is
a $\hat q_i(\cdot)$, so **$B$'s gauge rows vanish identically, at every
reference**. The Riccati solve therefore runs on the tangent-reduced
$(6 + 4n + 6n_Q)$ pair and the gain is pushed back through the same
projection.

$A$'s gauge rows do not vanish in general, but they do **at any static trim** —
a stronger statement than M5's, which is about a hanging equilibrium. At a
trim the link row's own $\xi_i$ block is
$[-M_{0i}\hat\alpha^d + \ell_i \hat G_i]\hat q_i^d = -\hat v_i \hat q_i^d$
with $v_i = M_{0i}\alpha^d - \ell_i G_i$, and the static condition
$\hat q_i v_i = 0$ says exactly $v_i \parallel q_i$, whence
$-\lambda \hat q_i \hat q_i = \lambda(I - q_iq_i^\top)$, whose $q_i$ row is
zero, so $\|(I - WW^\top)AW\| / \|A\|$ vanishes to round-off at the trim.
Off it, the leak is set by the reference's aggressiveness and grows with it.

## Horizon

The reference uses a **finite-horizon** LQR: it integrates the Riccati
equation backwards with `ode45` over $[t_f, t_0]$ along one specific
trajectory, caches $P(t)$ into a `.mat` file, and interpolates at run time.
This implementation solves the **infinite-horizon** problem instead, once,
about the static trim. The backward sweep needs the entire reference in
advance and produces a schedule tied to one manoeuvre; a controller in this
package has to work against a `setpoint` callable it has never seen.

The cost is that the linearisation moves with the reference while the gain
does not. Against `CrazyTrajectory` references of increasing amplitude and
frequency at $n = 3$, vehicles at $(0, 3)$, the relative drift
$\|A - A_0\|/\|A_0\|$ from the trim linearisation $A_0$ stays within a few
percent on a gentle reference ($a_x = f_1 = 0.15$), reaches tens of percent on a
moderate one $(0.4, 0.3)$ and exceeds 100 % on an aggressive one
$(0.8, 0.45)$ (`test_drift_over_trajectory_aggressiveness` pins these bands).
With the feedforward carrying the reference, a gain solved once at the trim
tracks the gentle and moderate references closely. Nothing warns when a
reference leaves that range: M6's flat map does not refuse the aggressive
cases, since at $n_Q \ge 2$ the hanging-branch test is off by construction — a
hose held at both ends *has* links pointing upwards. On the most aggressive
references the 200 Hz zero-order-hold loop with four Euler substeps and the
20 N actuator limits are themselves limiting, independently of the gain.

Gain scheduling is two lines away —
`ctrl.set_gain_from(*ctrl.linearize(ref, ff), ref.link_attitudes)` — and is
deliberately not automatic: a Riccati solve costs milliseconds against a 5 ms
control period, and it buys nothing on a gentle reference.

## Tracking

Feedforward is what carries the reference; the gain regulates a small error
about it. Over a gentle Lissajous started on the reference, feedforward
reduces root-position tracking error by well over the factor of five the tests
require, in every attachment pattern tested, and a 0.44 m step on the root from
the spawn shape settles to within the physical tolerance with the hose ending
in the trim's shape.

"Started on the reference" means the **whole** state — root position and
velocity, every link direction and rate, every attitude and body rate — taken
from the flat map at $t = 0$, which is what
{py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.reset` accepts and
what the tests do. Starting with only `root_position` on the reference and the
hose in its spawn shape — what the CLI does — adds a transient that takes
several seconds to decay at the closed loop's slowest pole.

The feedback-only comparison regulates against the *static trim* at the moving
setpoint, so most of its error is the lag of a slow loop behind the reference.
Both runs come out of
{py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.simulate`, and the
only difference between them is whether the bound setpoint exposes `get_jet`.

## What is and is not stabilised

**Authority.** The PBH margin $\min_\lambda \sigma_{\min}[A_r - \lambda I\ \
B_r]$ on the reduced pair, at the trim, on the shipped parameters, is
weakest when a single vehicle sits at the top of the chain — M5's
configuration, $(4,)$, which reproduces M5's collapse of authority with $n$ —
and grows as the vehicles are spread along the hose:
$(4,) < (0, 4) < (0, 2, 4)$, with a vehicle at *every* vertex the strongest of
all. The ordering is by where the vehicles are, not by how many there are: a
four-link span with nothing in the middle has a poorly-driven interior.

**Which mode is slowest depends on the pattern**, and this is the part worth
knowing before tuning. At $(0, 2, 4)$ it is the root-position mode. At
$(0, 4)$ it is a lightly damped hose-bending mode coupled to the airframes'
body rates.

**Weights.** Uniform scaling of $Q$ says nothing — for an LQR it is the same
as scaling $R^{-1}$, so it slides along the cheap-control locus. Targeted
weights do move the slowest pole: raising the *position* weights
(`q_root_position`, `q_link`) makes it faster, while raising any *rate* weight
makes it slower. The rate weights still damp the hose's bending modes, as in
M5; what differs is which mode is slowest. In M5 the bending mode is the
slowest one. In M6's spread patterns the root translation is already the
slowest, so heavy rate weighting only slows the translational pole toward the
generic heavily-rate-weighted double-integrator root
$-\sqrt{q_{\text{root position}} / q_{\text{root velocity}}}$. M6 at $(4,)$,
the M5-like configuration, behaves like M5 outright.

Weighting the *positions* helps, so there is no structural floor of the kind
M5 documents; the plant here is simply better actuated.

**Input weighting.** The published $R = 0.2 I_{4n_Q}$ weights a thrust and a
moment equally though their columns of $B$ are orders of magnitude apart on the
shipped airframe — the thrust columns scale as $1/M_{00}$ and the moment
columns as $J^{-1}$. With equal weights the attitude pole lies well beyond
what a 5 ms hold can track (more than $90°$ of phase lag per half period),
and the loop never settles. The shipped moment weight brings it under
$20°$, and on M6 it barely moves the slowest pole.

(m6-errata)=
## Errata in the reference

Three, none of them in `get_dyn_dq`, which the dynamics page reports as intact:

* **The cost matrix does not line up with the error vector.** `example1.m`
  L68-72 builds $Q = \operatorname{blkdiag}(0.5 I_6,\ 0.75 I_{6n},\ I_{3n_Q},\
  0.75 I_{3n_Q})$, whose first two blocks group $[x, v]$ and then the hose.
  The error vector it multiplies is
  $[e_x;\ \xi;\ e_v;\ \delta\omega;\ \eta;\ \delta\Omega]$ (`error.m` L47),
  which interleaves them. Six diagonal entries land in the wrong block: the
  first link's $\xi$ is weighted 0.5 instead of 0.75, and $e_v$ is weighted
  0.75 instead of 0.5. Small in effect and clearly not the intent; the weights
  in {py:mod}`udaan.core.defaults` are the intent.
* **$R = 0.2 I$** weights thrust and moment equally — see above.
* **`optimize_for_des_state`, the setpoint solver, is on the published path**
  (`computeSetpoint` calls it, and the paper's Sim 1.1/1.2 call
  `computeSetpoint`). Its `cost_fun` is commented "sum of the square of the
  tension norms" but sums the squared *thrusts*; minimising thrust is the
  stated intent, so it is the comment that is wrong. More consequentially it
  leaves the tension magnitudes unbounded in sign, so nothing stops it
  returning a setpoint with links in compression — a hose cannot push. The
  closed-form catenary trim used here cannot: its tensions are bounded below
  by $H > 0$ by construction.

Two further defects — `get_dyn_dq_damped` adding a gyroscopic term that does
not dissipate, and `parse_inputs` accepting an unsorted or out-of-range
`setI` — are recorded on {doc}`../dynamics/multi-quad-hose`.

```{footbibliography}
```
