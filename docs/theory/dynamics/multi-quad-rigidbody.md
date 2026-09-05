# Multi-quadrotor with cable-suspended rigid-body payload

A team of $n_Q$ quadrotors cooperatively transports a single **rigid-body**
payload, each attached to a distinct point of the load through its own
taut, massless, inextensible cable. This is
{doc}`multi-quad-cspointmass` with two additions — the payload has an
attitude, and the cables attach at distinct body-frame points $r_i$ —
and those two additions are exactly what lets the team exert a *moment*
on the load. The configuration manifold is

$$
Q = \mathbb{R}^3 \times SO(3) \times \left(S^2 \times SO(3)\right)^{n_Q},
$$

so the state space has dimension $12 + 10\, n_Q$ against M3's
$6 + 10\, n_Q$. The corresponding simulation model is
{py:class}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase`.

:::{admonition} This model is a derivation, not a port
:class: important

There is no working analytical M4 in the legacy code. The MATLAB
`odefun_multiQuad_rigid` assembles every coupling block and then never
solves them, returning six variables that are never assigned; its Python
sibling is a `pass` stub. The equations below were re-derived from
scratch and validated against six independent checks, catalogued in the
module docstring and exercised in
`tests/test_multi_quad_rigidbody_base.py`.
:::

## Coordinates and constraints

Let $q_i \in S^2$ be the unit cable vector oriented from quadrotor $i$
*to* its attachment point, $\omega_i \in T_{q_i} S^2$ its angular
velocity, and $\ell_i > 0$ the cable length. Quadrotor $i$ has mass
$m_i$, inertia $J_i$, attitude $R_i \in SO(3)$ and body rate $\Omega_i$.
The payload has mass $m_L$, inertia $J_L = J_L^\top \succ 0$, attitude
$R_L \in SO(3)$ and body rate $\Omega_L$, with
$\dot R_L = R_L \widehat{\Omega_L}$.

Cable $i$ attaches at the fixed body-frame point $r_i$, so the
world-frame attachment point and the quadrotor position are both
*derived*:

```{math}
:label: eq-m4-geom
x_{P_i} = x_L + R_L r_i,
\qquad
x_{Q_i} = x_L + R_L r_i - \ell_i q_i,
\qquad
\dot q_i = \omega_i \times q_i .
```

Setting every $r_i = 0$ collapses {math:numref}`eq-m4-geom` to M3's
$x_{Q_i} = x_L - \ell_i q_i$. That is the sign check that matters: the
convention is fixed by internal consistency with M3 and
{py:mod}`udaan.flatness.cable`, not by any external source.

Differentiating {math:numref}`eq-m4-geom` twice, and writing
$A_i := R_L \widehat{r_i}$,

```{math}
:label: eq-m4-attach
\dot x_{P_i} = v_L - A_i \Omega_L,
\qquad
\ddot x_{P_i} = a_L - A_i \dot\Omega_L + c_i,
\qquad
c_i := R_L \widehat{\Omega_L}\widehat{\Omega_L} r_i .
```

## Equations of motion

Write $u_i := f_i R_i e_3$ for the thrust vector, $P_i := q_i q_i^\top$,
$P_i^\perp := I - P_i$, and let $\mathbf{T}_i = T_i q_i$ with $T_i \ge 0$
be the force the cable applies *to quadrotor $i$* — it pulls the vehicle
towards the load, along $+q_i$, so by Newton's third law the payload
feels $-\mathbf{T}_i$ at $x_{P_i}$. Newton–Euler then reads

```{math}
:label: eq-m4-newton
\begin{aligned}
m_i a_{Q_i} &= u_i - m_i g\, e_3 + \mathbf{T}_i, \\
m_L a_L &= -\sum_i \mathbf{T}_i - m_L g\, e_3, \\
J_L \dot\Omega_L + \Omega_L \times J_L \Omega_L
  &= \sum_i r_i \times \bigl(R_L^\top(-\mathbf{T}_i)\bigr)
   = \sum_i A_i^\top \mathbf{T}_i ,
\end{aligned}
```

the last equality because
$\widehat{r_i} R_L^\top = -(R_L \widehat{r_i})^\top = -A_i^\top$.
The quadrotor attitude line is unchanged from M1.

**The parallel projection eliminates the tensions.** Since
$\dot\omega_i \times q_i \perp q_i$, projecting the first line of
{math:numref}`eq-m4-newton` onto $q_i$ contains no $\dot\omega_i$ at all.
With $\alpha := a_L + g e_3$ and
$b_i := m_i \ell_i \|\dot q_i\|^2 q_i - P_i u_i$ this gives

```{math}
:label: eq-m4-tension
\mathbf{T}_i = m_i P_i\bigl(\alpha - A_i \dot\Omega_L + c_i\bigr) + b_i ,
```

an *affine function of $(\alpha, \dot\Omega_L)$ alone*. Substituting it
into the payload force and moment lines leaves a $6 \times 6$ symmetric
system in those six unknowns,

```{math}
:label: eq-m4-mass
\begin{bmatrix} M_{11} & M_{12} \\ M_{12}^\top & M_{22}\end{bmatrix}
\begin{bmatrix}\alpha \\ \dot\Omega_L\end{bmatrix}
=
\begin{bmatrix}h_1 \\ h_2\end{bmatrix},
```

```{math}
\begin{aligned}
M_{11} &= m_L I + \sum_i m_i P_i, &
M_{12} &= -\sum_i m_i P_i A_i, &
M_{22} &= J_L + \sum_i m_i A_i^\top P_i A_i, \\
h_1 &= -\sum_i \bigl(m_i P_i c_i + b_i\bigr), &
h_2 &= \sum_i A_i^\top\bigl(m_i P_i c_i + b_i\bigr) &&\!\!\!\!\!- \Omega_L \times J_L\Omega_L .
\end{aligned}
```

Crossing the quadrotor line of {math:numref}`eq-m4-newton` with $q_i$
annihilates $\mathbf{T}_i \parallel q_i$ and back-substitutes the cable
accelerations:

```{math}
:label: eq-m4-cable
\dot\omega_i = \frac{1}{\ell_i}\,\widehat{q_i}\bigl(\alpha - A_i\dot\Omega_L + c_i\bigr)
  - \frac{1}{m_i \ell_i}\,\widehat{q_i}\, u_i^{\perp} .
```

Three structural facts, each load-bearing in the implementation:

* **Only a $6 \times 6$ solve is ever performed.** Assembling
  {math:numref}`eq-m4-newton` in $(\alpha, \dot\Omega_L, \dot\omega_1,
  \dots)$ gives a $(6 + 3n_Q) \times (6 + 3n_Q)$ system — M3's, with a
  payload-rotation block bordered on. But
  {math:numref}`eq-m4-tension`–{math:numref}`eq-m4-cable` *are* its exact
  block elimination: the parallel/perpendicular split of the quadrotor
  equation decouples $T_i$ from $\dot\omega_i$, so the reduction onto
  $(\alpha, \dot\Omega_L)$ is available in closed form. This is the
  same reduction M3 makes to $3 \times 3$.

  Said precisely: that assembled system is **singular by exactly $n_Q$**.
  Its per-cable diagonal block is $m_i \ell_i^2 P_i^{\perp}$, of rank 2,
  because $\dot\omega_i$ along $q_i$ does not enter the dynamics at all —
  it is the gauge direction the tangency constraint
  $q_i \cdot \omega_i = 0$ removes. "Schur complement" is therefore well
  defined here only through the pseudo-inverse of those blocks, or
  equivalently on the tangent spaces $T_{q_i}S^2$ where they are
  invertible. On that quotient the elimination is exact, not approximate.
* **The mass matrix is always invertible.** Each cable contributes
  $m_i \begin{bmatrix} I \\ -A_i^\top\end{bmatrix} P_i \begin{bmatrix} I & -A_i\end{bmatrix} \succeq 0$,
  so {math:numref}`eq-m4-mass` is $\mathrm{diag}(m_L I, J_L)$ plus
  positive semi-definite terms — positive definite for *every* cable
  configuration and *every* attachment geometry, including all-parallel
  cables, collinear attachment points and $n_Q = 1$.
* **The centripetal coefficient is $\|\dot q_i\|^2$**, for exactly the
  reason spelled out in {doc}`multi-quad-cspointmass`.

## The grasp map, and what a rigid-body payload demands

Cutting the payload out and setting every derivative to zero gives the
static trim:

```{math}
:label: eq-m4-trim
\sum_i T_i q_i = -m_L g\, e_3,
\qquad
\sum_i T_i \bigl(r_i \times (R_L^\top q_i)\bigr) = 0,
\qquad
u_i = m_i g\, e_3 - T_i q_i .
```

The middle equation is the constraint a point mass does not impose: the
grasp geometry must be able to produce **zero net moment**. It is not
automatic, and a geometry that cannot satisfy it has no static hover at
all.

More generally, the body-frame cable forces $\tau_i$ applied to the load
map to the load wrench through the **grasp matrix**
({py:func}`udaan.utils.grasp.grasp_map`)

```{math}
:label: eq-m4-grasp
\Phi =
\begin{bmatrix} I & \cdots & I \\ \widehat{r_1} & \cdots & \widehat{r_{n_Q}}\end{bmatrix}
\in \mathbb{R}^{6 \times 3 n_Q},
\qquad
\Phi\,\tau = \begin{bmatrix}\sum_i \tau_i \\ \sum_i r_i \times \tau_i\end{bmatrix} .
```

$\operatorname{rank}\Phi$ is the number of wrench directions the grasp
can produce **from arbitrary force vectors $\tau_i$**, and for the
default ring of
{py:func}`~udaan.utils.grasp.ring_attachment_points` it is

| $n_Q$ | 1 | 2 | $\ge 3$ | any, collinear $r_i$ |
|---|---|---|---|---|
| $\operatorname{rank}\Phi$ | 3 | 5 | 6 | 5 |

A single cable produces no moment whatsoever; two leave one moment axis
unactuated; collinear attachment points cannot twist about their own
line however many there are. **The payload attitude is therefore
controllable only from $n_Q \ge 3$ non-collinear attachments** — a real
limitation of the mechanism, not of any particular controller.

Rank 6 is *necessary* for full SE(3) authority, and it is not
sufficient. A taut cable pulls only along its own direction, so what the
team can apply **at a given instant** is not $\Phi\tau$ over all $\tau$
but $G t$ over scalar tensions $t_i \ge 0$ along the current $q_i$ — the
$(6 \times n_Q)$ map {py:func}`~udaan.utils.grasp.hover_wrench_map`. At
the default ring that map has **rank 3 for every $n_Q \ge 3$**: the
instantaneous wrench authority is 3-dimensional, and the remaining SE(3)
directions become available only as the cascade reorients the cables,
one attitude-loop time constant later. That lag is what shows up as the
$\sim 11^\circ$ steady-state yaw-rate lag of the default ring; it is a
property of the mechanism, not of the allocation.

Note carefully that both are *static* statements about allocation. They
say nothing about whether the dynamics are well posed: the mass matrix
of {math:numref}`eq-m4-mass` stays positive definite at rank-3 grasps
just as it does at rank-6 ones.

## Relationship to the point-mass case

Setting every $r_i = 0$ in {math:numref}`eq-m4-mass` gives $A_i = 0$,
$c_i = 0$, hence $M_{12} = 0$ and $M_{22} = J_L$: the payload rotation
decouples entirely (it becomes a free rigid body, and stays at rest if
it starts there), and the translational block is
$M_{11}\alpha = h_1$ — {doc}`multi-quad-cspointmass` verbatim. The
implementation reproduces M3 **to round-off** in that limit, which is
the strongest available check on the derivation: M3 was verified
independently, and the two code paths share nothing but two geometry
helpers — M3 assembles its $3 \times 3$ inline, M4 assembles blocks and
solves a $6 \times 6$. Round-off is the honest claim rather than
bit-exactness: over 400 steps the payload trajectory does come out
bit-identical at every $n_Q$ tested, but individual cable directions
drift by up to $4\times10^{-17}$, and two different linear solves are
only ever guaranteed to agree to rounding.

Everything M4 adds is therefore carried by $A_i = R_L \widehat{r_i}$:
the $M_{12}$ coupling between payload translation and payload rotation,
the $A_i^\top P_i A_i$ inflation of the rotational inertia by the cable
masses, and the $-A_i \dot\Omega_L + c_i$ correction that turns the
payload's acceleration into the *attachment point's* acceleration in
every cable equation.

## Control

{py:class}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController`
is the rigid-body sibling of M3's cascade. Where M3 splits a commanded
force with a {py:class}`~udaan.flatness.TensionSplitPolicy`, M4 splits a
commanded *wrench* through $\Phi$:

$$
\tau = \Phi^{+} W^d + \bigl(I - \Phi^{+}\Phi\bigr)\,\tau^{\mathrm{hov}},
\qquad
W^d = \begin{bmatrix} R_L^\top m_L (a_L^d + g e_3) \\ M_L^d\end{bmatrix} .
$$

$\Phi^{+}$ answers every command, whether or not $W^d$ lies in the range
of $\Phi$: the unachievable component $(I - \Phi\Phi^{+})W^d$ is simply
dropped. At $n_Q \le 2$ that is the whole moment command, which is why
the payload attitude drifts there while position tracking still looks
perfect. The controller counts those commands
({py:attr}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.unachievable_wrenches`)
and warns on the first one, and reports the deficiency at construction
through
{py:attr}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.grasp_rank`.

The null-space term plays the role M3's internal forces play: without
it the least-norm solution makes every cable force parallel and the
formation collapses onto one line. Biasing towards the hover forces of
the model's own `reset()` formation also makes the split **exact at
hover**, since $\Phi\tau^{\mathrm{hov}} = W^{\mathrm{hov}}$ implies the
expression returns $\tau^{\mathrm{hov}}$ identically.

The bias is published as
{py:attr}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.tension_policy`,
and the allocation itself runs through the very
{py:class}`~udaan.flatness.WrenchAllocation` the flat map below uses, so
there is one copy of $\Phi^{+}$ and one null-space bias rather than two
that can drift apart.

## Differential flatness

{py:class}`udaan.flatness.MultiQuadRigidbody` is the flat-to-state map,
and {py:meth}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.simulate`
consumes it: when the payload controller's `setpoint` is a trajectory's
bound `get_full`, the map recovers each quadrotor's reference attitude
$R_i$, body rate $\Omega_i$ and body angular acceleration $\dot\Omega_i$,
and all three reach the attitude loops as `desired_att`.
Binding `get_full` is the whole switch — no controller change required —
and the payload's attitude reference is read off `attitude_setpoint` at
the same time, so a commanded manoeuvre is fed forward as well as fed
back.

The attitude loop uses all three, but it splits them by degree of
freedom.
{py:meth}`~udaan.control.quadrotor.GeometricAttitudeController.compute`
always rebuilds $b_3$ of its $R_d$ from the commanded thrust direction,
because that is the only path the position loop has into the attitude
command — substituting the open-loop $R_i$ *whole* severs it and the
closed loop diverges. What it takes from $R_i$ is the one degree of
freedom left over: the heading of $b_1$ about $b_3$.

$R_i$ used to be discarded outright, and $R_i = I$ then gave a
bit-identical run at every $n_Q$, with the entire 1.38x tracking
improvement carried by the rate channel ($\dot\Omega_i$ on its own is
worth 0.07%). That is no longer so: $R_i = I$ commands $\psi = 0$
instead of the reference's heading. On the default $\psi \equiv 0$
reference the two still agree to within 0.2%, because the heading the
composition supplies is then the world $x$ axis the fallback would have
used anyway; the channel earns its place when
{py:attr}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.yaw_setpoint`
commands a heading that is not zero. The behaviour is shared with M1 and
M3 and is not specific to M4.

The map is built from *this controller's* `tension_policy`. That is not
housekeeping: $\Phi^{+} W$ is only half the split, and a reference
allocated at a different null-space bias is the reference for a different
formation. A payload controller that does not publish a policy therefore
gets no feedforward at all rather than one built at a guessed default.

References that the map refuses — a cable that would go slack, a
quadrotor in free fall, or (at $n_Q \le 2$) a wrench outside
$\operatorname{range}\Phi$ — degrade that step to feedback-only rather
than aborting the run. The drops are counted in
{py:attr}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.feedforward_drops`
and the first one warns.

## References

```{footbibliography}
```
