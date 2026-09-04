(flatness-multi-quad-cspayload)=
# Multiple quadrotors with a cable-suspended point-mass payload

This page derives the differential-flatness property of $n_Q$ quadrotors
cooperatively transporting a single point-mass payload, each through its
own taut, massless cable — the cooperative-manipulation system of
Sreenath and Kumar {footcite}`sreenath2013cooperative`, whose dynamics
are collected in {doc}`../dynamics/multi-quad-cspointmass`.

The essential difference from the single-quadrotor case of
{doc}`quadrotor-cspayload` is that the payload's force balance no longer
determines the cable forces. It fixes only their **sum**: three equations
in $3 n_Q$ unknowns. The remaining $3(n_Q - 1)$ dimensions are a genuine
free choice, and they are part of the flat output. Fixing that choice —
here, with a pluggable **tension-split policy** — is the whole design
content of the multi-quadrotor map; everything downstream of it is the
single-quadrotor recovery applied $n_Q$ times.

## System

Let $x_L \in \mathbb{R}^3$ be the payload position and, for
$i = 1, \dots, n_Q$, let $q_i \in S^2$ be the unit cable vector oriented
from quadrotor $i$ *to* the payload, with cable angular velocity
$\omega_i \in T_{q_i} S^2$ (so $\dot q_i = \omega_i \times q_i$ and
$\omega_i \cdot q_i = 0$). Each quadrotor has mass $m_i$, inertia
$J_i = J_i^\top \succ 0$, attitude $R_i \in SO(3)$, body rate
$\Omega_i \in \mathbb{R}^3$, and hangs from a cable of length
$\ell_i > 0$. The payload has mass $m_L$.

Cables are taut and inextensible, so quadrotor positions are *derived*:

```{math}
:label: eq-mq-cable-geom
x_{Q_i} \;=\; x_L \,-\, \ell_i\, q_i,
\qquad
\dot q_i \;=\; \omega_i \times q_i.
```

This is the same sign convention as {doc}`quadrotor-cspayload`, as
{py:mod}`udaan.flatness.cable`, and as the simulation model
{py:class}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`:
$q_i \approx -e_3$ in the hanging equilibrium.

Writing $u_i := f_i R_i e_3$ for the thrust vector of quadrotor $i$, the
equations of motion are

```{math}
:label: eq-mq-eom
\begin{aligned}
\Bigl(m_L I + \textstyle\sum_i m_i\, q_i q_i^\top\Bigr)\bigl(\ddot x_L + g e_3\bigr)
  &= \sum_i \Bigl( q_i q_i^\top u_i \,-\, m_i \ell_i \|\dot q_i\|^2 q_i \Bigr), \\[2pt]
\dot\omega_i &= \tfrac{1}{\ell_i}\, \widehat{q_i}\, \bigl(\ddot x_L + g e_3\bigr)
   \,-\, \tfrac{1}{m_i \ell_i}\, \widehat{q_i}\, u_i^{\perp}, \\[2pt]
\dot R_i &= R_i\, \widehat{\Omega_i}, \\[2pt]
J_i\, \dot\Omega_i \,+\, \Omega_i \times J_i\, \Omega_i &= M_i,
\end{aligned}
```

with $u_i^{\perp} = (I - q_i q_i^\top) u_i$. The control inputs are the
$n_Q$ pairs $(f_i, M_i)$, so $\dim u = 4 n_Q$.

## The cable-force split is under-determined

Cut the payload out of {math:numref}`eq-mq-eom` and write Newton's law
for the point mass with each cable reaction explicit. Following the
convention of {doc}`quadrotor-cspayload`, define the **tension vector**

```{math}
:label: eq-mq-Tvec
\mathbf{T}_i \;:=\; T_i\, q_i \;\in\; \mathbb{R}^3,
\qquad T_i = \|\mathbf{T}_i\| \geq 0,
```

as the force the cable applies *to quadrotor $i$* (it pulls the
quadrotor towards the payload, i.e. along $+q_i$). By Newton's third law
the force on the payload is $-\mathbf{T}_i$, so

```{math}
:label: eq-mq-payload-balance
m_L \ddot x_L \;=\; -\sum_{i=1}^{n_Q} \mathbf{T}_i \,-\, m_L\, g\, e_3
\quad\Longleftrightarrow\quad
\sum_{i=1}^{n_Q} \mathbf{T}_i \;=\; F_{\mathrm{net}}
 \;:=\; -\, m_L \bigl(\ddot x_L + g\, e_3\bigr).
```

At $n_Q = 1$ this is exactly {math:numref}`eq-payload-Tvec` of
{doc}`quadrotor-cspayload` — the single tension vector is *determined*.
For $n_Q > 1$, {math:numref}`eq-mq-payload-balance` is 3 scalar
equations in $3 n_Q$ unknowns; its solution set is the affine space

```{math}
:label: eq-mq-split-space
\Bigl\{ (\mathbf{T}_1, \dots, \mathbf{T}_{n_Q}) \;:\;
  \mathbf{T}_i = \alpha_i F_{\mathrm{net}} + N_i,\quad
  \textstyle\sum_i \alpha_i = 1,\ \sum_i N_i = 0 \Bigr\},
```

of dimension $3(n_Q - 1)$. The $N_i$ are **internal forces**: they
cancel at the payload and do no net work on it, but they are exactly what
spreads the quadrotors into a formation. This is the standard
over-actuation of cable-suspended cooperative manipulation
{footcite}`sreenath2013cooperative`.

## Flat output

Take

```{math}
:label: eq-mq-flat-output
y \;=\; \bigl(x_L,\; \psi_1, \dots, \psi_{n_Q},\;
             N_1, \dots, N_{n_Q-1}\bigr)
\;\in\; \mathbb{R}^3 \times \mathbb{R}^{n_Q} \times \mathbb{R}^{3(n_Q-1)},
```

where $\psi_i$ is quadrotor $i$'s yaw angle and the internal forces
$N_1, \dots, N_{n_Q-1}$ parameterise the split (the last one is fixed by
$\sum_i N_i = 0$). Then

$$
\dim y \;=\; 3 + n_Q + 3(n_Q - 1) \;=\; 4 n_Q \;=\; \dim u,
$$

so the dimension-matching necessary condition of {doc}`index` holds. At
$n_Q = 1$ the internal block is empty and {math:numref}`eq-mq-flat-output`
collapses to the $(x_L, \psi)$ flat output of
{doc}`quadrotor-cspayload`.

:::{admonition} The split is a *choice*, not a consequence
:class: important

Nothing in the dynamics prefers one point of
{math:numref}`eq-mq-split-space` over another. Choosing $(\alpha_i, N_i)$
is a design decision with real consequences — it sets the hover
formation geometry, decides which vehicle absorbs which part of the
trajectory's inertial load, and determines how much payload acceleration
the system can take before a cable goes slack. It is *not* something the
flatness derivation hands you.
:::

## The tension-split policy

Because the recovery consumes jets rather than point values, a usable
policy must produce $\mathbf{T}_i$ **and all of its time derivatives**
in closed form. That rules out running a numerical allocator (an
`fmincon`-style min-norm tension solver, say) at each sample: it returns
a point value with no derivatives, and finite-differencing it would
destroy the exactness the flat map exists to provide.

`udaan` therefore implements the split as a pluggable strategy object,
{py:class}`~udaan.flatness.TensionSplitPolicy`, with one concrete family:

```{math}
:label: eq-mq-affine-split
\mathbf{T}_i(t) \;=\; \bar{\mathbf{T}}_i
  \,+\, \alpha_i\,\bigl(F_{\mathrm{net}}(t) - \bar F_{\mathrm{net}}\bigr),
\qquad
\bar F_{\mathrm{net}} := \sum_j \bar{\mathbf{T}}_j,
\qquad
\sum_i \alpha_i = 1,
```

({py:class}`~udaan.flatness.AffineTensionSplit`). Here
$\bar{\mathbf{T}}_i$ is a constant **static anchor** — the cable force at
the reference hover condition, which fixes the formation because
$\bar q_i = \bar{\mathbf{T}}_i / \|\bar{\mathbf{T}}_i\|$ and
$x_{Q_i} = x_L - \ell_i \bar q_i$ — and $\alpha_i$ is a constant
**dynamic share**, the fraction of the trajectory's inertial load
quadrotor $i$ absorbs. Matching against
{math:numref}`eq-mq-split-space` gives
$N_i = \bar{\mathbf{T}}_i - \alpha_i \bar F_{\mathrm{net}}$, and
$\sum_i N_i = 0$ holds by construction.

Two properties make this the right shape:

**The constraint holds at every derivative order, exactly.** Since
$\bar{\mathbf{T}}_i$ and $\bar F_{\mathrm{net}}$ are constants,

```{math}
:label: eq-mq-split-derivs
\sum_i \mathbf{T}_i = F_{\mathrm{net}},
\qquad
\mathbf{T}_i^{(k)} \;=\; \alpha_i\, F_{\mathrm{net}}^{(k)}
 \;=\; -\,\alpha_i\, m_L\, x_L^{(k+2)}
\quad (k \geq 1),
```

so $\sum_i \mathbf{T}_i^{(k)} = F_{\mathrm{net}}^{(k)}$ for every $k$.
The derivatives are free — no differentiation of the policy is ever
needed — and they cost no extra jet depth beyond what
$F_{\mathrm{net}}$ already needs.

### Default policy — a ring anchor with mass-weighted dynamics

{py:class}`~udaan.flatness.RingTensionSplit` builds the anchors from a
horizontal ring of radius $r$ centred above the payload, with quadrotor
$i$ at bearing $\theta_i = 2\pi i / n_Q$ and lift
$dz_i = \sqrt{\ell_i^2 - r^2}$:

```{math}
:label: eq-mq-ring-anchor
\bar{\mathbf{T}}_i \;=\; -\,c\,
  \begin{bmatrix} r\cos\theta_i \\ r\sin\theta_i \\ dz_i \end{bmatrix},
\qquad
c \;=\; \frac{m_L\, g}{\sum_j dz_j},
\qquad
\alpha_i \;=\; \frac{m_i}{\sum_j m_j}.
```

The horizontal terms cancel around the ring
($\sum_i \cos\theta_i = \sum_i \sin\theta_i = 0$ for $n_Q \geq 2$), so
$\bar F_{\mathrm{net}} = -m_L g\, e_3$ as required, and the tension
magnitudes come out $\bar T_i = c\, \ell_i$: every cable pulls with the
same *horizontal* force $c\,r$, while longer (hence more vertical) cables
carry proportionally more of the weight. The resulting $\bar q_i$
coincide with
{py:func}`~udaan.utils.grasp.formation_cable_attitudes`,
so the flat reference at hover *is* the simulation model's default reset
configuration.

The rationale for each half:

* **Anchor — length-weighted, symmetric.** {math:numref}`eq-mq-ring-anchor`
  is the unique equal-horizontal-pull equilibrium of the requested ring.
  Under identical quadrotors it is exactly symmetric, and it degrades
  gracefully with heterogeneous cable lengths.
* **Share — mass-weighted.** $\alpha_i \propto m_i$ makes every
  quadrotor participate at *every* derivative order, in proportion to a
  proxy for its lift capacity.
* **Radius — the non-degeneracy knob.** Minimising $\sum_i \|\mathbf{T}_i\|^2$
  subject only to {math:numref}`eq-mq-payload-balance` returns
  $\mathbf{T}_i = F_{\mathrm{net}}/n_Q$ — all $N_i = 0$, every $q_i$
  parallel, and every quadrotor stacked at the same point above the
  payload. The min-norm objective must be constrained by geometry before
  it means anything (the static solver in the multi-quadrotor
  flexible-hose codebase does this with unit-norm, endpoint-closure and
  per-vertex balance constraints; it is a setpoint solver, so it cannot
  be used on a trajectory). Here $r$ plays that role directly, and
  $r \to 0$ is rejected as degenerate.

### Taut-cable certificate

Because {math:numref}`eq-mq-affine-split` is affine, tautness has a
closed-form sufficient condition. Write
$\Delta := F_{\mathrm{net}} - \bar F_{\mathrm{net}}$, so that
$\mathbf{T}_i = \bar{\mathbf{T}}_i + \alpha_i \Delta$ exactly. The
triangle inequality gives

```{math}
:label: eq-mq-taut-margin
\|\mathbf{T}_i\| \;\geq\; \|\bar{\mathbf{T}}_i\| - |\alpha_i|\, \|\Delta\|,
\qquad\text{so}\qquad
\|\Delta\| \;<\; \min_{i \,:\, \alpha_i \neq 0}
  \frac{\|\bar{\mathbf{T}}_i\| - T_{\min}}{|\alpha_i|}
  \;=:\; \mathrm{margin}
```

keeps every cable above $T_{\min}$. The share enters through
$|\alpha_i|$, not $\alpha_i$: only $\alpha_i = 0$ freezes
$\mathbf{T}_i$, while a *negative* share — legal, and physically a
vehicle that counteracts while another over-absorbs — moves it exactly
as fast as a positive one. This is
{py:meth}`~udaan.flatness.TensionSplitPolicy.net_force_margin`.

Turning that into a bound on $\|\ddot x_L\|$ needs one more step.
Substituting {math:numref}`eq-mq-Fnet-derivs`,

```{math}
:label: eq-mq-delta
\Delta \;=\; -m_L\, \ddot x_L \;+\; d,
\qquad
d \;:=\; -m_L g\, e_3 - \bar F_{\mathrm{net}},
```

so $\Delta = -m_L \ddot x_L$ **only when the anchors balance the payload
weight**, $\bar F_{\mathrm{net}} = -m_L g\, e_3$. Both shipped policies
do — it is exactly the hover-equilibrium condition — but
{py:class}`~udaan.flatness.AffineTensionSplit` accepts any anchors, and a
set that is not in equilibrium at rest already spends $\|d\|$ of the
margin standing still. With $\|\Delta\| \leq m_L \|\ddot x_L\| + \|d\|$,
every cable stays taut with margin $T_{\min}$ whenever

```{math}
:label: eq-mq-taut-bound
\|\ddot x_L\| \;<\; \frac{\mathrm{margin} - \|d\|}{m_L},
\qquad
\mathrm{margin} := \min_{i \,:\, \alpha_i \neq 0}
  \frac{\|\bar{\mathbf{T}}_i\| - T_{\min}}{|\alpha_i|}.
```

This is exposed as
{py:meth}`~udaan.flatness.MultiQuadCsPayload.payload_acceleration_bound`;
$\|d\| \geq \mathrm{margin}$ certifies nothing and the bound reads 0. It
is conservative, not tight — two triangle inequalities — so the recovery
additionally checks the true $\mathbf{T}_i$ at every sample and raises a
`ValueError` naming the offending cable, the same failure mode
{py:func}`~udaan.flatness.normalize_derivatives` raises on at zero norm,
but caught earlier and with physical units. For the shipped defaults at
$n_Q = 3$, $\ell_i = 1$ m, $r = 0.5$ m, identical quadrotors, $d = 0$ and
{math:numref}`eq-mq-taut-bound` evaluates to
$\|\ddot x_L\| < 11.33\ \mathrm{m/s^2}$.

:::{admonition} The tension magnitude alone cannot see the boundary
:class: warning

$T_i = \|\mathbf{T}_i\|$ is a norm, so it is positive on *both* sides of
the slack point: once $\alpha_i \Delta$ overshoots $-\bar{\mathbf{T}}_i$
the tension grows again while $q_i$ has flipped, putting quadrotor $i$
below the payload. A floor on $T_i$ therefore refuses only to *land* on
the boundary, never to cross it. The recovery additionally requires

```{math}
:label: eq-mq-sign-check
\mathbf{T}_i \cdot \bar q_i \;>\; 0,
\qquad \bar q_i = \bar{\mathbf{T}}_i / \|\bar{\mathbf{T}}_i\|,
```

which makes the refusal two-sided. It is sufficient, not necessary: it
also rejects a cable that has swung more than $90°$ from its hover
direction without ever going slack. Inside
{math:numref}`eq-mq-taut-bound` it can never fire, since there
$\|\alpha_i \Delta\| < \|\bar{\mathbf{T}}_i\|$ holds for every $i$.
:::

### Why the published fixed split is not the default

The reference implementation
(`Flat2state_mQpoint.m`, lines 21–31, of the
`quadrotor-flexible-cables-payload` GeoControl toolbox) hard-codes, for
$n_Q = 3$ exactly,

$$
\bar{\mathbf{T}}_{2,3} = -\tfrac{1}{3} m_L g\, R_x(\mp \pi/6)\, e_3,
\qquad
\mathbf{T}_1 = F_{\mathrm{net}} - \bar{\mathbf{T}}_2 - \bar{\mathbf{T}}_3,
$$

with $\mathbf{T}_2^{(k)} = \mathbf{T}_3^{(k)} = 0$ for all $k \geq 1$.
In the language of {math:numref}`eq-mq-affine-split` that is precisely an
{py:class}`~udaan.flatness.AffineTensionSplit` with
$\alpha = (1, 0, 0)$ — it is shipped as
{py:class}`~udaan.flatness.LegacyFixedTensionSplit` so the published
$n_Q = 3$ results stay reproducible. Four reasons it is not the default:

1. **Asymmetric.** Cables 2 and 3 are frozen at a constant world-frame
   force and contribute nothing at any derivative order; quadrotor 1
   absorbs 100 % of the trajectory dynamics, so its slack margin is the
   whole system's margin. By {math:numref}`eq-mq-taut-bound` that margin
   is $0.423\,g \approx 4.15\ \mathrm{m/s^2}$ — under 40 % of the ring
   default's.
2. **Mass-agnostic.** The shares ignore $m_i$ entirely, even though the
   same reference deliberately assigns the three quadrotors unequal
   masses.
3. **$n_Q = 3$ only.** There is no statement of it for general $n_Q$.
4. **Geometrically degenerate.** $\bar q_1 = -e_3$ and
   $\bar q_{2,3}$ tilt by $30°$ about the $x$-axis, so all three
   quadrotors lie in the plane $x = 0$: the formation has no extent along
   $x$ and cannot resist a moment about the $z$-axis of the payload
   (harmless for a point mass, fatal for a rigid body).

## Constructive recovery

Fix a policy. The recovery then proceeds per cable, and each cable's
chain is the single-quadrotor recovery of {doc}`quadrotor-cspayload`
with $\mathbf{T}_i$ in place of $\mathbf{T}$.

### Step 1 — net cable force and its derivatives

From {math:numref}`eq-mq-payload-balance`, with gravity constant,

```{math}
:label: eq-mq-Fnet-derivs
F_{\mathrm{net}} = -m_L\bigl(\ddot x_L + g\, e_3\bigr),
\qquad
F_{\mathrm{net}}^{(k)} = -\, m_L\, x_L^{(k+2)}, \quad k \geq 1.
```

Four derivative levels are needed ($k = 0, \dots, 4$), which is what
makes $x_L^{(6)}$ the deepest rung.

### Step 2 — split into per-cable force jets

Apply the policy: {math:numref}`eq-mq-affine-split` and
{math:numref}`eq-mq-split-derivs` give
$\bigl[\mathbf{T}_i, \dot{\mathbf{T}}_i, \dots, \mathbf{T}_i^{(4)}\bigr]$
for each $i$, in closed form.

### Step 3 — tension magnitude and cable direction, per cable

Each $\mathbf{T}_i = T_i q_i$ is decomposed by the *same* primitive the
single-quadrotor map uses, {py:func}`~udaan.flatness.tension_direction_jet`
(a thin wrapper over
{py:func}`~udaan.flatness.normalize_derivatives`), which takes the cable
**force** jet rather than a payload acceleration precisely so that it
works when the tension is a free/flat quantity:

```{math}
:label: eq-mq-T-q
T_i = \|\mathbf{T}_i\|,
\qquad
q_i = \frac{\mathbf{T}_i}{T_i},
\qquad
T_i\, q_i^{(n)} = \mathbf{T}_i^{(n)}
  - \sum_{k=1}^{n} \binom{n}{k}\, T_i^{(k)}\, q_i^{(n-k)},
```

the last being the general Leibniz recursion
{math:numref}`eq-payload-q-leibniz` of {doc}`quadrotor-cspayload`
{footcite}`apostol1967calculus`, evaluated through $n = 4$.

### Step 4 — quadrotor positions and their derivatives

Differentiate {math:numref}`eq-mq-cable-geom` term by term:

```{math}
:label: eq-mq-xQ-derivs
x_{Q_i}^{(k)} \;=\; x_L^{(k)} \,-\, \ell_i\, q_i^{(k)},
\qquad k = 0, \dots, 4.
```

The worst rung is $x_{Q_i}^{(4)}$ (quadrotor snap), which needs
$q_i^{(4)}$ and therefore $x_L^{(6)}$.

### Step 5 — cable rates

Exactly as in {math:numref}`eq-payload-omega`–{math:numref}`eq-payload-domega`
of {doc}`quadrotor-cspayload`, per cable:

```{math}
:label: eq-mq-omega
\omega_i \;=\; q_i \times \dot q_i,
\qquad
\dot\omega_i \;=\; q_i \times \ddot q_i.
```

The second line of {math:numref}`eq-mq-eom` is *not* needed to invert
$\omega_i$; it is a consistency relation, satisfied automatically by the
construction of Step 6 (crossing the quadrotor's Newton equation with
$q_i$ annihilates $\mathbf{T}_i \parallel q_i$ and reproduces it exactly).

### Step 6 — thrust vectors

Newton's law for quadrotor $i$, whose cable reaction is $+\mathbf{T}_i$,

$$
m_i\, \ddot x_{Q_i} \;=\; f_i R_i\, e_3 \,-\, m_i\, g\, e_3 \,+\, \mathbf{T}_i,
$$

gives the thrust vector and its two derivatives

```{math}
:label: eq-mq-B
\begin{aligned}
B_i &\;:=\; f_i R_i\, e_3
   \;=\; m_i\bigl(\ddot x_{Q_i} + g\, e_3\bigr) \,-\, \mathbf{T}_i, \\
\dot B_i &\;=\; m_i\, x_{Q_i}^{(3)} \,-\, \dot{\mathbf{T}}_i, \\
\ddot B_i &\;=\; m_i\, x_{Q_i}^{(4)} \,-\, \ddot{\mathbf{T}}_i.
\end{aligned}
```

At $n_Q = 1$, substituting
$\mathbf{T} = -m_L(\ddot x_L + g e_3)$ recovers
{math:numref}`eq-payload-B` of {doc}`quadrotor-cspayload` verbatim.

### Step 7 — attitude, body rate, moment

With $(B_i, \psi_i)$ in hand, $(R_i, \Omega_i, \dot\Omega_i, M_i)$ come
from the standalone-quadrotor recovery of {doc}`quadrotor` applied per
quadrotor — the $b_3$-projector chain, the yaw-hint frame
$b_{1,d}(\psi_i) = [\cos\psi_i, \sin\psi_i, 0]^\top$, the skew-projector
body rates, and

```{math}
:label: eq-mq-M
M_i \;=\; J_i\, \dot\Omega_i \,+\, \Omega_i \times J_i\, \Omega_i.
```

Each quadrotor carries its own yaw, so the $n_Q$ applications are fully
independent once the split is fixed.

:::{admonition} Singularities
:class: note

Three singularities bound the regular region. **(i) Slack cable:**
$\mathbf{T}_i = 0$ for some $i$, where $q_i$ is undefined —
{math:numref}`eq-mq-taut-bound` is a sufficient condition for avoiding
it, and the implementation checks the actual $\mathbf{T}_i$ regardless,
through both the magnitude floor and the sign test
{math:numref}`eq-mq-sign-check` that catches an overshoot of the
boundary rather than a landing on it. At
$n_Q = 1$ this degenerates to the free-fall condition
$\ddot x_L + g e_3 = 0$ of {doc}`quadrotor-cspayload`. **(ii) Vanishing
thrust:** $B_i = 0$, quadrotor $i$ in free fall. **(iii) Yaw-aligned:**
$b_{3,i} \parallel b_{1,d}(\psi_i)$, the singularity of the
single-quadrotor map, now per quadrotor.
:::

## Theorem

```{prf:theorem} Flatness of $n_Q$ quadrotors with a shared cable-suspended point-mass payload
:label: thm-mq-flat

Fix a smooth tension-split policy of the form
{math:numref}`eq-mq-affine-split`. The system
{math:numref}`eq-mq-eom` is differentially flat with flat output
$y = (x_L, \psi_1, \dots, \psi_{n_Q}, N_1, \dots, N_{n_Q-1})$ as in
{math:numref}`eq-mq-flat-output`. Every state and every input is an
algebraic function of the sixth-order jet of $x_L$, the second-order
jets of the $\psi_i$, and the (constant) internal forces.
```

```{prf:proof}
Dimension matching: $\dim y = 4 n_Q = \dim u$. Steps 1–7 construct the
smooth map

$$
\Phi : J^6 \mathbb{R}^3 \times \bigl(J^2 \mathbb{R}\bigr)^{n_Q}
       \times \mathbb{R}^{3(n_Q-1)}
      \;\longrightarrow\; \mathbb{R}^{6 + 10 n_Q} \times \mathbb{R}^{4 n_Q},
$$

$$
\Phi \;=\;
\Bigl(\bigl(x_L, \dot x_L, \{q_i, \omega_i, R_i, \Omega_i\}\bigr),\;
      \{(f_i, M_i)\}\Bigr),
$$

without integrating the dynamics: $F_{\mathrm{net}}$ and its derivatives
from $x_L$ by {math:numref}`eq-mq-Fnet-derivs`; the per-cable force jets
by {math:numref}`eq-mq-affine-split`; $(T_i, q_i)$ and $q_i^{(k)}$ by
{math:numref}`eq-mq-T-q`; $x_{Q_i}^{(k)}$ by {math:numref}`eq-mq-xQ-derivs`;
$(\omega_i, \dot\omega_i)$ by {math:numref}`eq-mq-omega`; $B_i$ by
{math:numref}`eq-mq-B`; and $(R_i, \Omega_i, \dot\Omega_i, M_i)$ by the
recovery of {doc}`quadrotor`, closing with {math:numref}`eq-mq-M`.

That the result satisfies {math:numref}`eq-mq-eom` is a two-line check.
Substituting $\mathbf{T}_i = m_i(\ddot x_L - \ell_i \ddot q_i + g e_3) - u_i$
from {math:numref}`eq-mq-B` into
$\sum_i \mathbf{T}_i = F_{\mathrm{net}}$ and projecting each term onto
$q_i$ — using the tangency identity
$q_i \cdot \ddot q_i = -\|\dot q_i\|^2$ — gives the first line of
{math:numref}`eq-mq-eom`; crossing the same expression with $q_i$
annihilates $\mathbf{T}_i \parallel q_i$ and gives the second, using
$\dot\omega_i = q_i \times \ddot q_i$.

Smoothness of $\Phi$ holds on the open set where no $T_i$ vanishes, no
$B_i$ vanishes, and no $b_{3,i}$ is parallel to $b_{1,d}(\psi_i)$. This
is the form {math:numref}`eq-flat-to-state-on-jets` on the product jet
space, so the system is differentially flat.
```

The $n_Q = 1$ specialisation of $\Phi$ is exactly
{prf:ref}`thm-payload-flat`: the split space
{math:numref}`eq-mq-split-space` is a point, $\alpha_1 = 1$, $N_1 = 0$,
and {math:numref}`eq-mq-affine-split` degenerates to
$\mathbf{T}_1 = F_{\mathrm{net}}$. `udaan` asserts this numerically —
the two maps agree to machine precision on random flat inputs. That test
pins the *plumbing*, not the algebra: the two implementations share the
cable and attitude primitives verbatim, so agreement at $n_Q = 1$ is
expected by construction and exercises none of the $n_Q > 1$ split
machinery. (For the same reason
{py:class}`~udaan.flatness.RingTensionSplit` has no ring at $n_Q = 1$:
the lone quadrotor hovers on the payload's vertical axis and the radius
$r$ is ignored, along with its $0 < r < \min_i \ell_i$ validation.)

## Derivative-order accounting

| Recovered quantity                             | Needs                                            |
| ---------------------------------------------- | ------------------------------------------------ |
| $F_{\mathrm{net}},\ \mathbf{T}_i,\ T_i,\ q_i$   | $\ddot x_L$                                      |
| $\dot{\mathbf{T}}_i,\ \dot q_i,\ \omega_i$      | $x_L^{(3)}$                                      |
| $\ddot{\mathbf{T}}_i,\ \ddot q_i,\ \dot\omega_i$| $x_L^{(4)}$                                      |
| $q_i^{(3)}$                                     | $x_L^{(5)}$                                      |
| $q_i^{(4)}$                                     | $x_L^{(6)}$                                      |
| $x_{Q_i},\ \dot x_{Q_i},\ \ddot x_{Q_i}$        | $\ddot x_L,\ x_L^{(3)},\ x_L^{(4)}$              |
| $x_{Q_i}^{(3)}$                                 | $x_L^{(5)}$ (via $q_i^{(3)}$)                    |
| $x_{Q_i}^{(4)}$ (snap)                          | $x_L^{(6)}$ (via $q_i^{(4)}$)                    |
| $f_i,\ b_{3,i},\ R_i$                           | $x_L^{(4)},\ \psi_i$                             |
| $\Omega_i$                                      | $x_L^{(5)},\ \dot\psi_i$                         |
| $\dot\Omega_i,\ M_i$                            | $x_L^{(6)},\ \ddot\psi_i$                        |

The worst case is again the moment, closing the chain at $x_L^{(6)}$
(pop) and $\ddot\psi_i$ — **identical to the single-quadrotor case**.
Adding quadrotors costs no jet depth: the policy contributes
$\mathbf{T}_i^{(k)} = \alpha_i F_{\mathrm{net}}^{(k)}$, so every cable's
chain bottoms out on the same $F_{\mathrm{net}}$ jet. A time-*varying*
internal force $N_i(t)$ would add its own derivative requirement (through
$\ddot N_i$, to reach $\ddot B_i$) without deepening the $x_L$ chain.

## Jet input

The map $\Phi$ of {prf:ref}`thm-mq-flat` takes one order-6
{py:class}`udaan.flatness.Jet` for the payload position and one order-2
scalar jet per quadrotor, packaged in a
{py:class}`udaan.flatness.MultiQuadCsPayloadFlats`:

```python
import numpy as np
from udaan.flatness import (
    Jet,
    MultiQuadCsPayload,
    MultiQuadCsPayloadFlats,
    RingTensionSplit,
)

nQ = 4

# Payload position and its first six time derivatives at t
x_L_jet = Jet(np.stack([
    x_L,            # row 0 — payload position
    x_L_dot,        # row 1 — payload velocity
    x_L_ddot,       # row 2 — acceleration (gives F_net, T_i, q_i)
    x_L_dddot,      # row 3 — jerk    (gives q̇_i, ω_i, ẋ_Q_i)
    x_L_snap,       # row 4 — snap    (gives q̈_i, ẍ_Q_i, f_i, b3_i)
    x_L_crackle,    # row 5 — crackle (gives q_i⁽³⁾, Ω_i)
    x_L_pop,        # row 6 — pop     (gives q_i⁽⁴⁾, Ω̇_i, M_i)
]))

# One yaw jet per quadrotor
psi_jets = [Jet(np.array([psi[i], psi_dot[i], psi_ddot[i]])) for i in range(nQ)]

flats = MultiQuadCsPayloadFlats(x_L=x_L_jet, psi=psi_jets)
f2s = MultiQuadCsPayload(
    num_quadrotors=nQ,
    masses=[0.7, 0.9, 1.1, 1.3],
    cable_lengths=[0.9, 1.1, 1.3, 1.5],
    payload_mass=0.2,
)                       # default policy: RingTensionSplit
ref, inputs = f2s.recover(flats)

# ref.payload_position, ref.payload_velocity,
# ref.positions[i], ref.velocities[i], ref.accelerations[i],
# ref.cable_attitudes[i], ref.cable_angular_velocities[i],
# ref.orientations[i], ref.angular_velocities[i]
# inputs.thrusts[i], inputs.moments[i], inputs.tensions[i]
# inputs.wrench     -> (nQ, 4) array, ready for model.step(...)
```

`ref` is a {py:class}`udaan.flatness.MultiQuadCsPayloadRefState` (its
`as_state()` projects onto the simulation model's state struct) and
`inputs` is a {py:class}`udaan.flatness.MultiQuadCsPayloadInputs`.
Building the map with
{py:meth}`~udaan.flatness.MultiQuadCsPayload.from_model` copies the
physical parameters *and* the formation radius off a
{py:class}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`, so
the flat hover reference coincides with the model's `reset()`
configuration. Supplying `tension_policy=` overrides the split — for
example `LegacyFixedTensionSplit()` to reproduce the published $n_Q = 3$
results, or a hand-built
{py:class}`~udaan.flatness.AffineTensionSplit` with custom anchors and
shares.

## References

```{footbibliography}
```
