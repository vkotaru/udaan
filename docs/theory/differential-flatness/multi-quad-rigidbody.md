(flatness-multi-quad-rigidbody)=
# Multiple quadrotors with a cable-suspended rigid-body payload

This page derives the differential-flatness property of $n_Q$ quadrotors
cooperatively transporting a single **rigid-body** payload, each through
its own taut, massless cable attached at a distinct point of the load —
the system of {footcite}`lee2018geometric`, whose dynamics are collected
in {doc}`../dynamics/multi-quad-rigidbody`.

The step up from {doc}`multi-quad-cspayload` is that the payload now has
an **attitude**. Two things follow, and everything on this page is a
consequence of one or the other:

1. The payload's balance is a *wrench* balance, not a force balance. It
   fixes six numbers, not three, so the free part of the cable-force
   split is the $3n_Q - 6$ dimensional **null space of the grasp map**
   rather than M3's $3(n_Q-1)$ dimensional internal-force space. Fixing
   that choice — with a pluggable **tension-distribution policy** — is
   the design content of the map.
2. Each cable hangs off a *moving, rotating* attachment point
   $x_{P_i} = x_L + R_L r_i$. The payload rotation therefore enters every
   cable's chain twice: once through the attachment kinematics, and once
   through the frame change between the body-frame allocation and the
   world-frame Newton equations.

## System

Let $x_L \in \mathbb{R}^3$ and $R_L \in SO(3)$ be the payload position
and attitude, with body angular velocity $\Omega_L$ ($\dot R_L = R_L
\widehat{\Omega_L}$), mass $m_L$ and inertia $J_L = J_L^\top \succ 0$.
Cable $i$ attaches at the fixed **body-frame** point $r_i$, has length
$\ell_i > 0$, and has direction $q_i \in S^2$ pointing *from quadrotor
$i$ to the load*, with $\omega_i \in T_{q_i}S^2$ and $\dot q_i = \omega_i
\times q_i$. Quadrotor $i$ has mass $m_i$, inertia $J_i$, attitude
$R_i$, body rate $\Omega_i$ and thrust vector $u_i = f_i R_i e_3$.

Cables are taut and inextensible, so the quadrotor positions are
*derived*:

```{math}
:label: eq-m4f-geom
x_{P_i} \;=\; x_L + R_L\, r_i,
\qquad
x_{Q_i} \;=\; x_L + R_L\, r_i - \ell_i\, q_i .
```

Setting every $r_i = 0$ collapses {math:numref}`eq-m4f-geom` to M3's
$x_{Q_i} = x_L - \ell_i q_i$, which is the check that fixes the sign
convention.

Writing $\mathbf{T}_i = T_i q_i$ for the force the cable applies *to
quadrotor $i$* (the same convention as {py:mod}`udaan.flatness.cable`,
so the force on the payload is $-\mathbf{T}_i$), the equations of motion
are

```{math}
:label: eq-m4f-eom
\begin{aligned}
m_i\, \ddot x_{Q_i} &= u_i - m_i g e_3 + \mathbf{T}_i, \\
m_L\, \ddot x_L &= -\sum_i \mathbf{T}_i - m_L g e_3, \\
J_L\, \dot\Omega_L + \Omega_L \times J_L \Omega_L
  &= \sum_i r_i \times \bigl(R_L^\top(-\mathbf{T}_i)\bigr), \\
\dot R_i = R_i \widehat{\Omega_i}, \qquad
J_i \dot\Omega_i + \Omega_i \times J_i \Omega_i &= M_i .
\end{aligned}
```

The control inputs are the $n_Q$ pairs $(f_i, M_i)$, so
$\dim u = 4 n_Q$.

## The wrench balance is under-determined

Work in the payload body frame and write

```{math}
:label: eq-m4f-tau
\tau_i \;:=\; R_L^\top\bigl(-\mathbf{T}_i\bigr) \;\in\; \mathbb{R}^3
```

for the body-frame force cable $i$ applies *to the load*, equivalently
$\mathbf{T}_i = -R_L \tau_i$. Then the second and third lines of
{math:numref}`eq-m4f-eom` read, together,

```{math}
:label: eq-m4f-balance
\Phi\, \tau \;=\; W,
\qquad
\Phi \;=\;
\begin{bmatrix} I & \cdots & I \\ \hat r_1 & \cdots & \hat r_{n_Q}\end{bmatrix}
\in \mathbb{R}^{6 \times 3 n_Q},
\qquad
W \;=\;
\begin{bmatrix}
 m_L R_L^\top\bigl(\ddot x_L + g e_3\bigr) \\
 J_L \dot\Omega_L + \Omega_L \times J_L \Omega_L
\end{bmatrix},
```

with $\tau = (\tau_1, \dots, \tau_{n_Q})$ stacked. $\Phi$ is the **grasp
map** {footcite}`murray2017mathematical`, implemented as
{py:func}`udaan.utils.grasp.grasp_map`; it is built from the $r_i$ alone,
which are *constants of the mechanism*, so $\Phi$, $\Phi^{+}$ and any
basis of $\ker\Phi$ are constant matrices — a fact the derivative
accounting below leans on heavily.

Six equations in $3n_Q$ unknowns. For $n_Q \geq 3$ non-collinear
attachments $\operatorname{rank}\Phi = 6$ and the solution set is the
affine space

```{math}
:label: eq-m4f-split-space
\bigl\{\, \tau \;:\; \tau = \Phi^{+}W + N\lambda,\;\;
  \lambda \in \mathbb{R}^{3n_Q - 6} \,\bigr\},
\qquad
\operatorname{range} N = \ker \Phi,
```

of dimension $3n_Q - 6$. This is the rigid-body analogue of M3's
internal-force space: the columns of $N$ are cable-force patterns that
produce *no* net wrench on the load, so they are invisible to the payload
and are exactly what spreads the team into a formation.

:::{admonition} Rank of $\Phi$ is not instantaneous authority
:class: warning

$\Phi$ maps arbitrary force *vectors* $\tau_i$. What a taut cable can
actually apply at a frozen instant is a scalar tension along its
*current* $q_i$, i.e. the $(6, n_Q)$ map
{py:func}`udaan.utils.grasp.hover_wrench_map`, whose rank at the default
ring is **3 for every** $n_Q \geq 3$. Full $SE(3)$ authority emerges only
as the cables reorient, one attitude-loop time constant later, and that
lag is a real closed-loop cost.

It is not a limitation of *this* map, and the distinction matters. A
flatness map is not choosing a wrench at a frozen instant: it plans the
whole trajectory, and $q_i := \mathbf{T}_i / \|\mathbf{T}_i\|$ is
*derived* from the allocated force. The reference it returns is one in
which the cables have already reoriented — the taut-cable constraint
$\mathbf{T}_i \parallel q_i$ holds by construction, at every instant, for
free. Nothing here claims more authority than the mechanism has.
:::

## Flat output

Take

```{math}
:label: eq-m4f-flat-output
y \;=\; \bigl(x_L,\; R_L,\; \psi_1, \dots, \psi_{n_Q},\; \lambda\bigr)
\;\in\; \mathbb{R}^3 \times SO(3) \times \mathbb{R}^{n_Q}
        \times \mathbb{R}^{3n_Q - 6},
```

the payload **pose** — position *and* attitude, unlike M3 — one yaw
angle per quadrotor, and the null-space coordinates of
{math:numref}`eq-m4f-split-space`. Then

$$
\dim y \;=\; 3 + 3 + n_Q + (3 n_Q - 6) \;=\; 4 n_Q \;=\; \dim u,
$$

so the dimension-matching necessary condition of {doc}`index` holds.

:::{admonition} The tension split is a *free choice*
:class: important

Nothing in the dynamics prefers one point of
{math:numref}`eq-m4f-split-space` over another. Choosing $\lambda$ sets
the hover formation geometry, decides how the trajectory's load is shared,
and determines how much payload wrench the system can take before a cable
goes slack. It is not something the flatness derivation hands you, and
`udaan` therefore exposes it as a pluggable policy object
({py:class}`~udaan.flatness.TensionDistributionPolicy`) rather than a
constant.
:::

## The tension-distribution policy

Because the recovery consumes jets rather than point values, a usable
policy must produce $\tau$ **and all of its time derivatives** in closed
form — which rules out running a numerical allocator (a min-norm QP with
tension bounds, say) at each sample.

`udaan` parameterises the null space by a constant **bias force stack**
$\bar\tau \in \mathbb{R}^{3n_Q}$: the body-frame forces, in newtons, the
policy would *like* the cables to carry. Writing $\bar W := \Phi\bar\tau$
for the wrench they produce, the allocation is

```{math}
:label: eq-m4f-alloc
\tau(W) \;=\; \Phi^{+} W + \bigl(I - \Phi^{+}\Phi\bigr)\bar\tau
       \;=\; \bar\tau + \Phi^{+}\bigl(W - \bar W\bigr),
```

the second equality holding exactly for *any* $\Phi$ (substitute
$\bar W = \Phi\bar\tau$ and use $\Phi^{+}\Phi\bar\tau$ twice). Three
consequences:

* **Exact at the bias point.** $\tau(\bar W) = \bar\tau$ identically, not
  approximately.
* **Affine in the wrench**, so tautness has the closed-form certificate
  below.
* **The derivatives are free.** $\Phi^{+}$ and the null-space term are
  constant, so

  ```{math}
  :label: eq-m4f-alloc-derivs
  \tau^{(k)} \;=\; \Phi^{+} W^{(k)}, \qquad k \geq 1 .
  ```

  {math:numref}`eq-m4f-alloc-derivs` is the step the published reference
  takes silently. Here it is *true by construction*, because the
  null-space coordinate is chosen constant; a time-varying $\lambda(t)$
  is a legitimate extension — it is a genuine flat output, free to move —
  but it would then have to supply its own derivative tuple, and
  {py:meth}`~udaan.flatness.WrenchAllocation.allocate_jet` re-checks
  {math:numref}`eq-m4f-balance` at every order either way.

### Default policy — the ring hover trim

{py:class}`~udaan.flatness.RingTensionDistribution` takes the bias to be
the model's own $\texttt{reset()}$ formation: the quadrotors sit on a
horizontal circle of radius $r$ centred above **their own attachment
points**, and the scalar tensions solve the static trim

```{math}
:label: eq-m4f-trim
G\, t = \begin{bmatrix} -m_L g e_3 \\ 0 \end{bmatrix},
\qquad
G = \bigl[\, \bar q_i \;;\; r_i \times \bar q_i \,\bigr]_{i=1\ldots n_Q},
\qquad
\bar\tau_i = -t_i\, \bar q_i .
```

Then $\Phi\bar\tau = [\,m_L g e_3;\, 0\,] = \bar W$ is exactly the hover
wrench, so {math:numref}`eq-m4f-alloc` returns $\bar\tau$ *identically* at
hover: the flat reference at rest **is** the simulation model's
$\texttt{reset()}$ configuration, to round-off, in cable directions,
quadrotor positions and thrusts alike. A non-zero least-squares residual
in {math:numref}`eq-m4f-trim` means the grasp geometry admits no static
trim at all, and is reported rather than absorbed.

The bias is not optional. Minimising $\sum_i \|\tau_i\|^2$ subject to
{math:numref}`eq-m4f-balance` alone — that is, $\bar\tau = 0$ — makes the
cable forces near-parallel and collapses the team onto one point above
the payload, the same degeneracy M3's ring split documents. Geometry has
to enter the objective somewhere; here the ring radius is that knob.

### Taut-cable certificate

Because {math:numref}`eq-m4f-alloc` is affine,
$\tau_i - \bar\tau_i = \Phi^{+}_i (W - \bar W)$ exactly, with
$\Phi^{+}_i$ the $(3,6)$ block of rows $3i \ldots 3i+2$. Hence

```{math}
:label: eq-m4f-margin
\|\tau_i\| \;\geq\; \|\bar\tau_i\|
  - \sigma_{\max}\!\bigl(\Phi^{+}_i\bigr)\,\|W - \bar W\|,
\qquad\text{so}\qquad
\|W - \bar W\| \;<\;
  \min_i \frac{\|\bar\tau_i\| - T_{\min}}{\sigma_{\max}(\Phi^{+}_i)}
  \;=:\; \mathrm{margin}
```

keeps every cable above $T_{\min}$. This is
{py:meth}`~udaan.flatness.WrenchAllocation.wrench_margin`. It is
*sufficient*, not tight — one triangle inequality and one operator norm —
so the recovery additionally checks the true $\tau_i$ at every sample,
and raises a `ValueError` naming the offending cable.

As in M3, the magnitude alone cannot see the boundary: $\|\tau_i\|$ is a
norm, positive on *both* sides of the slack point, so a floor refuses
only to *land* on the boundary and never to cross it. The recovery
therefore also requires

```{math}
:label: eq-m4f-sign
\tau_i \cdot \bar\tau_i \;>\; 0,
```

which makes the refusal two-sided. Inside {math:numref}`eq-m4f-margin` it
can never fire.

### Why the published `LAMBDA` is not the default

The reference implementation
(`differential_flatness/matlab/@Flat2State/multiQuadRigidLoad.m`, lines
20–114) writes {math:numref}`eq-m4f-alloc` as $\Phi^{+}W + N\Lambda$ with
a hard-coded

$$
\Lambda \;=\; \tfrac{1}{4} m_L g
\begin{bmatrix}
  \langle R_y(\pi/6) e_3,\, e_1\rangle \\
  \langle R_y(\pi/6) e_3,\, e_2\rangle \\
  \langle R_x(\pi/6) e_3,\, e_1\rangle \\
  R_y(-\pi/6)\, e_3
\end{bmatrix} .
$$

Four reasons it is not the default here:

1. **Inhomogeneous units.** Entries 1–3 are scalar inner products and
   4–6 are the components of a vector, so what each coordinate *means*
   depends entirely on the column ordering of an $N$ that the same file
   never builds. Parameterising by $\bar\tau$ instead keeps every
   coordinate in newtons; the equivalent $\Lambda = N^\top\bar\tau$ in
   the module's own orthonormal basis is still available as
   {py:attr}`~udaan.flatness.WrenchAllocation.null_coordinates`.
2. **Not derived, and not general.** It is four numbers that happen to
   give non-negative tensions in hover for one particular $r$, with no
   statement for any other geometry or any other $n_Q$.
3. **No tautness guarantee.** Nothing certifies $t_i > 0$ away from that
   one hover.
4. **The matrix it multiplies does not exist.** $\Phi^{+}$ and $N$ are
   read from `params` in that file but constructed nowhere: the block
   that would build them (lines 52–69) is commented out, so the function
   raises on any input.

Where $N$ *is* built —
`quadrotor-flexible-cables-payload/Multiple-Quad-Rigid-Load/main_multiquad_rigid_flexible.m`,
lines 27–52, under the author's own `(FIX THIS - write code to automate
...)` comment — it is hard-coded for exactly $n_Q = 4$ from the six
attachment-pair directions
$u_{ij} = (r_j - r_i)/\|r_j - r_i\|$, with column $(i,j)$ carrying
$u_{ij}$ on cable $i$ and $-u_{ij}$ on cable $j$. Those columns *are* in
$\ker\Phi$, since
$\Phi\,[u_{ij};\, -u_{ij}] = [\,0;\, (r_i - r_j)\times u_{ij}\,] = 0$,
but the recipe produces $\binom{n_Q}{2}$ of them for a kernel of
dimension $3n_Q - 6$ — the wrong *count* for every $n_Q \geq 5$ — and
even at $n_Q = 4$, on that file's own attachment geometry, the six
columns have **rank 5**. One whole direction of the tension null space is
unreachable by any $\Lambda$.

`udaan` replaces the recipe with an SVD basis
({py:func}`udaan.flatness.null_space_basis`): the right singular vectors
of $\Phi$ whose singular values vanish span $\ker\Phi$ exactly and come
out orthonormal {footcite}`golub2013matrix`. That is correct for **any**
$n_Q$ and **any** attachment geometry, and the dimension is *discovered*
($3n_Q - \operatorname{rank}\Phi$) rather than assumed — which matters
for degenerate grasps, where the kernel is larger than $3n_Q - 6$.

:::{admonition} One more thing the reference gets wrong: frames
:class: note

`multiQuadRigidLoad.m` stacks a **world**-frame force
$m_L(\ddot x_L + g e_3)$ on a **body**-frame moment
$J_L\dot\Omega_L + \Omega_L \times J_L\Omega_L$ inside one $W$, then
feeds the resulting $\tau_i$ straight into a world-frame Newton equation
without rotating it. The two agree only at $R_L = I$. The C++ sibling
`qrotor_ros/qrotor_control/include/qrotor_control/floating_rigidbody_control.hpp`
gets this right — `R.transpose() * force` in, `R * τ_i` out — and this
page follows the C++. (That same C++ is min-norm only: no null-space
term, no bias, no tautness guard, so its $\tau_i$ can point the wrong way
down a cable that can only pull.)
:::

## Constructive recovery

Fix a policy. The recovery is seven steps; steps 4–7 are M3's per-cable
chain verbatim, with $\mathbf{T}_i$ now coming from the allocation
instead of from a force split.

### Step 1 — the payload attitude jet

Differentiating $\dot R_L = R_L \widehat{\Omega_L}$ with the Leibniz rule
gives the exact recursion

```{math}
:label: eq-m4f-Rjet
R_L^{(k+1)} \;=\; \sum_{j=0}^{k} \binom{k}{j}\, R_L^{(j)}\,
                  \widehat{\Omega_L^{(k-j)}},
```

so $m$ derivatives of $R_L$ cost $m-1$ derivatives of $\Omega_L$
({py:func}`udaan.flatness.rotation_jet`). Carrying the attitude flat
output as $(R_L, J^5\Omega_L)$ rather than as a jet of matrices keeps it
chart-free: no Euler angles, no quaternion sign, and the derivatives stay
on $SO(3)$'s tangent structure.

### Step 2 — the payload wrench and its derivatives

Both blocks of {math:numref}`eq-m4f-balance` differentiate in closed form.
The force block is a Leibniz product against the $R_L^\top$ jet, using
$\bigl(\ddot x_L + g e_3\bigr)^{(k)} = x_L^{(k+2)}$ for $k \geq 1$:

```{math}
:label: eq-m4f-Wf
W_f^{(k)} \;=\; m_L \sum_{j=0}^{k} \binom{k}{j}
   \bigl(R_L^{(j)}\bigr)^{\!\top}
   \bigl(\ddot x_L + g e_3\bigr)^{(k-j)} .
```

The moment block is the Leibniz rule on the gyroscopic cross product:

```{math}
:label: eq-m4f-Wm
W_m^{(k)} \;=\; J_L\, \Omega_L^{(k+1)}
   \;+\; \sum_{j=0}^{k} \binom{k}{j}\,
     \Omega_L^{(j)} \times J_L\, \Omega_L^{(k-j)} .
```

Four derivative levels are needed, $k = 0, \dots, 4$, which is what makes
$x_L^{(6)}$ and $\Omega_L^{(5)}$ the deepest rungs.

### Step 3 — allocate, and change frames

Apply {math:numref}`eq-m4f-alloc` and {math:numref}`eq-m4f-alloc-derivs` to
get $[\tau_i, \dot\tau_i, \dots, \tau_i^{(4)}]$, then rotate into the
world frame with one more Leibniz product:

```{math}
:label: eq-m4f-Tjet
\mathbf{T}_i^{(k)} \;=\; -\sum_{j=0}^{k} \binom{k}{j}\,
   R_L^{(j)}\, \tau_i^{(k-j)} .
```

This is the step that has no M3 counterpart: M3's split is already in the
world frame.

### Step 4 — tension magnitude and cable direction

Each $\mathbf{T}_i = T_i q_i$ is decomposed by the *same* primitive M2
and M3 use, {py:func}`~udaan.flatness.tension_direction_jet` (a thin
wrapper over {py:func}`~udaan.flatness.normalize_derivatives`), which
takes the cable **force** jet rather than a payload acceleration
precisely so that it works when the tension is a free/flat quantity:

```{math}
:label: eq-m4f-T-q
T_i = \|\mathbf{T}_i\|,
\qquad
q_i = \frac{\mathbf{T}_i}{T_i},
\qquad
T_i\, q_i^{(n)} = \mathbf{T}_i^{(n)}
  - \sum_{k=1}^{n} \binom{n}{k}\, T_i^{(k)}\, q_i^{(n-k)},
```

the general Leibniz recursion {footcite}`apostol1967calculus`, evaluated
through $n = 4$.

### Step 5 — quadrotor positions

Differentiate {math:numref}`eq-m4f-geom` term by term:

```{math}
:label: eq-m4f-xQ
x_{Q_i}^{(k)} \;=\; x_L^{(k)} \,+\, R_L^{(k)} r_i \,-\, \ell_i\, q_i^{(k)},
\qquad k = 0, \dots, 4 .
```

The middle term is the moving attachment point; it is why $R_L^{(4)}$ is
needed, and it vanishes at $r_i = 0$.

### Step 6 — cable rates and thrust vectors

As in M3, per cable, $\omega_i = q_i \times \dot q_i$ and
$\dot\omega_i = q_i \times \ddot q_i$. Newton for quadrotor $i$, whose
cable reaction is $+\mathbf{T}_i$, gives

```{math}
:label: eq-m4f-B
B_i := f_i R_i e_3 = m_i\bigl(\ddot x_{Q_i} + g e_3\bigr) - \mathbf{T}_i,
\qquad
\dot B_i = m_i x_{Q_i}^{(3)} - \dot{\mathbf{T}}_i,
\qquad
\ddot B_i = m_i x_{Q_i}^{(4)} - \ddot{\mathbf{T}}_i .
```

### Step 7 — attitude, body rate, moment

With $(B_i, \psi_i)$ in hand, $(R_i, \Omega_i, \dot\Omega_i, M_i)$ come
from the standalone-quadrotor recovery of {doc}`quadrotor` applied per
quadrotor: the $b_3$-projector chain, the yaw-hint frame
$b_{1,d}(\psi_i) = [\cos\psi_i, \sin\psi_i, 0]^\top$, the skew-projector
body rates, and $M_i = J_i \dot\Omega_i + \Omega_i \times J_i \Omega_i$.

:::{admonition} Singularities
:class: note

Four bound the regular region. **(i) Rank-deficient grasp:** $W \notin
\operatorname{range}\Phi$, discussed below. **(ii) Slack cable:**
$\tau_i = 0$, guarded by {math:numref}`eq-m4f-margin` and
{math:numref}`eq-m4f-sign`. **(iii) Vanishing thrust:** $B_i = 0$,
quadrotor $i$ in free fall. **(iv) Yaw-aligned:** $b_{3,i} \parallel
b_{1,d}(\psi_i)$, per quadrotor.
:::

## Theorem

```{prf:theorem} Flatness of $n_Q$ quadrotors with a shared cable-suspended rigid-body payload
:label: thm-m4-flat

Let the attachment points satisfy $\operatorname{rank}\Phi = 6$ (which
requires $n_Q \geq 3$ non-collinear $r_i$) and fix a constant null-space
coordinate $\lambda$. The system {math:numref}`eq-m4f-eom` is
differentially flat with flat output
$y = (x_L, R_L, \psi_1, \dots, \psi_{n_Q}, \lambda)$ as in
{math:numref}`eq-m4f-flat-output`. Every state and every input is an
algebraic function of the sixth-order jet of $x_L$, the payload attitude
$R_L$ together with the fifth-order jet of $\Omega_L$, the second-order
jets of the $\psi_i$, and $\lambda$.
```

```{prf:proof}
Dimension matching: $\dim y = 4 n_Q = \dim u$. Steps 1–7 construct the
smooth map

$$
\Phi_{\mathrm{flat}} :
J^6\mathbb{R}^3 \times SO(3) \times J^5\mathbb{R}^3
\times \bigl(J^2\mathbb{R}\bigr)^{n_Q} \times \mathbb{R}^{3n_Q-6}
\;\longrightarrow\;
\mathbb{R}^{12 + 10 n_Q} \times \mathbb{R}^{4 n_Q}
$$

without integrating the dynamics: $R_L^{(k)}$ by
{math:numref}`eq-m4f-Rjet`; $W^{(k)}$ by {math:numref}`eq-m4f-Wf` and
{math:numref}`eq-m4f-Wm`; $\tau^{(k)}$ by {math:numref}`eq-m4f-alloc` and
{math:numref}`eq-m4f-alloc-derivs`; $\mathbf{T}_i^{(k)}$ by
{math:numref}`eq-m4f-Tjet`; $(T_i, q_i^{(k)})$ by
{math:numref}`eq-m4f-T-q`; $x_{Q_i}^{(k)}$ by {math:numref}`eq-m4f-xQ`;
$(\omega_i, \dot\omega_i)$, $B_i$ and $(R_i, \Omega_i, \dot\Omega_i,
M_i)$ as in Steps 6–7.

That the result satisfies {math:numref}`eq-m4f-eom` is a check in three
parts. The first line holds by the definition {math:numref}`eq-m4f-B` of
$B_i$. The second and third hold because $\Phi\tau = W$ *is* those two
lines, rewritten in the body frame through
{math:numref}`eq-m4f-tau` — and $\operatorname{rank}\Phi = 6$ is exactly
the condition under which $\Phi^{+}W$ solves it rather than
least-squaring it. The cable equation is not independent: crossing the
first line with $q_i$ annihilates $\mathbf{T}_i \parallel q_i$ and
reproduces $\dot\omega_i = q_i \times \ddot q_i$, which
{math:numref}`eq-m4f-T-q` already supplies.

Smoothness holds on the open set where no $T_i$ vanishes, no $B_i$
vanishes, and no $b_{3,i}$ is parallel to $b_{1,d}(\psi_i)$. This is the
form {math:numref}`eq-flat-to-state-on-jets` on the product jet space, so
the system is differentially flat.
```

### What happens at $n_Q \leq 2$

The rank hypothesis is not a technicality. At the default ring
$\operatorname{rank}\Phi$ is $3$ at $n_Q = 1$ and $5$ at $n_Q = 2$, so
the payload attitude is **not** a flat output there: only wrenches inside
$\operatorname{range}\Phi$ can be produced at all, and
$\Phi^{+}$ answers every other command by silently dropping
$(I - \Phi\Phi^{+})W$. `udaan` raises
{py:class}`~udaan.flatness.WrenchAllocationError` instead, because the
dropped component is precisely the difference between a reference that
solves {math:numref}`eq-m4f-eom` and one that does not.

Concretely: level hover still recovers at every $n_Q \geq 1$ — its wrench
happens to lie in the range — but a payload-attitude manoeuvre does not,
and neither does a lateral acceleration in the direction the grasp cannot
resist (at $n_Q = 2$ the ring's co-kernel is
$W_{m,x} + 0.05\,W_{f,y} = 0$, so a $y$-acceleration leaves the range;
the two attachments lie along $x$ and a moment about that line is the one
wrench the pair cannot produce). The special case $n_Q = 1$ with
$r_1 = 0$ removes the moment channel entirely and reduces the map exactly
to {doc}`quadrotor-cspayload`. This is a property of the mechanism, and
it is the same one
{py:attr}`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.grasp_rank`
reports in closed loop.

## Derivative-order accounting

| Recovered quantity                                    | Needs                                    |
| ----------------------------------------------------- | ---------------------------------------- |
| $W,\ \tau_i,\ \mathbf{T}_i,\ T_i,\ q_i$               | $\ddot x_L,\ R_L,\ \dot\Omega_L$         |
| $\dot W,\ \dot{\mathbf{T}}_i,\ \dot q_i,\ \omega_i$   | $x_L^{(3)},\ \dot R_L,\ \Omega_L^{(2)}$  |
| $\ddot W,\ \ddot{\mathbf{T}}_i,\ \ddot q_i,\ \dot\omega_i$ | $x_L^{(4)},\ R_L^{(2)},\ \Omega_L^{(3)}$ |
| $W^{(3)},\ q_i^{(3)}$                                 | $x_L^{(5)},\ R_L^{(3)},\ \Omega_L^{(4)}$ |
| $W^{(4)},\ q_i^{(4)}$                                 | $x_L^{(6)},\ R_L^{(4)},\ \Omega_L^{(5)}$ |
| $x_{Q_i},\ \dot x_{Q_i},\ \ddot x_{Q_i}$              | $\ddot x_L,\ x_L^{(3)},\ x_L^{(4)}$      |
| $x_{Q_i}^{(3)}$                                       | $x_L^{(5)}$ (via $q_i^{(3)}$)            |
| $x_{Q_i}^{(4)}$ (snap)                                | $x_L^{(6)}$ (via $q_i^{(4)}$)            |
| $f_i,\ b_{3,i},\ R_i$                                 | $x_L^{(4)},\ \Omega_L^{(3)},\ \psi_i$    |
| $\Omega_i$                                            | $x_L^{(5)},\ \Omega_L^{(4)},\ \dot\psi_i$ |
| $\dot\Omega_i,\ M_i$                                  | $x_L^{(6)},\ \Omega_L^{(5)},\ \ddot\psi_i$ |

The chain that fixes everything: the moment $M_i$ needs $\ddot B_i$,
hence $x_{Q_i}^{(4)}$, hence $q_i^{(4)}$, hence $\mathbf{T}_i^{(4)}$,
hence $\tau^{(4)} = \Phi^{+}W^{(4)}$. **So the wrench is needed through
its fourth derivative** — and the wrench is *second order* in the payload
pose, in both channels. In the translation channel that gives
$x_L^{(6)}$, identical to M3. In the attitude channel it gives
$\Omega_L^{(5)}$, equivalently $R_L^{(6)}$: measured in the flat
variable's own derivative order, the two channels are symmetric and both
bottom out at the sixth derivative of the payload pose.

The attitude column has no M3 counterpart, and the difference is worth
stating precisely. M3's flat output is a *position*, so "sixth
derivative" and "$x_L^{(6)}$" are the same statement. M4's attitude flat
output is carried as $(R_L, \Omega_L\text{-jet})$, and
{math:numref}`eq-m4f-Rjet` costs one rung: $R_L^{(6)}$ is an order-5
$\Omega_L$ jet, which is what
{py:class}`~udaan.flatness.MultiQuadRigidbodyFlats` validates. Separately,
$R_L^{(4)}$ alone is needed for the attachment point
{math:numref}`eq-m4f-xQ` and the frame change
{math:numref}`eq-m4f-Tjet` — a shallower requirement the wrench chain
subsumes.

Adding quadrotors costs no jet depth: $\Phi^{+}$ is constant, so every
cable's chain bottoms out on the same $W$ jet. A time-varying $\lambda(t)$
would add its own requirement (through $\ddot\lambda$, to reach $\ddot
B_i$) without deepening the pose chain.

## Jet input

The map takes one order-6 {py:class}`udaan.flatness.Jet` for the payload
position, the payload attitude matrix plus an order-5 jet for its body
rate, and one order-2 scalar jet per quadrotor, packaged in a
{py:class}`udaan.flatness.MultiQuadRigidbodyFlats`:

```python
import numpy as np
from udaan.flatness import (
    Jet,
    MultiQuadRigidbody,
    MultiQuadRigidbodyFlats,
    RingTensionDistribution,
)

nQ = 4

# Payload position and its first six time derivatives at t
x_L_jet = Jet(np.stack([
    x_L,            # row 0 — payload position
    x_L_dot,        # row 1 — payload velocity
    x_L_ddot,       # row 2 — acceleration (gives W_f, tau_i, T_i, q_i)
    x_L_dddot,      # row 3 — jerk    (gives q̇_i, ω_i, ẋ_Q_i)
    x_L_snap,       # row 4 — snap    (gives q̈_i, ẍ_Q_i, f_i, b3_i)
    x_L_crackle,    # row 5 — crackle (gives q_i⁽³⁾, Ω_i)
    x_L_pop,        # row 6 — pop     (gives q_i⁽⁴⁾, Ω̇_i, M_i)
]))

# Payload attitude: the matrix, plus the body rate through its 5th derivative
Omega_L_jet = Jet(np.stack([
    Omega_L, Omega_L_d1, Omega_L_d2, Omega_L_d3, Omega_L_d4, Omega_L_d5,
]))

psi_jets = [Jet(np.array([psi[i], psi_dot[i], psi_ddot[i]])) for i in range(nQ)]

flats = MultiQuadRigidbodyFlats(
    x_L=x_L_jet, R_L=R_L, Omega_L=Omega_L_jet, psi=psi_jets
)
f2s = MultiQuadRigidbody(
    num_quadrotors=nQ,
    masses=[0.7, 0.9, 1.1, 1.3],
    cable_lengths=[0.9, 1.1, 1.3, 1.5],
    payload_mass=1.0,
)                       # default policy: RingTensionDistribution
ref, inputs = f2s.recover(flats)

# ref.payload_position, ref.payload_velocity,
# ref.payload_orientation, ref.payload_angular_velocity,
# ref.positions[i], ref.velocities[i], ref.accelerations[i],
# ref.cable_attitudes[i], ref.cable_angular_velocities[i],
# ref.orientations[i], ref.angular_velocities[i]
# inputs.thrusts[i], inputs.moments[i], inputs.tensions[i], inputs.cable_forces[i]
# inputs.wrench     -> (nQ, 4) array, ready for model.step(...)
```

`ref` is a {py:class}`udaan.flatness.MultiQuadRigidbodyRefState` (its
`as_state()` projects onto the simulation model's state struct) and
`inputs` is a {py:class}`udaan.flatness.MultiQuadRigidbodyInputs`.
Building the map with
{py:meth}`~udaan.flatness.MultiQuadRigidbody.from_model` copies the
physical parameters, the attachment geometry *and* the formation radius
off a
{py:class}`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase`. For
*ring* attachment points the ring hover trim is exact and the flat hover
reference then coincides with the model's `reset()` configuration; for any
other grasp the trim is only least-squares, so the hover reference is a
different — still exact — equilibrium, and
{py:attr}`~udaan.flatness.RingTensionDistribution.hover_residual` reports
the gap. Supplying `tension_policy=` overrides the null-space
choice — for instance a
{py:class}`~udaan.flatness.FixedTensionDistribution` with a hand-built
bias stack.

The allocation is also usable on its own, as
{py:class}`udaan.flatness.WrenchAllocation`, for anything that needs
wrench→tension without the rest of the flat map.

## References

```{footbibliography}
```
