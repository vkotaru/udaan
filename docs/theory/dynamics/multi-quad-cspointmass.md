# Multi-quadrotor with cable-suspended point-mass payload

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

A team of $n_Q$ quadrotors cooperatively transports a single point-mass
payload, each attached through its own taut, massless, inextensible
cable {footcite}`sreenath2013cooperative`. The configuration manifold is

$$
Q = \mathbb{R}^3 \times \left(SO(3) \times S^2\right)^{n_Q},
$$

with one $SO(3)$ attitude and one $S^2$ cable direction per quadrotor,
plus the shared payload position $x_L \in \mathbb{R}^3$; the state space
$\mathbb{R}^3 \times \mathbb{R}^3 \times (TSO(3) \times TS^2)^{n_Q}$ has
dimension $6 + 10\, n_Q$. The corresponding simulation model is
{py:class}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`.

## Coordinates and constraints

Let $q_i \in S^2$ be the unit cable vector oriented from quadrotor $i$
*to* the payload, $\omega_i \in T_{q_i} S^2$ its angular velocity, and
$\ell_i > 0$ the cable length. Quadrotor $i$ has mass $m_i$, inertia
$J_i = J_i^\top \succ 0$, attitude $R_i \in SO(3)$ and body rate
$\Omega_i$. The payload has mass $m_L$. Because the cables are taut and
inextensible, the quadrotor positions are *derived*, not independent:

```{math}
:label: eq-mqd-geom
x_{Q_i} = x_L - \ell_i\, q_i,
\qquad
\dot x_{Q_i} = \dot x_L - \ell_i\, \dot q_i,
\qquad
\dot q_i = \omega_i \times q_i .
```

This is the sign convention used throughout `udaan` (see
{doc}`quadrotor-cspayload` and {py:mod}`udaan.flatness.cable`): at rest
$q_i \approx -e_3$, the payload hanging below the formation. The
$S^2$/$TS^2$ conventions are collected in {doc}`../preliminaries`.

## Equations of motion

Write $u_i := f_i R_i e_3$ for the thrust vector of quadrotor $i$, and
split it along and across the cable,

$$
u_i^{\parallel} = q_i q_i^\top u_i,
\qquad
u_i^{\perp} = \bigl(I - q_i q_i^\top\bigr) u_i .
$$

The Lagrangian derivation on $Q$ under the taut-cable constraint
{math:numref}`eq-mqd-geom` yields
{footcite}`sreenath2013cooperative`

```{math}
:label: eq-mqd-eom
\begin{aligned}
\Bigl(m_L I + \sum_{i=1}^{n_Q} m_i\, q_i q_i^\top\Bigr)
  \bigl(\ddot x_L + g\, e_3\bigr)
  &= \sum_{i=1}^{n_Q}
     \Bigl( u_i^{\parallel} \,-\, m_i\, \ell_i\, \|\dot q_i\|^2\, q_i \Bigr), \\[3pt]
\dot\omega_i &= \frac{1}{\ell_i}\, \widehat{q_i}\,\bigl(\ddot x_L + g\, e_3\bigr)
   \,-\, \frac{1}{m_i \ell_i}\, \widehat{q_i}\, u_i^{\perp}, \\[3pt]
\dot R_i &= R_i\, \widehat{\Omega_i}, \\[3pt]
J_i\, \dot\Omega_i \,+\, \Omega_i \times J_i\, \Omega_i &= M_i .
\end{aligned}
```

Three structural facts are worth recording, because each of them is
load-bearing in the implementation:

* **The mass matrix is always invertible.** $m_L I + \sum_i m_i q_i q_i^\top$
  is a sum of a positive multiple of the identity and positive
  semi-definite rank-one terms, hence positive definite for $m_L > 0$
  regardless of the cable configuration — including the degenerate case
  in which every $q_i$ is parallel.
* **The centripetal coefficient is $\|\dot q_i\|^2$, not $\|\Omega_i\|^2$
  or $\|\omega_i\|^2$.** The exact identity obtained by differentiating
  $q_i \cdot \dot q_i = 0$ is $q_i \cdot \ddot q_i = -\|\dot q_i\|^2$,
  and $\|\dot q_i\|^2 = \|\omega_i\|^2 - (q_i \cdot \omega_i)^2$ agrees
  with $\|\omega_i\|^2$ only while $\omega_i$ is exactly tangent. A
  discrete integrator lets $q_i \cdot \omega_i$ drift off zero, so the
  two forms are not interchangeable.
* **The cable equation stays on $T_{q_i} S^2$ by construction.** The
  right-hand side of the $\dot\omega_i$ line is a cross product with
  $q_i$, hence automatically orthogonal to it, so the tangency constraint
  is preserved by the continuous dynamics.

The $n_Q = 1$ case collapses to the single-quadrotor system of
{doc}`quadrotor-cspayload`: the mass matrix becomes
$m_L I + m_Q q q^\top$, whose action on the (necessarily parallel)
right-hand side is scalar multiplication by $(m_L + m_Q)^{-1}$ along $q$.

## Cable tensions and the allocation problem

The constrained form {math:numref}`eq-mqd-eom` hides the cable tensions.
Making them explicit is what the control and trajectory-generation layers
actually need. Cut the payload out and let

```{math}
:label: eq-mqd-Tvec
\mathbf{T}_i := T_i\, q_i, \qquad T_i = \|\mathbf{T}_i\| \geq 0,
```

be the force the cable applies *to quadrotor $i$* (pulling it towards the
payload, along $+q_i$); by Newton's third law the force on the payload is
$-\mathbf{T}_i$. Newton's law for the point mass is then

```{math}
:label: eq-mqd-balance
\sum_{i=1}^{n_Q} \mathbf{T}_i \;=\; -\, m_L \bigl(\ddot x_L + g\, e_3\bigr) .
```

Newton's law for quadrotor $i$, using {math:numref}`eq-mqd-geom`, is

```{math}
:label: eq-mqd-quad-newton
m_i \bigl(\ddot x_L - \ell_i \ddot q_i + g\, e_3\bigr)
  \;=\; u_i \,+\, \mathbf{T}_i .
```

Equations {math:numref}`eq-mqd-balance`–{math:numref}`eq-mqd-quad-newton`
are equivalent to {math:numref}`eq-mqd-eom`: projecting
{math:numref}`eq-mqd-quad-newton` onto $q_i$, substituting into
{math:numref}`eq-mqd-balance` and using
$q_i \cdot \ddot q_i = -\|\dot q_i\|^2$ reproduces the payload line;
crossing it with $q_i$ annihilates $\mathbf{T}_i \parallel q_i$ and
reproduces the cable line via $\dot\omega_i = q_i \times \ddot q_i$.

**Rank deficiency.** {math:numref}`eq-mqd-balance` is 3 scalar equations
in the $3 n_Q$ unknowns $\{\mathbf{T}_i\}$. For $n_Q = 1$ it determines
the tension outright — the single-quadrotor system is *not*
over-actuated. For $n_Q > 1$ its solution set is an affine space of
dimension $3(n_Q - 1)$:

```{math}
:label: eq-mqd-split
\mathbf{T}_i = \alpha_i \Bigl(-m_L(\ddot x_L + g e_3)\Bigr) + N_i,
\qquad
\sum_i \alpha_i = 1,
\qquad
\sum_i N_i = 0 .
```

The $N_i$ are **internal forces**: they cancel at the payload and do no
net work on it, but they are what spreads the quadrotors into a
formation. If every $N_i = 0$ then every $\mathbf{T}_i$ — and hence every
$q_i$ — is parallel and the whole team stacks up along one line above the
payload. The rank deficiency is therefore not a nuisance to be
regularised away; it is the degree of freedom the formation lives in.

Resolving {math:numref}`eq-mqd-split` is a design choice, made in
`udaan` by a pluggable *tension-split policy*
({py:class}`udaan.flatness.TensionSplitPolicy`). It also constrains what
is achievable: cables can only pull, so a split is physically realisable
only while every $T_i > 0$. Slack cables put the system in a different
hybrid mode that {math:numref}`eq-mqd-eom` does not describe.

## Relationship to the single-quadrotor case

Setting $n_Q = 1$ in {math:numref}`eq-mqd-eom` gives exactly the taut
cable model of {doc}`quadrotor-cspayload`, and
{math:numref}`eq-mqd-balance` gives its determined tension
$\mathbf{T} = -m_L(\ddot x_L + g e_3)$. Everything the multi-quadrotor
system adds is the allocation freedom
{math:numref}`eq-mqd-split`: the per-cable kinematics, the per-quadrotor
rigid-body dynamics and the tension→direction algebra are all unchanged,
one copy per cable.

## Differential flatness

Once the split {math:numref}`eq-mqd-split` is fixed, the system is
differentially flat, with the internal forces themselves forming part of
the flat output. See
{doc}`../differential-flatness/multi-quad-cspayload` for the flat-output
count $3 + n_Q + 3(n_Q-1) = 4 n_Q$, the constructive recovery, and the
tension-split policies `udaan` ships.

## References

```{footbibliography}
```
