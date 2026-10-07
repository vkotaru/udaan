# Quadrotor with a flexible (n-link) cable-suspended payload

A single quadrotor carries a payload through a cable that is no longer
assumed taut and straight: it is discretised into $n$ rigid links joined
by ideal spherical joints, each link carrying a point mass at its distal
end {footcite}`goodarzi2014geometric`. This is {doc}`quadrotor-cspayload` with the one cable replaced by a
chain, and at $n = 1$ the two are *the same equations* — a fact the
implementation asserts to round-off. The configuration manifold is

$$
Q = \mathbb{R}^3 \times SO(3) \times \left(S^2\right)^{n},
$$

so the integrated state is $(x_L, v_L, R, \Omega, \{q_i, \omega_i\})$:
$18 + 6n$ stored floats with $R$ as a matrix, or $12 + 6n$ manifold
dimensions. The corresponding simulation model is
{py:class}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`.

:::{admonition} The $12 + 6n$ in `RESEARCH_MAP.md` is a different state
:class: note

That figure is the *error/variational* state an LQR linearisation carries
— it counts $SO(3)$ and each $S^2$ by their tangent dimensions and drops
nothing else. It is not the state this model integrates, which stores the
attitude as a full rotation matrix.
:::

## Coordinates and constraints

Index the links **from the airframe down to the payload**: link $0$ is
hinged at the quadrotor, link $n-1$ carries the load. Let $q_i \in S^2$
be the unit vector along link $i$, oriented from its proximal joint *to*
its distal joint — that is, from the quadrotor towards the payload, the
same direction {py:mod}`udaan.flatness.cable`, {doc}`quadrotor-cspayload`
and {doc}`multi-quad-cspointmass` use. With $\ell_i > 0$ the link length
and $m_i$ the point mass at its distal end, the vertex positions are

```{math}
:label: eq-m5-geom
x_i = x_Q + \sum_{k \le i} \ell_k q_k,
\qquad
x_L = x_n = x_Q + \sum_{k=1}^{n} \ell_k q_k,
\qquad
\dot q_i = \omega_i \times q_i .
```

The model **integrates the payload** $(x_L, v_L)$ and derives the
quadrotor from it, $x_Q = x_L - \sum_i \ell_i q_i$. Both the legacy
MATLAB reference and M2/M3/M4 do the same, so the choice costs nothing;
the block solve below still produces $\ddot x_Q$ first and recurses
$\ddot x_L$ forward from it.

## Mass coefficients

Substituting {math:numref}`eq-m5-geom` into
$T = \tfrac12 m_Q\|\dot x_Q\|^2 + \tfrac12\sum_i m_i \|\dot x_i\|^2$ and
collecting terms gives, with

```{math}
:label: eq-m5-coeff
M_{00} = m_Q + \sum_{k=1}^{n} m_k,
\qquad
M_{0i} = M_{i0} = \Bigl(\sum_{k \ge i} m_k\Bigr)\ell_i,
\qquad
M_{ij} = \Bigl(\sum_{k \ge \max(i,j)} m_k\Bigr)\ell_i \ell_j ,
```

the Lagrangian

```{math}
:label: eq-m5-lagrangian
\begin{aligned}
T &= \tfrac12 M_{00}\|\dot x_Q\|^2
     + \sum_k M_{0k}\,\dot x_Q \cdot \dot q_k
     + \tfrac12 \sum_{k,j} M_{kj}\,\dot q_k \cdot \dot q_j
     + \tfrac12 \Omega^\top J \Omega , \\
V &= g\, e_3 \cdot \Bigl(M_{00} x_Q + \sum_k M_{0k} q_k\Bigr) .
\end{aligned}
```

{math:numref}`eq-m5-coeff` is therefore not postulated — it is read off
the kinetic energy — and it is the sign of $q_i$ in
{math:numref}`eq-m5-geom` that fixes the signs of $M_{0i}$ and of the
gravity term.

## Equations of motion

The $x_Q$ equation is ordinary Euler–Lagrange. For each $q_i$ the
constrained variation $\delta q_i = \xi_i \times q_i$ with
$\xi_i \perp q_i$ reduces stationarity to

$$
\widehat{q_i}\Bigl[\tfrac{d}{dt}\tfrac{\partial L}{\partial \dot q_i}
  - \tfrac{\partial L}{\partial q_i}\Bigr] = 0 .
$$

Using $\ddot q_j = \dot\omega_j \times q_j - c_j q_j$ with
$c_j := \|\dot q_j\|^2$, and writing $\alpha := \ddot x_Q + g e_3$ — at
which point gravity cancels *identically* from every link row — the whole
system is one symmetric $(3+3n) \times (3+3n)$ block solve:

```{math}
:label: eq-m5-system
\begin{bmatrix}
  M_{00} I & -M_{01}\widehat{q_1} & \cdots & -M_{0n}\widehat{q_n} \\
  M_{01}\widehat{q_1} & M_{11} I & \cdots & -M_{1n}\widehat{q_1}\widehat{q_n} \\
  \vdots & & \ddots & \vdots \\
  M_{0n}\widehat{q_n} & -M_{n1}\widehat{q_n}\widehat{q_1} & \cdots & M_{nn} I
\end{bmatrix}
\begin{bmatrix}\alpha \\ \dot\omega_1 \\ \vdots \\ \dot\omega_n\end{bmatrix}
=
\begin{bmatrix}
  f R e_3 + \sum_k M_{0k} c_k q_k \\
  \sum_{j \ne 1} M_{1j} c_j\, \widehat{q_1} q_j \\
  \vdots \\
  \sum_{j \ne n} M_{nj} c_j\, \widehat{q_n} q_j
\end{bmatrix}.
```

Symmetry follows from $\widehat{q}^\top = -\widehat{q}$, which maps each
off-diagonal block onto its transpose. The payload acceleration is then
recursed forward,

```{math}
:label: eq-m5-payload
\ddot x_Q = \alpha - g e_3,
\qquad
\ddot x_L = \ddot x_Q + \sum_i \ell_i \bigl(\dot\omega_i \times q_i - c_i q_i\bigr),
```

and the quadrotor attitude line is unchanged from {doc}`quadrotor`.

### Why the diagonal block is $M_{ii} I$

The Lagrangian diagonal block is
$-M_{ii}\widehat{q_i}\widehat{q_i} = M_{ii}(I - q_i q_i^\top)$, of rank
2: $\dot\omega_i$ along $q_i$ does not enter the dynamics at all, being
the gauge freedom of the $S^2$ parametrisation. Replacing it by
$M_{ii} I$ adds $M_{ii} q_i q_i^\top$ and *nothing else*, because every
other term in link row $i$ is a cross product with $q_i$ and so has no
$q_i$ component. The substitution is therefore exact rather than
approximate: it appends the gauge-fixing equation
$\dot\omega_i \cdot q_i = 0$, which is precisely the tangency condition
$\tfrac{d}{dt}(q_i \cdot \omega_i) = \dot\omega_i \cdot q_i = 0$, and it
turns an $n$-fold singular matrix into a positive definite one.

**Invertibility.** With the Lagrangian diagonal, {math:numref}`eq-m5-system`
is the Hessian of $T$ in $(\dot x_Q, \dot q_1, \dots)$ pulled back
through $\dot q_i = -\widehat{q_i}\omega_i$, hence positive semi-definite
with kernel exactly $\{(0,\dots,\omega_i \parallel q_i,\dots)\}$; the
added $\sum_i M_{ii} q_i q_i^\top$ is strictly positive there. So the
system is **positive definite whenever $m_Q > 0$ and every $m_i > 0$**,
for *every* link configuration — all links parallel, $n = 1$, arbitrarily
unequal masses and lengths. Geometry never makes it singular.

### The parallel/perpendicular elimination does not apply

In {doc}`quadrotor-cspayload`, {doc}`multi-quad-cspointmass` and
{doc}`multi-quad-rigidbody` each cable joins the payload directly to one
vehicle, so $\dot\omega_i$ appears only in link row $i$ and in the
payload row. Projecting the *quadrotor* equation onto $q_i$ annihilates
$\dot\omega_i$ and yields the scalar tension $T_i$ as an affine function
of the payload accelerations alone — which is what collapses those
systems to $3\times3$ and $6\times6$ at any number of vehicles.

Here the links form a **serial chain**. The constraint force in joint $i$
is transmitted through every downstream link, and link row $i$ couples to
link row $j$ through $-M_{ij}\widehat{q_i}\widehat{q_j}$ for every
$j \ne i$. Projecting link row $i$ onto $q_i$ annihilates *every* term,
not just $\dot\omega_i$, and so returns the gauge condition
$\dot\omega_i \cdot q_i = 0$ and no information whatsoever about
$\alpha$. Equivalently, before the $M_{ii}I$ gauge fix the system has
rank exactly $3 + 2n$ — each link row loses exactly one equation, to its
own gauge direction — and the lost equation says nothing about $\alpha$.
**No per-link projection decouples the chain**, so M4's elimination has
no analogue here and the dense $(3+3n)$ solve is the natural direct form;
$n=1$ is the only case that reduces to a closed form by hand (that closed
form being M2's).

This bounds *this* family of eliminations, not the problem. A serial
chain is precisely what the articulated-body / recursive Newton–Euler
algorithms are built for, and they solve it in $O(n)$ rather than the
$O(n^3)$ of a dense factorisation. It is not used here because at the
link counts the ZOH-Euler stability limit permits ($n \lesssim 5$, below)
the solve is not the bottleneck, and a single assembly path that reads
like the Lagrangian is worth more than a faster one that does not.

One reduction *is* available and is deliberately not taken. The $(0,0)$
block is $M_{00} I$, a scalar multiple of the identity, so $\alpha$ can
be eliminated to leave a $3n \times 3n$ system in $\dot\omega$ with
off-diagonal blocks
$\bigl(M_{0i}M_{0j}/M_{00} - M_{ij}\bigr)\widehat{q_i}\widehat{q_j}$.
That saves three unknowns out of $3+3n$ — half at $n=1$, nothing
asymptotically — in exchange for a second assembly path to keep in
agreement with the first.

### The centripetal coefficient

{math:numref}`eq-m5-system` is assembled from the ansatz
$\ddot q_i = \dot\omega_i \times q_i - c_i q_i$ with
$c_i = \|\dot q_i\|^2 = \|\omega_i \times q_i\|^2$, where the legacy
MATLAB writes $\|\omega_i\|^2$. Differentiating
$\dot q_i = \omega_i \times q_i$ twice gives the exact kinematics

$$
\ddot q_i = \dot\omega_i \times q_i
            + (\omega_i\cdot q_i)\,\omega_i - \|\omega_i\|^2 q_i ,
$$

so the ansatz drops

$$
(\omega_i\cdot q_i)\bigl[\omega_i - (\omega_i\cdot q_i) q_i\bigr] ,
$$

which vanishes identically when $\omega_i \perp q_i$. **The form used is
therefore exact on the tangent manifold $T_{q_i}S^2$, and wrong off it at
first order in the tangency residual $\omega_i \cdot q_i$** — as is
$\|\omega_i\|^2$, by a different first-order amount.

The identity $q_i \cdot \ddot q_i = -\|\dot q_i\|^2$ is often quoted here
and settles less than it looks. The dropped term is orthogonal to $q_i$
— dot it with $q_i$ and the two pieces cancel — so the identity
constrains only the **radial** component of $\ddot q_i$ and is blind to
the discrepancy entirely. It fixes the *coefficient* $c_i$ given the
$-c_i q_i$ ansatz; it does not establish the ansatz.

This is a modelling choice with a bounded cost, not an oversight. The
tangency residual is a pure integration artefact: the continuous dynamics
conserve $q_i\cdot\omega_i = 0$ exactly (the $M_{ii} I$ gauge row makes
$\dot\omega_i\cdot q_i = 0$ an equation of the solve) and ZOH Euler lets
it drift at $O(h)$, so the dropped term is $O(h)$ — the same order as the
integrator's own error, and the same trade M2, M3 and M4 make. At the
worst drift measured in the invariants test ($q_i\cdot\omega_i \approx
0.21$ at $n = 3$, $h = 5$ ms) the dropped term is $\sim 0.2$ rad/s² per
link and falls with $h$ like everything else. Choosing $\|\dot q_i\|^2$
over $\|\omega_i\|^2$ is what keeps the $n=1$ reduction to M2 exact
rather than merely close, since M2–M4 use the same coefficient under the
same ansatz.

## Static trim

Setting every derivative and rate to zero in {math:numref}`eq-m5-system`
leaves $\widehat{q_i}\,e_3 = 0$ from each link row — every link vertical,
with $q_i = -e_3$ the hanging branch — and then

```{math}
:label: eq-m5-trim
f R e_3 = M_{00}\, g\, e_3
\quad\Longrightarrow\quad
R = I, \qquad f = \Bigl(m_Q + \sum_k m_k\Bigr) g, \qquad M = 0 .
```

The thrust carries the whole system weight and the moment is zero. There
is no grasp condition to satisfy, unlike {doc}`multi-quad-rigidbody`: a
single chain can hover from any hanging configuration.

## Numerical stability

M5 is the first system here whose *discretisation* has a stability limit
worth stating, and it is a property of the chain rather than a defect of
the algebra. Linearising about the hanging equilibrium with the airframe
free to translate — writing $q_i = -e_3 + u_i$ with $u_i$ horizontal, so
that $e_3 \cdot q_i \approx -1 + \tfrac12\|u_i\|^2$ — gives, per
horizontal axis,

```{math}
:label: eq-m5-modes
\mathrm{diag}\bigl(0,\, g M_{0k}\bigr)\,\phi = \lambda
\begin{bmatrix} M_{00} & M_{0j} \\ M_{0i} & M_{ij}\end{bmatrix}\phi ,
```

whose zero eigenvalue is the free translation of the whole system and
whose remaining $n$ are the chain's pendulum modes
({py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.natural_frequencies`).
The top frequency grows with the link count roughly like $n^{3/2}$ and
with the payload-to-link mass ratio like $\sqrt{m_L/m_i}$: a 1 m cable of
10 g in five links already carries a 135 rad/s mode.

Forward Euler on an undamped oscillator multiplies amplitudes by
$\sqrt{1 + (\omega h)^2}$ per substep, i.e. $e^{\omega^2 h/2}$ per second
of simulation.
{py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.stability_growth_rate`
returns that exponent and the constructor warns above $1.0$, a threshold
calibrated by measurement: over a 3 s taut run at $h = 1.25$ ms a chain
at $0.45$ stays bounded, one at $1.46$ winds up to $\sim 20$ rad/s, and
one at $3.4$ overflows.

Four consequences, none of them a bug:

* the defaults ($n = 3$, a 0.1 kg cable) sit at $0.45$, with about a
  factor of two of margin — and both constants are *new with M5*, so
  nothing that existed before it moved when they were added;
* the warning is a property of the whole configuration, not of the cable
  mass alone: holding the default 0.1 kg cable and raising only the link
  count gives $0.008$ ($n=1$), $0.45$ ($n=3$), $1.46$ ($n=5$), $6.39$
  ($n=10$), so `num_links=10` warns on the defaults. That is expected —
  $6.39$ is far past the measured transition and such a chain really does
  wind up under ZOH Euler — and the remedy is the MuJoCo backend or a
  smaller $h$, not a quieter threshold;
* a **slack** chain has no limit at all — under zero thrust the system is
  in free fall, the links carry no tension and every internal frequency
  is zero, which is why the energy and momentum checks run at every $n$
  while the closed-loop ones do not;
* the MuJoCo backend has no such restriction, so a configuration this
  model warns about is one to run there.

## Relationship to the single-cable case

At $n = 1$, {math:numref}`eq-m5-system` is $6 \times 6$ and eliminates by
hand. The link row gives
$\dot\omega = -\tfrac{1}{\ell}\widehat{q}\,\alpha$; substituting into the
quadrotor row and splitting along and across $q$ gives

$$
q \cdot \alpha = \frac{q \cdot f R e_3 + m_1 \ell c}{m_Q + m_1},
\qquad
\alpha^{\perp} = \frac{(f R e_3)^{\perp}}{m_Q} ,
$$

and hence, through {math:numref}`eq-m5-payload`,

$$
\ddot x_L = \frac{q \cdot f R e_3 - m_Q \ell\, c}{m_Q + m_1}\, q - g e_3 ,
\qquad
\dot\omega = -\frac{1}{m_Q \ell}\,\widehat{q}\, f R e_3 ,
$$

which is {doc}`quadrotor-cspayload` term for term, for *any* $c$. That is
the strongest available check on the derivation: M2 was verified
independently, the two code paths share nothing (M2 evaluates the closed
form, M5 solves a $6\times6$), and over 400 steps of an identical
initial condition and wrench the worst component gap is $2.3\times
10^{-14}$. Round-off is the honest claim rather than bit-exactness — two
different linear solves are only ever guaranteed to agree to rounding.

## Validation

Six independent checks, exercised in
`tests/test_quadrotor_flexcable_base.py`:

1. **reduction to M2 at $n = 1$** (above), the only one that compares
   against code verified some other way;
2. the **static trim** of {math:numref}`eq-m5-trim`, re-derived in the
   test from Newton's laws without reading the model, applied open loop;
3. **energy** under zero input, converging at $O(h)$;
4. **linear and angular momentum** under gravity alone;
5. **conditioning** of {math:numref}`eq-m5-system` on degenerate
   geometries — all links parallel, $n=1$, six orders of magnitude of
   mass ratio;
6. the **manifold invariants** $\|q_i\| = 1$ (exact) and
   $q_i \cdot \omega_i = 0$ (drifting at $O(h)$, as in M2–M4).

A seventh, cheap and entirely independent of the block solve: whatever
the chain does internally its constraint forces are internal, so
$\sum_i m_i a_i = f R e_3 - M_{00} g e_3$ must hold exactly. It does, to
$10^{-15}$, at every link count tested.

## References

```{footbibliography}
```
