# Multiple quadrotors carrying a flexible hose

Several quadrotors are bolted at chosen points along **one** flexible
hose, discretised into $n$ rigid links joined by ideal spherical joints.
This is {doc}`quadrotor-flexcable` with the single airframe at the head
of the chain replaced by an arbitrary attachment pattern, and
{doc}`multi-quad-cspointmass`'s cooperative allocation applied to a load
that is a chain rather than a point. The configuration manifold is

$$
Q = \mathbb{R}^3 \times \left(S^2\right)^{n} \times SO(3)^{n_Q},
$$

so the integrated state is $(x_0, v_0, \{q_i, \omega_i\}, \{R_k,
\Omega_k\})$: $6 + 6n + 12 n_Q$ stored floats with each $R_k$ as a
matrix, or $6 + 4n + 6 n_Q$ manifold dimensions. The simulation model is
{py:class}`~udaan.models.multi_quad_hose.MultiQuadHoseBase`.

:::{admonition} One hose, not one per vehicle
:class: note

A vehicle is a property of a **vertex**, not of a segment of the chain.
The hose has $n + 1$ vertices; an indicator over them says which carry an
airframe, and $n$ and $n_Q$ are independent. Nothing in the equations
below assumes the vehicles are at the ends, evenly spaced, or even more
than one.
:::

## Coordinates and constraints

Index the vertices $j = 0 \dots n$ and the links $i = 0 \dots n-1$, with
link $i$ running from vertex $i$ to vertex $i+1$. Let $q_i \in S^2$ be
the unit vector along link $i$ oriented *down the chain*, $\ell_i > 0$
its length, and $\bar m_j$ the total mass lumped at vertex $j$ — the
hose's own share plus the airframe's, where there is one. Then

```{math}
:label: eq-m6-geom
x_{i+1} = x_i + \ell_i q_i,
\qquad
x_j = x_0 + \sum_{i<j} \ell_i q_i,
\qquad
\dot q_i = \omega_i \times q_i .
```

The model **integrates vertex 0** and derives every other vertex from
it. The legacy MATLAB reference integrates the same pair. This is the
opposite of {doc}`quadrotor-flexcable`, which integrates the payload at
the *distal* end and derives the airframe — that model has exactly one
airframe and it is at the proximal end, so there is a distinguished node
to derive; here neither end is special, so the root is the state.

## Mass coefficients

Substituting {math:numref}`eq-m6-geom` into $T = \tfrac12 \sum_j \bar m_j
\|\dot x_j\|^2$ and collecting terms gives, with $S_i = \sum_{j>i} \bar
m_j$ the mass hanging strictly below link $i$,

```{math}
:label: eq-m6-coeff
M_{00} = \sum_j \bar m_j,
\qquad
M_{0i} = M_{i0} = S_i \ell_i,
\qquad
M_{ij} = S_{\max(i,j)}\, \ell_i \ell_j ,
```

the Lagrangian

```{math}
:label: eq-m6-lagrangian
\begin{aligned}
T &= \tfrac12 M_{00}\|\dot x_0\|^2
     + \sum_k M_{0k}\,\dot x_0 \cdot \dot q_k
     + \tfrac12 \sum_{k,j} M_{kj}\,\dot q_k \cdot \dot q_j
     + \tfrac12 \sum_k \Omega_k^\top J_k \Omega_k , \\
V &= g\, e_3 \cdot \Bigl( M_{00} x_0 + \sum_k M_{0k} q_k \Bigr).
\end{aligned}
```

This is **exactly** {doc}`quadrotor-flexcable`'s Lagrangian, with that
model's $m_Q$ generalised to $\bar m_0$. The two systems differ only in
where the external forces are applied.

## Generalised forces: implicit tension propagation

Let $F_j = f_k R_k e_3$ be the thrust at the vertex vehicle $k$ sits on,
and zero at every unattached vertex. From $\delta x_j = \delta x_0 +
\sum_{i<j} \ell_i \delta q_i$,

```{math}
:label: eq-m6-work
\delta W
= \Bigl(\sum_j F_j\Bigr)\cdot \delta x_0
+ \sum_i \ell_i\, G_i \cdot \delta q_i ,
\qquad
G_i := \sum_{j>i} F_j ,
```

so the generalised force conjugate to $q_i$ is $\ell_i G_i$: the net
external force on **everything below link $i$**. That single cumulative
sum is how tension propagates in this model. There is no per-link scalar
tension anywhere in the derivation and none is ever assembled — $G_i$,
projected perpendicular to $q_i$, carries all of it. It is the analogue
of {doc}`quadrotor-flexcable`'s recursion, and it is what lets the same
equations serve any attachment pattern.

## Equations of motion

The $x_0$ equation is ordinary; for each $q_i$ the constrained variation
$\delta q_i = \xi_i \times q_i$, $\xi_i \perp q_i$, reduces stationarity
to $\hat q_i[\,\cdot\,] = 0$. Substituting $\ddot q_j = \dot\omega_j
\times q_j - c_j q_j$ with $c_j := \|\dot q_j\|^2$, and writing $\alpha
:= \ddot x_0 + g e_3$ — gravity then cancels identically from every link
row, because $\ell_i S_i = M_{0i}$ — gives

```{math}
:label: eq-m6-system
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
```

a symmetric $(3 + 3n) \times (3 + 3n)$ solve, followed by $\ddot x_0 =
\alpha - g e_3$, the chain recursion for every other vertex, and the
usual per-vehicle attitude dynamics

$$
\dot R_k = R_k \hat\Omega_k,
\qquad
J_k \dot\Omega_k = M_k - \Omega_k \times J_k \Omega_k .
$$

**Given the same $\bar m$, the matrix does not depend on the attachment
pattern.** It is then {doc}`quadrotor-flexcable`'s matrix, coefficient for
coefficient, and the vehicles' *inputs* enter only through $\sum_j F_j$
and $G_i$ on the right.

The pattern is not absent from the matrix, though — it is most of what
sets $\bar m$ in the first place, by deciding which vertices carry an
$m_Q$. At $n = 4$ on the shipped link geometry, $\|M\|_F$ runs 3.17 for
vehicles at $(2,)$, 4.89 for $(0,4)$, 6.84 for $(0,2,4)$ and 10.75 for
$(0,1,2,3,4)$. What is pattern-blind is the map $\bar m \mapsto M$, and
that is what the implementation asserts: two models with different
patterns but equal $\bar m$ produce the same matrix.

Setting $n_Q = 1$ with the vehicle at vertex 0 makes every $G_i = 0$ and
recovers {doc}`quadrotor-flexcable` exactly — which the implementation
asserts numerically, to $5\times10^{-15}$.

```{note}
That reduction is weaker evidence than it reads. The two assembly
routines share no code in the import sense — separate methods in separate
modules, neither importing the other — but they are near-verbatim
structural clones: same loop, same $M_{ii} I$ gauge substitution, same
$c_i = \|\dot q_i\|^2$, same dense solve. A derivation error common to
both would reproduce itself faithfully. The reduction is a good
consistency check on the port; what actually validates the equations is
the constrained-DAE re-derivation — different unknown vector, Lagrange
multipliers on $\|q_i\| = 1$, a KKT solve, nothing read off the
coefficients above — which agrees to $2\times10^{-12}$ across
$n \in \{1,\dots,8\}$, every attachment pattern and heterogeneous
parameters.
```

### Why the diagonal block is $M_{ii} I$

Unchanged from {doc}`quadrotor-flexcable`, and for the same reason. The
Lagrangian block is $-M_{ii}\hat q_i \hat q_i = M_{ii}(I - q_i
q_i^\top)$, rank 2: $\dot\omega_i$ along $q_i$ is the gauge freedom of
the $S^2$ parametrisation and appears nowhere in the dynamics. Replacing
it by $M_{ii} I$ adds $M_{ii} q_i q_i^\top$ and *nothing else*, because
every other term in link row $i$ — the new force term $\ell_i \hat q_i
G_i$ included — is a cross product with $q_i$. The substitution is
therefore not an approximation: it appends the exact tangency condition
$\dot\omega_i \cdot q_i = 0$ and turns an $n$-fold singular matrix into a
positive definite one.

Positive definiteness needs $\bar m_j > 0$ at *every* vertex. A vertex
whose total mass is zero lets the two links meeting there hinge freely
without moving any mass — an extra three-dimensional kernel no gauge fix
removes — and the model refuses it. The hose's own mass at a vertex may
be zero; an airframe there supplies $\bar m_j$ on its own, which is
exactly the configuration the reduction above uses.

### No elimination decouples a serial chain

In {doc}`multi-quad-cspointmass` and {doc}`multi-quad-rigidbody` each
cable joins the payload directly to one quadrotor, so $\dot\omega_i$
appears only in link row $i$ and in the payload row; projecting the
*quadrotor* equation onto $q_i$ yields the scalar tension as an affine
function of the payload accelerations and collapses the system to
$3 \times 3$ or $6 \times 6$ at any number of vehicles.

Here link row $i$ couples to link row $j$ through $-M_{ij}\hat q_i \hat
q_j$ for every $j \ne i$. Projecting link row $i$ onto $q_i$ annihilates
*every* term — each is a $\hat q_i(\cdot)$, the force term included — so
it returns the gauge condition and nothing about $\alpha$. Attaching more
vehicles to more vertices changes only the right-hand side, so it cannot
change the answer: {doc}`quadrotor-flexcable`'s finding carries over
verbatim, and the dense $(3+3n)$ solve is the natural direct form.

## Static trim

Set every rate and acceleration to zero and introduce the **tension
vector** $\tau_i$ of link $i$ through $G_i = S_i g e_3 + \tau_i$. Link
row $i$ collapses to $\hat q_i \tau_i = 0$: each link points along its
own tension. Telescoping $F_j = G_{j-1} - G_j$ with $\tau_{-1} = \tau_n =
0$ gives the vertex balance

```{math}
:label: eq-m6-trim
F_j = \bar m_j\, g\, e_3 + \tau_{j-1} - \tau_j ,
\qquad
q_i = \frac{\tau_i}{\|\tau_i\|} ,
\qquad
\sum_j F_j = M_{00}\, g\, e_3 ,
```

so an equilibrium is exactly a choice of $\{\tau_i\}$ making $F_j = 0$ at
every vertex *without* a vehicle. This is the discrete catenary, it is
closed form, and
{py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.hover_trim`
propagates it.

Two branches are worth naming. Putting the whole system weight on the
topmost vehicle gives $\tau_i = -g S_i e_3$: every link vertical, every
tension the weight hanging under it, and at $n_Q = 1$ the familiar $f =
M_{00} g$, $R = I$, $M = 0$.

```{warning}
Fed open loop this trim holds *exactly* — root drift 0.0 over a second —
but that number is conditional on the actuator, not only on the algebra.
The default split asks one airframe for the whole system weight: 22.6 N
at $n_Q = 2$, 49.1 N at $n_Q = 5$, against a shipped `_max_thrust` of
20 N. On shipped limits `step()` clips the very command that would have
held, and the root falls 0.56 m ($n_Q = 2$) to 2.9 m ($n_Q = 5$) in that
same second. `hover_trim` warns when the split exceeds the limit, and the
test reporting 0.0 raises the limit first — deliberately, because the
claim under test is about the equations of motion. A trim you intend to
*fly* wants a split the airframes can deliver.
```

Spreading the thrust drives the cumulative
"thrust above minus weight above" through zero somewhere, so the hose
*folds* — the links below the crossing point pointing back up — which is
just as much an equilibrium. Landing exactly on zero leaves a slack link
with no defined direction, and that configuration (one vehicle per
vertex, each carrying its own weight, say) simply has no vertical trim;
give the vehicles horizontal thrust components and it becomes a catenary
with tension everywhere.

The legacy reference solves the same conditions numerically, minimising
total thrust subject to {math:numref}`eq-m6-trim` plus a chain-closure
constraint pinning the vehicles to prescribed positions. That is the
useful general form — a *setpoint solver* rather than part of the plant —
and it is not reimplemented here.

## Numerical stability

Linearising about the free-hanging equilibrium gives modes solving
$\mathrm{diag}(0, g M_{0k})\phi = \lambda M \phi$, and forward Euler on
an undamped oscillator of frequency $\omega$ multiplies amplitudes by
$\sqrt{1 + (\omega h)^2}$ per substep — $e^{\omega^2 h/2}$ per second.
{py:meth}`~udaan.models.multi_quad_hose.MultiQuadHoseBase.stability_growth_rate`
returns that exponent and the constructor warns above 1.0.

On the shipped parameters, at $n_Q = 2$:

| $n$  | 1 | 2 | 4 | 6 | 8 | 16 |
|------|---|---|---|---|---|----|
| rate | 0.006 | 0.09 | 0.49 | 1.13 | 2.02 | 8.22 |

That row alone would put the usable region at $n \le 5$ — the same limit
{doc}`quadrotor-flexcable` reports for the same integrator on the same
kind of chain. It is one row, and the wrong one to generalise from.

$n_Q$ matters much less than $n$, but it is not negligible and it is not
monotonic: what the fastest mode responds to is the mass *contrast*
between neighbouring vertices, not the vehicle count. At $n = 4$, going
from vehicles at $(0,4)$ to $(0,2,4)$ raises the exponent slightly, 0.49
to 0.55, while bolting one to every vertex evens the chain out and drops
it to 0.13. Clustering the vehicles at one end does the opposite.
Enumerating *every* pattern on the shipped parameters:

| $n$ | patterns | over the 1.0 warn | worst |
|-----|----------|-------------------|-------|
| 3   | 15       | 0                 | 0.42 at $(2,3)$ |
| 4   | 31       | 0                 | 1.00 at $(2,3,4)$ |
| 5   | 63       | 23                | 1.94 at $(2,3,4,5)$ |
| 6   | 127      | 101               | 3.34 at $(2,3,4,5,6)$ |

**so the honest envelope at any $n_Q$ is $n \le 4$.** At $n = 5$, 23 of
the 63 patterns cross the warn threshold — $(2,3,4,5)$ at 1.94,
$(3,4,5)$ at 1.83, $(0,3,4,5)$ at 1.69 — even though the endpoints-only
$(0,5)$ of the table above sits at a comfortable 0.78. Above $n = 4$ the
limit is pattern-dependent, and $n = 4$ itself is only just inside it:
$(2,3,4)$ reaches 0.997. The shipped default, $n = 4$ with vehicles at
$(0,4)$, scores 0.49 and is not close to trouble.

A configuration this model warns about is one to run on the MuJoCo
backend, which integrates the same continuous dynamics with its own
solver.

## The reference

The legacy `multiple-quadrotor-flexible-hose` repository is the best
written in the collection, and its dynamics routine survived
re-derivation intact: the mass coefficients, the cumulative force sum,
the gauge substitution and the sign of every term agree with
{math:numref}`eq-m6-system`. What did not survive is a *damped* variant
that adds $-c\,\omega_k = -c\,(q_k \times \dot q_k)$ to $\ddot q_k$ after
the mass-matrix solve. That vector is orthogonal to $\dot q_k$, so its
power vanishes identically: the term is gyroscopic and precesses each
link rather than dissipating. The correct term is $-c\,\dot q_k$, as a
generalised force inside the solve. No damped variant is ported here.

**It is dead code, and no published result depends on it**: the variant
is reachable only by asking for `'dynamics', 'damped'`, all five example
scripts pass `'default'` or `'mex'`, and the codegen path sets
`damp_coeff = 0`. It is a latent bug in an unused branch, not an error in
a result.

One finding is on the published path, and is worth separating from that.
`optimize_for_des_state` — reached from `computeSetpoint`, which the
$n_Q = 2$ and $n_Q = 3$ setpoint simulations both call — carries a cost
comment reading "sum of the square of the tension norms" over an
expression that sums squared *thrusts*. Minimising thrust is the stated
intent of the method, so the comment is what is wrong. More
consequential: the solver leaves the tension magnitudes unbounded in
sign, so nothing stops it returning a setpoint with links in
compression, and a hose cannot push.
