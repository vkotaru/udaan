(flatness-quadrotor)=
# Quadrotor

This page derives the differential-flatness property of a rigid-body
quadrotor under collective-thrust/body-moment actuation, as established
by Mellinger and Kumar {footcite}`mellinger2011minimum`. The flat
output is $y = (x_Q,\, \psi)$, with $x_Q \in \mathbb{R}^3$ the
centre-of-mass position and $\psi \in \mathbb{R}$ the yaw angle. The
flat-to-state recovery requires the position $x_Q$ up to the snap
$y^{(4)}$ and the yaw up to $\ddot\psi$.

## System

Let $x_Q \in \mathbb{R}^3$ be the quadrotor centre-of-mass position in
the world frame, $R \in SO(3)$ the attitude (body-to-world), and
$\Omega \in \mathbb{R}^3$ the body-frame angular velocity so that
$\dot R = R\,\widehat{\Omega}$ (see {doc}`../preliminaries`). The
collective thrust $f \in \mathbb{R}_{\geq 0}$ acts along the body
$e_3$-axis and the moment $M \in \mathbb{R}^3$ acts in the body frame.
With mass $m > 0$, inertia $J = J^\top \succ 0$, and gravity $g$ along
the world up-axis $e_3$, the equations of motion are

```{math}
:label: eq-quad-eom
\begin{aligned}
m\, \ddot x_Q &= f\, R\, e_3 \,-\, m\, g\, e_3, \\[2pt]
\dot R &= R\, \widehat{\Omega}, \\[2pt]
J\, \dot\Omega + \Omega \times J\, \Omega &= M.
\end{aligned}
```

The four scalar control inputs are $(f,\, M) \in \mathbb{R}^4$, so
$\dim u = 4$. The full state $(x_Q,\, \dot x_Q,\, R,\, \Omega)$ lives
in $\mathbb{R}^3 \times \mathbb{R}^3 \times TSO(3)$, which has
dimension $12$.

## Flat output

Take

```{math}
:label: eq-quad-flat-output
y \;=\; (x_Q,\, \psi) \;\in\; \mathbb{R}^3 \times \mathbb{R}.
```

Then $\dim y = 4 = \dim u$, so the dimension-matching necessary
condition for differential flatness (see {doc}`index`) is satisfied.
The remainder of this page constructs the algebraic map from the jet
of $y$ to $(R,\, \Omega,\, f,\, M)$, confirming that $y$ is a flat
output. Inspection of the recovery below shows that $x_Q$ is needed
up to $y^{(4)}$ (snap) and $\psi$ is needed up to $\ddot\psi$.

## Constructive recovery

### Step 1 — thrust magnitude and direction

The translational equation of {math:numref}`eq-quad-eom` rearranges to
$f\, R\, e_3 = m\, (\ddot x_Q + g\, e_3)$. Collect the right-hand side
into a single vector,

```{math}
:label: eq-quad-A
A \;=\; m\,\bigl(\ddot x_Q + g\, e_3\bigr) \;\in\; \mathbb{R}^3.
```

Since $R\, e_3$ is a unit vector, the thrust magnitude is $f = \|A\|$
and the thrust direction is

```{math}
:label: eq-quad-b3
b_3 \;=\; R\, e_3 \;=\; \frac{A}{\|A\|}.
```

Both quantities are functions of $\ddot x_Q$ alone (through $A$).
Differentiating {math:numref}`eq-quad-b3` with respect to time and
using $\dot{\|A\|} = (A \cdot \dot A) / \|A\| = \dot A \cdot b_3$,

```{math}
:label: eq-quad-db3
\dot b_3
 \;=\; \frac{\dot A}{\|A\|}
      - \frac{A \cdot \dot A}{\|A\|^3}\, A
 \;=\; \frac{1}{\|A\|}\Bigl(\dot A - (\dot A \cdot b_3)\, b_3\Bigr),
```

i.e. $\dot b_3$ is the projection of $\dot A / \|A\|$ onto the plane
$T_{b_3} S^2$ orthogonal to $b_3$ — automatic because $\|b_3\| = 1$
forces $b_3 \cdot \dot b_3 = 0$ (see the $S^2$ tangent-space discussion
in {doc}`../preliminaries`). Since $\dot A = m\, x_Q^{(3)}$, computing
$\dot b_3$ needs the jerk.

Differentiating once more, the second derivative of the norm is

```{math}
:label: eq-quad-d2norm-A
\ddot{\|A\|}
 \;=\; \frac{\dot A \cdot \dot A \,+\, A \cdot \ddot A \,-\, \dot{\|A\|}^{\,2}}{\|A\|},
```

so

```{math}
:label: eq-quad-d2b3
\ddot b_3
 \;=\; \frac{1}{\|A\|}\Bigl(\ddot A \,-\, 2\,\dot b_3\, \dot{\|A\|}
        \,-\, b_3\, \ddot{\|A\|}\Bigr),
```

which requires $\ddot A = m\, x_Q^{(4)}$, i.e. the snap.

### Step 2 — desired attitude

Fix a yaw-heading hint in the world frame,

```{math}
:label: eq-quad-b1d
b_{1,d}(\psi) \;=\; [\cos\psi,\; \sin\psi,\; 0]^\top,
```

with derivatives

```{math}
:label: eq-quad-db1d
\begin{aligned}
\dot b_{1,d}  &\;=\; [-\sin\psi,\; \cos\psi,\; 0]^\top\, \dot\psi, \\[2pt]
\ddot b_{1,d} &\;=\; [-\cos\psi,\; -\sin\psi,\; 0]^\top\, \dot\psi^{\,2}
                     \,+\, [-\sin\psi,\; \cos\psi,\; 0]^\top\, \ddot\psi.
\end{aligned}
```

Provided $b_3$ is not parallel to $b_{1,d}$, the pair
$(b_3,\, b_{1,d})$ determines a right-handed orthonormal frame via

```{math}
:label: eq-quad-b2-b1
b_2 \;=\; \frac{b_3 \times b_{1,d}}{\|b_3 \times b_{1,d}\|},
\qquad
b_1 \;=\; b_2 \times b_3.
```

The attitude is then

```{math}
:label: eq-quad-R
R \;=\; [\, b_1 \;\; b_2 \;\; b_3 \,] \;\in\; SO(3).
```

This is the same construction performed by
`SO3.from_two_vectors(b3, b1d)` in {doc}`../preliminaries` — but the
implementation in {py:class}`udaan.utils.flatness.Quadrotor` builds
$R$ inline as ``np.column_stack([b1, b2, b3])`` rather than calling
the constructor, because the intermediate columns $b_1$ and $b_2$ are
also needed downstream (their derivatives feed Step 3 / Step 4 via
{math:numref}`eq-quad-dR`). The third column is fixed by Step 1; the
yaw angle $\psi$ chooses the in-plane rotation of $(b_1,\, b_2)$
about $b_3$.

:::{admonition} Singularity
:class: note

When $b_3$ is parallel to $b_{1,d}$ — i.e. the quadrotor is pointed
straight up/down while the yaw hint would lie along the body axis —
the cross product in {math:numref}`eq-quad-b2-b1` vanishes and $R$ is
not uniquely defined. The implementation in
{py:class}`udaan.utils.flatness.Quadrotor` raises an error at this
configuration; the regular region for the flatness map is the open
subset of jet space on which $b_3 \times b_{1,d} \neq 0$.
:::

Using $b_1 = b_2 \times b_3$ rather than a second direct projection
yields a compact recursion for the derivatives of the columns:

```{math}
:label: eq-quad-dR
\begin{aligned}
\dot b_2 &= \frac{1}{\|b_3 \times b_{1,d}\|}\Bigl(
  \frac{d}{dt}(b_3 \times b_{1,d})
  \,-\, b_2\, \frac{d}{dt}\|b_3 \times b_{1,d}\|
\Bigr), \\[2pt]
\dot b_1 &= \dot b_2 \times b_3 + b_2 \times \dot b_3,
\end{aligned}
```

where the cross-product and norm derivatives are

```{math}
:label: eq-quad-cross-derivs
\frac{d}{dt}(b_3 \times b_{1,d})
  \;=\; \dot b_3 \times b_{1,d} \,+\, b_3 \times \dot b_{1,d},
\qquad
\frac{d}{dt}\|b_3 \times b_{1,d}\|
  \;=\; \frac{(b_3 \times b_{1,d}) \cdot \tfrac{d}{dt}(b_3 \times b_{1,d})}{\|b_3 \times b_{1,d}\|}.
```

A second differentiation of the same shape (with the usual
$\|\cdot\|$-second-derivative formula from
{math:numref}`eq-quad-d2norm-A`) produces $\ddot b_1,\, \ddot b_2$, and
hence $\ddot R = [\ddot b_1\;\; \ddot b_2\;\; \ddot b_3]$.

### Step 3 — body-frame angular velocity

From $\dot R = R\, \widehat{\Omega}$ the body rate is

```{math}
:label: eq-quad-Omega
\Omega \;=\; \bigl(R^\top \dot R\bigr)^{\!\vee} \;\in\; \mathbb{R}^3,
```

using the vee map of {doc}`../preliminaries`. By Step 2, $\dot R$ is
known in terms of $(x_Q^{(3)},\, \dot\psi)$, so $\Omega$ requires the
jerk and $\dot\psi$.

### Step 4 — body-frame angular acceleration

Recovering the moment through {math:numref}`eq-quad-eom` requires
$\dot\Omega$. The implementation in
{py:class}`udaan.utils.flatness.Quadrotor` computes it through a
skew-projector on $\ddot R$:

```{math}
:label: eq-quad-dOmega
\dot\Omega \;=\; \bigl(\operatorname{skew}(R^\top \ddot R)\bigr)^{\!\vee},
\qquad
\operatorname{skew}(X) \;=\; \tfrac{1}{2}\bigl(X - X^\top\bigr).
```

The skew-projector form is chosen for a specific numerical reason,
worth unpacking against the more obvious cancellation form.

**Step 4a — expand $R^\top \ddot R$.** Differentiating
$\dot R = R\,\widehat{\Omega}$ once more gives
$\ddot R = R\,\widehat{\Omega}^{\,2} + R\,\dot{\widehat{\Omega}}$, so

$$
R^\top \ddot R
\;=\; \widehat{\Omega}^{\,2} \,+\, \dot{\widehat{\Omega}}.
$$

Rearranging gives the **cancellation form**,

$$
\dot{\widehat{\Omega}} \;=\; R^\top \ddot R \,-\, \widehat{\Omega}^{\,2},
$$

in which the $\widehat{\Omega}^{\,2}$ term is subtracted explicitly
using the previously computed body rate.

**Step 4b — skew-projector view.** Note that
$(\widehat{\Omega}^{\,2})^\top = (-\widehat{\Omega})(-\widehat{\Omega}) = \widehat{\Omega}^{\,2}$,
so $\widehat{\Omega}^{\,2}$ is *symmetric*.
The linear projector $\operatorname{skew}(X) = \tfrac{1}{2}(X - X^\top)$
annihilates symmetric matrices and is the identity on skew-symmetric
matrices. Applying it to $R^\top \ddot R = \widehat{\Omega}^{\,2} + \dot{\widehat{\Omega}}$,

$$
\operatorname{skew}(R^\top \ddot R)
\;=\; \underbrace{\operatorname{skew}(\widehat{\Omega}^{\,2})}_{=\,0}
      \,+\, \underbrace{\operatorname{skew}(\dot{\widehat{\Omega}})}_{=\,\dot{\widehat{\Omega}}}
\;=\; \dot{\widehat{\Omega}}.
$$

Applying vee recovers {math:numref}`eq-quad-dOmega`.

**Step 4c — why prefer the projector form.** Both forms yield the
same answer when $R^\top R = I$ holds exactly. Under floating-point
integration, however, $R$ drifts off $SO(3)$ and
$R^\top R = I + \mathcal{O}(\varepsilon)$ only. The cancellation form
subtracts a precomputed $\widehat{\Omega}^{\,2}$ from $R^\top \ddot R$
and then takes vee, leaving any symmetric residual from drift on the
diagonal of the result. The projector form, by contrast, rejects any
symmetric component of $R^\top \ddot R$ regardless of its source, so
it remains numerically faithful to
$\dot{\widehat{\Omega}} \in \mathfrak{so}(3)$ even in the presence of
drift — a small but real robustness benefit in long trajectory
rollouts.

:::{admonition} It is not about diagonal entries being zero
:class: note

For $\Omega = (1, 2, 3)$,

$$
\widehat{\Omega}^{\,2}
\;=\; \begin{bmatrix}
-13 & 2 & 3 \\
 2  & -10 & 6 \\
 3  & 6 & -5
\end{bmatrix},
$$

whose diagonal entries $(-13, -10, -5)$ are non-zero. It is the
*symmetry* of $\widehat{\Omega}^{\,2}$ — every off-diagonal pair
$(i, j)$ and $(j, i)$ agrees — that makes the skew projector
annihilate it, not any vanishing of its entries.
:::

With $\dot\Omega$ in hand, the body-frame Newton–Euler equation from
{math:numref}`eq-quad-eom` returns the feedforward moment:

```{math}
:label: eq-quad-M
M \;=\; J\, \dot\Omega \,+\, \Omega \times J\,\Omega.
```

Since $\dot\Omega$ depends on $\ddot R$, which by Step 2 requires
$x_Q^{(4)}$ (snap) and $\ddot\psi$, the moment requires the full
fourth-order position jet and second-order yaw jet.

## Theorem

```{prf:theorem} Flatness of the rigid-body quadrotor
:label: thm-quad-flat

The rigid-body quadrotor described by {math:numref}`eq-quad-eom` is
differentially flat, with flat output $y = (x_Q,\, \psi)$ as in
{math:numref}`eq-quad-flat-output`. Every state and every input is
an algebraic function of the fourth-order jet of $x_Q$ and the
second-order jet of $\psi$.
```

```{prf:proof}
Dimension matching: $\dim y = 4 = \dim u$. Steps 1–4 above construct
the smooth map

$$
\Phi : J^4\mathbb{R}^3 \times J^2\mathbb{R}
      \;\longrightarrow\; \mathbb{R}^{12} \times \mathbb{R}^4,
\qquad
\Phi\bigl(J^4_t\, x_Q,\, J^2_t\, \psi\bigr) \;=\;
\bigl((x_Q, \dot x_Q, R, \Omega),\; (f, M)\bigr),
$$

which recovers the full state and input without integrating the
dynamics: $(f,\, b_3)$ from $\ddot x_Q$ via
{math:numref}`eq-quad-A`–{math:numref}`eq-quad-b3`, $R$ from
$(\ddot x_Q,\, \psi)$ via {math:numref}`eq-quad-b2-b1`–{math:numref}`eq-quad-R`,
$\Omega$ from $(x_Q^{(3)},\, \dot\psi)$ via
{math:numref}`eq-quad-Omega`, and $M$ from $(x_Q^{(4)},\, \ddot\psi)$
via {math:numref}`eq-quad-dOmega`–{math:numref}`eq-quad-M`. Smoothness
of $\Phi$ holds on the open subset
$\{\,\ddot x_Q + g\, e_3 \neq 0\,\} \cap \{\,b_3 \not\parallel
b_{1,d}(\psi)\,\}$ — the former excludes the free-fall singularity
where the thrust direction is undefined, the latter excludes the
configuration where the yaw hint aligns with the thrust axis. This is
exactly the form {math:numref}`eq-flat-to-state-on-jets` on the
product jet space, so the system is differentially flat with flat
output $y$.
```

This recovery is the substrate of the minimum-snap trajectory
optimisation in {footcite}`mellinger2011minimum`; the same feedforward
is paired with the geometric controller of
{footcite}`lee2010geometric` for closed-loop tracking. Historical
context: differential flatness itself is due to Fliess, Lévine,
Martin, and Rouchon
{footcite}`fliess1995flatness,levine2009analysis`; the quadrotor is
one of its cleanest mechanical examples.

## Derivative-order accounting

| Recovered quantity                 | Needs                                   |
| ---------------------------------- | --------------------------------------- |
| $f$                                | $\ddot x_Q$                             |
| $b_3 = R\, e_3$                    | $\ddot x_Q$                             |
| $R$                                | $\ddot x_Q,\, \psi$                     |
| $\Omega$                           | $x_Q^{(3)},\, \dot\psi$                 |
| $\dot\Omega$                       | $x_Q^{(4)},\, \ddot\psi$                |
| $M$                                | $x_Q^{(4)},\, \ddot\psi$                |

The worst case is the moment $M$, which closes the chain at
$x_Q^{(4)}$ (snap) and $\ddot\psi$ — the orders that give the map its
$J^4 \times J^2$ type.

## Jet input

The map $\Phi$ of {prf:ref}`thm-quad-flat` takes a pair of
{py:class}`udaan.utils.flatness.Jet` objects — one of order 4 for the
position, one of order 2 for the yaw — packaged into a
{py:class}`udaan.utils.flatness.QuadrotorFlats` struct:

```python
import numpy as np
from udaan.utils.flatness import Jet, Quadrotor, QuadrotorFlats

# Centre-of-mass position and its first four time derivatives at t
x_jet = Jet(np.stack([
    x_Q,       # row 0 — position
    x_Q_dot,   # row 1 — velocity
    x_Q_ddot,  # row 2 — acceleration (gives f, b3)
    x_Q_dddot, # row 3 — jerk          (gives Ω)
    x_Q_snap,  # row 4 — snap          (gives dΩ, M)
]))
x_jet.order  # 4
x_jet.dim    # 3

# Yaw and its first two derivatives
psi_jet = Jet(np.array([psi, psi_dot, psi_ddot]))
psi_jet.order  # 2
psi_jet.dim    # 1

flats = QuadrotorFlats(x=x_jet, psi=psi_jet)
ref, inputs = Quadrotor(mass=m, inertia=J).recover(flats)
# ref.position, ref.velocity, ref.acceleration,
# ref.orientation, ref.angular_velocity, ref.angular_acceleration
# inputs.thrust, inputs.moment
```

Truncation ({prf:ref}`def-jet-trunc`) extracts the lower-order jets
needed by intermediate stages of the recovery (for example,
`x_jet.truncate(2)` is enough to produce $(f,\, R)$ but not
$(\Omega,\, M)$).

## References

```{footbibliography}
```
