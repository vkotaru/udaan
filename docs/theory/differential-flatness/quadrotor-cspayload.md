(flatness-quadrotor-cspayload)=
# Quadrotor with cable-suspended payload

This page derives the differential-flatness property of a single
quadrotor carrying a point-mass payload via a taut, massless cable, as
established by Sreenath, Michael, and Kumar
{footcite}`sreenath2013trajectory` and the companion geometric-control
paper of Sreenath, Lee, and Kumar {footcite}`sreenath2013geometric`.
The flat output is $y = (x_L,\, \psi)$, with $x_L \in \mathbb{R}^3$
the payload position and $\psi \in \mathbb{R}$ the quadrotor yaw angle.
The flat-to-state recovery requires $x_L$ up to the sixth derivative
$y^{(6)}$ and the yaw up to $\ddot\psi$.

## System

Let $x_L \in \mathbb{R}^3$ be the payload position and
$x_Q \in \mathbb{R}^3$ the quadrotor centre-of-mass position, both in
the world frame. Let $q \in S^2$ be the unit cable vector in the world
frame, oriented from the quadrotor *to* the payload, and
$\omega \in T_q S^2$ the
cable angular velocity (so $\omega \cdot q = 0$); see
{doc}`../preliminaries` for the $S^2 / TS^2$ conventions. Let
$R \in SO(3)$ be the quadrotor attitude and $\Omega \in \mathbb{R}^3$
the body-frame angular velocity, so $\dot R = R\,\widehat{\Omega}$.
The collective thrust $f \in \mathbb{R}_{\geq 0}$ acts along the body
$e_3$-axis and the moment $M \in \mathbb{R}^3$ acts in the body frame.
Quadrotor mass is $m_Q$, payload mass $m_L$, cable length $\ell > 0$,
and quadrotor inertia $J = J^\top \succ 0$.

The taut-cable kinematic tie between the two positions is

```{math}
:label: eq-payload-cable-geom
x_Q \;=\; x_L \,-\, \ell\, q,
\qquad
\dot q \;=\; \omega \times q.
```

With this sign convention, $q \approx -e_3$ in the hanging
equilibrium. Under the taut-cable assumption the system is a hybrid
rigid body; from {footcite}`sreenath2013geometric` its equations of
motion are

```{math}
:label: eq-payload-eom
\begin{aligned}
(m_Q + m_L)\bigl(\ddot x_L + g\, e_3\bigr)
  &= \bigl(q \cdot f R\, e_3 \,-\, m_Q\, \ell\, \|\dot q\|^2\bigr)\, q, \\[2pt]
m_Q\, \ell\, \dot\omega &= -\, q \times f R\, e_3, \\[2pt]
\dot R &= R\, \widehat{\Omega}, \\[2pt]
J\, \dot\Omega \,+\, \Omega \times J\, \Omega &= M.
\end{aligned}
```

The first line of {math:numref}`eq-payload-eom` is Newton's law for the
combined system resolved along $q$ (the only direction in which the
cable transmits force), the second is the cable angular dynamics on
$T_q S^2$, and the third–fourth are the rigid-body attitude dynamics.
The four scalar control inputs are $(f,\, M) \in \mathbb{R}^4$, so
$\dim u = 4$. The full state
$(x_L,\, \dot x_L,\, q,\, \omega,\, R,\, \Omega)$ lives in
$\mathbb{R}^3 \times \mathbb{R}^3 \times TS^2 \times TSO(3)$, which has
dimension $16$ (agreeing with the 8 DOF / 4-degree underactuation
count of {footcite}`sreenath2013trajectory`: $3 + 3 + 4 + 6 = 16$,
since $\dim TS^2 = 4$ and $\dim TSO(3) = 6$).

## Flat output

Take

```{math}
:label: eq-payload-flat-output
y \;=\; (x_L,\, \psi) \;\in\; \mathbb{R}^3 \times \mathbb{R}.
```

Then $\dim y = 4 = \dim u$, so the dimension-matching necessary
condition for differential flatness (see {doc}`index`) is satisfied.
The remainder of this page constructs the algebraic map from the jet
of $y$ to $(x_L,\, \dot x_L,\, q,\, \omega,\, R,\, \Omega,\, f,\, M)$,
confirming that $y$ is a flat output. Inspection of the recovery below
shows that $x_L$ is needed up to $y^{(6)}$ and $\psi$ is needed up to
$\ddot\psi$.

## Constructive recovery

### Step 1 — cable direction from payload dynamics

The cable direction can be recovered two equivalent ways: from the
constrained-Lagrangian first line of {math:numref}`eq-payload-eom`, or
from a Newton–Euler cut on the payload alone with the cable tension
made explicit. Both yield the same unit vector $q$.

**Lagrangian form.** Group the gravity-augmented payload acceleration
with the *combined* mass into a single vector,

```{math}
:label: eq-payload-A
A \;:=\; (m_Q + m_L)\bigl(\ddot x_L + g\, e_3\bigr) \;\in\; \mathbb{R}^3.
```

The first equation of {math:numref}`eq-payload-eom` reads
$A = \lambda\, q$ for some scalar $\lambda = q \cdot f R\, e_3 -
m_Q\, \ell\, \|\dot q\|^2$, so $A$ is parallel to $q$. In the
hanging/taut regime the thrust $f R\, e_3$ has a positive component
along $(-q)$ — it pulls the quadrotor away from the payload — so
$q \cdot f R\, e_3 < 0$ and hence $\lambda < 0$. Combined with
$\|q\| = 1$, this fixes both magnitude and sign:

```{math}
:label: eq-payload-q-from-A
q \;=\; -\,\frac{A}{\|A\|}, \qquad \lambda \;=\; -\,\|A\|.
```

**Newton–Euler-with-tension form.** Equivalently, isolate the payload
and write Newton's law with the cable reaction $-T q$ explicit
(following {footcite}`sreenath2013trajectory`, eq. 24):

```{math}
:label: eq-payload-tension
m_L\, \ddot x_L \;=\; -\,T\, q \,-\, m_L\, g\, e_3
\quad\Longleftrightarrow\quad
T\, q \;=\; -\, m_L\bigl(\ddot x_L + g\, e_3\bigr),
```

where $T \in \mathbb{R}_{\geq 0}$ is the scalar cable tension. It is
convenient to package the cable reaction into a single vector, the
**tension vector**

```{math}
:label: eq-payload-Tvec
\mathbf{T} \;:=\; T\, q \;=\; -\, m_L\bigl(\ddot x_L + g\, e_3\bigr)
            \;\in\; \mathbb{R}^3.
```

Taking norms (with $\|q\| = 1$) and dividing yields the scalar
tension and unit cable direction in closed form:

```{math}
:label: eq-payload-T-q
T \;=\; \|\mathbf{T}\|
 \;=\; m_L\,\|\ddot x_L + g\, e_3\|,
\qquad
q \;=\; \frac{\mathbf{T}}{T}.
```

Both routes give the *same* $q$: the mass prefactor — $(m_Q + m_L)$
in $A$ from the Lagrangian form, $m_L$ in $\mathbf{T}$ from the
Newton–Euler form — cancels in the unit-vector normalisation. The
scalars differ ($\lambda$ is the constrained reaction along $q$ from
the combined-system equation; $T$ is the physical cable tension), and
they are related by $\lambda = -(m_Q + m_L)/m_L \cdot T$. Both $T$
and $q$ are algebraic functions of $\ddot x_L$ alone.

:::{admonition} Which form to compute
:class: tip

For implementation, prefer the Newton–Euler form
{math:numref}`eq-payload-Tvec`. It depends only on the payload mass
$m_L$ and gives the physical tension $T$ as a free byproduct — useful
both as a slack-cable diagnostic ($T \to 0$ marks the hybrid-mode
boundary at which the cable goes slack) and as a sanity-check when
comparing to the simulation. The Lagrangian form requires the combined
mass $(m_Q + m_L)$ for the same answer and yields $\lambda$, which has
no direct physical meaning and would have to be back-computed from
$f$, $R$, $\dot q$ if needed. The derivation chain below uses
$\mathbf{T}$ accordingly.
:::

Differentiating $\mathbf{T} = T q$ gives, with
$\dot T = (\mathbf{T} \cdot \dot{\mathbf{T}})/T = \dot{\mathbf{T}} \cdot q$,

```{math}
:label: eq-payload-qdot
\dot q
 \;=\; \frac{\dot{\mathbf{T}}}{T}
       \,-\, \frac{\dot T}{T}\, q
 \;=\; \frac{1}{T}\Bigl(\dot{\mathbf{T}}
       \,-\, (\dot{\mathbf{T}} \cdot q)\, q\Bigr).
```

This is the projection of $\dot{\mathbf{T}}/T$ onto the tangent plane
$T_q S^2$, automatically perpendicular to $q$ because $\|q\| = 1$
forces $q \cdot \dot q = 0$ (see the $S^2$ tangent-space discussion
in {doc}`../preliminaries`). Since
from {math:numref}`eq-payload-Tvec`,
$\dot{\mathbf{T}} = -\, m_L\, x_L^{(3)}$, so computing $\dot q$ requires
the payload jerk.

Differentiating $T q = \mathbf{T}$ once more (Leibniz, $n = 2$),

$$
\ddot T\, q \,+\, 2\,\dot T\, \dot q \,+\, T\, \ddot q
 \;=\; \ddot{\mathbf{T}},
$$

and solving for $\ddot q$,

$$
\ddot q
 \;=\; \frac{1}{T}\Bigl(\ddot{\mathbf{T}}
        \,-\, \ddot T\, q
        \,-\, 2\,\dot T\, \dot q\Bigr).
$$

Substituting the tension derivatives
$\dot T = \dot{\mathbf{T}} \cdot q$ and
$\ddot T = \ddot{\mathbf{T}} \cdot q + \dot{\mathbf{T}} \cdot \dot q$
(both consequences of $T = \mathbf{T} \cdot q$, see also
{math:numref}`eq-payload-T-derivs`),

$$
\ddot q
 \;=\; \frac{1}{T}\Bigl(\ddot{\mathbf{T}}
        \,-\, (\ddot{\mathbf{T}} \cdot q)\, q
        \,-\, (\dot{\mathbf{T}} \cdot \dot q)\, q
        \,-\, 2\,(\dot{\mathbf{T}} \cdot q)\, \dot q\Bigr).
$$

The middle term collapses into the manifest tangency form. From
{math:numref}`eq-payload-qdot`, $T\, \dot q = \dot{\mathbf{T}} -
(\dot{\mathbf{T}} \cdot q)\, q$. Dotting with $\dot q$ and using
$q \cdot \dot q = 0$ gives
$T\, \|\dot q\|^2 = \dot{\mathbf{T}} \cdot \dot q$, so
$(\dot{\mathbf{T}} \cdot \dot q)/T = \|\dot q\|^2$. Substituting:

```{math}
:label: eq-payload-qddot
\ddot q
 \;=\; \frac{1}{T}\Bigl(\ddot{\mathbf{T}}
        \,-\, (\ddot{\mathbf{T}} \cdot q)\, q
        \,-\, 2\,(\dot{\mathbf{T}} \cdot q)\, \dot q\Bigr)
       \,-\, \|\dot q\|^2\, q,
```

where the trailing $-\|\dot q\|^2\, q$ packages the
$(\dot{\mathbf{T}} \cdot \dot q)/T$ residue and enforces the
second-order tangency condition $q \cdot \ddot q = -\|\dot q\|^2$
obtained by differentiating $q \cdot \dot q = 0$.

The higher derivatives are cleanest via the general Leibniz rule
$\,(fg)^{(n)} = \sum_{k=0}^{n} \binom{n}{k}\, f^{(k)} g^{(n-k)}\,$
{footcite}`apostol1967calculus`. Applied to $T\, q = \mathbf{T}$ and
solving for $q^{(n)}$,

```{math}
:label: eq-payload-q-leibniz
T\, q^{(n)}
 \;=\; \mathbf{T}^{(n)}
       \,-\, \sum_{k=1}^{n} \binom{n}{k}\, T^{(k)}\, q^{(n-k)},
```

which solves for $q^{(n)}$ once the lower-order $q^{(0)}, \ldots,
q^{(n-1)}$ and the tension derivatives are known. The tension
derivatives are obtained by differentiating the definition
$\mathbf{T} = T q$ and projecting onto $q$ at each order — equivalent
to differentiating $T = \mathbf{T} \cdot q$ (the projection
$\mathbf{T} \cdot q = T q \cdot q = T$, using $\|q\| = 1$). The
sub-leading $\mathbf{T} \cdot q^{(k)}$ pieces collapse via the tangency
identities $q \cdot \dot q = 0$, $q \cdot \ddot q = -\|\dot q\|^2$,
etc., leaving:

```{math}
:label: eq-payload-T-derivs
\begin{aligned}
\dot T     &\;=\; \dot{\mathbf{T}} \cdot q, \\[2pt]
\ddot T    &\;=\; \ddot{\mathbf{T}} \cdot q \,+\, \dot{\mathbf{T}} \cdot \dot q, \\[2pt]
T^{(3)}    &\;=\; \mathbf{T}^{(3)} \cdot q \,+\, 2\,\ddot{\mathbf{T}} \cdot \dot q
                   \,+\, \dot{\mathbf{T}} \cdot \ddot q, \\[2pt]
T^{(4)}    &\;=\; \mathbf{T}^{(4)} \cdot q \,+\, 3\,\mathbf{T}^{(3)} \cdot \dot q
                   \,+\, 3\,\ddot{\mathbf{T}} \cdot \ddot q
                   \,+\, \dot{\mathbf{T}} \cdot q^{(3)}.
\end{aligned}
```

Substituting $n = 3$ and $n = 4$ into {math:numref}`eq-payload-q-leibniz`,

```{math}
:label: eq-payload-q3
q^{(3)}
 \;=\; \frac{1}{T}\Bigl(
        \mathbf{T}^{(3)}
        \,-\, 3\,\dot T\, \ddot q
        \,-\, 3\,\ddot T\, \dot q
        \,-\, T^{(3)}\, q
       \Bigr),
```

```{math}
:label: eq-payload-q4
q^{(4)}
 \;=\; \frac{1}{T}\Bigl(
        \mathbf{T}^{(4)}
        \,-\, 4\,\dot T\, q^{(3)}
        \,-\, 6\,\ddot T\, \ddot q
        \,-\, 4\,T^{(3)}\, \dot q
        \,-\, T^{(4)}\, q
       \Bigr).
```

The two earlier expressions {math:numref}`eq-payload-qdot` and
{math:numref}`eq-payload-qddot` are the $n = 1, 2$ special cases of
this recursion, with the tangency conditions on $T_q S^2$ written
explicitly. Tracking the input order:
$\dot{\mathbf{T}} = -m_L\, x_L^{(3)}$, so
$q^{(3)}$ requires $x_L^{(5)}$ (crackle) and $q^{(4)}$ requires
$x_L^{(6)}$ (pop) — the deepest payload-jet rung in the recovery.

### Step 2 — quadrotor position and its derivatives

Apply $x_Q = x_L - \ell\, q$ from {math:numref}`eq-payload-cable-geom`
and differentiate term-by-term:

```{math}
:label: eq-payload-xQ-derivs
\begin{aligned}
x_Q       &\;=\; x_L \,-\, \ell\, q, \\
\dot x_Q  &\;=\; \dot x_L \,-\, \ell\, \dot q, \\
\ddot x_Q &\;=\; \ddot x_L \,-\, \ell\, \ddot q, \\
x_Q^{(3)} &\;=\; x_L^{(3)} \,-\, \ell\, q^{(3)}, \\
x_Q^{(4)} &\;=\; x_L^{(4)} \,-\, \ell\, q^{(4)}.
\end{aligned}
```

The differentiation order combines additively with Step 1: each
$x_Q^{(k)}$ requires the payload jet through $x_L^{(k)}$ for the
direct term *and* through $x_L^{(k+2)}$ for the cable term. The
worst-case rung is $x_Q^{(4)}$ (quadrotor snap), which needs $q^{(4)}$
and therefore $x_L^{(6)}$ — the source of the sixth-order requirement
in the flat output.

### Step 3 — cable angular velocity and acceleration

The cable kinematics $\dot q = \omega \times q$ inverts on $T_q S^2$
by the same BAC-CAB move used for {math:numref}`eq-eomega-s2` in
{doc}`../preliminaries`:

```{math}
:label: eq-payload-omega
\omega \;=\; q \times \dot q \;\in\; T_q S^2.
```

Indeed,
$q \times \dot q = q \times (\omega \times q) =
\omega\,(q \cdot q) - q\,(q \cdot \omega) = \omega$, using
$\|q\| = 1$ and the convention $\omega \cdot q = 0$. Differentiating
{math:numref}`eq-payload-omega` once,

```{math}
:label: eq-payload-domega
\dot\omega \;=\; q \times \ddot q,
```

since $\dot q \times \dot q = 0$. Recovering $\omega$ therefore needs
$\dot q$ — i.e. $x_L^{(3)}$ — and recovering $\dot\omega$ needs
$\ddot q$ — i.e. $x_L^{(4)}$.

The cable angular dynamics in the second line of
{math:numref}`eq-payload-eom` are *not* needed to invert $\omega$:
they serve as a consistency relation between $\dot\omega$ and the
moment-arm of the thrust vector across the payload, which is
automatically satisfied by the construction in Step 4.

### Step 4 — thrust vector

Newton's law for the quadrotor in the world frame is

$$
m_Q\, \ddot x_Q \;=\; f R\, e_3 \,-\, m_Q\, g\, e_3 \,+\, \mathbf{T},
$$

where $\mathbf{T}$ is the cable reaction acting on the quadrotor
(Newton's third law: the payload pulls the quadrotor along $+q$ via
the cable, so the force on the quadrotor is $+T q = +\mathbf{T}$;
on the payload it is $-\mathbf{T}$, as in {math:numref}`eq-payload-tension`).
Solving for the thrust vector,

```{math}
:label: eq-payload-B
B \;:=\; f R\, e_3
   \;=\; m_Q\bigl(\ddot x_Q + g\, e_3\bigr) \,-\, \mathbf{T}.
```

Both terms on the right-hand side are already in hand: $\ddot x_Q$
from Step 2, $\mathbf{T}$ from Step 1. Equivalently, eliminating
$\mathbf{T}$ via $\mathbf{T} = -m_L(\ddot x_L + g\, e_3)$ and
$\ddot x_Q = \ddot x_L - \ell\, \ddot q$ produces the two combined-system
forms

```{math}
:label: eq-payload-thrust-vector
B \;=\; m_Q\, \ddot x_Q \,+\, m_L\, \ddot x_L \,+\, (m_Q + m_L)\, g\, e_3
 \;=\; (m_Q + m_L)\bigl(\ddot x_L + g\, e_3\bigr)
   \,-\, m_Q\, \ell\, \ddot q,
```

obtained by adding the quadrotor and payload Newton's laws (the
$\pm \mathbf{T}$ pieces cancel). The thrust magnitude and direction
are then

```{math}
:label: eq-payload-f-b3
f \;=\; \|B\|, \qquad b_3 \;=\; R\, e_3 \;=\; \frac{B}{\|B\|}.
```

Compared to the standalone-quadrotor flatness map of {doc}`quadrotor`,
$B$ replaces that page's $A = m_Q(\ddot x_Q + g\, e_3)$ — same role
(the thrust vector), corrected by the cable reaction $-\mathbf{T}$.
Once $B$ is computed, every downstream quantity is identical.

### Step 5 — attitude, body rate, moment (via the quadrotor recovery)

With $(B,\, \psi)$ in hand, the remaining quantities $(R,\, \Omega,\,
\dot\Omega,\, M)$ are recovered by the **same** machinery as the
standalone-quadrotor flatness map of {doc}`quadrotor`, applied with
$B$ in place of that page's $A$. The algebra is not repeated here;
the relevant pieces are:

- **Attitude $R$** from $b_3$ and the yaw-heading hint
  $b_{1,d}(\psi) = [\cos\psi,\, \sin\psi,\, 0]^\top$ — see
  {math:numref}`eq-quad-b2-b1`–{math:numref}`eq-quad-R` in
  {doc}`quadrotor`. The two derivatives $\dot R$, $\ddot R$ follow the
  recursion of {math:numref}`eq-quad-dR`–{math:numref}`eq-quad-cross-derivs`,
  which feed off $\dot b_3$, $\ddot b_3$ via the projector
  {math:numref}`eq-quad-db3`–{math:numref}`eq-quad-d2b3`.
- **Body rate $\Omega$ and angular acceleration $\dot\Omega$** via the
  skew-projector form {math:numref}`eq-quad-Omega`–{math:numref}`eq-quad-dOmega`
  of {doc}`quadrotor`. The projector form is preferred over the
  cancellation form for the numerical-robustness reasons unpacked
  there.
- **Moment $M$** from the body-frame Newton–Euler equation,

  ```{math}
  :label: eq-payload-M
  M \;=\; J\, \dot\Omega \,+\, \Omega \times J\, \Omega,
  ```

  identical to {math:numref}`eq-quad-M` of {doc}`quadrotor` (the
  rotational dynamics are unchanged by the cable, since the cable
  applies a pure force at the body-fixed attachment point and exerts
  no moment about the centre of mass under the point-mass payload
  assumption).

Tracking the derivative chain: $B$ depends on $\ddot x_Q$ (and hence
$x_L^{(4)}$ via Step 2), so $\dot B$ needs $x_L^{(5)}$ and $\ddot B$
needs $x_L^{(6)}$. Through the recovery, $\Omega$ requires $\dot b_3
\sim \dot B$ and hence $x_L^{(5)}$, and $\dot\Omega$ (and therefore
$M$) requires $\ddot b_3 \sim \ddot B$ and hence $x_L^{(6)}$ — the
deepest payload-jet rung.

:::{admonition} Singularity
:class: note

Two singularities limit the regular region of the flatness map. The
first is the *slack-cable* limit $\ddot x_L + g\, e_3 = 0$, where
$\mathbf{T}$ in {math:numref}`eq-payload-Tvec` vanishes and the cable
direction $q$ is undefined — physically, the payload is in free-fall
and the cable carries no tension ($T = 0$). The second is the
*yaw-aligned* limit where $b_3 \parallel b_{1,d}(\psi)$, identical to
the singularity of the single-quadrotor map ({doc}`quadrotor`). Both
must be excluded from the open subset of jet space on which the
recovery is smooth.
:::

## Theorem

```{prf:theorem} Flatness of the quadrotor with cable-suspended point-mass payload
:label: thm-payload-flat

The quadrotor with taut, massless-cable-suspended point-mass payload
described by {math:numref}`eq-payload-eom` is differentially flat,
with flat output $y = (x_L,\, \psi)$ as in
{math:numref}`eq-payload-flat-output`. Every state and every input is
an algebraic function of the sixth-order jet of $x_L$ and the
second-order jet of $\psi$.
```

```{prf:proof}
Dimension matching: $\dim y = 4 = \dim u$. Steps 1–7 above construct
the smooth map

$$
\Phi : J^6 \mathbb{R}^3 \times J^2 \mathbb{R}
      \;\longrightarrow\; \mathbb{R}^{16} \times \mathbb{R}^4,
\qquad
\Phi\bigl(J^6_t\, x_L,\, J^2_t\, \psi\bigr) \;=\;
\bigl((x_L, \dot x_L, q, \omega, R, \Omega),\; (f, M)\bigr),
$$

which recovers the full state and input without integrating the
dynamics: $q$ from $\ddot x_L$ via {math:numref}`eq-payload-T-q`;
$\omega$ from $x_L^{(3)}$ via {math:numref}`eq-payload-omega`;
$x_Q$ and its derivatives up to $x_Q^{(4)}$ from $x_L$ up to
$x_L^{(6)}$ via {math:numref}`eq-payload-xQ-derivs`; the thrust vector
$B = f R\, e_3$ from $\ddot x_Q$ and $\mathbf{T}$ via
{math:numref}`eq-payload-B`; and finally $(R,\, \Omega,\, \dot\Omega,\, M)$
from $(B,\, \psi)$ by the standalone-quadrotor recovery of
{doc}`quadrotor` (Steps 2–4 there) — closing with $M$ via
{math:numref}`eq-payload-M`. Smoothness of $\Phi$ holds on the open
subset $\{\,\ddot x_L + g\, e_3 \neq 0\,\} \cap \{\,b_3 \not\parallel
b_{1,d}(\psi)\,\}$ — the former excludes the slack-cable / free-fall
configuration, the latter the yaw-aligned configuration in which the
desired attitude is not uniquely defined. This is exactly the form
{math:numref}`eq-flat-to-state-on-jets` on the product jet space, so
the system is differentially flat with flat output $y$.
```

This recovery is the substrate of the trajectory generation and
aggressive-maneuver experiments in
{footcite}`sreenath2013trajectory`; it generalises to multi-lift and
flexible-cable variants in {footcite}`kotaru2017differential`.

## Derivative-order accounting

| Recovered quantity         | Needs                                          |
| -------------------------- | ---------------------------------------------- |
| $q$                        | $\ddot x_L$                                    |
| $\dot q,\ \omega$          | $x_L^{(3)}$                                    |
| $\ddot q,\ \dot\omega$     | $x_L^{(4)}$                                    |
| $q^{(3)}$                  | $x_L^{(5)}$                                    |
| $q^{(4)}$                  | $x_L^{(6)}$                                    |
| $x_Q,\ \dot x_Q,\ \ddot x_Q$ | $\ddot x_L$ (via $q$, $\dot q$, $\ddot q$)   |
| $x_Q^{(3)}$                | $x_L^{(5)}$ (via $q^{(3)}$)                    |
| $x_Q^{(4)}$ (snap)         | $x_L^{(6)}$ (via $q^{(4)}$)                    |
| $f,\ b_3$                  | $\ddot x_L$ and $\ddot x_Q$, i.e. $x_L^{(4)}$  |
| $R$                        | $\ddot x_L,\ x_L^{(4)},\ \psi$                 |
| $\Omega$                   | $x_L^{(5)},\ \dot\psi$                         |
| $\dot\Omega$               | $x_L^{(6)},\ \ddot\psi$                        |
| $M$                        | $x_L^{(6)},\ \ddot\psi$                        |

The worst case is the moment $M$, which closes the chain at
$x_L^{(6)}$ (pop) and $\ddot\psi$ — two orders deeper in the payload
position than the bare quadrotor case of {prf:ref}`thm-quad-flat`,
where snap was sufficient. The two extra orders are exactly the cost
of recovering the quadrotor snap $x_Q^{(4)}$ through the cable
constraint $x_Q = x_L - \ell\, q$.

## Jet input

The map $\Phi$ of {prf:ref}`thm-payload-flat` takes a pair of
{py:class}`udaan.utils.flatness.Jet` objects — one of order 6 for the
payload position, one of order 2 for the yaw — packaged into a
{py:class}`udaan.utils.flatness.QuadrotorCsPayloadFlats` struct:

```python
import numpy as np
from udaan.utils.flatness import (
    Jet,
    QuadrotorCsPayload,
    QuadrotorCsPayloadFlats,
)

# Payload position and its first six time derivatives at t
x_L_jet = Jet(np.stack([
    x_L,            # row 0 — payload position
    x_L_dot,        # row 1 — payload velocity
    x_L_ddot,       # row 2 — payload acceleration (gives q)
    x_L_dddot,      # row 3 — jerk     (gives q̇, ω, ẋ_Q)
    x_L_snap,       # row 4 — snap     (gives q̈, ẍ_Q, f, b3)
    x_L_crackle,    # row 5 — crackle  (gives q⁽³⁾, x_Q⁽³⁾, Ω)
    x_L_pop,        # row 6 — pop      (gives q⁽⁴⁾, x_Q⁽⁴⁾, dΩ, M)
]))
x_L_jet.order   # 6
x_L_jet.dim     # 3

# Yaw and its first two derivatives
psi_jet = Jet(np.array([psi, psi_dot, psi_ddot]))
psi_jet.order   # 2
psi_jet.dim     # 1

flats = QuadrotorCsPayloadFlats(x_L=x_L_jet, psi=psi_jet)
ref, inputs = QuadrotorCsPayload(
    mass=m_Q,
    inertia=J,
    payload_mass=m_L,
    cable_length=ell,
).recover(flats)
# ref.payload_position, ref.payload_velocity,
# ref.position, ref.velocity, ref.acceleration,
# ref.cable_attitude, ref.cable_angular_velocity,
# ref.orientation, ref.angular_velocity, ref.angular_acceleration
# inputs.thrust, inputs.moment, inputs.tension
```

The returned `ref` is a
{py:class}`udaan.utils.flatness.QuadrotorCsPayloadRefState` and
`inputs` is a {py:class}`udaan.utils.flatness.QuadrotorCsPayloadInputs`.
Truncation ({prf:ref}`def-jet-trunc`) extracts the lower-order jets
needed by intermediate stages of the recovery (for example,
`x_L_jet.truncate(4)` is enough to produce $(q,\, \dot q,\, \ddot q,\,
x_Q,\, \dot x_Q,\, \ddot x_Q,\, f,\, b_3)$ but not $(\Omega,\, M)$).

## References

```{footbibliography}
```
