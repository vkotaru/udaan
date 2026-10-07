(flatness-quadrotor-flexcable)=
# Quadrotor with a flexible cable-suspended payload

This page derives the differential-flatness property of a single quadrotor
carrying a point-mass payload through a **flexible cable**, modelled as a
serial chain of $n$ rigid links — the system of
{footcite}`kotaru2017differential`, whose dynamics are collected in
{doc}`../dynamics/quadrotor-flexcable`.

The step up from {doc}`quadrotor-cspayload` is one of *degree*, not of kind.
The flat output is the same — payload position and quadrotor yaw — and the
recovery still ends in the same thrust-vector-to-attitude machinery. What
changes is the middle: instead of one cable whose direction falls out of the
payload's acceleration, there are $n$, and each one's direction can only be
computed once the one below it has been. That recursion is the content of this
page, and it has a consequence with no analogue anywhere else in this package:

> **The number of derivatives of the flat output that the map consumes grows
> with the size of the system.** M1 needs snap, M2 needs pop, M3 and M4 need
> the sixth derivative no matter how many quadrotors are added. Here the
> requirement is $2n+4$: order 6 at one link, order 14 at five.

That is not an artefact of how the recursion is organised. It is a statement
about the system — each additional link genuinely interposes two more
differentiations between the payload trajectory and the airframe's thrust — and
{ref}`the accounting section <flexcable-order-accounting>` derives it, states
what it costs in practice, and cites the experiment that confirms it.

## System

Let $x_L \in \mathbb{R}^3$ be the payload position and $\psi \in \mathbb{R}$
the quadrotor yaw. The cable is discretised into $n$ rigid links indexed
$i = 0, \dots, n-1$ **from the airframe downwards**; link $i$ has length
$\ell_i > 0$, direction $q_i \in S^2$ pointing *from the quadrotor towards the
payload*, angular velocity $\omega_i \in T_{q_i}S^2$ with
$\dot q_i = \omega_i \times q_i$, and a point mass $m_i > 0$ lumped at its own
**distal** end. The quadrotor has mass $m_Q$, inertia $J$, attitude $R$, body
rate $\Omega$, thrust vector $f R e_3$ and body moment $M$.

Write $x_0 = x_Q$ for the airframe and $x_{i+1}$ for the distal end of link
$i$, so the chain is inextensible:

```{math}
:label: eq-m5f-geometry
x_{i+1} = x_i + \ell_i q_i , \qquad
x_n = x_L = x_Q + \sum_{i=0}^{n-1} \ell_i q_i .
```

The equations of motion, derived from the Lagrangian on
$\mathbb{R}^3 \times SO(3) \times (S^2)^n$ in
{doc}`../dynamics/quadrotor-flexcable`, are a coupled $(3+3n)$ system in
$(\ddot x_Q, \dot\omega_0, \dots, \dot\omega_{n-1})$ together with the
airframe's attitude dynamics. The flat map does not invert that linear system.
It never has to: **as a statics problem the chain is determinate**, and that is
the whole reason it is flat.

## Newton on a sub-chain

Isolate the sub-chain consisting of the masses $m_i, m_{i+1}, \dots, m_{n-1}$
— everything at or below the distal end of link $i$. The only external forces
on it are gravity and whatever link $i$ transmits through its proximal joint.
Newton's second law for that free body therefore reads

```{math}
:label: eq-m5f-subchain
\sum_{k \ge i} m_k \ddot x_{k+1}
  \;=\; W_i \;-\; g\, e_3 \sum_{k \ge i} m_k ,
```

with $W_i$ the joint force. Define the **cable-force jet**

```{math}
:label: eq-m5f-force
F_i \;:=\; -W_i \;=\; -\sum_{k \ge i} m_k \bigl(\ddot x_{k+1} + g e_3\bigr) ,
```

which by Newton's third law is also the force the sub-chain pulls the body
above it with. A link of a taut cable can only pull, so $F_i$ is anti-parallel
to the pull direction $q_i$ from the top down — with this sign convention,
simply

```{math}
:label: eq-m5f-Tq
T_i = \|F_i\| \;\ge\; 0 , \qquad q_i = \frac{F_i}{T_i} .
```

At $n = 1$, {math:numref}`eq-m5f-force` is
$F_0 = -m_L(\ddot x_L + g e_3)$ — precisely M2's cable force
({doc}`quadrotor-cspayload`, Step 1), term for term.

:::{note}
**Sign convention.** $q_i$ points *quadrotor → payload*, so a chain at rest has
$q_i = -e_3$. This matches
{py:class}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableState`,
{py:mod}`udaan.flatness.cable`, and every other cable in `udaan`.

The legacy MATLAB reference
(`Single-Quad-Flex-Cable/get_desired_traj.m`, L38–121) uses the **same**
direction. It accumulates the opposite *force*,
$B_i = -F_i = +\sum_{k\ge i} m_k(\ddot x_{k+1} + g e_3)$, and then writes
$q_i^{\text{ref}} = -B_i/\|B_i\|$ — so at rest $B_i = +m g e_3$ and
$q_i^{\text{ref}} = -e_3$, hanging downwards, exactly as here. The only
difference is where the negation lives: the reference carries it in every
direction formula, while `udaan` folds it into $F_i$ once, inside
{py:func}`udaan.flatness.cable_chain_cascade`, and normalises $F_i$ unchanged.
(Its link *indexing* does run the other way — index $1$ at the quadrotor,
index $n$ at the payload — the reverse of the top-down order used here.)
:::

## Flat output

```{math}
:label: eq-m5f-flat-output
y \;=\; (x_L,\, \psi) \;\in\; \mathbb{R}^3 \times \mathbb{R} ,
```

the same four numbers as M2. Dimension matching holds: the input is
$(f, M) \in \mathbb{R}^4$.

Adding links adds *state* ($6$ manifold dimensions each) but no inputs, and
therefore no flat outputs. The chain is entirely **underactuated**, driven only
through the single joint at the airframe — which is exactly why its
configuration is pinned by the payload trajectory alone, and why the price of
each extra link is paid in derivative order rather than in extra flat outputs.

## Constructive recovery

### Step 1 — the chain cascade

Sweep {math:numref}`eq-m5f-force` and {math:numref}`eq-m5f-geometry` from the
payload upwards. Starting at the tip, $x_n = x_L$ is given, so

```{math}
:label: eq-m5f-cascade
\begin{aligned}
F_{n-1}^{(k)} &= -m_{n-1}\bigl(x_L^{(k+2)} + g e_3\,\delta_{k0}\bigr), \\[2pt]
(T_i, q_i)^{(k)} &= \text{the order-}k\text{ jet of }
                 \bigl(\|F_i\|,\; F_i/\|F_i\|\bigr), \\[2pt]
x_i^{(k)}     &= x_{i+1}^{(k)} - \ell_i\, q_i^{(k)}, \\[2pt]
F_{i-1}^{(k)} &= F_i^{(k)} - m_{i-1}\bigl(x_i^{(k+2)} + g e_3\,\delta_{k0}\bigr),
\end{aligned}
```

for $i = n-1, n-2, \dots, 0$, ending at the airframe $x_0 = x_Q$ and the force
$F_0$ the chain pulls it with.

Two things are worth naming. First, the per-link step is *not new*: the
normalisation jet in the second line is exactly
{py:func}`udaan.flatness.tension_direction_jet`, the primitive M2, M3 and M4
already use, with its closed-form arbitrary-order recursion
({doc}`../differential-flatness/jet` and
{py:func}`udaan.flatness.normalize_derivatives`). Only the *recursion between*
the links is new. Second, nothing here is solved — it is a sweep of explicit
assignments, which is what makes the map algebraic.

The link rates follow from the direction jets in the usual way,

```{math}
:label: eq-m5f-rates
\omega_i = q_i \times \dot q_i , \qquad
\dot\omega_i = q_i \times \ddot q_i ,
```

the $\dot q_i \times \dot q_i$ term vanishing in the second.

The cascade is implemented once, as
{py:func}`udaan.flatness.cable_chain_cascade`, and returns a
{py:class}`udaan.flatness.CableChain` carrying every node position, direction,
tension and force jet.

### Step 2 — the thrust vector

Newton on the airframe alone. The chain pulls on it with $F_0$, so

```{math}
:label: eq-m5f-B
B \;:=\; f R e_3 \;=\; m_Q\bigl(\ddot x_Q + g e_3\bigr) \;-\; F_0 ,
```

with $\dot B = m_Q x_Q^{(3)} - \dot F_0$ and
$\ddot B = m_Q x_Q^{(4)} - \ddot F_0$. At $n = 1$ this is M2's
$B = m_Q(\ddot x_Q + g e_3) + m_L(\ddot x_L + g e_3)$, since
$F_0 = -m_L(\ddot x_L + g e_3)$ there.

### Step 3 — attitude, body rate, moment

Unchanged from M1 and M2: given $(B, \dot B, \ddot B)$ and the yaw chain
$(\psi, \dot\psi, \ddot\psi)$,

```{math}
f = \|B\|, \quad b_3 = B/f, \quad
b_2 = \frac{b_3 \times b_{1,d}}{\|b_3 \times b_{1,d}\|}, \quad
b_1 = b_2 \times b_3, \quad R = [\,b_1\ b_2\ b_3\,],
```

with $b_{1,d} = (\cos\psi, \sin\psi, 0)^\top$, then
$\widehat\Omega = R^\top \dot R$, $\widehat{\dot\Omega} =
\operatorname{skew}(R^\top \ddot R)$ and

```{math}
:label: eq-m5f-moment
M \;=\; J\dot\Omega + \Omega \times J\Omega .
```

This is {py:func}`udaan.flatness.quadrotor._attitude_from_thrust_vector`,
shared verbatim with every other thrust-vector map here; see
{doc}`quadrotor` Steps 2–4 for the derivation of the projector chain.

## Theorem

```{prf:theorem} Flatness of a quadrotor with an $n$-link flexible cable
:label: thm-m5-flat

Let $m_Q > 0$ and $m_i, \ell_i > 0$ for every link. On the open set where no
link force vanishes ($F_i \ne 0$ for all $i$), the thrust vector is non-zero
($B \ne 0$), and $b_3$ is not parallel to $b_{1,d}(\psi)$, the system is
differentially flat with flat output $y = (x_L, \psi)$ as in
{math:numref}`eq-m5f-flat-output`. Every state and both inputs are algebraic
functions of the $(2n+4)$-th order jet of $x_L$ and the second-order jet of
$\psi$.
```

```{prf:proof}
Dimension matching: $\dim y = 4 = \dim(f, M)$. Steps 1–3 construct the smooth
map

$$
\Phi_{\mathrm{flat}} : J^{2n+4}\mathbb{R}^3 \times J^2\mathbb{R}
\;\longrightarrow\;
\mathbb{R}^{18+6n} \times \mathbb{R}^4
$$

without integrating anything: the recursion {math:numref}`eq-m5f-cascade`
terminates in $n$ steps because it strictly descends the link index, and each
of its lines is an explicit assignment.

That the result satisfies the equations of motion is a check in two parts.
{math:numref}`eq-m5f-subchain` is the *definition* of $F_i$, so each sub-chain's
translational balance holds by construction, and differencing consecutive ones
recovers the balance of each individual lumped mass. The airframe's balance is
{math:numref}`eq-m5f-B`, again by construction, and the attitude dynamics are
{math:numref}`eq-m5f-moment`. The per-link *rotational* equations are not
independent: crossing the balance of mass $i$ with $q_i$ annihilates the
component of the joint forces along $q_i$ and leaves
$\dot\omega_i = q_i \times \ddot q_i$, which {math:numref}`eq-m5f-rates`
already supplies. Equivalently — and this is the observation the whole map
rests on — the $(3+3n)$ mass matrix of the dynamics is invertible, so the state
and inputs constructed above are the *unique* ones consistent with the given
$\ddot x_L$, and the map inverts the dynamics rather than merely satisfying a
subset of them.

Smoothness on the stated open set follows from the smoothness of
$v \mapsto (\|v\|, v/\|v\|)$ away from the origin. This is the form
{math:numref}`eq-flat-to-state-on-jets`, so the system is differentially flat.
```

### What the singular set means

$F_i \to 0$ is a real physical boundary, not a bookkeeping artefact: it is the
instant the sub-chain below link $i$ is in free fall and the link carries
nothing. For a *cable* that is the slack transition — the hybrid-mode boundary
M2 also has — and the map's domain genuinely stops there.
{py:attr}`~udaan.flatness.QuadrotorFlexCableInputs.tensions` reports every
$T_i$, so the margin to it is available at every instant.

For M5's *rigid-link* cable there is a subtlety worth stating. A rigid link
carries compression as happily as tension, so past the crossing the dynamics
remain perfectly well posed and the recovery continues with $T_i > 0$ and $q_i$
flipped — the payload ends up *above* the airframe. That branch solves the
link-chain equations but not the cable ones it stands in for, so it is outside
the physical validity of the model.

$T_i$ alone cannot police it. Being a norm it is positive again on the far side
of the crossing, so a check on $T_i$ refuses only to *land* on $F_i = 0$ — a
measure-zero event that a sampled trajectory essentially never hits. Sweeping
a dive with $\ddot z_L$ passing through $-g$ at 2001 samples produced **no
refusals at all** and returned an inverted reference at a quarter of them, at
tensions indistinguishable from the taut side's.

The cascade therefore also enforces the **hanging branch**,
$F_i \cdot (-e_3) \ge 0$: link $i$ may not rotate more than $90°$ away from
straight down. That is the chain's version of M3's
$\mathbf{T}_i \cdot \bar q_i > 0$ test ({doc}`multi-quad-cspayload`) with hover
direction $-e_3$, and like it, it is deliberately conservative — a link that
swings past horizontal without ever unloading is rejected too. Between the two,
$F_i \to 0$ is refused on the boundary and $F_i$ crossing it is refused
afterwards, both naming the link.

(flexcable-order-accounting)=
## Derivative-order accounting

This is the interesting result of the page, so it is worth deriving rather than
quoting.

**Each rung of the cascade costs two derivatives.** Read
{math:numref}`eq-m5f-cascade` from the bottom: the force in link $i$ is built
from the *acceleration* of the node below it, so $F_i$ — and hence $q_i$ — is
available two orders shallower than $x_{i+1}$. The node above then inherits
$q_i$'s depth through $x_i = x_{i+1} - \ell_i q_i$. With the payload jet
supplied at order $K$:

| Quantity                       | Order available |
| ------------------------------ | --------------- |
| $x_n = x_L$                    | $K$             |
| $F_{n-1},\ T_{n-1},\ q_{n-1}$  | $K-2$           |
| $x_{n-1}$                      | $K-2$           |
| $F_{n-2},\ q_{n-2}$            | $K-4$           |
| $\vdots$                       | $\vdots$        |
| $F_0,\ q_0$                    | $K-2n$          |
| $x_0 = x_Q$                    | $K-2n$          |

The consumer at the top fixes $K$. The thrust {math:numref}`eq-m5f-B` needs
$\ddot x_Q$; $\Omega$ needs one more derivative of $B$ and $\dot\Omega$ — hence
the moment — one more again. So $x_Q$ is needed **through its snap**, i.e.
$K - 2n = 4$:

```{math}
:label: eq-m5f-order
\boxed{\;K \;=\; 2n + 4\;}
```

and the yaw is needed through $\ddot\psi$, unchanged.

Three consequences.

**It reduces correctly.** At $n = 1$, $K = 6$ — M2's pop, exactly. The
derivative-order accounting of {doc}`quadrotor-cspayload` is the $n=1$ row of
the table above.

**It is genuinely $O(n)$ deep, and that is unusual here.** M3 and M4 add
vehicles without adding jet depth, because every cable in those systems hangs
directly off the payload and its chain bottoms out on the same payload jet.
M5's links are in *series*, so their chains are nested rather than parallel.
The same will be true of M6.

**Practically, it is a demand on the trajectory generator, not on the map.**
The map itself costs $O(nK^2) = O(n^3)$ flops — the Leibniz recursions are
quadratic in the order — which is nothing. The real cost is upstream: a
reference trajectory for a five-link cable must be $C^{14}$ and have its
fourteenth derivative available in closed form. Polynomial splines of degree
$\ge 2n+4$ (or analytic trajectories such as the sinusoids the tests use)
supply that; a spline fitted for minimum-snap does not. This is the practical
ceiling on $n$ for flatness-based planning, and it bites well before the
$O(n^3)$ dense solve in the *simulation* model does. It is an API and
mathematical constraint that holds regardless of what the numbers do: on the
*shipped light cable* the deepest rows happen to buy nothing measurable at
$n \gtrsim 4$ (next admonition), but that is a property of those particular
masses and lengths, not a licence to supply a shallower jet.

:::{admonition} Confirmed by experiment, not by assertion
:class: tip

Necessity is checked in `tests/flatness/test_quadrotor_flexcable.py`, for
$n = 1, 2, 3, 5$: perturbing row $2n+4$ of the payload jet moves the last
link's own top derivative $q_{n-1}^{(2n+2)}$ by an $O(1)$ amount, and supplying
only $2n+3$ rows raises. End to end, the open-loop round trip
(`test_quadrotor_flexcable_feedforward.py`) closes to integrator round-off at
exactly this order — see below.

**Sufficiency is analytic, and the tests do not add to it.** The depth table
above is the argument. The companion "poison every row above $2n+4$ and the
output is bit-identical" test is a *truncation regression guard*, not
evidence: both `cable_chain_cascade` and `QuadrotorFlexCable.chain` slice the
jet to $K+1$ rows before doing any arithmetic, so surplus rows cannot move the
answer whatever the recursion then does. What it catches is the truncation
drifting — a mutant keeping one row fewer dies there — and it is labelled that
way in the test file.
:::

:::{admonition} The deepest row is real at every $n$, but not always visible
:class: tip

The *map-level* sensitivity to row $2n+4$ is attenuated by roughly
$m_i \ell_i / T_i$ per rung, and on the shipped defaults it disappears into
round-off: a unit bump of the deepest row moves the feedforward moment by
$2.6\times10^{-5}$, $1.4\times10^{-10}$, $2.9\times10^{-17}$, and *exactly zero*
at $n = 1, 3, 5, 7$.

**That is a statement about $m_i\ell_i/T_i$, not about $n$.** The default chain
divides a *fixed* $0.1$ kg / $1.0$ m cable into $n$ links, so $m_i \ell_i$ falls
as $1/n^2$ while $T_i$ barely moves — the attenuation is manufactured by the
refinement, not by the link count. Hold the links fixed instead (a heavy hose,
$m_i = 1$ kg, $\ell_i = 1$ m under a $20$ kg airframe) and the same bump moves
the moment by $3.4\times10^{-5}$, $5.4\times10^{-8}$, $2.6\times10^{-11}$ and
$6.0\times10^{-15}$ at $n = 1, 3, 5, 7$ — plainly observable throughout, and
scaling exactly linearly with the bump, which is what separates it from
round-off. `test_the_2n_plus_4_row_reaches_the_moment_on_a_heavy_chain` runs
that parameterisation at $n = 5$ and $7$.

So $K = 2n+4$ is analytically necessary, correctly derived and correctly
enforced: it is **not** over-specified, and the effective depth is not lower.
It is merely numerically unobservable for $n \gtrsim 4$ on the light default
cable — which is also why the per-rung necessity test measures at the bottom
link, where the dependence is $O(1)$ at every $n$.
:::

## Validation

The load-bearing check is the **open-loop round trip**: the flat-derived
$(f, M)$ is fed into the simulation model with no controller, and the resulting
trajectory is compared against the flat outputs it was built from. Integrated
with DOP853 at $\texttt{rtol} = \texttt{atol} = 10^{-12}$ over 5 s, the final
residuals are

| $n$ | jet order | hover | dynamic (Lissajous + yaw) |
| --- | --------- | ----- | ------------------------- |
| 1   | 6         | $0$   | $\lesssim 1.1\times10^{-12}$ |
| 2   | 8         | $0$   | $\lesssim 1.6\times10^{-11}$ |
| 3   | 10        | $0$   | $\lesssim 1.4\times10^{-11}$ |
| 5   | 14        | $2\times10^{-14}$ | $\lesssim 1.4\times10^{-11}$ |
| 7   | 18        | $2\times10^{-14}$ | $\lesssim 1.3\times10^{-11}$ |

(worst component over payload position/velocity, quadrotor position, every
link's $q_i$ and $\omega_i$, attitude and body rate; the $10^{-11}$ figures are
all the link angular velocities, everything else being $10^{-13}$ or below).
Heterogeneous link masses *and* lengths are covered at $n = 2, 4$.

Two notes on the integrator choice. The shipped ZOH-Euler loop is also
exercised, but only for $n \le 3$: Step 9 established that this system is
*conditionally stable* under ZOH-Euler, with
{py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.stability_growth_rate`
crossing 1 at $n = 5$ on the default chain. Inside that regime the ZOH residual
is $\sim3\times10^{-3}$ m after 3 s and **falls linearly with the step size**,
which is what identifies it as integrator error rather than a formula defect.
An adaptive RK has no such limit, so the larger $n$ are covered there — a
choice about the *integrator*, not about the map.

Independently of any integration, the recovered state and wrench are pushed
back through the model's own equations of motion and required to reproduce
$\ddot x_L$ and every $\dot\omega_i$ to $10^{-10}$. That check touches no other
part of `udaan.flatness`, so it is evidence independent of the primitives the
map is built from — but only for the cascade's statics and its kinematics
through second derivatives. It says **nothing about the moment**: the map sets
$M = J\dot\Omega + \Omega\times J\Omega$ and the model's `accelerations`
computes $\dot\Omega = J^{-1}(M - \Omega\times J\Omega)$, so feeding $M$ back in
returns the $\dot\Omega$ it was built from, by construction. Mutants of the
$(B,\dot B,\ddot B) \to (\Omega,\dot\Omega,M)$ half survive it untouched.

The attitude half is covered separately, by finite differences. The recovered
$R(t)$ is a function of $B(t)$ and $\psi(t)$ alone — no derivative of the thrust
vector enters it — so re-evaluating the map at neighbouring times and
differentiating $R$ numerically gives $\Omega = (R^\top \dot R)^\vee$ and
$\dot\Omega$ by a route that never touches the analytic $(\dot B, \ddot B)$
chain the moment comes from. (The second is just the product rule on
$\widehat\Omega = R^\top\dot R$: $R^\top\ddot R = \dot{\widehat\Omega} +
\widehat\Omega^2$, and $\widehat\Omega^2$ is symmetric, so the skew part is
$\dot{\widehat\Omega}$.) Seeded with that numerically differentiated body rate,
the model's Euler equation must return the numerically differentiated
$\dot\Omega$ when driven by the recovered $M$. Both mutants named above die
there.

The $n = 1$ reduction to M2 agrees to bit-exactness, but M5 and M2 *share*
{py:func}`~udaan.flatness.tension_direction_jet` and the attitude primitive —
so that comparison validates the cascade's assembly (sign, gravity, truncation,
ordering) against a long-verified assembly of the same pieces, and is not an
independent derivation.

## Jet input

The map takes one payload {py:class}`udaan.flatness.Jet` of order $2n+4$ and
one order-2 scalar yaw jet, packaged in a
{py:class}`udaan.flatness.QuadrotorFlexCableFlats`:

```python
import numpy as np
from udaan.flatness import Jet, QuadrotorFlexCable, QuadrotorFlexCableFlats

n = 3
f2s = QuadrotorFlexCable(num_links=n)  # or .from_model(model)
f2s.jet_order  # -> 10  == 2n + 4

# Payload position and its first 2n+4 time derivatives at t
x_L_jet = Jet(
    np.stack(
        [
            x_L,  # row  0 — payload position
            x_L_d1,  # row  1 — velocity
            x_L_d2,  # row  2 — acceleration   (gives F_{n-1}, T_{n-1}, q_{n-1})
            x_L_d3,  # row  3                  (gives q̇_{n-1}, ω_{n-1})
            x_L_d4,  # row  4                  (gives q̈_{n-1}, ω̇_{n-1}, and F_{n-2})
            # ...                                     two more rows per link up the chain
            x_L_d10,  # row 2n+4 — the deepest row; reaches q_0⁽⁴⁾, hence Ω̇ and M
        ]
    )
)
psi_jet = Jet(np.array([psi, psi_dot, psi_ddot]))

ref, inputs = f2s.recover(QuadrotorFlexCableFlats(x_L=x_L_jet, psi=psi_jet))

# ref.payload_position, ref.payload_velocity, ref.payload_acceleration
# ref.position, ref.velocity, ref.acceleration          -> the quadrotor
# ref.link_attitudes[i], ref.link_angular_velocities[i],
# ref.link_angular_accelerations[i]                      -> per link, quad → payload
# ref.link_positions, ref.link_velocities                -> (n, 3) lumped-mass nodes
# ref.orientation, ref.angular_velocity, ref.angular_acceleration
# inputs.thrust, inputs.moment, inputs.tensions          -> tensions[i] is the slack margin
```

`ref` is a {py:class}`udaan.flatness.QuadrotorFlexCableRefState` (its
`as_state()` projects onto the simulation model's state struct, ready for
`model.reset(...)`) and `inputs` is a
{py:class}`udaan.flatness.QuadrotorFlexCableInputs`, whose
`[thrust, *moment]` is directly a wrench for `model.step(...)`.
{py:meth}`~udaan.flatness.QuadrotorFlexCable.from_model` copies the link masses,
lengths, airframe mass and inertia off a
{py:class}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`; built
by hand, the constructor mirrors the model's `cable_mass` / `payload_mass` /
`cable_length` convenience so the two always agree.

The chain half is usable on its own, as
{py:func}`udaan.flatness.cable_chain_cascade`, for anything that needs the
tension and geometry of a hanging chain without the vehicle at the top. An
external force enters through the primitive's `node_forces` argument, and two
facts about that hook are part of its contract:

* **Nothing can be applied at node $0$.** `node_forces` has one entry per
  *link*, and entry $i$ lands at link $i$'s distal end — nodes $1 \dots n$.
  Node $0$ is the top attachment, whose force balance belongs to the body
  holding the chain up, not to the chain.
* **The caller owns the jet depth.** An external-force jet truncates everything
  above it, so a short one silently costs derivative depth at the top. The
  cascade rejects anything shallower than link $i$'s own force order,
  $K - 2(n-i)$ — which is exactly what a vehicle at node $i+1$ gets for free,
  since that node's position comes out at $K - 2(n-1-i)$ and its thrust vector
  costs the usual two.

{doc}`multi-quad-hose` is the primitive's second consumer, and it needed two
things this page did not anticipate. Its vehicles are *interior* to the chain,
so their thrusts are the unknowns rather than the data — what a hose knows is
the **link force**, and `link_forces` is the dual hook that prescribes it and
reports the applied force that balances the rung. And a hose is held up at both
ends, so its far links point upwards on purpose and the hanging-branch test
above has to be turned off (`require_hanging_branch=False`). Everything else —
the depth accounting, the per-rung normalisation, the position propagation, the
slack refusal — was reused unchanged, with M6 calling the cascade once on the
index-reversed chain.

## References

```{footbibliography}
```
