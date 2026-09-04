(flatness-multi-quad-hose)=
# Multiple quadrotors carrying a flexible hose

This page derives the differential-flatness property of $n_Q$ quadrotors bolted
at chosen vertices of **one** flexible hose, discretised into $n$ rigid links —
the system of {footcite}`kotaru2020hose`, whose dynamics are collected in
{doc}`../dynamics/multi-quad-hose`.

It is the last system in the package and the only one whose flat output is not
built from positions alone.

> **The free coordinates are the *tensions*, not the vehicle positions.** The
> flat output is the root vertex's position, one **vector tension** for each
> vehicle except the last, and a yaw per vehicle. And unlike every other map
> here, flatness is **conditional on the mechanism**: a hose with no vehicle at
> its last vertex is not flat in these coordinates at all.

Both facts have the same source, and it is worth naming it before the algebra:
prescribing a trajectory closes *one* vertex's force balance, and a hose has
$n+1$ vertices in series rather than a single load body.

## System

Vertices are indexed $j = 0, \dots, n$ and link $i$ runs from vertex $i$ to
vertex $i+1$, so $q_i \in S^2$ points **down the chain** and the hose is
inextensible:

```{math}
:label: eq-m6f-geometry
x_{j+1} \;=\; x_j + \ell_j q_j ,
\qquad
\dot q_i = \omega_i \times q_i .
```

Vertex $j$ carries a lumped mass $\bar m_j > 0$ — the hose's own share plus an
airframe's, where there is one. The attachment set
$\mathcal{A} \subset \{0, \dots, n\}$ is strictly increasing and vehicle $k$
sits at vertex $a_k$; write $u_j = f_j R_j e_3$ for the thrust vector at vertex
$j$, zero where $j \notin \mathcal{A}$. The configuration manifold is
$\mathbb{R}^3 \times (S^2)^n \times SO(3)^{n_Q}$ and the input is $4 n_Q$
dimensional: a collective thrust and a body moment per vehicle.

Write $\boldsymbol\tau_i$ for the force link $i$ exerts on its **proximal**
vertex $i$. A massless rigid link transmits equal and opposite forces at its
ends, so link $i$ exerts $-\boldsymbol\tau_i$ on vertex $i+1$, and a taut link
pulls along itself:

```{math}
:label: eq-m6f-direction
q_i \;=\; \frac{\boldsymbol\tau_i}{\|\boldsymbol\tau_i\|} ,
\qquad
T_i \;=\; \|\boldsymbol\tau_i\| \;\ge\; 0 .
```

Vertex $j$'s translational balance is then

```{math}
:label: eq-m6f-vertex
\bar m_j\bigl(\ddot x_j + g e_3\bigr)
\;=\; u_j \;-\; \boldsymbol\tau_{j-1} \;+\; \boldsymbol\tau_j ,
\qquad
\boldsymbol\tau_{-1} := 0, \quad \boldsymbol\tau_n := 0 ,
```

with the two boundary conventions saying only that nothing hangs above vertex
$0$ or below vertex $n$.

## Why the flat output cannot be positions alone

Count the unknowns {math:numref}`eq-m6f-vertex` has to resolve. Suppose the
root trajectory $x_0(\cdot)$ is given. Vertex $0$'s balance then contains two
unknowns, $u_0$ and $\boldsymbol\tau_0$, and one equation. If there is no
vehicle at vertex $0$ then $u_0 = 0$ and $\boldsymbol\tau_0$ falls out; the
geometry {math:numref}`eq-m6f-geometry` gives $x_1$, and the sweep continues.
**A vertex without a vehicle is determinate.** A vertex *with* one is not: the
thrust and the onward link force are three equations short between them.

That shortfall is exactly the flat-output budget. Dimension matching demands
$\dim y = \dim u = 4 n_Q$; the root position and the yaws account for
$3 + n_Q$, leaving

$$
4 n_Q - 3 - n_Q \;=\; 3\,(n_Q - 1)
$$

free components — one 3-vector per vehicle **except one**. Handing over
$\boldsymbol\tau_{a_k}$ at each such vehicle closes every balance, and the
recursion becomes a definition at every step:

| Vertex $j$ | {math:numref}`eq-m6f-vertex` gives |
| --- | --- |
| no vehicle, $j < n$ | $\boldsymbol\tau_j = \bar m_j(\ddot x_j + g e_3) + \boldsymbol\tau_{j-1}$ |
| vehicle, $j < n$ ($\boldsymbol\tau_j$ flat) | $u_j = \bar m_j(\ddot x_j + g e_3) + \boldsymbol\tau_{j-1} - \boldsymbol\tau_j$ |
| vehicle at $j = n$ | $u_n = \bar m_n(\ddot x_n + g e_3) + \boldsymbol\tau_{n-1}$ |
| **no vehicle at $j = n$** | **nothing — the system is not flat** |

This is the same move {doc}`multi-quad-cspayload` makes with its internal cable
forces and {doc}`multi-quad-rigidbody` with the grasp null space. What is
different is *which* quantity is free. There the load's own balance pins the
sum and the split is left over; here the load is a chain, and what is left over
is the force in each link a vehicle hands its load to.

:::{admonition} Tensions are the better coordinates anyway
:class: note

The choice is forced by the counting, but it is also the one a planner wants.
It is $T_i > 0$, not any vehicle position, that has to hold for a hose to be a
hose — a hose cannot push. Working in these coordinates makes tautness a
*constraint on the flat output*, enforceable by construction, instead of
something to be checked after the fact.

The price is that the vehicle positions are no longer prescribed: they come out
of {math:numref}`eq-m6f-geometry`, and steering them to chosen points is an
inverse problem. The reference solves it numerically
(`optimize_for_des_state`); it is a planner's job, not this map's.
:::

## The last vertex must carry a vehicle

Read the last row of the table again. Vertex $n$ has no link below it, so
$\boldsymbol\tau_n = 0$ is not available as a free choice — it is fixed by the
mechanism. With a vehicle there, $u_n$ absorbs the balance and the sweep
terminates. Without one, $u_n = 0$ as well and
{math:numref}`eq-m6f-vertex` becomes

$$
\bar m_n \bigl(\ddot x_n + g e_3\bigr) \;+\; \boldsymbol\tau_{n-1} \;=\; 0 ,
$$

three *constraints* the flat output must already satisfy rather than three
quantities it determines. A generic trajectory violates them, so the map is not
defined on an open set of flat outputs, and the system is not differentially
flat in these coordinates.

{py:class}`~udaan.flatness.MultiQuadHose` refuses to be constructed on such an
attachment pattern, raising {py:class}`~udaan.flatness.NotFlatError` — a
distinct exception type, because it is a statement about the *mechanism* rather
than about a trajectory, and no replanning fixes it. The legacy reference has
the same guard (`flat2state.m`, line 107,
`error('Flatness Error: This system is not differentially-flat')`), and every
one of its example scripts writes `I = [I, n+1]` with the comment
*"(n+1 is always part of I: differential flatness)"*.

:::{admonition} The asymmetry is in the coordinates, not the physics
:class: note

Nothing distinguishes the two ends of a hose physically. What breaks the
symmetry is the decision to drive the sweep from vertex $0$: a vehicle is then
required at the *far* end. Driving it from vertex $n$ instead would give an
equally valid map requiring a vehicle at vertex $0$, with the flat tension of
each vehicle being the link *above* it. The requirement is "a vehicle at the
end the sweep terminates at", and this package — following the reference, and
matching the simulation model, which integrates vertex $0$ as its root — puts
that end at $n$.
:::

## Flat output

```{math}
:label: eq-m6f-flat-output
y \;=\;
\bigl(\,
  x_0 ,\;
  \{\boldsymbol\tau_{a_k}\}_{a_k \ne n} ,\;
  \{\psi_k\}_{k=1}^{n_Q}
\,\bigr)
\;\in\; \mathbb{R}^{3} \times \mathbb{R}^{3(n_Q-1)} \times \mathbb{R}^{n_Q} ,
\qquad
\dim y = 4 n_Q .
```

## Constructive recovery

### Step 1 — the sweep

Run the table above from $j = 0$ to $j = n$, interleaving
{math:numref}`eq-m6f-geometry` and {math:numref}`eq-m6f-direction`. Nothing is
integrated and nothing is solved: each line is an explicit assignment, and the
sweep terminates in $n+1$ steps because it strictly ascends the vertex index.
Link rates follow from the direction jet as they do in every cable map here,

```{math}
:label: eq-m6f-rates
\omega_i \;=\; q_i \times \dot q_i ,
\qquad
\dot\omega_i \;=\; q_i \times \ddot q_i .
```

### Step 2 — attitude, body rate, moment

Each vehicle now has its thrust vector $u_j$ with two derivatives, which is
exactly the input of the shared primitive
{py:func}`udaan.flatness.quadrotor._attitude_from_thrust_vector`:
$f_k = \|u_{a_k}\|$, $b_3 = u_{a_k}/f_k$, the yaw hint
$b_{1,d} = (\cos\psi_k, \sin\psi_k, 0)$ completes the frame, $\hat\Omega_k =
R_k^\top \dot R_k$, and

```{math}
:label: eq-m6f-moment
M_k \;=\; J_k \dot\Omega_k + \Omega_k \times J_k \Omega_k .
```

See {doc}`quadrotor` Steps 2–4 for the projector chain.

### It is the M5 cascade, run backwards

{py:func}`udaan.flatness.cable_chain_cascade` — the primitive
{doc}`quadrotor-flexcable` introduced — *is* this recursion. Its $F_i$ is the
force link $i$ exerts on the body above it, its $q_i = F_i/\|F_i\|$ points
top→tip, and its rung

$$
F_i \;=\; -m_i(\ddot x_{i+1} + g e_3) \;+\; u_i \;+\; F_{i+1}
$$

is {math:numref}`eq-m6f-vertex` rearranged. The one difference is which end
drives it: the cascade sweeps tip→top from a prescribed **tip**, and M6
prescribes its **root**. So {py:class}`~udaan.flatness.MultiQuadHose` reverses
the chain — cascade node $c$ is hose vertex $n-c$ — and calls the primitive
once. Under that relabelling

* the hose's root $x_0$ is the cascade's tip, which is why the jet-order law
  {py:func}`~udaan.flatness.chain_jet_order` constrains $x_0$;
* the hose's **last vertex** is the cascade's node $0$ — the body "holding the
  chain up", whose thrust the primitive already documents as
  $u = m(\ddot x_0 + g e_3) - F_0$. **The not-flat guard is the cascade's own
  contract**: node $0$ is where the chain hands its load to a body, and a hose
  with no vehicle there has no such body;
* directions and forces flip sign, $F_{n-1-i} = -\boldsymbol\tau_i$, because
  the cascade points the other way along the same links.

Two things the second consumer of that primitive needed, and one it did not:

`link_forces`
: **New.** `node_forces` prescribes the *applied* force at a node — the right
  hook when a vehicle's thrust is known. M6's vehicles are interior and their
  thrusts are the unknowns; what it knows is the link force. `link_forces` is
  the dual: it fixes $F_i$ outright and reports the applied force that makes
  the rung balance, so each vehicle's thrust vector comes back out of
  {py:attr}`~udaan.flatness.CableChain.applied_forces`. The two write the same
  rung from opposite ends, share one depth rule, and are mutually exclusive per
  link.

`require_hanging_branch=False` at $n_Q \ge 2$
: **New, and a removal — a partial one.** M5's cascade refuses
  $q_i \cdot e_3 > 0$, which is sound for a chain suspended from its top and is
  how it makes the slack refusal two-sided. A hose strung between two or more
  vehicles has links that point *upwards* on purpose, and the test refuses such
  a configuration outright, so at $n_Q \ge 2$ it has to go. It does **not** go
  at $n_Q = 1$: a hose whose single vehicle sits at the last vertex hangs off
  that vehicle exactly as M5's cable does — measured, $(n, \text{attachments})$
  of $(2,(2))$, $(3,(3))$ and $(4,(4))$ all pass the test — so the map leaves
  it on there and keeps M5's full protection. What the removal costs at
  $n_Q \ge 2$, and what replaces it, is {ref}`below <m6-guards>`.

The depth accounting, the per-rung normalisation
({py:func}`~udaan.flatness.normalize_derivatives`), the position propagation
and the slack refusal needed **nothing** and are used unchanged.

:::{admonition} What replaced the Mathematica codegen
:class: tip

The reference's tension→direction step is
`getUnitvectorsFromTensions`, a Mathematica-generated file of 1,378 lines with
higher-order siblings running to 32k, 588k and 976k lines (73 MB). It is
{py:func}`udaan.flatness.normalize_derivatives`: the closed-form Leibniz
recursion for $\bigl(\|v\|, v/\|v\|\bigr)^{(k)}$ at arbitrary order, in forty
lines, shared with every other cable map in the package. Nothing was ported.
:::

## Theorem

```{prf:theorem} Flatness of $n_Q$ quadrotors on an $n$-link hose
:label: thm-m6-flat

Let $\bar m_j > 0$ at every vertex, $\ell_i > 0$ at every link, and let the
attachment set contain vertex $n$. On the open set where no link force vanishes
($\boldsymbol\tau_i \ne 0$ for all $i$), every thrust vector is non-zero
($u_{a_k} \ne 0$), and no $b_3$ is parallel to its yaw hint, the system is
differentially flat with flat output {math:numref}`eq-m6f-flat-output`. Every
state and every input is an algebraic function of the $(2n+4)$-th order jet of
$x_0$, the $\bigl(2(n-a_k)+2\bigr)$-th order jets of the flat tensions, and the
second-order jets of the yaws.

If the attachment set does **not** contain vertex $n$, the system is not flat
with this flat output.
```

```{prf:proof}
Dimension matching is {math:numref}`eq-m6f-flat-output`. Steps 1–2 construct a
smooth map on jet space without integrating anything: the sweep strictly
ascends the vertex index and each of its lines is an explicit assignment.

That the result satisfies the equations of motion is again a check in two
parts. {math:numref}`eq-m6f-vertex` is imposed at every vertex *by
construction* — it is what each line of the table solves — so every
translational balance holds, and the attitude dynamics are
{math:numref}`eq-m6f-moment`. The per-link rotational equations are not
independent: crossing vertex $i$'s balance with $q_i$ annihilates the joint
forces along $q_i$ and leaves $\dot\omega_i = q_i \times \ddot q_i$, which
{math:numref}`eq-m6f-rates` supplies. Equivalently, the $(3+3n)$ mass matrix of
{doc}`../dynamics/multi-quad-hose` is positive definite whenever every
$\bar m_j > 0$, so the state and inputs constructed above are the *unique* ones
consistent with the given $\ddot x_0$ and $\{\boldsymbol\tau_{a_k}\}$: the map
inverts the dynamics rather than satisfying a subset of them.

Smoothness on the stated open set follows from the smoothness of
$v \mapsto (\|v\|, v/\|v\|)$ away from the origin.

For the converse, suppose $n \notin \mathcal{A}$. Then $u_n = 0$ and
$\boldsymbol\tau_n = 0$, and {math:numref}`eq-m6f-vertex` at $j = n$ reads
$\bar m_n(\ddot x_n + g e_3) + \boldsymbol\tau_{n-1} = 0$. Both terms are
determined by the flat output through the sweep, so this is three non-trivial
scalar constraints on $y$ and its derivatives; the flat output cannot be chosen
freely on any open set, contradicting {math:numref}`eq-df-recovery`.
```

(m6-guards)=
### Singularities, and what is actually guarded

$\|\boldsymbol\tau_i\| \to 0$ is the slack transition, the same hybrid-mode
boundary M2 and M5 have, and the map's domain genuinely stops there.
{py:attr}`~udaan.flatness.MultiQuadHoseInputs.tensions` reports every $T_i$, so
the margin is available at every instant. But it is a **two-sided** margin only
on the links whose force is a flat output. Only $n_Q - 1$ links are prescribed
that way; the other

$$
n_{\text{der}} \;=\; n - (n_Q - 1)
$$

are *derived*, $\boldsymbol\tau_j = \bar m_j(\ddot x_j + g e_3) +
\boldsymbol\tau_{j-1}$, and their directions come out of a norm,
$q_j = \boldsymbol\tau_j / \|\boldsymbol\tau_j\|$, exactly as M5's do. A
derived tension that passes *through* zero folds its link to the other side of
its vertex and comes back out with the tension positive and rising, and the
slack refusal — which only declines to *land* on zero — says nothing about it.
Neither do the prescribed tensions upstream: they pin their own links, not the
derived ones below.

So, three cases:

$n_Q = 1$
: **Fully guarded.** There are no prescribed rungs, the hose *is* an M5 chain
  hanging off its last vertex, and the map leaves M5's hanging-branch test on.
  A root dive through $\ddot x_0 \cdot e_3 = -g$ is refused rather than
  silently folded. (Without this, the whole hose flips to the other side of the
  root — vertex heights jumping by metres — with every tension positive and
  rising, every thrust positive, every attitude well defined, and no exception:
  `tests/flatness/test_multi_quad_hose.py` pins the refusal.)

$n_Q \ge 2$, prescribed links
: **Two-sided by construction**, and only here. $\boldsymbol\tau_{a_k}$ is
  handed over as a vector, so its direction is the planner's choice and its
  magnitude walks to zero with no branch to flip onto.

$n_Q \ge 2$, derived links
: **Guarded by a margin only.** No per-rung direction test is available: a
  derived link may legitimately point down while a prescribed one points up, so
  there is no fixed half-space to demand of every link. What the map does
  instead is what M3 and M4 do — a
  {py:attr}`~udaan.flatness.MultiQuadHose.min_tension` floor, applied to the
  derived rungs and refusing by link index. Because the fold always happens
  where a derived tension crosses zero, the floor refuses the *neighbourhood*
  of the fold; a sample grid coarse enough to step clean over that
  neighbourhood can still land past a fold undetected. The default is M3/M4's
  slack floor, which detects little more than the crossing itself: a planner
  should set it to a fraction of the tension it means to hold.

$u_{a_k} \to 0$ is a vehicle in free fall: its attitude is undefined, and the
map refuses by name rather than returning a degraded reference. It is reached
by asking a vehicle to carry nothing, which the flat tensions make easy to do
by accident — setting $\boldsymbol\tau_0 = \bar m_0 g e_3$ in hover leaves the
vehicle at vertex $0$ with nothing to lift. It is a *different* boundary from
the fold: a derived link can flip at thrusts nowhere near zero, so the free-fall
guard does not stand in for the tension floor.

## Derivative-order accounting

Each rung of the sweep costs two derivatives, for M5's reason: the force in
link $j$ is built from the *acceleration* of vertex $j$, and the vertex below
inherits the direction's depth through {math:numref}`eq-m6f-geometry`. With the
root jet supplied at order $K$:

| Quantity | Order available |
| --- | --- |
| $x_0$ (root) | $K$ |
| $\boldsymbol\tau_0,\ q_0,\ x_1$ | $K-2$ |
| $\boldsymbol\tau_1,\ q_1,\ x_2$ | $K-4$ |
| $\vdots$ | $\vdots$ |
| $x_n$ (last vertex) | $K-2n$ |

The binding consumer is the mandatory vehicle at vertex $n$: its thrust
{math:numref}`eq-m6f-vertex` needs $\ddot x_n$, $\Omega_n$ one derivative more,
and $\dot\Omega_n$ — hence $M_n$ — one more again. So $x_n$ is needed through
its snap:

```{math}
:label: eq-m6f-order
\boxed{\;K \;=\; 2n + 4\;}
```

the same law M5 obeys, for a different consumer. A flat tension enters at the
rung carrying force order $K - 2(a_k+1)$, so

```{math}
:label: eq-m6f-tension-order
\boldsymbol\tau_{a_k} \text{ through its } 2(n - a_k) + 2 \text{-th derivative,}
```

which is what {py:attr}`~udaan.flatness.MultiQuadHose.tension_jet_orders`
returns and what the cascade's own $K - 2(n-i)$ rule demands of a prescribed
force at that rung. Yaws are needed through $\ddot\psi_k$, unchanged.

:::{admonition} Confirmed by experiment — and $2n+4$ is tight only at $n_Q = 1$
:class: tip

Necessity is measured in `tests/flatness/test_multi_quad_hose.py` by perturbing
one row of the root jet and watching the last vehicle's moment. Supplying
$2n+3$ rows raises; supplying more than $2n+4$ changes nothing bit-for-bit.

The experiment also turns up something the reference does not mention. The
sweep reaches $x_0$'s deepest row by climbing through the accelerations, two
rows per rung — but a **prescribed** flat tension does not depend on $x_0$ at
all, so the climb terminates at that rung and only the direct
$x_j = x_0 + \sum \ell_i q_i$ term survives it. *Every vehicle below the last
therefore buys two rows back.* Measured, at $n = 4$:

| attachments | deepest row of $x_0$ that moves $M_{\text{last}}$ |
| --- | --- |
| $(4)$ | $2n+4 = 12$ |
| $(0, 4)$ or $(2, 4)$ | $2n+2 = 10$ |
| $(0, 2, 4)$ | $2n = 8$ |

The closed form, verified at every one of the 63 $(n, \text{pattern})$
combinations with $n \le 5$, is the derived-rung count again — the deepest row
read is

```{math}
:label: eq-m6f-depth-tight
d_{\max} \;=\; 4 + 2\,n_{\text{der}} \;=\; 4 + 2\bigl(n - n_Q + 1\bigr),
```

which depends on the *number* of vehicles and not on where they sit. So
$K = 2n+4$ is tight only at $n_Q = 1$, where it reduces to it.

The map still demands $2n+4$ uniformly: it is what the shared primitive
contracts for, extra rows are free, and the alternative is a depth law that
depends on the attachment pattern in a way no caller wants to reason about. But
the reference's $d_{\max} = 4 + 2n$ is *conservative by two rows per extra
vehicle*, and a trajectory generator that has to produce a $C^{2n+4}$ reference
may care.
:::

:::{admonition} Are the deep rungs more observable than M5's?
:class: tip

Yes — but because of the tension, not the mass.

{doc}`quadrotor-flexcable` established that the per-rung attenuation goes as
$m_i \ell_i / T_i$ rather than as $n$, and a hose does carry much heavier
vertices than a cable. But at $n_Q = 1$ a hose *is* an M5 chain (see the
reduction below), and there the mass **cancels**: the tension in link $i$ is
the weight it carries, so $m_i \ell_i / T_i$ is scale-free in $m$. Multiplying
every vertex mass by 15 moves the deepest-row sensitivity by less than a factor
of two.

What M6 has and M5 cannot have is a tension that is a *flat output*, chosen
independently of the mass. Holding a heavy hose at low tension is a
configuration no M5 chain can reach, and it is exactly the one in which the
deep rows matter most. At $n = 4$ with vehicles at $(0, 4)$, the sensitivity of
$M_{\text{last}}$ to row $2n+2$ of the root jet runs

| flat pull $\|\boldsymbol\tau_0\|_z$ | 1 N | 3 N | 10 N | 30 N | 100 N | 300 N | 1000 N | 10000 N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\partial\|M\|/\partial x_0^{(10)}$ | $9.58\!\times\!10^{-9}$ | $9.17\!\times\!10^{-9}$ | $2.87\!\times\!10^{-10}$ | $1.09\!\times\!10^{-12}$ | $4.97\!\times\!10^{-15}$ | $5.36\!\times\!10^{-17}$ | $4.14\!\times\!10^{-19}$ | $4.07\!\times\!10^{-23}$ |
| local log-log slope | | $-0.04$ | $-2.88$ | $-5.08$ | $-4.48$ | $-4.12$ | $-4.04$ | $-4.01$ |

Fourteen orders of magnitude, and the exponent it settles on is **$-4$**, not
the $-3$ the three derived rungs alone would give. The first few columns are
not on the power law at all: at 1 N the flat pull is no larger than the vertex
weights, so it is gravity rather than $\boldsymbol\tau_0$ that sets the link
tensions — read a slope off that end (the 1 N–30 N pair gives $-2.67$) and it
means nothing. Only from about 100 N, where every link's tension is dominated
by the pull, does the slope settle, and what it settles on is $-4$.

The extra factor over the rung count is the last vehicle itself: its thrust has
to react $\boldsymbol\tau_{n-1}$, so $\|u_n\|$ grows with $T$, and recovering
its attitude divides by that norm once more. The general law, measured across
attachment patterns at $n = 2, 3, 4$, is one factor per derived rung plus that
one:

```{math}
:label: eq-m6f-observability
\frac{\partial \|M_{\text{last}}\|}{\partial x_0^{(k)}} \;\sim\;
\frac{\ell^{\,n_{\text{der}}}}{T^{\,n_{\text{der}}+1}},
\qquad n_{\text{der}} \;=\; n - (n_Q - 1),
```

with $n_{\text{der}}$ the number of links whose tension the recursion *derives*
rather than reading off the flat output. Measured slopes at
$n_{\text{der}} = 3, 2, 1$: $-4.01$, $-3.01$, $-2.01$.

The $\ell$ half is exact rather than asymptotic. Doubling the hose multiplies
the sensitivity by $2^{n_{\text{der}}}$ to five decimal places — ratio
$8.00000$ across four successive doublings at $n_{\text{der}} = 3$ (log-log
slope $3.00000$), and $4.00000$ / $2.00000$ at $n_{\text{der}} = 2$ and $1$.

Both exponents are pinned in `tests/flatness/test_multi_quad_hose.py`. So the
deep rows are plainly observable on a slack hose and plainly not on a taut one,
at the same $n$, the same masses and the same lengths.
:::

## Reduction to M5

Set $n_Q = 1$ with the single vehicle at vertex $n$. There is then no flat
tension ($n_Q - 1 = 0$), the sweep has no prescribed rung, and the mechanism is
a chain hanging off one airframe — {doc}`quadrotor-flexcable` exactly, read
from the payload end. M6's root vertex is M5's payload, M6's last vertex is
M5's airframe, and reversing the per-index parameters
($\bar m_j = m^{M5}_{n-1-j}$ for $j < n$, $\bar m_n = m_Q$, $\ell$ reversed)
must reproduce M5 term for term, with $q_i^{M6} = -q_{n-1-i}^{M5}$ because the
two conventions point opposite ways along the same links. The tests assert
exactly that — thrust, moment, attitude, body rate, every vertex position and
every tension — and it holds to $10^{-10}$, which is the tightest possible
statement given that the two maps reach the answer through the same primitive
under a double index reversal.

## Validation

The load-bearing check is the **open-loop round trip**
(`tests/flatness/test_multi_quad_hose_feedforward.py`): the flat-derived
$(f_k, M_k)$ is fed into
{py:class}`~udaan.models.multi_quad_hose.MultiQuadHoseBase` with no controller,
integrated with DOP853 at $10^{-12}$ tolerances, and the resulting trajectory
compared against the flat outputs at $t = 5$ s. It closes to $10^{-16} \dots
10^{-11}$ — integrator round-off — across seven $(n, n_Q, \text{pattern})$
combinations including an interior vehicle, a hose whose root carries no
vehicle, a vehicle at every vertex, and heterogeneous vertex masses, link
lengths, airframe masses and inertias.

Two cautions carried over from earlier steps shape that harness.

First, **a "check" that reaches the moment the way the map built it proves
nothing.** The map computes $M = J\dot\Omega + \Omega \times J\Omega$ and the
model inverts precisely that expression for $\dot\Omega$; comparing the two
would return zero for *any* $M$. The round trip is not tautological because the
translational half of the state is compared against flat outputs the model
never saw, and it closes only if the attitude, the thrust magnitude, the link
geometry and the hose's mass matrix all agree.

Second, **the ZOH-Euler envelope is $n \le 4$ at any $n_Q$**
({doc}`../dynamics/multi-quad-hose`), so the shipped `step()` path — which also
exercises actuator clipping — is kept inside it, and everything beyond runs on
DOP853. The ZOH residual is four orders larger than the DOP853 one and shrinks
in proportion to the step, which is what identifies it as integrator error
rather than a formula defect.

## What is wrong in the reference

{doc}`../dynamics/multi-quad-hose` records what the dynamics re-derivation
found. The flat map has its own list, and it is longer.

**`flat2state.m` handles only vehicles at vertices $0$ and $n$.** The
quadrotor branch tests `if i == 0 ... elseif i == obj.n` with **no `else`** —
`flat2state.m` L40-89 and, identically, the codegen twin `afh_flat2state.m`
L64-117. An *interior* vehicle therefore falls through: neither its vertex
position nor the tension in the link below it is ever written. The vehicle
counter `iq = iq+1` sits *outside* that inner branch, so it still advances, and
the next iteration reads empty structs. The general rule is the one-line
unification in the table above,
$u_j = \bar m_j(\ddot x_j + g e_3) + \boldsymbol\tau_{j-1} - \boldsymbol\tau_j$
with $\boldsymbol\tau_{-1} = \boldsymbol\tau_n = 0$, of which the reference's
two branches are the $j = 0$ and $j = n$ cases.

This matters because the paper's own $n_Q = 3$ setpoint simulation,
`sim_1b_SETPOINT_nQ3.m`, builds `I = [1, 6, 11]` — a vehicle at interior vertex
5. It survives only because it passes `'xd', 'fixed'`, which matches *neither*
branch of `parse_inputs.m` L63-68: the `'flatness'` case installs
`flat2state`, the `'fixed'` case is commented out, so `get_desired_states` is
simply never bound, and the script then overwrites the reference with
`computeSetpoint`'s numerical result. Every example that actually calls
`flat2state` has vehicles only at the ends.

**All flat tensions are set to the same value.** The loop over `ft_ind` assigns
`T.T = 1.*flats.T.T` for every vehicle from a single shared struct, marked
`>>> temperorary <<<`. With more than two vehicles that is not the map, it is
one tension used $n_Q - 1$ times.

**The codegen twin is dead on its own no-vehicle-at-vertex-0 path.**
`afh_flat2state.m` reads `states.positions(i)` at `i = 0` — a MATLAB index
error — and then calls `flats_n_states.set_tension_in_link`, a method of an
object that does not exist in that function's scope. The branch cannot ever
have run.

**`dmax = 4 + 2n` is right but conservative**, by two rows for every vehicle
below the last: the row actually read is $4 + 2(n - n_Q + 1)$
({math:numref}`eq-m6f-depth-tight`), so the two agree only at $n_Q = 1$.

And one thing it gets exactly right, worth recording because the re-derivation
looked hard for a reason to disagree: the sign conventions, the recursion
$T_{i+1} = \bar m_i(\ddot x_i + g e_3) + T_i$, the thrust formulas at both
ends including the sign flip between them, and the not-flat guard are all
correct as written.

## Jet input

The map takes one root {py:class}`udaan.flatness.Jet` of order $2n+4$, one
tension jet per vehicle except the last, and one order-2 scalar yaw jet per
vehicle, packaged in a
{py:class}`udaan.flatness.MultiQuadHoseFlats`:

```python
import numpy as np
from udaan.flatness import Jet, MultiQuadHose, MultiQuadHoseFlats

f2s = MultiQuadHose(num_links=4, attachments=(0, 2, 4))   # or .from_model(model)
f2s.jet_order            # -> 12  == 2n + 4
f2s.tension_jet_orders   # -> (10, 6)  == 2(n - a) + 2 for a in (0, 2)

# Root vertex position and its first 2n+4 time derivatives at t
x_0_jet = Jet(np.stack([
    x_0,            # row  0 — root position
    x_0_d1,         # row  1 — velocity
    x_0_d2,         # row  2 — acceleration  (gives u_0 at vertex 0)
    # ...                                      two more rows per rung down the hose
    x_0_d12,        # row 2n+4 — the deepest row
]))

# One vector-tension jet per vehicle except the one at the last vertex.
tension_jets = [Jet(rows_0), Jet(rows_2)]     # orders 10 and 6
psi_jets = [Jet(np.array([psi, psi_dot, psi_ddot])) for _ in range(3)]

ref, inputs = f2s.recover(
    MultiQuadHoseFlats(x_0=x_0_jet, tensions=tension_jets, psi=psi_jets)
)

# ref.root_position, ref.root_velocity, ref.root_acceleration
# ref.link_attitudes[i], ref.link_angular_velocities[i],
# ref.link_angular_accelerations[i]              -> per link, root -> last vertex
# ref.vertex_positions, ref.vertex_velocities    -> (n+1, 3), every vertex
# ref.positions[k], ref.velocities[k]            -> per vehicle
# ref.orientations[k], ref.angular_velocities[k], ref.angular_accelerations[k]
# inputs.thrusts[k], inputs.moments[k]
# inputs.tensions[i]                             -> the slack margin, per link
# inputs.as_wrench()                             -> (nQ, 4), ready for model.step(...)
```

`ref` is a {py:class}`udaan.flatness.MultiQuadHoseRefState` (its
`as_state(attachments=...)` projects onto the simulation model's state struct,
ready for `model.reset(...)`) and `inputs` is a
{py:class}`udaan.flatness.MultiQuadHoseInputs`.
{py:meth}`~udaan.flatness.MultiQuadHose.cascade` returns just the sweep — every
vertex position, link direction, link force and vehicle thrust vector with all
the derivatives the recursion produced — for anything that wants the geometry
without the attitude machinery.

## References

```{footbibliography}
```
