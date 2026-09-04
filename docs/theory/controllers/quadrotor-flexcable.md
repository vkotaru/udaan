(control-quadrotor-flexcable)=
# Payload tracking through a flexible $n$-link cable

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

{py:class}`udaan.control.quadrotor_flexcable.QuadrotorFlexCableController`
tracks a payload trajectory with one quadrotor whose cable is discretised
into $n$ rigid links. It closes the loop on the dynamics of
{doc}`../dynamics/quadrotor-flexcable` and is fed forward by the flat map of
{doc}`../differential-flatness/quadrotor-flexcable`
{footcite}`kotaru2017differential`.

It is the first controller in this package that is **not** a geometric
cascade. That is worth stating plainly, because the three systems before it
— {doc}`quadrotor-payload`, {doc}`multi-quad-payload`, and the rigid-body
generalisation of the latter — are all cascades, and the departure here is
structural rather than a matter of taste.

## Why the cascade does not generalise

Notation follows the dynamics page: $q_i \in S^2$ points from the top of the
chain towards the payload, $\ell_i$ and $m_i$ are link $i$'s length and its
lumped distal mass, $x_Q$ and $R$ are the airframe's position and attitude,
and

$$
M_{00} = m_Q + \sum_k m_k, \qquad
M_{0i} = \Bigl(\sum_{k \ge i} m_k\Bigr)\ell_i, \qquad
M_{ij} = \Bigl(\sum_{k \ge \max(i,j)} m_k\Bigr)\ell_i \ell_j
$$

are the mass coefficients of {eq}`eq-m5-coeff`.

Every cascade in this package is built on the same move: command a desired
cable direction $q_i^d$, and hand it to a body that can push on that cable
directly. In M2 there is one cable and one quadrotor; in M3 and M4 there are
$n_Q$ cables and $n_Q$ quadrotors, one per cable, and the extra freedom is
resolved by a tension-split policy or a grasp map. In each case the number of
independently actuated bodies equals the number of cables.

A flexible cable breaks that correspondence completely. There is **one**
actuated body and $n$ links, and the airframe is attached to link $0$ only.
Links $1 \dots n-1$ can be moved solely through the chain's own internal
forces. The plant has four inputs — collective thrust $f$ and body moment
$M$ — and $12 + 4n$ degrees of freedom, so for $n \ge 1$ there is no
$q_i^d$ that any actuator can be told to chase, and no loop that can be
inverted one level at a time.

Backstepping through $n$ nested error variables is possible in principle
{footcite}`goodarzi2014geometric` and needs derivatives of every intermediate
desired quantity. The published route for this system is the other one: a
linear-quadratic regulator on the *variational* linearisation, which is what
is implemented.

## Error state

The controller regulates a variational error about a reference
$\bigl(x_Q^d, v_Q^d, R_d, \Omega_d, \{q_i^d, \omega_i^d\}\bigr)$:

$$
s = \begin{bmatrix}
e_{x_Q} \\ e_{v_Q} \\ \eta \\ \delta\Omega \\ \xi \\ \delta\omega
\end{bmatrix},
\qquad
\begin{aligned}
e_{x_Q} &= x_Q - x_Q^d, &  e_{v_Q} &= v_Q - v_Q^d, \\
\eta &= \tfrac12 \operatorname{vee}\bigl(R_d^\top R - R^\top R_d\bigr),
     & \delta\Omega &= \Omega - R^\top R_d\, \Omega_d, \\
\xi_i &= q_i^d \times q_i,
     & \delta\omega_i &= \omega_i + q_i \times (q_i \times \omega_i^d).
\end{aligned}
$$ (eq-m5c-error)

The $SO(3)$ pair is the geometric one of {doc}`quadrotor-se3`; the $S^2$ pair
is the one M2, M3 and M4 already use for their cables, and it is *their*
convention rather than the legacy MATLAB reference's that is followed here, so
that a reader moving between the four systems meets one definition.

Two features are worth pausing on.

**The translational block is the quadrotor's, not the payload's.** The
equations of motion have exactly one row that carries the thrust — the $x_Q$
row — and the linearisation below is written in it. Regulating the payload
position directly would leave the chain's *shape* unconstrained, which for a
flexible cable is most of the state. The payload is tracked through the
chain instead: $x_Q^d = x_L^d - \sum_i \ell_i q_i^d$, so
$e_{x_Q} \to 0$ together with $\xi \to 0$ puts the payload at $x_L^d$.

**$2n$ of the $12 + 6n$ coordinates are gauge.** $\xi_i$ is a cross product
with $q_i^d$, hence orthogonal to it; $\omega_i$ is tangent to $q_i$ by
construction. The components along $q_i^d$ are therefore identically zero (to
first order) and identically unactuated — the same $S^2$ gauge freedom that
the dynamics page removes by writing the diagonal block as $M_{ii} I$ rather
than $M_{ii}(I - q_i q_i^\top)$. The physical error state has
$12 + 4n$ dimensions, and {ref}`the Riccati solve <m5c-controllability>`
has to be run there.

## Variational linearisation

Differentiating the equations of motion at the reference and eliminating the
mass matrix gives $\dot s = A s + B\,\delta u$ with $\delta u = (\delta f,
\delta M)$. Writing $\mathcal{M}$ for the reference mass matrix of
{eq}`eq-m5-system`, the airframe and link rows come from a single solve

$$
\mathcal{M}
\begin{bmatrix} \delta\alpha \\ \delta\dot\omega \end{bmatrix}
= \mathcal{F}\, s + \mathcal{G}\, \delta u ,
$$ (eq-m5c-varsolve)

whose thrust-direction entry is $-f_d\, R_d \hat e_3$ (in the $\eta$ columns)
and whose input matrix is $\mathcal{G} = \bigl[\,R_d e_3\;\; 0\,\bigr]$ on
the airframe row and zero on the link rows. The kinematic rows close the
system:

$$
\dot e_{x_Q} = e_{v_Q}, \qquad
\dot\eta = -\hat\Omega_d\, \eta + \delta\Omega, \qquad
\dot{\delta\Omega} = J^{-1}\bigl(\widehat{J\Omega_d} - \hat\Omega_d J\bigr)\,
                     \delta\Omega + J^{-1} \delta M,
$$

$$
\dot\xi_i = q_i^d (q_i^d)^\top \hat\omega_i^d\, \xi_i
            + \bigl(I - q_i^d (q_i^d)^\top\bigr)\, \delta\omega_i .
$$ (eq-m5c-xidot)

{eq}`eq-m5c-xidot` is where the gauge shows, though not quite as cleanly as
one would like. Dotting it with $q_i^d$ kills the *input* term exactly —
$q_i^d \cdot (I - q_i^d (q_i^d)^\top) = 0$ for every $\delta\omega_i$, at
every reference — but the first term survives:

$$
q_i^d \cdot \bigl(q_i^d (q_i^d)^\top \hat\omega_i^d\, \xi_i\bigr)
 = \xi_i \cdot (q_i^d \times \omega_i^d) ,
$$ (eq-m5c-gaugeleak)

which vanishes only when the reference link is at rest ($\omega_i^d = 0$),
or when $\xi_i$ happens to be orthogonal to $q_i^d \times \omega_i^d$. So the
gauge direction is *uncontrollable* everywhere — no input reaches it — but it
is *invariant* only at a stationary reference. What that costs the reduction
below is quantified in {ref}`m5c-controllability`.

{py:meth}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.linearize`
implements this in full reference-dependent form. Doing so is not decoration
— it is what makes the argument for a *constant* gain, below, a measurement
rather than an assertion.

(m5c-controllability)=
## What is controllable

Two separate statements, and both matter.

**Structurally.** Let $W$ be the $(12+6n) \times (12+4n)$ isometry that keeps
the twelve rigid-body coordinates and, per link, the two directions of
$(q_i^d)^\perp$. That subspace contains the whole range of $B$ at every
reference, so

$$
A_r = W^\top A W, \qquad B_r = W^\top B
$$

is well posed, and $K = K_r W^\top$ pushes the reduced gain back out without
ever acting on a gauge direction. The reduction is necessary rather than
tidy: handing the *embedded* pair $(A, B)$ to an algebraic Riccati solver
asks it to stabilise $2n$ integrators no input reaches, and it fails
outright. (The MATLAB reference never meets this, because a finite-horizon
Riccati sweep needs no stabilisability at all.)

Whether it *loses nothing* is a separate question, and {eq}`eq-m5c-gaugeleak`
is the reason it has to be asked. The subspace is invariant only where the
reference links are at rest. At the hanging equilibrium, which is where the
shipped gain is solved, $\omega_i^d = 0$ and

$$
\bigl\|(I - WW^\top)\, A\, W\bigr\| = 0
$$

exactly — so the shipped gain rests on an exact reduction. Off it the leak is
small but real; walked along the gentle Lissajous below, the relative residual
$\|(I - WW^\top) A W\| / \|A\|$ peaks at $2.2\times10^{-3}$, $3.0\times10^{-4}$
and $1.1\times10^{-4}$ at $n = 1, 2, 3$. The consequence is confined to
{py:meth}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.set_gain_from`
used as a *scheduling* hook: at a moving reference it solves a nearby problem,
not the exact one.

**Practically.** The reduced pair is controllable at every link count tested,
and saying only that would be misleading. What actually happens is that the
*authority* runs out. Writing the PBH margin as
$\sigma_{\min}\bigl[\,A_r - \lambda I \;\; B_r\,\bigr]$ minimised over the
open-loop spectrum, on the shipped defaults:

| $n$ | PBH margin | slowest closed-loop pole | damping |
|-----|-----------|--------------------------|---------|
| 1 | $0.62$ | $-1.00$ | (real) |
| 2 | $0.085$ | $-0.64 \pm 15.1\mathrm{j}$ | 4.2 % |
| 3 | $0.026$ | $-0.34 \pm 26.7\mathrm{j}$ | 1.3 % |
| 5 | $0.0082$ | $-0.17 \pm 48.3\mathrm{j}$ | 0.4 % |

Two things follow, and they are the honest statement of this controller's
reach.

First, **the mode that ends up slowest is not the payload swing — it is the
chain's *fastest* internal bending mode.** Its frequency is, to two decimal
places, the maximum of
{py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.natural_frequencies`.
The payload pendulum, the thing one would expect to be hard, is damped
briskly; it is the high-frequency shape of the cable that the airframe cannot
reach.

Second, **the weights do reach it — for a price.** It is tempting to say they
do not, and the usual experiment appears to confirm it: scaling $Q$ by $1000$
— a factor of $60$ in gain magnitude — moves the $n = 3$ figure from $-0.339$
to $-0.356$, about 5 %. But that experiment cannot show anything. For an LQR
the gain depends on $Q$ and $R$ only through $R^{-1/2} B^\top$ and $Q$, so
scaling $Q$ *uniformly* is identical to scaling $R^{-1}$ uniformly: it walks
the cheap-control locus, along which the finite closed-loop poles converge to
the plant's stable modes and the mirror images of its unstable ones. Uniform
scaling is the one perturbation guaranteed to leave the plant's own modes
where they are.

Weighting the link *rates* specifically does move the mode. Raising
$q_{\omega}$ (`q_link_rate`) alone by $50\times$, at $3.3\times$ the gain
norm:

| $n$ | top bending mode, defaults | $q_{\omega} \times 50$ |
|-----|---------------------------|------------------------|
| 2 | $-0.64 \pm 15.1\mathrm{j}$ (4.2 %) | $-4.02 \pm 15.4\mathrm{j}$ (25.3 %) |
| 3 | $-0.34 \pm 26.7\mathrm{j}$ (1.3 %) | $-2.19 \pm 26.8\mathrm{j}$ (8.2 %) |

and the slowest pole overall at $n = 3$ improves from $-0.339$ to $-0.514$.
Nonlinearly, an $n = 3$ chain station-keeping with every link kicked to
$\pm 2$ rad/s shows a bending residual that is flat over $10$ s on the
defaults and falls by three orders of magnitude with the same reweighting.

The price is tracking. On the gentle Lissajous the tail RMS payload error at
$n = 3$ goes from $0.00056$ m to $0.0138$ m — a factor of $25$. So the honest
statement is a **damping-versus-tracking trade-off**, not a structural limit:
the shipped defaults buy tracking, and a caller who would rather have a quiet
cable than $14$ mm of payload error should raise `q_link_rate`. What *is*
structural, and what no weighting improves, is the PBH margin in the table
above. Both quantities are published as
{py:attr}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.controllability_margin`
and
{py:attr}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.closed_loop_poles`.

There is a consolation, and it is not a coincidence. The mode the controller
cannot damp is the same one that sets the ZOH-Euler stability limit of
{py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.stability_growth_rate`,
because both are $\max_k \omega_k$. At any chain the analytical integrator
can hold at all, that mode is slow enough to stay small.

## The gain

With $Q$ block-diagonal over the six blocks of {eq}`eq-m5c-error` and
$R = \operatorname{diag}(r_f, r_M, r_M, r_M)$, the gain is the
infinite-horizon solution

$$
A_r^\top P + P A_r - P B_r R^{-1} B_r^\top P + Q_r = 0,
\qquad
K_r = -R^{-1} B_r^\top P ,
$$ (eq-m5c-care)

solved once, at the hanging equilibrium $q_i^d = -e_3$, $R_d = I$,
$\Omega_d = 0$, $f_d = M_{00} g$, by
{py:func}`scipy.linalg.solve_continuous_are`. The command is

$$
(f, M) = (f_d, M_d) + K s .
$$

### Infinite horizon, not the reference's finite one

{footcite}`kotaru2017differential`'s implementation integrates the Riccati
differential equation backwards over $[10, 0]$ along one specific trajectory,
caches $P(t)$ to a file, and interpolates it at run time. Three reasons for
solving the algebraic equation instead.

*It is causal and self-contained.* The backward sweep needs the entire
reference in advance and yields a schedule tied to one horizon and one
manoeuvre. A controller in this package has to work against a `setpoint`
callable it has never seen.

*The linearisation moves little on gentle references.* The flat map refuses
any reference on which a link leaves the hanging branch, so every trajectory
that can be fed forward keeps each $q_i^d$ within $90^\circ$ of $-e_3$. That
bound is far too loose to justify a constant gain on its own, and it should
not be read as if it were: admissibility is not the same as *gentleness*, and
the flat map accepts references that move $A$ a very long way. Measured at
$n = 3$ over three `CrazyTrajectory` amplitudes, every one of which the map
accepts on **every** sample (zero feedforward drops):

| reference | max link tilt | $\|A(\text{ref}) - A(\text{hov})\| / \|A(\text{hov})\|$ |
|-----------|---------------|--------------------------------------------------------|
| gentle Lissajous | $3.2^\circ$ | $5.7\,\%$ |
| `ax=0.6, f1=0.4` | $21.3^\circ$ | $37\,\%$ |
| `ax=1.0, f1=0.5` | $45.4^\circ$ | $93\,\%$ |

So the constant gain is justified **for gentle references** — which is what
the demos and the CLI fly, and what the feedforward is sized for — and
degrades as the reference gets aggressive. It is a design choice with a
stated domain, not a bound over the admissible set.

*The time variation a schedule captures is feedforward, and the flat map
supplies it exactly.* $(f_d, M_d)$ and the whole reference state come from
{py:class}`udaan.flatness.QuadrotorFlexCable`; what is left for $K$ is a
small regulation problem.

Gain scheduling remains two lines away —
`ctrl.set_gain_from(*ctrl.linearize(ref, inputs), ref.link_attitudes)` — and
is deliberately not automatic. It is the right thing to reach for on an
aggressive reference, subject to the caveat above: at a moving reference the
tangent reduction it solves on is approximate.

### Weights

The $Q$ blocks are the reference's, unchanged:
$0.5$ on position and velocity, $0.75$ on attitude and body rate, $1.0$ on
the link directions and $0.75$ on their rates. $R$ is not. The reference
weights all four inputs equally at $0.2$, though the moment channel's input
matrix is $J^{-1}$ — spectral norm $435$ for the default airframe — while the
thrust channel's is $1/M_{00}$, order $1$; the two differ by
$\|J^{-1}\| M_{00} = 522$. Equal weights therefore buy an attitude loop
roughly $500\times$ faster than the thrust loop, and the fastest closed-loop
pole lands at $-842$ rad/s.

**What that costs.** On a $0.62$ m payload step, tail RMS error over the
second half of a $20$ s run on the analytical model:

| $n$ | $r_M = 0.2$ | $r_M = 20$ |
|-----|-------------|------------|
| 1 | $4.96$ m | $10^{-5}$ m |
| 2 | $0.37$ m | $10^{-5}$ m |
| 3 | $1.46$ m | $1.6\times10^{-5}$ m |

At $0.2$ the loop simply never settles, at any link count. That measurement,
not an argument, is why the deviation ships.

**The mechanism is sampled-data phase loss, not discrete-time divergence.**
It is worth being exact, because the tempting explanation is wrong: a
$-842$ rad/s pole is *not* unstable under this integrator. Exactly held over
the $5$ ms outer step it maps to $e^{-842 \cdot 0.005} = 0.015$, and the model
integrates with forward Euler at $h = \Delta t / 4 = 1.25$ ms, where
$|1 - 842h| = 0.053$. Both sit deep inside the unit circle. What breaks the
loop is that the *command* is frozen across the whole $5$ ms hold while the
closed loop wants to act at $842$ rad/s: the hold's half-sample delay is
$\omega \Delta t / 2 = 2.1$ rad $\approx 121^\circ$ of phase lag right at the
crossover the design created — more than any of its phase margin. The
reference's adaptive `ode15s` never meets this because it re-evaluates the
control continuously.

Weighting the moments at $20$ moves the fastest pole to $-84$ rad/s, or
$12^\circ$ of hold lag. Being an *input* weight it leaves the chain modes
nearly alone — but "nearly" grows with the link count, and from $n = 2$ up it
is always in the *worse* direction. The slowest closed-loop real part moves by

| $n$ | 1 | 2 | 3 | 5 |
|-----|---|---|---|---|
| shift | $0.02\,\%$ (better) | $1.54\,\%$ | $4.61\,\%$ | $13.10\,\%$ |

so at $n = 5$ this input weight costs more chain damping than the $60\times$
$Q$ scaling above buys. The old claim that "the slowest pole moves by under
$2\,\%$" is true only at $n \le 2$. It is a real price, paid knowingly for a
loop that settles at all.

## Feedforward, and the price of $2n + 4$

The flat map of {doc}`../differential-flatness/quadrotor-flexcable` needs the
payload position through its $(2n+4)$-th derivative, because the chain
cascade spends two derivatives on every rung. That is the one place where the
link count reaches out of the controller and into the *trajectory*: at
$n = 3$ a feedforward reference must be differentiable ten times in closed
form, and at $n = 5$, fourteen.

Most trajectory classes here stop at snap, so
{py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.simulate`
accepts two protocols. `get_full(t)` is zero-padded past snap — the recovered
positions and link directions are unaffected, since the top node's
*acceleration* only needs the tip through order $2n+2$, but the recovered
$\Omega$, $\dot\Omega$ and $M_d$ degrade — and the fallback is warned about
once. `get_jet(t, order)` supplies every row exactly;
{py:class}`~udaan.utils.trajectory.CrazyTrajectory` implements it, being a
sum of sinusoids for which every derivative is a phase shift.

Measured on a Lissajous payload trajectory, steady-state payload tracking:

| $n$ | feedback only | `get_full`, padded | `get_jet`, exact |
|-----|--------------|--------------------|------------------|
| 1 | 0.316 m | 0.0039 m | 0.0003 m |
| 2 | 0.354 m | 0.0047 m | 0.0004 m |
| 3 | 0.380 m | 0.0050 m | 0.0006 m |

So the padded jet is worth about $75\times$ and the last $2n$ derivatives
another $8$–$10\times$ on top.

As in M3 and M4, the flat map is taken from the *controller's* own
{py:attr}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.flat_map`
rather than rebuilt from the model, so the reference and the gain always
describe the same machine; a controller that does not publish one gets no
feedforward at all. Samples the map refuses — a link gone slack, an airframe
in free fall — fall back to the hanging-equilibrium reference for that step
and are counted in
{py:attr}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.feedforward_drops`.

## References

```{footbibliography}
```
