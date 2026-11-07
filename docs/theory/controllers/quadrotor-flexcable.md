(control-quadrotor-flexcable)=
# Payload tracking through a flexible $n$-link cable

{py:class}`udaan.control.quadrotor_flexcable.QuadrotorFlexCableController`
tracks a payload trajectory with one quadrotor whose cable is discretised
into $n$ rigid links. It closes the loop on the dynamics of
{doc}`../dynamics/quadrotor-flexcable` and is fed forward by the flat map of
{doc}`../differential-flatness/quadrotor-flexcable`
{footcite}`kotaru2017differential`.

Unlike {doc}`quadrotor-payload`, {doc}`multi-quad-payload` and the
rigid-body generalisation of the latter, it is **not** a geometric cascade,
and the difference is structural rather than a matter of taste.

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
convention rather than the reference MATLAB implementation's that is followed here, so
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
implements this in full reference-dependent form, which is what lets the
argument for a *constant* gain, below, be checked against the linearisation at
any reference.

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
small but nonzero: along a gentle Lissajous the relative residual
$\|(I - WW^\top) A W\| / \|A\|$ stays below $5\times10^{-3}$, and it grows
with how hard the reference works. The consequence is confined to
{py:meth}`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController.set_gain_from`
used as a *scheduling* hook: at a moving reference it solves a nearby problem,
not the exact one.

**Practically.** The reduced pair is controllable at every link count tested,
but the *authority* runs out as $n$ grows. The PBH margin
$\sigma_{\min}\bigl[\,A_r - \lambda I \;\; B_r\,\bigr]$, minimised over the
open-loop spectrum on the shipped defaults, falls by well over an order of
magnitude from $n = 1$ to $n = 5$, and the slowest closed-loop pole becomes a
lightly damped oscillation whose damping shrinks with $n$.

Two things follow, and they bound this controller's reach.

First, **the mode that ends up slowest is not the payload swing — it is the
chain's *fastest* internal bending mode.** Its frequency is, to two decimal
places, the maximum of
{py:meth}`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.natural_frequencies`.
The payload pendulum, the thing one would expect to be hard, is damped
briskly; it is the high-frequency shape of the cable that the airframe cannot
reach.

Second, **the weights do reach it — for a price.** Scaling $Q$ uniformly
does not: it multiplies the gain severalfold and moves the slowest pole by only
a few percent. That is expected rather than informative. For an LQR
the gain depends on $Q$ and $R$ only through $R^{-1/2} B^\top$ and $Q$, so
scaling $Q$ *uniformly* is identical to scaling $R^{-1}$ uniformly: it walks
the cheap-control locus, along which the finite closed-loop poles converge to
the plant's stable modes and the mirror images of its unstable ones. Uniform
scaling is the one perturbation guaranteed to leave the plant's own modes
where they are.

Weighting the link *rates* specifically does move the mode. Raising
$q_{\omega}$ (`q_link_rate`) alone by $50\times$, with a modest increase in
gain norm, raises the top bending mode's damping several-fold.

The price is tracking: with the same reweighting the payload tracking error on
a gentle Lissajous grows several-fold. So this is a **damping-versus-tracking
trade-off**, not a structural limit: the shipped defaults buy tracking, and a
caller who would rather have a quiet cable should raise `q_link_rate`. What *is*
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
bound is far too loose to justify a constant gain on its own: admissibility
is not the same as *gentleness*, and the flat map accepts references that move
$A$ a long way. On a gentle Lissajous the relative change
$\|A(\text{ref}) - A(\text{hov})\| / \|A(\text{hov})\|$ stays under 10 %; on
an aggressive `CrazyTrajectory` that tilts the links by tens of degrees, and
that the map still accepts on every sample, it is several times larger.

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
pole lands beyond $-800$ rad/s. With those weights the loop does not settle
on a payload step at any link count; with $r_M = 20$ it does.

**The mechanism is sampled-data phase loss, not discrete-time divergence.**
A pole near $-842$ rad/s is *not* unstable under this integrator. Exactly held
over the $5$ ms outer step it maps to $e^{-842 \cdot 0.005} = 0.015$, and the
model integrates with forward Euler at $h = \Delta t / 4 = 1.25$ ms, where
$|1 - 842h| = 0.053$. Both sit deep inside the unit circle. What breaks the
loop is that the *command* is frozen across the whole $5$ ms hold while the
closed loop wants to act at $842$ rad/s: the hold's half-sample delay is
$\omega \Delta t / 2 = 2.1$ rad $\approx 121^\circ$ of phase lag right at the
crossover the design created — more than any of its phase margin. The
reference's adaptive `ode15s` never meets this because it re-evaluates the
control continuously.

Weighting the moments at $20$ moves the fastest pole under $100$ rad/s, about
$12^\circ$ of hold lag. Being an *input* weight it leaves the chain modes
nearly alone, but the slowest closed-loop real part moves in the *worse*
direction from $n = 2$ up, by an amount that grows with the link count. That
cost is accepted for a loop that settles at all.

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

On a Lissajous payload trajectory, feedforward reduces steady-state payload
tracking error substantially relative to feedback only, and the exact jet
improves further on the zero-padded one.

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
