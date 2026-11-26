(control-multi-quad-payload)=
# Cooperative payload transport with $n_Q$ quadrotors

{py:class}`udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController`
tracks a payload trajectory with $n_Q$ quadrotors, each attached to the
payload by its own taut cable. It is the generic-$n_Q$ sibling of the
single-quadrotor law of {py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`, built on the dynamics of
{doc}`../dynamics/multi-quad-cspointmass` and closed by the attitude loop
of {doc}`quadrotor-se3`.

Notation follows the dynamics page: $q_i \in S^2$ is the unit cable vector
oriented **from quadrotor $i$ to the payload**, $\ell_i$ its length,
$m_i$ the quadrotor mass, $m_L$ the payload mass, and
$\mathbf{T}_i = T_i q_i$ the cable force *on quadrotor $i$*.

## Why one extra stage

The single-quadrotor cascade has three levels: payload position, cable
attitude, vehicle attitude. Here a fourth sits between the first two,
and it is the only genuinely new design content. Newton's law on the
point-mass payload,

$$
\sum_{i=1}^{n_Q} \mathbf{T}_i \;=\; F_{\mathrm{net}}
    \;:=\; -m_L\,(a_L + g e_3),
$$ (eq-mqc-balance)

is three equations in $3 n_Q$ unknowns. For $n_Q = 1$ it *determines*
the cable force, which is why {py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController` can go straight
from the payload loop to $q_d = -A/\|A\|$. For $n_Q > 1$ it leaves a
$3(n_Q-1)$-dimensional space of internal, self-cancelling forces
undetermined, and something has to choose.

That choice is a **tension-split policy**
({py:class}`udaan.flatness.TensionSplitPolicy`), and it is the same
object the flat-to-state map of {doc}`../differential-flatness/multi-quad-cspayload`
uses. It is not a second, controller-private allocator: the closed loop
and the flatness feedforward *must* split $F_{\mathrm{net}}$ the same
way, or the reference attitude fed to the inner loop belongs to a
different formation than the one the outer loop is driving towards.

## Payload loop

Let $x_L^d(t)$ be the desired payload position and

$$
e_{x_L} = x_L - x_L^d, \qquad e_{v_L} = \dot x_L - \dot x_L^d .
$$

For positive gains $k_p^L, k_d^L$ the loop commands a payload
*acceleration*,

$$
a_L^d = \ddot x_L^d - k_p^L e_{x_L} - k_d^L e_{v_L},
$$ (eq-mqc-payload-pd)

and hence, through {math:numref}`eq-mqc-balance`, a desired net cable
force $F_{\mathrm{net}}^d = -m_L (a_L^d + g e_3)$.

:::{note}
The gains act in acceleration units, not force units as in
{py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`. A force-space PD normalised by the total mass
$m_L + \sum_i m_i$ would make the closed-loop bandwidth shrink as $n_Q$
grows — the same gains that are critically damped for one quadrotor are
sluggish for four. Commanding acceleration keeps the payload loop's
response $n_Q$-invariant, which is what "generic in $n_Q$" has to mean
for a controller whose whole point is that the vehicle count is a
parameter.

The invariance holds for $n_Q \ge 2$, and only there: across
$n_Q = 2 \dots 6$ a lateral step from hover has nearly the same rise time
and overshoot, while at $n_Q = 1$ the response is faster with markedly more
overshoot. A single vertical cable is a different geometry rather than the
one-vehicle case of the ring.
:::

## Distribution

The policy receives the *jet* $[F_{\mathrm{net}}^d, \dot F_{\mathrm{net}},
\ddot F_{\mathrm{net}}]$ — the higher entries being pure reference
feedforward $-m_L x_L^{(k+2)d}$ — and returns per-cable force jets
$[\mathbf{T}_i^d, \dot{\mathbf{T}}_i^d, \ddot{\mathbf{T}}_i^d]$
satisfying $\sum_i \mathbf{T}_i^{(k)} = F_{\mathrm{net}}^{(k)}$ at every
order. The shipped default,
{py:class}`udaan.flatness.RingTensionSplit`, anchors the quadrotors on a
horizontal circle above the payload and shares the *dynamic* load in
proportion to mass.

Normalising each force jet
({py:func}`udaan.flatness.tension_direction_jet`) gives the cable
reference $q_i^d, \dot q_i^d, \ddot q_i^d$. The derivatives come for
free precisely because the policy is differentiable in closed form; this
is why the abstraction is stated on jets rather than on point values.

## Cable loop

With $\omega_i^d = q_i^d \times \dot q_i^d$ and
$\dot\omega_i^d = q_i^d \times \ddot q_i^d$, the $S^2$ errors are those
of {py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`, applied per cable:

$$
e_{q_i} = \widehat{q}_i^{\,2} q_i^d, \qquad
e_{\dot q_i} = \dot q_i - (q_i^d \times \dot q_i^d) \times q_i .
$$

Writing the thrust vector of quadrotor $i$ as
$u_i = u_i^{\parallel} + u_i^{\perp}$ relative to $q_i$, and using
$x_{Q_i} = x_L - \ell_i q_i$ with
$q_i \cdot \ddot q_i = -\|\dot q_i\|^2$ in

$$
u_i = m_i (a_L + g e_3) - m_i \ell_i \ddot q_i - \mathbf{T}_i ,
$$

the two channels are

$$
u_i^{\parallel} = \min\bigl(A_i \cdot q_i,\; -T_i^{\min}\bigr)\, q_i, \qquad
A_i = m_i (a_L^d + g e_3) - \mathbf{T}_i^d
      + m_i \ell_i \|\dot q_i\|^2 q_i ,
$$ (eq-mqc-parallel)

Because $q_i$ points down the cable, $A_i \cdot q_i < 0$ is thrust up it.
The floor $T_i^{\min} = 0.1\,\|\bar{\mathbf{T}}_i\|$
(`MULTI_MIN_CABLE_TENSION_FRACTION` of the cable's hover tension under the
tension-split policy) keeps that component pointing up the cable when the
payload loop asks for a net downward force; without it the parallel channel
would ask the airframe to push down its cable and the commanded attitude
would turn over.

$$
u_i^{\perp} = \underbrace{-\,m_i \widehat{q}_i^{\,2}(a_L^d + g e_3)}_{\text{lateral share}}
  + m_i \ell_i \bigl(k_p^q e_{q_i} + k_d^q e_{\dot q_i}\bigr)
  - m_i \ell_i (q_i \cdot \omega_i^d)(q_i \times \dot q_i)
  - m_i \ell_i\,(\dot\omega_i^d \times q_i).
$$ (eq-mqc-perp)

### The lateral-share term

The first term of {math:numref}`eq-mqc-perp` has no counterpart in
{py:class}`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`, and its absence there is not an omission. With
one cable the *plant's* payload acceleration is always parallel to it —
the cable is the only thing pulling on the payload — so
$\widehat{q}^{\,2}(a_L + g e_3)$ vanishes at the reference and the term
is invisible there.

It does **not** vanish identically at $n_Q = 1$ for the law as
implemented: the projector is built from the *measured* $q$, not from
$q^d$, so the term is zero only where the cable-attitude error is, and it
grows with the cable's tilt off its reference. M2's law and this one are therefore genuinely different
at $n_Q = 1$ — this one is the more complete of the two — and the hover
identity below is unaffected either way.

With $n_Q > 1$ the cables are tilted off vertical by the formation
radius, $a_L + g e_3$ has a component across each of them, and each
quadrotor must supply the lateral force that holds its own weight
against a slanted cable. That missing force points *inwards*: on a ring
of radius $r$ with the quadrotors at height $\Delta z = \sqrt{\ell^2 - r^2}$
its horizontal part is $-m_i g\,(\Delta z\, r/\ell^2)(\cos\theta_i,
\sin\theta_i)$. Drop the term and each quadrotor therefore settles
**further out** until the cable-attitude PD's steady-state error supplies
the missing force by itself: the ring settles wider than the commanded
formation radius, with a corresponding payload position error.

:::{note}
The implementation deviates from {footcite}`sreenath2013cooperative`
here in one deliberate respect: their $u_i^\perp$ carries
$-\widehat q_i^{\,2} A_i$ in full, of which only the
$-m_i \widehat q_i^{\,2}(a_L^d + g e_3)$ half is kept. The dropped half is
$\widehat q_i^{\,2}\mathbf{T}_i^d = T_i^d e_{q_i}$ (the centripetal part
of $A_i$ is annihilated by $\widehat q_i^{\,2}$), an effective cable
stiffness lower by $T_i^d/(m_i \ell_i) \approx 0.84\ \mathrm{s^{-2}}$
against $k_p^q = \omega_c^2 \approx 39\ \mathrm{s^{-2}}$ at $\ell = 1$ m
and $n_Q = 3$ — about 2 %, and vanishing at zero cable error.
:::

The check is the hover condition: set $e = 0$, $q_i = \bar q_i$,
$\mathbf{T}_i = \bar{\mathbf{T}}_i$, $a_L^d = 0$. Then
{math:numref}`eq-mqc-parallel` and {math:numref}`eq-mqc-perp` sum to

$$
u_i = m_i g (e_3 \cdot q_i) q_i - \bar{\mathbf{T}}_i
      + m_i \bigl(g e_3 - (e_3 \cdot q_i) g q_i\bigr)
    = m_i g e_3 - \bar{\mathbf{T}}_i,
$$

which is exactly the equilibrium thrust — and only because the lateral
share is there.

### Tilt bound

The thrust vector $u_i = u_i^{\parallel} + u_i^{\perp}$ is what the attitude
loop turns into the commanded thrust axis $b_3^d = u_i/\|u_i\|$. A large
cable-direction error makes the perpendicular channel outgrow the parallel
one: released 30° off its hover direction, a cable's PD alone asks
$m_i \ell_i k_p^q \sin 30° = 2 m_i g$ sideways wherever
$\omega_c = 2\sqrt{g/\ell}$. The cascade therefore bounds the direction it
hands the attitude loop to $\theta_{\max}$ from vertical
(`MULTI_MAX_COMMANDED_TILT_DEG`, 50°, the controller's `max_tilt_deg`).
With $h_i = (u_{i,x}, u_{i,y})$ and $z_i = \max(u_{i,z}, T_i^{\min})$,

$$
u_i \leftarrow
\Bigl(\min\Bigl(1, \tfrac{z_i\tan\theta_{\max}}{\|h_i\|}\Bigr) h_i,\; z_i\Bigr).
$$

The vertical part is kept wherever it is at least the tension floor
$T_i^{\min}$, and the horizontal part is scaled down only where it leans
past $\theta_{\max}$. A command that points level or down becomes a small
upward one in its own horizontal direction: at the bound when its horizontal
part exceeds $T_i^{\min}\tan\theta_{\max}$, straight up when it has none.
The map is continuous in $u_i$.

## Attitude loop and flatness feedforward

Each $u_i$ is handed to the inner geometric loop of {prf:ref}`def-geom-att`:
$f_i = u_i \cdot R_i e_3$ and $R_i^d$ from $b_3^d = u_i/\|u_i\|$.

When the bound payload trajectory exposes higher derivatives
(`get_full(t) -> (pos, vel, acc, jerk, snap)`),
{py:meth}`udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.simulate`
additionally builds the flat map of {prf:ref}`thm-mq-flat` at the
controller's own tension-split policy and feeds the recovered
$(R_i^d, \Omega_i^d, \dot\Omega_i^d)$ into the attitude loop as feedforward.
{py:class}`~udaan.control.quadrotor.GeometricAttitudeController` keeps
$b_3^d$ from the commanded thrust vector it is given and takes from the
recovered $R_i^d$ only its heading, the $b_1$ direction about $b_3^d$; the
two rate terms enter as they are. Two things then improve at once: the
attitude loop stops lagging the reference, and the cable loop's
$\dot q_i^d, \ddot q_i^d$ become non-zero. Nothing in the controller
changes — growing derivatives on the trajectory is the whole switch.

Crackle and pop are taken as zero: trajectories in this package stop at
snap, and $x_L^{(5)}, x_L^{(6)}$ enter only the moment feedforward
$\dot\Omega_i^d$. The map refuses samples at which a cable would go
slack, and marks samples whose recovered airframes are not all upright
($b_3 \cdot e_3 \le 0$, {py:attr}`~udaan.flatness.MultiQuadCsPayloadRefState.upright`):
a cable-suspended airframe whirling around its payload can need a level or
inverted attitude although the payload's own net force points up. On
either kind of sample the feedforward is dropped for that step and the
feedback cascade carries on alone. Dropping is *not* silent — the total
is available as
{py:attr}`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.feedforward_drops`
and one warning at the end of the run gives the intervals, and
{py:meth}`~udaan.flatness.MultiQuadCsPayload.check_reference` finds them
before flight, so a reference outside the formation's reach can be
recognised rather than mistaken for a badly tuned loop. A policy that violates the split contract
$\sum_i \mathbf{T}_i^{(k)} = F_{\mathrm{net}}^{(k)}$ is a different
matter and raises
{py:class}`~udaan.flatness.TensionSplitContractError` instead of being
absorbed.

The feedforward is also skipped outright when the payload controller
exposes no `tension_policy`: without it there is no way to know how the
closed loop is splitting $F_{\mathrm{net}}$, and building the reference
at a default split would recreate exactly the mismatch this section
opened by ruling out.

## Gains and stability

Exponential stability of the cascade, under a separation of time scales
between the payload, cable and attitude loops, is established for the
cooperative system in {footcite}`sreenath2013cooperative`. The gains are a
time-scale design
({py:func}`~udaan.control.multi_quad_cspayload.multi_cable_cascade_gains`),
the one the single-quadrotor cascade uses, computed from the model's
parameters and recomputed when they change. Every loop has damping
$\zeta = 0.7$; the attitude, cable and vertical loops are separated by a
ratio $r = 5$, and the payload's lateral loop sits a ratio $r_p = 3.5$
(`MULTI_PAYLOAD_LOOP_RATIO`) below the slowest cable:

$$
\omega_a = 40\ \mathrm{rad/s}, \qquad
\omega_{c,i} = \min\bigl(2\sqrt{g/\ell_i},\ \omega_a / r\bigr), \qquad
\omega_p = \min_i \omega_{c,i} / r_p, \qquad
\omega_z = \omega_a / r^2 .
$$

The cable loop follows each cable's pendulum frequency, so its bandwidth
does not fall behind a long cable, and the payload loop sits below the
slowest cable. In this controller's acceleration units the payload gains
are $k_p^L = \omega^2$, $k_d^L = 2\zeta\omega$ ($\omega = \omega_p$ on
$x, y$ and $\omega_z$ on $z$) and the cable gains
$k_p^q = \omega_{c,i}^2$, $k_d^q = 2\zeta\omega_{c,i}$; the attitude
gains are moments, $k_R = J_{jj}\,\omega_a^2$ and
$k_\Omega = 2\zeta J_{jj}\,\omega_a$ on roll and pitch, with the
quadrotor's default yaw gains.

The taut-cable assumption is not free either: the policy's
{py:meth}`~udaan.flatness.TensionSplitPolicy.net_force_margin` and the
map's
{py:meth}`~udaan.flatness.MultiQuadCsPayload.payload_acceleration_bound`
give a sufficient bound on $\|a_L\|$ below which every cable is provably
taut, and a reference that violates it is refused rather than silently
tracked.

## References

```{footbibliography}
```
