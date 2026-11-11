# Quadrotor with cable-suspended payload

We consider a point-mass payload $x_L \in \mathbb{R}^3$ attached to the
quadrotor through a taut, massless, inextensible cable of length $\ell$.
Let $q \in S^2$ be the unit vector pointing from the quadrotor to the
payload, so that

$$
x_L = x_Q + \ell\, q, \qquad \|q\| = 1.
$$

The configuration manifold is $Q = \mathbb{R}^3 \times S^2 \times SO(3)$,
and the state space $\mathbb{R}^3 \times \mathbb{R}^3 \times TS^2 \times
TSO(3)$ has dimension 16. The quadrotor's own position is a *derived*
quantity, not an independent state — the simulator recomputes
$x_Q = x_L - \ell q$ and $v_Q = v_L - \ell \dot q$ after every substep,
so the length constraint holds exactly rather than to integrator
accuracy.

## Equations of motion

Following {footcite}`sreenath2013geometric`, the Lagrangian derivation yields

$$
\begin{aligned}
(m_Q + m_L)\, \ddot{x}_L &=
  \left(q \cdot f R e_3 - m_Q \ell\, \dot{q}\cdot\dot{q}\right) q
  - (m_Q + m_L)\, g e_3, \\
m_Q \ell\, \ddot{q} &=
  q \times \left(q \times f R e_3\right)
  - m_Q \ell\, (\dot{q}\cdot\dot{q})\, q, \\
\dot{R} &= R\,\widehat{\Omega}, \\
J\dot{\Omega} + \Omega \times J\Omega &= M.
\end{aligned}
$$

Here $\omega \in \mathbb{R}^3$ with $\omega \cdot q = 0$ is the cable
angular velocity and $\dot{q} = \omega \times q$. The first line splits
the thrust along the cable and the second across it: only $q \cdot fRe_3$
accelerates the combined mass, and only $q \times (q \times fRe_3)$ swings
the cable. Note that the centripetal coefficient is
$\dot q \cdot \dot q$, from the exact identity $q \cdot \ddot q =
-\|\dot q\|^2$; it coincides with $\|\omega\|^2$ only while $\omega$ is
exactly tangent to the sphere, and a discretisation that lets
$q \cdot \omega$ drift makes the two forms disagree.

## Cable tension

The tension is not a state — with one cable it is *determined* by
Newton's law on the payload,

$$
\mathbf{T} = T q, \qquad T q = -m_L\bigl(\ddot x_L + g e_3\bigr),
$$

$\mathbf{T}$ being the force the cable applies to the quadrotor. This is
what makes the single-quadrotor case special: with $n_Q > 1$ cables the
same balance is three equations in $3 n_Q$ unknowns and something has to
choose the split.

The flat-output structure of this system is summarised in
{doc}`../differential-flatness/quadrotor-cspayload`, and the cascaded
tracking law built on it in {doc}`../controllers/quadrotor-payload`.

## Taut-cable assumption

The derivation above assumes $\|x_Q - x_L\| = \ell$ at all times, and it
has no slack mode: {py:class}`udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase`
will happily drive the cable into compression. In practice the
assumption fails when the payload swings above the quadrotor.

{py:class}`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadMujoco`
therefore offers two cable models, selected with `cable_model=`:

| `cable_model` | Representation | Notes |
|---------------|----------------|-------|
| `tendon` (default) | massless one-sided spatial tendon with an upper length limit $\ell$ | carries no force when slack; its limit's time constant is two MuJoCo timesteps, so a taut cable holds its length |
| `links` | 10-link chain of ball joints, 10 g of cable mass | the cable's own mass and shape; `cable_mass` reports the 10 g, which the payload controller carries as part of $m_L$ |

Because `links` is a genuine link chain it does not reproduce the
$S^2$ model above exactly; it reports a single equivalent
$(q, \omega)$ to keep the state interface common with the base model:
$q$ from the airframe to the payload, the payload's velocity as MuJoCo
computes it for the payload body, and $\omega = q \times (v_L - v_Q)/\ell$.
Each link's ball joint sits at its proximal end and the payload is welded
to the last link's distal end, so the chain hangs the payload exactly
$\ell = 1$ m below the airframe. Its `reset()` places the airframe and
turns the straight chain about its first ball joint to `cable_attitude`
at `cable_angular_velocity`; the payload's velocity then follows, so
`payload_velocity` is refused.

On the tendon and the link chain the MJCF is the source of the machine:
$m_Q$, the full inertia tensor $J$, $m_L$ and $\ell$ are read from it, and
both describe the analytical model's defaults — the M1 airframe
($m_Q = 0.75$ kg, $J = \mathrm{diag}(0.0053, 0.0049, 0.0098)$ kg·m²) with
$m_L = 0.15$ kg on $\ell = 1$ m.

## Discretisation in the simulator

The base model uses a zero-order-hold Euler scheme at three rates: a
$\Delta t = 5\,\text{ms}$ outer step (the 200 Hz payload loop),
$N = $ `physics_rate_hz` $\cdot\,\Delta t$ substeps of $h = \Delta t/N$
(eight of 0.625 ms at the default 1600 Hz), and the attitude loop at
`attitude_rate_hz` (800 Hz by default, each wrench held over two
substeps). Per substep the payload state is advanced at frozen
acceleration, the cable through the tangency-preserving `manif.zoh_step`,
the attitude by the exponential map, and the quadrotor position and
velocity are then derived from the cable geometry. The MuJoCo backend
writes $h$ into MuJoCo's timestep, so a step advances exactly $\Delta t$
on both backends, and both clocks count substeps.

## References

```{footbibliography}
```
