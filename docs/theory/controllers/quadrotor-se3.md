# Geometric control on SE(3)

The default attitude controller in this package is the geometric SE(3)
controller of {footcite}`lee2010geometric`. We reproduce its key statements below
for reference.

## Attitude error

Given a desired orientation $R_d(t) \in SO(3)$ and desired body-rate
$\Omega_d(t) \in \mathbb{R}^3$, define the attitude and angular-velocity errors

$$
\begin{aligned}
e_R &= \tfrac{1}{2}\left(R_d^\top R - R^\top R_d\right)^\vee, \\
e_\Omega &= \Omega - R^\top R_d \Omega_d,
\end{aligned}
$$

where $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$ is the vee map.

## Control law

```{prf:definition} Geometric attitude control law
:label: def-geom-att

For positive-definite diagonal gains $k_R, k_\Omega \succ 0$, the control
moment is

$$
M = -k_R\, e_R - k_\Omega\, e_\Omega + \Omega \times J\Omega
   - J\!\left(\widehat{\Omega} R^\top R_d \Omega_d - R^\top R_d \dot{\Omega}_d\right).
$$
```

## Stability

```{prf:theorem} Almost-global exponential stability
:label: thm-geom-att-stability

Consider the closed loop formed by the quadrotor attitude dynamics and the
control law of {prf:ref}`def-geom-att`. Suppose the initial attitude-error
configuration $e_R(0)$ satisfies $\Psi(R(0), R_d(0)) < 2$, where $\Psi$ is
the logarithmic attitude error. Then the zero equilibrium
$(e_R, e_\Omega) = (0, 0)$ is exponentially stable and the region of
attraction covers almost all of $SO(3) \times \mathbb{R}^3$.
```

```{prf:proof}
A Lyapunov function of the form

$$
V(e_R, e_\Omega) = \tfrac{1}{2} e_\Omega^\top J e_\Omega + k_R\, \Psi(R, R_d)
$$

is shown in {footcite}`lee2010geometric` to be positive definite on the
specified sublevel set and to decrease along trajectories. Exponential
bounds on $V$ give the stated convergence. See the reference for the
detailed derivation.
```

## Position control

For trajectory tracking, the outer position loop produces a desired thrust
vector $F_d$, from which the inner loop extracts a scalar thrust
$f = F_d \cdot R e_3$ and a desired body attitude $R_d$ whose third column
aligns with $F_d$. See {footcite}`mellinger2011minimum` for the minimum-snap
feed-forward trajectory generation that accompanies this controller.

## Recovering $R_d$ from a thrust command

$b_3^d = F_d/\|F_d\|$ fixes only two of the three attitude degrees of
freedom; the heading of $b_1^d$ about $b_3^d$ is free, and something has
to supply it. `udaan` resolves it from a **hint** vector:

* With no flatness reference, the hint is the world $x$ axis — that is
  heading-hold at $\psi = 0$.
* With a reference attitude from
  {doc}`../differential-flatness/quadrotor`, the hint is the
  reference's own $b_1$. The reference then contributes *only* its
  heading, while $b_3^d$ still comes from the position loop's force
  command.

The second point is the design decision worth stating explicitly,
because the obvious alternative — using the recovered $R_d$ *whole* —
overwrites $b_3^d$ as well and thereby severs the position loop from the
attitude loop. Composing on the heading alone keeps position feedback
intact and costs nothing when the reference does not yaw, where the hint
it supplies is close to the world-$x$ fallback anyway. The
{py:meth}`~udaan.control.quadrotor.GeometricAttitudeController.compute`
docstring carries the measured tracking numbers on both sides of that
choice.

The construction divides by $\|b_3^d \times \text{hint}\|$, so the hint
must stay clear of $b_3^d$ — and unlike the fixed world axis, a
reference's $b_1$ tilts with the reference and can swing onto it, at
which point $R_d$ jumps discontinuously and the jump lands straight in
$e_R$. Below
{py:data}`~udaan.core.defaults.MIN_ATTITUDE_HINT_CONDITIONING` the hint
is therefore blended, in the plane orthogonal to $b_3^d$, towards the
reference frame carried onto $b_3^d$ by the minimal rotation — a vector
orthogonal to $b_3^d$ by construction, hence perfectly conditioned. The
blend weight is the conditioning over the threshold, so it is continuous
at the threshold and complete at the singularity.

## Adaptive augmentation

{doc}`geometric-l1` adds an $L_1$ disturbance estimator and compensator
on top of this law, in both the attitude and the position loop, without
changing the interface: the augmented controllers subclass the ones
above and collapse onto them exactly when there is no disturbance to
estimate.

## References

```{footbibliography}
```
