# Geometric control on SE(3)

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

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


## References

```{footbibliography}
```
