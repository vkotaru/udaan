# Quadrotor dynamics

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

## Equations of motion

Let $x_Q \in \mathbb{R}^3$ be the quadrotor centre-of-mass position and
$R \in SO(3)$ its orientation (body frame to world). With mass $m_Q$,
inertia $J$, collective thrust $f$, body moment $M$, and
gravity $g$, the standard rigid-body equations of motion are

$$
\begin{aligned}
\dot{x}_Q &= v_Q, \\
m_Q \dot{v}_Q &= f\, R e_3 - m_Q g e_3, \\
\dot{R} &= R\, \widehat{\Omega}, \\
J \dot{\Omega} + \Omega \times J\Omega &= M.
\end{aligned}
$$

Here $\widehat{\cdot} : \mathbb{R}^3 \to \mathfrak{so}(3)$ is the hat map;
see {footcite}`murray2017mathematical` for the mechanical-systems formalism.

## Properties

```{prf:property} Configuration manifold
:label: prop-quad-config

The configuration manifold of the quadrotor is $Q = \mathbb{R}^3 \times SO(3)$
with the natural trivialisation of its tangent bundle
$TQ = \mathbb{R}^3 \times \mathbb{R}^3 \times SO(3) \times \mathbb{R}^3$.
```

```{prf:property} Underactuation
:label: prop-quad-underact

The quadrotor has four control inputs $(f, M_1, M_2, M_3)$ but six
position-and-orientation degrees of freedom. It is therefore underactuated
with underactuation degree two — translational acceleration is constrained
to the thrust axis $R e_3$.
```

## Discretisation in the simulator

The base dynamics in {py:class}`udaan.models.quadrotor.QuadrotorBase` are
integrated with a zero-order-hold Euler scheme at a $\Delta t = 5\,\text{ms}$
outer step with four inner substeps (effective $h = 1.25\,\text{ms}$).
See the source of `QuadrotorBase._zoh()` for the exact update rules.


## References

```{footbibliography}
```
