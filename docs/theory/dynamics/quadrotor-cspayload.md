# Quadrotor with cable-suspended payload

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

We consider a point-mass payload $x_L \in \mathbb{R}^3$ attached to the
quadrotor through a taut, massless, inextensible cable of length $\ell$.
Let $q \in S^2$ be the unit vector pointing from the quadrotor to the
payload, so that

$$
x_L = x_Q - \ell\, q, \qquad \|q\| = 1.
$$

The configuration manifold is $Q = \mathbb{R}^3 \times S^2 \times SO(3)$.

## Equations of motion

Following {footcite}`sreenath2013geometric`, the Lagrangian derivation yields

$$
\begin{aligned}
(m_Q + m_L)\, \ddot{x}_L &=
  \left(q \cdot f R e_3 - m_Q \ell\, \dot{q}\cdot\dot{q}\right) q
  - (m_Q + m_L)\, g e_3, \\
m_Q \ell\, \ddot{q} &=
  -q \times \left(q \times f R e_3\right)
  - m_Q \ell\, (\dot{q}\cdot\dot{q})\, q, \\
\dot{R} &= R\,\widehat{\Omega}, \\
J\dot{\Omega} + \Omega \times J\Omega &= M.
\end{aligned}
$$

Here $\omega \in \mathbb{R}^3$ with $\omega \cdot q = 0$ is the cable
angular velocity and $\dot{q} = \omega \times q$.

The flat-output structure of this system is summarised in
{doc}`../differential-flatness/quadrotor-cspayload`.

## Taut-cable assumption

The derivation above assumes $\|x_Q - x_L\| = \ell$ at all times. In
practice the assumption fails when the payload swings above the quadrotor
(the cable goes slack). The links-cable MuJoCo backend
({py:class}`udaan.models.quadrotor_cspayload.QuadrotorCsPayloadMujoco`
with `cable_model="links"`) models the cable as a rigid chain to capture
slack-cable behaviour; the tendon backend uses a one-sided spatial tendon
constraint.

## References

```{footbibliography}
```
