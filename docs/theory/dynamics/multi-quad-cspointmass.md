# Multi-quadrotor with cable-suspended point-mass payload

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

:::{note}
This page is a stub. Content pending — see
{py:class}`~udaan.models.mujoco.MultiQuadrotorCSPointmass` for the
corresponding simulation model.
:::

A team of $N$ quadrotors cooperatively transports a single point-mass
payload, each attached through its own taut cable. The configuration
manifold is

$$
Q = \mathbb{R}^3 \times \left(SO(3) \times S^2\right)^N,
$$

with one $SO(3)$ attitude and one $S^2$ cable direction per quadrotor, plus
the shared payload position $x_L \in \mathbb{R}^3$.

## To be written

- Full Lagrangian derivation under the taut-cable assumption for all $N$
  cables simultaneously.
- The cable-tension allocation problem and its rank deficiency.
- Relationship to the single-quadrotor case of
  {doc}`quadrotor-cspayload`.
