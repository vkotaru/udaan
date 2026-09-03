# Geometric L1-adaptive control

:::{warning}
**Under construction.** This page is a draft; content and notation
may change and cross-references may be incomplete.
:::

:::{note}
This page is a stub. Content pending — see
{py:class}`~udaan.control.quadrotor.GeometricL1AttitudeController` and
{py:class}`~udaan.control.quadrotor.PositionL1Controller` for the
corresponding implementations.
:::

The $L_1$ adaptive augmentation adds a bandwidth-limited disturbance
estimator and feed-forward compensator to the geometric SE(3) baseline
of {doc}`quadrotor-se3`. It is designed to recover nominal performance
under unmodelled mass, inertia, or external-force disturbances without
amplifying high-frequency noise.

## To be written

- State-predictor and adaptation-law derivation on $SO(3)$ and
  $\mathbb{R}^3$.
- Choice of the low-pass filter bandwidth and its coupling to the
  nominal attitude gains.
- Stability argument and quiescent-response guarantee.
- Recommended gains and tuning recipe for the implementation shipped
  with `udaan`.
