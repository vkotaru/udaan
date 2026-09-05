# Quadrotor dynamics

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

## From command to wrench

The equations above take $(f, M)$, but no controller in the package
produces that pair directly — the position loop produces a thrust
*vector* and the propeller model produces four scalar forces. Every
command therefore passes through one funnel,
`QuadrotorBase._repackage_input()`, before it reaches the integrator, and
which path it takes is set by the model's
{py:class}`~udaan.core.types.InputType`:

| `InputType` | Command | Conversion |
|-------------|---------|------------|
| `ACCELERATION` | thrust force vector $F \in \mathbb{R}^3$ | attitude controller: $f = F \cdot R e_3$, $M$ from {prf:ref}`def-geom-att` |
| `WRENCH` | $[f, M_1, M_2, M_3]$ | used as-is |
| `PROP_FORCES` | $[f_1, \dots, f_4]$ | $\;[f, M]^\top = A\,[f_1 \dots f_4]^\top$ |

The allocation matrix $A$ encodes the $\times$-configuration rotor
geometry — arm length $d$, rotor angles at odd multiples of $\pi/4$, and
alternating spin directions — with the torque-to-thrust ratio
$c = k_m/k_f$ in its last row:

$$
A = \begin{bmatrix}
1 & 1 & 1 & 1 \\
d\sin\theta_1 & \cdots & \cdots & d\sin\theta_4 \\
-d\cos\theta_1 & \cdots & \cdots & -d\cos\theta_4 \\
-c & c & -c & c
\end{bmatrix},
\qquad
\theta_i = \tfrac{\pi}{4}(2i-1).
$$

The resulting wrench is then **clipped to the actuator limits** — thrust
into $[0, 20]\,\text{N}$, moments into $\pm[5, 5, 2]\,\text{N·m}$ — and,
when the model's {py:class}`~udaan.core.types.ForceType` is
`PROP_FORCES`, round-tripped through $A^{+}$, clipped again per rotor
into $[0, 10]\,\text{N}$, and mapped back. That second round trip is what
makes a physically unrealisable wrench — one that would need a negative
rotor thrust — visible in the simulation instead of silently granted.
The limits live in {py:mod}`udaan.core.defaults` as
{py:data}`~udaan.core.defaults.DEFAULT_MIN_THRUST` /
{py:data}`~udaan.core.defaults.DEFAULT_MAX_THRUST`,
{py:data}`~udaan.core.defaults.DEFAULT_MIN_TORQUE` /
{py:data}`~udaan.core.defaults.DEFAULT_MAX_TORQUE` and
{py:data}`~udaan.core.defaults.DEFAULT_PROP_MIN_FORCE` /
{py:data}`~udaan.core.defaults.DEFAULT_PROP_MAX_FORCE`, and every model's
constructor reads them from there; a new model should hook into this funnel
rather than writing $(f, M)$ straight to the integrator.

## Discretisation in the simulator

The base dynamics in {py:class}`udaan.models.quadrotor.QuadrotorBase` are
integrated with a zero-order-hold Euler scheme at a $\Delta t = 5\,\text{ms}$
outer step with four inner substeps (effective $h = 1.25\,\text{ms}$).
The repackaged wrench is computed once per outer step and held across all
four substeps; within a substep the translation is advanced by
$x_Q \leftarrow x_Q + v_Q h + \tfrac12 a h^2$, $v_Q \leftarrow v_Q + a h$
at frozen acceleration, and the attitude by the exponential map
$R \leftarrow R\,\exp(\widehat{\Omega}\,h)$, so $R$ stays on $SO(3)$ to
machine precision rather than drifting off it and needing
re-orthonormalisation.

{py:class}`~udaan.models.quadrotor.QuadrotorMujoco` and the vfx backend
subclass this model and override only `step()`, `reset()` and
`_query_latest_state()`; the input funnel, the controllers and
`simulate()` are inherited unchanged, which is what keeps a controller
written against the analytical model working against the physics engine.

## References

```{footbibliography}
```
