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
  -q \times \left(q \times f R e_3\right)
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
choose the split. See {doc}`multi-quad-cspointmass`.

The flat-output structure of this system is summarised in
{doc}`../differential-flatness/quadrotor-cspayload`, and the cascaded
tracking law built on it in {doc}`../controllers/quadrotor-payload`.

## Taut-cable assumption

The derivation above assumes $\|x_Q - x_L\| = \ell$ at all times, and it
has no slack mode: {py:class}`udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase`
will happily drive the cable into compression. In practice the
assumption fails when the payload swings above the quadrotor.

{py:class}`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadMujoco`
therefore offers three cable models, selected with `cable_model=`:

| `cable_model` | Representation | Notes |
|---------------|----------------|-------|
| `links` (default) | rigid link chain with ball joints | most stable; captures slack |
| `tendon` | one-sided spatial tendon with an upper length limit | fastest; carries no force when slack |
| `cable` | MuJoCo composite cable | experimental — warns on construction, and its reset path is still a stub |

Because `links` is a genuine $N$-link chain it does not reproduce the
$S^2$ model above exactly; it reports a single equivalent
$(q, \omega)$ to keep the state interface common with the base model.
The flexible-cable system where that chain *is* the model, rather than an
approximation of one, is {doc}`quadrotor-flexcable`.

## Discretisation in the simulator

The base model uses the same zero-order-hold Euler scheme as
{doc}`quadrotor`: a $\Delta t = 5\,\text{ms}$ outer step, four
$h = 1.25\,\text{ms}$ inner substeps, the wrench held across all four.
Per substep the payload state is advanced at frozen acceleration, the
cable through the tangency-preserving `manif.zoh_step`, the attitude by
the exponential map, and the quadrotor position and velocity are then
derived from the cable geometry.

## References

```{footbibliography}
```
