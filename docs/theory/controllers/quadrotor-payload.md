# Payload-tracking geometric control

The controller used by
{py:class}`udaan.control.quadrotor_cspayload.QuadCSPayloadController`
is a cascaded geometric design built on the taut-cable model of
{doc}`../dynamics/quadrotor-cspayload` and the attitude loop of
{doc}`quadrotor-se3`.

## Payload-position loop

Let $x_L^d(t)$ be the desired payload position with bounded first and
second time derivatives $\dot{x}_L^d,\, \ddot{x}_L^d$. Define

$$
e_{x_L} = x_L - x_L^d, \qquad e_{v_L} = \dot{x}_L - \dot{x}_L^d.
$$

For positive gains $k_p^L, k_d^L$, the desired net force applied to the
payload through the cable is

$$
A = -k_p^L\, e_{x_L} - k_d^L\, e_{v_L}
  + (m_Q + m_L)\left(\ddot{x}_L^d + g e_3\right)
  + m_Q \ell\,(\dot{q}\cdot\dot{q})\, q.
$$

Projecting $A$ onto $-q$ gives the thrust magnitude along the cable; the
component orthogonal to $q$ is delivered via the cable-attitude loop.

## Cable-attitude loop

The desired cable direction is

$$
q_d = -\frac{A}{\|A\|}
$$

with associated desired rate $\dot{q}_d, \ddot{q}_d$ (available from the
flatness derivation in {prf:ref}`thm-payload-flat`). Define the
$S^2$ errors

$$
e_q = \widehat{q}^{\,2}\, q_d, \qquad
e_\omega = \omega - (q_d \times \dot{q}_d) \times q,
$$

and for positive gains $k_p^q, k_d^q$ form the desired thrust force

$$
F = A\cdot q\, q - k_p^q\, e_q - k_d^q\, e_\omega
  + m_Q \ell\,\bigl(q \cdot (q_d \times \dot{q}_d)\bigr)(q \times \dot{q})
  + m_Q \ell\,(q_d \times \ddot{q}_d) \times q.
$$

## Attitude loop and collective thrust

The scalar thrust $f = F \cdot R e_3$ and the desired body attitude
$R_d = [b_1^d\; b_2^d\; b_3^d]$ with $b_3^d = F/\|F\|$ are passed to the
inner geometric attitude loop of {prf:ref}`def-geom-att`.

## Stability

Exponential stability of the cascaded system, under an appropriate
separation of time scales between the payload, cable, and attitude loops,
is established in {footcite}`sreenath2013geometric`. In practice the
time-scale separation is enforced by the gain hierarchy
$k_p^L \ll k_p^q \ll k_R$; values that satisfy this ordering
are the defaults shipped with the package.


## References

```{footbibliography}
```
