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

Projecting $A$ onto $-q$ gives the cable tension the payload loop asks
for, $T = -A\cdot q$ ($q$ points from the quadrotor to the payload, so
$T = \|A\|$ at hover); the component orthogonal to $q$ is delivered via
the cable-attitude loop.

The controller is built with the machine's $m_Q$, $m_L$, $\ell$ and $J$:
a model's default controller takes them from the model (on the MuJoCo
backend, from the MJCF), and `sync_from_model(model)` re-reads them. A cable
with mass of its own is carried as part of the payload: the controller's
$m_L$ is the model's `payload_mass` plus its `cable_mass` (0 on the analytical
model and the tendon, the chain's 10 g on `links`), so the hover feed-forward
$(m_Q + m_L)\,g$ is the machine's whole weight. The payload loop has no
integral action, so a weight the controller does not carry would stand off as
$m g / k_{x,z}$.

## Cable-attitude loop

The desired cable direction is

$$
q_d = -\frac{A}{\|A\|}
$$

with associated desired rate $\dot{q}_d, \ddot{q}_d$, taken from the
flatness derivation in {prf:ref}`thm-payload-flat`: the derivatives of the
reference's own cable direction $-(\ddot x_L^d + g e_3)/\|\cdot\|$, from
the setpoint's jerk and snap (zero when the setpoint returns only position,
velocity and acceleration). Define the
$S^2$ errors

$$
e_q = \widehat{q}^{\,2}\, q_d, \qquad
e_\omega = \omega - (q_d \times \dot{q}_d) \times q,
$$

and for positive gains $k_p^q, k_d^q$ form the desired thrust force

$$
F = -\max\!\left(-A\cdot q,\; T_{\min}\right) q - k_p^q\, e_q - k_d^q\, e_\omega
  + m_Q \ell\,\bigl(q \cdot (q_d \times \dot{q}_d)\bigr)(q \times \dot{q})
  + m_Q \ell\,(q_d \times \ddot{q}_d) \times q.
$$

The first term is the tension along the cable, floored at
$T_{\min} = 0.1\, m_L g$
({py:data}`~udaan.core.defaults.MIN_CABLE_TENSION_FRACTION`). Without the
floor it is $(A\cdot q)\,q$, which points *down* the cable whenever the
payload loop asks for a net downward force ($A\cdot q > 0$, a large
downward step does it); the attitude loop would then be asked for an
upside-down $R_d$. With it the commanded thrust always has a component up
the cable, the cable stays taut and $R_d$ stays upright.

## Attitude loop and collective thrust

The scalar thrust $f = F \cdot R e_3$ and the desired body attitude
$R_d = [b_1^d\; b_2^d\; b_3^d]$ with $b_3^d = F/\|F\|$ are passed to the
inner geometric attitude loop of {prf:ref}`def-geom-att`, which closes at
`attitude_rate_hz` (800 Hz by default) inside the 200 Hz payload loop, on
every input path.

**Attitude feed-forward.** When the bound payload trajectory exposes
`get_jet(t, order)`, `simulate()` reads the payload position through its
sixth derivative, recovers $(R_{ref}, \Omega_{ref}, \dot\Omega_{ref})$ from
the flat map of {doc}`../differential-flatness/quadrotor-cspayload` (built
with the model's own parameters, and the model's `yaw_setpoint` as $\psi$)
and passes it to the attitude loop. As for the bare quadrotor, the two
commands meet on orthogonal degrees of freedom: $b_3^d$ stays the thrust
command's, $R_{ref}$'s $b_1$ is the heading hint, and $\Omega_{ref}$,
$\dot\Omega_{ref}$ are the rate feed-forward. A trajectory without
`get_jet` flies with $\Omega_d = \dot\Omega_d = 0$ and heading $\psi = 0$.

**Reference feasibility.** The flat map returns an attitude for any non-zero
thrust vector, including one pointing down. The airframe whirls around the
payload, and the thrust it needs along the cable is
$(m_Q+m_L)(\ddot x_L + g e_3)\cdot(-q) - m_Q \ell \|\dot q\|^2$, so a fast loop
can need an inverted airframe although the payload's own net force points up.
Each recovered sample is flagged `upright` when $b_3 \cdot e_3 > 0$;
`simulate()` flies a sample that is not upright (or that the map refuses)
without the attitude feed-forward, counts it in `feedforward_drops` and logs
one warning with the intervals, and `QuadrotorCsPayload.check_reference`
finds them before flight.

## Gains: a time-scale design

Exponential stability of the cascaded system, under an appropriate
separation of time scales between the payload, cable, and attitude loops,
is established in {footcite}`sreenath2013geometric`. The default gains place
each loop, linearised about hover with its inner loop taken as instantaneous,
at its own natural frequency and damping:

$$
M\ddot e_x + k_v \dot e_x + k_x e_x = 0, \qquad
m_Q \ell\, \ddot e_q + k_\omega \dot e_q + k_q e_q = 0, \qquad
J \ddot e_R + k_\Omega \dot e_R + k_R e_R = 0,
$$

with $M = m_Q + m_L$, so that

$$
k_x = M\omega_p^2,\; k_v = 2\zeta M\omega_p, \qquad
k_q = m_Q\ell\,\omega_c^2,\; k_\omega = 2\zeta m_Q\ell\,\omega_c, \qquad
k_R = J\omega_a^2,\; k_\Omega = 2\zeta J\omega_a .
$$

The frequencies are separated by a ratio $r$, so each inner loop settles on
the time scale its outer loop acts on, and the cable loop follows the
pendulum frequency $\sqrt{g/\ell}$:

$$
\omega_a = 40\ \mathrm{rad/s}, \quad
\omega_c = \min\!\left(c\sqrt{g/\ell},\ \omega_a/r\right), \quad
\omega_p = \omega_c/r, \quad
\omega_z = \omega_a/r^2,
$$

with $r = 5$, $c = 2$ and $\zeta = 0.7$ on every loop ($\omega_z$ is the
payload's vertical loop, which carries no pendulum; yaw keeps the quadrotor's
own gains). A cable gain that is a fixed force would give a cable bandwidth
$\sqrt{k_q/(m_Q\ell)}$ that falls with the cable length; computing the gains
from $m_Q$, $m_L$, $\ell$ and $J$ keeps the loops separated and damped at every
length, and on a MuJoCo backend from the MJCF's machine.
`cable_cascade_gains` computes them, and `sync_from_model` recomputes them.

Fixed gains can be passed instead through the controller's `position_kp`,
`position_kd`, `cable_kp` and `cable_kd` keywords, and `sync_from_model` keeps
them. `core/defaults.py` holds one such set, `DEFAULT_PAYLOAD_POS_KP`,
`DEFAULT_PAYLOAD_POS_KD`, `DEFAULT_PAYLOAD_CABLE_KP` and
`DEFAULT_PAYLOAD_CABLE_KD` ($k_x = (4, 4, 8)$, $k_v = (3, 3, 6)$, $k_q = 24$,
$k_\omega = 8$): with the model's own attitude gains, the default machine's
closed loop at hover is stable with them for $0.25 \le \ell \le 1.76$ m.


## References

```{footbibliography}
```
