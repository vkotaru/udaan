# Hose LQR for a pinned hose

The controller of M7 and M8 is M6's hose LQR ({doc}`multi-quad-hose`) on the
pinned equations of motion,
{py:class}`~udaan.control.tethered_hose.TetheredHoseController`:

$$
(f_k, M_k) = (f_k^d, M_k^d) + \bigl[K\,T_\theta\,s\bigr]_k,
\qquad
s = \bigl[\{\xi_i\}, \{\delta\omega_i\}, \{\eta_k\}, \{\delta\Omega_k\}\bigr].
$$

## Error state, linearisation and cost

The error blocks are M6's without the root's position and velocity, which the
anchor fixes: $\xi_i = q_i^d \times q_i$, the link rate error $\delta\omega_i$,
the attitude error $\eta_k = \tfrac12 (R_k^{d\top}R_k - R_k^\top R_k^d)^\vee$
and the body-rate error $\delta\Omega_k$. The linearisation is M6's
variational one on the reduced system {eq}`eq-m8-system`, with
$M_{0i}\hat q_i\alpha$ on the right-hand side, linearised at
$\alpha_d = g e_3$. The Riccati equation is solved on the tangent-reduced
pair at the static trim, with M6's cost weights for the link, link-rate,
attitude, body-rate, thrust and moment blocks.

At the default M7 machine (2 m, 0.25 kg/m, four links, vehicle at
$0.5L$ out and $0.8L$ up) the slowest closed-loop mode is
$-0.60 \pm 12.7j$ (a hose-bending mode, damping ratio 0.048) and the
controllability margin is 0.098, above the free hose's 0.084 at the same
pattern. Spreading vehicles along the hose raises the damping: M8 at two
links per segment has a damping ratio near 0.2.

## Carrying the gain round the anchor

The trim depends on where the formation stands round the anchor, so a gain
solved at one azimuth and held in world coordinates is wrong elsewhere; at
$180°$ it is unstable on the default M7 machine. The system is symmetric
under a rotation $E = R_z(\Delta\theta)$ about the anchor's vertical axis,
except that every vehicle keeps $\psi \equiv 0$: the rotated trim's airframe
is $E R_k^0$, the reference flown is $R_k^d$, and the two share a thrust axis
and differ by

$$
Y_k = R_k^{d\top} E\, R_k^0 ,
$$

a rotation about the body $z$ axis. The gain solved at azimuth $\theta_0$ is
applied at $\theta_0 + \Delta\theta$ to the error rotated back into its own
frame: each link block by $E^\top$, each vehicle's $\eta_k$ and
$\delta\Omega_k$ by $Y_k^\top$, and each moment it returns by $Y_k$. The map
is exact where $Y_k$ commutes with the airframe inertia (at $180°$) and close
elsewhere, the difference coming from $J_x \ne J_y$.

One gain serves a whole flight. It is solved at the static trim at the
controller's `setpoint(0)`, and assigning a new setpoint re-solves it there;
the transport accounts for a change of azimuth only, so a climb, a different
standoff or height, and the whirl part of a circle are flown on the start
trim's gain.
