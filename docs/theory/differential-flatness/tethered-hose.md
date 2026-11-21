# Pinned hose: flat output and references

{py:class}`~udaan.flatness.TetheredHose`, the flat map of M7 and M8.

## The flat output

$$
y = \bigl(\tau_0,\ \{\tau_a : a \text{ a vehicle vertex},\ a \ne n\},\ \{\psi_k\}\bigr),
$$

the force the anchor link exerts on the anchor vertex, M6's vector tension
in the link leaving every vehicle but the top one, and the yaws: $3 + 3(n_Q -
1) + n_Q = 4 n_Q$ components, the input count. Pinning vertex 0 removes the
root position from M6's flat output, and the anchor's reaction absorbs vertex
0's force balance $\bar m_0 g e_3 = \lambda + \tau_0$, which frees $\tau_0$.

That is M6's flat output for a hose with one more vehicle, at vertex 0, whose
"thrust" is the anchor reaction. The map is therefore M6's map with
attachments $(0, a_1, \dots, a_{n_Q})$, a placeholder vehicle at the anchor
and a constant root jet; nothing is re-derived. $\tau_0$ is needed through
its $(2n + 2)$-th derivative.

The vehicle position is **not** a flat output. Holding the top vehicle still
leaves the hose's pinned-pinned modes as undamped zero dynamics (0.74 Hz on
the default 2 m tether), so a trajectory is planned in the tensions and the
vehicle paths come out of the map.

## References

All three place the vehicles evenly on the line from the anchor to the top
vehicle.

{py:class}`~udaan.flatness.TetherHold`
: the static trim, every flat tension constant.

{py:class}`~udaan.flatness.TetherBlend`
: from one static trim to another, every flat tension blended by one smooth
  step $s(t)$, so the vehicle paths are outputs of the blend.

{py:class}`~udaan.flatness.TetherCircle`
: a circle about the anchor, started from rest. The formation turns about
  the anchor's vertical axis at $\Omega(t) = \Omega\, s(t)$, and in the turning
  frame each flat tension blends from the static trim to the steady whirl at
  $\Omega$ through the same vehicle positions,
  $\tau_k(t) = R_z(\varphi(t))\bigl(\tau_k^s + s(t)(\tau_k^w - \tau_k^s)\bigr)$.
  After the ramp the reference is the exact whirl, whose jet is closed form.

The smooth step is $s = I_u(m+1, m+1)$, the regularised incomplete beta
function with $m$ the map's jet order, so it is continuous through every
derivative the map reads. Its derivatives are evaluated from $w = u(1-u)$,
not from the monomial form, whose alternating coefficients lose every digit
near the ends at the degrees a long hose needs.

A long hose needs a slow start. Each rung of the recursion spends two
derivatives, so the top vehicle's reference carries the step's
$2n$-th derivative scaled by roughly $(\bar m \ell/\tau)^n$: on 20 m with
twenty links a circle ramped up over 6 s asks for more than $10^6$ N, and over
24 s it stays within the 20 N limit. Check every reference with the flat map
before flying it; `simulate()` itself warns before flight when the
reference asks for more thrust or moment than the actuators have.

{py:func}`~udaan.flatness.tethered_hose.circle_rate_bound` gives the circle
rate the scenarios use: $\min(\tfrac12\sqrt{g/\ell}, \Omega_f)$, with $\ell$ the
link length and $\Omega_f$ the largest rate whose steady whirl asks no vehicle
for more than 95 % of the thrust limit
(`TETHER_CIRCLE_THRUST_FRACTION`), leaving 1 N of the 20 N to the controller.
