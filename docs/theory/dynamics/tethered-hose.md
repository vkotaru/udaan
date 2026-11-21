# Quadrotors on a hose pinned to a ground anchor

M8 is $n_Q$ quadrotors bolted in series along one hose of $n$ rigid links
whose bottom end, vertex 0, is pinned to an anchor on the ground. M7 is the
case $n_Q = 1$: one quadrotor at the top of an $n$-link tether. Both are
{doc}`multi-quad-hose` with its root vertex held still, and the simulation
models are {py:class}`~udaan.models.tethered_hose.TetheredHoseBase` and
{py:class}`~udaan.models.tethered_hose.TetheredQuadrotorBase`.

$$
Q = \left(S^2\right)^{n} \times SO(3)^{n_Q}.
$$

The conventions are M6's: link $i$ runs from vertex $i$ to vertex $i+1$, so
with the anchor at vertex 0 every $q_i$ points **up** the hose,
$x_{i+1} = x_i + \ell_i q_i$ and $x_0 = x_{\text{anchor}}$. Every vehicle sits
at a vertex $\ge 1$ and one sits at the top vertex $n$.

## The pin deletes a coordinate

M6 integrates the root $x_0$ as a generalised coordinate. The constraint
$x_0 = x_{\text{anchor}}$ therefore fixes a coordinate, and it is solved by
deleting it rather than by a Lagrange multiplier. In M6's system the root row
and column go, the unknowns are $\dot\omega_1, \dots, \dot\omega_n$, and
$\alpha = \ddot x_0 + g e_3$ is the known $g e_3$:

```{math}
:label: eq-m8-system
\bar M\,\dot\omega = b_{\text{links}} - \bigl[M_{0i}\,\hat q_i\bigr]_i\, g\, e_3 ,
```

where $\bar M$ is M6's link block of the mass matrix (with its $M_{ii} I$
gauge fix, so it stays positive definite) and $b_{\text{links}}$ is M6's link
rows of the right-hand side. Gravity no longer cancels from the link rows, as
it does in M6. The deleted row is what the anchor supplies:

```{math}
:label: eq-m8-anchor
\lambda = M_{00}\, g\, e_3 - \sum_i M_{0i}\,\hat q_i\,\dot\omega_i
          - \sum_j F_j - \sum_k M_{0k}\, c_k\, q_k ,
```

the force the anchor exerts on the hose. Vertex 0's lumped mass drops out of
the motion because the anchor carries it. The airframes' rotational dynamics
are M6's, $J_k\dot\Omega_k = M_k - \Omega_k\times J_k\Omega_k$.

The MuJoCo backend pins the hose the same way: the generated tree's root body
has no joint, so MuJoCo welds it to the world. The two backends then differ
only by the vertex spheres' own rotational inertia, $\tfrac25 m r^2$ each,
which the point-mass model leaves out; the disagreement in the vertex
accelerations scales as $r^2$.

## Steady states

With every rate zero, each link points along its own tension and the vertex
balance $\tau_j = \tau_{j-1} + \bar m_j g e_3$ holds at every bare vertex,
with $\tau_i$ the force link $i$ exerts on its lower vertex. A vehicle vertex
$v$ carries the thrust $u_v = \bar m_v g e_3 + \tau_{v-1} - \tau_v$, with
$\tau_n = 0$ at the top.

A trim is given by the **vehicle positions**: the top vehicle's, and the rest
evenly on the line from the anchor to it. Each segment between two held
points is then a pinned-pinned discrete catenary with one taut solution, a
three-equation root solve on the force in its first link
({py:func}`~udaan.utils.hose.tethered_trim`). A steady rotation at rate
$\Omega$ about the anchor's vertical axis is the same solve with the
centripetal acceleration $-\Omega^2 (x_j - x_{\text{anchor}})_{xy}$ at every
vertex.

## The ground

The anchor sits on the ground plane $z = 0$ unless `anchor_height` lifts it,
and neither backend has contacts. A tether slacker than the one whose anchor
link leaves horizontally ($V = \tau_0 \cdot e_3 = 0$) sags below the anchor
and, at ground level, through the floor. For a uniform continuous catenary
the boundary is pure geometry: at a standoff of $0.5L$ the top must be at
least $0.796L$ high. The lumped chain is more lenient and the margin depends
on $n$ ($0.699L$ at four links), so the check is made in the model's own
discretisation: {py:meth}`~udaan.models.tethered_hose.TetheredHoseBase.simulate`
evaluates the reference's lowest vertex before it flies and warns when it is
below the ground.

## Stability of the analytical loop

The analytical model integrates by zero-order hold over the 5 ms step and
the forward-Euler substeps (four by default). A closed loop that decays in continuous time can
grow in that sampled one: M7 on 2 m at eight links decays at $0.29\,\text{s}^{-1}$
and its sampled loop grows at $0.25\,\text{s}^{-1}$. The model's
{py:meth}`~udaan.models.tethered_hose.TetheredHoseBase.stability_growth_rate`
is that sampled loop's growth rate, $\ln\rho(\Phi + \Gamma K)/\Delta t$ with
$\Phi = (I + hA)^4$ and $\Gamma = \sum_j (I + hA)^j hB$ at the trim, and the
constructor warns when it is positive. M6's bound (the free hanging chain's
fastest mode, without the controller) does not describe a pinned trim.
