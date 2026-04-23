# Preliminaries

Symbols, Lie groups, and conventions used throughout the theory pages.

## Symbol table

| Symbol | Meaning |
|---|---|
| $x_Q \in \mathbb{R}^3$ | quadrotor position in world frame |
| $v_Q = \dot{x}_Q$ | quadrotor velocity in world frame |
| $a_Q = \ddot{x}_Q$ | quadrotor acceleration in world frame |
| $R \in SO(3)$ | quadrotor orientation (body $\to$ world) |
| $\Omega \in \mathbb{R}^3$ | angular velocity of the body w.r.t. the world frame, expressed in the body frame |
| $f \in \mathbb{R}_{\geq 0}$ | collective thrust (scalar) |
| $M \in \mathbb{R}^3$ | body-frame moment |
| $m_Q, J$ | quadrotor mass, inertia matrix |
| $x_L \in \mathbb{R}^3$ | payload position |
| $q \in S^2$ | cable unit vector (quadrotor $\to$ payload) |
| $\omega \in \mathbb{R}^3$ | cable angular velocity, $\omega \cdot q = 0$ |
| $\ell$ | cable length |
| $m_L$ | payload mass |
| $g, e_3$ | gravity magnitude, world up-axis $[0,0,1]^\top$ |
| $\widehat{\cdot}$ | hat map $\mathbb{R}^3 \to \mathfrak{so}(3)$ |
| $(\cdot)^\vee$ | vee map $\mathfrak{so}(3) \to \mathbb{R}^3$ |
| $\exp$ | matrix exponential $\mathfrak{so}(3) \to SO(3)$ |

## Hat and vee maps

The *hat map* $\widehat{\cdot} : \mathbb{R}^3 \to \mathfrak{so}(3)$ turns a
3-vector into the skew-symmetric matrix that implements the cross product:

$$
\omega = \begin{bmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \end{bmatrix},
\qquad
\widehat{\omega} =
\begin{bmatrix}
0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0
\end{bmatrix},
\qquad
\widehat{\omega}\, v = \omega \times v \ \ \forall v \in \mathbb{R}^3.
$$

The *vee map* $(\cdot)^\vee : \mathfrak{so}(3) \to \mathbb{R}^3$ is its
inverse:

$$
(\widehat{\omega})^\vee = \omega.
$$

Implemented in {py:func}`udaan.manif.utils.hat` and
{py:func}`udaan.manif.utils.vee`.

## Rotation group $SO(3)$

### Definition

The *special orthogonal group*

$$
SO(3) = \bigl\{\, R \in \mathbb{R}^{3 \times 3} \,\big|\, R^\top R = I,\ \det R = +1 \,\bigr\}
$$

is the configuration manifold of a rotating rigid body. An element
$R \in SO(3)$ maps a vector expressed in the body frame to the same vector
in the world frame: $v_{\text{world}} = R\, v_{\text{body}}$.

$SO(3)$ is a three-dimensional compact matrix Lie group. Its Lie algebra is

$$
\mathfrak{so}(3) = \bigl\{\, \widehat{\Omega} \in \mathbb{R}^{3 \times 3} \,\big|\, \widehat{\Omega}^{\top} = -\widehat{\Omega} \,\bigr\},
$$

the space of $3 \times 3$ skew-symmetric matrices. Implemented in
{py:class}`udaan.manif.SO3`.

### Exponential map $\exp : \mathfrak{so}(3) \to SO(3)$

The matrix exponential sends a skew-symmetric matrix to the rotation
produced by rotating about the corresponding axis by the corresponding
angle. Closed-form (Rodrigues):

$$
\exp(\widehat{\eta}) =
I + \frac{\sin \|\eta\|}{\|\eta\|}\,\widehat{\eta}
+ \frac{1 - \cos \|\eta\|}{\|\eta\|^2}\,\widehat{\eta}^{\,2},
\qquad \eta \in \mathbb{R}^3.
$$

:::{note} Zero-rotation caveat
The formula above has a $0/0$ at $\|\eta\| = 0$. The limit (via the
Taylor expansions $\sin x / x \to 1$ and $(1 - \cos x)/x^2 \to 1/2$) gives
$\exp(\widehat{0}) = I$. The implementation in
{py:func}`udaan.manif.utils.rodrigues_expm` branches on a small-angle
threshold $\|\eta\| < 10^{-10}$ and returns the quadratic Taylor expansion
$I + \widehat{\eta} + \tfrac{1}{2}\widehat{\eta}^{\,2}$, which avoids the
numerical singularity while remaining exact to $\mathcal{O}(\|\eta\|^3)$.
:::

### Operations on `udaan.manif.SO3`

The `SO3` type implements the group structure through operator overloads
so that the code reads the same as the math.

**Group composition and rigid-body action.** For $R_1, R_2 \in SO(3)$ and
$v \in \mathbb{R}^3$,

$$
R_1\, R_2 \in SO(3), \qquad R\, v \in \mathbb{R}^3,
\qquad R^{-1} = R^\top.
$$

```python
R3 = R1 @ R2          # group composition
v_world = R @ v_body  # action on a vector
Rinv = R.inv()        # equals R.T
```

**Integration step.** Given a body-frame angular velocity $\Omega$ and a
time step $h$, the discrete-time integrator is

```{math}
:label: eq-so3-step
R_{k+1} = R_k \, \exp\!\bigl(\widehat{\Omega}\, h\bigr).
```

Invoked as `R.step(Omega * dt)` or equivalently `R + TSO3(Omega * dt)`.

**Constructors.** Three non-trivial ways to build an `SO3`:

| Method | Math | Typical use |
|---|---|---|
| `SO3.from_angle_axis(η)` | $\exp(\widehat{\eta})$ | rotate by $\|\eta\|$ about axis $\eta / \|\eta\|$ |
| `SO3.from_two_vectors(b3, b1)` | columns $(b_1', b_2, b_3)$, $b_3$ preserved, $b_1'$ the component of $b_1$ orthogonal to $b_3$, $b_2 = b_3 \times b_1'$ | desired attitude from thrust direction $b_3$ and heading hint $b_1$ |
| `SO3.from_tilt_yaw(tilt, ψ)` | $b_3 = \exp(\widehat{\mathrm{tilt}})\, e_3$, $b_1 = [\cos\psi,\sin\psi,0]^\top$ | tilt/yaw parameterisation of the desired attitude |

### Tangent space $TSO(3)$

The *tangent bundle* $TSO(3) = SO(3) \times \mathfrak{so}(3)$ pairs each
orientation with an angular velocity. Throughout, $\Omega \in \mathbb{R}^3$
is the angular velocity of the body with respect to the world, *expressed
in the body frame*:

$$
\dot{R} = R\,\widehat{\Omega}.
$$

Represented by {py:class}`udaan.manif.TSO3`.

#### Operations on `udaan.manif.TSO3`

`TSO3` wraps a 3-vector and behaves like a tangent-space element:

$$
\omega_1 + \omega_2, \quad \omega_1 - \omega_2, \quad \alpha\, \omega
\ \in \mathfrak{so}(3) \quad \forall \alpha \in \mathbb{R},
$$

with `omega.hat()` returning the skew-symmetric $\widehat{\omega} \in \mathfrak{so}(3)$
and `omega.norm` returning $\|\omega\|$.

#### Frame transport

You cannot subtract $\Omega - \Omega_d$ directly: $\Omega$ is the actual
angular velocity expressed in the actual body frame $R$, while $\Omega_d$
is the reference expressed in the desired frame $R_d$. Both are 3-vectors,
but they live in different tangent spaces — $T_R SO(3)$ versus
$T_{R_d} SO(3)$.

The computation has two steps.

**Step 1 — lift to the world frame.** A body-frame angular velocity
transforms to the world frame by the attitude itself:

$$
\Omega_d^{\,\text{world}} = R_d\, \Omega_d,
\qquad
\Omega^{\,\text{world}} = R\, \Omega.
$$

**Step 2 — pull back to the actual body frame.** Re-express the desired
world-frame rate in $R$'s frame:

```{math}
:label: eq-transport
\Omega_d^{(R)} = R^\top\, \Omega_d^{\,\text{world}} = R^\top R_d\, \Omega_d.
```

Now both vectors live in the same body frame and subtracting is
well-defined:

```{math}
:label: eq-eOmega
e_\Omega = \Omega - \Omega_d^{(R)} = \Omega - R^\top R_d\, \Omega_d.
```

This is the construction used in the geometric SE(3) controller
{footcite}`lee2010geometric`; see {doc}`controllers/quadrotor-se3`.

```python
e_Omega = Omega - Omega_d.transport(R_from=R_d, R_to=R)
# equivalent to:  e_Omega = Omega - R.T @ R_d @ Omega_d
```

{py:meth}`udaan.manif.TSO3.transport` executes exactly
{math:numref}`eq-transport`.

### Configuration error $e_R$

For a desired orientation $R_d$ and actual $R$, the configuration error in
the Lie algebra is

```{math}
:label: eq-config-error
e_R = \tfrac{1}{2} \bigl( R_d^\top R - R^\top R_d \bigr)^{\!\vee} \in \mathbb{R}^3,
```

as in {footcite}`lee2010geometric`, Eq. 9. The construction proceeds in
three steps.

**Step 1 — the relative rotation.** The error is *not* $R - R_d$; that
difference is not even in $SO(3)$. Instead, compose to get the rotation
that takes the desired frame to the actual frame:

$$
R_e \;\triangleq\; R_d^\top R \in SO(3).
$$

$R_e = I$ iff $R = R_d$; any other value means the two frames disagree.

**Step 2 — skew-symmetric part.** The anti-symmetric piece of $R_e$ lives
in $\mathfrak{so}(3)$:

$$
\mathrm{skew}(R_e) = \tfrac{1}{2}\bigl(R_e - R_e^\top\bigr)
                   = \tfrac{1}{2}\bigl(R_d^\top R - R^\top R_d\bigr)
                   \in \mathfrak{so}(3).
$$

This is guaranteed skew-symmetric because
$\mathrm{skew}(R_e)^\top = -\mathrm{skew}(R_e)$.

**Step 3 — vee to $\mathbb{R}^3$.** Apply the inverse of the hat map:

$$
e_R = \bigl(\mathrm{skew}(R_e)\bigr)^{\!\vee}
     = \tfrac{1}{2}\bigl(R_d^\top R - R^\top R_d\bigr)^{\!\vee}.
$$

In code:

```python
Re = R_d.T @ R                              # Step 1 — relative rotation
skew = 0.5 * (Re - Re.T)                    # Step 2 — skew part in so(3)
e_R = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])  # Step 3 — vee

# Or, using the SO3 type directly:
e_R = R - R_d                               # TSO3, identical result
```

`udaan.manif.SO3.__sub__` executes exactly steps 1–3 (see
{py:meth}`udaan.manif.SO3.__sub__`).

**Relation to the axis-angle parameterisation.** Write $R_e =
\exp(\widehat{\eta})$ for some $\eta \in \mathbb{R}^3$ with
$\|\eta\| < \pi$. By Rodrigues,

$$
R_e - R_e^\top
= \exp(\widehat{\eta}) - \exp(\widehat{\eta})^\top
= 2\,\frac{\sin \|\eta\|}{\|\eta\|}\,\widehat{\eta},
$$

since the $I$ and $\widehat{\eta}^{\,2}$ terms of Rodrigues are symmetric
and drop out under the transpose. Dividing by 2 and applying vee,

```{math}
:label: eq-eR-axis-angle
e_R = \frac{\sin \|\eta\|}{\|\eta\|}\,\eta.
```

For small angles $e_R \approx \eta$; near $\|\eta\| = \pi$, the factor
$\sin \|\eta\| / \|\eta\|$ vanishes, which is why the geometric controller
of {footcite}`lee2010geometric` guarantees only *almost-global* stability
— the sublevel set $\Psi < 2$ excludes the $180°$-error antipode.

**Scalar Morse error.**

$$
\Psi(R, R_d) = \tfrac{1}{2}\,\mathrm{tr}\!\bigl(I - R_d^\top R\bigr)
$$

is a smooth Morse function on
$SO(3) \setminus \{R : \mathrm{tr}(R_d^\top R) = -1\}$; it vanishes iff
$R = R_d$ and reaches its maximum $\Psi = 2$ at the $180°$-error
configuration. In axis-angle form, $\Psi = 1 - \cos \|\eta\|$, making the
connection with $e_R$ explicit.

```python
psi = R.config_error(R_d)  # scalar Morse error
```

## Sphere $S^2$

### Definition

The 2-sphere

$$
S^2 = \bigl\{\, q \in \mathbb{R}^3 \,\big|\, \|q\| = 1 \,\bigr\}
$$

is the configuration manifold of a unit vector — used here for the cable
direction. Implemented in {py:class}`udaan.manif.S2`.

### Tangent space $TS^2$

The tangent space at $q$ is the plane of vectors orthogonal to $q$:

$$
T_q S^2 = \bigl\{\, \omega \in \mathbb{R}^3 \,\big|\, \omega \cdot q = 0 \,\bigr\}.
$$

The cable angular velocity lives in this tangent space,
$\omega \in T_q S^2$, so $\omega \cdot q = 0$ is a membership condition on
$\omega$ rather than an extra constraint imposed on top of it.

:::{note} Choice of representation
Any $\omega' \in \mathbb{R}^3$ produces the same $\dot{q} = \omega' \times q$
as its projection $\omega = \omega' - (\omega' \cdot q)\, q$ onto $T_q S^2$
— the component along $q$ is annihilated by the cross product. We always
use the projected representative $\omega \in T_q S^2$, matching the
convention in {footcite}`sreenath2013geometric`.
:::

Cable kinematics:

$$
\dot{q} = \omega \times q = \widehat{\omega}\, q, \qquad \omega \in T_q S^2.
$$

The bundle $TS^2 = \{\,(q, \omega) : \omega \in T_q S^2\,\}$ is represented
by {py:class}`udaan.manif.TS2`.

### Integration step

The analogue of the $SO(3)$ exponential step advances $q \in S^2$ under
a perpendicular angular velocity $\omega$:

$$
q_{k+1} = \exp\!\bigl(\widehat{\omega}\, h\bigr)\, q_k,
$$

available via {py:meth}`udaan.manif.S2.step`. The same zero-rotation
caveat as for the $SO(3)$ exponential applies — the implementation falls
back to the Taylor expansion below $\|\omega\| < 10^{-10}$.

### Configuration error $e_q$

For a desired cable direction $q_d \in S^2$ and actual $q \in S^2$,
the configuration error used in the payload controller
{footcite}`sreenath2013geometric` is

```{math}
:label: eq-eq
e_q = \widehat{q}^{\,2}\, q_d \in T_q S^2.
```

The construction proceeds in two steps.

**Step 1 — cross-product twice.** Apply $\widehat{q} = q\times$ repeatedly:

$$
\widehat{q}\, q_d = q \times q_d,
\qquad
\widehat{q}^{\,2}\, q_d = q \times (q \times q_d).
$$

The vector $q \times q_d$ is normal to both $q$ and $q_d$, with magnitude
$\sin \theta$ where $\theta$ is the angle between them. A second cross
product with $q$ rotates it $90°$ within the plane $T_q S^2$.

**Step 2 — BAC-CAB identity.** Expanding,

$$
e_q = q \times (q \times q_d)
    = q\,(q \cdot q_d) - q_d\,(q \cdot q)
    = (q \cdot q_d)\, q - q_d,
$$

i.e. $e_q$ is the *negative projection* of $q_d$ onto the plane
$T_q S^2$. A quick sanity check: $q \cdot e_q = (q\cdot q_d)(q\cdot q) -
q\cdot q_d = 0$, so $e_q \in T_q S^2$ as claimed. $e_q = 0$ when
$q = q_d$; the error degenerates at the antipode $q_d = -q$, where both
candidate axes of rotation in $T_q S^2$ are equivalent.

```python
e_q = np.cross(q, np.cross(q, q_d))   # equivalent to hat(q) @ hat(q) @ q_d
```

### Angular-velocity error $e_\omega$

With $\dot q = \omega \times q$ and $\dot q_d = \omega_d \times q_d$, the
angular-velocity error used by the cable loop is

```{math}
:label: eq-eomega-s2
e_\omega = \dot q - (q_d \times \dot q_d) \times q.
```

**Why this form?** Because $\omega_d \in T_{q_d}S^2$, we cannot write
$\omega - \omega_d$ directly — the two vectors live in different tangent
spaces, as with the $SO(3)$ case. The trick is to substitute the
kinematic identity

$$
q_d \times \dot q_d
= q_d \times (\omega_d \times q_d)
= \omega_d\,(q_d \cdot q_d) - q_d\,(q_d \cdot \omega_d)
= \omega_d,
$$

using BAC-CAB and $\omega_d \cdot q_d = 0$. Equation
{math:numref}`eq-eomega-s2` therefore reads

$$
e_\omega = \dot q - \omega_d \times q = (\omega - \omega_d) \times q,
$$

which is in $T_q S^2$ by construction and independent of the component
of $\omega_d$ along $q_d$. Crossing with $q$ is the lift that reconciles
the two tangent spaces — the counterpart of the transport map
{math:numref}`eq-transport` on $SO(3)$.

```python
dq   = np.cross(omega, q)
dqd  = np.cross(omega_d, q_d)
e_om = dq - np.cross(np.cross(q_d, dqd), q)   # matches eq. (eq-eomega-s2)
```

Both $e_q$ and $e_\omega$ appear in the payload cable-attitude PD law
derived in {doc}`controllers/quadrotor-payload`.

## References

```{footbibliography}
```
