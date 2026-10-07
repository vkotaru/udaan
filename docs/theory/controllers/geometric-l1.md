(control-geometric-l1)=
# Geometric $L_1$-adaptive control

The $L_1$ augmentation shipped with `udaan` adds disturbance estimation
and bandwidth-limited compensation on top of the geometric baseline of
{doc}`quadrotor-se3`. It is two independent pieces, one per loop of the
single-quadrotor (M1) cascade, and either can be used without the other:

| Loop | Class | Baseline it extends |
|------|-------|---------------------|
| Attitude, on $SO(3)$ | {py:class}`~udaan.control.quadrotor.GeometricL1AttitudeController` | {py:class}`~udaan.control.quadrotor.GeometricAttitudeController` |
| Position, on $\mathbb{R}^3$ | {py:class}`~udaan.control.quadrotor.PositionL1Controller` | {py:class}`~udaan.control.quadrotor.PositionPDController` |

```{warning}
{py:class}`~udaan.control.quadrotor.PositionL1Controller` is **experimental and
does not work**: its adaptation is frozen, and at no gain does it outperform the
PD baseline it is meant to improve on. See
[below](#the-position-channel-is-experimental-and-does-not-adapt). The page
still documents the law as designed, because that is what the class is meant to
become.
```

Both are drop-in replacements — each subclasses its baseline, keeps its
`compute()` signature and its `setpoint` protocol, and is installed
through the model's controller properties:

```python
from udaan.control.quadrotor import GeometricL1AttitudeController, PositionL1Controller

mdl.position_controller = PositionL1Controller(mass=mdl.mass, setpoint=traj.get)
mdl.attitude_controller = GeometricL1AttitudeController(inertia=mdl.inertia)
```

The attitude law is the geometric $L_1$ design of
{footcite}`kotaru2020geometric`; the equation numbers quoted below are
the ones the implementation's own comments name, so the code and the
paper can be read side by side.

## The architecture, in one paragraph

Both controllers share the $L_1$ skeleton, which is what separates them
from plain model-reference adaptive control. A **state predictor** (the
code calls it the MRAC model) is integrated alongside the plant from the
same initial condition, driven by the nominal dynamics and the *current*
disturbance estimate. The mismatch between predictor and plant is the
only signal the **adaptation law** sees, and adaptation is deliberately
fast — the gain $\Gamma$ is large. What the plant receives is not that
fast estimate but its **low-pass filtered** version. Estimation speed and
loop bandwidth are therefore decoupled: $\Gamma$ can be pushed until the
estimate converges within a step, while the filter cutoff $a$ alone
decides how much of the estimate reaches the actuators, and hence how
much sensor noise is amplified. Both controllers also bound the estimate
with a smooth **projection operator**, so the adaptation cannot wind up
against an unmodelled input the compensator can never cancel.

## Attitude channel

### Plant and predictor

The plant is the rigid-body attitude of {doc}`../dynamics/quadrotor`
with an unknown, bounded, matched body-torque disturbance $\sigma$:

$$
\dot R = R\,\widehat\Omega,
\qquad
J\dot\Omega + \Omega \times J\Omega = M + \sigma,
\qquad
\|\sigma\| \le \sigma_{\max}.
$$

The predictor is a second copy of the same equations *without* $\sigma$,
carrying its own state $(R_m, \Omega_m)$ and driven by its own moment
$M_m$:

```{math}
:label: eq-l1-att-predictor
\dot R_m = R_m \widehat{\Omega_m},
\qquad
J \dot\Omega_m = M_m - \Omega_m \times J \Omega_m .
```

{py:meth}`~udaan.control.quadrotor.GeometricL1AttitudeController.compute`
integrates {math:numref}`eq-l1-att-predictor` explicitly, once per call,
with $M_m$ held at the value computed on the previous call and $\Delta t$
taken from the caller's clock. On the first call the predictor is *seeded
from the plant* ($R_m \leftarrow R$, $\Omega_m \leftarrow \Omega$) and
nothing else happens, so the two states start coincident;
{py:meth}`~udaan.control.quadrotor.GeometricL1AttitudeController.reset`
puts the controller back into that state for a fresh run.

### Errors

Three error pairs are in play. All of them use the manifold operators of
{doc}`../preliminaries` — $R_1 - R_2$ is
$\tfrac12\bigl(R_2^\top R_1 - R_1^\top R_2\bigr)^\vee$, and
`transport` carries a body rate between frames:

```{math}
:label: eq-l1-att-errors
\begin{aligned}
\text{plant vs.\ reference:}
  &\quad e_R = R - R_d, & e_\Omega &= \Omega - R^\top R_d\, \Omega_d, \\
\text{predictor vs.\ reference:}
  &\quad e_R^m = R_m - R_d, & e_\Omega^m &= \Omega_m - R_m^\top R_d\, \Omega_d, \\
\text{predictor vs.\ plant (Eq.\ 30–31):}
  &\quad \tilde e_R = R_m - R, & \tilde e_\Omega &= \Omega_m - R_m^\top R\, \Omega .
\end{aligned}
```

$R_d$ is built exactly as in the baseline: the thrust command fixes
$b_3$ and the flatness reference attitude, when one is supplied,
contributes only its heading. See
{py:meth}`~udaan.control.quadrotor.GeometricAttitudeController.compute`
for why the reference is composed rather than substituted.

### Adaptation law

Write $Q := R^\top R_m \in SO(3)$ for the plant-to-predictor frame map,
and

```{math}
:label: eq-l1-att-P
P := J\, Q^\top J^{-1} Q .
```

$P$ is the disturbance-input matrix expressed through that frame
mismatch; it is the identity when predictor and plant coincide. The
adaptation direction and law (Eq. 37–39) are

```{math}
:label: eq-l1-att-adapt
y = -\Bigl( P^\top \tilde e_\Omega + c\, P^\top J^{-\top} \tilde e_R \Bigr),
\qquad
\dot{\hat\sigma} = \operatorname{Proj}_\Gamma(\hat\sigma, y),
```

with $c > 0$ the Lyapunov cross-term constant that weights the attitude
half of the mismatch against the rate half. $\Gamma \succ 0$ is diagonal.

The projection (Eq. 42, Remark 5) keeps $\hat\sigma$ inside a ball of
radius $\sigma_{\max}$ with a smoothing margin $\varepsilon$. With

$$
f(\hat\sigma) = \frac{\|\hat\sigma\|^2 - \sigma_{\max}^2}
                      {\varepsilon\, \sigma_{\max}^2},
\qquad
\nabla f = \frac{2\,\hat\sigma}{\varepsilon\, \sigma_{\max}^2},
$$

the operator is

```{math}
:label: eq-l1-proj
\operatorname{Proj}_\Gamma(\hat\sigma, y) =
\begin{cases}
\Gamma y - \dfrac{(\Gamma \nabla f)\,(\nabla f^\top \Gamma y)}{\|\nabla f\|^2}\, f
  & \text{if } f > 0 \text{ and } y^\top \Gamma \nabla f > 0, \\[6pt]
\Gamma y & \text{otherwise.}
\end{cases}
```

Outside the ball, and only while the adaptation is pushing further out,
the component of $\Gamma y$ along $\nabla f$ is scaled away in proportion
to $f$ — so the law is continuous at the boundary, unlike a hard clip.
The implementation *also* renormalises $\hat\sigma$ to $\sigma_{\max}$
after the integration step; that clip is a belt-and-braces guard against
a large $\Gamma\,\Delta t$ overshooting the smoothing band in one
discrete step, not part of the continuous design.

### Filter and the two moments

The compensator the plant sees is the estimate through a first-order
low-pass $C(s) = a/(s+a)$:

```{math}
:label: eq-l1-att-filter
\dot\sigma_f = -a\, \sigma_f + a\, \hat\sigma .
```

The predictor is then driven by the *unfiltered* estimate and the plant
by the filtered one — this asymmetry is the whole point of the
architecture. With $\mu_1^m := -k_R e_R^m - k_\Omega e_\Omega^m$, the
predictor moment (Eq. 34) is

```{math}
:label: eq-l1-att-Mm
M_m = \mu_1^m
    + \underbrace{P\,\hat\sigma - \sigma_f}_{\mu_2^m}
    + \Omega_m \times J\Omega_m
    - J\Bigl(\widehat{\Omega_m}\, R_m^\top R_d\, \Omega_d
             - R_m^\top R_d\, \dot\Omega_d\Bigr),
```

and the moment actually commanded to the vehicle is

```{math}
:label: eq-l1-att-M
\begin{aligned}
\mu_1 &= J\,Q\,J^{-1}\Bigl(\mu_1^m - \sigma_f
          + k_R\,\tilde e_R + k_\Omega\,\tilde e_\Omega\Bigr), \\
\mu_2 &= J\,Q\,\widehat{\tilde e_\Omega}\,Q^\top e_\Omega, \\
M &= \mu_1 + \mu_2 + \Omega \times J\Omega
   - J\Bigl(\widehat{\Omega}\, R^\top R_d\, \Omega_d
            - R^\top R_d\, \dot\Omega_d\Bigr).
\end{aligned}
```

Two things are worth reading off {math:numref}`eq-l1-att-M` directly.
First, the last two terms are the *baseline* gyroscopic and
feedforward terms of {prf:ref}`def-geom-att`, untouched — the
augmentation is genuinely additive on that part of the law. Second, the
baseline PD is no longer applied to $e_R, e_\Omega$ directly: it is
applied to the *predictor's* errors and then transported into the plant
frame by $J Q J^{-1}$, with the predictor–plant mismatch fed back at the
same gains. When $R_m \equiv R$ and $\sigma_f \equiv 0$ we have
$Q = I$, $\tilde e_R = \tilde e_\Omega = 0$, $\mu_2 = 0$ and
$\mu_1 = \mu_1^m = -k_R e_R - k_\Omega e_\Omega$, so
{math:numref}`eq-l1-att-M` collapses exactly onto the baseline law. That
is the quiescent property: with no disturbance to estimate, the
augmented controller *is* the geometric controller.

The scalar thrust is passed through unchanged, $f = F \cdot R e_3$.

### Parameters

| Symbol | Constructor kwarg | Default | Role |
|--------|-------------------|---------|------|
| $\Gamma$ | `Gamma` | $[100, 100, 100]$ | adaptation gain, diagonal, per body axis |
| $c$ | `c` | $1.0$ | Lyapunov cross-term weight in {math:numref}`eq-l1-att-adapt` |
| $\sigma_{\max}$ | `sigma_max` | $30.0$ | projection radius, N·m |
| $\varepsilon$ | `eps` | $0.5$ | projection smoothing band |
| $a$ | `lpf_cutoff` | $5.0$ | filter pole in {math:numref}`eq-l1-att-filter` |
| $k_R, k_\Omega$ | inherited | `DEFAULT_ATT_KP`, `DEFAULT_ATT_KD` | baseline attitude PD |

$\hat\sigma$ and $\sigma_f$ are published after every call as the
`sigma_hat` and `sigma_filtered` attributes, which is the intended way to
plot what the estimator is doing.

:::{note}
`lpf_cutoff` is applied directly as the pole $a$ of
{math:numref}`eq-l1-att-filter`, i.e. in rad/s, giving a time constant
$1/a = 0.2$ s at the default. The source comment calls it a cutoff
*frequency*; read it as rad/s, not Hz.
:::

## Position channel

{py:class}`~udaan.control.quadrotor.PositionL1Controller` follows the
same skeleton on the translational double integrator, but it is a
separate design rather than a transcription of the attitude one, and
three things differ: the baseline is LQR rather than the inherited PD,
the adaptation direction comes from the Riccati solution, and the
projection applies $\Gamma$ *outside* rather than inside.

### Baseline and predictor

The design model is the double integrator with force input,

$$
\dot\eta = F\eta + G u,
\qquad
F = \begin{bmatrix} 0 & I \\ 0 & 0\end{bmatrix},
\qquad
G = \frac{1}{m_Q}\begin{bmatrix} 0 \\ I \end{bmatrix},
\qquad
\eta = \begin{bmatrix} x_Q - x^d \\ v_Q - v^d\end{bmatrix},
$$

and the feedback gain is the LQR solution for $Q = I_6$,
$R = 10^{-2} I_3$, computed once at construction together with its
Riccati matrix $P$:

$$
(K, P) = \operatorname{lqr}(F, G, I_6, 10^{-2} I_3),
\qquad
K_y := -G^\top P .
$$

$K_y \in \mathbb{R}^{3 \times 6}$ is what turns the predictor–plant
mismatch into an adaptation direction; taking it from the Riccati
solution rather than picking it freely is what makes
the adaptation Lyapunov-consistent with the baseline it augments.

The predictor carries $(x_m, v_m)$, integrated with its own input
$u_m$ held constant across the step, and is seeded from the plant on
the first call exactly as the attitude one is (the first call returns a
zero command). Its error is $\eta_m$, formed against the *same*
reference $(x^d, v^d)$.

### Law

```{math}
:label: eq-l1-pos
\begin{aligned}
\tilde\eta &= \eta_m - \eta,
&
y &= K_y\, \tilde\eta,
&
\dot{\hat\delta} &= \Gamma \odot \operatorname{Proj}(\hat\delta, y), \\[4pt]
u &= -K\eta + m_Q\,(a^d + g e_3) - \delta_f,
&
u_m &= -K\eta_m + m_Q\,(a^d + g e_3) + \hat\delta - \delta_f .
\end{aligned}
```

The plant is compensated with the filtered estimate only; the predictor
carries the unfiltered residual $\hat\delta - \delta_f$ on top of it,
which is the translational form of the same asymmetry as
{math:numref}`eq-l1-att-Mm`. The projection is
{math:numref}`eq-l1-proj` with $\Gamma$ removed from the geometry — the
cut is taken on $y$ alone and the diagonal gain is applied to the result
elementwise, which is why $\Gamma$ may be strongly anisotropic here
without skewing the projection direction. The filter is the same
first-order lag, written as an explicit smoothing step
$\delta_f \leftarrow \delta_f + \alpha\,(\hat\delta - \delta_f)$ with
$\alpha = \min(1,\, a\,\Delta t)$, so it degrades to a one-step
pass-through rather than going unstable at a large $\Delta t$.

The return value is the thrust force *vector* in the world frame, which
is the `InputType.ACCELERATION` shape the M1 pipeline expects, so the
attitude loop — baseline or $L_1$ — consumes it unchanged.

### Parameters

| Symbol | Attribute | Default | Role |
|--------|-----------|---------|------|
| $\Gamma$ | `Gamma` | $[10^3, 10^3, 10^5]$ | adaptation gain, per world axis |
| $\delta_{\max}$ | `deltamax` | $30.0$ | projection radius, N |
| $\varepsilon$ | `eps` | $0.5$ | projection smoothing band |
| $a$ | `lpf_cutoff` | $5.0$ | filter rate |

The $10^5$ on the vertical axis is deliberate: a payload or mass error
shows up almost entirely as a constant vertical force, and that is the
channel the estimator is expected to close fastest.

:::{warning}
$K$ and $P$ come from `control.lqr`, so
{py:class}`~udaan.control.quadrotor.PositionL1Controller` — alone among
the controllers in `udaan` — needs the optional `control` package,
installed with the `l1` extra (`pip install 'udaan[l1]'`). The import is
inside `__init__`, so the rest of `udaan.control.quadrotor` still imports
without it, but constructing this class does not.
:::

:::{note}
The predictor integrates $u_m$ directly as an acceleration
($v_m \leftarrow v_m + u_m \Delta t$) while the design model above treats it
as a force ($G = I/m_Q$). The two agree only at $m_Q = 1$, and the
class's own default `mass` is $1.0$; at the package default
$m_Q = 0.9$ the predictor runs a mass it was not designed for. This is
recorded here as an open question against the implementation rather than
as intended behaviour — it has not been resolved from the source.
:::

## Trying it

The CLI ships a three-vehicle side-by-side comparison, which is the
quickest way to exercise both loops end to end. Read what it prints with the
section below in hand: the position channel does not adapt, so the demo shows
the attitude augmentation plus two sets of non-adaptive position gains, not the
design this page describes.

```bash
uv run udaan run fleet --demo l1-comparison
```

It flies a hover setpoint with three quadrotors: baseline PD plus
geometric attitude with no disturbance, the same pair carrying an
undeclared 0.25 kg mass offset from the centre of gravity, and the full
$L_1$ pair — position *and* attitude — carrying the identical
disturbance. The offset mass perturbs both channels at once, a force the
position loop does not know about and a torque the attitude loop does
not, which is why the demo replaces both controllers rather than one.
See {py:mod}`udaan.cli.demos` for the exact configuration.

`examples/quadrotor_geometric_l1.py` is the smaller, headless version,
and isolates the position channel: one quadrotor whose position
controller is handed a mass 0.25 kg lighter than the plant, flown to a
setpoint once with the PD baseline and once with
{py:class}`~udaan.control.quadrotor.PositionL1Controller`, printing the
steady-state sag each leaves behind. It currently prints 0.3028 m for PD
against 0.3096 m for $L_1$ — the $L_1$ run is slightly *worse*, and the example
asserts only that neither diverged.

## The position channel is experimental and does not adapt

```{warning}
{py:class}`~udaan.control.quadrotor.PositionL1Controller` is **experimental**.
Everything above describes the law as written; the implementation has never
executed it. Use {py:class}`~udaan.control.quadrotor.PositionPDController` for
anything that has to fly.
```

Two independent defects, both found by instrumenting a live run rather than by
reading:

**1. The step is zero.** `compute()` assigns `self.last_t = t` at the *top* of
the method, before the step $\Delta t = t - t_{\mathrm{last}}$ is formed a few
lines below, so $\Delta t$ is identically zero and is clamped to $10^{-6}$ —
5000 times smaller than the real 5 ms step. The adaptation integrator, the MRAC
reference model and the low-pass filter are all frozen and the controller
reduces to the LQR baseline of {math:numref}`eq-l1-pos`.

**2. The MRAC reference state was aliased to the plant.** The initialisation
branch used to *bind* rather than copy — `self.mrac_position = pos`,
`self.mrac_velocity = vel`, where `pos` and `vel` are
{py:class}`~udaan.models.quadrotor.QuadrotorBase`'s own state arrays. Checked
live, `self.mrac_position is mdl.state.position` was `True` on every call. That
made $\eta_m \equiv \eta$, hence $\tilde\eta \equiv 0$ and
$\hat\sigma \equiv 0$ **exactly**, for the whole run and at *any* $\Gamma$:
the adaptation could not run even with defect 1 repaired. It also ran the wrong
way round — the in-place updates to `mrac_position` / `mrac_velocity` wrote
straight into the plant, injecting $4.2\times10^{-4}$ m of position and
$4.4\times10^{-2}$ m/s of velocity into the model over a 25 s run. Defect 2 is
**fixed**: the states are copied. It was a correctness bug regardless of the
adaptation — a controller must not write into the plant.

The MuJoCo backend happened to be immune to defect 2, because
`_query_latest_state()` rebuilds those arrays from `mjData` every step. The
analytical backend — the one the example, the README and this page all use —
was not.

### Why the one-line fix is not applied

Defect 1 is a one-line deletion (the correct assignment already exists at the
end of `compute()`). It is deliberately left in place, because deleting it does
not do what it appears to do:

| variant | single-quad hover error | final $\hat\sigma$ |
|---|---|---|
| as shipped | 0.2451 m | **0** |
| delete the `dt` line only | 0.1606 m | **still exactly 0** |
| copy the MRAC state only (what ships now) | 0.3096 m | 36.6 |
| both | NaN | NaN |

The 0.2451 m → 0.1606 m improvement previously quoted for the `dt` fix was
**not** adaptation — $\hat\sigma$ is still identically zero in that row. It was
defect 2: the controller corrupting the plant's own velocity, by up to
0.143 m/s in a single step. Measuring the plant you are perturbing is not a
tracking result.

With both defects fixed the run diverges: at the real 5 ms step the adaptation
gain $\Gamma$ (up to $10^5$) gives $\Gamma\Delta t \approx 500$, far outside
the stability limit of the explicit Euler update the law uses, and
`udaan run fleet --demo l1-comparison` NaNs at $t = 0.34$ s. Nor is there a gain
that rescues it. Sweeping $\Gamma$ down over 1000 / 200 / 100 keeps every run
finite but leaves the L1 quadrotor at 0.72–0.89 m of $z$-error in the fleet demo
against the PD baseline's 0.001 m, and 0.55–0.64 m on the single quadrotor
against PD's 0.303 m.

So the honest summary is: this is not a working controller with a known
one-line fix. It is a law that has never once adapted, and for which no gain
setting beats the PD baseline. Finishing it means redesigning the adaptation
update — most plausibly replacing the explicit Euler integration of
$\dot{\hat\sigma}$ with the piecewise-constant adaptation law of the $L_1$
literature — retuning $\Gamma$, and adding the regression test the controller
has never had. That is control-design work, which is why the controller is
labelled experimental rather than quietly footnoted.

A third issue to settle at the same time: the smooth projection in
{py:meth}`PositionL1Controller.compute() <udaan.control.quadrotor.PositionL1Controller.compute>`
computes
$\mathrm{proj} = y - \frac{\langle \nabla f, y\rangle}{\|\nabla f\|^2}\nabla f\,f$
without clamping $f$ to $1$, so once $\|\hat\sigma\|$ exceeds $\delta_{\max}$
the correction is scaled by $f > 1$ and over-subtracts — which would accelerate
a divergence rather than arrest it.

## What is not on this page

The stability argument is not reproduced here. The attitude design's
Lyapunov function, the bound relating the achievable tracking error to
$\Gamma$ and $a$, and the quiescent-response guarantee are all in
{footcite}`kotaru2020geometric`; the summary above is a reading of the
implementation, not an independent proof. Neither controller carries a
regression test in the current suite, so the defaults tabulated above
should be treated as the author's working values rather than as
validated tuning.

## References

```{footbibliography}
```
