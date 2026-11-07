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
fast — the paper uses $\Gamma = 10^6 I$ (the default here is $100\,I$). What the plant receives is not that
fast estimate but its **low-pass filtered** version. Estimation speed and
loop bandwidth are therefore decoupled: estimation can be pushed until the
estimate converges within a step, while the filter cutoff alone decides how
much of the estimate reaches the actuators, and hence how much sensor noise is
amplified. The two channels reach fast estimation differently. The attitude
channel uses a gradient law with gain $\Gamma$, bounded by a smooth
**projection operator** so it cannot wind up against an input the compensator
can never cancel. The position channel uses the piecewise-constant law, which
estimates in one sample by construction and has no gain at all.

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
\Gamma y - \dfrac{(\Gamma \nabla f)\,(\nabla f^\top \Gamma y)}{\nabla f^\top \Gamma \nabla f}\, f
  & \text{if } f > 0 \text{ and } y^\top \Gamma \nabla f > 0, \\[6pt]
\Gamma y & \text{otherwise.}
\end{cases}
```

Outside the ball, and only while the adaptation is pushing further out,
the component of $\Gamma y$ along $\nabla f$ is scaled away in proportion
to $f$ — so the law is continuous at the boundary, unlike a hard clip.
This is the $\Gamma$-projection of Lavretsky and Gibson (arXiv:1112.4232,
Definition 11), which {footcite}`kotaru2020geometric` cites; the code uses
that definition's denominator $\nabla f^\top\Gamma\nabla f$ (the paper's
Eq. (42) prints $\nabla f^\top\nabla f$).

The implementation *also* renormalises $\hat\sigma$ to $\sigma_{\max}$
after every integration step, and in practice that clip, not the projection,
is what bounds the estimate: it leaves $f \le 0$ at the start of each step,
so the projection branch essentially never fires.

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
| $c$ | `c` | $\tfrac12 k_\Omega(k_R+\Gamma)/\Gamma$, per axis | Lyapunov cross-term weight in {math:numref}`eq-l1-att-adapt`; see [the stability bound](#stability-bound-on-c) |
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

### Stability bound on c

Linearised about perfect prediction, one body axis of the predictor–adaptation
loop is $J\dot{\tilde\omega} = \hat\sigma - k_R\tilde\theta - k_\Omega\tilde\omega$
with $\dot{\hat\sigma} = -\Gamma(\tilde\omega + c J^{-1}\tilde\theta)$, whose
characteristic polynomial is

$$
J s^3 + k_\Omega s^2 + (k_R + \Gamma) s + \Gamma c / J .
$$

Routh's condition $k_\Omega(k_R+\Gamma) > J\cdot\Gamma c/J$ has the inertia
cancel exactly, leaving

$$
c < \frac{k_\Omega (k_R + \Gamma)}{\Gamma},
$$

which depends on the gains alone. At the default gains and $\Gamma = 100$ the
bound is $0.358$ on $x$/$y$ and $0.228$ on $z$; the controller uses half of it,
$c = [0.179, 0.179, 0.114]$, recomputed from the current gains on every call,
and logs a warning if an explicit `c` breaks the bound. At the derived $c$,
on MuJoCo, a 5° start with no disturbance keeps $|\hat\sigma|$ under
0.05 N·m, and a constant 0.1 N·m torque is estimated to within 5 mN·m.

## Position channel

{py:class}`~udaan.control.quadrotor.PositionL1Controller` is the
**piecewise-constant** $L_1$ law of Hovakimyan and Cao (*L1 Adaptive Control
Theory*, SIAM 2010, §3.3) around the inherited
{py:class}`~udaan.control.quadrotor.PositionPDController`. It has no
adaptation gain to bound, and the PD baseline makes "$L_1$ = PD + adaptation"
testable directly.

### Baseline and control

```{math}
:label: eq-l1-pos
u = u_{PD} - C(s)\,\hat\delta,
\qquad
u_{PD} = -k_p e - k_d \dot e + \hat m\,(a^d + g e_3),
```

so with $\hat\delta \equiv 0$ the controller *is* the PD controller, sample for
sample. The return value is the world-frame thrust-force vector the
`InputType.ACCELERATION` pipeline expects.

### Predictor

On $x = (p, v)$ with the controller's mass belief $\hat m$,

$$
\dot{\hat x} = A_s(\hat x - x) + f(x, u) + \hat\sigma,
\qquad
f(x, u) = \bigl(v,\ \mathrm{clip}(u\cdot b_3)\,b_3/\hat m - g e_3\bigr),
\qquad
A_s = -a I_6 ,
$$

with $b_3 = Re_3$ the attitude when $u$ was applied: the predictor uses the
thrust the vehicle actually produces. The attitude loop delivers only the
component of $u$ along $b_3$, and only after rotating $b_3$ toward $u$. A
predictor driven by the commanded $u$ reads that rotation lag as a disturbance,
and on a moving reference the compensation feeds back through the attitude
loop. $R$ arrives as an optional third element of the
state, `compute(t, (p, v, R))`, which `QuadrotorBase.simulate()` and
`QuadrotorFleet` pass and every other position controller ignores. Without it
the predictor falls back to the commanded $u$. $\mathrm{clip}$ is the model's
collective-thrust clip (`thrust_limits`, by default the `core.defaults` limits),
so thrust saturation is not read as a disturbance. The torque clip and the
per-rotor clip are not modelled; they show up only as attitude lag.

Between samples $u$ and $\hat\sigma$ are held. The model part is integrated
exactly from the last measured state (a double integrator under a constant
force), and the error $\tilde x = \hat x - x$ decays by $e^{-aT}$. On a plant
that matches the model the predictor is exact, $\tilde x \equiv 0$ and
$\hat\sigma \equiv 0$.

This is a deliberate sampled-data choice, not the exact discretisation of the
continuous predictor. That one would integrate $A_s(\hat x - x(t))$ against a
plant state the controller never sees between samples. Restarting the model
part from each measured state is what makes the nominal case exact. The two
settle at different fractions of a constant disturbance $d$: this predictor at
$aT/(e^{aT}-1)\,d = 0.847\,d$ at $aT = 0.32$, the exact continuous one at
$e^{-aT} d = 0.724\,d$.

### Adaptation

At every sample, $T$ being the time since the previous one,

$$
\hat\sigma(kT) = -\Phi(T)^{-1} e^{A_s T}\,\tilde x(kT),
\qquad
\Phi(T) = \int_0^T e^{A_s\tau} d\tau = \frac{1 - e^{-aT}}{a} I_6 ,
$$

the value that cancels the current prediction error over the next step.
$\hat\sigma \in \mathbb R^6$: its velocity half gives the matched estimate
$\hat\delta$ (a force, below); its position half is unmatched
and only feeds the predictor. The error recursion is deadbeat at any $T$, with
no adaptation gain and so no step-size limit.

For a constant disturbance acceleration $d$ (the force is $\hat m d$), $\hat\sigma_v$ settles one step late at

$$
\hat\sigma_v = d\,\frac{aT}{e^{aT} - 1},
$$

0.847 of $d$ at $aT = 0.32$ (the defaults at 5 ms). The matched estimate scales
it back:

$$
\hat\delta = \kappa\,\hat m\,\hat\sigma_v,
\qquad
\kappa = \frac{e^{aT} - 1}{aT},
$$

so $\hat\delta = \hat m d$ exactly and the PD loop is left nothing to do. Only
the estimate sent to the control is scaled. Scaling the $\hat\sigma$ fed back
into the predictor would turn the deadbeat error recursion into
$e^{-aT}(1 - \kappa)$ and move its fixed point off $d$ again.

### Filter

$C(s) = \omega_c/(s + \omega_c)$, as a zero-delay update that uses the *new*
estimate (one sample ahead of the zero-order-hold form, which would use the
previous one):
$\delta_f \leftarrow \delta_f + (1 - e^{-\omega_c T})(\hat\delta - \delta_f)$.

### Bandwidths

Both are derived from the gains and $\hat m$ unless passed
(`predictor_pole`, `lpf_cutoff`, both in rad/s). Per axis the PD loop is
$\hat m\ddot e = -k_p e - k_d\dot e$, with complementary sensitivity

$$
T(s) = \frac{k_d s + k_p}{\hat m s^2 + k_d s + k_p}.
$$

$\omega_b$ is the largest $-3$ dB frequency over the three axes,
$a = 5\,\omega_b$, and $\omega_c = \sqrt{\omega_b\,a} = \sqrt5\,\omega_b$, the
geometric mean of the two. At the default gains:

| $\hat m$ | $\omega_b$ ($x$/$y$, $z$) | $a$ | $\omega_c$ |
|---|---|---|---|
| 0.75 kg | 5.29, **12.90** rad/s | 64.5 rad/s | 28.8 rad/s |

On the MuJoCo backend, with the realised-thrust predictor, a 1 N side force
and a 20 % mass under-estimate each leave a steady error under 1 mm, where the
PD baseline is left tens of centimetres off. Under the mass under-estimate the
bandwidths derive from the believed 0.6 kg, giving $a = 79.5$ and
$\omega_c = 35.5$.

## Trying it

```bash
uv run udaan run fleet --demo l1-comparison
```

The demo flies a hover setpoint with three quadrotors:
- baseline PD plus geometric attitude, with no disturbance;
- the same pair carrying an undeclared 0.25 kg mass offset from the centre of
  gravity;
- the full $L_1$ pair — position *and* attitude — carrying the same
  disturbance.

`examples/quadrotor_geometric_l1.py` is the headless version of the position
channel alone: a position controller handed a mass 0.25 kg lighter than the
plant, flown with PD and with $L_1$, printing the steady-state sag each leaves.

## What is not on this page

The stability argument is not reproduced here. The attitude design's
Lyapunov function and its exponential input-to-state bound, which shrinks
with $\Gamma$, are in {footcite}`kotaru2020geometric`. The paper puts no
condition on the filter beyond $C(0) = 1$, and gives no equations for the
position controller (its Remark 6 cites a non-geometric position $L_1$ law);
the position design on this page follows Hovakimyan and Cao. The summary above is a reading of the
implementation, not an independent proof. Both controllers are exercised by
`tests/test_quadrotor_l1.py`, on an exact point-mass plant and on the MuJoCo
backend.

## References

```{footbibliography}
```
