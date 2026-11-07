r"""L1 adaptive position controller for quadrotor.

Piecewise-constant L1 adaptation (Hovakimyan & Cao, *L1 Adaptive Control
Theory*, SIAM 2010, §3.3) wrapped around :class:`PositionPDController`.
"""

import numpy as np
from scipy.optimize import brentq

from ...core.defaults import DEFAULT_MAX_THRUST, DEFAULT_MIN_THRUST
from .position_pd import PositionPDController

#: Predictor pole as a multiple of the PD loop's closed-loop bandwidth.
PREDICTOR_POLE_RATIO = 5.0


def pd_bandwidth(kp, kd, mass) -> float:
    r"""The PD position loop's closed-loop bandwidth, in rad/s.

    Per axis the loop is :math:`m\ddot e = -k_p e - k_d \dot e`, whose
    complementary sensitivity is

    .. math:: T(s) = \frac{k_d s + k_p}{m s^2 + k_d s + k_p}.

    The bandwidth of an axis is the frequency where :math:`|T(j\omega)|` falls
    to :math:`1/\sqrt 2`; this returns the largest over the three axes, the
    loop the adaptation has to stay ahead of. At the default gains and the
    MuJoCo airframe's 0.75 kg it is 12.90 rad/s, set by ``z``
    (5.29 rad/s on ``x`` and ``y``).
    """
    kp, kd = np.broadcast_to(kp, (3,)), np.broadcast_to(kd, (3,))
    bandwidths = []
    for p, d in zip(kp, kd, strict=True):

        def excess(w, p=p, d=d):
            s = 1j * w
            return abs((d * s + p) / (mass * s**2 + d * s + p)) - 1 / np.sqrt(2)

        # |T| = 1 at DC and → 0 as ω → ∞, so the crossing is bracketed.
        bandwidths.append(brentq(excess, 1e-6, 1e6))
    return float(max(bandwidths))


class PositionL1Controller(PositionPDController):
    r"""Quadrotor position control: PD plus piecewise-constant L1 adaptation.

    **Baseline.** The control is the parent :class:`PositionPDController`'s
    output minus a filtered disturbance estimate,

    .. math:: u = u_{PD} - C(s)\,\hat\delta,

    so with :math:`\hat\delta \equiv 0` this controller *is* the PD controller,
    sample for sample.

    **Predictor.** On the state :math:`x = (p, v)` with the controller's mass
    belief :math:`\hat m`,

    .. math::

        \dot{\hat x} = A_s(\hat x - x) + f(x, u) + \hat\sigma, \qquad
        f(x, u) = \bigl(v,\; \mathrm{clip}(u\cdot b_3)\,b_3/\hat m - g e_3\bigr),
        \qquad A_s = -a I_6,

    with :math:`b_3 = R e_3` the attitude when :math:`u` was applied. That is
    the thrust the vehicle actually produces: the attitude loop delivers only
    the component of :math:`u` along :math:`b_3`, and only after it has
    rotated :math:`b_3` toward :math:`u`. A predictor driven by the commanded
    :math:`u` reads that rotation lag as a disturbance, and on a moving
    reference the compensation feeds back through the attitude loop.
    :math:`R` is read as an optional third element of the state,
    ``compute(t, (p, v, R))``, which every other position controller ignores.
    Without it the predictor falls back to the commanded :math:`u`.
    :math:`\mathrm{clip}` is the model's collective-thrust clip
    (``thrust_limits``, by default the ``core.defaults`` limits
    ``_repackage_input`` uses), so thrust saturation is not mistaken for a
    disturbance. The torque clip and, on the ``PROP_FORCES`` force path, the
    per-rotor clip are not modelled: they bound how fast :math:`b_3` turns, not
    the thrust along it, and show up only as attitude lag.

    Between samples :math:`u` and :math:`\hat\sigma` are held; the model part
    :math:`f` is integrated exactly from the last measured state (a double
    integrator under a constant force), and the error part
    :math:`\tilde x = \hat x - x` decays as :math:`e^{-aT}`. With no
    disturbance and an exact double-integrator plant this reproduces the
    measured state exactly, :math:`\tilde x \equiv 0` and
    :math:`\hat\sigma \equiv 0`.

    This is a deliberate sampled-data choice, **not** the exact discretisation
    of the continuous predictor above. That one would integrate
    :math:`A_s(\hat x - x(t))` against the plant's continuously varying
    :math:`x(t)`, which the controller never sees between samples; restarting
    the model part from each measured state is what makes the nominal case
    exact. The two settle at different fractions of a constant disturbance
    :math:`d`: this predictor at :math:`aT/(e^{aT} - 1)\,d` (0.847 at
    :math:`aT = 0.32`), the exact continuous one at :math:`e^{-aT} d` (0.724).

    **Adaptation.** At each sample, with :math:`T` the time since the last one,

    .. math::

        \hat\sigma(kT) = -\Phi(T)^{-1} e^{A_s T}\,\tilde x(kT), \qquad
        \Phi(T) = \int_0^T e^{A_s \tau} d\tau = \frac{1 - e^{-aT}}{a} I_6,

    the value that would cancel the current prediction error over the next
    step. :math:`\hat\sigma \in \mathbb R^6`; its velocity half is the matched
    estimate, :math:`\hat\delta = \hat m\,\hat\sigma_v` (a force), and its
    position half is unmatched and only feeds the predictor. The error
    recursion is deadbeat for every :math:`T`: there is no adaptation gain and
    no step-size limit.

    **Filter.** :math:`C(s) = \omega_c / (s + \omega_c)`, as the zero-delay
    update :math:`\delta_f \leftarrow \delta_f + (1 - e^{-\omega_c T})
    (\hat\delta - \delta_f)` using the *new* estimate :math:`\hat\delta`. That
    is the exact step for a :math:`\hat\delta` held over the step that ends
    now — one sample ahead of the zero-order-hold form, which would use the
    previous estimate.

    **Bandwidths.** Derived from the gains and :math:`\hat m` unless given:
    :math:`a = 5\,\omega_b` and :math:`\omega_c = \sqrt{\omega_b a} =
    \sqrt 5\,\omega_b`, the geometric mean, with :math:`\omega_b` from
    :func:`pd_bandwidth`. At the defaults and 0.75 kg: :math:`\omega_b =
    12.90`, :math:`a = 64.5`, :math:`\omega_c = 28.8` rad/s.

    **Unbiased estimate.** For a constant disturbance acceleration :math:`d`
    (force :math:`\hat m d`) the predictor's
    :math:`\hat\sigma_v` settles one step late at
    :math:`d\cdot aT/(e^{aT} - 1)` (0.847 of it at :math:`aT = 0.32`, the
    defaults at 5 ms). The matched estimate is scaled back by
    :math:`\kappa = (e^{aT} - 1)/(aT)`,
    :math:`\hat\delta = \kappa\,\hat m\,\hat\sigma_v`, so it equals the
    disturbance force exactly. The :math:`\hat\sigma` fed back into the
    predictor is not scaled: that would change the error recursion from
    deadbeat to :math:`e^{-aT}(1 - \kappa)` and move its fixed point off
    :math:`d` again.

    Args:
        mass: the controller's mass belief :math:`\hat m`, in kg.
        predictor_pole: :math:`a`, in rad/s. Default ``5 ω_b``.
        lpf_cutoff: :math:`\omega_c`, in rad/s. Default ``√(ω_b a)``.
        thrust_limits: ``(f_min, f_max)`` of the collective-thrust clip the
            predictor assumes, in N. Default the ``core.defaults`` limits.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bandwidth = pd_bandwidth(self._gains.kp, self._gains.kd, self.mass)
        self.thrust_limits = tuple(
            kwargs.get("thrust_limits", (DEFAULT_MIN_THRUST, DEFAULT_MAX_THRUST))
        )
        pole, cutoff = kwargs.get("predictor_pole"), kwargs.get("lpf_cutoff")
        self.predictor_pole = PREDICTOR_POLE_RATIO * bandwidth if pole is None else float(pole)
        self.lpf_cutoff = (
            float(np.sqrt(bandwidth * self.predictor_pole)) if cutoff is None else float(cutoff)
        )
        self.reset()

    def reset(self):
        """Forget the predictor and the estimate, for a new run."""
        self.control_initialized = False
        self.x_hat = np.zeros(6)
        self.sigma_hat = np.zeros(6)
        self.delta = np.zeros(3)  # matched estimate δ̂, a force
        self.lpf_delta = np.zeros(3)  # C(s) δ̂, what the control subtracts
        self._last_t = 0.0
        self._last_x = np.zeros(6)
        self._last_u = np.zeros(3)
        self._last_b3 = None  # b₃ when _last_u was applied; None: not supplied

    def compute(self, *args):
        t = args[0]
        pos, vel = args[1][0], args[1][1]
        b3 = None if len(args[1]) < 3 else np.asarray(args[1][2], dtype=float)[:, 2]
        u_pd = super().compute(t, (pos, vel))
        x = np.concatenate((pos, vel)).astype(float)

        if self.control_initialized and t < self._last_t:
            # Time ran backwards: a new run (a second `simulate()`, an env
            # reset) on a model that did not reset us. Without this every
            # sample until t passes the previous run's end has T <= 0 and
            # returns the stale u, so the loop runs open.
            self.reset()
        if not self.control_initialized:
            self.x_hat = x.copy()
            self._last_t, self._last_x = t, x
            self._last_u, self._last_b3 = u_pd.copy(), b3
            self.control_initialized = True
            return u_pd

        T = t - self._last_t
        if T <= 0.0:
            # Same instant as the last sample: nothing new to adapt on.
            return self._last_u.copy()

        # Propagate the predictor over [t - T, t] with u and σ̂ held.
        a = self.predictor_pole
        decay = np.exp(-a * T)
        phi = (1.0 - decay) / a
        p0, v0 = self._last_x[:3], self._last_x[3:]
        if self._last_b3 is None:
            force = self._last_u
        else:
            # The thrust the airframe produces: u's component along b₃, clipped
            # to the same collective-thrust limits the model's
            # `_repackage_input` applies, so saturation is not read as a
            # disturbance.
            lo, hi = self.thrust_limits
            force = float(np.clip(self._last_u @ self._last_b3, lo, hi)) * self._last_b3
        accel = force / self.mass - self._ge3
        model = np.concatenate((p0 + v0 * T + 0.5 * accel * T**2, v0 + accel * T))
        error = decay * (self.x_hat - self._last_x) + phi * self.sigma_hat
        self.x_hat = model + error

        # Piecewise-constant adaptation, then the low-pass.
        x_tilde = self.x_hat - x
        self.sigma_hat = -decay * x_tilde / phi
        # σ̂ settles one step late at aT/(e^{aT} − 1) of a constant disturbance
        # (see the class docstring). The matched estimate is scaled back by the
        # reciprocal so that δ̂ is exactly the disturbance force; σ̂ itself, fed
        # back into the predictor, is left alone — scaling it would make the
        # error recursion e^{-aT}(1 − κ) instead of deadbeat and move its fixed
        # point off the disturbance again.
        aT = a * T
        self.delta = self.mass * self.sigma_hat[3:] * (np.expm1(aT) / aT)
        alpha = 1.0 - np.exp(-self.lpf_cutoff * T)
        self.lpf_delta = self.lpf_delta + alpha * (self.delta - self.lpf_delta)

        u = u_pd - self.lpf_delta
        self._last_t, self._last_x, self._last_u, self._last_b3 = t, x, u.copy(), b3
        return u
