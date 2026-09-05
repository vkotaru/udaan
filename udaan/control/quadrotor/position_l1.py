"""L1 adaptive position controller for quadrotor.

.. warning::

   :class:`PositionL1Controller` is **experimental** and is not a working
   implementation of the law it is named after. See the class docstring.
"""

import warnings

import numpy as np

from .position_pd import PositionPDController


class PositionL1Controller(PositionPDController):
    """Quadrotor position controller using L1 adaption. **Experimental.**

    .. warning::

       **This controller has never adapted, and no gain setting makes it beat
       the PD baseline.** It is shipped as a starting point for the control
       work needed to finish it, not as a usable controller. Use
       :class:`~udaan.control.quadrotor.PositionPDController` for anything that
       has to fly.

       Two independent defects, found by instrumenting a live run:

       1. ``compute()`` assigns ``self.last_t = t`` at the *top* of the method,
          before ``dt = t - self.last_t`` is formed below it, so ``dt`` is
          always ``0`` and is clamped to ``1e-6``. The adaptation integrator,
          the MRAC reference model and the low-pass filter are all frozen.
       2. The MRAC reference state used to be *bound* to the plant's own arrays
          rather than copied (``self.mrac_position = pos``). On
          :class:`~udaan.models.quadrotor.QuadrotorBase` — the analytical
          backend the example, the README and the docs all use — that made
          ``mrac_eta`` identically equal to ``eta``, hence ``eta_tilde ≡ 0``
          and ``delta ≡ 0`` *exactly*, for the whole run and at any ``Gamma``.
          It also let the controller write into the plant's state: a 25 s run
          had 4.2e-4 m of position and 4.4e-2 m/s of velocity injected into
          the model by its own controller. Defect 2 is **fixed** (the states
          are copied); it was a correctness bug regardless of the adaptation.

       Because of defect 2, the improvement previously quoted for the one-line
       ``dt`` fix (0.2451 m → 0.1606 m of hover sag) was not adaptation — with
       that fix alone ``delta`` still ends at exactly ``0``, and what moved the
       number was the controller corrupting the plant's velocity by up to
       0.143 m/s in a single step. With the aliasing repaired, deleting the
       ``dt`` line as well makes the run diverge to NaN, and sweeping ``Gamma``
       down (1000 / 200 / 100) leaves 0.55–0.64 m of error against the PD
       baseline's 0.303 m. So the ``dt`` line is deliberately left in place:
       there is no known gain at which this law, as written, works.

       Finishing it is control-design work — most likely replacing the explicit
       Euler adaptation update with the piecewise-constant adaptation law, then
       retuning ``Gamma`` — plus the regression test the controller has never
       had.

    See :doc:`/theory/controllers/geometric-l1` for the derivation and the full
    write-up of both defects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # `warnings`, not the module logger: the library installs a NullHandler
        # by default, so a logged warning would never reach anyone who has not
        # called `udaan.utils.logging.setup_logging()` — and this one has to be
        # seen before someone flies with it.
        warnings.warn(
            "PositionL1Controller is experimental and does not work: its adaptation "
            "is frozen by a known defect, and at no gain does it outperform "
            "PositionPDController. See the class docstring.",
            UserWarning,
            stacklevel=2,
        )
        self.control_initialized = False

        # MRAC reference states
        self.mrac_position = np.array([0.0, 0.0, 0.0])
        self.mrac_velocity = np.array([0.0, 0.0, 0.0])

        self.Gamma = np.array([1000.0, 1000.0, 100000.0])

        I3, O3 = np.eye(3), np.zeros((3, 3))
        self.F = np.concatenate(
            (np.concatenate((O3, I3), axis=1), np.concatenate((O3, O3), axis=1))
        )
        self.G = np.concatenate((O3, I3)) * (1 / self.mass)

        import control as ctrl

        self.K, self.P, _ = ctrl.lqr(self.F, self.G, np.eye(6), np.eye(3) * 1e-2)
        self.yGain = -self.G.T @ self.P

        # MRAC parameters
        self.deltamax = 30.0
        self.eps = 0.5
        self.lpf_cutoff = 5.0  # Low-pass filter cutoff frequency [Hz]

        # Unmodeled disturbance states.
        self.delta = np.zeros(3)
        self.lpf_delta = np.zeros(3)
        self.delta_dot = np.zeros(3)

        self.last_t = 0.0
        self.last_mrac_input = np.zeros(3)

    def compute(self, *args):
        t = args[0]
        # KNOWN DEFECT — deliberately left in place; see the class docstring and
        # the CHANGELOG's "Known issues". This assignment makes `dt` below
        # identically zero (it is clamped to 1e-6), which freezes the adaptation
        # integrator, the MRAC reference model and the low-pass filter: the
        # controller reduces to its LQR baseline. The correct assignment already
        # exists at the end of this method, so deleting this line is a one-line
        # change — but it does NOT unfreeze the adaptation and it is not safe:
        #
        #   * with this line deleted and nothing else changed, `delta` still
        #     ends a 25 s run at exactly 0 (the aliasing defect, now fixed
        #     below, was what actually held it at zero);
        #   * with both defects fixed, the run diverges to NaN — at the real
        #     5 ms step the gain Gamma (up to 1e5) gives Gamma*dt ~ 500, far
        #     outside the stability limit of this explicit Euler update — and
        #     `udaan run fleet --demo l1-comparison` NaNs at t = 0.34 s;
        #   * sweeping Gamma down (1000 / 200 / 100) keeps it finite but leaves
        #     0.55-0.64 m of hover error against the PD baseline's 0.303 m.
        #
        # There is no known gain at which this law, as written, works, so the
        # fix is a redesign of the adaptation update plus a regression test —
        # control-design work, not a packaging fix.
        self.last_t = t
        pos, vel = args[1][0], args[1][1]
        # Desired (reference) trajectory
        pos_d, vel_d, acc_d = self.setpoint(t)

        if not self.control_initialized:
            # Copy, never bind: `pos`/`vel` are the model's own state arrays.
            # Binding made the MRAC reference *be* the plant state, so
            # `eta_tilde` was identically zero and the adaptation could never
            # run; worse, the in-place `+=` updates below then wrote straight
            # into the plant, injecting 4.2e-4 m and 4.4e-2 m/s over a 25 s run
            # on QuadrotorBase. (The MuJoCo backend happened to be immune:
            # `_query_latest_state()` rebuilds these arrays every step.)
            self.mrac_position = np.array(pos, dtype=float)
            self.mrac_velocity = np.array(vel, dtype=float)
            self.last_t = t
            self.control_initialized = True
            return np.zeros(3)

        # Compute time step
        dt = t - self.last_t
        if dt <= 0:
            dt = 1e-6

        # Update MRAC reference model states
        self.mrac_position += self.mrac_velocity * dt + 0.5 * self.last_mrac_input * dt * dt
        self.mrac_velocity += self.last_mrac_input * dt

        # Update disturbance estimate
        self.delta += self.delta_dot * dt

        # Low-pass filter: lpf_delta += dt * cutoff * (delta - lpf_delta)
        alpha = min(1.0, dt * self.lpf_cutoff)
        self.lpf_delta = self.lpf_delta + alpha * (self.delta - self.lpf_delta)

        # Compute errors
        eta = np.concatenate((pos - pos_d, vel - vel_d))
        mrac_eta = np.concatenate((self.mrac_position - pos_d, self.mrac_velocity - vel_d))

        # Compute input
        u = -self.K @ eta + self.mass * (acc_d + self._ge3)
        u_mrac = -self.K @ mrac_eta + self.mass * (acc_d + self._ge3)

        # Remove unmodeled disturbance
        u -= self.lpf_delta
        u_mrac += self.delta - self.lpf_delta

        # Projection operator for disturbance prediction
        eta_tilde = mrac_eta - eta
        y = self.yGain @ eta_tilde

        f_ = (np.linalg.norm(self.delta) ** 2 - self.deltamax**2) / (self.eps * self.deltamax**2)
        df = 2 * self.delta / (self.eps * self.deltamax**2)
        df_norm_sq = np.dot(df, df)

        if f_ > 0 and np.dot(y, df) > 0 and df_norm_sq > 1e-10:
            proj = y - (np.dot(df, y) / df_norm_sq) * df * f_
        else:
            proj = y

        self.delta_dot = self.Gamma * proj

        # Update internal state
        self.last_mrac_input = u_mrac
        self.last_t = t

        self.pos_setpoint = pos_d
        return u
