"""L1 adaptive position controller for quadrotor."""

import numpy as np

from .position_pd import PositionPDController


class PositionL1Controller(PositionPDController):
    """Quadrotor position controller using L1 adaption."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.last_t = t
        pos, vel = args[1][0], args[1][1]
        # Desired (reference) trajectory
        pos_d, vel_d, acc_d = self.setpoint(t)

        if not self.control_initialized:
            self.mrac_position = pos
            self.mrac_velocity = vel
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
