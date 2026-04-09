"""Geometric L1 adaptive attitude controller on SO(3).

Kotaru, Wu, Sreenath (2020) https://doi.org/10.1115/1.4045558

NOTE: Work in progress — equations being implemented by author.
"""

import numpy as np

from ...utils import hat, rodrigues_expm, vee
from .geometric_attitude import GeometricAttitudeController


class GeometricL1AttitudeController(GeometricAttitudeController):
    r"""Geometric L₁ adaptive attitude control on SO(3).

    Extends GeometricAttitudeController with L₁ adaptive disturbance
    rejection for unmodeled torque disturbances.

    From: Kotaru, Wu, Sreenath (2020) https://doi.org/10.1115/1.4045558
    """

    class State:
        def __init__(self):
            self.R = np.eye(3)
            self.Omega = np.zeros(3)
            self.last_M = np.zeros(3)
            self.last_t = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False

        self._mrac = self.State()
        self._plant = self.State()

        J = self.inertia

        # Adaptation gain Γ (diagonal, per-axis)
        self._Gamma = kwargs.get("Gamma", np.array([100.0, 100.0, 100.0]))

        # Lyapunov constant c (positive scalar, Eq. 38)
        self._c = kwargs.get("c", 1.0)

        # Projection bounds
        self._sigma_max = kwargs.get("sigma_max", 30.0)
        self._eps_proj = kwargs.get("eps", 0.5)

        # Low-pass filter cutoff
        self._omega_c = kwargs.get("lpf_cutoff", 5.0)

        # Disturbance estimate and filtered version
        self._sigma_hat = np.zeros(3)
        self._sigma_hat_dot = np.zeros(3)
        self._sigma_filtered = np.zeros(3)

    def _projection(self, theta_hat, y):
        r"""Γ-Projection operator (Eq. 42, Remark 5).

        Proj_Γ(θ̂, y) = {
            Γy - Γ ∇f (∇f)ᵀ / ((∇f)ᵀ ∇f) Γy f(θ̂),  if f(θ̂) > 0 and yᵀΓ∇f > 0
            Γy,                                          otherwise
        }
        where f(θ̂) = (‖θ̂‖² - θ_max²) / (ε θ_max²)
        """
        Gamma = np.diag(self._Gamma)
        Gy = Gamma @ y
        grad_f = 2 * theta_hat / (self._eps_proj * self._sigma_max**2)
        f = (np.dot(theta_hat, theta_hat) - self._sigma_max**2) / (
            self._eps_proj * self._sigma_max**2
        )

        if f > 0 and y @ Gamma @ grad_f > 0:
            grad_norm_sq = np.dot(grad_f, grad_f)
            if grad_norm_sq < 1e-10:
                return Gy
            return Gy - (Gamma @ grad_f) * (grad_f @ Gy) / grad_norm_sq * f
        return Gy

    def compute(self, *args):
        """Compute thrust and torque with L₁ adaptive disturbance rejection.

        Args: t, (R, Omega), thrust_force
        Returns: (f, M) scalar thrust and 3D torque
        """
        t = args[0]
        R, Omega = args[1][0], args[1][1]
        thrust_force = args[2]
        J = self.inertia
        Jinv = self._inertia_inv

        # Desired attitude from thrust direction
        Rd = self._cmd_accel_to_cmd_att(thrust_force)
        Omegad = np.zeros(3)
        dOmegad = np.zeros(3)

        dt = t - self._plant.last_t
        if dt <= 0:
            dt = 1e-6

        # Integrate MRAC reference model (using previous M_mrac)
        if self._initialized:
            self._mrac.R = self._mrac.R @ rodrigues_expm(self._mrac.Omega * dt)
            dOmega_m = Jinv @ (self._mrac.last_M - np.cross(self._mrac.Omega, J @ self._mrac.Omega))
            self._mrac.Omega += dOmega_m * dt
        else:
            self._mrac.R = R.copy()
            self._mrac.Omega = Omega.copy()
            self._initialized = True

        # Plant errors: eR, eΩ from (R, Rd)
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        eOmega = Omega - R.T @ Rd @ Omegad

        # MRAC errors: eR_m, eΩ_m from (R_m, Rd)
        eR_mrac = 0.5 * vee(Rd.T @ self._mrac.R - self._mrac.R.T @ Rd)
        eOmega_mrac = self._mrac.Omega - self._mrac.R.T @ Rd @ Omegad

        # Plant-MRAC error (geometric on SO(3), Eq. 30-31)
        eR_tilde = 0.5 * vee(R.T @ self._mrac.R - self._mrac.R.T @ R)
        eOmega_tilde = self._mrac.Omega - self._mrac.R.T @ R @ Omega

        # Adaptation law (Eq. 37-39)
        P = J @ self._mrac.R.T @ R @ Jinv @ R.T @ self._mrac.R
        y = -(P.T @ eOmega_tilde + self._c * P.T @ Jinv.T @ eR_tilde)
        self._sigma_hat_dot = self._projection(self._sigma_hat, y)
        self._sigma_hat += self._sigma_hat_dot * dt
        # Hard clip as safety (projection should handle this, but Euler overshoot can escape)
        sigma_norm = np.linalg.norm(self._sigma_hat)
        if sigma_norm > self._sigma_max:
            self._sigma_hat *= self._sigma_max / sigma_norm

        # Low-pass filter C(s) = a/(s+a), Euler integration of ẋ = -a x + a θ̂
        a = self._omega_c
        self._sigma_filtered += dt * (-a * self._sigma_filtered + a * self._sigma_hat)

        # MRAC moment (Eq. 34)
        mrac_mu1 = -self._gains.kp * eR_mrac - self._gains.kd * eOmega_mrac
        mrac_mu2 = (
            J @ self._mrac.R.T @ R @ Jinv @ R.T @ self._mrac.R @ self._sigma_hat
            - self._sigma_filtered
        )
        M_mrac = (
            mrac_mu1
            + mrac_mu2
            + np.cross(self._mrac.Omega, J @ self._mrac.Omega)
            - J
            @ (hat(self._mrac.Omega) @ self._mrac.R.T @ Rd @ Omegad - self._mrac.R.T @ Rd @ dOmegad)
        )

        # Plant moment
        mu1 = (
            J
            @ R.T
            @ self._mrac.R
            @ Jinv
            @ (
                mrac_mu1
                - self._sigma_filtered
                + self._gains.kp * eR_tilde
                + self._gains.kd * eOmega_tilde
            )
        )
        mu2 = J @ R.T @ self._mrac.R @ hat(eOmega_tilde) @ self._mrac.R.T @ R @ eOmega
        M = (
            mu1
            + mu2
            + np.cross(Omega, J @ Omega)
            - J @ (hat(Omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ dOmegad)
        )

        # Store for next step
        self._mrac.last_M = M_mrac
        self._mrac.last_t = t
        self._plant.R = R
        self._plant.Omega = Omega
        self._plant.last_M = M
        self._plant.last_t = t

        # Diagnostics
        self.sigma_hat = self._sigma_hat.copy()
        self.sigma_filtered = self._sigma_filtered.copy()

        f = thrust_force.dot(R[:, 2])
        return f, M
