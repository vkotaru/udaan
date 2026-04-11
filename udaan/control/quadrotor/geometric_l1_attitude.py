"""Geometric L1 adaptive attitude controller on SO(3).

Kotaru, Edmonson, Sreenath (2020) https://doi.org/10.1115/1.4045558
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.types import Vec3
from ...manif import SO3, TSO3
from ...manif.utils import hat
from .geometric_attitude import GeometricAttitudeController


@dataclass
class AttitudeState:
    """Internal attitude state for reference model tracking."""

    R: SO3 = field(default_factory=SO3)
    Omega: TSO3 = field(default_factory=TSO3)
    last_M: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_t: float = 0.0


class GeometricL1AttitudeController(GeometricAttitudeController):
    r"""Geometric L1 adaptive attitude control on SO(3).

    Extends GeometricAttitudeController with L1 adaptive disturbance
    rejection for unmodeled torque disturbances.

    From: Kotaru, Edmonson, Sreenath (2020) https://doi.org/10.1115/1.4045558
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initialized: bool = False

        self._mrac: AttitudeState = AttitudeState()
        self._plant: AttitudeState = AttitudeState()

        # Adaptation gain Gamma (diagonal, per-axis)
        self._Gamma: Vec3 = kwargs.get("Gamma", np.array([100.0, 100.0, 100.0]))

        # Lyapunov constant c (positive scalar, Eq. 38)
        self._c: float = kwargs.get("c", 1.0)

        # Projection bounds
        self._sigma_max: float = kwargs.get("sigma_max", 30.0)
        self._eps_proj: float = kwargs.get("eps", 0.5)

        # Low-pass filter cutoff
        self._omega_c: float = kwargs.get("lpf_cutoff", 5.0)

        # Disturbance estimate and filtered version
        self._sigma_hat: Vec3 = np.zeros(3)
        self._sigma_hat_dot: Vec3 = np.zeros(3)
        self._sigma_filtered: Vec3 = np.zeros(3)

    def reset(self) -> None:
        """Reset adaptation state for a new simulation."""
        self._initialized = False
        self._mrac = AttitudeState()
        self._plant = AttitudeState()
        self._sigma_hat = np.zeros(3)
        self._sigma_hat_dot = np.zeros(3)
        self._sigma_filtered = np.zeros(3)

    def _projection(self, theta_hat: Vec3, y: Vec3) -> Vec3:
        r"""Gamma-Projection operator (Eq. 42, Remark 5)."""
        Gamma: np.ndarray = np.diag(self._Gamma)
        Gy: Vec3 = Gamma @ y
        grad_f: Vec3 = 2 * theta_hat / (self._eps_proj * self._sigma_max**2)
        f: float = (np.dot(theta_hat, theta_hat) - self._sigma_max**2) / (
            self._eps_proj * self._sigma_max**2
        )

        if f > 0 and y @ Gamma @ grad_f > 0:
            grad_norm_sq: float = np.dot(grad_f, grad_f)
            if grad_norm_sq < 1e-10:
                return Gy
            return Gy - (Gamma @ grad_f) * (grad_f @ Gy) / grad_norm_sq * f
        return Gy

    def compute(
        self,
        t: float,
        state: tuple[SO3, TSO3],
        thrust_force: Vec3,
        desired_att: tuple[SO3, TSO3, Vec3] | None = None,
    ) -> tuple[float, Vec3]:
        """Compute thrust and torque with L1 adaptive disturbance rejection.

        Args:
            t: current time.
            state: (R, Omega) — SO3 rotation and TSO3 angular velocity.
            thrust_force: desired thrust force vector in world frame.
            desired_att: optional (Rd, Omegad, dOmegad) feedforward.

        Returns:
            (f, M) — scalar thrust and 3D torque vector.
        """
        R: SO3 = state[0]
        Omega: TSO3 = state[1]
        J: np.ndarray = self.inertia
        Jinv: np.ndarray = self._inertia_inv

        # Desired attitude
        Rd: SO3 = self._cmd_accel_to_cmd_att(thrust_force)
        Omegad: TSO3
        dOmegad: Vec3
        if desired_att is not None:
            _, Omegad, dOmegad = desired_att
        else:
            Omegad = TSO3()
            dOmegad = np.zeros(3)

        dt: float = t - self._plant.last_t
        if dt <= 0:
            dt = 1e-6

        # Integrate MRAC reference model
        if self._initialized:
            self._mrac.R = self._mrac.R.step(self._mrac.Omega * dt)
            Om_m_arr: Vec3 = self._mrac.Omega.vector
            dOmega_m: Vec3 = Jinv @ (self._mrac.last_M - np.cross(Om_m_arr, J @ Om_m_arr))
            self._mrac.Omega += TSO3(dOmega_m * dt)
        else:
            self._mrac.R = SO3(R)
            self._mrac.Omega = TSO3(Omega)
            self._initialized = True

        # Geometric errors using SO3/TSO3 operators
        eR: TSO3 = R - Rd
        eOm: TSO3 = Omega - Omegad.transport(Rd, R)

        eR_mrac: TSO3 = self._mrac.R - Rd
        eOm_mrac: TSO3 = self._mrac.Omega - Omegad.transport(Rd, self._mrac.R)

        # Plant-MRAC error (Eq. 30-31)
        eR_tilde: TSO3 = self._mrac.R - R
        eOm_tilde: TSO3 = self._mrac.Omega - Omega.transport(R, self._mrac.R)

        # Precompute rotation products for control law
        RtRm: np.ndarray = (R.T @ self._mrac.R).arr
        RmtR: np.ndarray = RtRm.T
        RtRd: np.ndarray = (R.T @ Rd).arr
        RmtRd: np.ndarray = (self._mrac.R.T @ Rd).arr

        # Adaptation law (Eq. 37-39)
        P: np.ndarray = J @ RmtR @ Jinv @ RtRm
        y: Vec3 = -(P.T @ eOm_tilde.vector + self._c * P.T @ Jinv.T @ eR_tilde.vector)
        self._sigma_hat_dot = self._projection(self._sigma_hat, y)
        self._sigma_hat += self._sigma_hat_dot * dt
        # Hard clip as safety
        sigma_norm: float = np.linalg.norm(self._sigma_hat)
        if sigma_norm > self._sigma_max:
            self._sigma_hat *= self._sigma_max / sigma_norm

        # Low-pass filter C(s) = a/(s+a)
        a: float = self._omega_c
        self._sigma_filtered += dt * (-a * self._sigma_filtered + a * self._sigma_hat)

        # MRAC moment (Eq. 34)
        Om_m: Vec3 = self._mrac.Omega.vector
        Omd: Vec3 = Omegad.vector
        mrac_mu1: Vec3 = -self._gains.kp * eR_mrac.vector - self._gains.kd * eOm_mrac.vector
        mrac_mu2: Vec3 = J @ RmtR @ Jinv @ RtRm @ self._sigma_hat - self._sigma_filtered
        M_mrac: Vec3 = (
            mrac_mu1
            + mrac_mu2
            + np.cross(Om_m, J @ Om_m)
            - J @ (hat(Om_m) @ RmtRd @ Omd - RmtRd @ dOmegad)
        )

        # Plant moment
        Om_p: Vec3 = Omega.vector
        mu1: Vec3 = (
            J
            @ RtRm
            @ Jinv
            @ (
                mrac_mu1
                - self._sigma_filtered
                + self._gains.kp * eR_tilde.vector
                + self._gains.kd * eOm_tilde.vector
            )
        )
        mu2: Vec3 = J @ RtRm @ hat(eOm_tilde.vector) @ RmtR @ eOm.vector
        M: Vec3 = (
            mu1 + mu2 + np.cross(Om_p, J @ Om_p) - J @ (hat(Om_p) @ RtRd @ Omd - RtRd @ dOmegad)
        )

        # Store for next step
        self._mrac.last_M = M_mrac
        self._mrac.last_t = t
        self._plant.R = R
        self._plant.Omega = Omega
        self._plant.last_M = M
        self._plant.last_t = t

        # Diagnostics
        self.sigma_hat: Vec3 = self._sigma_hat.copy()
        self.sigma_filtered: Vec3 = self._sigma_filtered.copy()

        f: float = thrust_force.dot(R[:, 2])
        return f, M
