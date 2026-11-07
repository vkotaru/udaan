"""Geometric L1 adaptive attitude controller on SO(3).

Kotaru, Edmonson, Sreenath (2020) https://doi.org/10.1115/1.4045558
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...core.types import Vec3
from ...manif import SO3, TSO3
from ...manif.utils import hat
from ...utils.logging import get_logger
from .geometric_attitude import GeometricAttitudeController

_logger = get_logger(__name__)


def routh_bound(kR: Vec3, kOmega: Vec3, Gamma: Vec3) -> Vec3:
    r"""Per-axis upper bound on the adaptation coupling ``c`` for stability.

    Linearised about perfect prediction, one body axis of the
    predictor–adaptation loop is

    .. math::

        J\dot{\tilde\omega} = \hat\sigma - k_R\tilde\theta - k_\Omega\tilde\omega,
        \qquad
        \dot{\hat\sigma} = -\Gamma\,(\tilde\omega + c\,J^{-1}\tilde\theta),

    with characteristic polynomial
    :math:`J s^3 + k_\Omega s^2 + (k_R + \Gamma) s + \Gamma c / J`. Routh's
    condition :math:`k_\Omega (k_R + \Gamma) > J \cdot \Gamma c / J` has the
    inertia cancel exactly, so the bound

    .. math:: c < k_\Omega (k_R + \Gamma) / \Gamma

    depends on the gains alone, not on the airframe.
    """
    kOmega, kR, Gamma = (np.asarray(x, dtype=float) for x in (kOmega, kR, Gamma))
    return np.asarray(kOmega * (kR + Gamma) / Gamma, dtype=float)


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
        self._Gamma: Vec3 = np.asarray(kwargs.get("Gamma", [100.0, 100.0, 100.0]), dtype=float)

        # Adaptation coupling c (Eq. 38), per body axis. ``None`` means derive it
        # from the gains at every call — half the Routh bound — so it follows a
        # gain change. Above the bound the predictor–adaptation loop is
        # unstable. An explicit ``c`` is kept as given and checked against the
        # bound.
        c = kwargs.get("c")
        self._c_explicit: Vec3 | None = (
            None if c is None else np.broadcast_to(np.asarray(c, dtype=float), (3,)).copy()
        )
        self._c_checked_for: tuple | None = None

        # Projection bounds
        self._sigma_max: float = kwargs.get("sigma_max", 30.0)
        self._eps_proj: float = kwargs.get("eps", 0.5)

        # Low-pass filter cutoff
        self._omega_c: float = kwargs.get("lpf_cutoff", 5.0)

        # Disturbance estimate and filtered version
        self._sigma_hat: Vec3 = np.zeros(3)
        self._sigma_hat_dot: Vec3 = np.zeros(3)
        self._sigma_filtered: Vec3 = np.zeros(3)

        self.c  # noqa: B018 — checks an explicit c against the gains now, not on first use

    @property
    def c(self) -> Vec3:
        """The per-axis adaptation coupling in use (see :func:`routh_bound`)."""
        bound = routh_bound(self._gains.kp, self._gains.kd, self._Gamma)
        if self._c_explicit is None:
            return 0.5 * bound
        key = (tuple(self._gains.kp), tuple(self._gains.kd), tuple(self._Gamma))
        if key != self._c_checked_for:
            self._c_checked_for = key
            if np.any(self._c_explicit >= bound):
                _logger.warning(
                    "GeometricL1AttitudeController: c = %s violates the Routh stability "
                    "bound c < k_Omega (k_R + Gamma) / Gamma = %s; the predictor-adaptation "
                    "loop is unstable on those axes.",
                    np.array2string(self._c_explicit, precision=4),
                    np.array2string(bound, precision=4),
                )
        return self._c_explicit

    def reset(self) -> None:
        """Reset adaptation state for a new simulation."""
        self._initialized = False
        self._mrac = AttitudeState()
        self._plant = AttitudeState()
        self._sigma_hat = np.zeros(3)
        self._sigma_hat_dot = np.zeros(3)
        self._sigma_filtered = np.zeros(3)

    def _projection(self, theta_hat: Vec3, y: Vec3) -> Vec3:
        r"""Gamma-Projection operator (Lavretsky & Gibson, arXiv:1112.4232, Def. 11).

        The denominator is :math:`\nabla f^\top\Gamma\nabla f`, as in that
        definition (the paper's Eq. 42 prints :math:`\nabla f^\top\nabla f`).
        In practice the hard clip in :meth:`compute` keeps :math:`\hat\sigma`
        inside the ball, so this branch rarely fires.
        """
        Gamma: np.ndarray = np.diag(self._Gamma)
        Gy: Vec3 = Gamma @ y
        grad_f: Vec3 = 2 * theta_hat / (self._eps_proj * self._sigma_max**2)
        f: float = (np.dot(theta_hat, theta_hat) - self._sigma_max**2) / (
            self._eps_proj * self._sigma_max**2
        )

        if f > 0 and y @ Gamma @ grad_f > 0:
            grad_weighted_sq: float = grad_f @ Gamma @ grad_f
            if grad_weighted_sq < 1e-10:
                return Gy
            return Gy - (Gamma @ grad_f) * (grad_f @ Gy) / grad_weighted_sq * f
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
            desired_att: optional (Rd, Omegad, dOmegad) feedforward. ``Rd``
                contributes its heading only, exactly as in
                :meth:`GeometricAttitudeController.compute`; ``b₃`` stays with
                the thrust command so the position loop keeps its path in.

        Returns:
            (f, M) — scalar thrust and 3D torque vector.
        """
        R: SO3 = state[0]
        Omega: TSO3 = state[1]
        J: np.ndarray = self.inertia
        Jinv: np.ndarray = self._inertia_inv

        # Desired attitude: b3 from the thrust command, heading from the
        # reference attitude when there is one. See
        # :meth:`GeometricAttitudeController._cmd_accel_to_cmd_att`.
        Rd: SO3 = self._cmd_accel_to_cmd_att(
            thrust_force, reference=None if desired_att is None else desired_att[0]
        )
        Omegad: TSO3
        dOmegad: Vec3
        if desired_att is not None:
            _, Omegad, dOmegad = desired_att
        else:
            Omegad = TSO3()
            dOmegad = np.zeros(3)

        if self._initialized and t < self._plant.last_t:
            # Time ran backwards: a new run on a model that did not reset us.
            # Carrying the predictor and σ̂ across would start the run from the
            # previous one's end state.
            self.reset()
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
        y: Vec3 = -(P.T @ eOm_tilde.vector + P.T @ Jinv.T @ (self.c * eR_tilde.vector))
        self._sigma_hat_dot = self._projection(self._sigma_hat, y)
        self._sigma_hat += self._sigma_hat_dot * dt
        # Hard clip: this, not the projection, is what bounds ||sigma_hat|| (the
        # projection's band lies outside sigma_max, so it starts inside the ball).
        sigma_norm: float = float(np.linalg.norm(self._sigma_hat))
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
