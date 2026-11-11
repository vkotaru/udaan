"""Payload position + cable attitude controller.

Sreenath, Lee, Kumar (2013) https://ieeexplore.ieee.org/abstract/document/6760219
"""

import numpy as np

from ...control import Gains, PDController
from ...core.defaults import (
    DEFAULT_M1_QUAD_INERTIA,
    DEFAULT_M1_QUAD_MASS,
    DEFAULT_M2_CABLE_LENGTH,
    DEFAULT_M2_PAYLOAD_MASS,
    MIN_CABLE_TENSION_FRACTION,
)
from ...core.validation import MIN_PHYSICAL_MASS, check_positive
from ...flatness.quadrotor_cspayload import cable_direction_jet
from ...utils import hat
from .gains import CableCascadeGains, cable_cascade_gains

_GAIN_KEYWORDS = ("position_kp", "position_kd", "cable_kp", "cable_kd")


class QuadCSPayloadController(PDController):
    """Geometric payload-position and cable-attitude controller.

    Returns the thrust-force vector ``F`` (world frame) that the quadrotor's
    attitude loop then tracks.

    Keyword Args:
        mass_q: quadrotor mass ``m_Q``, in kg (default
            :data:`~udaan.core.defaults.DEFAULT_M1_QUAD_MASS`).
        mass_l: mass ``m_L`` at the cable's end, in kg (default
            :data:`~udaan.core.defaults.DEFAULT_M2_PAYLOAD_MASS`): the payload,
            plus the cable's own mass where the cable has one.
        cable_length: cable length ``ℓ``, in m (default
            :data:`~udaan.core.defaults.DEFAULT_M2_CABLE_LENGTH`).
        inertia: quadrotor inertia ``J``, ``(3,)`` or ``(3, 3)`` in kg·m²
            (default :data:`~udaan.core.defaults.DEFAULT_M1_QUAD_INERTIA`).

        position_kp: payload-loop gain ``k_x``, ``(3,)`` in N/m.
        position_kd: payload-loop gain ``k_v``, ``(3,)`` in N·s/m.
        cable_kp: cable-loop gain ``k_q``, ``(3,)`` in N.
        cable_kd: cable-loop gain ``k_ω``, ``(3,)`` in N·s.

    Gains not given are :func:`~udaan.control.quadrotor_cspayload.cable_cascade_gains`
    of the machine (the time-scale design: the payload loop five times slower
    than the cable loop, the cable loop tied to the pendulum frequency and five
    times slower than the attitude loop, ζ = 0.7), recomputed whenever the
    parameters change, so they follow the cable length and the masses. A gain
    given explicitly is kept as given.

    The remaining keywords (``setpoint``) are
    :class:`~udaan.control.PDController`'s. A model's default controller is
    built with that model's parameters, and :meth:`sync_from_model` re-reads
    them.

    The command along the cable is floored at the tension
    ``T_min = MIN_CABLE_TENSION_FRACTION · m_L · g``
    (:data:`~udaan.core.defaults.MIN_CABLE_TENSION_FRACTION`): see
    :meth:`compute`.
    """

    def __init__(
        self,
        *,
        mass_q=DEFAULT_M1_QUAD_MASS,
        mass_l=DEFAULT_M2_PAYLOAD_MASS,
        cable_length=DEFAULT_M2_CABLE_LENGTH,
        inertia=DEFAULT_M1_QUAD_INERTIA,
        position_kp=None,
        position_kd=None,
        cable_kp=None,
        cable_kd=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        given = (position_kp, position_kd, cable_kp, cable_kd)
        self._fixed_gains = {
            name: np.array(value, dtype=float) * np.ones(3)
            for name, value in zip(_GAIN_KEYWORDS, given, strict=True)
            if value is not None
        }
        self._gain_pos = Gains(kp=np.zeros(3), kd=np.zeros(3))
        self._gain_cable = Gains(kp=np.zeros(3), kd=np.zeros(3))
        self._set_parameters(mass_q, mass_l, cable_length, inertia)

    def _set_parameters(self, mass_q, mass_l, cable_length, inertia):
        check_positive(mass_q, "mass_q", warn_below=MIN_PHYSICAL_MASS)
        check_positive(mass_l, "mass_l", warn_below=MIN_PHYSICAL_MASS)
        check_positive(cable_length, "cable_length")
        J: np.ndarray = np.array(inertia, dtype=float)
        if J.ndim == 1:
            J = np.diag(J)
        if J.shape != (3, 3):
            raise ValueError(f"inertia must be (3,) or (3, 3); got {J.shape}")
        self._quad_mass = float(mass_q)
        self._payload_mass = float(mass_l)
        self._cable_length = float(cable_length)
        self._quad_inertia = J
        self._quad_inertia_inv = np.linalg.inv(J)
        self._design_gains = cable_cascade_gains(mass_q, mass_l, cable_length, J)
        designed = {
            "position_kp": self._design_gains.position_kp,
            "position_kd": self._design_gains.position_kd,
            "cable_kp": self._design_gains.cable_kp,
            "cable_kd": self._design_gains.cable_kd,
        }
        gains = {name: self._fixed_gains.get(name, designed[name]).copy() for name in designed}
        self._gain_pos.kp, self._gain_pos.kd = gains["position_kp"], gains["position_kd"]
        self._gain_cable.kp, self._gain_cable.kd = gains["cable_kp"], gains["cable_kd"]

    def sync_from_model(self, model):
        """Take ``m_Q``, ``m_L``, ``ℓ`` and ``J`` from ``model``.

        ``model`` is a :class:`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase`
        or one of its backends, whose ``mass``, ``payload_mass``,
        ``cable_mass``, ``cable_length`` and ``inertia`` are the machine it
        simulates. The controller's ``m_L`` is ``payload_mass + cable_mass``:
        a cable with mass of its own is carried as part of the payload, so the
        hover feed-forward ``(m_Q + m_L) g`` is the machine's whole weight. The
        gains not given explicitly are recomputed for it.
        """
        self._set_parameters(
            model.mass,
            model.payload_mass + model.cable_mass,
            model.cable_length,
            model.inertia,
        )

    @property
    def design_gains(self) -> CableCascadeGains:
        """The time-scale design's gains for this controller's machine.

        Its ``attitude_kp``/``attitude_kd`` are what the model writes into its
        own attitude controller; the payload and cable gains are this
        controller's, except where given explicitly.
        """
        return self._design_gains

    @property
    def position_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """``(k_x, k_v)`` in use, each ``(3,)`` (copies)."""
        return self._gain_pos.kp.copy(), self._gain_pos.kd.copy()

    @property
    def cable_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """``(k_q, k_ω)`` in use, each ``(3,)`` (copies)."""
        return self._gain_cable.kp.copy(), self._gain_cable.kd.copy()

    @property
    def mass_q(self) -> float:
        """Quadrotor mass ``m_Q`` this controller is built for, in kg."""
        return self._quad_mass

    @property
    def mass_l(self) -> float:
        """Mass ``m_L`` at the cable's end this controller is built for, in kg."""
        return self._payload_mass

    @property
    def cable_length(self) -> float:
        """Cable length ``ℓ`` this controller is built for, in m."""
        return self._cable_length

    @property
    def inertia(self) -> np.ndarray:
        """Quadrotor inertia ``J`` this controller is built for, ``(3, 3)`` in kg·m²."""
        return np.array(self._quad_inertia, dtype=float)

    @property
    def min_tension(self) -> float:
        """``T_min = MIN_CABLE_TENSION_FRACTION · m_L · g``, in N."""
        return MIN_CABLE_TENSION_FRACTION * self._payload_mass * self._g

    def compute(self, *args):
        """Thrust-force vector for the payload state ``args[1]`` at time ``args[0]``.

        Payload loop: ``A = (m_Q + m_L)(ẍ_Ld + g e₃) + m_Q ℓ ‖q̇‖² q − k_x e_x − k_v e_v``
        and the cable command ``q_c = −A/‖A‖``. Cable loop: the component of
        ``F`` orthogonal to ``q`` tracks ``q_c`` (with ``q̇_d``, ``q̈_d`` from
        the reference's flat map when the setpoint returns jerk and snap), and
        the component along ``q`` is the tension the payload loop asks for,
        ``T = −A·q`` (``q`` points from the airframe to the payload, so
        ``T = ‖A‖`` at hover), floored at :attr:`min_tension`:
        ``F_n = −max(T, T_min) q``. The floor keeps the commanded thrust
        pointing up the cable when the payload loop asks for a net downward
        force, rather than through the airframe's underside.
        """
        t = args[0]
        s = args[1]  # quadrotor payload state
        sp = self.setpoint(t)
        sd, dsd, d2sd = sp[0], sp[1], sp[2]
        # Higher payload derivatives (jerk, snap) feed the cable-attitude
        # feedforward below; default to zero when the setpoint omits them
        # (e.g. a stationary set-point), which disables that feedforward.
        d3sd = sp[3] if len(sp) > 3 else np.zeros(3)
        d4sd = sp[4] if len(sp) > 4 else np.zeros(3)

        ex = s.payload_position - sd
        ev = s.payload_velocity - dsd

        q = np.asarray(s.cable_attitude)
        dq = s.dq()

        mQ = self._quad_mass
        mL = self._payload_mass
        l = self._cable_length

        Fff = (mQ + mL) * (d2sd + self._ge3) + mQ * l * np.dot(dq, dq) * q
        Fpd = -self._gain_pos.kp * ex - self._gain_pos.kd * ev
        A = Fff + Fpd

        #  desired load attitude
        qc = -A / np.linalg.norm(A)
        # load-attitude ctrl
        qd = qc
        # Desired cable-attitude rates from differential flatness of the
        # payload trajectory: the flat cable direction is
        # q_des = -(a_L + g e3)/‖·‖ with payload-acceleration jet
        # [a_L, ȧ_L, ä_L] = [d2sd, d3sd, d4sd]; dqd, d2qd are its 1st/2nd time
        # derivatives, from the same flat-to-state map as
        # udaan.flatness.quadrotor_cspayload. These supply the cable
        # feedforward; q_des coincides with qc in the nominal (feedback-free)
        # case, so qd is left as qc.
        try:
            _, q_flat = cable_direction_jet([d2sd, d3sd, d4sd], mL)
            dqd, d2qd = q_flat[1], q_flat[2]
        except ValueError:
            # Payload near free-fall: cable direction undefined, drop feedforward.
            dqd = np.zeros(3)
            d2qd = np.zeros(3)
        # calculating errors
        err_q = hat(q) @ hat(q) @ qd
        err_dq = dq - np.cross(np.cross(qd, dqd), q)

        Fpd = -self._gain_cable.kp * err_q - self._gain_cable.kd * err_dq
        Fff = (mQ * l) * np.dot(q, np.cross(qd, dqd)) * np.cross(q, dq) + (mQ * l) * np.cross(
            np.cross(qd, d2qd), q
        )
        # Tension along the cable, floored so the cable stays taut and the
        # thrust keeps pointing from the payload up through the airframe.
        tension = max(-float(np.dot(A, q)), self.min_tension)
        Fn = -tension * q
        thrust_force = -Fpd - Fff + Fn
        return thrust_force
