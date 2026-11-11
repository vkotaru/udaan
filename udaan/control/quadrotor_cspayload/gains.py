"""Time-scale gain design for the cable-suspended payload cascade.

The cascade nests three loops: the attitude loop turns the airframe, the cable
loop swings the cable through the airframe, and the payload loop moves the
payload through the cable. Each loop is a second-order system about hover,
with the inner one treated as instantaneous, so each is placed at its own
natural frequency ``ω`` and damping ``ζ``:

==========  =============================  ==========================================
loop        error dynamics                 gains
==========  =============================  ==========================================
payload     ``M ë + k_v ė + k_x e = 0``    ``k_x = M ω_p²``, ``k_v = 2ζ M ω_p``
cable       ``m_Q ℓ ë_q + k_ω ė_q + k_q``  ``k_q = m_Q ℓ ω_c²``, ``k_ω = 2ζ m_Q ℓ ω_c``
attitude    ``J ë_R + k_Ω ė_R + k_R e_R``  ``k_R = J ω_a²``, ``k_Ω = 2ζ J ω_a``
==========  =============================  ==========================================

with ``M = m_Q + m_L``. The frequencies are separated by a ratio ``r`` so that
each inner loop settles on the time scale its outer loop acts on (time-scale
separation), and the cable loop is tied to the pendulum frequency ``√(g/ℓ)``,
so the gains follow the cable length::

    ω_a = CABLE_CASCADE_ATTITUDE_BANDWIDTH
    ω_c = min(c √(g/ℓ), ω_a / r)
    ω_p = ω_c / r          (payload x, y)
    ω_z = ω_a / r²         (payload z)

``r``, ``c`` and ``ζ`` are :data:`~udaan.core.defaults.CABLE_CASCADE_LOOP_RATIO`,
:data:`~udaan.core.defaults.CABLE_CASCADE_PENDULUM_FACTOR` and
:data:`~udaan.core.defaults.CABLE_CASCADE_DAMPING`. The vertical axis carries no
pendulum, so its frequency is fixed by the attitude loop alone. Yaw does not
couple to the cable and keeps the quadrotor's own attitude gains
(:data:`~udaan.core.defaults.DEFAULT_ATT_KP`, ``DEFAULT_ATT_KD``, third entry).

Every gain is a force (payload, cable) or a moment (attitude): the payload
controller's ``A`` and cable command are forces in N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...core.defaults import (
    CABLE_CASCADE_ATTITUDE_BANDWIDTH,
    CABLE_CASCADE_DAMPING,
    CABLE_CASCADE_LOOP_RATIO,
    CABLE_CASCADE_PENDULUM_FACTOR,
    DEFAULT_ATT_KD,
    DEFAULT_ATT_KP,
    GRAVITY,
)
from ...core.validation import check_positive


@dataclass(frozen=True)
class LoopFrequencies:
    """Natural frequencies of the cable cascade's loops, in rad/s."""

    #: ``ω_a``, roll and pitch.
    attitude: float
    #: ``ω_c``.
    cable: float
    #: ``ω_p``, payload x and y.
    payload: float
    #: ``ω_z``, payload z.
    vertical: float


def loop_frequencies(
    cable_length: float,
    *,
    attitude: float = CABLE_CASCADE_ATTITUDE_BANDWIDTH,
    ratio: float = CABLE_CASCADE_LOOP_RATIO,
    pendulum_factor: float = CABLE_CASCADE_PENDULUM_FACTOR,
    gravity: float = GRAVITY,
) -> LoopFrequencies:
    """The loop frequencies of the time-scale design for a cable of length ``cable_length``.

    ``ω_c = min(c √(g/ℓ), ω_a / r)``, ``ω_p = ω_c / r``, ``ω_z = ω_a / r²``.
    """
    check_positive(cable_length, "cable_length")
    check_positive(attitude, "attitude")
    check_positive(ratio, "ratio")
    check_positive(pendulum_factor, "pendulum_factor")
    w_c = min(pendulum_factor * math.sqrt(gravity / cable_length), attitude / ratio)
    return LoopFrequencies(
        attitude=float(attitude),
        cable=float(w_c),
        payload=float(w_c / ratio),
        vertical=float(attitude / ratio**2),
    )


@dataclass(frozen=True)
class CableCascadeGains:
    """Gains of the cable cascade, each a ``(3,)`` array."""

    #: ``k_x``, payload position, in N/m.
    position_kp: np.ndarray
    #: ``k_v``, payload velocity, in N·s/m.
    position_kd: np.ndarray
    #: ``k_q``, cable direction, in N.
    cable_kp: np.ndarray
    #: ``k_ω``, cable rate, in N·s.
    cable_kd: np.ndarray
    #: ``k_R``, in N·m.
    attitude_kp: np.ndarray
    #: ``k_Ω``, in N·m·s.
    attitude_kd: np.ndarray
    #: The :class:`LoopFrequencies` they place.
    frequencies: LoopFrequencies


def cable_cascade_gains(
    mass_q: float,
    mass_l: float,
    cable_length: float,
    inertia,
    *,
    damping: float = CABLE_CASCADE_DAMPING,
    **frequency_kwargs,
) -> CableCascadeGains:
    """The cascade's gains for an airframe ``mass_q`` carrying ``mass_l`` on ``cable_length``.

    Args:
        mass_q: airframe mass ``m_Q``, in kg.
        mass_l: payload mass ``m_L``, in kg.
        cable_length: ``ℓ``, in m.
        inertia: airframe inertia ``J``, ``(3,)`` or ``(3, 3)`` in kg·m²; the
            roll and pitch gains use its diagonal.
        damping: ``ζ`` of every loop.
        **frequency_kwargs: passed to :func:`loop_frequencies`.

    Returns:
        :class:`CableCascadeGains`: ``k_x = M ω²``, ``k_v = 2ζ M ω`` (``ω = ω_p``
        on x, y and ``ω_z`` on z, ``M = m_Q + m_L``); ``k_q = m_Q ℓ ω_c²``,
        ``k_ω = 2ζ m_Q ℓ ω_c``; ``k_R = J ω_a²``, ``k_Ω = 2ζ J ω_a`` on roll and
        pitch, and the quadrotor's default attitude gains on yaw.
    """
    check_positive(mass_q, "mass_q")
    check_positive(mass_l, "mass_l")
    check_positive(damping, "damping")
    J = np.asarray(inertia, dtype=float)
    Jd = np.diag(J) if J.ndim == 2 else J
    if Jd.shape != (3,):
        raise ValueError(f"inertia must be (3,) or (3, 3); got {J.shape}")
    w = loop_frequencies(cable_length, **frequency_kwargs)
    M = mass_q + mass_l
    w_xyz = np.array([w.payload, w.payload, w.vertical])
    mql = mass_q * cable_length
    return CableCascadeGains(
        position_kp=M * w_xyz**2,
        position_kd=2.0 * damping * M * w_xyz,
        cable_kp=np.full(3, mql * w.cable**2),
        cable_kd=np.full(3, 2.0 * damping * mql * w.cable),
        attitude_kp=np.array([Jd[0] * w.attitude**2, Jd[1] * w.attitude**2, DEFAULT_ATT_KP[2]]),
        attitude_kd=np.array(
            [
                2.0 * damping * Jd[0] * w.attitude,
                2.0 * damping * Jd[1] * w.attitude,
                DEFAULT_ATT_KD[2],
            ]
        ),
        frequencies=w,
    )
