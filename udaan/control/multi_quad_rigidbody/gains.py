"""Time-scale gain design for the rigid-body payload cascade (M4).

The cascade nests the same three loops as the single cable-suspended payload
(:mod:`udaan.control.quadrotor_cspayload.gains`): each airframe's attitude
loop, each cable's direction loop and the payload's position loop, placed at
their own natural frequencies with damping ``ζ`` on every loop::

    ω_a      = CABLE_CASCADE_ATTITUDE_BANDWIDTH
    ω_c,i    = min(c √(g/ℓ_i), ω_a / r)     cable i
    ω_p      = min_i ω_c,i / r               payload x, y
    ω_z      = ω_a / r²                      payload z
    ω_L      = ρ ω_p                         payload attitude

The payload loop sits a ratio ``r`` below the slowest cable, so every cable is
at least ``r`` times faster than the payload it carries.

M4's payload and cable loops are in **acceleration** units: the payload PD
commands an acceleration and each cable PD carries its own ``m_i ℓ_i``, so the
gains are ``k_p = ω²``, ``k_d = 2ζω`` and no mass enters them. The airframe
attitude gains are moments: ``k_R = J_ii ω_a²``, ``k_Ω = 2ζ J_ii ω_a`` on roll
and pitch from each airframe's inertia, and the quadrotor's default gains on
yaw (:data:`~udaan.core.defaults.DEFAULT_ATT_KP`, ``DEFAULT_ATT_KD``, third
entry), which does not couple to the cables.

The payload's attitude loop, which M4 has and the point-mass systems do not,
sits at ``ω_L = ρ ω_p`` with ``ρ =``
:data:`~udaan.core.defaults.RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO`. Its output
is a moment, so its gains are designed on the payload's own inertia ``J_L``:
``k_R = J_L,ii ω_L²``, ``k_Ω = 2ζ J_L,ii ω_L`` per body axis, from the
diagonal of ``J_L``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.defaults import (
    CABLE_CASCADE_DAMPING,
    DEFAULT_ATT_KD,
    DEFAULT_ATT_KP,
    RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO,
)
from ...core.validation import check_positive
from ..quadrotor_cspayload.gains import LoopFrequencies, loop_frequencies

__all__ = ["RigidPayloadCascadeGains", "rigid_payload_cascade_gains"]


@dataclass(frozen=True)
class RigidPayloadCascadeGains:
    """Gains of the rigid-body payload cascade."""

    #: ``(3,)`` payload position gain, in 1/s².
    position_kp: np.ndarray
    #: ``(3,)`` payload velocity gain, in 1/s.
    position_kd: np.ndarray
    #: ``(nQ, 3)`` cable direction gain per cable, in 1/s².
    cable_kp: np.ndarray
    #: ``(nQ, 3)`` cable rate gain per cable, in 1/s.
    cable_kd: np.ndarray
    #: ``(nQ, 3)`` airframe ``k_R`` per airframe, in N·m, or ``None`` when no
    #: inertias were given.
    attitude_kp: np.ndarray | None
    #: ``(nQ, 3)`` airframe ``k_Ω`` per airframe, in N·m·s, or ``None``.
    attitude_kd: np.ndarray | None
    #: ``(3,)`` payload attitude ``k_R``, in N·m, or ``None`` when no payload
    #: inertia was given.
    payload_attitude_kp: np.ndarray | None
    #: ``(3,)`` payload attitude ``k_Ω``, in N·m·s, or ``None``.
    payload_attitude_kd: np.ndarray | None
    #: Payload attitude-loop natural frequency ``ω_L``, in rad/s.
    payload_attitude_frequency: float
    #: One :class:`~udaan.control.quadrotor_cspayload.gains.LoopFrequencies`
    #: per cable, each carrying the shared payload frequency.
    frequencies: tuple[LoopFrequencies, ...]


def rigid_payload_cascade_gains(
    cable_lengths,
    inertias=None,
    payload_inertia=None,
    *,
    damping: float = CABLE_CASCADE_DAMPING,
    payload_attitude_ratio: float = RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO,
    **frequency_kwargs,
) -> RigidPayloadCascadeGains:
    """The M4 cascade's gains for cables ``cable_lengths``, airframes ``inertias``
    and a payload of inertia ``payload_inertia``.

    Args:
        cable_lengths: ``(nQ,)`` cable lengths, in m.
        inertias: ``(nQ, 3, 3)`` airframe inertias, in kg·m²; the roll and pitch
            gains use each diagonal. ``None`` leaves the attitude gains out.
        payload_inertia: ``J_L``, a ``(3,)`` diagonal or ``(3, 3)`` tensor, in
            kg·m²; the payload attitude gains use its diagonal. ``None`` leaves
            them out.
        damping: ``ζ`` of every loop.
        payload_attitude_ratio: ``ρ``, the payload attitude-loop frequency as a
            multiple of the payload's lateral frequency ``ω_p``.
        **frequency_kwargs: passed to
            :func:`~udaan.control.quadrotor_cspayload.gains.loop_frequencies`.
    """
    check_positive(damping, "damping")
    check_positive(payload_attitude_ratio, "payload_attitude_ratio")
    lengths = np.atleast_1d(np.asarray(cable_lengths, dtype=float))
    per_cable = [loop_frequencies(float(l_i), **frequency_kwargs) for l_i in lengths]
    slowest = min(per_cable, key=lambda w: w.cable)
    shared = tuple(
        LoopFrequencies(
            attitude=w.attitude, cable=w.cable, payload=slowest.payload, vertical=w.vertical
        )
        for w in per_cable
    )
    w_xyz = np.array([slowest.payload, slowest.payload, slowest.vertical])
    w_c = np.array([w.cable for w in per_cable])
    attitude_kp = attitude_kd = None
    if inertias is not None:
        J = np.asarray(inertias, dtype=float)
        if J.shape != (lengths.size, 3, 3):
            raise ValueError(f"inertias must be ({lengths.size}, 3, 3); got {J.shape}")
        w_a = np.array([w.attitude for w in per_cable])
        Jd = np.array([np.diag(J_i) for J_i in J])
        n = lengths.size
        attitude_kp = np.column_stack(
            [Jd[:, 0] * w_a**2, Jd[:, 1] * w_a**2, np.full(n, DEFAULT_ATT_KP[2])]
        )
        attitude_kd = np.column_stack(
            [
                2.0 * damping * Jd[:, 0] * w_a,
                2.0 * damping * Jd[:, 1] * w_a,
                np.full(n, DEFAULT_ATT_KD[2]),
            ]
        )
    w_L = payload_attitude_ratio * slowest.payload
    payload_kp = payload_kd = None
    if payload_inertia is not None:
        J_L = np.asarray(payload_inertia, dtype=float)
        if J_L.shape == (3, 3):
            J_L = np.diag(J_L)
        if J_L.shape != (3,):
            raise ValueError(f"payload_inertia must be (3,) or (3, 3); got {J_L.shape}")
        payload_kp = J_L * w_L**2
        payload_kd = 2.0 * damping * J_L * w_L
    return RigidPayloadCascadeGains(
        position_kp=w_xyz**2,
        position_kd=2.0 * damping * w_xyz,
        cable_kp=np.repeat((w_c**2)[:, None], 3, axis=1),
        cable_kd=np.repeat((2.0 * damping * w_c)[:, None], 3, axis=1),
        attitude_kp=attitude_kp,
        attitude_kd=attitude_kd,
        payload_attitude_kp=payload_kp,
        payload_attitude_kd=payload_kd,
        payload_attitude_frequency=float(w_L),
        frequencies=shared,
    )
