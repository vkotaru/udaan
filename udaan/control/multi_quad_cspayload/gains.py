"""Time-scale gain design for the multi-quadrotor payload cascade (M3).

The same design as the single-vehicle cascade
(:mod:`udaan.control.quadrotor_cspayload.gains`): attitude, cable and payload
loops, each a second-order system placed at its own natural frequency with
damping ``ζ``, the frequencies separated by a ratio ``r``::

    ω_a     = CABLE_CASCADE_ATTITUDE_BANDWIDTH
    ω_c,i   = min(c √(g/l_i), ω_a / r)          (cable i)
    ω_p     = min_i ω_c,i / r_p                 (payload x, y)
    ω_z     = ω_a / r²                          (payload z)

with ``r_p = MULTI_PAYLOAD_LOOP_RATIO``.

M3's payload and cable loops are written in acceleration units (the payload PD
commands ``a_L^d``; the cable PD is multiplied by ``m_i l_i`` inside the
controller), so their gains are ``ω²`` and ``2ζω`` with no mass. The payload
loop is placed a ratio ``r_p`` below the slowest cable, the longest one. The attitude gains are
moments: ``k_R = J_ii ω_a²``, ``k_Ω = 2ζ J_ii ω_a`` on roll and pitch, and the
quadrotor's default yaw gains, per airframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.defaults import (
    CABLE_CASCADE_DAMPING,
    DEFAULT_ATT_KD,
    DEFAULT_ATT_KP,
    MULTI_PAYLOAD_LOOP_RATIO,
)
from ...core.validation import check_positive
from ..quadrotor_cspayload.gains import LoopFrequencies, loop_frequencies

__all__ = ["MultiCableCascadeGains", "multi_cable_cascade_gains"]


@dataclass(frozen=True)
class MultiCableCascadeGains:
    """Gains of the M3 cascade."""

    #: ``(3,)`` payload position gain, in 1/s².
    position_kp: np.ndarray
    #: ``(3,)`` payload velocity gain, in 1/s.
    position_kd: np.ndarray
    #: ``(nQ, 3)`` cable direction gains, in 1/s².
    cable_kp: np.ndarray
    #: ``(nQ, 3)`` cable rate gains, in 1/s.
    cable_kd: np.ndarray
    #: ``(nQ, 3)`` attitude gains ``k_R``, in N·m.
    attitude_kp: np.ndarray
    #: ``(nQ, 3)`` body-rate gains ``k_Ω``, in N·m·s.
    attitude_kd: np.ndarray
    #: One :class:`~udaan.control.quadrotor_cspayload.gains.LoopFrequencies` per
    #: cable; the payload entries of each are the shared ``ω_p``, ``ω_z``.
    frequencies: tuple[LoopFrequencies, ...]


def multi_cable_cascade_gains(
    cable_lengths,
    inertias,
    *,
    damping: float = CABLE_CASCADE_DAMPING,
    payload_ratio: float = MULTI_PAYLOAD_LOOP_RATIO,
    **frequency_kwargs,
) -> MultiCableCascadeGains:
    """The M3 cascade's gains for cables ``cable_lengths`` and airframes ``inertias``.

    Args:
        cable_lengths: ``(nQ,)`` cable lengths, in m.
        inertias: ``(nQ, 3, 3)`` airframe inertias, in kg·m²; the roll and pitch
            gains use each diagonal.
        damping: ``ζ`` of every loop.
        payload_ratio: ``r_p``, the ratio between the slowest cable loop and the
            payload's lateral loop, ``ω_p = min_i ω_c,i / r_p``.
        **frequency_kwargs: passed to
            :func:`~udaan.control.quadrotor_cspayload.gains.loop_frequencies`.
    """
    check_positive(damping, "damping")
    check_positive(payload_ratio, "payload_ratio")
    lengths = np.atleast_1d(np.asarray(cable_lengths, dtype=float))
    J = np.asarray(inertias, dtype=float)
    if J.shape != (lengths.size, 3, 3):
        raise ValueError(f"inertias must be ({lengths.size}, 3, 3); got {J.shape}")
    per_cable = [loop_frequencies(float(l_i), **frequency_kwargs) for l_i in lengths]
    slowest = min(per_cable, key=lambda w: w.cable)
    w_p = slowest.cable / payload_ratio
    shared = [
        LoopFrequencies(attitude=w.attitude, cable=w.cable, payload=w_p, vertical=w.vertical)
        for w in per_cable
    ]
    w_xyz = np.array([w_p, w_p, slowest.vertical])
    w_c = np.array([w.cable for w in per_cable])
    w_a = np.array([w.attitude for w in per_cable])
    Jd = np.array([np.diag(J_i) for J_i in J])
    att_kp = np.column_stack(
        [Jd[:, 0] * w_a**2, Jd[:, 1] * w_a**2, np.full(lengths.size, DEFAULT_ATT_KP[2])]
    )
    att_kd = np.column_stack(
        [
            2.0 * damping * Jd[:, 0] * w_a,
            2.0 * damping * Jd[:, 1] * w_a,
            np.full(lengths.size, DEFAULT_ATT_KD[2]),
        ]
    )
    return MultiCableCascadeGains(
        position_kp=w_xyz**2,
        position_kd=2.0 * damping * w_xyz,
        cable_kp=np.repeat((w_c**2)[:, None], 3, axis=1),
        cable_kd=np.repeat((2.0 * damping * w_c)[:, None], 3, axis=1),
        attitude_kp=att_kp,
        attitude_kd=att_kd,
        frequencies=tuple(shared),
    )
