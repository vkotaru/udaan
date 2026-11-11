"""Feasibility of a flatness reference: does it keep every airframe upright?

A flat map recovers each airframe's attitude ``R`` from its thrust vector,
``b₃ = B/‖B‖``. The map returns an attitude for any non-zero ``B``, including one
that points down: a reference that needs ``b₃·e₃ ≤ 0`` asks the airframe to fly
level or inverted, which no upright controller can follow. On a cable-suspended
payload this happens although the payload's own net force points up: the
airframe whirls around the payload, and with ``q`` the unit cable direction
(airframe to payload) the thrust it needs along the cable is::

    (m_Q + m_L)(ẍ_L + g e₃)·(−q) − m_Q ℓ ‖q̇‖²,

which the centripetal term ``m_Q ℓ ‖q̇‖²`` can drive negative at the top of a
fast loop.

A sample is *upright* when every airframe's ``b₃·e₃`` exceeds
:data:`~udaan.core.defaults.MIN_UPRIGHT_THRUST_Z`. A sample the map refuses
(free fall, a slack cable) is not upright either.

- :func:`is_upright` judges one sample's attitudes.
- :func:`scan_reference` runs a sampler along ``[t0, tf]`` and returns a
  :class:`ReferenceFeasibility` with the infeasible intervals; each system's
  flat map provides the sampler (see
  :meth:`~udaan.flatness.QuadrotorCsPayload.check_reference`).
- :func:`infeasible_intervals` and :func:`format_intervals` turn per-sample
  flags into ``[(t_start, t_end), …]`` and a message, for a model's
  ``simulate()`` that collects the flags as it flies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.defaults import MIN_UPRIGHT_THRUST_Z

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import ArrayLike


def is_upright(orientations) -> bool:
    """Whether every rotation in ``orientations`` has ``b₃·e₃ > MIN_UPRIGHT_THRUST_Z``.

    Args:
        orientations: one ``(3, 3)`` rotation (an :class:`~udaan.manif.SO3` or an
            array), or an iterable of them (one per airframe), or ``None`` for
            a sample the flat map refused (not upright).
    """
    if orientations is None:
        return False
    R = np.asarray(orientations, dtype=float)
    if R.ndim == 2:
        R = R[None]
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"expected (3, 3) rotations; got shape {R.shape}")
    return bool(np.all(R[:, 2, 2] > MIN_UPRIGHT_THRUST_Z))


def infeasible_intervals(
    times: ArrayLike, upright: ArrayLike, dt: float | None = None
) -> list[tuple[float, float]]:
    """The maximal runs of samples that are not upright, as ``(t_start, t_end)``.

    ``t_end`` is the last infeasible sample's time plus ``dt`` (the sampling
    period, inferred from ``times`` when not given), so a single infeasible
    sample has a non-empty interval.
    """
    t = np.asarray(times, dtype=float)
    ok = np.asarray(upright, dtype=bool)
    if t.shape != ok.shape:
        raise ValueError(f"times and upright differ in shape: {t.shape} vs {ok.shape}")
    if t.size == 0:
        return []
    if dt is None:
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    out: list[tuple[float, float]] = []
    start = None
    for k in range(t.size):
        if not ok[k] and start is None:
            start = t[k]
        if ok[k] and start is not None:
            out.append((float(start), float(t[k - 1] + dt)))
            start = None
    if start is not None:
        out.append((float(start), float(t[-1] + dt)))
    return out


def format_intervals(intervals: Iterable[tuple[float, float]]) -> str:
    """``"[3.270, 3.440] s, [5.215, 5.390] s"``."""
    return ", ".join(f"[{a:.3f}, {b:.3f}] s" for a, b in intervals)


@dataclass(frozen=True)
class ReferenceFeasibility:
    """Per-sample uprightness of a reference along ``times``."""

    #: Sample times, in s.
    times: np.ndarray
    #: ``True`` where every airframe's ``b₃·e₃`` exceeds
    #: :data:`~udaan.core.defaults.MIN_UPRIGHT_THRUST_Z`.
    upright: np.ndarray
    #: The smallest ``b₃·e₃`` over the airframes at each sample (``nan`` where
    #: the map refused the sample).
    min_thrust_z: np.ndarray
    #: The sampling period, in s.
    dt: float

    @property
    def feasible(self) -> bool:
        """``True`` when every sample is upright."""
        return bool(np.all(self.upright))

    @property
    def infeasible_samples(self) -> int:
        """How many samples are not upright."""
        return int(np.count_nonzero(~self.upright))

    @property
    def intervals(self) -> list[tuple[float, float]]:
        """The infeasible intervals, see :func:`infeasible_intervals`."""
        return infeasible_intervals(self.times, self.upright, self.dt)

    def __str__(self) -> str:
        if self.feasible:
            return f"feasible: every airframe upright on {self.times.size} samples"
        return (
            f"infeasible on {self.infeasible_samples} of {self.times.size} samples: the reference "
            f"needs an airframe that is not upright on {format_intervals(self.intervals)}"
        )


def scan_reference(
    orientations: Callable[[float], object],
    tf: float,
    *,
    t0: float = 0.0,
    dt: float = 0.005,
) -> ReferenceFeasibility:
    """Run ``orientations(t)`` on ``t = t0, t0 + dt, …`` below ``tf`` and judge each sample.

    Args:
        orientations: ``t -> R`` or ``t -> [R_1, …, R_n]``: the airframe
            attitudes a flat map recovers at ``t``, or ``None`` where it
            refuses the sample.
        tf: end time, in s (exclusive).
        t0: start time, in s.
        dt: sampling period, in s (a model's position-loop period samples
            exactly what its ``simulate()`` feeds forward).
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    n = max(int(np.ceil((tf - t0) / dt - 1e-9)), 0)
    times = t0 + dt * np.arange(n)
    upright = np.zeros(n, dtype=bool)
    min_z = np.full(n, np.nan)
    for k, t in enumerate(times):
        R = orientations(float(t))
        upright[k] = is_upright(R)
        if R is not None:
            min_z[k] = float(np.min(np.asarray(R, dtype=float).reshape(-1, 3, 3)[:, 2, 2]))
    return ReferenceFeasibility(times=times, upright=upright, min_thrust_z=min_z, dt=float(dt))
