"""Validation of the loop rates a model's ``step()`` is built from.

A model's ``step()`` advances the position-loop period ``dt``. Inside it the
physics takes ``N = physics_rate_hz · dt`` substeps of ``dt / N``, and the
attitude loop closes ``k = attitude_rate_hz · dt`` times, each update held over
``N / k`` substeps. Both counts must be whole, and ``k`` must divide ``N``, so
that nothing is rounded and the clock never drifts; these functions compute the
two counts or raise ``ValueError`` listing the valid rates.
"""

from __future__ import annotations

import numpy as np

from ..core.defaults import PHYSICS_RATE_RANGE_HZ
from ..core.validation import check_positive

__all__ = ["attitude_updates_per_step", "physics_steps_per_step"]


def physics_steps_per_step(dt: float, rate: float) -> int:
    """``N = rate · dt`` for a valid ``physics_rate_hz``.

    Raises:
        ValueError: unless the rate lies in
            :data:`~udaan.core.defaults.PHYSICS_RATE_RANGE_HZ` and ``rate · dt``
            is a whole number — so that ``N`` substeps of ``dt/N`` are exactly
            ``dt``, with nothing rounded.
    """
    lo, hi = PHYSICS_RATE_RANGE_HZ
    check_positive(dt, "dt")
    check_positive(rate, "physics_rate_hz")
    n = rate * dt
    if lo <= rate <= hi and abs(n - round(n)) <= 1e-9 * max(1.0, n):
        return int(round(n))
    whole = range(int(np.ceil(lo * dt - 1e-9)), int(np.floor(hi * dt + 1e-9)) + 1)
    valid = ", ".join(f"{k / dt:g}" for k in whole if k >= 1) or "none"
    raise ValueError(
        f"physics_rate_hz = {rate:g} Hz must lie in [{lo:g}, {hi:g}] Hz and be a whole "
        f"number of substeps per dt = {dt:g} s step; valid rates at this dt are "
        f"{valid} Hz. A rate r needs a dt that is a multiple of 1/r (500 Hz: of 2 ms)."
    )


def attitude_updates_per_step(
    dt: float, rate: float | None, substeps: int, *, is_default: bool = False
) -> int:
    """Attitude updates per outer step for ``rate`` over ``substeps`` physics substeps.

    ``None`` is once per step.

    Raises:
        ValueError: unless the rate is ``k/dt`` for a whole ``k`` dividing
            ``substeps`` — so that every update is held for the same whole
            number of substeps and the clock never drifts. ``is_default`` adds
            that the offending rate is the default and has to be passed
            explicitly with this ``physics_rate_hz`` or ``dt``.
    """
    if rate is None:
        return 1
    check_positive(rate, "attitude_rate_hz")
    k = rate * dt
    divisors = [d for d in range(1, substeps + 1) if substeps % d == 0]
    if abs(k - round(k)) > 1e-9 * max(1.0, k) or round(k) not in divisors:
        valid = ", ".join(f"{d / dt:g}" for d in divisors)
        raise ValueError(
            f"attitude_rate_hz = {rate:g} Hz is not a whole number of updates per "
            f"{dt:g} s step dividing its {substeps} substeps; valid rates are "
            f"{valid} Hz (or None for once per step)."
            + (
                f" {rate:g} Hz is the default (DEFAULT_ATTITUDE_RATE_HZ): with this "
                "physics_rate_hz or dt, pass attitude_rate_hz explicitly."
                if is_default
                else ""
            )
        )
    return int(round(k))
