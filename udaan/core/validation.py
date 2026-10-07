"""Shared validation for physical model parameters.

Every model in :mod:`udaan.models` builds a mass matrix out of masses, lengths
and inertias, and every one of them is only invertible while those numbers are
strictly positive and finite. A non-positive mass does not raise where it is
*set*; it raises — or worse, quietly does not — deep inside the integrator, as
a ``LinAlgError`` from ``np.linalg.solve`` or a ``nan`` behind a
``RuntimeWarning``. The guard belongs at assignment, where the caller can still
see which parameter was wrong.

This module is the single implementation. M4, M5 and M6 each grew a private
copy of it as they were written; M2 and M3 had none at all.
"""

from __future__ import annotations

import logging

import numpy as np

# ``core`` is the bottom of the layering and imports nothing from ``utils``, so
# the logger is taken straight from stdlib under the same ``udaan.*`` namespace
# ``udaan.utils.logging.get_logger`` would have produced.
_logger = logging.getLogger(__name__)

#: Below this, a mass is almost certainly a typo or a unit error rather than a
#: deliberately light body. The dynamics stay well posed, so this warns instead
#: of raising.
MIN_PHYSICAL_MASS = 1e-9

__all__ = ["MIN_PHYSICAL_MASS", "check_positive"]


def check_positive(values, name: str, *, warn_below: float | None = None) -> None:
    """Reject non-finite or non-positive physical parameters.

    Args:
        values: scalar or array of parameters to check.
        name: parameter name, used verbatim in the error message.
        warn_below: if given, log a warning (but do not raise) when any entry
            falls below this threshold. Used for masses that are positive but
            implausibly small.

    Raises:
        ValueError: if any entry is non-finite or not strictly positive.
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite; got {arr.tolist()}")
    if not np.all(arr > 0.0):
        raise ValueError(f"{name} must be strictly positive; got {arr.tolist()}")
    if warn_below is not None and np.any(arr < warn_below):
        _logger.warning(
            "%s contains a value below %g (%s); the dynamics stay well posed "
            "but this is almost certainly a typo or a unit error",
            name,
            warn_below,
            arr.tolist(),
        )
