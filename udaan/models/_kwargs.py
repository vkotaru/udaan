"""Validation for the ``**kwargs`` initial conditions taken by ``reset()``.

Every model's ``reset()`` — and therefore ``simulate()``, which forwards to it —
takes its initial condition as keyword arguments. Historically an unrecognised
keyword was simply not matched by any of the ``if key in kwargs`` branches, so
it was discarded and the run went ahead from the default state. That is the
worst failure mode a renamed or retired keyword can have: a caller gets a clean
exit code and a silently different simulation.

It bit for real across the v1.3.0 → v2.0.0 break. v1.3.0's own CLI called
``simulate(tf=time, xL=x0)`` on the retired ``MultiQuadrotorCSPointmass``; the
M3 model that replaces it names the same quantity ``payload_position``, so a
1.x user who only re-points the import gets ``simulate(tf=0.5, xL=[-1, 2, 0.5])``
running from the default formation with no diagnostic at all.

So: reject. :func:`check_reset_kwargs` is the one place that decides how, and
three MuJoCo backends already phrased their bespoke version of it as "cannot
honour", which is the wording kept here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

__all__ = ["check_reset_kwargs"]


def check_reset_kwargs(
    obj: Any,
    kwargs: Iterable[str],
    supported: Collection[str],
    detail: str | None = None,
) -> None:
    """Raise if ``kwargs`` holds a keyword ``obj.reset()`` will not act on.

    Args:
        obj: The model instance, used only for the type name in the message.
        kwargs: The keyword names passed to ``reset()``.
        supported: The keywords this ``reset()`` actually reads.
        detail: Optional sentence explaining *why* the rest are unsupported,
            for backends that can honour less than their analytical base.

    Raises:
        ValueError: If any keyword is outside ``supported``.
    """
    unsupported = sorted(k for k in kwargs if k not in supported)
    if not unsupported:
        return

    reason = f": {detail}" if detail else "."
    raise ValueError(
        f"{type(obj).__name__}.reset() cannot honour {unsupported}{reason} "
        f"Supported: {', '.join(sorted(supported))}."
    )
