"""Validation for the ``**kwargs`` initial conditions taken by ``reset()``.

Every model's ``reset()`` — and therefore ``simulate()``, which forwards to it —
takes its initial condition as keyword arguments. An unrecognised keyword that
matched none of the ``if key in kwargs`` branches would be discarded and the run
would start from the default state: a misspelled or renamed keyword gives a
clean exit code and a silently different simulation. For example, the 1.x
keyword ``xL`` names what the M3 model calls ``payload_position``.

Unrecognised keywords are therefore rejected. :func:`check_reset_kwargs` is the
one place that decides how, using the "cannot honour" wording the MuJoCo
backends use for keywords they recognise but cannot apply.
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
