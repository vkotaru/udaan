"""Base class for per-system differential-flatness state recovery.

A subclass declares three dataclass types as class attributes ``Flats``,
``RefState``, ``Inputs`` and implements ``recover``. Physical parameters
(mass, inertia, cable length, …) live on the instance; the
flat-to-state map itself is stateless in time.
"""

from __future__ import annotations

from typing import ClassVar


class Flat2State:
    """Differential-flatness map for a specific robotic system.

    Subclass contract:
        - Declare three dataclasses as class attributes:
              ``Flats``     flat-output derivative bundle at a single time
              ``RefState``  recovered kinematic reference state
              ``Inputs``    recovered feedforward inputs
        - Implement ``recover(self, flats) -> (RefState, Inputs)``.

    Optional: provide a ``from_model(cls, model)`` classmethod that
    reads physical parameters off the corresponding simulation model.

    Intermediate abstract classes (ones that only partially specify the
    triple and are subclassed again before instantiation) can opt out of
    the declaration check with ``_abstract = True``.
    """

    Flats: ClassVar[type]
    RefState: ClassVar[type]
    Inputs: ClassVar[type]

    _abstract: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract", False):
            return
        missing = [name for name in ("Flats", "RefState", "Inputs") if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"{cls.__name__} must declare class attributes: {', '.join(missing)}")

    def recover(self, flats):
        """Map flat-output derivatives to (reference state, feedforward inputs).

        Subclasses must override.
        """
        raise NotImplementedError
