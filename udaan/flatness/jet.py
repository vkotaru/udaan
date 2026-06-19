"""Jet — a time-varying signal packaged with its time derivatives.

A *k-th order jet* at time t₀ of a (possibly vector-valued) signal y(t) is the
tuple

    Jₖ(y, t₀) = (y(t₀), ẏ(t₀), ÿ(t₀), …, y⁽ᵏ⁾(t₀))

In differential-flatness theory (Fliess, Lévine, et al.) the state and input
of a flat system are algebraic functions of such a jet of the flat output.
This module provides a lightweight container for these tuples, used
throughout ``udaan.flatness`` as the atomic data type for flat-output
derivative bundles.

Convention
----------
The underlying storage is a 2-D numpy array of shape ``(order+1, dim)``.
Row ``k`` is the k-th time derivative of the signal::

    data[0]  →  y
    data[1]  →  ẏ
    data[2]  →  ÿ
    ...
    data[order]  →  y⁽ᵒʳᵈᵉʳ⁾

Scalar signals (``dim == 1``) may be constructed from a 1-D array of length
``order+1``; it's promoted to 2-D internally. Indexing a scalar jet returns
a Python ``float``; indexing a vector jet returns a 1-D ``np.ndarray`` of
shape ``(dim,)``.

Strictly, the k-jet in the differential-geometry literature is the truncated
Taylor polynomial ``Σ yᵏ(t₀)/k! · zᵏ``; ``Jet`` stores the raw derivative
tuple (no factorial scaling), which is the conventional representation in
the applied flatness and controls literature.

See also
--------
https://en.wikipedia.org/wiki/Jet_(mathematics)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Jet:
    """Time-derivative tuple ``(y, ẏ, ÿ, …)`` of a signal at a single time.

    Parameters
    ----------
    data:
        Either a 1-D array of shape ``(order+1,)`` — which promotes to
        a scalar jet with ``dim == 1`` — or a 2-D array of shape
        ``(order+1, dim)``. Dtype is coerced to float64.

    Attributes
    ----------
    data:
        2-D numpy array of shape ``(order+1, dim)``.

    Examples
    --------
    Scalar signal (e.g. yaw angle with three derivative levels)::

        >>> psi = Jet(np.array([0.0, 0.1, 0.2]))
        >>> psi.order
        2
        >>> psi.dim
        1
        >>> psi[1]
        0.1

    Vector signal (e.g. position with four derivative levels)::

        >>> x = Jet(np.stack([[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]], axis=0))
        >>> x.order
        3
        >>> x.dim
        3
        >>> x[1]
        array([1., 0., 0.])
    """

    data: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Jet.data must be 1-D (scalar signal) or 2-D "
                f"(vector signal); got shape {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("Jet requires at least one row (the 0-th derivative).")
        self.data = arr

    # ─── Shape / type introspection ───────────────────────────────────

    @property
    def order(self) -> int:
        """Highest derivative index stored; ``data.shape[0] - 1``."""
        return self.data.shape[0] - 1

    @property
    def dim(self) -> int:
        """Signal dimension; ``data.shape[1]`` (``1`` for scalar signals)."""
        return self.data.shape[1]

    @property
    def is_scalar(self) -> bool:
        """``True`` if the signal is scalar (``dim == 1``)."""
        return self.dim == 1

    # ─── Indexed access ───────────────────────────────────────────────

    def __getitem__(self, k: int) -> float | np.ndarray:
        """The k-th derivative.

        Returns ``float`` for scalar signals (``dim == 1``), otherwise
        a 1-D ``np.ndarray`` of shape ``(dim,)``.
        """
        if not 0 <= k <= self.order:
            raise IndexError(f"Jet derivative index {k} out of range [0, {self.order}]")
        row = self.data[k]
        return float(row[0]) if self.is_scalar else row

    def __len__(self) -> int:
        """Number of derivative levels (``order + 1``)."""
        return self.data.shape[0]

    def __iter__(self):
        """Iterate over derivative levels in order (y, ẏ, ÿ, …)."""
        for k in range(self.order + 1):
            yield self[k]

    # ─── Named accessors for common low orders ────────────────────────
    #
    # These are pure conveniences. ``jet.acceleration`` is exactly the
    # same as ``jet[2]`` — use whichever reads best in a given line.

    @property
    def value(self) -> float | np.ndarray:
        return self[0]

    @property
    def velocity(self) -> float | np.ndarray:
        if self.order < 1:
            raise AttributeError("Jet has order 0; no velocity stored")
        return self[1]

    @property
    def acceleration(self) -> float | np.ndarray:
        if self.order < 2:
            raise AttributeError(f"Jet has order {self.order}; no acceleration stored")
        return self[2]

    @property
    def jerk(self) -> float | np.ndarray:
        if self.order < 3:
            raise AttributeError(f"Jet has order {self.order}; no jerk stored")
        return self[3]

    @property
    def snap(self) -> float | np.ndarray:
        if self.order < 4:
            raise AttributeError(f"Jet has order {self.order}; no snap stored")
        return self[4]

    @property
    def crackle(self) -> float | np.ndarray:
        if self.order < 5:
            raise AttributeError(f"Jet has order {self.order}; no crackle stored")
        return self[5]

    @property
    def pop(self) -> float | np.ndarray:
        if self.order < 6:
            raise AttributeError(f"Jet has order {self.order}; no pop stored")
        return self[6]

    # ─── numpy interop ────────────────────────────────────────────────

    def __array__(self, dtype=None) -> np.ndarray:
        """Return the underlying 2-D derivative array."""
        return self.data if dtype is None else self.data.astype(dtype)

    # ─── Constructors / transformations ───────────────────────────────

    @classmethod
    def from_list(cls, derivatives: list) -> Jet:
        """Construct from an explicit list of derivative arrays/scalars.

        Examples
        --------
        >>> Jet.from_list([pos, vel, acc, jerk, snap])
        """
        stacked = np.stack([np.atleast_1d(d) for d in derivatives])
        return cls(stacked)

    @classmethod
    def zeros(cls, order: int, dim: int = 1) -> Jet:
        """A jet of all zeros with the given ``order`` and ``dim``."""
        return cls(np.zeros((order + 1, dim)))

    def truncate(self, new_order: int) -> Jet:
        """Drop derivatives above ``new_order`` and return a new jet."""
        if new_order < 0 or new_order > self.order:
            raise ValueError(f"truncate(new_order={new_order}) out of range [0, {self.order}]")
        return Jet(self.data[: new_order + 1])

    def differentiate(self) -> Jet:
        """Shift derivatives down by one level and return a new jet.

        Given ``J = (y, ẏ, ÿ, …, y⁽ᵒʳᵈᵉʳ⁾)`` returns
        ``J' = (ẏ, ÿ, …, y⁽ᵒʳᵈᵉʳ⁾)`` — i.e. the jet of the time derivative
        of the original signal, with ``order`` reduced by one.
        """
        if self.order < 1:
            raise ValueError("Cannot differentiate an order-0 jet; the derivative is not stored.")
        return Jet(self.data[1:])

    # ─── Debug / display ──────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.is_scalar:
            vals = ", ".join(f"{float(row[0]):.4g}" for row in self.data)
        else:
            vals = ", ".join(np.array2string(row, precision=4, separator=", ") for row in self.data)
        return f"Jet(order={self.order}, dim={self.dim}; [{vals}])"
