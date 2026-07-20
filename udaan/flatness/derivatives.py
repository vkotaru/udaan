"""Arbitrary-order time-derivative algebra for differential-flatness maps.

Every flat-to-state map in this package pushes the time derivatives of the flat
output through the system dynamics.  Doing so by hand (or with a one-off
symbolic dump) does not scale past the first system, so the recurring operations
are collected here as a small **shared derivative engine**: given the
time-derivative tuples of one or more signals, build the derivative tuple of
their product, inner product, norm, and normalized direction via the Leibniz
rule.

A *derivative tuple* is ``[s, ṡ, s̈, …, s^(K)]`` — a signal's value together with
all of its time derivatives up to order ``K`` at one instant (the same object
carried by :class:`~udaan.flatness.jet.Jet`).  These closed-form recursions are
exact at arbitrary order and are reused by every per-system map (the
quadrotor / cable-suspended-payload maps today; the multi-cable systems next).
"""

from __future__ import annotations

from math import comb

import numpy as np

__all__ = ["leibniz_product", "leibniz_dot", "normalize_derivatives"]


def leibniz_product(a: list, b: list) -> list:
    """Derivative tuple of the product ``a(t) · b(t)`` (elementwise / scalar).

    Given derivative tuples ``a = [a, ȧ, …, a^(K)]`` and ``b`` of equal length,
    return ``c`` with ``c[k] = (d/dt)^k (a b) = Σ_j C(k,j) a^(j) b^(k-j)``.

    ``a`` and ``b`` may be scalars or NumPy arrays (broadcast elementwise); the
    result matches their shape.
    """
    if len(a) != len(b):
        raise ValueError("leibniz_product: derivative tuples must have equal length")
    K = len(a) - 1
    return [sum(comb(k, j) * a[j] * b[k - j] for j in range(k + 1)) for k in range(K + 1)]


def leibniz_dot(a: list[np.ndarray], b: list[np.ndarray]) -> list[float]:
    """Derivative tuple of the inner product ``⟨a(t), b(t)⟩``.

    Given derivative tuples of two vector signals of equal length, return the
    scalar derivative tuple ``c`` with
    ``c[k] = (d/dt)^k ⟨a, b⟩ = Σ_j C(k,j) ⟨a^(j), b^(k-j)⟩``.
    """
    if len(a) != len(b):
        raise ValueError("leibniz_dot: derivative tuples must have equal length")
    K = len(a) - 1
    out = []
    for k in range(K + 1):
        s = 0.0
        for j in range(k + 1):
            s += comb(k, j) * float(np.dot(a[j], b[k - j]))
        out.append(s)
    return out


def normalize_derivatives(
    v: list[np.ndarray],
) -> tuple[list[float], list[np.ndarray]]:
    """Derivative tuples of the norm and unit direction of a vector signal.

    Given the derivative tuple ``v = [v, v̇, v̈, …, v^(K)]`` of a non-vanishing
    vector signal, return ``(n, u)`` where ``n[k] = (d/dt)^k ‖v‖`` (scalar) and
    ``u[k] = (d/dt)^k (v / ‖v‖)`` (vector) for ``k = 0, 1, …, K``.

    The recursions follow from ``‖v‖² = ⟨v, v⟩`` and ``v = ‖v‖ · u``:

        (‖v‖²)^(k) = Σ_{j=0}^{k} C(k,j) ⟨v^(j), v^(k-j)⟩        (Leibniz)
        n^(k)      = ((‖v‖²)^(k) − Σ_{j=1}^{k-1} C(k,j) n^(j) n^(k-j)) / (2 n)
        u^(k)      = (v^(k) − Σ_{j=1}^{k} C(k,j) n^(j) u^(k-j)) / n

    Raises
    ------
    ValueError
        If ``v`` vanishes (``‖v‖ ≈ 0``): the direction is undefined — this is the
        free-fall / slack-cable singularity of the flatness maps that call it.
    """
    K = len(v) - 1
    n_val = float(np.linalg.norm(v[0]))
    if n_val < 1e-9:
        raise ValueError(
            "normalize_derivatives: input vector has zero norm — singularity in the flatness map."
        )

    # n²[k] := (d/dt)^k ‖v‖² via the Leibniz inner-product rule.
    n_sq = leibniz_dot(v, v)

    # n[k] by differentiating ‖v‖² = ‖v‖·‖v‖ and solving for the top term:
    #   (‖v‖²)^(k) = Σ_j C(k,j) n^(j) n^(k-j)  ⇒  2 n n^(k) = (‖v‖²)^(k) − (inner terms)
    n = [n_val]
    for k in range(1, K + 1):
        s = n_sq[k]
        for j in range(1, k):
            s -= comb(k, j) * n[j] * n[k - j]
        n.append(s / (2.0 * n_val))

    # u[k] by differentiating v = n·u and solving for the top term:
    #   v^(k) = Σ_j C(k,j) n^(j) u^(k-j)  ⇒  n u^(k) = v^(k) − (inner terms)
    u = [v[0] / n_val]
    for k in range(1, K + 1):
        s = np.array(v[k], dtype=float)
        for j in range(1, k + 1):
            s -= comb(k, j) * n[j] * u[k - j]
        u.append(s / n_val)

    return n, u
