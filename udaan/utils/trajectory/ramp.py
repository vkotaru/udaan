r"""The smooth step every circle-type reference starts from rest with.

``S_m(u) = I_u(m + 1, m + 1)``, the regularised incomplete beta function: the
polynomial of degree ``2m + 1`` with ``S_m(0) = 0``, ``S_m(1) = 1`` and its
first ``m`` derivatives zero at both ends, so ``S_m`` meets the constants on
either side continuously through order ``m``. A rate ``ω S_m(t/T)`` therefore
gives a phase, and a path, continuous through order ``m + 1`` where the ramp
meets rest and the cruise. ``S_m(u) + S_m(1 - u) = 1``, so the ramp covers
half of its full-rate phase.

A reference read by a flat map of jet order ``k`` needs ``m = k - 1``; the
caller that binds the trajectory to a model chooses it. ``m = 5`` is the
degree-11 smoothstep ``462u⁶ − 1980u⁷ + 3465u⁸ − 3080u⁹ + 1386u¹⁰ − 252u¹¹``.

Derivatives are exact to any order: ``S_m' = w^m / B(m + 1, m + 1)`` with
``w = u(1 - u)``, and since ``w`` is quadratic

.. math::

    (w^m)^{(r)} = \sum_{j=0}^{\lfloor r/2 \rfloor}
        \frac{r!}{j!\,(r-2j)!\,2^j}\,
        \frac{m!}{(m-r+j)!}\, w^{m-r+j}\, (1-2u)^{r-2j}\, (-2)^j ,

each term zero where ``r − j > m``. This avoids the monomial form, whose
coefficients alternate in sign and lose every digit near the ends at the
degrees a long chain needs. The phase is closed form,
``∫₀ᵘ I_x(a, a) dx = u I_u(a, a) − ½ I_u(a + 1, a)``.
"""

from __future__ import annotations

from math import factorial

import numpy as np

#: Default ``smoothness`` of the circle-type references: the order through which
#: the reference is continuous where a ramp meets rest or the cruise. Six is the
#: deepest the M1–M4 flat maps read (M1 snap, M2–M4 the payload's sixth
#: derivative); a deeper map (M5–M8) binds its trajectory with its own jet order.
DEFAULT_RAMP_SMOOTHNESS = 6


def check_smoothness(smoothness) -> int:
    """``smoothness`` as an int ``>= 1``, or ``ValueError``."""
    k = int(smoothness)
    if k != smoothness or k < 1:
        raise ValueError(f"smoothness must be an integer >= 1, got {smoothness}")
    return k


def _check(smoothness: int) -> int:
    m = int(smoothness)
    if m != smoothness or m < 0:
        raise ValueError(f"smoothness must be a non-negative integer, got {smoothness}")
    return m


def smooth_step_derivatives(u: float, order: int, smoothness: int) -> np.ndarray:
    """``S_m`` and its first ``order`` derivatives in ``u``.

    ``m`` = ``smoothness`` (``>= 0``) is the order through which the step itself
    is continuous; the phase of a rate ramped through it is one order smoother
    (:func:`ramp_phase_jet`).

    Returns:
        ``(order + 1,)``; all zero for ``u <= 0``, ``[1, 0, …]`` for ``u >= 1``.
    """
    from scipy.special import betainc, betaln

    m = _check(smoothness)
    out = np.zeros(order + 1)
    u = float(u)
    if u <= 0.0:
        return out
    if u >= 1.0:
        out[0] = 1.0
        return out
    out[0] = float(betainc(m + 1, m + 1, u))
    w = u * (1.0 - u)
    w1 = 1.0 - 2.0 * u
    inv_beta = float(np.exp(-betaln(m + 1, m + 1)))
    for r in range(order):
        total = 0.0
        for j in range(r // 2 + 1):
            k = r - j  # derivative order of w^m in this term
            if k > m:
                continue
            coefficient = factorial(r) / (factorial(j) * factorial(r - 2 * j) * 2**j)
            falling = factorial(m) / factorial(m - k)
            total += coefficient * falling * w ** (m - k) * w1 ** (r - 2 * j) * (-2.0) ** j
        out[r + 1] = inv_beta * total
    return out


def smooth_step_integral(u: float, smoothness: int) -> float:
    """``∫₀ᵘ S_m``, ``m`` = ``smoothness``: ``0`` for ``u <= 0``, ``u - 1/2`` for ``u >= 1``."""
    from scipy.special import betainc

    m = _check(smoothness)
    u = float(u)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return u - 0.5
    a = m + 1
    return u * float(betainc(a, a, u)) - 0.5 * float(betainc(a + 1, a, u))


def step_jet(t: float, duration: float, order: int, smoothness: int) -> np.ndarray:
    """``s(t) = S_m(t/T)`` and its first ``order`` time derivatives.

    ``T`` = ``duration``, ``m`` = ``smoothness``, the order through which ``s``
    is continuous.
    """
    T = float(duration)
    out = smooth_step_derivatives(float(t) / T, order, smoothness)
    return out / T ** np.arange(order + 1)


def ramp_phase_jet(
    t: float, rate: float, duration: float, order: int, smoothness: int = DEFAULT_RAMP_SMOOTHNESS
) -> np.ndarray:
    """The phase ``φ`` of a rate ramped from rest, and its first ``order`` time derivatives.

    ``φ̇ = ω S_m(t/T)`` with ``ω`` = ``rate``, ``T`` = ``duration`` and
    ``m = smoothness - 1``, ``φ(0) = 0``: ``φ = 0`` for ``t <= 0``,
    ``ωT ∫₀^{t/T} S_m`` on the ramp and ``ω(t - T/2)`` after it. ``φ`` is
    continuous through order ``smoothness`` (``>= 1``) where the ramp starts
    and ends.

    Returns:
        ``(order + 1,)``: ``φ, φ̇, …``.
    """
    m = check_smoothness(smoothness) - 1
    T = float(duration)
    w = float(rate)
    out = np.zeros(order + 1)
    out[0] = w * T * smooth_step_integral(float(t) / T, m)
    if order >= 1:
        out[1:] = w * step_jet(t, T, order - 1, m)
    return out
