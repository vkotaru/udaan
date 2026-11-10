"""Tests for the shared derivative engine (:mod:`udaan.flatness.derivatives`).

The Leibniz recursions are verified two independent ways:

* **Exact algebraic identities** that must hold at *arbitrary* order for any
  input — polynomial ground truth for the products, and the defining relations
  ``v = ‖v‖·u`` and ``‖u‖ ≡ 1`` for :func:`normalize_derivatives`.  These carry no
  floating-point truncation error, so they pin the recursion at high order.
* A **finite-difference anchor** on a genuine analytic signal, cross-checking the
  norm/direction derivatives against an entirely different method at low order.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from udaan.flatness import leibniz_dot, leibniz_product, normalize_derivatives


def _poly_derivs(coeffs: list[float], t: float, order: int) -> list[float]:
    """Derivative tuple ``[p, ṗ, …, p^(order)]`` at ``t`` of the polynomial with
    the given power-basis ``coeffs`` (``coeffs[i]`` multiplies ``t**i``)."""
    out = []
    for k in range(order + 1):
        val = 0.0
        for i, c in enumerate(coeffs):
            if i >= k:
                # d^k/dt^k of c t^i = c * i!/(i-k)! * t^(i-k)
                fall = 1
                for m in range(k):
                    fall *= i - m
                val += c * fall * t ** (i - k)
        out.append(val)
    return out


def test_leibniz_product_matches_polynomial():
    # a(t) = t^2, b(t) = t^3  ->  a*b = t^5, whose derivatives are known exactly.
    t0 = 1.7
    order = 6
    a = _poly_derivs([0, 0, 1], t0, order)  # t^2
    b = _poly_derivs([0, 0, 0, 1], t0, order)  # t^3
    prod = _poly_derivs([0, 0, 0, 0, 0, 1], t0, order)  # t^5
    got = leibniz_product(a, b)
    np.testing.assert_allclose(got, prod, rtol=0, atol=1e-9)


def test_leibniz_dot_matches_polynomial():
    # a(t) = [t^2, t, 1], b(t) = [1, t^2, t]  ->  <a,b> = t^2 + t^3 + t.
    t0 = 0.9
    order = 5
    ax = [_poly_derivs(c, t0, order) for c in ([0, 0, 1], [0, 1], [1])]
    bx = [_poly_derivs(c, t0, order) for c in ([1], [0, 0, 1], [0, 1])]
    a = [np.array([ax[0][k], ax[1][k], ax[2][k]]) for k in range(order + 1)]
    b = [np.array([bx[0][k], bx[1][k], bx[2][k]]) for k in range(order + 1)]
    expected = _poly_derivs([0, 1, 1, 1], t0, order)  # t + t^2 + t^3
    got = leibniz_dot(a, b)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5, 6])
def test_normalize_reconstruction_and_unit_identities(order):
    """For an arbitrary derivative tuple, ``normalize_derivatives`` must satisfy
    the exact relations it is defined by, at every order:

    * n[0] = ‖v[0]‖,
    * v = ‖v‖·u   (Leibniz reconstruction of v from n and u),
    * ‖u‖ ≡ 1     (so leibniz_dot(u,u) = [1, 0, 0, …]).
    """
    rng = np.random.default_rng(20260717 + order)
    v = [rng.standard_normal(3) for _ in range(order + 1)]
    v[0] += 3.0 * np.sign(v[0] + 1e-9)  # keep it comfortably away from zero norm

    n, u = normalize_derivatives(v)

    assert n[0] == pytest.approx(float(np.linalg.norm(v[0])))

    # v^(k) == sum_j C(k,j) n^(j) u^(k-j)
    for k in range(order + 1):
        recon = sum(comb(k, j) * n[j] * u[k - j] for j in range(k + 1))
        np.testing.assert_allclose(recon, v[k], rtol=0, atol=1e-9)

    # ‖u‖^2 ≡ 1  ->  its 0th derivative is 1 and all higher derivatives vanish.
    uu = leibniz_dot(u, u)
    assert uu[0] == pytest.approx(1.0)
    for k in range(1, order + 1):
        assert uu[k] == pytest.approx(0.0, abs=1e-9)


def test_normalize_matches_finite_difference():
    """Anchor the norm/direction derivatives against central finite differences
    of a genuine analytic signal, independent of the Leibniz recursion."""
    t0, order = 0.35, 3

    def signal(t):
        return np.array([np.cos(t), np.sin(2.0 * t), t + 2.0])

    # exact derivative tuple of `signal` at t0
    def signal_derivs(t):
        d = []
        for k in range(order + 1):
            d.append(
                np.array(
                    [
                        np.cos(t + k * np.pi / 2),  # d^k cos t
                        (2.0**k) * np.sin(2.0 * t + k * np.pi / 2),  # d^k sin 2t
                        (t + 2.0) if k == 0 else (1.0 if k == 1 else 0.0),
                    ]
                )
            )
        return d

    n, u = normalize_derivatives(signal_derivs(t0))

    # central finite differences of ‖signal‖ and signal/‖signal‖
    h = 1e-3

    def norm_of(t):
        return float(np.linalg.norm(signal(t)))

    def dir_of(t):
        s = signal(t)
        return s / np.linalg.norm(s)

    # 1st–3rd central-difference stencils
    fd_n1 = (norm_of(t0 + h) - norm_of(t0 - h)) / (2 * h)
    fd_n2 = (norm_of(t0 + h) - 2 * norm_of(t0) + norm_of(t0 - h)) / h**2
    fd_n3 = (
        norm_of(t0 + 2 * h) - 2 * norm_of(t0 + h) + 2 * norm_of(t0 - h) - norm_of(t0 - 2 * h)
    ) / (2 * h**3)
    assert n[1] == pytest.approx(fd_n1, abs=1e-5)
    assert n[2] == pytest.approx(fd_n2, abs=1e-4)
    assert n[3] == pytest.approx(fd_n3, abs=1e-3)

    fd_u1 = (dir_of(t0 + h) - dir_of(t0 - h)) / (2 * h)
    fd_u2 = (dir_of(t0 + h) - 2 * dir_of(t0) + dir_of(t0 - h)) / h**2
    np.testing.assert_allclose(u[1], fd_u1, atol=1e-5)
    np.testing.assert_allclose(u[2], fd_u2, atol=1e-4)


def test_free_fall_singularity_raises():
    with pytest.raises(ValueError):
        normalize_derivatives([np.zeros(3), np.ones(3)])


def test_leibniz_length_mismatch_raises():
    with pytest.raises(ValueError):
        leibniz_product([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        leibniz_dot([np.ones(3)], [np.ones(3), np.ones(3)])
