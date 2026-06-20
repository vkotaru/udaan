(differential-flatness-index)=
# Differential flatness

Differential flatness is a property of certain nonlinear systems that
makes trajectory generation and feed-forward control significantly
easier. A system is *differentially flat* if there exists a set of
so-called **flat outputs** from which all states and all inputs can be
recovered **without integrating the dynamics** — purely algebraic
combinations of the flat outputs and a finite number of their time
derivatives.

The concept is due to Fliess, Lévine, Martin, and
Rouchon {footcite}`fliess1995flatness,levine2009analysis`.

## Definition

Consider a nonlinear control-affine system

```{math}
:label: eq-df-system
\dot x = f(x, u), \qquad x \in \mathbb{R}^n,\; u \in \mathbb{R}^m.
```

The system is *differentially flat* if there exists an $m$-vector of
smooth functions — the **flat output** —

```{math}
y = \phi(x,\, u,\, \dot u,\, \ddot u,\, \dots,\, u^{(p)}) \;\in\; \mathbb{R}^m,
```

such that the state and input can be expressed as smooth functions of
the flat output and its derivatives up to some finite order $q$:

```{math}
:label: eq-df-recovery
\begin{aligned}
x &= \phi_x\bigl(y,\, \dot y,\, \ddot y,\, \dots,\, y^{(q)}\bigr), \\
u &= \phi_u\bigl(y,\, \dot y,\, \ddot y,\, \dots,\, y^{(q)}\bigr).
\end{aligned}
```

A few consequences follow immediately:

- **Dimension matching.** The number of flat outputs equals the number
  of inputs, $\dim y = \dim u = m$.
- **Integration-free trajectory generation.** Any smooth curve
  $y(\cdot)$ (with enough derivatives) defines a valid *reference
  trajectory* in the full state/input space — just apply
  {math:numref}`eq-df-recovery`. No ODE solver needed.
- **Feed-forward control.** The resulting $\bigl(x(t),\, u(t)\bigr)$ is
  the nominal state and input pair for tracking $y(\cdot)$. A closed-
  loop controller then only has to reject deviations from it.

## A small example — double integrator

Consider the system

```{math}
\ddot x = u, \qquad x, u \in \mathbb{R}.
```

The choice $y = x$ is a flat output: the state and input fall out as

```{math}
x = y, \qquad \dot x = \dot y, \qquad u = \ddot y.
```

Any smooth reference $y(t)$ that's at least $C^2$ yields a valid
$(x(t), \dot x(t), u(t))$. Trivial in this case, but the same structure
extends to the systems we care about in `udaan`.

With the tuple $(y,\, \dot y,\, \ddot y,\, \dots,\, y^{(q)})$ written
as a $q$-th order **jet** $J^q_t\, y$ of the flat output, the
flat-to-state map {math:numref}`eq-df-recovery` reads compactly as

```{math}
:label: eq-flat-to-state-on-jets
(x, u) \;=\; \Phi\!\bigl(J^q_t\, y\bigr),
```

a single map from the jet space $J^q \mathbb{R}^m$ to the state-input
space $\mathbb{R}^n \times \mathbb{R}^m$. See {doc}`jet` for the formal
definition, the {py:class}`udaan.flatness.Jet` implementation,
and the operations (differentiation, truncation, prolongation) the
abstraction makes available.

```{toctree}
:maxdepth: 1
:hidden:

jet
quadrotor
quadrotor-cspayload
```

## References

```{footbibliography}
```
