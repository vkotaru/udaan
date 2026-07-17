(flatness-jet)=
# Jets of flat outputs

This page defines the jet, states its basic operations, and documents
{py:class}`udaan.flatness.Jet` — the Python container that every
flat-to-state map in `udaan` takes as input.

## Formal definition

Let $y : \mathbb{R} \to \mathbb{R}^m$ be a $C^q$-smooth signal
(for example, the flat output of a differentially flat system). Fix a
time $t_0 \in \mathbb{R}$.

```{prf:definition} Jet of a signal at a point
:label: def-jet

The **$q$-th order jet** of $y$ at $t_0$ is the tuple

$$
J^q_{t_0}\, y
\;=\;
\bigl(
  y(t_0),\;
  \dot y(t_0),\;
  \ddot y(t_0),\;
  \dots,\;
  y^{(q)}(t_0)
\bigr)
\;\in\;
\bigl(\mathbb{R}^m\bigr)^{q+1}.
$$

The jet space is the set of all such tuples; we write it
$J^q \mathbb{R}^m$.
```

Two signals agree "to order $q$" at $t_0$ iff their $q$-jets at $t_0$
are equal. This is an equivalence relation, and the equivalence class
is what the differential-geometry literature calls the *abstract jet*
(Wikipedia's [Jet (mathematics)](https://en.wikipedia.org/wiki/Jet_%28mathematics%29)
article defines it as the truncated Taylor polynomial
$\sum_{k=0}^{q} y^{(k)}(t_0) z^k / k!$, which is a canonical
representative of that class). Throughout `udaan` we store and
manipulate the **derivative-tuple representation** in
{prf:ref}`def-jet` — the convention used in the applied flatness and
controls literature (Fliess, Lévine, et al.
{footcite}`fliess1995flatness,levine2009analysis`). The two encodings
carry identical information: the raw-derivative tuple and the Taylor
polynomial are related by the diagonal scaling $1/k!$.

## Operations on jets

Thinking of the flat-to-state map as a map between jet spaces earns
access to a small toolkit of natural operations. Three that show up
repeatedly in flatness work:

```{prf:definition} Differentiation of a jet
:label: def-jet-diff

For $J^q_{t_0} y = (y, \dot y, \dots, y^{(q)})$, the *time-derivative
jet* is the $(q{-}1)$-order jet obtained by shifting every entry one
position down:

$$
D\, J^q_{t_0} y
\;=\;
\bigl(\dot y, \ddot y, \dots, y^{(q)}\bigr)
\;=\;
J^{q-1}_{t_0}\, \dot y.
$$
```

```{prf:definition} Truncation
:label: def-jet-trunc

For $k \leq q$, the *truncation* drops all derivatives above order
$k$:

$$
\pi_k\, J^q_{t_0} y \;=\; \bigl(y, \dot y, \dots, y^{(k)}\bigr).
$$
```

```{prf:definition} Prolongation
:label: def-jet-prolong

Going the other way requires knowing the dynamics — given
$J^q_{t_0} y$, the *prolongation* produces
$J^{q+1}_{t_0} y$ by differentiating $y^{(q)}$ using the system's ODE.
In a flat system, this is always algebraically well-defined.
```

Truncation and differentiation are always available; prolongation
requires side information (the dynamics). In code, the first two are
simple array slices; the third is the substantive content of each
per-system flat-to-state map.

## The flat-to-state map, rewritten

With jets as first-class objects, equation
{math:numref}`eq-flat-to-state-on-jets` from the parent page reads as
a map between jet spaces:

```{math}
:label: eq-flat-to-state-map

\Phi:\; J^q \mathbb{R}^m \;\longrightarrow\; \mathbb{R}^n \times \mathbb{R}^m,
\qquad
\Phi\!\bigl(J^q_t\, y\bigr) \;=\; (x, u).
```

Every system in {doc}`index` provides its own $\Phi$. The input type
is always a jet (or a collection of jets, one per component of the
flat output); the output is the full recovered state plus the
feedforward input.

## The `Jet` class

`udaan` exposes jets as the {py:class}`udaan.flatness.Jet`
dataclass. Storage is a 2-D numpy array of shape
$(\text{order} + 1,\; \text{dim})$; row $k$ is the $k$-th time
derivative of the signal.

### Construction

**Vector signals** (e.g. a 3-D position trajectory, up to snap):

```python
from udaan.flatness import Jet
import numpy as np

x = Jet(np.stack([
    position,       # row 0 → y
    velocity,       # row 1 → ẏ
    acceleration,   # row 2 → ÿ
    jerk,           # row 3 → y⁽³⁾
    snap,           # row 4 → y⁽⁴⁾
]))
x.order    # 4
x.dim      # 3
x[2]       # ndarray(3,) — acceleration
```

**Scalar signals** (e.g. a yaw angle) may be constructed from a 1-D
array of length `order + 1`; the jet promotes it to 2-D with
`dim == 1`:

```python
psi = Jet(np.array([psi_0, psi_dot_0, psi_ddot_0]))
psi.order   # 2
psi.dim     # 1
psi[1]      # float — yaw rate (scalar unwrapped by __getitem__)
```

Factory helpers:

```python
Jet.zeros(order=4, dim=3)          # zero-initialised
Jet.from_list([pos, vel, acc])     # stack derivatives from a Python list
```

### Accessors

| Order $k$ | Accessor         | Standard name |
|-----------|------------------|---------------|
| 0         | `.value`         | position / value |
| 1         | `.velocity`      | velocity |
| 2         | `.acceleration`  | acceleration |
| 3         | `.jerk`          | jerk |
| 4         | `.snap`          | snap (a.k.a. jounce) |
| 5         | `.crackle`       | crackle |
| 6         | `.pop`           | pop |

Each raises `AttributeError` if the jet's `order` is below the
requested derivative. For orders beyond `pop`, use numerical indexing
(`jet[7]`, `jet[8]`, …) — the flexible $N$-link cable derivation needs
this at $q = 2N + 2$.

### Transformations

Reflecting {prf:ref}`def-jet-diff` and {prf:ref}`def-jet-trunc`:

```python
jet.differentiate()     # shift down by one → J^{q-1}(ẏ)
jet.truncate(k)         # drop derivatives above order k
```

### Numpy interop

```python
np.asarray(jet)         # returns the underlying (order+1, dim) array
jet.data                # direct access (same array, no copy)
```

## See also

- {doc}`index` — differential-flatness definition and per-system flat
  outputs.
- {doc}`../preliminaries` — SO(3)/S² conventions used in the recovered
  states.

## References

```{footbibliography}
```
