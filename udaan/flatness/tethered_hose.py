r"""Differential flatness for quadrotors on a hose pinned to a ground anchor — M7 / M8.

Flat output::

    y = (𝛕_0, {𝛕_a : a a vehicle vertex, a ≠ n}, {ψ_k}_{k=1..nQ})

where ``𝛕_0`` is the force the **anchor link** exerts on the anchor vertex
(it points up the hose, along ``q_0``), ``𝛕_a`` is M6's vector tension in the
link leaving vehicle vertex ``a``, and ``ψ_k`` is vehicle ``k``'s yaw.
Counting: ``3 + 3(nQ − 1) + nQ = 4nQ = dim u``.  The dynamics being inverted
are :class:`~udaan.models.tethered_hose.TetheredHoseBase`.

How the pin changes M6's flat output
------------------------------------
M6's flat output is ``(x_0, {𝛕_a}, {ψ_k})`` with the root position ``x_0``
free.  Pinning vertex 0 fixes ``x_0``, and the anchor's reaction absorbs
vertex 0's force balance

.. math::

    \bar m_0\, g\, e_3 = \lambda + 𝛕_0 ,

so ``𝛕_0`` is no longer determined by the root's motion and joins the flat
output in its place.  That is exactly M6's flat output for a hose with one
more "vehicle", at vertex 0, whose "thrust" is the anchor reaction: so this
map **is** :class:`~udaan.flatness.MultiQuadHose` with attachments
``(0, a_1, …, a_{nQ})``, the anchor in the role of the vehicle at vertex 0
and a constant root jet.  Nothing is re-derived.  The placeholder vehicle's
mass (:data:`ANCHOR_PLACEHOLDER_MASS`) enters only its own recovered thrust,
which is discarded; the anchor force is reported from ``𝛕_0`` directly.

The vehicle position is **not** a flat output: holding the top vehicle still
leaves the hose's pinned-pinned modes as undamped zero dynamics.  Vehicle
positions are outputs of this map, as in M6.

Derivative orders required::

    𝛕_0  through its (2n + 2)-th derivative   (:attr:`TetheredHose.jet_order`)
    𝛕_a  through its (2(n − a) + 2)-th derivative
    ψ_k  through ψ̈_k

References
----------
Three flat-output trajectories, each placing the vehicles evenly on the line
from the anchor to the top vehicle
(:func:`~udaan.utils.hose.tethered_vehicle_positions`):

* :class:`TetherHold` — a static trim, every flat tension constant;
* :class:`TetherBlend` — from one static trim to another, every flat tension
  blended by one smooth step (:func:`smoothstep_jet`) continuous through the
  map's whole jet order, so the vehicle paths are outputs of the blend;
* :class:`TetherCircle` — a circle about the anchor started from rest: the
  formation turns rigidly about the anchor's vertical axis at a rate that
  ramps up smoothly to ``Ω``, while each flat tension blends, in the turning
  frame, from the static trim to the whirl trim at ``Ω`` (the steady rotation
  through the same vehicle positions).  After the ramp the reference is the
  exact whirl; during it the vehicles leave the circle by what the blend asks.

Each exposes ``flats(t)`` (what :meth:`TetheredHose.recover` takes) and the
controller setpoint protocol ``get(t) -> (pos, vel, acc)`` of the **top**
vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, factorial

import numpy as np

from ..core.defaults import (
    DEFAULT_GENERATED_QUAD_INERTIA,
    DEFAULT_GENERATED_QUAD_MASS,
    DEFAULT_HOSE_MIN_TENSION,
    DEFAULT_MAX_THRUST,
    DEFAULT_TETHER_ANCHOR_HEIGHT,
    DEFAULT_TETHER_LENGTH,
    DEFAULT_TETHER_LINEAR_DENSITY,
    DEFAULT_TETHER_NUM_LINKS,
    GRAVITY,
    TETHER_CIRCLE_THRUST_FRACTION,
)
from ..utils.hose import TetheredTrim, tethered_trim, tethered_vehicle_positions
from .base import Flat2State
from .jet import Jet
from .multi_quad_hose import MultiQuadHose, MultiQuadHoseFlats, MultiQuadHoseRefState

__all__ = [
    "ANCHOR_PLACEHOLDER_MASS",
    "TetherBlend",
    "TetherCircle",
    "TetherHold",
    "TetherReference",
    "TetheredHose",
    "TetheredHoseFlats",
    "TetheredHoseInputs",
    "circle_rate_bound",
    "smoothstep_jet",
]

_E3 = np.array([0.0, 0.0, 1.0])

#: Mass, in kg, of the placeholder vehicle M6's map is given at vertex 0.  Only
#: that vehicle's own recovered thrust ``(m + m_0) g e_3 - 𝛕_0`` depends on it;
#: a mass this large keeps that vector near vertical and far from zero, so its
#: attitude recovery (which this map discards) is never singular.
ANCHOR_PLACEHOLDER_MASS = 1.0e3  # kg


@dataclass
class TetheredHoseFlats:
    """Flat output of M7 / M8 at one instant."""

    #: dim-3 :class:`Jet` of ``𝛕_0``, the anchor link's force on the anchor
    #: vertex; order ≥ :attr:`TetheredHose.jet_order`.
    anchor_tension: Jet
    #: one dim-3 :class:`Jet` per vehicle **except the top one**, in vehicle
    #: order: ``𝛕_a`` in the link leaving that vehicle upwards.
    tensions: list[Jet] = field(default_factory=list)
    #: ``nQ`` scalar yaw jets, order ≥ 2.
    psi: list[Jet] = field(default_factory=list)


@dataclass
class TetheredHoseInputs:
    """Feed-forward inputs recovered from the flat output."""

    #: ``(nQ,)`` collective thrusts.
    thrusts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: ``(nQ, 3)`` body moments.
    moments: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: ``(n,)`` link tension magnitudes.
    tensions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: ``(3,)`` force the anchor exerts on the hose, ``m_0 g e_3 - 𝛕_0``.
    anchor_force: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_wrench(self) -> np.ndarray:
        """``(nQ, 4)`` stack of ``[thrust, Mx, My, Mz]``."""
        return np.column_stack([self.thrusts, self.moments])


class TetheredHose(Flat2State):
    """Flat-to-state map of ``nQ`` quadrotors on a hose pinned at vertex 0.

    Parameters mirror :class:`~udaan.models.tethered_hose.TetheredHoseBase`
    (``hose_mass`` defaults to 0.25 kg per metre of hose);
    ``attachments`` lists the vehicle vertices, all ``>= 1``, the last ``n``.
    ``vertex_masses[0]`` is the hose mass the anchor carries.

    Raises:
        ValueError: for a vehicle at vertex 0 (that vertex is the anchor) or
            no vehicle at the last vertex (the map is not flat).
    """

    Flats = TetheredHoseFlats
    RefState = MultiQuadHoseRefState
    Inputs = TetheredHoseInputs

    def __init__(
        self,
        *,
        num_links: int = DEFAULT_TETHER_NUM_LINKS,
        attachments=None,
        vertex_masses=None,
        link_lengths=None,
        hose_length: float = DEFAULT_TETHER_LENGTH,
        hose_mass: float | None = None,
        masses=None,
        inertias=None,
        gravity: float = GRAVITY,
        min_tension: float = DEFAULT_HOSE_MIN_TENSION,
        anchor=None,
    ):
        n = int(num_links)
        att = (n,) if attachments is None else tuple(int(a) for a in attachments)
        if not att or att[0] < 1:
            raise ValueError(
                f"vertex 0 is the anchor, so every vehicle must sit at a vertex >= 1; got {att}"
            )
        nQ = len(att)
        quad_masses = (
            np.full(nQ, DEFAULT_GENERATED_QUAD_MASS)
            if masses is None
            else np.broadcast_to(np.asarray(masses, dtype=float), (nQ,)).copy()
        )
        if inertias is None:
            tensors = np.tile(np.asarray(DEFAULT_GENERATED_QUAD_INERTIA, dtype=float), (nQ, 1, 1))
        else:
            arr = np.asarray(inertias, dtype=float)
            tensors = np.tile(arr, (nQ, 1, 1)) if arr.shape == (3, 3) else arr.reshape(nQ, 3, 3)
        self._inner = MultiQuadHose(
            num_links=n,
            attachments=(0, *att),
            vertex_masses=vertex_masses,
            link_lengths=link_lengths,
            hose_length=hose_length,
            hose_mass=(
                DEFAULT_TETHER_LINEAR_DENSITY
                * (float(hose_length) if link_lengths is None else float(np.sum(link_lengths)))
                if hose_mass is None
                else hose_mass
            ),
            masses=np.concatenate([[ANCHOR_PLACEHOLDER_MASS], quad_masses]),
            inertias=np.concatenate([np.eye(3)[None], tensors]),
            gravity=gravity,
            min_tension=min_tension,
        )
        self.num_links = n
        self.attachments = att
        self.num_quadrotors = nQ
        self.link_lengths = self._inner.link_lengths
        self.vertex_masses = self._inner.vertex_masses
        self.masses = quad_masses
        self.inertias = tensors
        self.gravity = float(gravity)
        self.min_tension = float(min_tension)
        #: ``m̄_j`` — hose plus airframe mass at each vertex; entry 0 is the
        #: anchor's share of the hose, carried by the anchor.
        self.vertex_total_masses = self._inner.vertex_total_masses.copy()
        self.vertex_total_masses[0] = self.vertex_masses[0]
        self.anchor = (
            np.array([0.0, 0.0, DEFAULT_TETHER_ANCHOR_HEIGHT])
            if anchor is None
            else np.asarray(anchor, dtype=float).reshape(3).copy()
        )

    @classmethod
    def from_model(cls, model) -> TetheredHose:
        """Build from a :class:`~udaan.models.tethered_hose.TetheredHoseBase`-like model."""
        return cls(
            num_links=model.num_links,
            attachments=model.attachments,
            vertex_masses=np.asarray(model.vertex_masses, dtype=float).copy(),
            link_lengths=np.asarray(model.link_lengths, dtype=float).copy(),
            masses=np.asarray(model.masses, dtype=float).copy(),
            inertias=np.asarray(model.inertias, dtype=float).copy(),
            anchor=np.asarray(model.anchor, dtype=float).copy(),
        )

    # ─── Derived parameters ────────────────────────────────────────

    @property
    def jet_order(self) -> int:
        """``2n + 2`` — the order the anchor tension's jet must reach."""
        return int(self._inner.tension_jet_orders[0])

    @property
    def tension_jet_orders(self) -> tuple[int, ...]:
        """Required jet order of each flat tension, ``𝛕_0`` first."""
        return tuple(int(o) for o in self._inner.tension_jet_orders)

    @property
    def hose_length(self) -> float:
        """Total taut hose length ``Σ_i ℓ_i``, in metres."""
        return float(np.sum(self.link_lengths))

    # ─── Steady states ─────────────────────────────────────────────

    def trim(self, top_position, whirl_rate: float = 0.0) -> TetheredTrim:
        """The steady state with the top vehicle at ``top_position``.

        The other vehicles sit evenly on the line from the anchor
        (:func:`~udaan.utils.hose.tethered_vehicle_positions`); with
        ``whirl_rate`` the state is the steady rotation about the anchor's
        vertical axis through those positions.  See
        :func:`~udaan.utils.hose.tethered_trim`.
        """
        positions = tethered_vehicle_positions(top_position, self.num_quadrotors, self.anchor)
        return tethered_trim(
            self.link_lengths,
            self.vertex_total_masses,
            self.attachments,
            positions,
            anchor=self.anchor,
            gravity=self.gravity,
            whirl_rate=whirl_rate,
        )

    def constant_flats(self, flat_tensions) -> TetheredHoseFlats:
        """The flat output holding ``flat_tensions`` (``(nQ, 3)``) constant, yaws zero."""
        values = np.asarray(flat_tensions, dtype=float).reshape(self.num_quadrotors, 3)
        jets = []
        for k, order in enumerate(self.tension_jet_orders):
            rows = np.zeros((order + 1, 3))
            rows[0] = values[k]
            jets.append(Jet(rows))
        return TetheredHoseFlats(
            anchor_tension=jets[0],
            tensions=jets[1:],
            psi=[Jet(np.zeros(3)) for _ in range(self.num_quadrotors)],
        )

    # ─── Recovery ──────────────────────────────────────────────────

    def recover(self, flats: TetheredHoseFlats) -> tuple[MultiQuadHoseRefState, TetheredHoseInputs]:
        """Reference state and feed-forward from the flat output.

        Raises:
            ValueError: for a jet shallower than the map needs, a wrong number
                of tensions or yaws, or a reference the M6 map refuses (a slack
                link, a derived tension under :attr:`min_tension`, a vehicle in
                free fall).
        """
        nQ = self.num_quadrotors
        if len(flats.tensions) != nQ - 1:
            raise ValueError(
                f"TetheredHose: expected {nQ - 1} vehicle tension jets (every vehicle but the "
                f"top one); got {len(flats.tensions)}"
            )
        if len(flats.psi) != nQ:
            raise ValueError(f"TetheredHose: expected {nQ} yaw jets; got {len(flats.psi)}")
        root = np.zeros((self._inner.jet_order + 1, 3))
        root[0] = self.anchor
        inner_flats = MultiQuadHoseFlats(
            x_0=Jet(root),
            tensions=[flats.anchor_tension, *flats.tensions],
            psi=[Jet(np.zeros(3)), *flats.psi],
        )
        ref, inputs = self._inner.recover(inner_flats)
        state = MultiQuadHoseRefState(
            root_position=ref.root_position,
            root_velocity=ref.root_velocity,
            root_acceleration=ref.root_acceleration,
            link_attitudes=ref.link_attitudes,
            link_angular_velocities=ref.link_angular_velocities,
            link_angular_accelerations=ref.link_angular_accelerations,
            vertex_positions=ref.vertex_positions,
            vertex_velocities=ref.vertex_velocities,
            positions=ref.positions[1:],
            velocities=ref.velocities[1:],
            orientations=ref.orientations[1:],
            angular_velocities=ref.angular_velocities[1:],
            angular_accelerations=ref.angular_accelerations[1:],
        )
        tau0 = np.asarray(flats.anchor_tension[0], dtype=float)
        out = TetheredHoseInputs(
            thrusts=np.asarray(inputs.thrusts[1:], dtype=float),
            moments=np.asarray(inputs.moments[1:], dtype=float),
            tensions=np.asarray(inputs.tensions, dtype=float),
            anchor_force=self.vertex_masses[0] * self.gravity * _E3 - tau0,
        )
        return state, out

    def top_acceleration(self, flats: TetheredHoseFlats) -> np.ndarray:
        """Reference acceleration of the top vehicle, from the sweep."""
        root = np.zeros((self._inner.jet_order + 1, 3))
        root[0] = self.anchor
        chain = self._inner.cascade(
            MultiQuadHoseFlats(
                x_0=Jet(root),
                tensions=[flats.anchor_tension, *flats.tensions],
                psi=[Jet(np.zeros(3)), *flats.psi],
            )
        )
        return np.asarray(chain.vertex_positions[self.num_links][2], dtype=float)


# ─── Smooth step ──────────────────────────────────────────────────────


def smoothstep_jet(t: float, duration: float, order: int, smoothness: int) -> np.ndarray:
    r"""Time jet of the smooth step from 0 at ``t = 0`` to 1 at ``t = duration``.

    ``s(u) = I_u(m + 1, m + 1)`` with ``u = t/T`` and ``m`` = ``smoothness``:
    the regularised incomplete beta function, the polynomial of degree
    ``2m + 1`` whose first ``m`` derivatives vanish at both ends, so the step
    is ``C^m`` where it meets the constants on either side.  Its derivatives
    are ``s^{(r+1)}(u) = (w^m)^{(r)} / B(m+1, m+1)`` with ``w = u(1 − u)``, and
    since ``w`` is quadratic

    .. math::

        (w^m)^{(r)} = \sum_{j=0}^{\lfloor r/2 \rfloor}
            \frac{r!}{j!\,(r-2j)!\,2^j}\,
            \frac{m!}{(m-r+j)!}\, w^{m-r+j}\, (1-2u)^{r-2j}\, (-2)^j ,

    each term zero where ``r − j > m``.  This avoids the monomial form, whose
    coefficients alternate in sign and lose every digit near the ends at the
    degrees a long hose needs.

    Returns:
        ``(order + 1,)``: ``s`` and its first ``order`` time derivatives.
    """
    from scipy.special import betainc, betaln

    m = int(smoothness)
    T = float(duration)
    out = np.zeros(order + 1)
    u = float(t) / T
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
            k = r - j  # derivative order of f(w) = w^m in this term
            if k > m:
                continue
            coefficient = factorial(r) / (factorial(j) * factorial(r - 2 * j) * 2**j)
            falling = factorial(m) / factorial(m - k)
            total += coefficient * falling * w ** (m - k) * w1 ** (r - 2 * j) * (-2.0) ** j
        out[r + 1] = inv_beta * total / T ** (r + 1)
    return out


# ─── References ───────────────────────────────────────────────────────


class TetherReference:
    """A flat-output trajectory of a :class:`TetheredHose` map.

    Subclasses implement :meth:`tension_jets`; this class turns them into
    :meth:`flats` and the top vehicle's setpoint :meth:`get`.
    """

    def __init__(self, flat_map: TetheredHose):
        self.flat_map = flat_map
        self._cached_t: float | None = None
        self._cached: tuple | None = None

    def tension_jets(self, t: float) -> list[np.ndarray]:
        """``nQ`` arrays, ``(order_k + 1, 3)`` each: ``𝛕_0`` first, then each ``𝛕_a``."""
        raise NotImplementedError

    def flats(self, t: float) -> TetheredHoseFlats:
        """The flat output at ``t`` (yaws held at zero)."""
        jets = [Jet(rows) for rows in self.tension_jets(t)]
        return TetheredHoseFlats(
            anchor_tension=jets[0],
            tensions=jets[1:],
            psi=[Jet(np.zeros(3)) for _ in range(self.flat_map.num_quadrotors)],
        )

    def recover(self, t: float):
        """``(RefState, Inputs)`` at ``t``, cached for repeated queries at one instant."""
        if self._cached_t != t or self._cached is None:
            flats = self.flats(t)
            ref, inputs = self.flat_map.recover(flats)
            self._cached = (ref, inputs, flats)
            self._cached_t = t
        return self._cached[0], self._cached[1]

    def get(self, t: float):
        """``(pos, vel, acc)`` of the top vehicle — the controller setpoint protocol."""
        ref, _ = self.recover(t)
        assert self._cached is not None
        acc = self.flat_map.top_acceleration(self._cached[2])
        return (
            np.asarray(ref.positions[-1], dtype=float),
            np.asarray(ref.velocities[-1], dtype=float),
            acc,
        )

    def _constant_jets(self, values) -> list[np.ndarray]:
        flats = self.flat_map.constant_flats(values)
        return [np.asarray(j, dtype=float) for j in (flats.anchor_tension, *flats.tensions)]


class TetherHold(TetherReference):
    """The static trim with the top vehicle at ``top_position``."""

    def __init__(self, flat_map: TetheredHose, top_position):
        super().__init__(flat_map)
        self.top_position = np.asarray(top_position, dtype=float).reshape(3).copy()
        self.trim = flat_map.trim(self.top_position)

    def tension_jets(self, t: float) -> list[np.ndarray]:
        return self._constant_jets(self.trim.flat_tensions)


class TetherBlend(TetherReference):
    r"""From the static trim at ``start`` to the one at ``end``, over ``duration`` seconds.

    Every flat tension follows ``𝛕_k(t) = 𝛕_k^A + s(t)(𝛕_k^B − 𝛕_k^A)`` with
    one :func:`smoothstep_jet` continuous through :attr:`TetheredHose.jet_order`,
    starting at ``start_time``.  The vehicle paths are what the flat map makes
    of that, not a straight line.
    """

    def __init__(
        self,
        flat_map: TetheredHose,
        start,
        end,
        duration: float,
        start_time: float = 0.0,
    ):
        super().__init__(flat_map)
        if not float(duration) > 0.0:
            raise ValueError(f"duration must be positive; got {duration}")
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.start_trim = flat_map.trim(start)
        self.end_trim = flat_map.trim(end)

    def tension_jets(self, t: float) -> list[np.ndarray]:
        orders = self.flat_map.tension_jet_orders
        s = smoothstep_jet(t - self.start_time, self.duration, max(orders), max(orders))
        a = self.start_trim.flat_tensions
        b = self.end_trim.flat_tensions
        out = []
        for k, order in enumerate(orders):
            rows = np.outer(s[: order + 1], b[k] - a[k])
            rows[0] += a[k]
            out.append(rows)
        return out


class TetherCircle(TetherReference):
    r"""A circle about the anchor, started from rest.

    The top vehicle starts at ``top_position`` in the static trim; the
    formation's azimuth about the anchor's vertical axis then turns at
    ``Ω(t) = rate · s(t/ramp_time)`` (:func:`smoothstep_jet`, continuous
    through the map's jet order), and every flat tension blends, in the
    turning frame, from the static trim to the whirl trim at ``rate`` through
    the same vehicle positions:

    .. math::

        𝛕_k(t) = R_z(φ(t))\,\bigl(𝛕_k^{s} + s(t)(𝛕_k^{w} - 𝛕_k^{s})\bigr),
        \qquad \dot φ = Ω(t),\ φ(0) = 0 .

    From ``ramp_time`` on this is the exact steady whirl: the top vehicle on
    the circle of the start position's radius and height.
    """

    def __init__(
        self,
        flat_map: TetheredHose,
        top_position,
        rate: float,
        ramp_time: float,
    ):
        super().__init__(flat_map)
        if not float(ramp_time) > 0.0:
            raise ValueError(f"ramp_time must be positive; got {ramp_time}")
        self.top_position = np.asarray(top_position, dtype=float).reshape(3).copy()
        self.rate = float(rate)
        self.ramp_time = float(ramp_time)
        self.static_trim = flat_map.trim(self.top_position)
        self.whirl_trim = flat_map.trim(self.top_position, whirl_rate=self.rate)

    def azimuth(self, t: float) -> float:
        """``φ(t)``, the formation's turn about the anchor since ``t = 0``, in rad."""
        if t <= 0.0:
            return 0.0
        T = self.ramp_time
        if t >= T:
            return self.rate * (0.5 * T + (t - T))
        from scipy.special import betainc

        m = max(self.flat_map.tension_jet_orders)
        u = t / T
        s = float(betainc(m + 1, m + 1, u))
        # ∫_0^u I_x(a, a) dx = u I_u(a, a) - ½ I_u(a + 1, a)
        return self.rate * T * (u * s - 0.5 * float(betainc(m + 2, m + 1, u)))

    def tension_jets(self, t: float) -> list[np.ndarray]:
        orders = self.flat_map.tension_jet_orders
        K = max(orders)
        s = smoothstep_jet(t, self.ramp_time, K, K)
        # φ^{(k)} = Ω s^{(k-1)} for k ≥ 1; z = e^{iφ} by z' = iφ' z (Leibniz).
        phi = np.zeros(K + 2)
        phi[0] = self.azimuth(t)
        phi[1:] = self.rate * s
        z = np.zeros(K + 1, dtype=complex)
        z[0] = np.exp(1j * phi[0])
        for k in range(K):
            z[k + 1] = sum(comb(k, j) * 1j * phi[j + 1] * z[k - j] for j in range(k + 1))
        out = []
        a = self.static_trim.flat_tensions
        b = self.whirl_trim.flat_tensions - a
        for idx, order in enumerate(orders):
            horizontal = np.zeros(order + 1, dtype=complex)
            vertical = np.zeros(order + 1)
            horizontal[0] = complex(a[idx, 0], a[idx, 1])
            vertical[0] = a[idx, 2]
            horizontal += s[: order + 1] * complex(b[idx, 0], b[idx, 1])
            vertical += s[: order + 1] * b[idx, 2]
            rows = np.zeros((order + 1, 3))
            for r in range(order + 1):
                value = sum(comb(r, j) * z[j] * horizontal[r - j] for j in range(r + 1))
                rows[r] = (value.real, value.imag, vertical[r])
            out.append(rows)
        return out


def circle_rate_bound(
    flat_map: TetheredHose,
    top_position,
    max_thrust: float = DEFAULT_MAX_THRUST,
    thrust_fraction: float = TETHER_CIRCLE_THRUST_FRACTION,
) -> float:
    r"""The largest whirl rate about the anchor a circle through ``top_position`` may use.

    ``min(½√(g/ℓ), Ω_f)``: ``ℓ`` the longest link, so the whirl stays well
    below the links' own pendulum rate, and ``Ω_f`` the largest rate whose
    steady whirl (:meth:`TetheredHose.trim` with ``whirl_rate``) asks no vehicle
    for more than ``thrust_fraction × max_thrust``, found by bisection; the rest
    of the thrust is left to the controller's corrections.  Returns 0 when even
    the static trim exceeds that.
    """
    max_thrust = float(thrust_fraction) * float(max_thrust)
    pendulum = 0.5 * float(np.sqrt(flat_map.gravity / float(np.max(flat_map.link_lengths))))

    def peak(rate: float) -> float:
        trim = flat_map.trim(top_position, whirl_rate=rate)
        return float(np.max(np.linalg.norm(trim.thrust_vectors, axis=1)))

    if peak(pendulum) <= max_thrust:
        return pendulum
    if peak(0.0) > max_thrust:
        return 0.0
    low, high = 0.0, pendulum
    for _ in range(60):
        mid = 0.5 * (low + high)
        if peak(mid) <= max_thrust:
            low = mid
        else:
            high = mid
    return low
