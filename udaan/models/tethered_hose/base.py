r"""TetheredHoseBase — quadrotors on a hose pinned to a ground anchor, pure dynamics.

M8 is ``nQ`` quadrotors bolted in series to one hose of ``n`` rigid links whose
bottom end, vertex 0, is pinned to an anchor; M7 is the one-vehicle case, a
quadrotor at the top of a tether
(:class:`TetheredQuadrotorBase`).  Both are M6
(:class:`~udaan.models.multi_quad_hose.MultiQuadHoseBase`) with its root
vertex held still.

.. math::

    Q = (S^2)^{n} \times SO(3)^{n_Q}

Conventions are M6's: link ``i`` runs from vertex ``i`` to vertex ``i + 1``,
so with the anchor at vertex 0 every ``q_i`` points *up* the hose, and
``attachments`` lists the vehicle vertices.  Here every vehicle sits at a
vertex ``>= 1`` (vertex 0 is the anchor) and one sits at the top vertex ``n``.

The pin deletes a coordinate
----------------------------
M6 integrates the root vertex ``x_0`` as a generalised coordinate, so the
constraint ``x_0 = x_{anchor}`` fixes a coordinate rather than adding a force:
it is solved by deleting it.  In M6's system (:eq:`m6-system`) the root row and
column go, the unknowns are ``\dot\omega_1 … \dot\omega_n`` alone, and
``α = \ddot x_0 + g e_3`` is the known ``g e_3``:

.. math::

    \bar M\,\dot\omega = b_{\text{links}} - \bigl[M_{0i}\,\hat q_i\bigr]_i\, g\, e_3 ,

with ``\bar M`` M6's link block (the ``M_{ii} I`` gauge fix included, so it
stays positive definite) and ``b_{\text{links}}`` M6's link rows of the
right-hand side.  Gravity therefore no longer cancels from the link rows.  The
deleted row is what the anchor has to supply:

.. math::

    \lambda = M_{00}\, g\, e_3 - \sum_i M_{0i}\,\hat q_i\,\dot\omega_i
              - \sum_j F_j - \sum_k M_{0k}\, c_k\, q_k ,

the anchor's force on the hose (:meth:`TetheredHoseBase.accelerations` returns
it as ``anchor_force``).  Vertex 0's lumped mass drops out of the motion: the
anchor carries it.  This is M6's Lagrangian with one coordinate removed, a
reparametrised chain, not M6 plus a constraint solver.

The state keeps M6's shape: ``root_position`` is the anchor and
``root_velocity`` is zero, and neither is integrated (the reduced solve returns
``\ddot x_0 = 0``) nor accepted by :meth:`reset`.

Steady states and the ground
----------------------------
A trim is given by the **vehicle positions** (the top vehicle's from the
controller's setpoint, the others evenly on the line from the anchor): each
segment between two held points is then a pinned-pinned discrete catenary with
one taut solution, :func:`~udaan.utils.hose.tethered_trim`.  The anchor sits
on the ground plane ``z = 0`` unless ``anchor_height`` lifts it, and neither
backend has contacts, so a hose slacker than the one whose anchor link leaves
horizontally sags below the anchor and, at ``anchor_height = 0``, through the
floor.  :meth:`TetheredHoseBase.simulate` checks every reference's lowest
vertex before it flies and warns when it is below ground.

The analytical loop's stability
-------------------------------
M6's :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.stability_growth_rate`
is the forward-Euler growth of the free hanging chain's fastest mode.  It does
not describe this machine: it ignores the controller, and the pinned trim's
modes are not the hanging chain's.  Here
:meth:`TetheredHoseBase.stability_growth_rate` is the growth rate of the
sampled closed loop itself — the LQR held over the 5 ms step, four
forward-Euler substeps, linearised at the trim — and the constructor warns
when it is above :data:`~udaan.core.defaults.TETHER_MAX_CLOSED_LOOP_GROWTH_RATE`.
A continuous loop that decays can grow in this one: M7 at eight links on 2 m
decays at 0.29 1/s and its sampled loop grows at 0.25 1/s.
"""

from __future__ import annotations

import time

import numpy as np

from ...control.quadrotor import GeometricAttitudeController, PositionPDController
from ...control.tethered_hose import TetheredHoseController
from ...core.defaults import (
    DEFAULT_TETHER_ANCHOR_HEIGHT,
    DEFAULT_TETHER_LENGTH,
    DEFAULT_TETHER_LINEAR_DENSITY,
    DEFAULT_TETHER_NUM_LINKS,
    DEFAULT_TETHERED_HOSE_LENGTH,
    DEFAULT_TETHERED_HOSE_LINKS_PER_SEGMENT,
    DEFAULT_TETHERED_HOSE_NUM_QUADROTORS,
    TETHER_MAX_CLOSED_LOOP_GROWTH_RATE,
)
from ...core.types import InputType
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from ..multi_quad_hose import MultiQuadHoseBase

_logger = get_logger(__name__)

__all__ = ["TetheredHoseBase", "TetheredQuadrotorBase"]

#: Step of the coarse grid :meth:`TetheredHoseBase.simulate`'s ground check
#: samples a bound reference on, in seconds.
_GROUND_CHECK_DT = 0.05

#: The initial-condition overrides :meth:`TetheredHoseBase.reset` acts on.
_SUPPORTED_RESET_KWARGS = frozenset(
    {"link_attitudes", "link_angular_velocities", "orientations", "angular_velocities"}
)


def _default_attachments(num_links: int, num_quadrotors: int) -> tuple[int, ...]:
    if num_links % num_quadrotors:
        raise ValueError(
            f"{num_links} links do not split evenly over {num_quadrotors} vehicles; "
            "pass attachments explicitly"
        )
    step = num_links // num_quadrotors
    return tuple(step * (k + 1) for k in range(num_quadrotors))


class TetheredHoseBase(MultiQuadHoseBase):
    """``nQ`` quadrotors on one ``n``-link hose pinned to an anchor — pure dynamics (M8).

    Keyword arguments are M6's, except:

    * ``num_quadrotors`` (default 3) and ``num_links`` (default two per
      vehicle) set the default ``attachments``, evenly spaced up the hose with
      one at the top; ``attachments`` overrides them;
    * ``hose_length`` defaults to 3 m and ``hose_mass`` to 0.25 kg/m of it;
    * ``anchor_height`` (default 0) lifts the anchor off the ground plane;
    * ``input`` defaults to ``"wrench"``, the hose LQR;
    * ``sag_ratio`` is not used (the trim comes from the setpoint).

    The anchor is at ``(0, 0, anchor_height)``.  The controller's
    :attr:`~udaan.control.tethered_hose.TetheredHoseController.setpoint` is the
    top vehicle's position; :meth:`reset` puts the system in the static trim
    there.
    """

    #: Defaults a subclass changes to make M7.
    _DEFAULT_NUM_QUADROTORS = DEFAULT_TETHERED_HOSE_NUM_QUADROTORS
    _DEFAULT_LINKS_PER_SEGMENT = DEFAULT_TETHERED_HOSE_LINKS_PER_SEGMENT
    _DEFAULT_LENGTH = DEFAULT_TETHERED_HOSE_LENGTH

    @classmethod
    def _apply_defaults(cls, kwargs: dict) -> dict:
        """The keyword arguments with this system's defaults filled in."""
        out = dict(kwargs)
        if "sag_ratio" in out:
            raise ValueError(
                f"{cls.__name__} takes no sag_ratio: its trim is set by the vehicle positions"
            )
        if "attachments" not in out:
            nQ = int(out.pop("num_quadrotors", cls._DEFAULT_NUM_QUADROTORS))
            n = int(out.get("num_links", cls._DEFAULT_LINKS_PER_SEGMENT * nQ))
            out["num_links"] = n
            out["attachments"] = _default_attachments(n, nQ)
        else:
            out.pop("num_quadrotors", None)
            out.setdefault("num_links", int(tuple(out["attachments"])[-1]))
        out.setdefault("hose_length", cls._DEFAULT_LENGTH)
        out.setdefault("hose_mass", DEFAULT_TETHER_LINEAR_DENSITY * float(out["hose_length"]))
        out.setdefault("input", "wrench")
        return out

    def __init__(self, **kwargs):
        kwargs = self._apply_defaults(kwargs)
        n = int(kwargs["num_links"])
        attachments = tuple(int(a) for a in kwargs["attachments"])
        if not attachments or attachments[0] < 1 or attachments[-1] != n:
            raise ValueError(
                f"a tethered hose needs every vehicle at a vertex in [1, {n}] and one at the "
                f"top vertex {n}; got {attachments}"
            )
        self._check_attachments(attachments)
        anchor_height = float(kwargs.pop("anchor_height", DEFAULT_TETHER_ANCHOR_HEIGHT))
        if not anchor_height >= 0.0:
            raise ValueError(
                f"anchor_height must be >= 0 (the ground is z = 0); got {anchor_height}"
            )
        self._anchor: np.ndarray = np.array([0.0, 0.0, anchor_height])
        self._ground_clearance = float("inf")
        super().__init__(**kwargs)
        TetheredHoseBase.reset(self)

    def _check_attachments(self, attachments: tuple[int, ...]) -> None:
        """Hook for a subclass that restricts the pattern (M7)."""

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        """The hose LQR (``WRENCH``) and the per-vehicle station-keeping PDs (``ACCELERATION``).

        The PDs are M6's baseline, each carrying its own airframe and an equal
        share of the hose as feed-forward.
        """
        nQ = self._num_quadrotors
        self._att_controllers = [
            GeometricAttitudeController(inertia=self._inertias[i]) for i in range(nQ)
        ]
        self._pos_controllers = [
            PositionPDController(mass=self._share_of_system_mass(i)) for i in range(nQ)
        ]
        for i in range(nQ):
            self._pos_controllers[i].setpoint = self._station_setpoint(i)
        setpoint = getattr(getattr(self, "_hose_controller", None), "setpoint", None)
        kwargs = {} if setpoint is None else {"setpoint": setpoint}
        self._hose_controller = TetheredHoseController.from_model(self, **kwargs)
        self._controllers_ready = True

    @property
    def hose_controller(self) -> TetheredHoseController:
        """The pinned hose LQR — what :meth:`simulate` drives under ``InputType.WRENCH``.

        Bind the top vehicle's setpoint, or a
        :class:`~udaan.flatness.tethered_hose.TetherReference`'s ``get`` to fly
        it with the flat feed-forward::

            mdl.hose_controller.setpoint = TetherCircle(...).get
        """
        assert self._hose_controller is not None
        return self._hose_controller  # type: ignore[return-value]

    @hose_controller.setter
    def hose_controller(self, controller):
        self._hose_controller = controller

    # ─── Geometry ──────────────────────────────────────────────────

    @property
    def anchor(self) -> np.ndarray:
        """Position of the anchor (vertex 0), ``(0, 0, anchor_height)``."""
        return np.array(self._anchor, dtype=float)

    @property
    def anchor_height(self) -> float:
        """Height of the anchor above the ground plane, in metres."""
        return float(self._anchor[2])

    @property
    def feedforward_drops(self) -> int:
        """Steps of the last :meth:`simulate` whose reference the flat map refused.

        Each such step ran against the last reference recovered before it.
        """
        return self._feedforward_drops

    @property
    def reference_ground_clearance(self) -> float:
        """Lowest reference height of any vertex but the anchor in the last :meth:`simulate`, m.

        Measured from the ground plane ``z = 0``, over the pre-flight grid and
        every step flown; ``inf`` before a run.
        """
        return self._ground_clearance

    # ─── Dynamics ──────────────────────────────────────────────────

    def accelerations(self, thrusts=None, torques=None) -> dict:
        """Every acceleration at the current state, with the anchor held.

        M6's system (:meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase._assemble`)
        with the root row and column deleted and ``α = g e₃``.

        Returns:
            M6's dict (``root_accel`` is zero) plus ``anchor_force``, the
            force the anchor exerts on the hose.
        """
        n = self._num_links
        nQ = self._num_quadrotors
        thrusts = np.zeros(nQ) if thrusts is None else np.asarray(thrusts, dtype=float).reshape(-1)
        torques = (
            np.zeros((nQ, 3))
            if torques is None
            else np.asarray(torques, dtype=float).reshape(nQ, 3)
        )
        matrix, rhs = self._assemble(thrusts)
        alpha = self._ge3
        link_ang_accels = np.linalg.solve(matrix[3:, 3:], rhs[3:] - matrix[3:, 0:3] @ alpha)
        anchor_force = matrix[0:3, 0:3] @ alpha + matrix[0:3, 3:] @ link_ang_accels - rhs[0:3]
        link_ang_accels = link_ang_accels.reshape(n, 3)

        link_accels = np.zeros((n, 3))
        vertex_accels = np.zeros((n + 1, 3))
        for i in range(n):
            q_i = np.asarray(self.state.link_attitudes[i], dtype=float)
            dq_i = self.state.dq(i)
            link_accels[i] = np.cross(link_ang_accels[i], q_i) - np.dot(dq_i, dq_i) * q_i
            vertex_accels[i + 1] = vertex_accels[i] + self._link_lengths[i] * link_accels[i]

        quad_ang_accels = np.zeros((nQ, 3))
        for k in range(nQ):
            Omega = np.asarray(self.state.angular_velocities[k], dtype=float)
            quad_ang_accels[k] = self._inertia_invs[k] @ (
                torques[k] - np.cross(Omega, self._inertias[k] @ Omega)
            )
        return {
            "root_accel": np.zeros(3),
            "link_ang_accels": link_ang_accels,
            "link_accels": link_accels,
            "vertex_accels": vertex_accels,
            "quad_ang_accels": quad_ang_accels,
            "anchor_force": anchor_force,
        }

    # ─── Stability ─────────────────────────────────────────────────

    def stability_growth_rate(self) -> float:
        """Growth rate, in 1/s, of the analytical model's sampled closed loop at the trim.

        The hose LQR held over one :attr:`dt` step of ``inner_loop_steps``
        (four by default) forward-Euler substeps, linearised at the static trim at the
        controller's ``setpoint(0)``:
        :meth:`~udaan.control.tethered_hose.TetheredHoseController.sampled_growth_rate`.
        Positive means a perturbation grows.
        """
        return self.hose_controller.sampled_growth_rate(self.dt, self._inner_loop_steps)

    def natural_frequencies(self) -> np.ndarray:
        """Open-loop oscillation frequencies of the pinned trim, rad/s, ascending.

        The distinct positive imaginary parts of the open-loop linearisation
        at the static trim (the airframes' rotational modes are integrators
        and contribute none).
        """
        ctrl = self.hose_controller
        ref, inputs = ctrl.equilibrium_reference(0.0)
        A, _ = ctrl.linearize(ref, inputs)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        imag = np.abs(np.linalg.eigvals(basis.T @ A @ basis).imag)
        return np.unique(np.round(imag[imag > 1e-9], 10))

    def _warn_if_stiff(self):
        rate = self.stability_growth_rate()
        if rate > TETHER_MAX_CLOSED_LOOP_GROWTH_RATE:
            _logger.warning(
                "the analytical model's sampled closed loop grows at %.3g 1/s at the trim "
                "(%d links, %d substeps of %.4g s): a perturbation grows instead of decaying, "
                "though the continuous loop may not. Use fewer links (currently %d), a smaller "
                "dt, or the MuJoCo backend.",
                rate,
                self._num_links,
                self._inner_loop_steps,
                self.h,
                self._num_links,
            )

    # ─── Static trim ───────────────────────────────────────────────

    def hover_trim(self, thrust_split=None) -> dict:
        """The static trim at the controller's ``setpoint(0)``.

        Returns M6's keys (``link_attitudes``, ``tensions`` as the ``(n, 3)``
        link force vectors, ``thrusts``, ``orientations``, ``wrench``) plus
        ``vertex_positions`` and ``anchor_force``.

        Raises:
            ValueError: if ``thrust_split`` is given: a pinned hose's split is
                set by its vehicle positions.
        """
        if thrust_split is not None:
            raise ValueError(
                "a pinned hose's thrust split follows from its vehicle positions; set the "
                "controller's setpoint instead of passing thrust_split"
            )
        ctrl = self.hose_controller
        trim = ctrl.trim_at(ctrl.setpoint(0.0)[0])
        ref, inputs = ctrl.equilibrium_reference(0.0)
        wrench = np.zeros((self._num_quadrotors, 4))
        wrench[:, 0] = inputs.thrusts
        wrench[:, 1:] = inputs.moments
        return {
            "link_attitudes": np.asarray(trim.directions),
            "tensions": np.asarray(trim.tensions),
            "thrusts": np.asarray(inputs.thrusts),
            "orientations": np.stack([np.asarray(R) for R in ref.orientations]),
            "wrench": wrench,
            "vertex_positions": np.asarray(trim.vertex_positions),
            "anchor_force": np.asarray(trim.anchor_force),
        }

    # ─── Reset / simulate ──────────────────────────────────────────

    def _reset_reference(self):
        """``(RefState, Inputs)`` the default reset places the system in."""
        ctrl = self.hose_controller
        setpoint_fn = ctrl.setpoint
        owner = getattr(setpoint_fn, "__self__", None)
        if owner is not None and hasattr(owner, "flats"):
            return ctrl.flat_map.recover(owner.flats(0.0))
        return ctrl.equilibrium_reference(0.0)

    def reset(self, **kwargs):
        """Reset time and state; by default to the reference at ``t = 0``.

        Without keywords every link, airframe attitude and rate is the
        reference's at ``t = 0``: the bound
        :class:`~udaan.flatness.tethered_hose.TetherReference`'s, or the static
        trim at the controller's setpoint.  Recognised overrides:
        ``link_attitudes``, ``link_angular_velocities``, ``orientations``,
        ``angular_velocities`` (repaired onto the manifold as M6's are).

        Raises:
            ValueError: for any other keyword, ``root_position`` and
                ``root_velocity`` included: vertex 0 is the anchor.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        ref, _ = self._reset_reference()
        defaults = {
            "link_attitudes": [np.asarray(q, dtype=float) for q in ref.link_attitudes],
            "link_angular_velocities": [
                np.asarray(w, dtype=float) for w in ref.link_angular_velocities
            ],
            "orientations": [np.asarray(R, dtype=float) for R in ref.orientations],
            "angular_velocities": [np.asarray(w, dtype=float) for w in ref.angular_velocities],
        }
        defaults.update(kwargs)
        MultiQuadHoseBase.reset(self, root_position=self._anchor, **defaults)

    def simulate(self, tf, **kwargs):
        """Run the closed loop to time ``tf``.

        Under ``WRENCH`` (the default) the hose LQR flies the controller's
        setpoint.  When the setpoint is a bound method of a
        :class:`~udaan.flatness.tethered_hose.TetherReference` (its ``get``),
        every step regulates against that reference's flat-map state and adds
        its feed-forward wrenches; otherwise against the static trim at the
        setpoint.  Under ``ACCELERATION`` the per-vehicle station-keeping PDs
        run instead.

        Under ``WRENCH``, before flying, the reference is recovered on a 50 ms grid: a lowest
        vertex below the ground plane is warned about (contacts are off, so the
        hose would pass through the floor), and so is a thrust or moment beyond
        the actuator limits (``step()`` would clip it).
        :attr:`reference_ground_clearance` keeps the lowest height seen, over
        that grid and every step flown.

        A step whose reference the flat map refuses runs against the last
        reference it recovered, and is counted in :attr:`feedforward_drops`;
        a refusal at ``t = 0`` raises.
        """
        self.reset(**kwargs)
        self._feedforward_drops = 0
        lqr = self._input_type == InputType.WRENCH
        ctrl = self.hose_controller
        owner = getattr(ctrl.setpoint, "__self__", None)
        flats_fn = getattr(owner, "flats", None)

        self._ground_clearance = float("inf")
        self._last_reference = None
        if lqr:
            grid = np.arange(0.0, float(tf) + 0.5 * _GROUND_CHECK_DT, _GROUND_CHECK_DT)
            peak_thrust, peak_moment = 0.0, 0.0
            for t in grid:
                ref = self._preflight_reference(flats_fn, float(t))
                if ref is not None:
                    self._note_clearance(ref[0])
                    peak_thrust = max(peak_thrust, float(np.max(ref[1].thrusts)))
                    peak_moment = max(
                        peak_moment, float(np.max(np.abs(ref[1].moments) / self._max_torque))
                    )
            if peak_thrust > self._max_thrust or peak_moment > 1.0:
                _logger.warning(
                    "the reference asks for up to %.3g N of thrust (limit %.3g N) and %.3g times "
                    "the moment limit; step() will clip it. Slow the reference down (a longer "
                    "ramp or blend).",
                    peak_thrust,
                    self._max_thrust,
                    peak_moment,
                )
            if self._ground_clearance < 0.0:
                _logger.warning(
                    "the reference takes the hose %.3g m below the ground plane (z = 0); the "
                    "anchor link stops pulling upwards there, and with contacts off the hose "
                    "would pass through the floor",
                    -self._ground_clearance,
                )

        start_t = time.time_ns()
        while self.t < tf:
            if lqr:
                reference = self._reference_at(flats_fn, self.t)
                if reference is not None:
                    self._note_clearance(reference[0])
                u = self.compute_hose_input(reference=reference)
            else:
                u = self.compute_default_input()
            self.step(u)
        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _reference_at(self, flats_fn, t: float):
        """The reference at ``t``: the bound reference's flat map, or the static trim.

        Where the flat map refuses the bound reference, the last reference it
        recovered (a refusal before any raises).
        """
        ctrl = self.hose_controller
        if flats_fn is None:
            return ctrl.equilibrium_reference(t)
        try:
            self._last_reference = ctrl.flat_map.recover(flats_fn(t))
            return self._last_reference
        except ValueError as err:
            if getattr(self, "_last_reference", None) is None:
                raise
            self._feedforward_drops += 1
            if self._feedforward_drops == 1:
                _logger.warning(
                    "flatness feedforward dropped at t=%.4f (%s); this step ran against the last "
                    "reference recovered. Further drops are counted in feedforward_drops.",
                    t,
                    err,
                )
            return self._last_reference

    def _preflight_reference(self, flats_fn, t: float):
        """The reference at ``t`` for the ground check, or ``None`` where the flat map refuses it."""
        ctrl = self.hose_controller
        if flats_fn is None:
            return ctrl.equilibrium_reference(t)
        try:
            return ctrl.flat_map.recover(flats_fn(t))
        except ValueError:
            return None

    def _note_clearance(self, ref) -> None:
        lowest = float(np.min(np.asarray(ref.vertex_positions, dtype=float)[1:, 2]))
        self._ground_clearance = min(self._ground_clearance, lowest)


class TetheredQuadrotorBase(TetheredHoseBase):
    """One quadrotor at the top of an ``n``-link tether pinned to an anchor (M7).

    :class:`TetheredHoseBase` with ``attachments = (n,)``: 2 m of tether at
    0.25 kg/m over 4 links by default.
    """

    _DEFAULT_NUM_QUADROTORS = 1
    _DEFAULT_LINKS_PER_SEGMENT = DEFAULT_TETHER_NUM_LINKS
    _DEFAULT_LENGTH = DEFAULT_TETHER_LENGTH

    def _check_attachments(self, attachments: tuple[int, ...]) -> None:
        if len(attachments) != 1:
            raise ValueError(
                f"{type(self).__name__} carries one quadrotor at the top of the tether; got "
                f"attachments {attachments} (use TetheredHoseBase for several)"
            )
