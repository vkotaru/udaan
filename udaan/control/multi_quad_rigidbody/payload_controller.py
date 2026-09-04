r"""Payload-tracking cascade for ``nQ`` quadrotors sharing one cable-suspended
**rigid-body** payload (M4).

.. admonition:: Provisional

   This cascade is **provisional**.  It works — it holds the payload's pose at
   every grasp geometry the mechanism allows — but it is deliberately not
   finished here: the cable-direction feedforward needs an M4 flat-to-state map
   that :mod:`udaan.flatness` does not have yet, and the controller's own test
   coverage is thinner than the model's.  Completing and testing it, and wiring
   the feedforward, is a later step; until then treat the M4 model, not this
   file, as the verified artefact, and do not extend the cascade in place.

The rigid-body sibling of
:class:`~udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController`.  M3
commands a net *force* on a point mass and splits it across the cables; M4 has
to command a net *wrench* — force **and** moment — and split that.  The cascade
is four stages:

1. **Payload pose loop.**  A PD on the payload position/velocity error fixes the
   desired payload acceleration, and a geometric SO(3) PD on the payload
   attitude fixes the desired body moment:

   .. math::

       a_L^d &= \ddot x_L^d - k_p^L e_{x_L} - k_d^L e_{v_L}, \\
       M_L^d &= -k_R e_R - k_\Omega e_\Omega + \Omega_L \times J_L \Omega_L .

   The position gains are in *acceleration* units for the same ``nQ``-invariance
   reason M3 documents.  The attitude gains cannot be: a moment command is
   divided across the cables by the grasp map, whose scale is set by the
   attachment geometry, so the physical torque is what has to be commanded.

2. **Grasp-map allocation.**  The desired body-frame wrench

   .. math::

       W^d = \begin{bmatrix} R_L^\top m_L (a_L^d + g e_3) \\ M_L^d \end{bmatrix}

   is distributed over the ``3 n_Q`` body-frame cable forces ``\tau_i`` applied
   *to the load* by

   .. math::
       :label: m4c-split

       \tau = \Phi^{+} W^d + \bigl(I - \Phi^{+}\Phi\bigr)\, \tau^{\mathrm{hov}},
       \qquad
       \Phi = \begin{bmatrix} I & \cdots & I \\ \hat r_1 & \cdots & \hat r_{n_Q}\end{bmatrix} .

   ``\Phi`` is :func:`udaan.utils.grasp.grasp_map`.  **The split is only as
   good as the rank of** ``\Phi``: ``\Phi^{+}`` returns the least-squares
   solution whether or not ``W^d`` lies in the range of ``\Phi``, so a
   rank-deficient grasp silently drops the unachievable component of the
   command.  At the default ring the rank is 3/5/6 for ``n_Q = 1/2/\ge 3``, so
   at ``n_Q \le 2`` the payload attitude is *structurally* uncontrollable — a
   property of the mechanism, not a bug.  Both the deficiency and every command
   that falls outside the range are reported: see :attr:`grasp_rank` and
   :attr:`unachievable_wrenches`.

   The second term is M4's
   answer to M3's tension-split policy: the least-norm solution ``\Phi^+ W^d``
   alone makes every cable force parallel and collapses the formation onto one
   line, exactly as M3's zero-internal-force split does.  Biasing the null space
   towards the hover forces ``\tau^{\mathrm{hov}}`` of the model's own ``reset()``
   formation both spreads the team and makes the allocation **exact at hover**:
   since ``\Phi \tau^{\mathrm{hov}} = W^{\mathrm{hov}}``, at ``W^d =
   W^{\mathrm{hov}}`` equation :eq:`m4c-split` returns ``\tau^{\mathrm{hov}}``
   identically, not merely approximately.

3. **Cable loop.**  Per cable, ``q_i`` is driven towards
   ``q_i^d = 𝐓_i^d/\|𝐓_i^d\|`` with the same ``S²``/``TS²`` error convention as
   M2 and M3, where ``𝐓_i^d = -R_L \tau_i`` is the commanded force on the
   *quadrotor* (Newton's third law: ``\tau_i`` acts on the load).

4. **Quadrotor loop.**  The resulting thrust *vector* per quadrotor is returned;
   the model's ``ACCELERATION`` input pipeline feeds it to the per-quadrotor
   :class:`~udaan.control.quadrotor.GeometricAttitudeController`.

Thrust law
----------
Newton for quadrotor ``i``, with ``x_{Q_i} = x_L + R_L r_i - l_i q_i``, is
``u_i = m_i(a_{Q_i} + g e_3) - 𝐓_i``.  The one difference from M3 is that the
cable now hangs off a *moving, rotating* attachment point, so the payload
acceleration that M3 uses is replaced by the attachment acceleration

.. math::

    \mathrm{acc}_i := a_L^d + g e_3
                      + R_L\bigl((R_L^\top R_d \dot\Omega_d) \times r_i\bigr)
                      + R_L\bigl(\Omega_L \times (\Omega_L \times r_i)\bigr),

with ``R_L^\top R_d \dot\Omega_d`` the **reference** angular acceleration from
:attr:`~MultiQuadRigidbodyController.attitude_setpoint`, *transported into the
payload body frame* — ``r_i`` is a body-frame vector, so a reference expressed
in the frame of ``R_d`` has to be rotated before it crosses with ``r_i``.  The
two agree only at ``R_L = R_d``, and the discrepancy scales with
``\|\dot\Omega_d\|\,\|r_i\|`` and the attitude error: at 0.4 rad of yaw error and
``\|\dot\Omega_d\| = 0.37`` rad/s² the raw vector already misdirects 0.03 m/s² of
every quadrotor's commanded acceleration.  It is *not* the
closed-loop
``J_L^{-1}(M_L^d - \Omega_L \times J_L\Omega_L)`` either, which reads like the
same thing; see :meth:`MultiQuadRigidbodyController.desired_wrench` for
what putting the closed-loop value here does to the payload attitude.  After
that the split along and across the cable is M3's verbatim:

.. math::

    u_i^{\parallel} &= (A_i \cdot q_i)\, q_i, \qquad
    A_i := m_i\,\mathrm{acc}_i - 𝐓_i^d + m_i l_i \|\dot q_i\|^2 q_i, \\
    u_i^{\perp} &= -m_i \hat q_i^2\, \mathrm{acc}_i
       + m_i l_i \bigl(k_p^q e_{q_i} + k_d^q e_{\dot q_i}\bigr) .

Substituting the hover condition (``e = 0``, ``q_i = \bar q_i``,
``𝐓_i = \bar{𝐓}_i``, ``\Omega_L = 0``, ``\mathrm{acc}_i = g e_3``) returns
``u_i = m_i g e_3 - \bar{𝐓}_i``, the exact equilibrium thrust of
:mod:`udaan.models.multi_quad_rigidbody` — which is the check that the two
re-derivations agree.

**No cable feedforward yet.**  M3 gets ``\dot q_i^d`` and ``\ddot q_i^d`` from
its tension-split policy, which differentiates the split exactly.  The M4
equivalent now exists — :class:`udaan.flatness.MultiQuadRigidbody` differentiates
:eq:`m4c-split` exactly through :meth:`udaan.flatness.WrenchAllocation.allocate_jet`
— but this cascade does not consume it yet, so the reference derivatives are
dropped rather than guessed.  They are zero for a hover setpoint anyway; wiring
the flat map in is what would unlock them, the same way M3's did.

References:
    K. Sreenath and V. Kumar, "Dynamics, Control and Planning for Cooperative
    Manipulation of Payloads Suspended by Cables from Multiple Quadrotor
    Robots", Robotics: Science and Systems (RSS), 2013 — the point-mass case
    this generalises.
"""

from __future__ import annotations

import numpy as np

from ...control import Gains, PDController
from ...core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_MULTI_PAYLOAD_CABLE_KD,
    DEFAULT_MULTI_PAYLOAD_CABLE_KP,
    DEFAULT_MULTI_PAYLOAD_POS_KD,
    DEFAULT_MULTI_PAYLOAD_POS_KP,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_QUAD_MASS,
    DEFAULT_RIGID_PAYLOAD_ATT_KD,
    DEFAULT_RIGID_PAYLOAD_ATT_KP,
    DEFAULT_RIGID_PAYLOAD_INERTIA,
    DEFAULT_RIGID_PAYLOAD_MASS,
    GRAVITY,
)
from ...utils import hat, vee
from ...utils.grasp import (
    formation_cable_attitudes,
    grasp_map,
    hover_wrench_map,
    ring_attachment_points,
)
from ...utils.logging import get_logger

__all__ = ["MultiQuadRigidbodyController"]

_logger = get_logger(__name__)

#: Below this commanded cable-force magnitude the desired direction is
#: undefined and the hover direction is used instead.
_MIN_CABLE_FORCE = 1e-9

#: Relative size above which the part of a commanded wrench that the grasp
#: cannot produce is reported. ``Φ⁺`` drops it silently otherwise.
_WRENCH_RESIDUAL_TOL = 1e-6

#: A grasp needs six independent wrench directions to hold an SE(3) pose.
_FULL_WRENCH_RANK = 6


class MultiQuadRigidbodyController(PDController):
    """Payload-tracking controller for ``nQ`` quadrotors and one rigid-body load.

    ``compute(t, state)`` returns an ``(nQ, 3)`` stack of thrust *force vectors*
    — one per quadrotor, in the world frame — which is exactly the
    ``InputType.ACCELERATION`` shape that
    :meth:`udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.step`
    expects.

    ``setpoint`` follows the usual protocol ``t -> (pos, vel, acc)`` for the
    payload's *translation*.  The payload's *attitude* reference is a separate
    callable, :attr:`attitude_setpoint`, returning ``(R_d, Ω_d, Ω̇_d)`` and
    defaulting to level and still — the two references are independent for a
    rigid-body load in a way they cannot be for a point mass.

    Args:
        num_quadrotors: ``nQ ≥ 1``.
        masses: scalar (broadcast) or ``(nQ,)`` quadrotor masses.
        payload_mass: ``m_L``.
        payload_inertia: ``J_L``, a ``(3,)`` diagonal or ``(3, 3)`` tensor.
        cable_lengths: scalar (broadcast) or ``(nQ,)`` cable lengths.
        attachment_points: ``(nQ, 3)`` body-frame attachment points; defaults
            to :func:`~udaan.utils.grasp.ring_attachment_points`.
        hover_cable_attitudes: ``(nQ, 3)`` cable directions of the model's
            ``reset()`` formation, used to bias the null space of the grasp
            map. Defaults to the ring at ``formation_radius``.
        formation_radius: radius of that default ring.
        gravity: ``g``.
        kp, kd: payload **position**-loop PD gains, in acceleration units.
        setpoint: forwarded to :class:`~udaan.control.PDController`.
    """

    def __init__(
        self,
        *,
        num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
        masses=DEFAULT_QUAD_MASS,
        payload_mass: float = DEFAULT_RIGID_PAYLOAD_MASS,
        payload_inertia=DEFAULT_RIGID_PAYLOAD_INERTIA,
        cable_lengths=DEFAULT_CABLE_LENGTH,
        attachment_points=None,
        hover_cable_attitudes=None,
        formation_radius: float = DEFAULT_FORMATION_RADIUS,
        gravity: float = GRAVITY,
        **kwargs,
    ):
        super().__init__(**kwargs)

        nQ = int(num_quadrotors)
        if nQ < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {nQ}")
        self._num_quadrotors = nQ

        # ``Controller.__init__`` already bound ``_g``/``_ge3``/``_gravity`` at
        # the package default; rebind all three together so a caller-supplied
        # ``gravity`` cannot leave them disagreeing.
        self._g = float(gravity)
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._gravity = np.array([0.0, 0.0, -self._g])

        #: Commands whose wrench fell outside the range of ``Φ``. See
        #: :attr:`unachievable_wrenches`.
        self._unachievable_wrenches = 0

        # Gains. Position loop in acceleration units (nQ-invariant, see M3);
        # attitude loop in N·m. ``PDController.__init__`` parked any kp/kd in
        # ``self._gains``, which ``compute()`` never reads — route them into the
        # position loop and alias ``_gains`` onto it, so there is exactly one
        # live gain object and tuning through the documented argument reaches
        # the command.
        self._gain_pos = Gains(
            kp=self._gain_vector(kwargs.get("kp"), DEFAULT_MULTI_PAYLOAD_POS_KP, "kp"),
            kd=self._gain_vector(kwargs.get("kd"), DEFAULT_MULTI_PAYLOAD_POS_KD, "kd"),
        )
        self._gains = self._gain_pos
        self._gain_att = Gains(
            kp=DEFAULT_RIGID_PAYLOAD_ATT_KP.copy(), kd=DEFAULT_RIGID_PAYLOAD_ATT_KD.copy()
        )
        self._gain_cable = Gains(
            kp=DEFAULT_MULTI_PAYLOAD_CABLE_KP.copy(), kd=DEFAULT_MULTI_PAYLOAD_CABLE_KD.copy()
        )

        #: Payload attitude reference, ``t -> (R_d, Ω_d, Ω̇_d)``.
        self.attitude_setpoint = lambda t: (np.eye(3), np.zeros(3), np.zeros(3))

        # Whether the hover formation below is ours to rebuild when the model's
        # parameters change. A caller-supplied one is the caller's to maintain.
        self._hover_is_default = hover_cable_attitudes is None
        self._set_parameters(
            masses=masses,
            payload_mass=payload_mass,
            payload_inertia=payload_inertia,
            cable_lengths=cable_lengths,
            attachment_points=attachment_points,
            formation_radius=formation_radius,
            hover_cable_attitudes=hover_cable_attitudes,
        )

    # ─── Parameter plumbing ────────────────────────────────────────

    @staticmethod
    def _gain_vector(value, default: np.ndarray, name: str) -> np.ndarray:
        """A 3-vector gain from a scalar, a 3-sequence, or the default."""
        if value is None:
            return np.array(default, dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(3, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must be a scalar or a 3-vector; got {arr.tolist()}")
        return np.array(arr, dtype=float)

    def _broadcast(self, value, name: str) -> np.ndarray:
        arr = np.array(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_quadrotors, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != self._num_quadrotors:
            raise ValueError(f"{name} must have {self._num_quadrotors} entries, got {arr.shape[0]}")
        # ``np.array`` copies: a caller's array must never be aliased into the
        # controller, the same rule the model follows for its own parameters.
        return arr

    def _set_parameters(
        self,
        *,
        masses,
        payload_mass,
        payload_inertia,
        cable_lengths,
        attachment_points,
        formation_radius,
        hover_cable_attitudes=None,
    ):
        """Validate the physical parameters and rebuild every cached quantity.

        Everything derived from the geometry — ``Φ``, ``Φ⁺``, the null-space
        projector, the hover tensions and the hover bias ``τ^hov`` of
        :eq:`m4c-split` — is cached, so it all has to be rebuilt together.
        :meth:`sync_from_model` is the path that does so after a model's
        parameters change; going through one routine is what keeps a re-synced
        controller identical to a freshly constructed one.

        The hover tensions are solved from the static conditions in least
        squares.  A geometry whose residual is non-zero cannot hold a static
        hover at all; that is reported here rather than left to show up as a
        slowly toppling payload.
        """
        nQ = self._num_quadrotors

        self._masses = self._broadcast(masses, "masses")
        if np.any(self._masses <= 0.0):
            raise ValueError(f"masses must be positive; got {self._masses.tolist()}")
        self._cable_lengths = self._broadcast(cable_lengths, "cable_lengths")
        if np.any(self._cable_lengths <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {self._cable_lengths.tolist()}")

        self._payload_mass = float(payload_mass)
        if self._payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {self._payload_mass}")

        inertia = np.array(payload_inertia, dtype=float)
        if inertia.shape == (3,):
            inertia = np.diag(inertia)
        if inertia.shape != (3, 3):
            raise ValueError(f"payload_inertia must be (3,) or (3, 3); got {inertia.shape}")
        self._payload_inertia = inertia
        self._payload_inertia_inv = np.linalg.inv(inertia)

        self._formation_radius = float(formation_radius)

        if attachment_points is None:
            attachment_points = ring_attachment_points(nQ)
        points = np.array(attachment_points, dtype=float)
        if points.shape != (nQ, 3):
            raise ValueError(f"attachment_points must be ({nQ}, 3); got {points.shape}")
        self._attachment_points = points

        if hover_cable_attitudes is None:
            hover_cable_attitudes = [
                np.asarray(q)
                for q in formation_cable_attitudes(nQ, self._cable_lengths, self._formation_radius)
            ]
        hover_q = np.array(hover_cable_attitudes, dtype=float)
        if hover_q.shape != (nQ, 3):
            raise ValueError(f"hover_cable_attitudes must be ({nQ}, 3); got {hover_q.shape}")
        self._hover_directions = hover_q

        self._grasp = grasp_map(self._attachment_points)
        self._grasp_pinv = np.linalg.pinv(self._grasp)
        self._grasp_null = np.eye(3 * nQ) - self._grasp_pinv @ self._grasp
        self._grasp_rank = int(np.linalg.matrix_rank(self._grasp))
        if self._grasp_rank < _FULL_WRENCH_RANK:
            _logger.warning(
                "grasp map of %d attachment point(s) has rank %d < 6: %d wrench "
                "direction(s) cannot be produced at all, so the payload attitude "
                "is structurally uncontrollable and this cascade will hold "
                "position while the attitude drifts. Full SE(3) authority needs "
                "nQ >= 3 non-collinear attachment points.",
                nQ,
                self._grasp_rank,
                _FULL_WRENCH_RANK - self._grasp_rank,
            )

        wrench_map = hover_wrench_map(hover_q, self._attachment_points)
        target = np.concatenate([-self._payload_mass * self._ge3, np.zeros(3)])
        tensions, *_ = np.linalg.lstsq(wrench_map, target, rcond=None)
        self._hover_residual = float(np.linalg.norm(wrench_map @ tensions - target))
        self._hover_tensions = tensions
        # τ^hov: body-frame force each cable applies *to the load*, which is
        # minus the force it applies to the quadrotor. At R_L = I the two
        # frames coincide, which is the configuration this trim describes.
        self._hover_forces = (-tensions[:, None] * hover_q).reshape(-1)

    def sync_from_model(self, model) -> None:
        """Re-read every physical parameter off ``model``, keeping the tuning.

        The gains, the translation setpoint and :attr:`attitude_setpoint` are
        left alone; everything the geometry determines is rebuilt.  The model
        calls this from its own parameter setters: ``Φ``, ``Φ⁺``, the
        null-space bias and every mass are cached here at construction, so a
        post-construction ``mdl.attachment_points = ...`` would otherwise leave
        this cascade allocating through the *old* grasp — which holds position
        perfectly while tumbling the payload, and is invisible unless the
        attitude is checked.

        Hover cable directions supplied explicitly at construction are kept;
        only the default ring is rebuilt, since only that one was derived from
        the parameters being re-read.

        Raises:
            ValueError: if ``model`` has a different ``num_quadrotors``. The
                cached shapes are all ``nQ``-sized, and a controller for a
                different team size is a different controller.
        """
        if int(model.num_quadrotors) != self._num_quadrotors:
            raise ValueError(
                f"cannot sync a {self._num_quadrotors}-quadrotor controller from a "
                f"{model.num_quadrotors}-quadrotor model; rebuild it with from_model()"
            )
        self._set_parameters(
            masses=model.masses,
            payload_mass=model.payload_mass,
            payload_inertia=model.payload_inertia,
            cable_lengths=model.cable_lengths,
            attachment_points=model.attachment_points,
            formation_radius=model.formation_radius,
            hover_cable_attitudes=None if self._hover_is_default else self._hover_directions,
        )

    @classmethod
    def from_model(cls, model, **kwargs):
        """Build from a
        :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase`-like
        model, so the controller and the plant agree on every parameter."""
        return cls(
            num_quadrotors=model.num_quadrotors,
            masses=model.masses,
            payload_mass=model.payload_mass,
            payload_inertia=model.payload_inertia,
            cable_lengths=model.cable_lengths,
            attachment_points=model.attachment_points,
            formation_radius=model.formation_radius,
            **kwargs,
        )

    @property
    def num_quadrotors(self) -> int:
        return self._num_quadrotors

    @property
    def attachment_points(self) -> np.ndarray:
        """Body-frame attachment points ``r_i``, ``(nQ, 3)``."""
        return self._attachment_points

    @property
    def grasp(self) -> np.ndarray:
        """The ``(6, 3 nQ)`` grasp matrix ``Φ`` this controller allocates with."""
        return self._grasp

    @property
    def grasp_rank(self) -> int:
        """``rank Φ`` — how many of the six wrench directions the grasp spans.

        Below 6 the missing directions cannot be commanded at all and the
        payload attitude is uncontrollable; the default ring gives 3/5/6 at
        ``nQ = 1/2/≥3``. Note that this is the rank over *arbitrary* cable
        force vectors: what the taut cables can apply at a given instant is
        smaller still — see :func:`udaan.utils.grasp.grasp_map`.
        """
        return self._grasp_rank

    @property
    def unachievable_wrenches(self) -> int:
        """Commands whose wrench had a component outside the range of ``Φ``.

        ``Φ⁺`` returns the least-squares split regardless, so that component is
        simply dropped — silently, unless it is counted. A non-zero count on a
        rank-6 grasp means the geometry went degenerate mid-run; on a
        rank-deficient one it is the structural limit showing up. The first
        occurrence is logged as a warning, the rest at debug level.
        """
        return self._unachievable_wrenches

    @property
    def hover_tensions(self) -> np.ndarray:
        """Scalar hover tensions ``t_i`` of the reference formation, ``(nQ,)``."""
        return np.asarray(self._hover_tensions)

    @property
    def hover_residual(self) -> float:
        """Least-squares residual of the static hover conditions.

        Non-zero means the grasp geometry admits no static trim: the team
        cannot simultaneously hold the payload's weight and zero its net
        moment. The controller still runs — the closed loop will simply hold
        the best wrench the geometry allows.
        """
        return self._hover_residual

    @property
    def position_gains(self) -> Gains:
        """Payload position-loop PD gains, in acceleration units. Mutable in place."""
        return self._gain_pos

    @property
    def attitude_gains(self) -> Gains:
        """Payload attitude-loop gains, in N·m. Mutable in place."""
        return self._gain_att

    @property
    def cable_gains(self) -> Gains:
        """Cable-loop (``S²``) PD gains. Mutable in place."""
        return self._gain_cable

    # ─── Control ───────────────────────────────────────────────────

    def desired_wrench(self, t: float, state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(a_L^d + g e_3, M_L^d, R_Lᵀ R_d Ω̇_d)`` — the payload pose loop's output.

        The first entry is in the world frame; the second and third are in the
        payload body frame, as the payload's rotational dynamics are.

        The third entry is the **reference** angular acceleration read off
        :attr:`attitude_setpoint` and *transported into the payload body
        frame*. ``Ω̇_d`` is expressed in the reference frame ``R_d``, while its
        only consumer crosses it with the body-frame ``r_i``; the two frames
        coincide only at ``R_L = R_d``, and the discrepancy grows with
        ``‖Ω̇_d‖ ‖r_i‖`` and the attitude error — 0.03 m/s² per quadrotor at
        0.4 rad of yaw and ``‖Ω̇_d‖ = 0.37`` rad/s², all of it pointing the
        wrong way.
        It is not the closed-loop ``J_L^{-1}(M_L^d - Ω×J_LΩ)`` either.
        It is only used as a feedforward inside ``acc_i``, and routing the
        closed-loop value there would put measured ``Ω_L`` through a gain of
        ``k_Ω |r_i| / J_L`` into every quadrotor's commanded acceleration —
        about 5 m/s² per rad/s for the default slab, comparable to ``g``. That
        makes what reads like a kinematic feedforward into the highest-gain
        feedback path in the cascade, and it destabilises the payload attitude
        outright: measured at ``nQ = 4`` with ``k_Ω = 0.9``, the formation
        diverges within 10 s, and even at ``k_Ω = 0.45`` the attitude settles
        into a 2-8° limit cycle instead of converging. With the reference used
        here, the same gains converge monotonically.
        """
        sp = self.setpoint(t)
        xd, vd, ad = np.asarray(sp[0]), np.asarray(sp[1]), np.asarray(sp[2])

        ex = np.asarray(state.payload_position) - xd
        ev = np.asarray(state.payload_velocity) - vd
        accel_plus_g = ad - self._gain_pos.kp * ex - self._gain_pos.kd * ev + self._ge3

        R_L = np.asarray(state.payload_orientation, dtype=float)
        Omega_L = np.asarray(state.payload_angular_velocity, dtype=float)
        R_d, Omega_d, dOmega_d = (np.asarray(v, dtype=float) for v in self.attitude_setpoint(t))

        # Geometric SO(3) errors (Lee 2010), the same pair the per-quadrotor
        # attitude controller uses.
        err_R = 0.5 * vee(R_d.T @ R_L - R_L.T @ R_d)
        err_Omega = Omega_L - R_L.T @ R_d @ Omega_d
        moment = (
            -self._gain_att.kp * err_R
            - self._gain_att.kd * err_Omega
            + np.cross(Omega_L, self._payload_inertia @ Omega_L)
            - self._payload_inertia
            @ (hat(Omega_L) @ R_L.T @ R_d @ Omega_d - R_L.T @ R_d @ dOmega_d)
        )
        # Body-frame reference angular acceleration: its consumer crosses it
        # with the body-frame r_i.
        return accel_plus_g, moment, R_L.T @ R_d @ dOmega_d

    def allocate(self, accel_plus_g: np.ndarray, moment: np.ndarray, R_L: np.ndarray) -> np.ndarray:
        """Commanded cable forces on the **quadrotors**, ``(nQ, 3)`` world frame.

        Implements :eq:`m4c-split` and negates into the quadrotor-side
        convention: ``𝐓_i = -R_L τ_i``.

        The split is checked before it is returned. ``Φ⁺`` is a least-squares
        inverse, so it answers *every* command, including one the grasp cannot
        produce: the unachievable component ``(I - ΦΦ⁺) W^d`` is dropped
        without a word. That is exactly what happens to the moment command at
        ``nQ ≤ 2``, and the payload then drifts off attitude while position
        tracking still looks perfect. Counted in
        :attr:`unachievable_wrenches`, with the first occurrence logged.
        """
        wrench = np.concatenate([R_L.T @ (self._payload_mass * accel_plus_g), moment])
        tau = self._grasp_pinv @ wrench + self._grasp_null @ self._hover_forces

        dropped = float(np.linalg.norm(self._grasp @ tau - wrench))
        if dropped > _WRENCH_RESIDUAL_TOL * max(1.0, float(np.linalg.norm(wrench))):
            self._unachievable_wrenches += 1
            if self._unachievable_wrenches == 1:
                _logger.warning(
                    "commanded wrench %s lies outside the range of the grasp map "
                    "(rank %d of 6): ‖(I - ΦΦ⁺)W‖ = %.4g is being dropped by the "
                    "least-squares allocation. The payload cannot track that "
                    "component. Further occurrences are logged at debug level; "
                    "see the unachievable_wrenches counter for the total.",
                    np.array2string(wrench, precision=4),
                    self._grasp_rank,
                    dropped,
                )
            else:
                _logger.debug("unachievable wrench component ‖·‖ = %.4g dropped", dropped)

        return np.asarray(-(tau.reshape(self._num_quadrotors, 3) @ R_L.T))

    def compute(self, *args) -> np.ndarray:
        """Per-quadrotor thrust force vectors, shape ``(nQ, 3)``.

        Args:
            args: ``(t, state)`` where ``state`` is a
                :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyState`.
        """
        t = args[0]
        state = args[1]

        accel_plus_g, moment, ref_ang_accel = self.desired_wrench(t, state)
        R_L = np.asarray(state.payload_orientation, dtype=float)
        Omega_L = np.asarray(state.payload_angular_velocity, dtype=float)
        cable_forces = self.allocate(accel_plus_g, moment, R_L)

        out = np.zeros((self._num_quadrotors, 3))
        for i in range(self._num_quadrotors):
            m_i = float(self._masses[i])
            l_i = float(self._cable_lengths[i])
            r_i = self._attachment_points[i]
            force = cable_forces[i]

            # The cable hangs off a rotating attachment point, so the reference
            # acceleration each quadrotor must supply is the attachment's, not
            # the payload centre's. Both extra terms vanish at r_i = 0, which is
            # what makes this reduce to M3's law. ``ref_ang_accel`` is already
            # in the payload body frame — ``desired_wrench`` transports it —
            # which is the frame ``r_i`` lives in.
            acc_i = (
                accel_plus_g
                + R_L @ np.cross(ref_ang_accel, r_i)
                + R_L @ np.cross(Omega_L, np.cross(Omega_L, r_i))
            )

            # A vanished commanded cable force (free fall) leaves the direction
            # undefined; fall back on the hover direction rather than
            # propagating a nan.
            norm_force = float(np.linalg.norm(force))
            qd = self._hover_directions[i] if norm_force < _MIN_CABLE_FORCE else force / norm_force

            q = np.asarray(state.cable_attitudes[i], dtype=float)
            dq = state.dq(i)
            q_hat = hat(q)

            # S²/TS² errors — same convention as M2/M3. The reference
            # derivatives q̇^d, q̈^d are zero: see "No cable feedforward yet".
            err_q = q_hat @ q_hat @ qd
            err_dq = dq

            u_perp = (m_i * l_i) * (self._gain_cable.kp * err_q + self._gain_cable.kd * err_dq)
            u_perp -= m_i * (q_hat @ q_hat @ acc_i)

            # Parallel channel: cable tension, weight share and centripetal term.
            A_i = m_i * acc_i - force + m_i * l_i * np.dot(dq, dq) * q
            out[i] = np.dot(A_i, q) * q + u_perp

        return out
