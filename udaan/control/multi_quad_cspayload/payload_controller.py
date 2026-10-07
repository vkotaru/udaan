"""Payload-tracking cascade for ``nQ`` quadrotors sharing one cable-suspended
point-mass payload (M3).

The controller is the generic-``nQ`` sibling of
:class:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`, and it is
built as a four-stage cascade:

1. **Payload loop.**  A PD on the payload position/velocity error, added to the
   reference acceleration, fixes the desired payload acceleration ``a_L^d`` and
   hence the desired *net* cable force

   .. math::

       a_L^d = \\ddot x_L^d - k_p^L e_{x_L} - k_d^L e_{v_L},
       \\qquad
       F_{\\mathrm{net}}^d = -m_L\\,(a_L^d + g e_3).

   The gains act in **acceleration** units, not the force units M2 uses.  A
   force-space PD would have to be normalised by the total system mass
   ``m_L + Σ_i m_i``, which grows with ``nQ``, so a gain pair that is well
   damped at ``nQ = 1`` goes sluggish at ``nQ = 4`` — and empirically unstable
   the other way round, once the payload loop is pushed fast enough to close
   on the cable loop.  Commanding acceleration keeps the payload response
   ``nQ``-invariant, which is what "generic in ``nQ``" has to mean here.
   Measured on a 1 m lateral step: 10-90% rise 0.80-0.87 s and 4.5-7.0%
   overshoot across ``nQ = 2…6``.  ``nQ = 1`` is a real outlier (0.59 s,
   17.8%) — one vertical cable is a different geometry, not the one-vehicle
   case of the ring — so the invariance claim is for ``nQ >= 2``.

2. **Distribution.**  ``F_net^d`` — together with its reference derivatives
   ``Ḟ_net = -m_L x_L^{(3)d}`` and ``F̈_net = -m_L x_L^{(4)d}`` — is handed to a
   :class:`~udaan.flatness.TensionSplitPolicy`, the very abstraction
   :mod:`udaan.flatness.multi_quad_cspayload` introduced for the flat-to-state
   map.  Newton on the payload only constrains ``Σ_i 𝐓_i``; the policy picks the
   remaining ``3(nQ − 1)`` internal degrees of freedom, guarantees a
   non-degenerate formation and — crucially here — supplies the *derivatives*
   of the split, which is what the cable loop needs for its feedforward.
   Defaulting to :class:`~udaan.flatness.RingTensionSplit` makes the
   controller's hover formation coincide with the model's ``reset()``
   configuration and with the flat reference.

3. **Cable loop.**  Per cable, ``q_i`` is driven towards
   ``q_i^d = 𝐓_i^d / ‖𝐓_i^d‖`` with the same ``S²``/``TS²`` error convention as
   M2 (:mod:`udaan.manif.S2`)::

       e_{q_i}  = q̂_i² q_i^d
       e_{q̇_i} = q̇_i - (q_i^d × q̇_i^d) × q_i

4. **Quadrotor loop.**  The resulting thrust *vector* per quadrotor is returned;
   the model's ``ACCELERATION`` input pipeline feeds it to the per-quadrotor
   :class:`~udaan.control.quadrotor.GeometricAttitudeController`.

Thrust law
----------
For quadrotor ``i``, Newton with ``x_{Q_i} = x_L - l_i q_i`` gives

.. math::

    u_i = m_i\\,(a_L + g e_3) - m_i l_i \\ddot q_i - \\mathbf{T}_i .

Splitting along and across the cable, and using ``q_i \\cdot \\ddot q_i =
-\\|\\dot q_i\\|^2``:

.. math::

    u_i^{\\parallel} = (A_i \\cdot q_i)\\, q_i, \\qquad
    A_i := m_i (a_L^d + g e_3) - \\mathbf{T}_i^d + m_i l_i \\|\\dot q_i\\|^2 q_i

.. math::

    u_i^{\\perp} = -m_i \\hat q_i^2 (a_L^d + g e_3)
       + m_i l_i \\bigl(k_p^q e_{q_i} + k_d^q e_{\\dot q_i}\\bigr)
       - m_i l_i (q_i \\cdot \\omega_i^d)(q_i \\times \\dot q_i)
       - m_i l_i\\, (\\dot\\omega_i^d \\times q_i)

with ``ω_i^d = q_i^d × q̇_i^d`` and ``ω̇_i^d = q_i^d × q̈_i^d``.

**The ``-m_i q̂_i² (a_L^d + g e_3)`` term is the one thing M2 does not have**,
and it is not an embellishment.  In the single-quadrotor *plant* the payload
acceleration is always parallel to the cable — the cable is the only thing
pulling on the load — so ``q̂²(a_L + g e_3)`` vanishes at the reference and M2
is free to omit it.  Note that the term is evaluated here at the **measured**
``q_i``, not at ``q_i^d``, so at ``nQ = 1`` it vanishes only where the cable
attitude error is zero: with the cable tilted 10° off its reference it is
already ~12% of ``‖u_i‖``.  M2 and M3 are therefore genuinely different laws at
``nQ = 1``, M3 being the more complete one; they are not two spellings of the
same expression.

With ``nQ > 1`` the cables are tilted off vertical, ``a_L + g e_3`` has a
component across every one of them, and dropping the term leaves each
quadrotor short of exactly the lateral force its own weight requires.  The
missing force points *inwards* (its horizontal part is
``-m_i g (Δz·r/l²)(cosθ, sinθ)`` on a ring of radius ``r`` whose quadrotors
sit at height ``Δz = sqrt(l² - r²)``), so without it each quadrotor settles
**further out**, not inwards: over a 10 s hover the formation radius drifts
0.5 → 0.668-0.673 m for ``nQ = 2,3,4``, with 2.6-4.7 cm of payload position
error.  Substituting the hover condition (``e = 0``, ``q_i = q̄_i``,
``𝐓_i = 𝐓̄_i``) into the two expressions above returns ``u_i = m_i g e_3 -
𝐓̄_i``, the exact equilibrium thrust, only because that term is present.

**Deliberate deviation from Sreenath–Kumar.**  Their ``u_i^⊥`` carries
``-q̂_i² A_i`` in full; the implementation above keeps only the
``-m_i q̂_i²(a_L^d + g e_3)`` half of it.  The dropped half is
``q̂_i² 𝐓_i^d = T_i^d e_{q_i}`` (the ``m_i l_i ‖q̇_i‖² q_i`` part of ``A_i``
is annihilated by ``q̂_i²``), i.e. an effective reduction of the cable
stiffness by ``T_i^d/(m_i l_i)`` — about 0.84 s⁻² against ``k_p^q = 30``, some
3%, and vanishing at zero cable-attitude error.  It is not an oversight: keep
the algebra as written.

References:
    K. Sreenath and V. Kumar, "Dynamics, Control and Planning for Cooperative
    Manipulation of Payloads Suspended by Cables from Multiple Quadrotor
    Robots", Robotics: Science and Systems (RSS), 2013.
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
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ...flatness.cable import tension_direction_jet
from ...flatness.multi_quad_cspayload import RingTensionSplit, TensionSplitPolicy
from ...utils import hat

__all__ = ["MultiQuadCsPayloadController"]


class MultiQuadCsPayloadController(PDController):
    """Payload-tracking controller for ``nQ`` quadrotors and one point-mass load.

    ``compute(t, state)`` returns an ``(nQ, 3)`` stack of thrust *force vectors*
    — one per quadrotor, in the world frame — which is exactly the
    ``InputType.ACCELERATION`` shape that
    :meth:`udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.step`
    expects.

    The ``setpoint`` follows the usual protocol ``t -> (pos, vel, acc)``, and
    like M2 it *may* return two further entries ``(jerk, snap)``.  When it does,
    the cable loop picks up its feedforward for free: ``Ḟ_net`` and ``F̈_net``
    become non-zero, the tension-split policy differentiates them exactly, and
    ``q̇_i^d``/``q̈_i^d`` stop being zero.  Binding a trajectory's ``get_full``
    is therefore enough to switch that on — no controller change required.

    Args:
        num_quadrotors: ``nQ ≥ 1``.
        masses: scalar (broadcast) or ``(nQ,)`` quadrotor masses.
        payload_mass: ``m_L``.
        cable_lengths: scalar (broadcast) or ``(nQ,)`` cable lengths.
        formation_radius: radius of the default :class:`RingTensionSplit`.
        gravity: ``g``.
        tension_policy: an explicit
            :class:`~udaan.flatness.TensionSplitPolicy`; defaults to a
            :class:`~udaan.flatness.RingTensionSplit` built from the arguments
            above, so the controller's hover formation is the model's own.
        kp, kd: **payload-loop** PD gains — scalar or 3-vector — replacing
            ``DEFAULT_MULTI_PAYLOAD_POS_KP``/``_KD``.  They are in
            *acceleration* units (see the module docstring), not the force
            units M2's payload PD uses, so gains ported from
            :class:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`
            need dividing by the mass they were normalised against.  The cable
            loop is a separate PD and keeps ``DEFAULT_MULTI_PAYLOAD_CABLE_*``;
            tune it through :attr:`cable_gains`.
        setpoint: forwarded to :class:`~udaan.control.PDController`.
    """

    def __init__(
        self,
        *,
        num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
        masses=DEFAULT_QUAD_MASS,
        payload_mass: float = DEFAULT_PAYLOAD_MASS,
        cable_lengths=DEFAULT_CABLE_LENGTH,
        formation_radius: float = DEFAULT_FORMATION_RADIUS,
        gravity: float = GRAVITY,
        tension_policy: TensionSplitPolicy | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        nQ = int(num_quadrotors)
        if nQ < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {nQ}")
        self._num_quadrotors = nQ

        self._masses = self._broadcast(masses, "masses")
        if np.any(self._masses <= 0.0):
            raise ValueError(f"masses must be positive; got {self._masses.tolist()}")
        self._cable_lengths = self._broadcast(cable_lengths, "cable_lengths")
        if np.any(self._cable_lengths <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {self._cable_lengths.tolist()}")

        self._payload_mass = float(payload_mass)
        if self._payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {self._payload_mass}")
        # ``Controller.__init__`` already bound ``_g``/``_ge3``/``_gravity`` at
        # the package default; rebind all three together so a caller-supplied
        # ``gravity`` cannot leave them disagreeing.
        self._g = float(gravity)
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._gravity = np.array([0.0, 0.0, -self._g])
        self._formation_radius = float(formation_radius)

        # Gains: acceleration units throughout, so the closed-loop bandwidths
        # do not move with nQ or with the per-quadrotor mass (see the module
        # docstring and ``udaan.core.defaults``).
        #
        # ``PDController.__init__`` has already parked any ``kp``/``kd`` in
        # ``self._gains``, which ``compute()`` never reads. Route them into the
        # payload loop instead and alias ``_gains`` onto it, so there is exactly
        # one live gain object and tuning through the documented argument
        # actually reaches the command.
        self._gain_pos = Gains(
            kp=self._gain_vector(kwargs.get("kp"), DEFAULT_MULTI_PAYLOAD_POS_KP, "kp"),
            kd=self._gain_vector(kwargs.get("kd"), DEFAULT_MULTI_PAYLOAD_POS_KD, "kd"),
        )
        self._gains = self._gain_pos
        self._gain_cable = Gains(
            kp=DEFAULT_MULTI_PAYLOAD_CABLE_KP.copy(), kd=DEFAULT_MULTI_PAYLOAD_CABLE_KD.copy()
        )

        # Whether the policy below is ours to rebuild when the model's
        # parameters change. A caller-supplied one is the caller's to maintain.
        self._policy_is_default = tension_policy is None
        if tension_policy is None:
            tension_policy = self._default_tension_policy()
        if tension_policy.num_quadrotors != nQ:
            raise ValueError(
                f"tension_policy is built for {tension_policy.num_quadrotors} "
                f"quadrotors, but this controller has {nQ}"
            )
        self._tension_policy = tension_policy

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
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_quadrotors, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != self._num_quadrotors:
            raise ValueError(f"{name} must have {self._num_quadrotors} entries, got {arr.shape[0]}")
        return arr

    def _default_tension_policy(self) -> TensionSplitPolicy:
        """The ring split built from this controller's current parameters."""
        return RingTensionSplit(
            num_quadrotors=self._num_quadrotors,
            payload_mass=self._payload_mass,
            masses=self._masses,
            cable_lengths=self._cable_lengths,
            radius=self._formation_radius,
            gravity=self._g,
        )

    def sync_from_model(self, model) -> None:
        """Re-read every physical parameter off ``model``, keeping the tuning.

        The gains and the setpoint are left alone; the masses, cable lengths
        and formation radius are re-read, and the default tension-split policy
        is rebuilt from them. The model calls this from its own parameter
        setters: every one of those quantities is cached here at construction,
        so ``mdl.payload_mass = 3.0`` after the fact would otherwise leave this
        cascade splitting the *old* load's weight — worth 1.6 m of steady-state
        payload error, with no other symptom.

        An explicitly supplied :attr:`tension_policy` is **not** rebuilt: it
        was constructed by the caller, with parameters that are the caller's to
        keep current.

        Raises:
            ValueError: if ``model`` has a different ``num_quadrotors``.
        """
        if int(model.num_quadrotors) != self._num_quadrotors:
            raise ValueError(
                f"cannot sync a {self._num_quadrotors}-quadrotor controller from a "
                f"{model.num_quadrotors}-quadrotor model; rebuild it with from_model()"
            )
        masses = self._broadcast(model.masses, "masses")
        if np.any(masses <= 0.0):
            raise ValueError(f"masses must be positive; got {masses.tolist()}")
        cable_lengths = self._broadcast(model.cable_lengths, "cable_lengths")
        if np.any(cable_lengths <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {cable_lengths.tolist()}")
        payload_mass = float(model.payload_mass)
        if payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {payload_mass}")

        self._masses = masses
        self._cable_lengths = cable_lengths
        self._payload_mass = payload_mass
        self._formation_radius = float(model.formation_radius)
        if self._policy_is_default:
            self._tension_policy = self._default_tension_policy()

    @classmethod
    def from_model(cls, model, tension_policy: TensionSplitPolicy | None = None, **kwargs):
        """Build from a
        :class:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`-like
        model, so the controller and the plant agree on every parameter."""
        return cls(
            num_quadrotors=model.num_quadrotors,
            masses=model.masses,
            payload_mass=model.payload_mass,
            cable_lengths=model.cable_lengths,
            formation_radius=model.formation_radius,
            tension_policy=tension_policy,
            **kwargs,
        )

    @property
    def num_quadrotors(self) -> int:
        return self._num_quadrotors

    @property
    def tension_policy(self) -> TensionSplitPolicy:
        """The :class:`~udaan.flatness.TensionSplitPolicy` that distributes
        ``F_net`` across the cables."""
        return self._tension_policy

    @tension_policy.setter
    def tension_policy(self, policy: TensionSplitPolicy):
        if policy.num_quadrotors != self._num_quadrotors:
            raise ValueError(
                f"tension_policy is built for {policy.num_quadrotors} quadrotors, "
                f"but this controller has {self._num_quadrotors}"
            )
        self._tension_policy = policy
        # An assigned policy is the caller's; :meth:`sync_from_model` leaves it.
        self._policy_is_default = False

    @property
    def position_gains(self) -> Gains:
        """Payload-loop PD gains, in acceleration units. Mutable in place."""
        return self._gain_pos

    @property
    def cable_gains(self) -> Gains:
        """Cable-loop (``S²``) PD gains. Mutable in place."""
        return self._gain_cable

    @property
    def total_mass(self) -> float:
        """``M_tot = m_L + Σ_i m_i`` — the mass the formation moves as a whole."""
        return self._payload_mass + float(np.sum(self._masses))

    # ─── Control ───────────────────────────────────────────────────

    def desired_net_force(self, t: float, payload_position, payload_velocity) -> list[np.ndarray]:
        """``[F_net^d, Ḟ_net, F̈_net]`` — the payload loop's output.

        The zeroth entry carries the PD correction; the higher entries are pure
        reference feedforward (``-m_L x_L^{(k+2)d}``) and are zero unless the
        setpoint supplies jerk and snap.
        """
        sp = self.setpoint(t)
        xd, vd, ad = np.asarray(sp[0]), np.asarray(sp[1]), np.asarray(sp[2])
        jerk = np.asarray(sp[3]) if len(sp) > 3 else np.zeros(3)
        snap = np.asarray(sp[4]) if len(sp) > 4 else np.zeros(3)

        ex = np.asarray(payload_position) - xd
        ev = np.asarray(payload_velocity) - vd

        accel_cmd = ad - self._gain_pos.kp * ex - self._gain_pos.kd * ev
        mL = self._payload_mass
        return [
            -mL * (accel_cmd + self._ge3),
            -mL * jerk,
            -mL * snap,
        ]

    def compute(self, *args) -> np.ndarray:
        """Per-quadrotor thrust force vectors, shape ``(nQ, 3)``.

        Args:
            args: ``(t, state)`` where ``state`` is a
                :class:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadState`.
        """
        t = args[0]
        state = args[1]

        net_force = self.desired_net_force(t, state.payload_position, state.payload_velocity)
        # The one distribution mechanism: the same policy the flat-to-state map
        # uses, so the closed loop and the feedforward reference agree on which
        # cable carries what.
        cable_forces = self._tension_policy.split(net_force)
        hover_dirs = np.asarray(self._tension_policy.hover_directions, dtype=float)

        # a_L^d + g e3, needed whole (not just along the cable) for the lateral
        # feedforward that M2 can omit and this system cannot.
        accel_plus_g = -net_force[0] / self._payload_mass

        out = np.zeros((self._num_quadrotors, 3))
        for i in range(self._num_quadrotors):
            m_i = float(self._masses[i])
            l_i = float(self._cable_lengths[i])
            force = cable_forces[i]

            try:
                _, q_jet = tension_direction_jet(force)
                qd, dqd, d2qd = q_jet[0], q_jet[1], q_jet[2]
            except ValueError:
                # Commanded cable force vanished (free fall): the direction is
                # undefined, so fall back on the policy's hover direction
                # rather than propagating a nan.
                qd, dqd, d2qd = hover_dirs[i], np.zeros(3), np.zeros(3)

            q = np.asarray(state.cable_attitudes[i], dtype=float)
            dq = state.dq(i)
            q_hat = hat(q)

            # S²/TS² errors — same convention as M2's payload controller.
            err_q = q_hat @ q_hat @ qd
            err_dq = dq - np.cross(np.cross(qd, dqd), q)

            # Perpendicular channel: PD, cable-rate feedforward, and the lateral
            # share of (a_L^d + g e3) that holds each quadrotor on its station
            # (drop it and the ring drifts *outwards* — see the module
            # docstring).  Only the -m_i q̂² (a_L^d + g e3) half of
            # Sreenath-Kumar's -q̂² A_i is kept; the deliberately dropped half
            # is a ~3% cable-stiffness reduction, also documented up there.
            u_perp = (m_i * l_i) * (self._gain_cable.kp * err_q + self._gain_cable.kd * err_dq)
            u_perp -= (m_i * l_i) * np.dot(q, np.cross(qd, dqd)) * np.cross(q, dq)
            u_perp -= (m_i * l_i) * np.cross(np.cross(qd, d2qd), q)
            u_perp -= m_i * (q_hat @ q_hat @ accel_plus_g)

            # Parallel channel: cable tension, weight share and centripetal term.
            A_i = m_i * accel_plus_g - force[0] + m_i * l_i * np.dot(dq, dq) * q
            out[i] = np.dot(A_i, q) * q + u_perp

        return out
