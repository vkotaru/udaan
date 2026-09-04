r"""Payload-tracking LQR for one quadrotor carrying a load on an ``n``-link
flexible cable (M5).

**This is the first controller in this package that is not a geometric
cascade, and the departure is structural rather than stylistic.**  M2, M3 and
M4 all close a payload loop, hand its output to a *cable* loop, and hand that
to an attitude loop.  What makes that work is that every cable in those systems
joins the payload directly to a vehicle: quadrotor ``i`` pushes on cable ``i``
and on nothing else, so ``q_i^d`` is a quantity some actuator can be told to
chase.  Here the cable is a **serial chain**.  The airframe touches link ``0``
only; links ``1 … n-1`` move solely through the chain, and the whole system has
four inputs — one thrust and three body moments — for ``12 + 4n`` states.
There is no ``q_i^d`` to command per link, so the *geometric* cascade M2-M4
use — invert one loop, hand its output to the next as a commanded direction —
has nothing to invert.  Inverting the chain a rung at a time is possible in
principle by backstepping — it is how Goodarzi, Lee and Lee close the
``n``-link loop in the paper this system's dynamics come from — at the price
of differentiating the whole chain once per rung.  The published law for
*this* system takes the other route, a
linear-quadratic regulator on the *variational* linearisation, and that is
what is implemented here.

.. math::

    (f, M) = (f_d, M_d) + K\,s ,
    \qquad
    s = \bigl[e_{x_Q},\; e_{v_Q},\; \eta,\; \delta\Omega,\;
              \{\xi_i\},\; \{\delta\omega_i\}\bigr] ,

with ``(f_d, M_d)`` and the whole reference state supplied by the flat map
:class:`udaan.flatness.QuadrotorFlexCable`, or — when no trajectory is bound —
by the hanging equilibrium at the setpoint (:meth:`equilibrium_reference`).

Error state
-----------
Six blocks, in the conventions M2/M3/M4 already use — **not** the reference
MATLAB's, where they differ:

.. math::

    e_{x_Q} &= x_Q - x_Q^d, \qquad e_{v_Q} = v_Q - v_Q^d, \\
    \eta &= \tfrac12 \operatorname{vee}(R_d^\top R - R^\top R_d), \qquad
    \delta\Omega = \Omega - R^\top R_d\,\Omega_d, \\
    \xi_i &= q_i^d \times q_i, \qquad
    \delta\omega_i = \omega_i + q_i \times (q_i \times \omega_i^d) .

Note that the *translational* block is the *quadrotor's*, not the payload's,
which is what the linearisation below is written in (the ``x_Q`` row of the
equations of motion is the one that carries the thrust).  The payload is
tracked through the chain: ``x_Q^d = x_L^d - \sum_i \ell_i q_i^d``, so driving
``e_{x_Q} \to 0`` and every ``\xi_i \to 0`` puts the payload at ``x_L^d``.
Regulating the payload position *directly* would leave the chain's shape
unconstrained, which for a flexible cable is most of the state.

Two decisions, and why
----------------------

**1. LQR, not a geometric cascade.**  See above: a cascade needs one
independently commandable body per cable, and a chain does not have them.  The
reference (``phd_research/quadrotor-flexible-cables-payload/Control/sQnlink/
sQ_nlink_control.m``) uses a **finite-horizon** LQR: it integrates the Riccati
equation backwards over ``[10, 0]`` along one specific trajectory, caches
``P(t)`` into a ``.mat`` file, and interpolates it at run time.  This
implementation solves the **infinite-horizon** problem instead, once, about the
hanging equilibrium, with :func:`scipy.linalg.solve_continuous_are`.  Three
reasons:

* *Self-contained and causal.*  The backward sweep needs the entire reference
  trajectory in advance and produces a gain schedule tied to one horizon and one
  manoeuvre; the cached ``.mat`` in the reference is the visible symptom.  A
  controller in this package has to work against a ``setpoint`` callable it has
  never seen.
* *The linearisation moves little on gentle references — and not at all
  little on aggressive ones.*  The flat map refuses any reference on which a
  link leaves the hanging branch (:mod:`udaan.flatness.chain`), so every
  trajectory this controller can be fed forward on keeps every ``q_i^d``
  within 90° of ``-e_3``.  That bound alone is nowhere near tight enough to
  justify a constant gain, and it should not be quoted as if it were.
  :meth:`linearize` is written in full reference-dependent form precisely so
  the real number can be *measured*, and
  ``tests/test_quadrotor_flexcable_control.py::TestLinearization`` measures it
  at ``n = 3`` across three ``CrazyTrajectory`` amplitudes, all of which the
  map accepts on every sample:

  ======================  ===============  =================================
  reference               max link tilt    ``\|A(ref) - A(hov)\|/\|A(hov)\|``
  ======================  ===============  =================================
  gentle Lissajous         3.2°            5.7 %
  ax 0.6, f1 0.4          21.3°            37 %
  ax 1.0, f1 0.5          45.4°            93 %
  ======================  ===============  =================================

  So the constant gain is justified *for gentle references*, and degrades as
  the reference gets aggressive.  It is a design choice with a stated domain,
  not a theorem.
* *The time variation the finite-horizon solve captures is feedforward, and the
  flat map already supplies it exactly.*  ``(f_d, M_d)`` and the full reference
  state come from :class:`udaan.flatness.QuadrotorFlexCable`; what is left for
  the gain to do is regulate a small error about it.

Gain scheduling is two lines away for a caller who wants it — ``A, B =
ctrl.linearize(ref, inputs)`` then :meth:`set_gain_from` — and deliberately not
done automatically: at ``n = 3`` a Riccati solve is a ``24 x 24`` eigenproblem
per control step, and on a gentle reference the measured benefit is nil.  On an
aggressive one it is the right thing to reach for, with the caveat in
:meth:`set_gain_from` that the reduction it solves on is then approximate.

**2. What is and is not controllable.**  Two separate statements, both
measured, both tested:

*Structurally.*  The embedded error state has ``12 + 6n`` entries but the
system has only ``12 + 4n`` degrees of freedom.  Each ``\xi_i`` is a cross
product with ``q_i^d`` and each ``\delta\omega_i`` is tangent to ``q_i``, so
``2n`` of those coordinates — the components along ``q_i^d`` — are the ``S^2``
gauge, exactly as ``\dot\omega_i \cdot q_i`` is in the dynamics (see "Why the
diagonal block is ``M_{ii} I``" in :mod:`udaan.models.quadrotor_flexcable.base`).
No input reaches them — ``B``'s rows there vanish identically, at every
reference — so handing the embedded pair to
:func:`~scipy.linalg.solve_continuous_are` asks it to stabilise ``2n``
integrators it cannot touch, and it fails.  That is why the Riccati solve here
runs on the tangent-reduced ``(12 + 4n)`` pair and the gain is pushed back out
through the same projection.  The reference does not hit this because a
finite-horizon Riccati sweep needs no stabilisability at all.

Whether the reduction is *exact* is a second question, and the answer depends
on the reference.  Invariance needs ``\dot\xi_i \cdot q_i^d = 0``, and from
``\dot\xi_i = q_i^d (q_i^d)^\top \hat\omega_i^d \xi_i +
(I - q_i^d (q_i^d)^\top)\,\delta\omega_i`` the second term drops but the first
does **not**: dotting it with ``q_i^d`` leaves
``\xi_i \cdot (q_i^d \times \omega_i^d)``, which is generically nonzero.  At
the hanging equilibrium — where the shipped gain is solved — ``\omega_i^d =
0`` and the residual ``\|(I - WW^\top) A W\| / \|A\|`` is exactly ``0``.  Off
it the residual is small but real: ``2.2e-3 / 3.0e-4 / 1.1e-4`` at
``n = 1/2/3`` over a gentle Lissajous.  So the shipped gain rests on an exact
reduction; :meth:`set_gain_from` at a moving reference does not, and says so.

*Practically.*  The reduced pair is controllable at every ``n`` tested, but the
margin collapses with the link count.  The smallest PBH singular value
``\sigma_{\min}[\,A - \lambda I \;\; B\,]`` over the open-loop spectrum runs

======  =======  ==========================  =========================
``n``   PBH      slowest closed-loop pole    damping of that mode
======  =======  ==========================  =========================
1       0.62     -1.00                       (real)
2       0.085    -0.64 ± 15.1j               4.2 %
3       0.026    -0.34 ± 26.7j               1.3 %
5       0.008    -0.17 ± 48.3j               0.4 %
======  =======  ==========================  =========================

and the mode that ends up slowest is *not* the payload swing — it is the
chain's **highest** internal bending mode, at exactly the frequency
:meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.natural_frequencies`
returns as its maximum.

*On the defaults*, that mode is barely damped, and it is worth being precise
about why, because the obvious experiment misleads.  Scaling the **whole** of
``Q`` by 1000 moves the ``n = 3`` figure from -0.339 to -0.356 — 5 % for a 60x
change in gain — and it is tempting to conclude that no weighting reaches the
mode.  That conclusion does not follow from that experiment: for an LQR,
scaling ``Q`` uniformly is the same as scaling ``R^{-1}``, so it slides along
the cheap-control locus, where the finite closed-loop poles converge to the
plant's stable modes and their mirror images.  A uniform scale is the one
perturbation guaranteed to say nothing.

*Targeted* weighting does reach it.  Raising ``q_link_rate`` alone by 50x, at
3.3x the gain norm:

======  ===========================  ===========================
``n``   top bending mode, default    ``q_link_rate`` x 50
======  ===========================  ===========================
2       -0.64 ± 15.1j  (4.2 %)       -4.02 ± 15.4j  (25.3 %)
3       -0.34 ± 26.7j  (1.3 %)       -2.19 ± 26.8j  (8.2 %)
======  ===========================  ===========================

and the slowest pole overall at ``n = 3`` moves from -0.339 to -0.514.  In a
nonlinear check — a ``n = 3`` chain station-keeping, every link kicked to
±2 rad/s — the bending residual is flat over 10 s on the defaults and falls by
three orders of magnitude with the same reweighting.

**So the honest claim is a trade-off, not a limit.**  The chain's top mode is
reachable by weighting its rates, and the price is tracking: on the gentle
Lissajous the tail RMS payload error goes from 0.00056 m to 0.0138 m at
``n = 3``, a factor of 25.  The shipped defaults buy tracking; a caller who
cares more about a quiet cable than about 14 mm of payload error should raise
``q_link_rate``.  What *is* structural is the authority — the PBH column
above — which no weighting improves.

That mode is also the one the ZOH-Euler integrator is conditionally stable
against — the same frequency drives
:meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.stability_growth_rate`
— so at the link counts where the analytical model is usable at all, it is
lightly excited and stays small.  Both quantities are exposed
(:attr:`controllability_margin`, :attr:`closed_loop_poles`) rather than left to
be discovered.

Errata in the reference
-----------------------
Three, found while re-deriving:

* ``get_linear_error_dynamics`` writes ``hat(dwd(:,i)) - norm(wd(:,i))^2``
  (``sQ_nlink_control.m:452``) and ``hat(dwd(:,j)) - norm(wd(:,j))^2``
  (``:465``).  In MATLAB a scalar subtracted from a matrix is subtracted from
  **every entry**, so this is
  ``\hat{\dot\omega} - \|\omega\|^2 \mathbf{1}\mathbf{1}^\top`` rather than
  ``\hat{\dot\omega} - \|\omega\|^2 I``.  It is a genuine defect, invisible at
  hover (where ``\omega_d = 0``) and quadratic in the reference link rate
  elsewhere.  :meth:`linearize` uses the identity at both sites, and
  ``tests/test_quadrotor_flexcable_control.py::TestNonHoverLinearization``
  pins it at a reference where ``\omega_d \ne 0`` — the only place it is
  visible at all.
* ``get_linear_control`` builds ``Q1 = blkdiag(0.5*eye(6), eye(3),
  0.75*eye(9))`` (``:364``) — an ``18 x 18`` matrix, i.e. hard-wired to
  ``n = 1`` — while ``odefun_riccati`` (``:512``), which is the one that
  actually solved for ``P``, uses
  ``blkdiag(0.5*eye(6), 0.75*eye(6), eye(3n), 0.75*eye(3n))``.  The two
  disagree; the first is dead code (``Q1`` is never read in that function) but
  it is the one a reader meets first.  The weights carried into
  :mod:`udaan.core.defaults` are ``odefun_riccati``'s.
* ``Q2 = 0.2 I_4`` (``:365`` and ``:514``) weights the thrust and the three
  moments equally, though their input matrices differ by
  ``\|J^{-1}\| M_{00} = 522`` on the default airframe.  Harmless under the
  reference's adaptive ``ode15s``, which re-evaluates the control
  continuously; on udaan's 200 Hz zero-order hold the resulting -842 rad/s
  pole sits at 121° of hold phase lag and the loop never settles.  See
  :data:`~udaan.core.defaults.DEFAULT_FLEXCABLE_LQR_R_MOMENT` for the measured
  cost and for what weighting the moments at 20 costs in return.

References:
    P. Kotaru, G. Wu, K. Sreenath, "Differential-Flatness and Control of
    Quadrotor(s) with a Payload Suspended through Flexible Cable(s)", Indian
    Control Conference, 2018.

    F. A. Goodarzi, D. Lee, T. Lee, "Geometric Control of a Quadrotor UAV
    Transporting a Payload Connected via Flexible Cable", American Control
    Conference, 2014 — the dynamics.
"""

from __future__ import annotations

import numpy as np

from ...control import PDController
from ...core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_MASS,
    DEFAULT_FLEXCABLE_LQR_Q_ANGULAR_RATE,
    DEFAULT_FLEXCABLE_LQR_Q_ATTITUDE,
    DEFAULT_FLEXCABLE_LQR_Q_LINK,
    DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE,
    DEFAULT_FLEXCABLE_LQR_Q_POSITION,
    DEFAULT_FLEXCABLE_LQR_Q_VELOCITY,
    DEFAULT_FLEXCABLE_LQR_R_MOMENT,
    DEFAULT_FLEXCABLE_LQR_R_THRUST,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ...flatness.quadrotor_flexcable import (
    QuadrotorFlexCable,
    QuadrotorFlexCableInputs,
    QuadrotorFlexCableRefState,
)
from ...utils import hat, vee

__all__ = ["QuadrotorFlexCableController"]

_E3 = np.array([0.0, 0.0, 1.0])

#: Cosine below which a link direction is treated as too close to the tangent
#: basis' seed axis for that seed to be used.  Purely a numerical guard on
#: :func:`_tangent_basis`; any value comfortably inside ``(0, 1)`` works.
_TANGENT_SEED_CUTOFF = 0.9


def _tangent_basis(q: np.ndarray) -> np.ndarray:
    """A ``(3, 2)`` orthonormal basis of ``q``'s tangent plane on ``S^2``.

    Only the *span* matters — it is used to project the gauge directions out of
    the Riccati solve — so any consistent choice does.
    """
    seed = (
        np.array([1.0, 0.0, 0.0]) if abs(q[0]) < _TANGENT_SEED_CUTOFF else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(q, seed)
    u = u / np.linalg.norm(u)
    return np.stack([u, np.cross(q, u)], axis=1)


class QuadrotorFlexCableController(PDController):
    """Infinite-horizon LQR tracking a payload trajectory through an ``n``-link cable.

    ``compute(t, state[, reference])`` returns a ``(4,)`` **wrench**
    ``[f, M_x, M_y, M_z]`` — not the thrust *vector* M2/M3/M4 return.  An LQR
    on a quadrotor's own inputs produces the body moment directly, so
    :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase` runs its
    default cascade under ``InputType.WRENCH`` and the geometric attitude loop
    is not in the path at all.

    The optional third argument is the ``(RefState, Inputs)`` pair recovered by
    :class:`udaan.flatness.QuadrotorFlexCable`;
    :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.simulate`
    supplies it when the bound trajectory can be differentiated deeply enough.
    Without it the reference is the hanging equilibrium at the setpoint
    (:meth:`equilibrium_reference`), which is the honest feedback-only law.

    Args:
        num_links: ``n >= 1``.
        mass: quadrotor mass ``m_Q``.
        inertia: ``J``, a ``(3,)`` diagonal or ``(3, 3)`` tensor.
        link_masses: scalar (broadcast) or ``(n,)`` point masses lumped at each
            link's **distal** end, so ``link_masses[-1]`` carries the payload.
            Defaults to ``cable_mass`` spread evenly with ``payload_mass``
            added to the last link, exactly as the model builds it.
        link_lengths: scalar (broadcast) or ``(n,)`` link lengths; defaults to
            ``cable_length / n``.
        cable_length, cable_mass, payload_mass: the convenience parameters that
            build the two arrays above.
        gravity: ``g``.
        q_position, q_velocity, q_attitude, q_angular_rate: diagonal weights of
            the four rigid-body blocks of the LQR state cost, replacing the
            matching ``DEFAULT_FLEXCABLE_LQR_Q_*`` constants.
        q_link, q_link_rate: diagonal weights of the per-link blocks, likewise.
        r_thrust, r_moment: input weights on ``\\delta f`` and on each
            ``\\delta M`` component.
        setpoint: forwarded to :class:`~udaan.control.PDController`; the
            **payload** position reference, ``t -> (pos, vel, acc, ...)``.

    Raises:
        ValueError: on any malformed parameter, and on ``kp``/``kd``, which
            this controller has no PD to tune (use the ``q_*``/``r_*``
            weights).
    """

    def __init__(
        self,
        *,
        num_links: int = DEFAULT_NUM_CABLE_LINKS,
        mass: float = DEFAULT_QUAD_MASS,
        inertia=DEFAULT_QUAD_INERTIA,
        link_masses=None,
        link_lengths=None,
        cable_length: float = DEFAULT_CABLE_LENGTH,
        cable_mass: float = DEFAULT_CABLE_MASS,
        payload_mass: float = DEFAULT_PAYLOAD_MASS,
        gravity: float = GRAVITY,
        q_position: float = DEFAULT_FLEXCABLE_LQR_Q_POSITION,
        q_velocity: float = DEFAULT_FLEXCABLE_LQR_Q_VELOCITY,
        q_attitude: float = DEFAULT_FLEXCABLE_LQR_Q_ATTITUDE,
        q_angular_rate: float = DEFAULT_FLEXCABLE_LQR_Q_ANGULAR_RATE,
        q_link: float = DEFAULT_FLEXCABLE_LQR_Q_LINK,
        q_link_rate: float = DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE,
        r_thrust: float = DEFAULT_FLEXCABLE_LQR_R_THRUST,
        r_moment: float = DEFAULT_FLEXCABLE_LQR_R_MOMENT,
        **kwargs,
    ):
        # ``PDController`` would silently park these in ``self._gains``, which
        # nothing here reads — the exact class of bug M3 and M4 document having
        # been bitten by. There is no PD in this cascade; say so.
        for name in ("kp", "kd", "ki"):
            if name in kwargs:
                raise ValueError(
                    f"{type(self).__name__} is an LQR, not a PD controller, so "
                    f"'{name}' has nothing to tune. Use the q_* / r_* cost weights "
                    f"(see udaan.core.defaults.DEFAULT_FLEXCABLE_LQR_*)."
                )
        super().__init__(**kwargs)

        n = int(num_links)
        if n < 1:
            raise ValueError(f"num_links must be >= 1, got {n}")
        self._num_links = n

        self._weights = self._cost_weights(
            q_position=q_position,
            q_velocity=q_velocity,
            q_attitude=q_attitude,
            q_angular_rate=q_angular_rate,
            q_link=q_link,
            q_link_rate=q_link_rate,
            r_thrust=r_thrust,
            r_moment=r_moment,
        )

        # ``Controller.__init__`` bound _g/_ge3/_gravity at the package default;
        # rebind all three together so a caller-supplied gravity cannot leave
        # them disagreeing.
        self._g = float(gravity)
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._gravity = np.array([0.0, 0.0, -self._g])

        if link_lengths is None:
            link_lengths = np.full(n, float(cable_length) / n)
        if link_masses is None:
            built = np.full(n, float(cable_mass) / n)
            built[-1] += float(payload_mass)
            link_masses = built

        self._set_parameters(
            mass=mass, inertia=inertia, link_masses=link_masses, link_lengths=link_lengths
        )

    # ─── Parameter plumbing ────────────────────────────────────────

    @staticmethod
    def _cost_weights(**kwargs) -> dict[str, float]:
        """Validate the eight cost weights; every one must be strictly positive.

        A zero weight on a block is not merely soft, it is a singular ``R`` (for
        the input weights) or a ``Q`` that no longer detects part of the state,
        and the Riccati solution stops being the one this class documents.
        """
        out = {}
        for name, value in kwargs.items():
            weight = float(value)
            if not weight > 0.0:
                raise ValueError(f"{name} must be strictly positive; got {weight}")
            out[name] = weight
        return out

    def _broadcast(self, value, name: str) -> np.ndarray:
        arr = np.array(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_links, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != self._num_links:
            raise ValueError(
                f"{name} must be a scalar or have {self._num_links} entries; got {arr.shape[0]}"
            )
        # ``np.array`` copies: a caller's array must never be aliased in.
        return arr

    def _set_parameters(self, *, mass, inertia, link_masses, link_lengths) -> None:
        """Validate the physical parameters and rebuild everything derived.

        The flat map, the linearisation and the gain are all cached, so they
        have to be rebuilt together; going through one routine is what keeps a
        re-synced controller identical to a freshly constructed one.
        """
        self._mass = float(mass)
        if self._mass <= 0.0:
            raise ValueError(f"mass must be positive; got {self._mass}")

        tensor = np.array(inertia, dtype=float)
        if tensor.shape == (3,):
            tensor = np.diag(tensor)
        if tensor.shape != (3, 3):
            raise ValueError(f"inertia must be (3,) or (3, 3); got {tensor.shape}")
        self._inertia = tensor
        self._inertia_inv = np.linalg.inv(tensor)

        self._link_masses = self._broadcast(link_masses, "link_masses")
        if np.any(self._link_masses <= 0.0):
            raise ValueError(
                f"link_masses must be strictly positive; got {self._link_masses.tolist()}"
            )
        self._link_lengths = self._broadcast(link_lengths, "link_lengths")
        if np.any(self._link_lengths <= 0.0):
            raise ValueError(
                f"link_lengths must be strictly positive; got {self._link_lengths.tolist()}"
            )

        # One flat map, built from *these* parameters. ``simulate`` reads it off
        # this attribute rather than building its own from the model, so the
        # feedforward reference cannot describe a different machine from the one
        # the gain was solved for — M4's tension_policy lesson, transposed.
        self._flat_map = QuadrotorFlexCable(
            num_links=self._num_links,
            mass_q=self._mass,
            inertia=self._inertia,
            link_masses=self._link_masses.copy(),
            link_lengths=self._link_lengths.copy(),
            gravity=self._g,
        )
        # The gain is *invalidated*, not re-solved. Solving here would make
        # every ``mdl.link_masses = ...`` pay for a Riccati factorisation, and
        # — worse — would make a legitimate assignment on the *plant* fail
        # whenever the resulting linearisation happens to be too
        # ill-conditioned for the solver: the dynamics of a 1e-6 kg link on a
        # 1e-3 m lever are perfectly well posed (see ``mass_matrix`` in
        # :mod:`udaan.models.quadrotor_flexcable.base`) even where the LQR is
        # not. Deferring means such a machine can be built and integrated, and
        # only refuses when something actually asks this controller to fly it.
        self._invalidate_gain()

    def _invalidate_gain(self) -> None:
        """Drop the cached gain, so the next use re-solves it at the new parameters."""
        self._gain: np.ndarray | None = None
        self._closed_loop_poles: np.ndarray | None = None
        self._controllability_margin: float | None = None

    def _ensure_gain(self) -> np.ndarray:
        """The current gain, solving the Riccati equation if it is stale."""
        if self._gain is None:
            self._solve_gain()
        assert self._gain is not None  # _solve_gain either sets it or raises
        return self._gain

    def sync_from_model(self, model) -> None:
        """Re-read every physical parameter off ``model``, keeping the weights.

        The gain matrix is solved from the masses, lengths and inertia at
        construction, so ``mdl.link_masses = ...`` after the fact would
        otherwise leave this controller regulating a machine that no longer
        exists — and, worse, feeding forward a reference from a flat map built
        on the old one. The model calls this from its own parameter setters.

        Raises:
            ValueError: if ``model`` has a different ``num_links``. Every cached
                shape is ``n``-sized, and a controller for a different chain
                length is a different controller.
        """
        if int(model.num_links) != self._num_links:
            raise ValueError(
                f"cannot sync a {self._num_links}-link controller from a "
                f"{model.num_links}-link model; rebuild it with from_model()"
            )
        self._set_parameters(
            mass=model.mass,
            inertia=model.inertia,
            link_masses=model.link_masses,
            link_lengths=model.link_lengths,
        )

    @classmethod
    def from_model(cls, model, **kwargs) -> QuadrotorFlexCableController:
        """Build from a
        :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`-like
        model, so the controller and the plant agree on every parameter."""
        return cls(
            num_links=model.num_links,
            mass=model.mass,
            inertia=model.inertia,
            link_masses=np.asarray(model.link_masses, dtype=float).copy(),
            link_lengths=np.asarray(model.link_lengths, dtype=float).copy(),
            **kwargs,
        )

    # ─── Properties ────────────────────────────────────────────────

    @property
    def num_links(self) -> int:
        """Number of rigid links in the chain."""
        return self._num_links

    @property
    def mass(self) -> float:
        """Quadrotor mass, in kg."""
        return self._mass

    @property
    def inertia(self) -> np.ndarray:
        """Quadrotor inertia tensor."""
        return self._inertia

    @property
    def link_masses(self) -> np.ndarray:
        """Point mass at each link's distal end, ``(n,)``."""
        return self._link_masses

    @property
    def link_lengths(self) -> np.ndarray:
        """Per-link lengths, ``(n,)``."""
        return self._link_lengths

    @property
    def total_mass(self) -> float:
        """``M₀₀ = m_Q + Σ_k m_k`` — the mass the thrust has to carry at hover."""
        return float(self._mass + np.sum(self._link_masses))

    @property
    def cost_weights(self) -> dict[str, float]:
        """The eight LQR cost weights, as supplied. Read-only snapshot."""
        return dict(self._weights)

    @property
    def state_dim(self) -> int:
        """``12 + 6n`` — the size of the embedded error state :meth:`error_state` returns."""
        return 12 + 6 * self._num_links

    @property
    def gain(self) -> np.ndarray:
        """The LQR gain ``K``, shape ``(4, 12 + 6n)``.

        ``du = K s``, added to the feedforward wrench. Its columns on the
        ``2n`` gauge directions are exactly zero — see "What is and is not
        controllable" in the module docstring.

        Solved lazily, on first use, from the parameters current at that
        moment.
        """
        return self._ensure_gain()

    @property
    def closed_loop_poles(self) -> np.ndarray:
        """Eigenvalues of the *reduced* closed loop ``A_r + B_r K_r``, ``(12 + 4n,)``.

        Sorted by real part, most positive first, so ``[0]`` is the mode that
        limits the response. All are strictly negative by construction; the
        slowest one is the chain's top bending mode and is essentially
        independent of the cost weights.
        """
        self._ensure_gain()
        assert self._closed_loop_poles is not None
        return self._closed_loop_poles

    @property
    def controllability_margin(self) -> float:
        """``min_λ σ_min[A_r - λI  B_r]`` over the open-loop spectrum.

        The PBH test on the tangent-reduced pair: strictly positive means
        controllable. It falls by roughly an order of magnitude every two links
        (0.62, 0.085, 0.026, 0.008 at ``n = 1, 2, 3, 5`` on the defaults), which
        is the quantitative form of "one four-input actuator at the top of a
        long chain has very little authority over the far end".
        """
        self._ensure_gain()
        assert self._controllability_margin is not None
        return self._controllability_margin

    @property
    def flat_map(self) -> QuadrotorFlexCable:
        """The :class:`udaan.flatness.QuadrotorFlexCable` built from *these* parameters.

        Published so that
        :meth:`udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.simulate`
        can build the feedforward at this controller's own configuration. A
        controller that does not expose it gets no feedforward at all rather
        than one built from the model's parameters, which may be a different
        machine.
        """
        return self._flat_map

    @property
    def jet_order(self) -> int:
        """``2n + 4`` — payload-jet depth the flat feedforward needs.

        The one number that decides which trajectories can be fed forward at
        all: a reference must be differentiable in closed form this many times.
        See :func:`udaan.flatness.chain.chain_jet_order`.
        """
        return self._flat_map.jet_order

    # ─── Linearisation and gain ────────────────────────────────────

    def linearize(
        self,
        reference: QuadrotorFlexCableRefState | None = None,
        inputs: QuadrotorFlexCableInputs | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Variational linearisation ``(A, B)`` of the error dynamics, ``ṡ = A s + B δu``.

        ``A`` is ``(12 + 6n, 12 + 6n)`` and ``B`` is ``(12 + 6n, 4)``, in the
        block order of :meth:`error_state`. With no arguments the reference is
        the hanging equilibrium — every ``q_i^d = -e_3``, ``R_d = I``,
        ``\Omega_d = 0``, ``a_Q^d = 0``, ``f_d = M_{00} g`` — which is the
        operating point the shipped gain is solved at.

        The assembly follows the reference MATLAB's
        ``get_linear_error_dynamics`` block for block, with the
        ``\hat{\dot\omega} - \|\omega\|^2`` scalar-broadcast defect corrected
        (see "Errata in the reference"). It is exposed publicly because the
        justification for using a *constant* gain is a claim about how little
        ``A`` moves over the admissible reference set, and a claim like that
        should be measurable.

        Args:
            reference: a
                :class:`~udaan.flatness.QuadrotorFlexCableRefState`; ``None``
                for the hanging equilibrium.
            inputs: the matching
                :class:`~udaan.flatness.QuadrotorFlexCableInputs`, read for
                ``f_d``. ``None`` uses the hover thrust ``M_{00} g``.

        Raises:
            ValueError: if ``reference`` describes a different link count.
        """
        n = self._num_links
        eye = np.eye(3)
        M00, M0i, Mij = self._mass_coefficients()

        if reference is None:
            qd = np.tile(-_E3, (n, 1))
            wd = np.zeros((n, 3))
            dwd = np.zeros((n, 3))
            R_d = np.eye(3)
            Omega_d = np.zeros(3)
            accel_Qd = np.zeros(3)
        else:
            if len(reference.link_attitudes) != n:
                raise ValueError(
                    f"reference describes {len(reference.link_attitudes)} links, but "
                    f"this controller has {n}"
                )
            qd = np.asarray(reference.link_attitudes, dtype=float)
            wd = np.asarray(reference.link_angular_velocities, dtype=float)
            dwd = np.asarray(reference.link_angular_accelerations, dtype=float)
            R_d = np.asarray(reference.orientation, dtype=float)
            Omega_d = np.asarray(reference.angular_velocity, dtype=float)
            accel_Qd = np.asarray(reference.acceleration, dtype=float)
        thrust_d = M00 * self._g if inputs is None else float(inputs.thrust)

        dim = self.state_dim
        qd_hat = [hat(qd[i]) for i in range(n)]

        # ── E: the mass matrix of the equations of motion, at the reference ──
        mass = np.zeros((3 + 3 * n, 3 + 3 * n))
        mass[0:3, 0:3] = M00 * eye
        for i in range(n):
            r = 3 * (i + 1)
            mass[0:3, r : r + 3] = -M0i[i] * qd_hat[i]
            mass[r : r + 3, 0:3] = M0i[i] * qd_hat[i]
            for j in range(n):
                c = 3 * (j + 1)
                mass[r : r + 3, c : c + 3] = (
                    Mij[i, j] * eye if i == j else -Mij[i, j] * (qd_hat[i] @ qd_hat[j])
                )

        # ── F: the state Jacobian of the same right-hand side ──
        state_jac = np.zeros((3 + 3 * n, dim))
        state_jac[0:3, 6:9] = -thrust_d * R_d @ hat(_E3)
        tail = np.cumsum(self._link_masses[::-1])[::-1]
        for i in range(n):
            link_col = 12 + 3 * i
            rate_col = 12 + 3 * n + 3 * i
            speed_sq = float(np.dot(wd[i], wd[i]))
            state_jac[0:3, link_col : link_col + 3] = (
                M0i[i] * (hat(dwd[i]) - speed_sq * eye) @ qd_hat[i]
            )
            state_jac[0:3, rate_col : rate_col + 3] = 2.0 * M0i[i] * np.outer(qd[i], wd[i])

        for i in range(n):
            r = 3 * (i + 1)
            for j in range(n):
                link_col = 12 + 3 * j
                rate_col = 12 + 3 * n + 3 * j
                if i == j:
                    block = M0i[i] * hat(accel_Qd) + tail[i] * self._g * self._link_lengths[
                        i
                    ] * hat(_E3)
                    for k in range(n):
                        if k == i:
                            continue
                        block = block - Mij[i, k] * (
                            hat(qd_hat[k] @ dwd[k]) + float(np.dot(wd[k], wd[k])) * qd_hat[k]
                        )
                    state_jac[r : r + 3, link_col : link_col + 3] = block @ (-qd_hat[i])
                else:
                    speed_sq = float(np.dot(wd[j], wd[j]))
                    state_jac[r : r + 3, link_col : link_col + 3] = (
                        Mij[i, j] * qd_hat[i] @ (hat(dwd[j]) - speed_sq * eye) @ qd_hat[j]
                    )
                    state_jac[r : r + 3, rate_col : rate_col + 3] = (
                        2.0 * Mij[i, j] * qd_hat[i] @ np.outer(qd[j], wd[j])
                    )

        # ── G: the input Jacobian of the same right-hand side ──
        input_jac = np.zeros((3 + 3 * n, 4))
        input_jac[0:3, 0] = R_d @ _E3

        solved_state = np.linalg.solve(mass, state_jac)
        solved_input = np.linalg.solve(mass, input_jac)

        A = np.zeros((dim, dim))
        A[0:3, 3:6] = eye
        A[3:6, :] = solved_state[0:3, :]
        A[6:9, 6:9] = -hat(Omega_d)
        A[6:9, 9:12] = eye
        A[9:12, 9:12] = self._inertia_inv @ (
            hat(self._inertia @ Omega_d) - hat(Omega_d) @ self._inertia
        )
        for i in range(n):
            r = 12 + 3 * i
            A[r : r + 3, 12 + 3 * i : 12 + 3 * i + 3] = np.outer(qd[i], qd[i]) @ hat(wd[i])
            A[r : r + 3, 12 + 3 * n + 3 * i : 12 + 3 * n + 3 * i + 3] = eye - np.outer(qd[i], qd[i])
        A[12 + 3 * n :, :] = solved_state[3:, :]

        B = np.zeros((dim, 4))
        B[3:6, :] = solved_input[0:3, :]
        B[9:12, 1:] = self._inertia_inv
        B[12 + 3 * n :, :] = solved_input[3:, :]
        return A, B

    def tangent_projection(self, link_directions=None) -> np.ndarray:
        r"""``W``, the ``(12 + 6n, 12 + 4n)`` basis of the non-gauge error subspace.

        The twelve rigid-body coordinates pass through untouched; each link
        contributes the two directions of ``q_i^{d\perp}`` and drops the one
        along ``q_i^d``. The subspace contains the whole range of ``B`` at
        every reference, so ``B_r = WᵀB`` never loses anything.

        ``A_r = WᵀAW`` is lossless too **at a reference whose links are not
        turning**, which the hanging equilibrium is and which is where the
        shipped gain is solved: there ``(I - WWᵀ) A W`` is exactly zero. At a
        moving reference it is not — the ``ξ̇`` row carries
        ``q_i^d (q_i^d)^\top \hat\omega_i^d``, whose ``q_i^d`` component
        survives — and the reduction becomes an approximation with a residual
        of order ``1e-4`` to ``1e-3`` relative to ``\|A\|`` on an admissible
        trajectory. See "What is and is not controllable" in the module
        docstring.
        """
        n = self._num_links
        if link_directions is None:
            link_directions = np.tile(-_E3, (n, 1))
        directions = np.asarray(link_directions, dtype=float)
        dim = self.state_dim
        basis = np.zeros((dim, 12 + 4 * n))
        basis[0:12, 0:12] = np.eye(12)
        for i in range(n):
            tangent = _tangent_basis(directions[i])
            basis[12 + 3 * i : 12 + 3 * i + 3, 12 + 2 * i : 12 + 2 * i + 2] = tangent
            basis[
                12 + 3 * n + 3 * i : 12 + 3 * n + 3 * i + 3,
                12 + 2 * n + 2 * i : 12 + 2 * n + 2 * i + 2,
            ] = tangent
        return basis

    def cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Q_r, R)`` — the reduced state cost and the input cost.

        ``Q_r`` is ``(12 + 4n)``-square: the six block weights of
        :attr:`cost_weights`, with the link blocks carrying two entries per
        link rather than three, since the third is the gauge direction the
        projection removes.
        """
        n = self._num_links
        w = self._weights
        diagonal = np.concatenate(
            [
                np.full(3, w["q_position"]),
                np.full(3, w["q_velocity"]),
                np.full(3, w["q_attitude"]),
                np.full(3, w["q_angular_rate"]),
                np.full(2 * n, w["q_link"]),
                np.full(2 * n, w["q_link_rate"]),
            ]
        )
        return np.diag(diagonal), np.diag([w["r_thrust"]] + [w["r_moment"]] * 3)

    def _solve_gain(self) -> None:
        """Solve the algebraic Riccati equation at the hanging equilibrium."""
        A, B = self.linearize()
        self.set_gain_from(A, B)

    def set_gain_from(self, A: np.ndarray, B: np.ndarray, link_directions=None) -> None:
        r"""Re-solve the Riccati equation at a given linearisation and adopt the gain.

        The hook a caller schedules gains through: ``ctrl.set_gain_from(
        *ctrl.linearize(ref, inputs), ref.link_attitudes)``. The solve runs on
        the tangent-reduced pair (:meth:`tangent_projection`) and the resulting
        ``K_r`` is pushed back out as ``K = K_r Wᵀ``, so the returned gain never
        acts on a gauge direction.

        **The reduction is exact only where the reference links are at rest.**
        At the hanging equilibrium — the default, and where the shipped gain is
        solved — ``(I - WWᵀ) A W`` is identically zero and this is the same
        problem as the embedded one. At a *moving* reference it is not: the
        ``ξ̇`` row's ``q_i^d (q_i^d)^\top \hat\omega_i^d`` term leaks out of
        the subspace, so scheduling gains this way solves a nearby problem
        rather than the exact one. The leak is of order ``1e-4`` to ``1e-3``
        relative to ``\|A\|`` on trajectories the flat map accepts, which is
        why the approximation is a usable one; it is not zero, and the
        docstrings that used to call the reduction exact everywhere were
        wrong.

        Raises:
            ValueError: if ``(A, B)`` is not stabilisable on the reduced
                subspace, which the Riccati solver reports.
        """
        from scipy.linalg import solve_continuous_are

        basis = self.tangent_projection(link_directions)
        A_r = basis.T @ A @ basis
        B_r = basis.T @ B
        Q_r, R = self.cost_matrices()
        try:
            P = solve_continuous_are(A_r, B_r, Q_r, R)
        except (np.linalg.LinAlgError, ValueError) as err:
            # Both failure modes land here: an unstabilisable pair, and a pair
            # whose generalised Schur reordering the solver refuses as too
            # ill-conditioned — which is what a chain of six orders of
            # magnitude in mass ratio produces.
            raise ValueError(
                f"no stabilising solution to the algebraic Riccati equation for this "
                f"{self._num_links}-link chain (masses {self._link_masses.tolist()}, "
                f"lengths {self._link_lengths.tolist()}): {err}"
            ) from err
        gain_r = -np.linalg.solve(R, B_r.T @ P)
        self._gain = gain_r @ basis.T
        closed = A_r + B_r @ gain_r
        poles = np.linalg.eigvals(closed)
        self._closed_loop_poles = poles[np.argsort(-poles.real)]
        self._controllability_margin = self._pbh_margin(A_r, B_r)

    @staticmethod
    def _pbh_margin(A_r: np.ndarray, B_r: np.ndarray) -> float:
        """``min_λ σ_min[A_r - λI  B_r]`` over the open-loop spectrum."""
        size = A_r.shape[0]
        margin = np.inf
        for eigenvalue in np.linalg.eigvals(A_r):
            stacked = np.hstack([A_r - eigenvalue * np.eye(size), B_r.astype(complex)])
            margin = min(margin, float(np.linalg.svd(stacked, compute_uv=False).min()))
        return margin

    def _mass_coefficients(self) -> tuple[float, np.ndarray, np.ndarray]:
        """``(M₀₀, M₀ᵢ, Mᵢⱼ)`` of the M5 derivation, from *this* controller's parameters.

        Deliberately recomputed rather than read off a model: the whole point of
        :meth:`sync_from_model` is that this object knows its own machine.
        """
        masses = self._link_masses
        lengths = self._link_lengths
        tail = np.cumsum(masses[::-1])[::-1]
        M00 = float(self._mass + masses.sum())
        M0i = tail * lengths
        indices = np.arange(self._num_links)
        Mij = tail[np.maximum.outer(indices, indices)] * np.outer(lengths, lengths)
        return M00, M0i, Mij

    # ─── Reference and error ───────────────────────────────────────

    def equilibrium_reference(
        self, t: float
    ) -> tuple[QuadrotorFlexCableRefState, QuadrotorFlexCableInputs]:
        """The hanging equilibrium at the setpoint — the feedback-only reference.

        Every link straight down, the airframe level and still, the thrust
        carrying the whole system weight, and the payload at ``setpoint(t)``'s
        position moving at its velocity. The setpoint's *acceleration* is
        deliberately **not** used: an accelerating reference is not an
        equilibrium, its links are not vertical, and pretending otherwise would
        smuggle in half a feedforward and make the with/without comparison
        meaningless. Accelerating references are what
        :attr:`flat_map` is for.
        """
        n = self._num_links
        sp = self.setpoint(t)
        payload_position = np.asarray(sp[0], dtype=float)
        payload_velocity = np.asarray(sp[1], dtype=float)
        down = -_E3
        offset = float(np.sum(self._link_lengths))
        position = payload_position + offset * _E3

        ref = QuadrotorFlexCableRefState(
            payload_position=payload_position,
            payload_velocity=payload_velocity,
            payload_acceleration=np.zeros(3),
            position=position,
            velocity=payload_velocity,
            acceleration=np.zeros(3),
            link_attitudes=[down.copy() for _ in range(n)],
            link_angular_velocities=[np.zeros(3) for _ in range(n)],
            link_angular_accelerations=[np.zeros(3) for _ in range(n)],
        )
        inputs = QuadrotorFlexCableInputs(
            thrust=self.total_mass * self._g,
            moment=np.zeros(3),
            tensions=np.cumsum(self._link_masses[::-1])[::-1] * self._g,
        )
        return ref, inputs

    def error_state(self, state, reference: QuadrotorFlexCableRefState) -> np.ndarray:
        r"""The ``(12 + 6n,)`` variational error of :math:`s` in the module docstring.

        Args:
            state: a
                :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableState`.
            reference: the matching
                :class:`~udaan.flatness.QuadrotorFlexCableRefState`.

        Raises:
            ValueError: if the state or the reference has a different link count.
        """
        n = self._num_links
        if len(state.link_attitudes) != n:
            raise ValueError(
                f"state has {len(state.link_attitudes)} links, but this controller has {n}"
            )
        if len(reference.link_attitudes) != n:
            raise ValueError(
                f"reference has {len(reference.link_attitudes)} links, but this controller has {n}"
            )

        R = np.asarray(state.orientation, dtype=float)
        Omega = np.asarray(state.angular_velocity, dtype=float)
        R_d = np.asarray(reference.orientation, dtype=float)
        Omega_d = np.asarray(reference.angular_velocity, dtype=float)

        error = np.zeros(self.state_dim)
        error[0:3] = np.asarray(state.position, dtype=float) - np.asarray(
            reference.position, dtype=float
        )
        error[3:6] = np.asarray(state.velocity, dtype=float) - np.asarray(
            reference.velocity, dtype=float
        )
        error[6:9] = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        error[9:12] = Omega - R.T @ R_d @ Omega_d
        for i in range(n):
            q = np.asarray(state.link_attitudes[i], dtype=float)
            omega = np.asarray(state.link_angular_velocities[i], dtype=float)
            q_d = np.asarray(reference.link_attitudes[i], dtype=float)
            omega_d = np.asarray(reference.link_angular_velocities[i], dtype=float)
            error[12 + 3 * i : 12 + 3 * i + 3] = np.cross(q_d, q)
            error[12 + 3 * n + 3 * i : 12 + 3 * n + 3 * i + 3] = omega + np.cross(
                q, np.cross(q, omega_d)
            )
        return error

    # ─── Control ───────────────────────────────────────────────────

    def compute(self, *args) -> np.ndarray:
        """The wrench ``[f, M_x, M_y, M_z]``.

        Args:
            args: ``(t, state)``, or ``(t, state, reference)`` where
                ``reference`` is the ``(RefState, Inputs)`` pair recovered by
                :attr:`flat_map`. Without it the hanging equilibrium at the
                setpoint is used.
        """
        t = args[0]
        state = args[1]
        reference = args[2] if len(args) > 2 and args[2] is not None else None
        if reference is None:
            ref, feedforward = self.equilibrium_reference(t)
        else:
            ref, feedforward = reference

        correction = self._ensure_gain() @ self.error_state(state, ref)
        return np.concatenate(
            [[float(feedforward.thrust) + correction[0]], feedforward.moment + correction[1:]]
        )
