"""Default physical parameters and control gains for Udaan.

All magic numbers used across the codebase are defined here.
Import from this module instead of hard-coding values.
"""

import numpy as np

# =============================================================================
# Physical constants
# =============================================================================

GRAVITY = 9.81  # m/s^2

# =============================================================================
# Default quadrotor parameters
# =============================================================================

# Two airframe parameter sets. `DEFAULT_QUAD_MASS` and `DEFAULT_QUAD_INERTIA`
# are the vehicle M2–M6 are built and tuned around. The M1 quadrotor's MuJoCo
# asset (`quadrotor_mj.xml`) describes a different airframe, and its analytical
# model must fly the same one or a base-vs-MuJoCo comparison compares two
# machines; so M1 uses `DEFAULT_M1_QUAD_*`, which equal the MJCF
# (tests/test_quadrotor_mujoco.py checks it). The M2–M6 gains and tests are
# tuned to the shared pair, which therefore stays separate.

DEFAULT_QUAD_MASS = 0.9  # kg
DEFAULT_QUAD_INERTIA = np.array(
    [[0.0023, 0.0, 0.0], [0.0, 0.0023, 0.0], [0.0, 0.0, 0.004]]
)  # kg*m^2

#: M1 airframe mass, in kg — the quadrotor body of ``quadrotor_mj.xml``.
DEFAULT_M1_QUAD_MASS = 0.75
#: M1 airframe inertia, in kg·m² — the quadrotor body of ``quadrotor_mj.xml``.
DEFAULT_M1_QUAD_INERTIA = np.diag([0.0053, 0.0049, 0.0098])
DEFAULT_ARM_LENGTH = 0.2  # m
#: Rotor angles about ``b₃`` from ``b₁``, in rad, rotor ``i`` sitting at
#: ``ℓ (cos θᵢ, sin θᵢ, 0)`` — an X layout. The rotor sites of ``quadrotor_mj.xml``.
DEFAULT_ROTOR_ANGLES = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
#: Rotor spin directions ``dᵢ = ±1``: rotor ``i``'s yaw moment is ``dᵢ c fᵢ`` with
#: ``c = k_M/k_F``. The sign of each rotor's yaw gear in ``quadrotor_mj.xml``.
DEFAULT_SPIN_DIRECTIONS = np.array([-1.0, 1.0, -1.0, 1.0])

#: Rate at which the M1 quadrotor closes its attitude loop, in Hz: 800 Hz, four
#: updates per 5 ms position step, each held over two of the eight physics
#: substeps, on every input path and every backend. It must be ``k/dt`` for a
#: whole ``k`` that divides the substeps per step; ``None`` means once per outer
#: step (``1/dt``). See :class:`~udaan.models.quadrotor.QuadrotorBase`'s
#: ``attitude_rate_hz``.
DEFAULT_ATTITUDE_RATE_HZ: float | None = 800.0

#: Rate at which the M1 quadrotor's physics integrates, in Hz: ``N = rate · dt``
#: substeps of ``h = dt/N`` per outer step, on both backends (the ZOH substeps of
#: :class:`~udaan.models.quadrotor.QuadrotorBase`, and the timestep
#: :class:`~udaan.models.quadrotor.QuadrotorMujoco` writes into MuJoCo). 1600 Hz
#: is ``N = 8`` substeps of 0.625 ms at the default 5 ms ``dt``.
DEFAULT_PHYSICS_RATE_HZ = 1600.0
#: The range ``physics_rate_hz`` must lie in, in Hz (inclusive).
PHYSICS_RATE_RANGE_HZ = (500.0, 2000.0)

# Motor constants
DEFAULT_FORCE_CONSTANT = 4.104890333e-6  # N/(rad/s)^2
DEFAULT_TORQUE_CONSTANT = 1.026e-07  # Nm/(rad/s)^2

# Actuator limits
#
# Every model's ``_repackage_input()`` clips the wrench it has assembled to
# these before handing it to the integrator, so they are what makes a
# controller's demand physically bounded.
#
# Sized around the 0.9 kg default airframe: hover needs 8.8 N, so
# :data:`DEFAULT_MAX_THRUST` leaves a thrust-to-weight of about 2.3, and the
# four-rotor sum of :data:`DEFAULT_PROP_MAX_FORCE` (40 N) sits comfortably above
# it so that the allocation matrix, not the per-rotor clip, is the binding
# constraint in normal flight. Yaw torque is the smaller limit because a
# quadrotor produces it from rotor drag rather than differential thrust.

#: Lower bound on collective thrust, in N. Zero, not negative: rotors that spin
#: one way cannot pull down.
DEFAULT_MIN_THRUST = 0.0
#: Upper bound on collective thrust, in N.
DEFAULT_MAX_THRUST = 20.0
#: Lower bound on body torque ``[M_x, M_y, M_z]``, in N·m.
DEFAULT_MIN_TORQUE = np.array([-5.0, -5.0, -2.0])
#: Upper bound on body torque ``[M_x, M_y, M_z]``, in N·m.
DEFAULT_MAX_TORQUE = np.array([5.0, 5.0, 2.0])
#: Lower bound on a single rotor's force, in N.
DEFAULT_PROP_MIN_FORCE = 0.0
#: Upper bound on a single rotor's force, in N.
DEFAULT_PROP_MAX_FORCE = 10.0

#: Collective-thrust ceiling for the multi-quadrotor rigid-body model (M4), in N.
#:
#: Higher than :data:`DEFAULT_MAX_THRUST` because each vehicle there carries a
#: share of a rigid-body payload on top of its own weight, and the formation
#: also has to supply the payload's attitude control through cable tensions.
DEFAULT_MULTI_QUAD_MAX_THRUST = 30.0
#: Body-torque bound for M4, in N·m — isotropic, unlike the single-vehicle case.
DEFAULT_MULTI_QUAD_MAX_TORQUE = np.array([3.0, 3.0, 3.0])

# =============================================================================
# Default payload parameters
# =============================================================================

DEFAULT_PAYLOAD_MASS = 0.2  # kg
DEFAULT_CABLE_LENGTH = 1.0  # m

#: Radius of the point-mass payload sphere, in metres (geometry only).
DEFAULT_PAYLOAD_RADIUS = 0.05  # m

# =============================================================================
# Default flexible-cable parameters (M5)
# =============================================================================
#
# M5 replaces M2's single taut cable with a serial chain of rigid links. The
# chain is a *discretisation*, so its two parameters are the resolution and the
# mass being discretised; the total length stays :data:`DEFAULT_CABLE_LENGTH`
# and the tip mass stays :data:`DEFAULT_PAYLOAD_MASS`, which is what makes M5
# at ``num_links=1, cable_mass=0`` literally M2.

#: Number of rigid links the flexible cable is discretised into.
#:
#: Three, not more, because the chain's fastest internal mode grows roughly
#: like ``n^{3/2}`` and udaan's ZOH-Euler integrator is only *conditionally*
#: stable — see "Numerical stability" in
#: :mod:`udaan.models.quadrotor_flexcable.base`. At the shared 1.25 ms substep
#: this default leaves ~2x margin;
#: :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.natural_frequencies`
#: is what to consult before raising it, and the constructor warns when the
#: combination of ``num_links`` and ``dt`` has left the stable region. The
#: growth exponent rises steeply with the link count at fixed cable mass, so a
#: long chain at this ``dt`` warns out of the constructor.
DEFAULT_NUM_CABLE_LINKS = 3

#: Total mass of the cable itself, in kg, spread evenly over its links.
#:
#: A tenth of the payload mass. The cable *could* be lighter — a real 1 m
#: tether is nearer 10 g, which is what
#: :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_flexible_cable_payload`
#: defaults to — but the fastest internal mode of a lumped-mass chain
#: scales like ``sqrt(m_payload / m_link)``, so a 10 g cable at
#: ``num_links = 3`` is already three times too fast for the 1.25 ms substep.
#: Both the analytical model and :func:`udaan.utils.assets.quad_flexcable` read
#: this constant, so they always describe the same cable whatever it is set to.
#:
#: The 0.1 kg here is the default for M5's *own* asset and analytical model
#: only. ``create_flexible_cable_payload`` keeps its own ``cable_mass = 0.01``
#: keyword default, which is what the fleet asset (``_generate_xml`` in
#: :mod:`udaan.models.fleet.quadrotor_cspayload_fleet`) uses.
DEFAULT_CABLE_MASS = 0.1  # kg

# =============================================================================
# Default multi-quadrotor (cooperative transport) parameters
# =============================================================================

#: Number of quadrotors in a cooperative-transport team.
DEFAULT_NUM_QUADROTORS = 3

#: Horizontal radius of the quadrotor ring around the payload, in metres.
#: Must stay strictly below the cable length so the cables can be taut.
DEFAULT_FORMATION_RADIUS = 0.5  # m

#: Height of the quadrotor ring above the ground at spawn, in metres. The
#: payload height is *derived* from this, the cable length and the formation
#: radius so that the cables start taut — never chosen independently.
DEFAULT_SPAWN_HEIGHT = 2.0  # m

# =============================================================================
# Default rigid-body payload parameters (M4)
# =============================================================================
#
# The M4 payload is a *rigid body*, not a point mass, so it needs a mass, an
# inertia tensor and a set of body-frame attachment points — the last of these
# is what makes the system able to exert a moment on the load at all.  The
# defaults describe a thin square slab, which is also what
# :func:`udaan.utils.assets.multi_quad_rigidbody` emits when it is not told
# otherwise — so the analytical model and the generated MJCF agree by
# construction.

#: Mass of the default rigid-body payload, in kg.
DEFAULT_RIGID_PAYLOAD_MASS = 1.0  # kg

#: Half-extents ``(a, b, c)`` of the default slab payload, in metres.
DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS = np.array([0.5, 0.5, 0.05])  # m

#: Inertia of the default slab payload about its centre of mass, in kg*m^2.
#: Uniform box: ``I = (m/3) diag(b²+c², a²+c², a²+b²)`` with *half*-extents,
#: which is exactly the formula MuJoCo's ``inertiafromgeom`` applies to a
#: ``type="box"`` geom — so the analytical model and the MJCF agree by
#: construction rather than by coincidence.
DEFAULT_RIGID_PAYLOAD_INERTIA = (DEFAULT_RIGID_PAYLOAD_MASS / 3.0) * np.diag(
    [
        DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS[1] ** 2 + DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS[2] ** 2,
        DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS[0] ** 2 + DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS[2] ** 2,
        DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS[0] ** 2 + DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS[1] ** 2,
    ]
)

#: Horizontal radius of the ring of cable attachment points on the payload,
#: measured in the payload body frame, in metres.
DEFAULT_ATTACHMENT_RADIUS = 0.5  # m

#: Height of the attachment ring above the payload centre of mass, in metres.
#: Matches the top face of the default slab.
DEFAULT_ATTACHMENT_HEIGHT = 0.05  # m

# =============================================================================
# Default flexible-hose parameters (M6)
# =============================================================================
#
# M6 is one flexible hose — a serial chain of ``n`` rigid links joined by ideal
# spherical joints — with quadrotors bolted to some of its ``n + 1`` vertices.
# It is M5's chain with the single airframe at the top replaced by an arbitrary
# attachment pattern, so the constants below are the chain constants again,
# sized for a *carried* hose rather than a hanging tether.

#: Number of rigid links the hose is discretised into.
#:
#: Sized on the same criterion as :data:`DEFAULT_NUM_CABLE_LINKS`: the
#: ZOH-Euler loop is only conditionally stable, and
#: :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.stability_growth_rate`
#: is the number that says where the limit is. The growth exponent rises with
#: the link count, so a long hose warns straight out of the constructor; the
#: limit also depends on the attachment pattern, so the default pattern (a
#: vehicle at each end) is not the envelope, and a non-default pattern should
#: be checked with ``stability_growth_rate`` before it is flown.
DEFAULT_NUM_HOSE_LINKS = 4

#: Total length of the hose, in metres, split evenly over its links.
DEFAULT_HOSE_LENGTH = 2.0  # m

#: Total mass of the hose itself, in kg, lumped evenly onto its ``n + 1``
#: vertices. The quadrotor masses are *added* to the vertices they sit on, so
#: the mass that enters the dynamics at vertex ``j`` is ``m_j + m_{Q_j}``.
#:
#: 0.25 kg/m over the default 2 m — a real layflat hose of that bore. It is
#: also about the lightest the ZOH-Euler loop tolerates: the fastest internal
#: mode of a lumped chain scales like the square root of the heavy/light vertex
#: mass ratio, so a lighter hose against a 0.9 kg airframe at each end pushes
#: :data:`DEFAULT_NUM_HOSE_LINKS` links past the warning threshold.
DEFAULT_HOSE_MASS = 0.5  # kg

#: Horizontal span of each hose segment at spawn, as a fraction of that
#: segment's own length.
#:
#: 1.0 would be a straight, perfectly taut segment between two quadrotors and
#: 0 would stack them on top of each other; 0.8 leaves a visible sag while
#: keeping every link taut. See :func:`udaan.utils.hose.hose_spawn_directions`
#: for the triangle this closes and the feasibility condition it has to meet.
DEFAULT_HOSE_SAG_RATIO = 0.8

#: MuJoCo integrator timestep written into generated MJCF, in seconds.
#:
#: It is the analytical models' *inner* substep (a 5 ms outer loop over four
#: 1.25 ms substeps), so the two backends integrate at the same rate and — the
#: load-bearing part — 5 ms divides by it exactly. A MuJoCo backend takes
#: ``round(dt / timestep)`` substeps per :meth:`step`, so at MuJoCo's own
#: default of 2 ms that is ``round(2.5) = 2``: 4 ms of physics per step, while
#: ``dt`` still reads 5 ms and ``self.t`` drifts against the analytical model's
#: by 20%.
DEFAULT_MJCF_TIMESTEP = 1.25e-3  # s

#: Outer control-loop period shared by every model, in seconds.
#:
#: The analytical models take it as ``dt`` and subdivide it into four
#: :data:`DEFAULT_MJCF_TIMESTEP` substeps; the MuJoCo backends take
#: ``round(dt / DEFAULT_MJCF_TIMESTEP) = 4`` integrator steps per
#: :meth:`step`. The exact division is the point — see
#: :data:`DEFAULT_MJCF_TIMESTEP`.
DEFAULT_SIM_DT = 5e-3  # s

# =============================================================================
# Default control gains
# =============================================================================

# Position PD gains
DEFAULT_POS_KP = np.array([4.1, 4.1, 8.1])
DEFAULT_POS_KD = np.array([3.0, 3.0, 9.0])

# Attitude geometric controller gains (Lee 2010)
DEFAULT_ATT_KP = np.array([2.4, 2.4, 1.35])
DEFAULT_ATT_KD = np.array([0.35, 0.35, 0.225])

#: Smallest ``‖b₃ × b₁_hint‖`` the commanded-attitude construction will use a
#: raw heading hint at.
#:
#: :meth:`udaan.manif.SO3.from_two_vectors` divides by that norm, so the hint's
#: usable heading information vanishes as ``b₁_hint`` swings onto ``b₃``.
#: The flatness reference's own ``b₁`` is a *tilted* hint — unlike a fixed
#: world axis, it degenerates whenever the commanded thrust direction
#: approaches the reference's forward axis, which a yawing reference at high
#: tilt reaches, and the commanded attitude can then jump by a large angle in
#: one step.
#:
#: Below it,
#: :meth:`~udaan.control.quadrotor.GeometricAttitudeController._cmd_accel_to_cmd_att`
#: blends the hint toward the reference frame carried onto ``b₃`` by the
#: minimal rotation, which is orthogonal to ``b₃`` by construction. ``0.3`` is
#: ``sin 17.5°``: close enough to the singularity that the raw hint's heading
#: is already unreliable, far enough that a well-tracked reference never
#: reaches it.
MIN_ATTITUDE_HINT_CONDITIONING = 0.3

# Payload controller gains (Sreenath 2013)
DEFAULT_PAYLOAD_POS_KP = np.array([4.0, 4.0, 8.0])
DEFAULT_PAYLOAD_POS_KD = np.array([3.0, 3.0, 6.0])
DEFAULT_PAYLOAD_CABLE_KP = np.array([24.0, 24.0, 24.0])
DEFAULT_PAYLOAD_CABLE_KD = np.array([8.0, 8.0, 8.0])

# Multi-quadrotor cooperative payload gains (Sreenath & Kumar 2013).
#
# Kept separate from the single-quadrotor ``DEFAULT_PAYLOAD_*`` set above
# because the two loops are normalised differently, and the units differ with
# them. M2's gains are *force* gains (N/m, N·s/m); M3's are **acceleration**
# gains (1/s², 1/s) for the payload loop and 1/s², 1/s for the cable loop,
# which carries its own ``m_i l_i``. That is what makes them mean the same
# thing at every ``nQ``: a force-space PD would have to be divided by the total
# system mass ``m_L + Σ_i m_i``, which grows with ``nQ``, so a pair that is
# well damped for one quadrotor is sluggish for four.
#
# Payload loop: ω_n ≈ 2.0-2.4 rad/s, ζ ≈ 0.75-0.8.
# Cable loop:   ω_n ≈ 5.5 rad/s, ζ ≈ 0.91.
# The ordering ``k_p^L ≪ k_p^q ≪ k_R`` is the time-scale separation the
# cascade's stability argument assumes.
DEFAULT_MULTI_PAYLOAD_POS_KP = np.array([4.0, 4.0, 6.0])
DEFAULT_MULTI_PAYLOAD_POS_KD = np.array([3.0, 3.0, 4.0])
DEFAULT_MULTI_PAYLOAD_CABLE_KP = np.array([30.0, 30.0, 30.0])
DEFAULT_MULTI_PAYLOAD_CABLE_KD = np.array([10.0, 10.0, 10.0])

# Rigid-body payload attitude loop (M4).
#
# The extra loop M4 has and M3 does not: a geometric SO(3) PD on the payload
# attitude whose output is a *moment*, in N·m per unit of the dimensionless
# ``e_R``/``e_Ω`` error pair — the same units as the per-quadrotor
# ``DEFAULT_ATT_*`` above, not the acceleration units of the payload position
# loop.  They cannot be normalised the way ``DEFAULT_MULTI_PAYLOAD_POS_*`` are:
# a moment command has to be divided across the cables by the grasp map, whose
# scale is set by the attachment geometry, so it is the physical torque that
# has to be commanded.  Sized against the default slab payload
# (``J_L ≈ diag(0.084, 0.084, 0.167)``) for ω_n ≈ 3 rad/s, ζ ≈ 0.9.
DEFAULT_RIGID_PAYLOAD_ATT_KP = np.array([0.8, 0.8, 1.5])
DEFAULT_RIGID_PAYLOAD_ATT_KD = np.array([0.45, 0.45, 0.9])

# Flexible-cable payload LQR (M5).
#
# M5 is the first system here whose published law is a linear-quadratic
# regulator rather than a geometric cascade, so its "gains" are the weights of
# the cost
#
#     J = ∫ sᵀ Q s + duᵀ R du dt,
#
# not a PD pair: the gain matrix ``K = -R⁻¹ BᵀP`` is solved from these once, at
# construction, by :class:`~udaan.control.quadrotor_flexcable.QuadrotorFlexCableController`.
# The six ``Q`` entries weight the six blocks of the variational error state
# ``s = [e_x, e_v, η, δΩ, ξ_i, δω_i]``; the two ``R`` entries weight the thrust
# and the three body moments.
#
# The ``Q`` block values are the MATLAB reference implementation's
# (``github.com/vkotaru/quadrotor-flexible-cables-payload/Control/sQnlink/
# sQ_nlink_control.m::odefun_riccati``, line 512,
# ``Q1 = blkdiag(0.5 I6, 0.75 I6, I_3n, 0.75 I_3n)``) unchanged.
#
# ``R`` is **not**.  The reference weights all four inputs equally at
# ``0.2 I4`` (line 514), which ignores that the moment channel's input matrix
# is ``J⁻¹`` — spectral norm 435 for the default airframe — while the thrust
# channel's is ``1/M₀₀``, order 1.  Equal weights therefore buy an attitude
# loop several hundred times faster than the thrust loop: its fastest
# closed-loop pole lies beyond −800 rad/s, against a 200 Hz outer loop.
#
# The mechanism is **sampled-data phase loss**, not discrete-time divergence:
# neither the exact 5 ms hold nor the 1.25 ms forward-Euler substep amplifies
# such a pole.  What breaks the loop is that the *command* is frozen across the
# whole 5 ms hold, and the hold's half-sample delay ``ω Δt / 2`` is then more
# phase lag at crossover than the design has margin for.  The reference's
# adaptive ``ode15s`` re-evaluates the control continuously and does not meet
# this.
#
# Weighting the moments at 20 brings the fastest pole inside −100 rad/s.
# Being an *input* weight it leaves the chain modes nearly alone, but "nearly"
# is n-dependent, and from n = 2 up it is in the *worse* direction: the
# slowest closed-loop real part moves 1.5 % at n = 2, 4.6 % at n = 3 and 13 %
# at n = 5 (pinned in ``tests/test_quadrotor_flexcable_control.py``).  It is a
# real price, paid for a loop that settles; see the damping-versus-tracking
# trade-off in :mod:`udaan.control.quadrotor_flexcable.payload_controller`.
DEFAULT_FLEXCABLE_LQR_Q_POSITION = 0.5
DEFAULT_FLEXCABLE_LQR_Q_VELOCITY = 0.5
DEFAULT_FLEXCABLE_LQR_Q_ATTITUDE = 0.75
DEFAULT_FLEXCABLE_LQR_Q_ANGULAR_RATE = 0.75
DEFAULT_FLEXCABLE_LQR_Q_LINK = 1.0
DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE = 0.75
DEFAULT_FLEXCABLE_LQR_R_THRUST = 0.2
DEFAULT_FLEXCABLE_LQR_R_MOMENT = 20.0


# ── M6: multi-quadrotor flexible hose, LQR cost weights ────────────────
#
# The published weights (``matlab/examples/example1.m`` L68-72) are
# ``Q = blkdiag(0.5 I_6, 0.75 I_{6n}, I_{3nQ}, 0.75 I_{3nQ})`` and
# ``R = 0.2 I_{4nQ}``, and the six ``Q`` blocks below are those, read as the
# *intent* rather than as the literal diagonal.  They are not the same thing:
# the reference's own error vector is ``[e_x; xi; e_v; delta-omega; eta;
# delta-Omega]`` (``error.m`` L47) while that ``blkdiag`` groups ``[x, v]``
# first and the hose second, so on the reference's own ordering the ``0.5``
# block lands on ``e_x`` *and the first link's* ``xi``, and ``e_v`` picks up
# ``0.75``.  The blocks here follow the intent — one weight per physical
# block — rather than that literal diagonal.
DEFAULT_HOSE_LQR_Q_ROOT_POSITION = 0.5
DEFAULT_HOSE_LQR_Q_ROOT_VELOCITY = 0.5
DEFAULT_HOSE_LQR_Q_LINK = 0.75
DEFAULT_HOSE_LQR_Q_LINK_RATE = 0.75
DEFAULT_HOSE_LQR_Q_ATTITUDE = 1.0
DEFAULT_HOSE_LQR_Q_ANGULAR_RATE = 0.75
DEFAULT_HOSE_LQR_R_THRUST = 0.2

# M5's input-weighting problem applies to M6 as well.  The
# published ``R = 0.2 I`` weights a thrust and a moment equally, though a
# newton of thrust and a newton-metre of moment do very different things: the
# thrust column of ``B`` scales as ``1/M00`` and the moment column as
# ``||J^-1||``.  The largest moment column is ``||J^-1|| = 435`` whatever the
# hose, while the thrust columns run 0.435 to 2.83 over the attachment patterns
# tabulated in that module, so the ratio is 150 to 1000 rather than one
# number.  The resulting
# attitude loop is far faster than udaan's 200 Hz zero-order hold can carry.
# See :mod:`udaan.control.multi_quad_hose.hose_controller` for what the
# moment weight costs the hose modes in return.
DEFAULT_HOSE_LQR_R_MOMENT = 20.0
