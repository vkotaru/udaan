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
# are the vehicle the payload and multi-vehicle models are built and tuned
# around. The M1 quadrotor's MuJoCo asset (`quadrotor_mj.xml`) describes a
# different airframe, and its analytical model must fly the same one or a
# base-vs-MuJoCo comparison compares two machines; so M1 uses
# `DEFAULT_M1_QUAD_*`, which equal the MJCF (tests/test_quadrotor_mujoco.py
# checks it). The payload and multi-vehicle gains and tests are tuned to the
# shared pair, which therefore stays separate.

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

# The quadrotor with a cable-suspended payload (M2) flies the M1 airframe
# (`DEFAULT_M1_QUAD_MASS`, `DEFAULT_M1_QUAD_INERTIA`) with the payload and cable
# below, which are the payload body and the cable of `quad_payload_mj.xml` and
# `quad_flxcbl_mj.xml` (tests/test_quadrotor_cspayload_mujoco.py checks it). The
# shared `DEFAULT_PAYLOAD_MASS` and `DEFAULT_CABLE_LENGTH` above belong to the
# multi-vehicle models.

#: M2 payload mass ``m_L``, in kg.
DEFAULT_M2_PAYLOAD_MASS = 0.15
#: M2 cable length ``ℓ``, in m: the airframe-to-payload distance of a taut cable.
DEFAULT_M2_CABLE_LENGTH = 1.0
#: Smallest cable tension the M2 payload controller commands, as a fraction of
#: the payload weight: ``T_min = MIN_CABLE_TENSION_FRACTION · m_L · g``. The
#: thrust component along the cable never falls below ``T_min``, so a payload
#: loop that asks for a net downward force keeps the cable taut and the
#: commanded attitude upright instead of turning it over.
MIN_CABLE_TENSION_FRACTION = 0.1

#: Radius of the point-mass payload sphere, in metres (geometry only).
DEFAULT_PAYLOAD_RADIUS = 0.05  # m

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
#: radius so that the cables start taut — never chosen independently. In the
#: generated M5 MJCF it is the airframe's height, and the payload spawns
#: ``cable_length`` below it.
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

#: Tip-to-tip span of the generated airframe, in metres: two 0.2 m arms
#: (:data:`DEFAULT_ARM_LENGTH`) plus a 0.1 m rotor disc at each tip, as emitted by
#: :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_quadrotor0`.
DEFAULT_ROTOR_SPAN = 0.6  # m

#: Margin between the rotor spans of neighbouring quadrotors on a ring, in
#: metres. A ring whose neighbours are closer than ``DEFAULT_ROTOR_SPAN`` plus
#: this is refused (:func:`udaan.utils.formation.check_formation`), and the
#: default radius grows with the number of quadrotors to keep it
#: (:func:`udaan.utils.formation.default_formation_radius`).
DEFAULT_FORMATION_CLEARANCE = 0.1  # m

#: Height of the payload at spawn in a generated ring formation, in metres. The
#: ring sits one taut-cable drop ``sqrt(l² − r²)`` above it
#: (:func:`udaan.utils.formation.default_spawn_height`).
DEFAULT_PAYLOAD_SPAWN_HEIGHT = 1.0  # m

#: Mass of the generated airframe, in kg: what the multi-vehicle and M5 generators pass
#: :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_quadrotor0`
#: as its ``mass``.
DEFAULT_GENERATED_QUAD_MASS = 0.9  # kg
#: Body-frame inertia of the generated airframe at
#: :data:`DEFAULT_GENERATED_QUAD_MASS`, in kg·m²: ``R diag(I) Rᵀ`` of MuJoCo's
#: principal moments ``I`` and principal frame ``R = R(body_iquat)``. The
#: analytical multi-vehicle models start from it so that they fly the airframe
#: their MuJoCo backends fly.
DEFAULT_GENERATED_QUAD_INERTIA = np.diag([4.842347925e-3, 5.933258135e-3, 1.048606061e-2])  # kg*m^2

#: Time constant of a generated cable tendon's length limit, in MuJoCo
#: timesteps: ``solreflimit = (DEFAULT_TENDON_LIMIT_TIMESTEPS · h, 1)`` with ``h``
#: the timestep. Two timesteps is the stiffest limit MuJoCo integrates stably,
#: so the cable holds its length.
DEFAULT_TENDON_LIMIT_TIMESTEPS = 2.0

#: Smallest along-cable thrust the M3 payload cascade commands, as a fraction of
#: each cable's hover tension (its share ``‖T̄_i‖`` of the payload weight under
#: the tension-split policy). The thrust component along cable ``i`` never falls
#: below it, so a payload loop that asks for a net downward force keeps the
#: cable taut and the commanded attitude upright.
MULTI_MIN_CABLE_TENSION_FRACTION = 0.1

#: Largest angle from vertical of the thrust vector the M3 payload cascade
#: hands each attitude loop, in degrees (``MultiQuadCsPayloadController``'s
#: ``max_tilt_deg``). The command's vertical part is floored at its cable's
#: tension floor and its horizontal part scaled down, where needed, to this angle.
MULTI_MAX_COMMANDED_TILT_DEG = 50.0

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

# Payload controller gains (Sreenath 2013): fixed payload-loop (k_x, k_v) and
# cable-loop (k_q, k_ω) gains for QuadCSPayloadController, which takes them
# through its position_kp, position_kd, cable_kp and cable_kd keywords. They do
# not depend on the machine. On the default M2 machine (DEFAULT_M1_QUAD_MASS,
# DEFAULT_M1_QUAD_INERTIA, DEFAULT_M2_PAYLOAD_MASS) with the cascade's own
# attitude gains, the closed loop at hover is stable for cable lengths
# 0.25 m ≤ ℓ ≤ 1.76 m and unstable outside it. The controller's default gains
# are the time-scale design below.

#: Payload position gain ``k_x``, in N/m.
DEFAULT_PAYLOAD_POS_KP = np.array([4.0, 4.0, 8.0])
#: Payload velocity gain ``k_v``, in N·s/m.
DEFAULT_PAYLOAD_POS_KD = np.array([3.0, 3.0, 6.0])
#: Cable direction gain ``k_q``, in N.
DEFAULT_PAYLOAD_CABLE_KP = np.array([24.0, 24.0, 24.0])
#: Cable rate gain ``k_ω``, in N·s.
DEFAULT_PAYLOAD_CABLE_KD = np.array([8.0, 8.0, 8.0])

# Cable-suspended payload cascade gains (M2): a time-scale design.
#
# The cascade has three nested loops, attitude ≫ cable ≫ payload, and each is
# placed at its own natural frequency so that each inner loop has settled on the
# time scale its outer loop acts on:
#
#   attitude  ω_a = CABLE_CASCADE_ATTITUDE_BANDWIDTH
#   cable     ω_c = min(CABLE_CASCADE_PENDULUM_FACTOR · √(g/ℓ), ω_a / r)
#   payload   ω_p = ω_c / r                    (lateral)
#   vertical  ω_z = ω_a / r²                   (payload z)
#
# with r = CABLE_CASCADE_LOOP_RATIO and damping ζ = CABLE_CASCADE_DAMPING on
# every loop. The cable loop follows the pendulum frequency √(g/ℓ), so the
# gains scale with the cable length rather than holding one length's tuning;
# the cap ω_a / r keeps it separated from the attitude loop on short cables.
# `udaan.control.quadrotor_cspayload.cable_cascade_gains` turns these into
# gains from m_Q, m_L, ℓ and J.

#: Attitude-loop natural frequency ``ω_a`` of the cable cascade, in rad/s.
CABLE_CASCADE_ATTITUDE_BANDWIDTH = 40.0
#: Frequency ratio ``r`` between neighbouring loops of the cable cascade.
CABLE_CASCADE_LOOP_RATIO = 5.0
#: Cable-loop frequency as a multiple ``c`` of the pendulum frequency ``√(g/ℓ)``.
CABLE_CASCADE_PENDULUM_FACTOR = 2.0
#: Damping ratio ``ζ`` of every loop of the cable cascade.
CABLE_CASCADE_DAMPING = 0.7

#: Smallest vertical component ``b₃·e₃`` of a flatness reference's thrust
#: direction for which the airframe counts as upright. A reference sample at or
#: below it needs the airframe level or inverted; see
#: :mod:`udaan.flatness.feasibility`.
MIN_UPRIGHT_THRUST_Z = 0.0

# Multi-quadrotor cooperative payload gains (M3) follow the same time-scale
# design, CABLE_CASCADE_* above, in **acceleration** units: the payload PD
# commands an acceleration (``k_p = ω²``, ``k_d = 2ζω``) and the cable PD carries
# its own ``m_i l_i`` (``k_q = ω_c,i²``, ``k_ω = 2ζ ω_c,i``), so no mass enters
# them and they mean the same thing at every ``nQ``. Each cable's ``ω_c,i``
# follows its own length, and the payload loop sits a ratio ``r`` below the
# slowest cable. `udaan.control.multi_quad_cspayload.multi_cable_cascade_gains`
# computes them from the model's cable lengths and airframe inertias.

# Rigid-body payload attitude loop (M4).
#
# The extra loop M4 has and M3 does not: a geometric SO(3) PD on the payload
# attitude whose output is a *moment*. It is placed in the same time-scale
# design as the payload's position loop, at a multiple of the payload's lateral
# frequency ω_p (which sits a ratio r below the slowest cable),
#
#   ω_L = RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO · ω_p,
#
# with damping CABLE_CASCADE_DAMPING, on the payload's own inertia J_L:
# k_R = J_L,ii ω_L², k_Ω = 2ζ J_L,ii ω_L per body axis, in N·m.
# `udaan.control.multi_quad_rigidbody.rigid_payload_cascade_gains` computes them.

#: Payload attitude-loop frequency ``ω_L`` as a multiple of the payload's
#: lateral position-loop frequency ``ω_p``.
RIGID_PAYLOAD_ATTITUDE_FREQUENCY_RATIO = 3.0

#: Floor on every tension the M4 payload cascade allocates, as a fraction of
#: that cable's own hover tension: the default tension policy
#: (:class:`~udaan.flatness.MinimumTensionDistribution`) moves the grasp's
#: null-space coordinate so that each cable's force along its hover direction
#: stays at or above it.
RIGID_MIN_CABLE_TENSION_FRACTION = 0.1

# =============================================================================
# Quadrotor with a payload on an n-link flexible cable (M5)
# =============================================================================

#: Number of rigid links the flexible cable is discretised into.
#:
#: Three, not more, because the chain's fastest internal mode grows roughly
#: like ``n^{3/2}`` and udaan's ZOH-Euler integrator is only *conditionally*
#: stable — see "Numerical stability" in
#: :mod:`udaan.models.quadrotor_flexcable.base`. At the shared 1.25 ms substep
#: this default leaves ~2x margin;
#: :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.natural_frequencies`
#: is what to consult before raising it, and the analytical model's constructor
#: warns when the combination of ``num_links`` and ``dt`` has left the stable
#: region. The growth exponent rises steeply with the link count at fixed cable
#: mass, so a long chain at this ``dt`` warns out of the constructor.
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
DEFAULT_CABLE_MASS = 0.1  # kg

#: MuJoCo integrator timestep written into the generated M5 MJCF, in seconds.
#:
#: It is the analytical model's *inner* substep (a 5 ms outer loop over four
#: 1.25 ms substeps), so the two backends integrate at the same rate and 5 ms
#: divides by it exactly: the MuJoCo backend takes ``round(dt / timestep)``
#: substeps per :meth:`step`.
DEFAULT_MJCF_TIMESTEP = 1.25e-3  # s

#: Largest step of the flat map's reference airframe acceleration, in m/s², that
#: :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.simulate`
#: accepts without a warning: a fifth of gravity, i.e. a step in the
#: feed-forward thrust of a fifth of hover. The flat map needs the payload path
#: continuous through order ``2n + 4``; a trajectory smooth to a lower order
#: steps the reference where the discontinuity is, and a long cable magnifies
#: the step.
MAX_FLEXCABLE_REFERENCE_ACCEL_STEP = 0.2 * GRAVITY  # m/s^2

# Flexible-cable payload LQR (M5).
#
# M5's published law is a linear-quadratic regulator rather than a geometric
# cascade, so its "gains" are the weights of the cost
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
# (``quadrotor-flexible-cables-payload/Control/sQnlink/sQ_nlink_control.m``,
# ``odefun_riccati``: ``Q1 = blkdiag(0.5 I6, 0.75 I6, I_3n, 0.75 I_3n)``)
# unchanged.
#
# ``R`` is **not**.  The reference weights all four inputs equally at
# ``0.2 I4``, which ignores that the moment channel's input matrix is ``J⁻¹``
# — spectral norm 207 for the default (generated) airframe — while the thrust
# channel's is ``1/M₀₀``, order 1.  Equal weights therefore buy an attitude
# loop a few hundred times faster than the thrust loop: its fastest
# closed-loop pole lies at −400 rad/s, against a 200 Hz outer loop.
#
# The cost of that pole is **sampled-data phase loss**, not discrete-time
# divergence: the *command* is frozen across the whole 5 ms hold, and the
# hold's half-sample delay ``ω Δt / 2`` is about 1 rad of phase lag at
# −400 rad/s.
#
# Weighting the moments at 20 brings the fastest pole to −40 rad/s (0.1 rad
# of hold lag).  Being an *input* weight it leaves the chain modes nearly
# alone, but "nearly" is n-dependent, and from n = 2 up it is in the *worse*
# direction: the slowest closed-loop real part moves 9.1 % at n = 2, 22 % at
# n = 3 and 43 % at n = 5 (pinned in
# ``tests/test_quadrotor_flexcable_control.py``).  See the
# damping-versus-tracking trade-off in
# :mod:`udaan.control.quadrotor_flexcable.payload_controller`.
DEFAULT_FLEXCABLE_LQR_Q_POSITION = 0.5
DEFAULT_FLEXCABLE_LQR_Q_VELOCITY = 0.5
DEFAULT_FLEXCABLE_LQR_Q_ATTITUDE = 0.75
DEFAULT_FLEXCABLE_LQR_Q_ANGULAR_RATE = 0.75
DEFAULT_FLEXCABLE_LQR_Q_LINK = 1.0
DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE = 0.75
DEFAULT_FLEXCABLE_LQR_R_THRUST = 0.2
DEFAULT_FLEXCABLE_LQR_R_MOMENT = 20.0

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

#: Radius of each hose vertex's mass sphere in the generated M6 MJCF, in metres.
#: The sphere carries ``2/5 m r^2`` of rotational inertia that the analytical
#: model, whose vertices are point masses, does not have.
DEFAULT_HOSE_VERTEX_RADIUS = 0.02  # m

#: Radius of the (massless) link cylinders in the generated M6 MJCF, in metres
#: (geometry only).
DEFAULT_HOSE_LINK_RADIUS = 0.008  # m

#: Smallest tension, in N, the M6 flat map and controller accept on a hose link
#: before refusing the reference as slack. A link tension is a norm, so it never
#: goes negative; this floor keeps the refusal away from the slack boundary,
#: where the link direction is undefined.
DEFAULT_HOSE_MIN_TENSION = 1e-6  # N

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
# ``||J^-1||``.  The largest moment column is ``||J^-1|| = 207`` (the generated
# airframe) whatever the hose, while the thrust columns run 0.435 to 2.83 over
# the attachment patterns tabulated in that module, so the ratio is 70 to 500
# rather than one number.  The resulting attitude pole lies near -400 rad/s,
# about 57° of half-sample lag at udaan's 200 Hz zero-order hold.
# See :mod:`udaan.control.multi_quad_hose.hose_controller` for what the
# moment weight costs the hose modes in return.
DEFAULT_HOSE_LQR_R_MOMENT = 20.0

#: Largest step of an airframe's reference acceleration, in m/s², that
#: :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.simulate` accepts
#: without a warning: M5's threshold, a fifth of gravity. The M6 flat map needs
#: the root path continuous through order ``2n + 4``; a trajectory smooth to a
#: lower order steps every airframe's reference where the discontinuity is.
MAX_HOSE_REFERENCE_ACCEL_STEP = MAX_FLEXCABLE_REFERENCE_ACCEL_STEP  # m/s^2

# ── M7 / M8: quadrotors on a hose pinned to a ground anchor ────────────
#
# M7 is one quadrotor at the top of an n-link tether; M8 is several quadrotors
# in series on one hose.  Both are ``udaan.models.tethered_hose``, M7 being the
# one-vehicle case, and both use M6's hose LQR weights without the two root
# blocks (the pinned vertex has no position or velocity to weight).

#: M7's tether length, in metres.
DEFAULT_TETHER_LENGTH = 2.0  # m

#: Links M7's tether is split into.  At the default station the analytical
#: model's ZOH-Euler loop decays at four links (−0.51 1/s) and six (−0.13 1/s)
#: and grows at eight (+0.25 1/s), although the continuous loop decays.  Four
#: keeps the most margin of the even link counts that decay (see
#: :meth:`~udaan.models.tethered_hose.TetheredHoseBase.stability_growth_rate`).
DEFAULT_TETHER_NUM_LINKS = 4

#: Mass per metre of the tether and of M8's hose, in kg/m: M6's hose
#: (0.5 kg over 2 m), so M7 is M8 with one vehicle.
DEFAULT_TETHER_LINEAR_DENSITY = 0.25  # kg/m

#: M8's hose length, in metres.
DEFAULT_TETHERED_HOSE_LENGTH = 3.0  # m

#: Vehicles on M8's hose.
DEFAULT_TETHERED_HOSE_NUM_QUADROTORS = 3

#: Links between consecutive M8 vehicles (and between the anchor and the
#: first).  Two keeps every segment a foldable chain (a one-link segment has a
#: static solution only when its two ends are exactly one link apart) and keeps
#: the analytical loop's sampled closed loop decaying at the default M8 machine.
DEFAULT_TETHERED_HOSE_LINKS_PER_SEGMENT = 2

#: Height of the anchor above the ground plane ``z = 0``, in metres.
DEFAULT_TETHER_ANCHOR_HEIGHT = 0.0  # m

#: Default station of the top vehicle: horizontal distance from the anchor and
#: height above it, as fractions of the hose length.  Inside the band where the
#: hose stays above the anchor's height (the anchor link pulls upwards).
DEFAULT_TETHER_STANDOFF_RATIO = 0.5
DEFAULT_TETHER_HEIGHT_RATIO = 0.8

#: Ramp of ``udaan run tethered-quad/tethered-hose --traj circle`` and blend of
#: ``--traj climb``, in seconds.  A long hose needs longer: the flat map reads
#: the anchor force through order ``2n + 2``, so pre-flight the reference.
DEFAULT_TETHER_RAMP_TIME = 6.0  # s

#: Fraction of an airframe's thrust limit the steady circle about the anchor
#: may ask of any vehicle (:func:`~udaan.flatness.tethered_hose.circle_rate_bound`):
#: 0.95 leaves 1 N of the 20 N for the controller's corrections.
TETHER_CIRCLE_THRUST_FRACTION = 0.95

#: Largest growth rate, in 1/s, of the analytical model's sampled closed loop
#: (the 5 ms zero-order hold over the forward-Euler substeps, four by default, with the LQR, at
#: the trim) that the constructor accepts without a warning.  Zero: above it a
#: perturbation grows.
TETHER_MAX_CLOSED_LOOP_GROWTH_RATE = 0.0  # 1/s
