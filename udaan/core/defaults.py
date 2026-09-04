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

DEFAULT_QUAD_MASS = 0.9  # kg
DEFAULT_QUAD_INERTIA = np.array(
    [[0.0023, 0.0, 0.0], [0.0, 0.0023, 0.0], [0.0, 0.0, 0.004]]
)  # kg*m^2
DEFAULT_ARM_LENGTH = 0.2  # m

# Motor constants
DEFAULT_FORCE_CONSTANT = 4.104890333e-6  # N/(rad/s)^2
DEFAULT_TORQUE_CONSTANT = 1.026e-07  # Nm/(rad/s)^2

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
#: combination of ``num_links`` and ``dt`` has left the stable region. Holding
#: this cable mass and raising only the link count, the growth exponent runs
#: 0.008 (n=1), 0.45 (n=3), 1.46 (n=5), 6.39 (n=10), so ``num_links=10`` warns
#: on these very defaults — expected, and the reason the default is 3.
DEFAULT_NUM_CABLE_LINKS = 3

#: Total mass of the cable itself, in kg, spread evenly over its links.
#:
#: A tenth of the payload mass. The cable *could* be lighter — a real 1 m
#: tether is nearer 10 g, which is what
#: :meth:`~udaan.utils.assets.mujoco_asset_creator.MujocoAssetCreator.create_flexible_cable_payload`
#: still defaults to — but the fastest internal mode of a lumped-mass chain
#: scales like ``sqrt(m_payload / m_link)``, so a 10 g cable at
#: ``num_links = 3`` is already three times too fast for the 1.25 ms substep.
#: Both the analytical model and :func:`udaan.utils.assets.quad_flexcable` read
#: this constant, so they always describe the same cable whatever it is set to.
#:
#: This constant is **new with M5 and changes nothing that existed before it.**
#: ``create_flexible_cable_payload`` carried ``cable_mass = 0.01`` as a
#: hardcoded local; that literal became the function's keyword default, at the
#: same value, and its one pre-existing caller (``_generate_xml`` in
#: :mod:`udaan.models.mujoco.quadrotor_cspayload_fleet`) passes none of the new
#: keywords and regenerates its ``cspayload_fleet_*`` asset byte-identically,
#: cache hash included. The 0.1 kg here is the default for M5's *own* asset and
#: analytical model only.
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

# =============================================================================
# Default control gains
# =============================================================================

# Position PD gains
DEFAULT_POS_KP = np.array([4.1, 4.1, 8.1])
DEFAULT_POS_KD = np.array([3.0, 3.0, 9.0])

# Attitude geometric controller gains (Lee 2010)
DEFAULT_ATT_KP = np.array([2.4, 2.4, 1.35])
DEFAULT_ATT_KD = np.array([0.35, 0.35, 0.225])

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
# The ``Q`` block values are the legacy MATLAB reference's
# (``phd_research/quadrotor-flexible-cables-payload/Control/sQnlink/
# sQ_nlink_control.m::odefun_riccati``, line 512,
# ``Q1 = blkdiag(0.5 I6, 0.75 I6, I_3n, 0.75 I_3n)``) carried over unchanged.
#
# ``R`` is **not**.  The reference weights all four inputs equally at
# ``0.2 I4`` (line 514), which ignores that the moment channel's input matrix
# is ``J⁻¹`` — spectral norm 435 for the default airframe — while the thrust
# channel's is ``1/M₀₀``, order 1.  Equal weights therefore buy an attitude
# loop roughly 500 times faster than the thrust loop: the fastest closed-loop
# pole lands at −842 rad/s, against a 200 Hz outer loop.
#
# What that costs, measured.  A 0.62 m payload step, tail RMS error over the
# second half of a 20 s run on the analytical model:
#
#     n           1         2         3
#     r = 0.2     4.96 m    0.37 m    1.46 m
#     r = 20      1e-5 m    1e-5 m    1.6e-5 m
#
# — i.e. at ``0.2`` the loop never settles at any link count tested.  That is
# the justification for the deviation, and it is empirical.
#
# The mechanism is **sampled-data phase loss**, not discrete-time divergence:
# neither integrator actually amplifies that pole.  Exactly held at the 5 ms
# outer step it maps to ``e^{-842·0.005} = 0.015``, and the model integrates
# with forward Euler at ``h = dt/4 = 1.25 ms``, where ``|1 − 842 h| = 0.053``.
# Both are well inside the unit circle.  What breaks the loop is that the
# *command* is frozen across the whole 5 ms hold while the closed loop wants
# to act at 842 rad/s: the hold's half-sample delay is ``ω Δt / 2 = 2.1``
# rad ≈ 121° of phase lag at the crossover the design put there, which is more
# phase margin than any of it has.  The reference's adaptive ``ode15s`` never
# meets this because it re-evaluates the control continuously.
#
# Weighting the moments at 20 moves the fastest pole to −84 rad/s (12° of hold
# lag).  Being an *input* weight it leaves the chain modes nearly alone, but
# "nearly" is n-dependent, and from n = 2 up it is always in the *worse*
# direction: the slowest closed-loop real part moves 0.02 % at n = 1
# (marginally the other way), then 1.54 % at n = 2, 4.61 % at n = 3 and
# 13.10 % at n = 5.  At n = 5 that costs more chain damping than the 60x Q
# scaling elsewhere in this system's docs buys.  It is a real price, paid
# deliberately for a loop that settles at all; see the damping-versus-tracking
# trade-off in :mod:`udaan.control.quadrotor_flexcable.payload_controller`.
DEFAULT_FLEXCABLE_LQR_Q_POSITION = 0.5
DEFAULT_FLEXCABLE_LQR_Q_VELOCITY = 0.5
DEFAULT_FLEXCABLE_LQR_Q_ATTITUDE = 0.75
DEFAULT_FLEXCABLE_LQR_Q_ANGULAR_RATE = 0.75
DEFAULT_FLEXCABLE_LQR_Q_LINK = 1.0
DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE = 0.75
DEFAULT_FLEXCABLE_LQR_R_THRUST = 0.2
DEFAULT_FLEXCABLE_LQR_R_MOMENT = 20.0
