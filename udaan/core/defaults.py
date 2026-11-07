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

# =============================================================================
# Default payload parameters
# =============================================================================

DEFAULT_PAYLOAD_MASS = 0.2  # kg
DEFAULT_CABLE_LENGTH = 1.0  # m

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
