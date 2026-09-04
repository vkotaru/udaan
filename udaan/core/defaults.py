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
