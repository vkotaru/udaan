"""Clean flat-output-to-state mapping for quadrotor differential flatness.

Given position derivatives up to snap (4th order), computes the desired
attitude, angular velocity, and angular acceleration using the differential
flatness property of the quadrotor (Mellinger & Kumar, ICRA 2011).
"""

import numpy as np

from ..core.defaults import GRAVITY
from ..manif import SO3, TSO3, vee


def flat2state(acc, jerk, snap, mass, inertia):
    """Compute desired attitude state from flat output derivatives.

    Args:
        acc: desired acceleration (3-vector), 2nd derivative of position.
        jerk: desired jerk (3-vector), 3rd derivative of position.
        snap: desired snap (3-vector), 4th derivative of position.
        mass: quadrotor mass (scalar).
        inertia: inertia matrix (3x3).

    Returns:
        (Rd, Omegad, dOmegad, f) where:
            Rd: desired rotation matrix (SO3)
            Omegad: desired angular velocity (TSO3)
            dOmegad: desired angular acceleration (3-vector)
            f: scalar thrust
    """
    g = GRAVITY
    e3 = np.array([0.0, 0.0, 1.0])
    b1d = np.array([1.0, 0.0, 0.0])
    db1d = np.zeros(3)

    # Thrust vector
    fb3 = mass * (acc + g * e3)
    norm_fb3 = np.linalg.norm(fb3)

    if norm_fb3 < 1e-6:
        return SO3(), TSO3(), np.zeros(3), 0.0

    f = norm_fb3
    b3 = fb3 / norm_fb3

    # Desired rotation from thrust direction + heading
    b3_b1d = np.cross(b3, b1d)
    norm_b3_b1d = np.linalg.norm(b3_b1d)
    if norm_b3_b1d < 1e-6:
        b1d = np.array([0.0, 1.0, 0.0])
        b3_b1d = np.cross(b3, b1d)
        norm_b3_b1d = np.linalg.norm(b3_b1d)
    b1 = -np.cross(b3, b3_b1d) / norm_b3_b1d
    b2 = np.cross(b3, b1)
    R = np.column_stack([b1, b2, b3])

    # First time derivative of thrust direction → angular velocity
    dfb3 = mass * jerk
    dnorm_fb3 = fb3.dot(dfb3) / norm_fb3
    db3 = (dfb3 * norm_fb3 - fb3 * dnorm_fb3) / norm_fb3**2

    db3_b1d = np.cross(db3, b1d) + np.cross(b3, db1d)
    dnorm_b3_b1d = b3_b1d.dot(db3_b1d) / norm_b3_b1d
    db1 = (-np.cross(db3, b3_b1d) - np.cross(b3, db3_b1d) - b1 * dnorm_b3_b1d) / norm_b3_b1d
    db2 = np.cross(db3, b1) + np.cross(b3, db1)
    dR = np.column_stack([db1, db2, db3])

    Omega = vee(dR @ R.T)

    # Second time derivative → angular acceleration
    d2fb3 = mass * snap
    d2norm_fb3 = (dfb3.dot(dfb3) + fb3.dot(d2fb3) - dnorm_fb3**2) / norm_fb3
    d2b3 = (
        (d2fb3 * norm_fb3 - fb3 * d2norm_fb3) * norm_fb3**2
        - (dfb3 * norm_fb3 - fb3 * dnorm_fb3) * 2 * norm_fb3 * dnorm_fb3
    ) / norm_fb3**4

    d2b1d = np.zeros(3)
    d2b3_b1d = np.cross(d2b3, b1d) + 2 * np.cross(db3, db1d) + np.cross(b3, d2b1d)
    d2norm_b3_b1d = (db3_b1d.dot(db3_b1d) + b3_b1d.dot(d2b3_b1d) - dnorm_b3_b1d**2) / norm_b3_b1d

    d2b1 = (
        -np.cross(d2b3, b3_b1d)
        - 2 * np.cross(db3, db3_b1d)
        - np.cross(b3, d2b3_b1d)
        - db1 * dnorm_b3_b1d
        - b1 * d2norm_b3_b1d
        - db1 * dnorm_b3_b1d
    ) / norm_b3_b1d
    d2b2 = np.cross(d2b3, b1) + 2 * np.cross(db3, db1) + np.cross(b3, d2b1)
    d2R = np.column_stack([d2b1, d2b2, d2b3])

    dOmega = vee(dR @ dR.T + d2R @ R.T)

    return SO3(R), TSO3(Omega), dOmega, f
