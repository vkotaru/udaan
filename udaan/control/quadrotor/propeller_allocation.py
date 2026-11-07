"""Direct propeller force allocation controller, and the quadrotor allocation matrix."""

import numpy as np

from ...control import Controller
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_ROTOR_ANGLES,
    DEFAULT_SPIN_DIRECTIONS,
    DEFAULT_TORQUE_CONSTANT,
)
from .geometric_attitude import GeometricAttitudeController
from .position_pd import PositionPDController


def quadrotor_allocation_matrix(
    arm_length=DEFAULT_ARM_LENGTH,
    rotor_angles=DEFAULT_ROTOR_ANGLES,
    spin_directions=DEFAULT_SPIN_DIRECTIONS,
    moment_ratio=DEFAULT_TORQUE_CONSTANT / DEFAULT_FORCE_CONSTANT,
):
    r"""The ``(4, n)`` matrix ``A`` taking rotor forces to the wrench ``[f, Mx, My, Mz]``.

    Rotor ``i`` sits at ``ℓ (cos θᵢ, sin θᵢ, 0)`` in the body frame, thrusts
    along ``b₃`` and spins in direction ``dᵢ = ±1``, so its column is

    .. math::

        A_{:,i} = \bigl[1,\; \ell\sin\theta_i,\; -\ell\cos\theta_i,\; c\,d_i\bigr]^\top,
        \qquad c = k_M / k_F.

    The defaults are the M1 airframe (``quadrotor_mj.xml``): ``ℓ = 0.2`` m, an X
    layout at ``π/4, 3π/4, 5π/4, 7π/4`` and ``d = (−1, 1, −1, 1)``::

             (1)CW    CCW(0) [-1]      y^
                  \_^_/                 |
                   |_|                  |
                  /   \                 |
            (2)CCW     CW(3)           z.------> x

    Args:
        arm_length: ``ℓ``, rotor distance from the centre, in m.
        rotor_angles: ``θᵢ``, each rotor's angle about ``b₃`` from ``b₁``, in rad.
        spin_directions: ``dᵢ``, the sign of each rotor's yaw moment.
        moment_ratio: ``c = k_M / k_F``, yaw moment per newton of thrust, in m.
    """
    angles = np.asarray(rotor_angles, dtype=float)
    spins = np.asarray(spin_directions, dtype=float)
    alloc = np.zeros((4, len(angles)))
    for i in range(len(angles)):
        alloc[0, i] = 1.0
        alloc[1, i] = arm_length * np.sin(angles[i])
        alloc[2, i] = -arm_length * np.cos(angles[i])
        alloc[3, i] = moment_ratio * spins[i]
    return alloc


class DirectPropellerForceController(Controller):
    """Position PD → geometric attitude → rotor forces, through an allocation matrix.

    Every keyword other than ``allocation_matrix`` goes to the inner
    :class:`~udaan.control.quadrotor.PositionPDController` and
    :class:`~udaan.control.quadrotor.GeometricAttitudeController`.

    Keyword Args:
        allocation_matrix: the ``(4, 4)`` matrix ``A`` the rotor forces are
            allocated through (forces = ``pinv(A) @ wrench``). Default:
            :func:`quadrotor_allocation_matrix` at its defaults. Pass the
            model's own (or use :meth:`from_model`) so that the controller and
            the vehicle allocate identically.
    """

    def __init__(self, allocation_matrix=None, **kwargs):
        super().__init__()
        if allocation_matrix is None:
            allocation_matrix = quadrotor_allocation_matrix()
        self._allocation_matrix = np.array(allocation_matrix, dtype=float)
        self._allocation_inv = np.linalg.pinv(self._allocation_matrix)
        self._pos_controller = PositionPDController(**kwargs)
        self._att_controller = GeometricAttitudeController(**kwargs)

    @classmethod
    def from_model(cls, model, **kwargs):
        """Build one for ``model``: its mass, inertia and allocation matrix."""
        return cls(
            allocation_matrix=model.allocation_matrix,
            mass=model.mass,
            inertia=model.inertia,
            **kwargs,
        )

    @property
    def allocation_matrix(self):
        """The ``(4, 4)`` rotor-forces → wrench matrix this controller inverts (a copy)."""
        return self._allocation_matrix.copy()

    def compute(self, *args):
        """Compute propeller forces given current state.

        Returns:
            ndarray: four propeller forces in N
        """
        t = args[0]
        # (p, v, R): R is what PositionL1Controller's predictor needs.
        thrust_force = self._pos_controller.compute(t, (args[1][0], args[1][1], args[1][2]))
        f, M = self._att_controller.compute(t, (args[1][2], args[1][3]), thrust_force)
        return self._allocation_inv @ np.append(f, M)
