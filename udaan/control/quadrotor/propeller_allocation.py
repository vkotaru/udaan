"""Direct propeller force allocation controller."""

import numpy as np

from ...control import Controller
from .geometric_attitude import GeometricAttitudeController
from .position_pd import PositionPDController


class DirectPropellerForceController(Controller):
    def __init__(self, **kwargs):
        super().__init__()
        self.compute_alloc_matrix()
        self._pos_controller = PositionPDController(**kwargs)
        self._att_controller = GeometricAttitudeController(**kwargs)

    def compute_alloc_matrix(self):
        r"""Compute propeller force allocation matrix.

        Propeller layout::

             (1)CW    CCW(0) [-1]      y^
                  \_^_/                 |
                   |_|                  |
                  /   \                 |
            (2)CCW     CW(3)           z.------> x
        """
        self._force_constant = 4.104890333e-6
        self._torque_constant = 1.026e-07
        self._force2torque_const = self._torque_constant / self._force_constant

        l = 0.2
        ang = [np.pi / 4.0, 3 * np.pi / 4.0, 5 * np.pi / 4.0, 7 * np.pi / 4.0]
        d = [-1.0, 1.0, -1.0, 1.0]

        self._allocation_matrix = np.zeros((4, 4))
        for i in range(4):
            self._allocation_matrix[0, i] = 1.0
            self._allocation_matrix[1, i] = l * np.sin(ang[i])
            self._allocation_matrix[2, i] = -l * np.cos(ang[i])
            self._allocation_matrix[3, i] = self._force2torque_const * d[i]

        self._allocation_inv = np.linalg.pinv(self._allocation_matrix)

    def compute(self, *args):
        """Compute propeller forces given current state.

        Returns:
            ndarray: four propeller forces in N
        """
        t = args[0]
        thrust_force = self._pos_controller.compute(t, (args[1][0], args[1][1]))
        f, M = self._att_controller.compute(t, (args[1][2], args[1][3]), thrust_force)
        return self._allocation_inv @ np.append(f, M)
