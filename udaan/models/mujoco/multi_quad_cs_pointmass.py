import numpy as np
from scipy.linalg import expm
import copy
import enum

from ..mujoco import MujocoModel, mujoco
from ..base import BaseModel
from ... import utils


class MultiQuadrotorCSPointmass(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    class INPUT_TYPE(enum.Enum):
        CMD_WRENCH = 0  # thrust [N] (scalar), torque [Nm] (3x1) : (4x1)
        CMD_PROP_FORCES = 1  # propeller forces [N] (4x1)
        CMD_ACCEL = 2  # acceleration [m/s^2] (3x1)

    class State(object):
        class Quadrotor(object):
            def __init__(self, **kwargs):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            def reset(self):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                return

        class Cable(object):
            def __init__(self, **kwargs):
                self.q = np.array([0.0, 0.0, -1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = 1.0
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            def reset(self):
                self.q = np.array([0.0, 0.0, -1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = 1.0
                return

        def __init__(self, nQ: int, **kwargs):
            self.num_quadrotors = nQ
            self.quads = [self.QuadrotorState() for _ in range(nQ)]
            self.cables = [self.CableState() for _ in range(nQ)]
            self.load_position = np.zeros(3)
            self.load_velocity = np.zeros(3)
            return

        def reset(self):
            for quad in self.quads:
                quad.reset()
            self.load_position = np.zeros(3)
            self.load_velocity = np.zeros(3)
            return
