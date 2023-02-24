from scipy.linalg import block_diag as blkdiag
import numpy as np
import scipy as sp
import time
import enum

from . import BaseModel
from ... import control
from ... import utils


class QuadrotorCSPayload(BaseModel):
  """Quadrotor with cable suspended paylaod

  Args:
      BaseModel (_type_): _description_
  """
  class INPUT_TYPE(enum.Enum):
    CMD_WRENCH = 0  # thrust [N] (scalar), torque [Nm] (3x1) : (4x1)
    CMD_PROP_FORCES = 1  # propeller forces [N] (4x1)
    CMD_ACCEL = 2  # acceleration [m/s^2] (3x1)
    
  class State(object):
      def __init__(self, l=1):
          self.cable_length = l
          # Quadrotor state
          self.position = np.array([0., 0., 1.])
          self.velocity = np.zeros(3)
          self.orientation = np.eye(3)
          self.angular_velocity = np.zeros(3)
          # Payload state
          self.payload_position = np.zeros(3)
          self.payload_velocity = np.zeros(3)
          self.cable_attitude = np.array([0., 0., -1.])
          self.cable_ang_velocity = np.zeros(3)
          return
      
      def reset(self):
          # Quadrotor state
          self.position = np.array([0., 0., 1.])
          self.velocity = np.zeros(3)
          self.orientation = np.eye(3)
          self.angular_velocity = np.zeros(3)
          # Payload state
          self.payload_position = np.zeros(3)
          self.payload_velocity = np.zeros(3)
          self.cable_attitude = np.array([0., 0., -1.])
          self.cable_ang_velocity = np.zeros(3)
          return

  def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self.state = QuadrotorCSPayload.State()

      # system parameters
      self._mass = 0.9 # kg
      self._inertia =np.array([[0.0023, 0., 0.], [0., 0.0023, 0.],
                                  [0., 0., 0.004]]) # kg m^2
      self._inertia_inv = np.linalg.inv(self._inertia)
      self._payload_mass = 0.2 # kg
      self._cable_length = 1.0 # m   
              
      self._min_thrust = 0.5
      self._max_thrust = 20.0
      self._min_torque = np.array([-5., -5., -2.])
      self._max_torque = np.array([5., 5., 2.])
      self._prop_min_force = 0.0
      self._prop_max_force = 10.0
      self._wrench_min = np.concatenate([np.array([self._min_thrust]), self._min_torque])
      self._wrench_max = np.concatenate([np.array([self._max_thrust]), self._max_torque])
