import numpy as np
from ..base import BaseModel
from ... import utils
from scipy.linalg import expm
import time
import enum



class Quadrotor(BaseModel):
    class INPUT_TYPE(enum.Enum):
        CMD_WRENCH = 0      # thrust [N] (scalar), torque [Nm] (3x1)
        CMD_PROP_FORCES = 1 # propeller forces [N] (4x1)
        

    class State(object):
        def __init__(self):
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            self.orientation = np.eye(3)
            self.angular_velocity = np.zeros(3)
            return
        
        def reset(self):
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            self.orientation = np.eye(3)
            self.angular_velocity = np.zeros(3)
            return

    def __init__(self, **kwargs):
        super().__init__()
        self.state = Quadrotor.State()

        # system parameters
        self.mass = 0.9 # kg
        self._inertia =np.array([[0.0023, 0., 0.], [0., 0.0023, 0.],
                                    [0., 0., 0.004]]) # kg m^2
        self._inertia_inv = np.linalg.inv(self._inertia)
        self._min_thrust = 0.0
        self._max_thrust = 20.0
        self._min_torque = np.array([-5., -5., -2.])
        self._max_torque = np.array([5., 5., 2.])

        self._render = False
        self._input_type = Quadrotor.INPUT_TYPE.CMD_WRENCH

        self._parse_args(**kwargs)
        if "timestep" in kwargs.keys():
            self._timestep = kwargs["timestep"]
        if "render" in kwargs.keys():
            self._render = kwargs["render"]
        if "input" in kwargs.keys():
            if kwargs["input"] == "prop_forces":
                self._input_type = Quadrotor.INPUT_TYPE.CMD_PROP_FORCES
            else:
                self._input_type = Quadrotor.INPUT_TYPE.CMD_WRENCH        
        

        return
    
    @property
    def inertia(self):
        return self._inertia
    @inertia.setter
    def inertia(self, inertia):
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)
        return

    def _zoh(self, thrust, torque):
        accel = -self._ge3 + thrust * self.state.orientation@self._e3
        self.state.position += self.state.velocity * self._timestep + 0.5 * accel * self._timestep**2
        self.state.velocity += accel * self._timestep
        self.state.orientation = self.state.orientation @ expm(utils.hat(torque) * self._timestep)
        ang_vel_dot = self._inertia_inv @ (torque - np.cross(self.state.angular_velocity, self._inertia @ self.state.angular_velocity))
        self.state.angular_velocity += ang_vel_dot * self._timestep
        return

    def _actuation_params(self):
        self._force_constant = 4.104890333e-6
        self._torque_constant = 1.026e-07
        self._force2torque_const = self._torque_constant/self._force_constant

    def _compute_allocation_matrix(self):
        self._actuation_params()
        """
         (1)CW    CCW(0)           y^
              \_^_/                 |
               |_|                  |
              /   \                 |
        (2)CCW     CW(3)           z.------> x
        """
        l = 0.175  # arm length
        ang = [np.pi/4.0, 3*np.pi/4.0, 5*np.pi/4.0, 7*np.pi/4.0]
        d = [-1., 1., -1., 1.]

        self._allocation_matrix = np.zeros((4, 4))
        for i in range(4):
            self._allocation_matrix[0, i] = 1.0
            self._allocation_matrix[1, i] = l * np.sin(ang[i])
            self._allocation_matrix[2, i] = -l * np.cos(ang[i])
            self._allocation_matrix[3, i] = self._force2torque_const*d[i]

        self._allocation_inv = np.linalg.pinv(self._allocation_matrix)
        print(self._allocation_matrix)
        print(self._allocation_inv)
        return

    def _wrench_to_propforces(self, wrench):
        """wrench is 4x1, (f, Mx, My, Mz)"""
        f = self._allocation_inv@wrench
        f = np.clip(f, np.array([1., 1., 1., 1.]),
                    np.array([10., 10., 10., 10.]))
        # TODO another level of abstraction for 1st order motor dynamics
        return f

    def _parse_input(self, input):
          return thrust, torque

    def _step(self, input):
          # integrate dynamics
          thrust, torque = self._parse_input(input)
          self._zoh(thrust, torque)
          self.t += self._timestep

    def simulate(self, x0, tf):
        self.reset(x0)

        start_t = time.time_ns()
        while self.t < tf:
            input = self._controller.compute_input()
            self._step(input)
        
        end_t = time.time_ns()
        time_taken += (end_t - start_t)*1e-9
        print("Took (%.4f)s for simulating (%.4f)s" % (time_taken, self.t))
        return
