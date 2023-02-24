import numpy as np
from ..base import BaseModel
from ... import control
from ... import utils
from scipy.linalg import expm
import time
import enum



class Quadrotor(BaseModel):
    class INPUT_TYPE(enum.Enum):
        CMD_WRENCH = 0      # thrust [N] (scalar), torque [Nm] (3x1) : (4x1)
        CMD_PROP_FORCES = 1 # propeller forces [N] (4x1)
        CMD_ACCEL = 2       # acceleration [m/s^2] (3x1)
        
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
        super().__init__(**kwargs)
        self.state = Quadrotor.State()

        # system parameters
        self._mass = 0.9 # kg
        self._inertia =np.array([[0.0023, 0., 0.], [0., 0.0023, 0.],
                                    [0., 0., 0.004]]) # kg m^2
        self._inertia_inv = np.linalg.inv(self._inertia)
        
        self._min_thrust = 0.5
        self._max_thrust = 20.0
        self._min_torque = np.array([-5., -5., -2.])
        self._max_torque = np.array([5., 5., 2.])
        self._prop_min_force = 0.0
        self._prop_max_force = 10.0
        self._wrench_min = np.concatenate([np.array([self._min_thrust]), self._min_torque])
        self._wrench_max = np.concatenate([np.array([self._max_thrust]), self._max_torque])

        self._input_type = Quadrotor.INPUT_TYPE.CMD_ACCEL
        self._n_action = 4  
        self._step_freq = 500.
        self._step_iter = max(1, int(1. / self._step_freq / self.sim_timestep))
        
        self._parse_args(**kwargs)
        if "sim_timestep" in kwargs.keys():
            self.sim_timestep = kwargs["sim_timestep"]
        if "input" in kwargs.keys():
            if kwargs["input"] == "prop_forces":
                self._input_type = Quadrotor.INPUT_TYPE.CMD_PROP_FORCES
                self._n_action = 4
                self._step_freq = 500.
                self._step_iter = max(1, int(1. / self._step_freq / self.sim_timestep))
            elif kwargs["input"] == "accel":
                self._input_type = Quadrotor.INPUT_TYPE.CMD_ACCEL
                self._n_action = 3
                self._step_freq = 100.
                self._step_iter = max(1, int(1. / self._step_freq / self.sim_timestep))
            else:
                self._input_type = Quadrotor.INPUT_TYPE.CMD_WRENCH
                self._n_action = 4  
                self._step_freq = 500.
                self._step_iter = max(1, int(1. / self._step_freq / self.sim_timestep))
                
        self._compute_allocation_matrix()
        self._init_default_controllers()
        return
      
    def _init_default_controllers(self):
        self._att_controller = control.QuadAttGeoPD(inertia=self.inertia)
        self._pos_controller = control.QuadPosPD(mass=self.mass)
        self._prop_controller = control.QuadPropForceController(mass=self.mass, inertia=self.inertia)
        return
    
    @property
    def mass(self):
          return self._mass
    @mass.setter
    def mass(self, mass):
        self._mass = mass
    
    @property
    def inertia(self):
        return self._inertia
    @inertia.setter
    def inertia(self, inertia):
        if inertia.ndim == 1:
            inertia = np.diag(inertia)
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)
        return

    def _zoh(self, thrust, torque):
        accel = -self._ge3 + thrust * self.state.orientation@self._e3
        self.state.position += self.state.velocity * self.sim_timestep + 0.5 * accel * self.sim_timestep**2
        self.state.velocity += accel * self.sim_timestep
        self.state.orientation = self.state.orientation @ expm(utils.hat(torque) * self.sim_timestep)
        ang_vel_dot = self._inertia_inv @ (torque - np.cross(self.state.angular_velocity, self._inertia @ self.state.angular_velocity))
        self.state.angular_velocity += ang_vel_dot * self.sim_timestep
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
        l = 0.2 # 0.175  # arm length
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

    def _propforces_to_wrench(self, prop_forces):
        """prop_forces is 4x1, (f1, f2, f3, f4)"""
        wrench = self._allocation_matrix@prop_forces
        return wrench


    def _parse_input(self, input):
        if self._input_type == Quadrotor.INPUT_TYPE.CMD_ACCEL:
            thrust, torque  = self._att_controller.compute(self.t, (self.state.orientation, self.state.angular_velocity), input)
        elif self._input_type == Quadrotor.INPUT_TYPE.CMD_PROP_FORCES:
            utils.printc_warn("TODO: Incorrect implementation verify")
            wrench = self._propforces_to_wrench(input)
            thrust, torque = wrench[0], wrench[1:]
        else:
            thrust, torque = input[0], input[1:]
        
        thrust = np.clip(thrust, self._min_thrust, self._max_thrust)
        torque = np.clip(torque, -self._max_torque, self._max_torque)
        return thrust, torque

    def step(self, input:np.ndarray):
        """Zero-order hold on the system equations of motion
        
        :param input: input to the quadrotor
        :type input: np.ndarray

        :return: None
        """
        for _ in range(self._step_iter):
            # integrate dynamics
            thrust, torque = self._parse_input(input)
            # physical input limits
            thrust = np.clip(thrust, self._min_thrust, self._max_thrust)
            torque = np.clip(torque, self._min_torque, self._max_torque)
            # dynamics zero-order hold integration (Euler integration)
            self._zoh(thrust, torque)
            # update time
            self.t += self.sim_timestep
        return

    def reset(self, **kwargs):
        self.t = 0.
        self.state.reset()

        k = ["position", "velocity", "orientation", "angular_velocity"]
        for key in k:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])
        return

    def simulate(self, tf, **kwargs):
        self.reset(**kwargs)

        start_t = time.time_ns()
        while self.t < tf:
            if self._input_type == Quadrotor.INPUT_TYPE.CMD_PROP_FORCES:
                u = self._prop_controller.compute(self.t, (self.state.position, self.state.velocity, self.state.orientation, self.state.angular_velocity))
            else:
                u = self._pos_controller.compute(self.t, (self.state.position, self.state.velocity))
            self.step(u)
        
        end_t = time.time_ns()
        time_taken = (end_t - start_t)*1e-9
        print("Took (%.4f)s for simulating (%.4f)s" % (time_taken, self.t))
        return
