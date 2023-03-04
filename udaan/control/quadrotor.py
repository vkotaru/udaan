from ..control import Gains, Controller, PDController
import numpy as np
from ..utils import printc_fail, hat, vee


class QuadPosPD(PDController):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mass = 1.
        if "mass" in kwargs.keys():
            self.mass = kwargs["mass"]

        self._gains.kp = np.array([4.1, 4.1, 8.1])
        self._gains.kd = 1.5 * np.array([2., 2., 6.])
        return

    def compute(self, *args):
        t = args[0]
        s, ds = args[1][0], args[1][1]
        sd, dsd, d2sd = self.setpoint(t)
        e = s - sd
        de = ds - dsd
        u = -self._gains.kp * e - self._gains.kd * de + d2sd + self._ge3
        # scale acceleration to force
        u = self.mass * u
        return u


class QuadAttGeoPD(Controller):

    def __init__(self, **kwargs):
        self._gains = Gains(kp=np.array([2.4, 2.4, 1.35]),
                            kd=np.array([0.35, 0.35, 0.225]))

        self._inertia = np.eye(3)
        self._inertia_inv = np.eye(3)
        self.mass = 1.

        if "inertia" in kwargs.keys():
            self.inertia = kwargs["inertia"]
        else:
            printc_fail("Inertia not provided")
        if "mass" in kwargs.keys():
            self.mass = kwargs["mass"]
        return

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, inertia):
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)

    def _cmd_accel_to_cmd_att(self, accel):
        """command attitude
        """
        norm_accel = np.linalg.norm(accel)
        b1d = np.array([1., 0., 0.])
        b3 = accel / norm_accel
        b3_b1d = np.cross(b3, b1d)
        norm_b3_b1d = np.linalg.norm(b3_b1d)
        b1 = (-1 / norm_b3_b1d) * np.cross(b3, b3_b1d)
        b2 = np.cross(b3, b1)
        Rd = np.hstack([
            np.expand_dims(b1, axis=1),
            np.expand_dims(b2, axis=1),
            np.expand_dims(b3, axis=1)
        ])
        return Rd

    def compute(self, *args):
        t = args[0]
        R, Omega = args[1][0], args[1][1]
        thrust_force = args[2]
        Rd = self._cmd_accel_to_cmd_att(thrust_force)
        # TODO: add angular velocity feedforward
        Omegad = np.zeros(3)  # self.des_state['Omega']
        dOmegad = np.zeros(3)  # self.des_state['dOmega']

        tmp = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])  # vee-map
        eOmega = Omega - R.T @ Rd @ Omegad
        M = -self._gains.kp * eR - self._gains.kd * eOmega + np.cross(
            Omega, self.inertia @ Omega)
        M += -1 * self.inertia @ (hat(Omega) @ R.T @ Rd @ Omegad -
                                  R.T @ Rd @ dOmegad)
        f = thrust_force.dot(R[:, 2])
        return f, M


class QuadPropForceController(Controller):

    def __init__(self, **kwargs):
        super().__init__()
        self.compute_alloc_matrix()
        self._pos_controller = QuadPosPD(**kwargs)
        self._att_controller = QuadAttGeoPD(**kwargs)

    def compute_alloc_matrix(self):
        """
         (1)CW    CCW(0) [-1]      y^
              \_^_/                 |
               |_|                  |
              /   \                 |
        (2)CCW     CW(3)           z.------> x
        """
        self._force_constant = 4.104890333e-6
        self._torque_constant = 1.026e-07
        self._force2torque_const = self._torque_constant / self._force_constant

        l = 0.2  # 0.175  # arm length
        ang = [np.pi / 4.0, 3 * np.pi / 4.0, 5 * np.pi / 4.0, 7 * np.pi / 4.0]
        d = [-1., 1., -1., 1.]

        self._allocation_matrix = np.zeros((4, 4))
        for i in range(4):
            self._allocation_matrix[0, i] = 1.0
            self._allocation_matrix[1, i] = l * np.sin(ang[i])
            self._allocation_matrix[2, i] = -l * np.cos(ang[i])
            self._allocation_matrix[3, i] = self._force2torque_const * d[i]

        self._allocation_inv = np.linalg.pinv(self._allocation_matrix)

    def compute(self, *args):
        """computes the propeller forces given the current state
        Returns:
            ndarray : four propeller forces in N
        """
        t = args[0]
        thrust_force = self._pos_controller.compute(t,
                                                    (args[1][0], args[1][1]))
        f, M = self._att_controller.compute(t, (args[1][2], args[1][3]),
                                            thrust_force)
        return self._allocation_inv @ np.append(f, M)
