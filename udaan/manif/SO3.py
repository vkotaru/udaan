import numpy as np
from .utils import hat, vee
import scipy.linalg


class SO3(np.ndarray):

    def __new__(cls, R=np.eye(3)):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(R).view(cls)
        # add the new attribute to the created instance
        obj.R = np.array(R, dtype=float)
        # Finally, we must return the newly created object:
        return obj

    def step(
            self,
            Omega=np.zeros(3, ),
    ):
        return SO3(self @ scipy.linalg.expm(hat(Omega)))


class RotationMatrix(object):

    def __init__(self):
        self._e3 = np.array([0.0, 0.0, 1.0])
        self._e2 = np.array([0.0, 1.0, 0.0])
        self._e1 = np.array([1.0, 0.0, 0.0])
        return

    def __call__(self, b3, b1):
        b3_b1d = np.cross(b3, b1)
        norm_b3_b1d = np.linalg.norm(b3_b1d)
        b1 = (-1 / norm_b3_b1d) * np.cross(b3, b3_b1d)
        b2 = np.cross(b3, b1)
        R = np.hstack([
            np.expand_dims(b1, axis=1),
            np.expand_dims(b2, axis=1),
            np.expand_dims(b3, axis=1),
        ])
        return R

    def tilt(
            self,
            ang_vec=np.zeros(3, ),
    ):
        R = scipy.linalg.expm(hat(ang_vec))
        return R @ self._e3

    def yaw(self, theta=0.0):
        return np.array([np.cos(theta), np.sin(theta), 0.0])


def Rot2Eul(R):
    phi_val = np.arctan2(R[2, 1], R[2, 2])
    theta_val = np.arcsin(-R[2, 0])
    psi_val = np.arctan2(R[1, 0], R[0, 0])

    if abs(theta_val - np.pi / 2) < 1.0e-3:
        phi_val = 0.0
        psi_val = np.arctan2(R[1, 2], R[0, 2])
    elif abs(theta_val + np.pi / 2) < 1.0e-3:
        phi_val = 0.0
        psi_val = np.arctan2(-R[1, 2], -R[0, 2])

    return np.array([phi_val, theta_val, psi_val])
