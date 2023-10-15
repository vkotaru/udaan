import numpy as np
from .utils import hat, vee
import scipy.linalg


class S2(np.ndarray):

    def __new__(cls, q=np.array([0.0, 0.0, 1.0])):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(q).view(cls)
        # add the new attribute to the created instance
        obj.q = np.array(q, dtype=float)
        # Finally, we must return the newly created object:
        return obj

    def step(
            self,
            omega=np.zeros(3, ),
    ):
        return S2(scipy.linalg.expm(hat(omega)) @ self)

    def config_error(self, other):
        return 1 - np.dot(self, other)

    def error_vec(self, other, version=2):
        if version == 2:
            return hat(self) @ hat(self) @ other
        else:
            return np.cross(other, self)

    @staticmethod
    def fromEuler(phi=0.0, th=0.0, psi=0.0):
        return np.array(
            [np.cos(phi) * np.sin(th),
             np.sin(phi) * np.sin(th),
             np.cos(th)])
