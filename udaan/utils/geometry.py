import numpy as np
import math
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as sp_rot


def hat(vector):
    return np.array([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def rodriguesExpm(vector):
    K = hat(vector)
    th = np.linalg.norm(vector)
    if abs(th) <= 1e-4:
        return np.eye(3)
    else:
        return np.eye(3) + K * np.sin(th) + (1 - np.cos(th)) * K @ K


def expmTaylorExpansion(M, order=2):
    R = np.eye(3)
    for i in range(1, order + 1):
        R += np.power(M, i) / math.factorial(i)
    return R
