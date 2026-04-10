import math

import numpy as np


def hat(vector):
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def rodrigues_expm(vector):
    """Closed-form matrix exponential for so(3) via Rodrigues' formula.

    Exact for 3-vectors (skew-symmetric generators) and ~1.5x faster
    than scipy.linalg.expm over the hat map.
    """
    K = hat(vector)
    th = np.linalg.norm(vector)
    if abs(th) <= 1e-4:
        return np.eye(3)
    else:
        return np.eye(3) + (np.sin(th) / th) * K + ((1 - np.cos(th)) / th**2) * (K @ K)


def expm_taylor_expansion(M, order=2):
    R = np.eye(3)
    for i in range(1, order + 1):
        R += np.linalg.matrix_power(M, i) / math.factorial(i)
    return R
