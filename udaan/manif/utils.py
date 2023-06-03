import numpy as np
# import casadi as ca
import math

def hat(vector):
    return np.array([[0., -vector[2], vector[1]], 
                        [vector[2], 0., -vector[0]],
                        [-vector[1], vector[0], 0.]])

def vee(matrix):
    return np.array([matrix[2,1], matrix[0,2], matrix[1,0]])

def rodriguesExpm(vector):
    K = hat(vector)
    th = np.linalg.norm(vector)
    if abs(th) <= 1e-4:
        return np.eye(3)
    else:
        return np.eye(3) + K*np.sin(th) + (1-np.cos(th))*K@K

def expmTaylorExpansion(M, order=2):
    R = np.eye(3)
    for i in range(1, order+1):
      R += np.power(M, i)/math.factorial(i)
      
# def ca_expmTaylorExpansion(M, order=2):
#     R = ca.DM.eye(3)
#     for i in range(1, order+1):
#       R += ca.mpower(M, i)/math.factorial(i)
#     return R

# def casadi_hat(x):
#     M = ca.vertcat(ca.horzcat(0, -x[2], x[1]),
#                     ca.horzcat(x[2], 0, -x[0]),
#                     ca.horzcat(-x[1], x[0], 0))
#     return M
