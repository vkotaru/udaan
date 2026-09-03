import numpy as np

from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

# base models
# from udaan.models.quadrotor import QuadrotorBase
# mdl = QuadrotorBase()
# mdl.simulate(tf=10, position=np.array([1., 1., 0.]))

# mujoco models
# from udaan.models.quadrotor import QuadrotorMujoco
# mdl = QuadrotorMujoco(render=True)
# mdl.simulate(tf=10, position=np.array([1., 1., 0.]))

# mdl = QuadrotorMujoco(render=True, force="prop_forces", input="wrench")
# mdl.simulate(tf=10, position=np.array([-1.0, 2.0, 0.0]))

# from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco
# mdl = QuadrotorCsPayloadMujoco(render=True, cable_model="tendon")
# mdl.simulate(tf=10, payload_position=np.array([-1., 2., 0.5]))

mdl = MultiQuadCsPayloadMujoco(render=True, num_quadrotors=3)
mdl.simulate(tf=10, payload_position=np.array([-1.0, 2.0, 0.5]))
