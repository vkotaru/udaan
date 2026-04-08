import numpy as np

import udaan as U

# base models
# mdl = U.models.base.Quadrotor(render=True)
# mdl.simulate(tf=10, x0=np.array([1., 1., 0.]))

# mujoco models
# mdl = U.models.mujoco.Quadrotor(render=True)
# mdl.simulate(tf=10, position=np.array([1., 1., 0.]))

# mdl = U.models.mujoco.Quadrotor(render=True, force="prop_forces", input="wrench")
# mdl.simulate(tf=10, position=np.array([-1.0, 2.0, 0.0]))

# mdl = U.models.base.QuadrotorCSPayload(render=True)
# mdl.simulate(tf=10, position=np.array([-1., 2., 0.]))

# mdl = U.models.mujoco.QuadrotorCSPayload(render=True, model="tendon")
# mdl.simulate(tf=10, payload_position=np.array([-1., 2., 0.5]))

mdl = U.models.mujoco.MultiQuadrotorCSPointmass(render=True, num_quadrotors=3)
mdl.simulate(tf=10, payload_position=np.array([-1.0, 2.0, 0.5]))
