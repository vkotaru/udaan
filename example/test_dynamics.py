import udaan as U
import numpy as np

# base models
# mdl = U.models.base.Quadrotor(render=True)
# mdl.simulate(tf=10, x0=np.array([1., 1., 0.]))

# mujoco models
# mdl = U.models.mujoco.Quadrotor(render=True)
# mdl.simulate(tf=10, x0=np.array([1., 1., 0.]))

# mdl = U.models.mujoco.Quadrotor(render=True, input="prop_forces")
# mdl.simulate(tf=10, position=np.array([-1., 2., 0.]))

mdl = U.models.base.QuadrotorCSPayload(render=True)
mdl.simulate(tf=10, position=np.array([-1., 2., 0.]))
