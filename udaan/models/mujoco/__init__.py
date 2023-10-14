import os
import numpy as np

try:
    import mujoco
except ImportError as e:
    raise ImportError(e)
try:
    import mujoco_viewer
except ImportError as e:
    raise ImportError(e)

from ... import _FOLDER_PATH

DEFAULT_SIZE = 480


class MujocoModel(object):
    def __init__(self, model_path, render=False):
        self.full_path = os.path.join(
            _FOLDER_PATH, "udaan", "models", "assets", "mjcf", model_path
        )
        if not os.path.exists(self.full_path):
            raise OSError(f"File {self.full_path} does not exist")

        self.render = render
        self.viewer = None

        self.frame_skip = 1
        self._initialize_simulation()

        self.viewer_setup()
        return

    def _initialize_simulation(self):
        print("Loading model from {}".format(self.full_path))
        self.model = mujoco.MjModel.from_xml_path(self.full_path)
        self.data = mujoco.MjData(self.model)

        # mujoco py not implemented
        # following for future reference
        # ---------------------------------
        # mj_path = mujoco_py.utils.discover_mujoco()
        # # xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
        # model = mujoco_py.load_model_from_path(self.full_path)
        # sim = mujoco_py.MjSim(model)
        # sim.forward()

        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        return

    def _step_mujoco_simulation(self, n_frames=1):
        mujoco.mj_step(self.model, self.data, n_frames)
        if self.render:
            self.viewer.render()
            self.viewer.add_marker(
                pos=[0, 0, 0],
                size=[0.025, 0.025, 0.025],
                rgba=[0, 0, 0.1, 0.8],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="origin",
            )
            # try:
            #     self.viewer.render()
            # except Exception as e:
            #     print(e)
        return

    def _quat2rot(self, q):
        return np.array(
            [
                [
                    2 * (q[0] * q[0] + q[1] * q[1]) - 1,
                    2 * (q[1] * q[2] - q[0] * q[3]),
                    2 * (q[1] * q[3] + q[0] * q[2]),
                ],
                [
                    2 * (q[1] * q[2] + q[0] * q[3]),
                    2 * (q[0] * q[0] + q[2] * q[2]) - 1,
                    2 * (q[2] * q[3] - q[0] * q[1]),
                ],
                [
                    2 * (q[1] * q[3] - q[0] * q[2]),
                    2 * (q[2] * q[3] + q[0] * q[1]),
                    2 * (q[0] * q[0] + q[3] * q[3]) - 1,
                ],
            ]
        )

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return

    def viewer_setup(self):
        if self.render:
            self.viewer.cam.trackbodyid = 0  # id of the body to track ()
            self.viewer.cam.distance = (
                self.model.stat.extent * 2.0
            )  # how much you "zoom in", model.stat.extent is
            # the max limits of the arena
            self.viewer.cam.lookat[
                0
            ] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
            self.viewer.cam.lookat[1] += 0.5
            self.viewer.cam.lookat[2] += 0.5
            self.viewer.cam.elevation = (
                -40
            )  # camera rotation around the axis in the plane going through the frame
            # origin (if 0 you just see a line)
            self.viewer.cam.azimuth = (
                0  # camera rotation around the camera's vertical axis
            )

    def add_marker_at(self, p, size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 0.75], label=""):
        if self.render:
            self.viewer.add_marker(
                pos=p,
                size=size,
                rgba=rgba,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                label=label,
            )
        return

    def add_arrow_at(self, p, R, s, label="", color=None):
        if color is None:
            color = [1, 0, 0, 0.75]
        if self.render:
            self.viewer.add_marker(
                pos=p,
                mat=R,
                size=s,
                rgba=color,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                label=label,
            )
        return


from .quadrotor import Quadrotor
from .quadrotor_cspayload import QuadrotorCSPayload
from .multi_quad_cs_pointmass import MultiQuadrotorCSPointmass
from .quadrotor_comparison import QuadrotorComparison