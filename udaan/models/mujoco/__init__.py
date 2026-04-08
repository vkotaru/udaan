"""MuJoCo physics backend for Udaan.

Requires mujoco package.
Install with: pip install udaan
"""

import os
import time

import mujoco
import numpy as np

from ... import _FOLDER_PATH
from ...utils.logging import get_logger

_logger = get_logger(__name__)


class _GlfwViewer:
    """Lightweight GLFW-based MuJoCo viewer (no third-party viewer dependency)."""

    def __init__(self, model, data, width=1200, height=900, title="udaan"):
        import glfw

        self._model = model
        self._data = data

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self._camera = mujoco.MjvCamera()
        self._option = mujoco.MjvOption()

        mujoco.mjv_defaultCamera(self._camera)
        mujoco.mjv_defaultOption(self._option)

        self._camera.trackbodyid = 0
        self._camera.distance = model.stat.extent * 2.0
        self._camera.lookat[0] += 0.5
        self._camera.lookat[1] += 0.5
        self._camera.lookat[2] += 0.5
        self._camera.elevation = -40
        self._camera.azimuth = 0

        self._last_render_time = 0.0

    @property
    def cam(self):
        return self._camera

    def render(self):
        import glfw

        if glfw.window_should_close(self._window):
            return

        # Throttle to ~60fps
        now = time.monotonic()
        if now - self._last_render_time < 1.0 / 60.0:
            return
        self._last_render_time = now

        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self._window))
        mujoco.mjv_updateScene(
            self._model, self._data, self._option, None, self._camera,
            mujoco.mjtCatBit.mjCAT_ALL, self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)
        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def close(self):
        import glfw

        if self._window:
            glfw.destroy_window(self._window)
            glfw.terminate()
            self._window = None

    def is_alive(self):
        import glfw

        return self._window is not None and not glfw.window_should_close(self._window)


class MujocoModel:
    def __init__(self, model_path, render=False):
        self.full_path = os.path.join(_FOLDER_PATH, "udaan", "models", "assets", "mjcf", model_path)
        if not os.path.exists(self.full_path):
            raise OSError(f"File {self.full_path} does not exist")

        self.render = render
        self._viewer = None

        self.frame_skip = 1
        self._initialize_simulation()

    def _initialize_simulation(self):
        _logger.info("Loading model from %s", self.full_path)
        self.model = mujoco.MjModel.from_xml_path(self.full_path)
        self.data = mujoco.MjData(self.model)
        self._wall_start = None

        if self.render:
            self._viewer = _GlfwViewer(self.model, self.data)

    def _step_mujoco_simulation(self, n_frames=1):
        mujoco.mj_step(self.model, self.data, n_frames)
        if self.render and self._viewer is not None:
            # Sync simulation to real-time
            if self._wall_start is None:
                self._wall_start = time.monotonic()
            sim_time = self.data.time
            wall_elapsed = time.monotonic() - self._wall_start
            sleep_time = sim_time - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._viewer.render()

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
        self._wall_start = None

    def add_marker_at(self, p, size=None, rgba=None, label=""):
        """Visual markers not yet supported with built-in viewer."""
        pass

    def add_arrow_at(self, p, R, s, label="", color=None):
        """Visual markers not yet supported with built-in viewer."""
        pass


from .multi_quad_cs_pointmass import MultiQuadrotorCSPointmass as MultiQuadrotorCSPointmass
from .multi_quad_rigidbody import MultiQuadRigidbody as MultiQuadRigidbody
from .quadrotor import Quadrotor as Quadrotor
from .quadrotor_comparison import QuadrotorComparison as QuadrotorComparison
from .quadrotor_cspayload import QuadrotorCSPayload as QuadrotorCSPayload
