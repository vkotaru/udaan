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
    """Lightweight GLFW-based MuJoCo viewer with mouse interaction."""

    def _mouse_button_callback(self, window, button, action, mods):
        import glfw

        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._button_left = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._button_right = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._button_middle = pressed

        x, y = self._glfw.get_cursor_pos(window)
        self._last_mouse_x = x
        self._last_mouse_y = y

    def _mouse_move_callback(self, window, xpos, ypos):
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos

        if not (self._button_left or self._button_right or self._button_middle):
            return

        width, height = self._glfw.get_window_size(window)

        if self._button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._button_middle:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return

        mujoco.mjv_moveCamera(
            self._model, action, dx / width, dy / height, self._scene, self._camera
        )

    def _scroll_callback(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self._scene, self._camera
        )

    def _key_callback(self, window, key, scancode, action, mods):
        import glfw

        if action == glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)

    @property
    def cam(self):
        return self._camera

    def __init__(self, model, data, width=1200, height=900, title="udaan", record=None):
        if record:
            width, height = 640, 480
        import glfw

        self._model = model
        self._data = data
        self._glfw = glfw
        self._record_path = record
        self._frames = [] if record else None
        self._record_every = 4  # capture every Nth rendered frame
        self._render_count = 0

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
        self._perturb = mujoco.MjvPerturb()

        mujoco.mjv_defaultCamera(self._camera)
        mujoco.mjv_defaultOption(self._option)
        mujoco.mjv_defaultPerturb(self._perturb)

        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = 1
        self._camera.distance = max(model.stat.extent * 1.5, 1.5)
        self._camera.elevation = -20
        self._camera.azimuth = 135

        # Mouse interaction state
        self._button_left = False
        self._button_right = False
        self._button_middle = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Register GLFW callbacks
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self._window, self._mouse_move_callback)
        glfw.set_scroll_callback(self._window, self._scroll_callback)
        glfw.set_key_callback(self._window, self._key_callback)

        self._last_render_time = 0.0
        self._overlay_text = ""

        # Visual markers: per-entity trails, starts, targets
        self.show_trails = True
        self._trails = {}  # key -> list of positions
        self._trail_colors = {}  # key -> rgba
        self._trail_max = 200
        self._starts = {}  # key -> position
        self._targets = {}  # key -> position

    def _add_geom(self, type_, size, pos, rgba, mat=np.eye(3)):
        """Add a custom geom to the scene for the current frame."""
        if self._scene.ngeom >= self._scene.maxgeom:
            return
        g = self._scene.geoms[self._scene.ngeom]
        mujoco.mjv_initGeom(g, type_, size, pos, mat.flatten(), rgba)
        self._scene.ngeom += 1

    def add_trail_point(self, pos, key=0, rgba=None):
        """Append a position to a keyed trail."""
        if key not in self._trails:
            self._trails[key] = []
            self._trail_colors[key] = (
                np.array(rgba, dtype=np.float32)
                if rgba is not None
                else np.array([0.2, 0.6, 1.0, 0.6], dtype=np.float32)
            )
        trail = self._trails[key]
        trail.append(np.array(pos, dtype=np.float64))
        if len(trail) > self._trail_max:
            self._trails[key] = trail[-self._trail_max :]

    def set_start(self, pos, key=0):
        self._starts[key] = np.array(pos, dtype=np.float64)

    def set_target(self, pos, key=0):
        self._targets[key] = np.array(pos, dtype=np.float64)

    def _render_markers(self):
        """Draw trails, start markers, and target markers."""
        if self.show_trails:
            trail_size = np.array([0.008, 0, 0], dtype=np.float64)
            for key, trail in self._trails.items():
                rgba = self._trail_colors.get(
                    key, np.array([0.2, 0.6, 1.0, 0.6], dtype=np.float32)
                )
                for pt in trail[::2]:
                    self._add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, trail_size, pt, rgba)

        marker_size = np.array([0.03, 0, 0], dtype=np.float64)
        green = np.array([0.2, 0.9, 0.3, 0.8], dtype=np.float32)
        red = np.array([0.9, 0.2, 0.2, 0.8], dtype=np.float32)
        for pos in self._starts.values():
            self._add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, marker_size, pos, green)
        for pos in self._targets.values():
            self._add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, marker_size, pos, red)

    def render(self):
        import glfw

        if glfw.window_should_close(self._window):
            return

        # Always process events so window responds to close/input
        glfw.poll_events()

        # Throttle rendering to ~60fps
        now = time.monotonic()
        if now - self._last_render_time < 1.0 / 60.0:
            return
        self._last_render_time = now

        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self._window))
        mujoco.mjv_updateScene(
            self._model,
            self._data,
            self._option,
            self._perturb,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        self._render_markers()
        mujoco.mjr_render(viewport, self._scene, self._context)

        # Overlay sim time top-right
        time_str = f"t = {self._data.time:.2f}s"
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            viewport,
            time_str,
            "",
            self._context,
        )

        # Overlay legend top-left (if set)
        if self._overlay_text:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                self._overlay_text,
                "",
                self._context,
            )

        glfw.swap_buffers(self._window)

        # Capture frame for recording (every Nth rendered frame)
        if self._frames is not None:
            self._render_count += 1
            if self._render_count % self._record_every == 0:
                width, height = glfw.get_framebuffer_size(self._window)
                rgb = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, viewport, self._context)
                self._frames.append(np.flipud(rgb))

    def save_recording(self):
        """Save captured frames to file (GIF or MP4)."""
        if not self._frames or not self._record_path:
            return
        import imageio.v3 as iio

        path = self._record_path
        frames = list(self._frames)
        self._frames = []  # Prevent double-save
        _logger.info("Saving %d frames to %s", len(frames), path)

        if path.endswith(".gif"):
            iio.imwrite(path, frames, duration=1000 // 15, loop=0)
        else:
            iio.imwrite(path, frames, fps=30)
        _logger.info("Saved recording to %s", path)

    def hold(self):
        """Keep window open until user closes it. Press ESC or Q to quit."""
        import glfw

        while self._window and not glfw.window_should_close(self._window):
            self.render()
            glfw.wait_events_timeout(1.0 / 60.0)
        self.close()

    def close(self):
        import glfw

        self.save_recording()
        if self._window:
            glfw.destroy_window(self._window)
            glfw.terminate()
            self._window = None

    def is_alive(self):
        import glfw

        return self._window is not None and not glfw.window_should_close(self._window)


class MujocoModel:
    def __init__(self, model_path, render=False, record=None):
        self.full_path = os.path.join(_FOLDER_PATH, "udaan", "models", "assets", "mjcf", model_path)
        if not os.path.exists(self.full_path):
            raise OSError(f"File {self.full_path} does not exist")

        self.render = render
        self._record = record
        self._viewer = None

        self.frame_skip = 1
        self._initialize_simulation()

    def _initialize_simulation(self):
        _logger.info("Loading model from %s", self.full_path)
        self.model = mujoco.MjModel.from_xml_path(self.full_path)
        self.data = mujoco.MjData(self.model)
        self._wall_start = None

        if self.render:
            self._viewer = _GlfwViewer(self.model, self.data, record=self._record)

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

    def wait_for_close(self):
        """Keep the viewer open until the user closes the window.
        If recording, save and close immediately instead of holding."""
        if self.render and self._viewer is not None:
            if self._viewer._record_path:
                self._viewer.close()
            else:
                self._viewer.hold()

    def set_overlay(self, text):
        """Set overlay text displayed in the top-left corner."""
        if self._viewer is not None:
            self._viewer._overlay_text = text

    def add_marker_at(self, p, size=None, rgba=None, label=""):
        """Visual markers not yet supported with built-in viewer."""
        pass

    def add_arrow_at(self, p, R, s, label="", color=None):
        """Visual markers not yet supported with built-in viewer."""
        pass


from ..quadrotor.mujoco import QuadrotorMujoco as Quadrotor
from .multi_quad_cs_pointmass import MultiQuadrotorCSPointmass as MultiQuadrotorCSPointmass
from .multi_quad_rigidbody import MultiQuadRigidbody as MultiQuadRigidbody
from .quadrotor_comparison import QuadrotorComparison as QuadrotorComparison
from .quadrotor_cspayload import QuadrotorCSPayload as QuadrotorCSPayload
from .quadrotor_fleet import QuadrotorFleet as QuadrotorFleet

__all__ = [
    "MujocoModel",
    "MultiQuadrotorCSPointmass",
    "MultiQuadRigidbody",
    "Quadrotor",
    "QuadrotorComparison",
    "QuadrotorCSPayload",
    "QuadrotorFleet",
]
