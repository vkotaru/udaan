"""Viewer camera interaction against the installed MuJoCo, without a window."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from udaan.models.mujoco import _GlfwViewer, _move_camera

pytestmark = pytest.mark.mujoco

_XML = "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>"


@pytest.fixture
def camera_setup():
    model = mujoco.MjModel.from_xml_string(_XML)
    scene = mujoco.MjvScene(model, maxgeom=10)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    # A translate reads the scene's frustum, which the first render fills in.
    mujoco.mjv_updateScene(
        model,
        mujoco.MjData(model),
        mujoco.MjvOption(),
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return model, scene, camera


def _camera_pose(camera):
    return np.array([camera.azimuth, camera.elevation, camera.distance, *camera.lookat])


@pytest.mark.parametrize(
    "action",
    [
        mujoco.mjtMouse.mjMOUSE_ROTATE_V,
        mujoco.mjtMouse.mjMOUSE_MOVE_V,
        mujoco.mjtMouse.mjMOUSE_ZOOM,
    ],
)
def test_move_camera_moves_the_camera(camera_setup, action):
    model, scene, camera = camera_setup
    before = _camera_pose(camera)
    _move_camera(model, action, 0.05, 0.05, scene, camera)
    assert not np.allclose(_camera_pose(camera), before)


def _viewer_stub(camera_setup, **buttons):
    model, scene, camera = camera_setup
    pressed = {"_button_left": False, "_button_right": False, "_button_middle": False}
    pressed.update(buttons)
    return SimpleNamespace(
        _model=model,
        _scene=scene,
        _camera=camera,
        _last_mouse_x=100.0,
        _last_mouse_y=100.0,
        _glfw=SimpleNamespace(get_window_size=lambda window: (800, 600)),
        **pressed,
    )


@pytest.mark.parametrize("button", ["_button_left", "_button_right", "_button_middle"])
def test_mouse_drag_moves_the_camera(camera_setup, button):
    viewer = _viewer_stub(camera_setup, **{button: True})
    before = _camera_pose(viewer._camera)
    _GlfwViewer._mouse_move_callback(viewer, None, 140.0, 130.0)
    assert not np.allclose(_camera_pose(viewer._camera), before)


def test_scroll_zooms_the_camera(camera_setup):
    viewer = _viewer_stub(camera_setup)
    before = viewer._camera.distance
    _GlfwViewer._scroll_callback(viewer, None, 0.0, 1.0)
    assert viewer._camera.distance != before
