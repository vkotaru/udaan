"""The rotor discs on the +x (forward, b1) side of every airframe carry the front colour.

Every quadrotor asset marks its heading: the rotors whose disc centre has a positive
x coordinate in the airframe body frame are ``FRONT_ROTOR_RGB``, and every other
rotor keeps the vehicle's own colour.

Red is reserved for the front. A colour counts as red when it is saturated and bright
enough to read as a hue (HSV ``S >= 0.3`` and ``V >= 0.2``) and its hue lies in
``[290, 360) U [0, 20)`` degrees, the band from magenta-pink through red that stops
short of orange. No visible geom other than a front rotor disc, and no fleet palette
entry, may be red. Sites are not checked: the body-axis triads in ``quadrotor_quat.xml``
and ``quadrotor_x.xml`` draw the +x axis red, which marks the same forward direction.
"""

import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from udaan.utils.assets.mujoco_asset_creator import FRONT_ROTOR_RGB, MujocoAssetCreator

_MJCF_DIR = Path(__file__).resolve().parent.parent / "udaan" / "models" / "assets" / "mjcf"
_PACKAGED_MJCF = sorted(f.name for f in _MJCF_DIR.glob("*.xml") if f.name != "scene.xml")


def _is_red(rgb):
    """True when ``rgb`` lies in the hue band reserved for the front rotors."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(float(c) for c in rgb[:3]))
    hue *= 360.0
    return saturation >= 0.3 and value >= 0.2 and (hue >= 290.0 or hue < 20.0)


def test_front_rotor_colour_is_red():
    assert _is_red(FRONT_ROTOR_RGB)


def _rotor_discs(model):
    """``(name, airframe, x in the airframe frame, rgba)`` for every rotor disc.

    A rotor disc is a cylinder geom named ``*prop*`` or ``thruster*``. Discs inside a
    ``rotor_prop_*`` child body are expressed in that body's parent, the airframe.
    """
    import mujoco

    discs = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            continue
        if "prop" not in name and not name.startswith("thruster"):
            continue
        body = model.geom_bodyid[g]
        pos = model.geom_pos[g].copy()
        while mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body).startswith("rotor_prop"):
            assert np.allclose(model.body_quat[body], [1.0, 0.0, 0.0, 0.0])
            pos = pos + model.body_pos[body]
            body = model.body_parentid[body]
        airframe = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        discs.append((name, airframe, pos[0], model.geom_rgba[g].copy()))
    return discs


@pytest.mark.mujoco
@pytest.mark.parametrize("filename", _PACKAGED_MJCF)
def test_packaged_mjcf_front_rotors_are_front_colour(filename):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_MJCF_DIR / filename))
    discs = _rotor_discs(model)
    if not discs:
        pytest.skip(f"{filename} has no rotor discs")

    airframes = {airframe for _, airframe, _, _ in discs}
    for airframe in airframes:
        own = [d for d in discs if d[1] == airframe]
        front = [d for d in own if d[2] > 1e-9]
        rear = [d for d in own if d[2] <= 1e-9]
        assert len(own) == 4, (filename, airframe)
        assert len(front) == 2, (filename, airframe, [d[0] for d in front])
        for name, _, _, rgba in front:
            np.testing.assert_allclose(rgba[:3], FRONT_ROTOR_RGB, err_msg=name)
        for name, _, _, rgba in rear:
            assert not _is_red(rgba), (name, rgba)


@pytest.mark.mujoco
@pytest.mark.parametrize("filename", _PACKAGED_MJCF)
def test_packaged_mjcf_only_front_rotors_are_red(filename):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_MJCF_DIR / filename))
    front = {name for name, _, x, _ in _rotor_discs(model) if x > 1e-9}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
        rgba = model.geom_rgba[g]
        if rgba[3] > 0 and name not in front:
            assert not _is_red(rgba), (filename, name, rgba)


def _generated_discs(**kwargs):
    writer = MujocoAssetCreator("rotor_colour_test")
    writer.create_quadrotor0(writer.worldbody, "q", np.array([0.0, 0.0, 1.0]), **kwargs)
    return _discs_in(writer.root)


def _discs_in(root):
    discs = []
    for geom in root.iter("geom"):
        if "_rotor_prop_geom_" not in geom.attrib["name"]:
            continue
        pos = np.array([float(v) for v in geom.attrib["pos"].split()])
        rgba = np.array([float(v) for v in geom.attrib["rgba"].split()])
        discs.append((geom.attrib["name"], pos, rgba, float(geom.attrib["mass"])))
    return discs


@pytest.mark.parametrize(
    ("xtype", "front_names"),
    [
        (True, {"q_rotor_prop_geom_0", "q_rotor_prop_geom_3"}),
        (False, {"q_rotor_prop_geom_0"}),
    ],
    ids=["x-layout", "plus-layout"],
)
def test_generator_front_rotors_are_front_colour(xtype, front_names):
    vehicle_rgb = [0.2, 0.5, 0.9]
    discs = _generated_discs(xtype=xtype, rgb=vehicle_rgb, alpha=0.5)
    assert len(discs) == 4
    assert {name for name, pos, _, _ in discs if pos[0] > 1e-9} == front_names
    for name, _, rgba, mass in discs:
        expected = FRONT_ROTOR_RGB if name in front_names else vehicle_rgb
        np.testing.assert_allclose(rgba, [*expected, 0.5], err_msg=name)
        assert mass == 0.05


def _visible_non_front_geoms(root):
    """``(name, rgba)`` of every visible geom that is not a front rotor disc."""
    front = {name for name, pos, _, _ in _discs_in(root) if pos[0] > 1e-9}
    out = []
    for geom in root.iter("geom"):
        rgba = [float(v) for v in geom.attrib["rgba"].split()]
        if rgba[3] > 0 and geom.attrib["name"] not in front:
            out.append((geom.attrib["name"], rgba))
    return out


def test_generator_vehicle_parts_are_not_red():
    writer = MujocoAssetCreator("rotor_colour_test")
    writer.create_quadrotor0(writer.worldbody, "q", np.array([0.0, 0.0, 1.0]), unmodeled_mass=0.1)
    for name, rgba in _visible_non_front_geoms(writer.root):
        assert not _is_red(rgba), (name, rgba)


def test_comparison_generator_vehicles_are_not_red(tmp_path):
    from udaan.utils.assets.xml_model_generator import quadrotor_comparison

    for kwargs in ({}, {"unmodeled_mass": 0.1}):
        path = tmp_path / "comparison.xml"
        quadrotor_comparison(filename=str(path), **kwargs)
        root = ET.parse(path).getroot()
        for name, rgba in _visible_non_front_geoms(root):
            assert not _is_red(rgba), (name, rgba)


@pytest.mark.mujoco
@pytest.mark.parametrize(
    "module",
    ["udaan.models.mujoco.quadrotor_fleet", "udaan.models.mujoco.quadrotor_cspayload_fleet"],
)
def test_fleet_palette_is_not_red_and_distinct(module):
    import importlib

    pytest.importorskip("mujoco")
    palette = importlib.import_module(module)._FLEET_COLORS
    colours = [np.array(rgb) for rgb, _ in palette]
    names = [name for _, name in palette]
    assert len(set(names)) == len(names)
    for rgb, name in palette:
        assert not _is_red(rgb), name
    for i in range(len(colours)):
        for j in range(i + 1, len(colours)):
            assert np.linalg.norm(colours[i] - colours[j]) >= 0.3, (names[i], names[j])
