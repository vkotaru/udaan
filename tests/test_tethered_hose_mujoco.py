"""M7 / M8 MuJoCo backend: the welded anchor, and agreement with the analytical model."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from udaan.flatness.tethered_hose import TetherCircle
from udaan.manif import SO3
from udaan.models.tethered_hose import (
    TetheredHoseBase,
    TetheredHoseMujoco,
    TetheredQuadrotorBase,
    TetheredQuadrotorMujoco,
)
from udaan.utils.assets.xml_model_generator import multi_quad_hose

pytestmark = pytest.mark.mujoco

G = 9.81
DATA = Path(__file__).parent / "data"


def _unit(v):
    return v / np.linalg.norm(v)


def _base_like(mj):
    """The analytical model of the machine the MuJoCo backend read from its MJCF."""
    cls = TetheredQuadrotorBase if isinstance(mj, TetheredQuadrotorMujoco) else TetheredHoseBase
    base = cls(num_links=mj.num_links, attachments=mj.attachments, hose_length=mj.hose_length)
    base.masses = mj.masses
    base.inertias = mj.inertias
    base.vertex_masses = mj.vertex_masses
    return base


class TestAsset:
    def test_the_root_body_is_welded_to_the_world(self):
        mj = TetheredQuadrotorMujoco()
        model = mj._mjMdl.model
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hose_vertex0")
        assert model.body_jntnum[root] == 0
        assert model.nq == 4 * (mj.num_links + mj.num_quadrotors)

    def test_the_anchor_stays_put_in_flight(self):
        mj = TetheredHoseMujoco(anchor_height=0.3)
        circle = TetherCircle(mj.hose_controller.flat_map, [1.5, 0.0, 2.7], rate=1.0, ramp_time=2.0)
        mj.hose_controller.setpoint = circle.get
        mj.simulate(3.0)
        root = mujoco.mj_name2id(mj._mjMdl.model, mujoco.mjtObj.mjOBJ_BODY, "hose_vertex0")
        np.testing.assert_array_equal(mj._mjMdl.data.xpos[root], [0.0, 0.0, 0.3])

    def test_m6_generator_output_is_unchanged(self, tmp_path):
        """The M6 generator writes the reference MJCF in ``tests/data``.

        Elements and attributes must match exactly; numbers must match to 1e-12,
        because the last bit of a written float depends on the platform's libm.
        """
        cases = {
            "multi_quad_hose_default.xml": {},
            "multi_quad_hose_six_links.xml": {"num_links": 6, "attachments": (0, 3, 6)},
        }
        for reference, kwargs in cases.items():
            path = multi_quad_hose(filename=str(tmp_path / "m6.xml"), verbose=False, **kwargs)
            _assert_same_mjcf(ET.parse(path).getroot(), ET.parse(DATA / reference).getroot())


def _assert_same_mjcf(got, expected):
    assert got.tag == expected.tag
    assert got.attrib.keys() == expected.attrib.keys(), got.tag
    for key, value in expected.attrib.items():
        try:
            want = np.array(value.split(), dtype=float)
        except ValueError:
            assert got.attrib[key] == value, (got.tag, key)
        else:
            have = np.array(got.attrib[key].split(), dtype=float)
            np.testing.assert_allclose(
                have, want, rtol=1e-12, atol=1e-12, err_msg=f"{got.tag}.{key}"
            )
    assert len(got) == len(expected), got.tag
    for child_got, child_expected in zip(got, expected, strict=True):
        _assert_same_mjcf(child_got, child_expected)


class TestAgreement:
    @pytest.mark.parametrize("cls", [TetheredQuadrotorMujoco, TetheredHoseMujoco])
    def test_vertex_accelerations_match_mujoco(self, cls):
        """The reduced (root-deleted) solve against MuJoCo's welded tree, at random states.

        The residual is the vertex spheres' own rotational inertia, which the
        point-mass model does not have (M6's ``r²`` term): well under 0.2 % here.
        """
        mj = cls()
        rng = np.random.default_rng(0)
        n, nQ = mj.num_links, mj.num_quadrotors
        model, data = mj._mjMdl.model, mj._mjMdl.data
        for _ in range(4):
            qs = []
            for _ in range(n):
                v = np.array([0.0, 0.0, 1.0]) + 0.6 * rng.normal(size=3)
                qs.append(v / np.linalg.norm(v))
            orientations = [
                np.asarray(
                    SO3.from_two_vectors(
                        _unit(np.array([0, 0, 1.0]) + 0.3 * rng.normal(size=3)),
                        np.array([1.0, 0, 0]),
                    )
                )
                for _ in range(nQ)
            ]
            mj.reset(
                link_attitudes=qs,
                link_angular_velocities=[0.5 * rng.normal(size=3) for _ in range(n)],
                orientations=orientations,
                angular_velocities=[0.5 * rng.normal(size=3) for _ in range(nQ)],
            )
            wrench = np.column_stack([rng.uniform(5, 15, nQ), 0.05 * rng.normal(size=(nQ, 3))])
            data.ctrl[:] = mj._repackage_input(wrench).reshape(-1)
            mujoco.mj_forward(model, data)
            mujoco.mj_rnePostConstraint(model, data)
            measured = np.zeros((n + 1, 3))
            for i in range(n):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"hose_vertex_geom{i + 1}")
                acc = np.zeros(6)
                mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_GEOM, gid, acc, 0)
                measured[i + 1] = acc[3:] - G * np.array([0.0, 0.0, 1.0])
            predicted = mj.accelerations(wrench[:, 0], wrench[:, 1:])["vertex_accels"]
            assert np.abs(predicted - measured).max() < 2e-3 * np.abs(measured).max()

    def test_a_circle_flies_the_same_on_both_backends(self):
        mj = TetheredQuadrotorMujoco()
        base = _base_like(mj)
        logs = []
        for mdl in (mj, base):
            circle = TetherCircle(
                mdl.hose_controller.flat_map, [1.0, 0.0, 1.6], rate=2.0, ramp_time=6.0
            )
            mdl.hose_controller.setpoint = circle.get
            mdl.reset()
            log = []
            while mdl.t < 8.0:
                mdl.step(mdl.compute_hose_input(reference=circle.recover(mdl.t)))
                log.append(np.asarray(mdl.state.positions[-1]).copy())
            logs.append(np.array(log))
        assert np.abs(logs[0] - logs[1]).max() < 5e-3

    def test_reset_keywords_are_honoured(self):
        mj = TetheredHoseMujoco()
        ref, _ = mj.hose_controller.equilibrium_reference(0.0)
        q = [_unit(np.asarray(x) + np.array([0.0, 0.05, 0.0])) for x in ref.link_attitudes]
        w = [np.cross(x, np.array([0.1, -0.2, 0.05])) for x in q]
        R = [
            np.asarray(
                SO3.from_two_vectors(_unit(np.array([0.05, 0.0, 1.0])), np.array([1.0, 0, 0]))
            )
        ] * 3
        Om = [np.array([0.1, 0.2, -0.1])] * 3
        mj.reset(link_attitudes=q, link_angular_velocities=w, orientations=R, angular_velocities=Om)
        st = mj.state
        for got, want in (
            (st.link_attitudes, q),
            (st.link_angular_velocities, w),
            (st.orientations, R),
            (st.angular_velocities, Om),
        ):
            for a, b in zip(got, want, strict=True):
                np.testing.assert_allclose(np.asarray(a), b, atol=1e-12)
        with pytest.raises(ValueError):
            mj.reset(root_position=np.zeros(3))
