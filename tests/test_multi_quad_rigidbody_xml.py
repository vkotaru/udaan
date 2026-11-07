"""Tests for the M4 MJCF pipeline: taut spawn geometry, ring azimuths, hashing.

Three properties of the generated asset, each needed for the MuJoCo backend
to simulate the machine the analytical model and the controller describe:

* **Taut spawn.** The payload height is derived from the cable length, the
  formation radius and the attachment height, so every cable starts exactly at
  its tendon limit. A hand-picked height (a quadrotor ring at 1.2071 m and the
  payload at ``z = 1.34``) spans only 0.7887 m of a 1.0 m cable: no tension at
  ``t = 0`` and, the tendons carrying no stiffness, a payload that free-falls
  until the limit constraint catches it. The analytical model has no slack
  mode, so a base-vs-MuJoCo comparison from such a spawn compares the drop
  rather than the integration.
* **Agreeing azimuths.** The ``pyld_poc`` sites sit on the ring of
  :func:`~udaan.utils.grasp.ring_attachment_points`, at the same ``2πi/nQ``
  azimuths as :func:`~udaan.utils.grasp.formation_cable_attitudes`. That
  pairing puts each quadrotor radially outboard of *its own* attachment and
  makes the hover trim moment-free. Corner attachments (45°, 135°, …) trim at
  round-off under their own 45° cable directions, but read through the model's
  0°/90° ring they admit no static hover: ``hover_residual`` is 2.7208 N for
  that pairing against round-off for a consistent one.
* **No free-joint damping.** ``MujocoAssetCreator.joint`` defaults to
  ``damping=0.1``; the generator sets it to zero on the quadrotors' and the
  payload's free joints, because the analytical model has no drag term.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path, PurePath

import mujoco
import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC
from udaan.core.defaults import (
    DEFAULT_ATTACHMENT_HEIGHT,
    DEFAULT_ATTACHMENT_RADIUS,
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_MJCF_TIMESTEP,
    DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS,
    DEFAULT_SPAWN_HEIGHT,
)
from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyMujoco
from udaan.models.multi_quad_rigidbody import mujoco as mj_module
from udaan.utils.assets import (
    MJCF_GENERATOR_VERSION,
    multi_quad_rigidbody,
    rigid_payload_height,
)
from udaan.utils.assets.cache import PACKAGED_MJCF_DIR, generated_mjcf_dir
from udaan.utils.grasp import ring_attachment_points

pytestmark = pytest.mark.mujoco

QUAD_COUNTS = [1, 2, 3, 4, 6]

#: Where generated MJCF lands: a user cache directory, not the installed
#: package. ``scene.xml`` ships with the package, so the two are separate.
ASSET_DIR = generated_mjcf_dir()

#: A hand-picked spawn geometry, the counter-example to the derived one: a
#: quadrotor ring at 45° azimuths, corner sites, and a payload height chosen
#: rather than derived.
_HAND_PICKED_QUAD_XY = np.array([0.8535533905932737, 0.8535533905932736])
_HAND_PICKED_PAYLOAD_HEIGHT = 1.34
_HAND_PICKED_SPAWN_DISTANCE = 0.7887331614684397


@pytest.fixture
def mjcf_dir(tmp_path) -> Path:
    """Scratch dir the generated MJCF can ``<include>`` its scene.xml from."""
    shutil.copy(PACKAGED_MJCF_DIR / "scene.xml", tmp_path / "scene.xml")
    return tmp_path


@pytest.fixture
def scratch_assets():
    """Remove any MJCF this test generated in the packaged asset directory."""
    before = set(ASSET_DIR.iterdir())
    yield
    for path in set(ASSET_DIR.iterdir()) - before:
        path.unlink()


def attitude_between_deg(a, b) -> float:
    """Angle between two unit vectors, in degrees."""
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _cable_spans(path) -> np.ndarray:
    """Quadrotor -> *attachment site* distances of a model in its spawn pose.

    Not quadrotor-to-payload-centre: the cable ends at ``pyld_poc{i}``, and the
    difference between the two is the whole of M4.
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spans = []
    i = 0
    while (bid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}")) >= 0:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pyld_poc{i}")
        assert site >= 0
        spans.append(float(np.linalg.norm(data.xpos[bid] - data.site_xpos[site])))
        i += 1
    return np.array(spans)


class TestGeneratedXml:
    """The emitted MJCF parses and is wired up for every quadrotor count."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_parses_in_mujoco(self, mjcf_dir, nQ):
        path = mjcf_dir / f"m4_{nQ}.xml"
        multi_quad_rigidbody(num_quadrotors=nQ, filename=str(path), verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))

        assert model.nu == 4 * nQ
        assert model.ntendon == nQ
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pyld") >= 0
        for i in range(nQ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}") >= 0
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pyld_poc{i}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{nQ}") < 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pyld_poc{nQ}") < 0

    @pytest.mark.parametrize("nQ", [1, 3])
    def test_actuators_live_in_one_block(self, mjcf_dir, nQ):
        path = mjcf_dir / f"m4_{nQ}.xml"
        multi_quad_rigidbody(num_quadrotors=nQ, filename=str(path), verbose=False)
        root = ET.parse(path).getroot()
        blocks = root.findall("actuator")
        assert len(blocks) == 1
        assert len(blocks[0].findall("motor")) == 4 * nQ

    def test_masses_and_slab_reach_the_model(self, mjcf_dir):
        path = mjcf_dir / "m4_masses.xml"
        multi_quad_rigidbody(
            num_quadrotors=3,
            filename=str(path),
            payload_mass=1.7,
            payload_half_extents=[0.4, 0.3, 0.04],
            quad_mass=1.23,
            attachment_radius=0.25,
            attachment_height=0.04,
            verbose=False,
        )
        model = mujoco.MjModel.from_xml_path(str(path))
        pyld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pyld")
        assert model.body_mass[pyld] == pytest.approx(1.7)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pyld")
        np.testing.assert_allclose(model.geom_size[geom], [0.4, 0.3, 0.04], atol=TOL_ANALYTIC)
        for i in range(3):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}")
            assert model.body_mass[bid] == pytest.approx(1.23)

    def test_payload_inertia_is_mujocos_box_formula(self, mjcf_dir):
        """``inertiafromgeom`` on a box is ``(m/3) diag(b²+c², a²+c², a²+b²)``.

        The same formula ``DEFAULT_RIGID_PAYLOAD_INERTIA`` uses, which is why
        the analytical default and the generated asset agree by construction
        rather than by coincidence.
        """
        a, b, c = 0.4, 0.3, 0.04
        mass = 1.7
        path = mjcf_dir / "m4_inertia.xml"
        multi_quad_rigidbody(
            num_quadrotors=3,
            filename=str(path),
            payload_mass=mass,
            payload_half_extents=[a, b, c],
            attachment_radius=0.25,
            attachment_height=c,
            verbose=False,
        )
        model = mujoco.MjModel.from_xml_path(str(path))
        pyld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pyld")
        expected = (mass / 3.0) * np.array([b**2 + c**2, a**2 + c**2, a**2 + b**2])
        np.testing.assert_allclose(model.body_inertia[pyld], expected, atol=TOL_ANALYTIC)


class TestTautSpawn:
    """The generated formation starts taut: every cable spans its tendon limit."""

    def test_payload_height_is_derived_not_literal(self):
        expected = (
            DEFAULT_SPAWN_HEIGHT
            - DEFAULT_ATTACHMENT_HEIGHT
            - np.sqrt(DEFAULT_CABLE_LENGTH**2 - DEFAULT_FORMATION_RADIUS**2)
        )
        assert rigid_payload_height() == pytest.approx(expected, abs=TOL_ANALYTIC)
        # The derived height is well away from the hand-picked literal.
        assert abs(rigid_payload_height() - _HAND_PICKED_PAYLOAD_HEIGHT) > 0.2

    def test_a_hand_picked_geometry_is_slack_and_the_derived_one_is_taut(self):
        """A chosen spawn leaves the cables slack; the derived one spans them exactly.

        Quadrotors at radius 1.2071 m and height 2.0, with the payload at 1.34
        and corner sites 0.5 m off-axis, span 0.7887 m of a 1.0 m cable. The
        derived spawn spans the cable exactly.
        """
        # Hand-picked: quadrotor at 45°, radius 1.2071; site at the (0.5, 0.5)
        # corner of a payload centred at z = 1.34, 0.05 above its centre.
        quad = np.array([*_HAND_PICKED_QUAD_XY, 2.0])
        site = np.array([0.5, 0.5, _HAND_PICKED_PAYLOAD_HEIGHT + 0.05])
        assert float(np.linalg.norm(quad - site)) == pytest.approx(
            _HAND_PICKED_SPAWN_DISTANCE, abs=1e-12
        )
        assert _HAND_PICKED_SPAWN_DISTANCE < 0.8 * DEFAULT_CABLE_LENGTH

        # Derived: the quadrotor is radially outboard of its own attachment by
        # exactly formation_radius, and lifted by the triangle's other leg.
        z_L = rigid_payload_height()
        radius_q = DEFAULT_ATTACHMENT_RADIUS + DEFAULT_FORMATION_RADIUS
        quad = np.array([radius_q, 0.0, DEFAULT_SPAWN_HEIGHT])
        site = np.array([DEFAULT_ATTACHMENT_RADIUS, 0.0, z_L + DEFAULT_ATTACHMENT_HEIGHT])
        assert float(np.linalg.norm(quad - site)) == pytest.approx(
            DEFAULT_CABLE_LENGTH, abs=TOL_ANALYTIC
        )

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    @pytest.mark.parametrize(
        ("cable_length", "formation_radius", "attachment_radius"),
        [(1.0, 0.5, 0.5), (1.5, 0.5, 0.3), (1.0, 0.25, 0.5), (2.0, 1.9, 0.2)],
    )
    def test_cables_start_taut(
        self, mjcf_dir, nQ, cable_length, formation_radius, attachment_radius
    ):
        path = mjcf_dir / f"m4_{nQ}_{cable_length}_{formation_radius}_{attachment_radius}.xml"
        multi_quad_rigidbody(
            num_quadrotors=nQ,
            filename=str(path),
            cable_length=cable_length,
            formation_radius=formation_radius,
            attachment_radius=attachment_radius,
            spawn_height=4.0,
            verbose=False,
        )
        spans = _cable_spans(path)
        assert spans.shape == (nQ,)
        np.testing.assert_allclose(spans, cable_length, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", [1, 3, 4])
    def test_every_free_joint_is_undamped(self, mjcf_dir, nQ):
        """No drag on any body — the analytical M4 model has no such term.

        ``MujocoAssetCreator.joint`` defaults to ``damping=0.1``, which on a
        *free* joint is a viscous drag on all six degrees of freedom. The
        generator overrides it to zero on the quadrotor root joints and the
        payload's, so the drag does not enter the base-vs-MuJoCo comparison.
        """
        path = mjcf_dir / f"m4_damping_{nQ}.xml"
        multi_quad_rigidbody(num_quadrotors=nQ, filename=str(path), verbose=False)

        # In the file, so a reader sees it too, not only the compiled model.
        for joint in ET.parse(path).getroot().iter("joint"):
            assert joint.attrib["damping"] == "0.0", joint.attrib["name"]

        model = mujoco.MjModel.from_xml_path(str(path))
        np.testing.assert_allclose(model.dof_damping, 0.0, atol=0.0)

    def test_the_timestep_divides_the_models_outer_step(self, mjcf_dir):
        """The MJCF timestep divides ``dt``, so a ``step()`` advances exactly ``dt``.

        A MuJoCo backend takes ``round(dt / timestep)`` substeps per
        :meth:`step`. MuJoCo's own 2 ms default against the model's 5 ms gives
        ``round(2.5) = 2``, i.e. 4 ms of physics per step, and the two backends
        would be compared at different simulation times.
        """
        path = mjcf_dir / "m4_timestep.xml"
        multi_quad_rigidbody(num_quadrotors=3, filename=str(path), verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.opt.timestep == pytest.approx(DEFAULT_MJCF_TIMESTEP)

        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
        assert mdl._mjMdl.model.opt.timestep == pytest.approx(DEFAULT_MJCF_TIMESTEP)
        # Exactly, not approximately: the reported dt is the time taken.
        assert mdl._step_iter * mdl._mjDt == pytest.approx(mdl.dt, abs=1e-15)
        before = mdl.t
        mdl.step(np.zeros((3, 3)))
        assert mdl.t - before == pytest.approx(mdl.dt, abs=1e-12)

    def test_tendons_sit_exactly_at_their_limit(self, mjcf_dir):
        path = mjcf_dir / "m4_tendon.xml"
        multi_quad_rigidbody(num_quadrotors=4, filename=str(path), cable_length=1.25, verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(model.tendon_range[:, 1], 1.25, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(data.ten_length, 1.25, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", [2, 3, 4])
    def test_model_spawns_taut(self, nQ):
        """Through the model: every cable is at its length the moment it loads."""
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=nQ)
        mdl.reset()
        for i in range(nQ):
            span = np.linalg.norm(mdl.attachment_position(i) - mdl.state.positions[i])
            assert span == pytest.approx(mdl.cable_lengths[i], abs=TOL_ANALYTIC)

    def test_model_stays_taut_after_translation(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3)
        mdl.reset(payload_position=np.array([-1.0, 2.0, 1.5]))
        for i in range(3):
            span = np.linalg.norm(mdl.attachment_position(i) - mdl.state.positions[i])
            assert span == pytest.approx(mdl.cable_lengths[i], abs=TOL_ANALYTIC)

    def test_a_slack_asset_is_refused_at_load(self):
        """A spawn with slack cables is refused, not warned about.

        The payload is raised 0.2 m without moving the quadrotors. The backend must refuse it rather than warn and carry on: the
        analytical model has no slack mode to match, so the two backends would
        be simulating different machines.
        """
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=4)
        # Raise the payload without moving the quadrotors: 0.2 m of slack.
        i = mdl._payload_qposadr
        mdl._mjMdl.data.qpos[i + 2] += 0.2
        with pytest.raises(RuntimeError, match="does not spawn taut"):
            mdl._check_spawn_is_taut()

    def test_a_stale_asset_with_the_wrong_formation_radius_is_reported(
        self, scratch_assets, caplog
    ):
        """A taut, well-formed, *wrong* file — the case the hover check is for.

        The stale-cache scenario, built for real: generate the asset the
        default constructor asks for, then overwrite that exact filename with
        one whose quadrotor ring sits 0.80 m outboard of each attachment
        instead of 0.50 m. Nothing else moves — the ``pyld_poc`` sites are
        untouched, the tendons are wired the same way and every cable is still
        exactly at its limit — so the file sails through
        ``_read_params_from_mjcf``, ``_check_tendon_wiring`` and
        ``_check_spawn_is_taut``.

        It is nonetheless not the machine the controller is flying. Its cable
        directions are 36 deg off the ones the cascade allocates through, and
        the cascade's own ``hover_residual`` cannot see that: it is solved
        from the *constructor's* ``formation_radius`` and never reads the
        file. Deriving the check from the loaded geometry is what makes the
        difference visible.
        """
        cable_length = 1.02  # a config no other test warms, so the cache is ours
        kwargs = {"render": False, "num_quadrotors": 4, "cable_lengths": cable_length}

        good = MultiQuadRigidbodyMujoco(**kwargs)
        assert good.mjcf_hover_residual < 1e-9
        generated = list(ASSET_DIR.glob("multi_quad_rigidbody_4q_*.xml"))
        generated = [p for p in generated if f"{cable_length}" in p.read_text()]
        assert len(generated) == 1, "could not identify the generated asset"

        multi_quad_rigidbody(
            num_quadrotors=4,
            filename=str(generated[0]),
            cable_length=cable_length,
            formation_radius=0.80,
            verbose=False,
        )

        with caplog.at_level("WARNING"):
            stale = MultiQuadRigidbodyMujoco(**kwargs)

        # It loads, and every existing load-time check is happy with it.
        for i in range(4):
            span = np.linalg.norm(stale.attachment_position(i) - stale.state.positions[i])
            assert span == pytest.approx(stale.cable_lengths[i], abs=TOL_ANALYTIC)
        np.testing.assert_allclose(
            stale.attachment_points, good.attachment_points, atol=TOL_ANALYTIC
        )
        # ... including the controller's own residual, which reads its
        # constructor arguments and not the file.
        assert stale.payload_controller.hover_residual < 1e-12

        # The file-derived one is what catches it.
        assert stale.mjcf_hover_residual > 1.0
        assert "does not hover" in caplog.text

        # And the reason: the file's cable directions are nowhere near the
        # controller's.
        for i in range(4):
            assert (
                attitude_between_deg(
                    np.asarray(stale.state.cable_attitudes[i]),
                    np.asarray(good.state.cable_attitudes[i]),
                )
                > 20.0
            )


class TestRingAzimuthsAgree:
    """The MJCF's sites and the model's ring are the same geometry."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_sites_are_the_models_ring(self, mjcf_dir, nQ):
        path = mjcf_dir / f"m4_ring_{nQ}.xml"
        multi_quad_rigidbody(num_quadrotors=nQ, filename=str(path), verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))
        sites = np.array(
            [
                model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pyld_poc{i}")]
                for i in range(nQ)
            ]
        )
        radius = 0.0 if nQ == 1 else DEFAULT_ATTACHMENT_RADIUS
        expected = ring_attachment_points(nQ, radius, DEFAULT_ATTACHMENT_HEIGHT)
        np.testing.assert_allclose(sites, expected, atol=1e-9)

    @pytest.mark.parametrize("nQ", [3, 4, 6])
    def test_each_quadrotor_is_outboard_of_its_own_attachment(self, mjcf_dir, nQ):
        """The invariant that makes the trim moment-free.

        Quadrotor ``i`` must sit on attachment ``i``'s own azimuth, further
        out. Sites rotated a half-sector away from the model's ring would
        satisfy this for the *file* and fail it for the pairing the model uses.
        """
        path = mjcf_dir / f"m4_outboard_{nQ}.xml"
        multi_quad_rigidbody(num_quadrotors=nQ, filename=str(path), verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for i in range(nQ):
            quad = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}")]
            site = data.site_xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pyld_poc{i}")
            ]
            # Same azimuth: the horizontal offset is purely radial.
            radial = site[:2] / np.linalg.norm(site[:2])
            offset = quad[:2] - site[:2]
            np.testing.assert_allclose(offset, DEFAULT_FORMATION_RADIUS * radial, atol=TOL_ANALYTIC)
            assert quad[2] > site[2]

    @pytest.mark.parametrize("nQ", [3, 4, 6])
    def test_the_loaded_geometry_admits_a_static_hover(self, nQ):
        """``hover_residual`` is at round-off for the generated asset.

        A grasp whose attachment azimuths disagree with its formation's cannot
        simultaneously hold the payload's weight and zero its net moment; the
        residual is how far short it falls, and a formation with a non-zero
        residual never settles.
        """
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=nQ)
        assert mdl.payload_controller.hover_residual < 1e-12
        assert mdl.payload_controller.grasp_rank == 6

    def test_corner_sites_hover_only_with_their_own_cable_directions(self):
        """The negative of the test above, and what it is the negative *of*.

        A non-zero residual is not a property of a corner grasp. It is the
        residual of reading corner attachments through the *model's*
        0 deg/90 deg cable directions, which is what the controller does
        (2.7208 N). Give the same corners the 45 deg directions a quadrotor
        ring on their own azimuths implies (``quad0`` at radius 1.20711,
        ``pyld_poc0`` at 0.70711, both at 45 deg) and they trim at round-off.

        So what matters is the *pairing* of file and model, not the corner
        geometry, and the generator derives both from the same ring.
        """
        from udaan.control.multi_quad_rigidbody import MultiQuadRigidbodyController
        from udaan.utils.grasp import formation_cable_attitudes

        corners = np.array(
            [
                [0.5, 0.5, 0.05],
                [-0.5, 0.5, 0.05],
                [-0.5, -0.5, 0.05],
                [0.5, -0.5, 0.05],
            ]
        )
        through_the_models_ring = MultiQuadRigidbodyController(
            num_quadrotors=4, attachment_points=corners
        )
        assert through_the_models_ring.hover_residual == pytest.approx(2.7208, abs=1e-3)

        # The corners' own directions: the same ring, rotated onto the
        # corners' 45 deg azimuths.
        half = np.pi / 4.0
        c, s = np.cos(half), np.sin(half)
        rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        directions = (
            np.array([np.asarray(q) for q in formation_cable_attitudes(4, np.ones(4), 0.5)])
            @ rot_z.T
        )
        consistent = MultiQuadRigidbodyController(
            num_quadrotors=4, attachment_points=corners, hover_cable_attitudes=directions
        )
        assert consistent.hover_residual < 1e-12
        assert np.all(consistent.hover_tensions > 0.0)


class TestConfigHash:
    """A differing config regenerates; an identical one reuses the cache."""

    _BASE = {
        "cable_length": 1.05,
        "payload_mass": 1.17,
        "quad_mass": 0.95,
        "payload_half_extents": np.array([0.45, 0.45, 0.06]),
        "attachment_radius": 0.4,
        "attachment_height": 0.06,
        "formation_radius": 0.45,
        "spawn_height": 2.25,
    }

    def _name(self, nQ=3, **overrides):
        cfg = {**self._BASE, **overrides}
        return MultiQuadRigidbodyMujoco._ensure_xml(
            nQ,
            cfg["cable_length"],
            cfg["payload_mass"],
            cfg["quad_mass"],
            cfg["payload_half_extents"],
            cfg["attachment_radius"],
            cfg["attachment_height"],
            cfg["formation_radius"],
            cfg["spawn_height"],
        )

    @staticmethod
    def _basename(generated: str) -> str:
        """``_ensure_xml`` returns a full path into the asset cache."""
        return PurePath(generated).name

    def test_identical_config_reuses_the_same_file(self, scratch_assets):
        first = self._name()
        stat = (ASSET_DIR / self._basename(first)).stat()
        second = self._name()
        assert first == second
        assert self._basename(first).startswith("multi_quad_rigidbody_3q_")
        # Reused, not rewritten.
        assert (ASSET_DIR / self._basename(first)).stat().st_mtime_ns == stat.st_mtime_ns

    @pytest.mark.parametrize(
        "override",
        [
            {"cable_length": 1.15},
            {"payload_mass": 1.27},
            {"quad_mass": 1.05},
            {"payload_half_extents": np.array([0.45, 0.45, 0.07])},
            {"attachment_radius": 0.3},
            {"attachment_height": 0.05},
            {"formation_radius": 0.35},
            {"spawn_height": 2.75},
        ],
    )
    def test_each_parameter_changes_the_hash(self, scratch_assets, override):
        assert self._name() != self._name(**override)

    def test_quadrotor_count_changes_the_name(self, scratch_assets):
        assert self._name(nQ=2) != self._name(nQ=3)

    def test_generator_version_changes_the_hash(self, scratch_assets, monkeypatch):
        """A release that reshapes the XML must not reuse a warm cache."""
        first = self._name()
        monkeypatch.setattr(mj_module, "MJCF_GENERATOR_VERSION", MJCF_GENERATOR_VERSION + 1)
        assert self._name() != first

    def test_generated_files_differ_in_content(self, scratch_assets):
        base = Path(self._name())
        other = Path(self._name(cable_length=1.15))
        assert base.read_text() != other.read_text()
        np.testing.assert_allclose(_cable_spans(base), 1.05, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(_cable_spans(other), 1.15, atol=TOL_ANALYTIC)

    def test_a_lone_quadrotor_ignores_both_radii(self, scratch_assets):
        """Neither ring is used at ``nQ = 1``, so neither may enter the hash."""
        first = self._name(nQ=1)
        assert first == self._name(nQ=1, formation_radius=0.9, attachment_radius=0.1)


class TestGeometryGuards:
    """Infeasible geometry is refused at generation, not logged and flown."""

    @pytest.mark.parametrize("cable_length", [0.5, 0.5 - 1e-12])
    def test_cable_shorter_than_the_formation_radius_raises(self, cable_length):
        with pytest.raises(ValueError, match="must be strictly greater than the formation"):
            rigid_payload_height(cable_length=cable_length, formation_radius=0.5)

    def test_negative_formation_radius_raises(self):
        with pytest.raises(ValueError, match="formation_radius must be >= 0"):
            rigid_payload_height(formation_radius=-0.1)

    def test_generator_rejects_an_infeasible_triangle(self, mjcf_dir):
        with pytest.raises(ValueError, match="must be strictly greater than the formation"):
            multi_quad_rigidbody(
                num_quadrotors=3,
                filename=str(mjcf_dir / "bad.xml"),
                cable_length=0.4,
                formation_radius=0.5,
                verbose=False,
            )

    @pytest.mark.parametrize("nQ", [0, -1])
    def test_non_positive_quadrotor_count_raises(self, mjcf_dir, nQ):
        with pytest.raises(ValueError, match="num_quadrotors must be >= 1"):
            multi_quad_rigidbody(
                num_quadrotors=nQ, filename=str(mjcf_dir / "bad.xml"), verbose=False
            )

    def test_negative_attachment_radius_raises(self, mjcf_dir):
        with pytest.raises(ValueError, match="attachment_radius must be >= 0"):
            multi_quad_rigidbody(
                num_quadrotors=3,
                filename=str(mjcf_dir / "bad.xml"),
                attachment_radius=-0.1,
                verbose=False,
            )

    def test_payload_below_the_ground_raises(self, mjcf_dir):
        with pytest.raises(ValueError, match="ground plane"):
            multi_quad_rigidbody(
                num_quadrotors=3,
                filename=str(mjcf_dir / "bad.xml"),
                spawn_height=0.5,
                verbose=False,
            )

    def test_an_attachment_ring_off_the_slab_raises(self, mjcf_dir):
        """A site anchored to thin air is a moment arm with no body behind it."""
        with pytest.raises(ValueError, match="leaves the payload footprint"):
            multi_quad_rigidbody(
                num_quadrotors=3,
                filename=str(mjcf_dir / "bad.xml"),
                attachment_radius=0.9,
                verbose=False,
            )

    def test_an_attachment_height_outside_the_slab_raises(self, mjcf_dir):
        with pytest.raises(ValueError, match="outside the slab"):
            multi_quad_rigidbody(
                num_quadrotors=3,
                filename=str(mjcf_dir / "bad.xml"),
                attachment_height=0.5,
                verbose=False,
            )

    @pytest.mark.parametrize("extents", [[0.5, 0.5], [0.5, 0.0, 0.05]])
    def test_a_malformed_slab_raises(self, mjcf_dir, extents):
        with pytest.raises(ValueError, match="payload_half_extents"):
            multi_quad_rigidbody(
                num_quadrotors=3,
                filename=str(mjcf_dir / "bad.xml"),
                payload_half_extents=extents,
                verbose=False,
            )


class TestBackendConstruction:
    """What the backend will and will not take at construction time."""

    def test_inertias_at_construction_raise(self):
        with pytest.raises(ValueError, match="derives inertias"):
            MultiQuadRigidbodyMujoco(render=False, inertias=np.eye(3))

    def test_attachment_points_at_construction_raise(self):
        """The sites come from the ring, so an arbitrary set would disagree."""
        with pytest.raises(ValueError, match="attachment_points"):
            MultiQuadRigidbodyMujoco(render=False, attachment_points=np.zeros((4, 3)))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cable_lengths": [1.0, 1.1, 1.0]},
            {"masses": [0.9, 1.0, 0.9]},
        ],
    )
    def test_non_uniform_construction_parameters_raise(self, kwargs):
        with pytest.raises(ValueError, match="needs a uniform"):
            MultiQuadRigidbodyMujoco(render=False, num_quadrotors=3, **kwargs)

    def test_near_uniform_parameters_are_not_silently_flattened(self):
        with pytest.raises(ValueError, match="needs a uniform"):
            MultiQuadRigidbodyMujoco(render=False, num_quadrotors=2, cable_lengths=[1.0, 1.0000001])

    def test_the_ring_knobs_move_both_the_asset_and_the_model(self):
        mdl = MultiQuadRigidbodyMujoco(
            render=False, num_quadrotors=4, attachment_radius=0.3, attachment_height=0.04
        )
        expected = ring_attachment_points(4, 0.3, 0.04)
        np.testing.assert_allclose(mdl.attachment_points, expected, atol=1e-9)
        model = mdl._mjMdl.model
        sites = np.array([model.site_pos[i] for i in mdl._attachment_site_id])
        np.testing.assert_allclose(sites, expected, atol=1e-9)

    def test_defaults_reach_the_model(self):
        mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=4)
        assert mdl.attachment_radius == pytest.approx(DEFAULT_ATTACHMENT_RADIUS)
        assert mdl.attachment_height == pytest.approx(DEFAULT_ATTACHMENT_HEIGHT)
        assert mdl.spawn_height == pytest.approx(DEFAULT_SPAWN_HEIGHT)
        np.testing.assert_allclose(
            mdl.payload_half_extents, DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS, atol=TOL_ANALYTIC
        )
        np.testing.assert_allclose(mdl.cable_lengths, DEFAULT_CABLE_LENGTH, atol=TOL_ANALYTIC)
