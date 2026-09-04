"""Tests for the M3 MJCF pipeline: taut spawn geometry, config hashing, guards.

The taut-geometry tests here are the regression cover for the retired
``zL = 1.34`` literal, which overwrote the derived payload height of 1.134 m —
putting the payload 0.206 m too high, so each cable spanned only 0.828 m of its
1.0 m tendon limit and started with 0.172 m of slack.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC
from udaan.core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_MASS,
    DEFAULT_SPAWN_HEIGHT,
)
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco
from udaan.models.multi_quad_cspayload import mujoco as mj_module
from udaan.models.multi_quad_cspayload.mujoco import _MJCF_DIR
from udaan.utils.assets import (
    MJCF_GENERATOR_VERSION,
    multi_quad_pointmass,
    payload_height,
)

pytestmark = pytest.mark.mujoco

QUAD_COUNTS = [1, 2, 3, 4, 6]


ASSET_DIR = Path(_MJCF_DIR)


@pytest.fixture
def mjcf_dir(tmp_path) -> Path:
    """Scratch dir the generated MJCF can ``<include>`` its scene.xml from."""
    shutil.copy(ASSET_DIR / "scene.xml", tmp_path / "scene.xml")
    return tmp_path


@pytest.fixture
def scratch_assets():
    """Remove any MJCF this test generated in the packaged asset directory."""
    before = set(ASSET_DIR.iterdir())
    yield
    for path in set(ASSET_DIR.iterdir()) - before:
        path.unlink()


def _quad_payload_distances(path) -> np.ndarray:
    """Quad -> payload distances of a freshly loaded model, in spawn pose."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    payload = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pyld")]
    distances = []
    i = 0
    while (bid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}")) >= 0:
        distances.append(float(np.linalg.norm(data.xpos[bid] - payload)))
        i += 1
    return np.array(distances)


class TestGeneratedXml:
    """The emitted MJCF parses and is wired up for every quadrotor count."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_parses_in_mujoco(self, mjcf_dir, nQ):
        path = mjcf_dir / f"m3_{nQ}.xml"
        multi_quad_pointmass(num_quadrotors=nQ, filename=str(path), verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))

        assert model.nu == 4 * nQ
        assert model.ntendon == nQ
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pyld") >= 0
        for i in range(nQ):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{nQ}") < 0

    @pytest.mark.parametrize("nQ", [1, 3])
    def test_actuators_live_in_one_block(self, mjcf_dir, nQ):
        """One merged ``<actuator>`` section, not one sibling block per quad."""
        path = mjcf_dir / f"m3_{nQ}.xml"
        multi_quad_pointmass(num_quadrotors=nQ, filename=str(path), verbose=False)
        root = ET.parse(path).getroot()
        blocks = root.findall("actuator")
        assert len(blocks) == 1
        assert len(blocks[0].findall("motor")) == 4 * nQ

    def test_masses_reach_the_model(self, mjcf_dir):
        path = mjcf_dir / "m3_masses.xml"
        multi_quad_pointmass(
            num_quadrotors=3,
            filename=str(path),
            payload_mass=0.31,
            quad_mass=1.23,
            verbose=False,
        )
        model = mujoco.MjModel.from_xml_path(str(path))
        pyld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pyld")
        assert model.body_mass[pyld] == pytest.approx(0.31)
        for i in range(3):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"quad{i}")
            assert model.body_mass[bid] == pytest.approx(1.23)

    def test_inertia_scales_with_the_requested_mass(self, mjcf_dir):
        """A heavier quadrotor is heavier *everywhere*, not just in the chassis.

        The geometry is fixed, so scaling every geom mass by the same factor is
        a uniform density change and the derived inertia must scale linearly.
        The retired construction — a fixed rotor assembly plus a chassis that
        absorbed the difference — turned a 3.3x mass into only a 1.3-1.8x
        inertia, because the propeller discs at the arm tips dominate.
        """
        inertias = {}
        for mass in (0.9, 3.0):
            path = mjcf_dir / f"m3_inertia_{mass}.xml"
            multi_quad_pointmass(
                num_quadrotors=1, filename=str(path), quad_mass=mass, verbose=False
            )
            model = mujoco.MjModel.from_xml_path(str(path))
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad0")
            assert model.body_mass[bid] == pytest.approx(mass)
            inertias[mass] = np.array(model.body_inertia[bid])
        np.testing.assert_allclose(inertias[3.0], (3.0 / 0.9) * inertias[0.9], rtol=1e-9)


class TestTautSpawn:
    """The generated formation starts taut — the zL regression."""

    def test_payload_height_is_derived_not_literal(self):
        expected = DEFAULT_SPAWN_HEIGHT - np.sqrt(
            DEFAULT_CABLE_LENGTH**2 - DEFAULT_FORMATION_RADIUS**2
        )
        assert payload_height() == pytest.approx(expected, abs=TOL_ANALYTIC)
        # The retired literal; keep it named so the regression stays legible.
        assert abs(payload_height() - 1.34) > 0.2

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    @pytest.mark.parametrize(
        ("cable_length", "radius"),
        [(1.0, 0.5), (1.5, 0.5), (1.0, 0.25), (2.0, 1.9)],
    )
    def test_cables_start_taut(self, mjcf_dir, nQ, cable_length, radius):
        path = mjcf_dir / f"m3_{nQ}_{cable_length}_{radius}.xml"
        multi_quad_pointmass(
            num_quadrotors=nQ,
            filename=str(path),
            cable_length=cable_length,
            formation_radius=radius,
            spawn_height=3.0,
            verbose=False,
        )
        distances = _quad_payload_distances(path)
        assert distances.shape == (nQ,)
        np.testing.assert_allclose(distances, cable_length, atol=TOL_ANALYTIC)

    def test_tendons_sit_exactly_at_their_limit(self, mjcf_dir):
        path = mjcf_dir / "m3_tendon.xml"
        multi_quad_pointmass(num_quadrotors=4, filename=str(path), cable_length=1.25, verbose=False)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(model.tendon_range[:, 1], 1.25, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(data.ten_length, 1.25, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", [2, 3, 4])
    def test_model_spawns_taut(self, nQ):
        """Through the model: every cable is at its length the moment it loads."""
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        mdl.reset()
        for i in range(nQ):
            d = np.linalg.norm(mdl.state.payload_position - mdl.state.positions[i])
            assert d == pytest.approx(mdl.cable_lengths[i], abs=TOL_ANALYTIC)

    def test_model_stays_taut_after_translation(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        mdl.reset(payload_position=np.array([-1.0, 2.0, 0.5]))
        for i in range(3):
            d = np.linalg.norm(mdl.state.payload_position - mdl.state.positions[i])
            assert d == pytest.approx(mdl.cable_lengths[i], abs=TOL_ANALYTIC)


class TestConfigHash:
    """A differing config regenerates; an identical one reuses the cache."""

    _BASE = {
        "cable_length": 1.05,
        "payload_mass": 0.17,
        "quad_mass": 0.95,
        "formation_radius": 0.45,
        "spawn_height": 2.25,
    }

    def _name(self, nQ=3, **overrides):
        cfg = {**self._BASE, **overrides}
        return MultiQuadCsPayloadMujoco._ensure_xml(
            nQ,
            cfg["cable_length"],
            cfg["payload_mass"],
            cfg["quad_mass"],
            cfg["formation_radius"],
            cfg["spawn_height"],
        )

    def test_identical_config_reuses_the_same_file(self, scratch_assets):
        first = self._name()
        stat = (ASSET_DIR / first).stat()
        second = self._name()
        assert first == second
        assert first.startswith("multi_quad_pointmass_3q_")
        # Reused, not rewritten.
        assert (ASSET_DIR / first).stat().st_mtime_ns == stat.st_mtime_ns

    @pytest.mark.parametrize(
        "override",
        [
            {"cable_length": 1.15},
            {"payload_mass": 0.27},
            {"quad_mass": 1.05},
            {"formation_radius": 0.35},
            {"spawn_height": 2.75},
        ],
    )
    def test_each_parameter_changes_the_hash(self, scratch_assets, override):
        assert self._name() != self._name(**override)

    def test_quadrotor_count_changes_the_name(self, scratch_assets):
        assert self._name(nQ=2) != self._name(nQ=3)

    def test_generator_version_changes_the_hash(self, scratch_assets, monkeypatch):
        """A release that reshapes the XML must not reuse a warm cache.

        Observed live: a source change kept the same hash, the stale XML was
        reloaded and three tests failed until the asset was deleted by hand.
        The real-world form of that is a user upgrading ``udaan`` on top of a
        populated asset directory.
        """
        first = self._name()
        monkeypatch.setattr(mj_module, "MJCF_GENERATOR_VERSION", MJCF_GENERATOR_VERSION + 1)
        assert self._name() != first

    def test_generated_files_differ_in_content(self, scratch_assets):
        base = ASSET_DIR / self._name()
        other = ASSET_DIR / self._name(cable_length=1.15)
        assert base.read_text() != other.read_text()
        np.testing.assert_allclose(_quad_payload_distances(base), 1.05, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(_quad_payload_distances(other), 1.15, atol=TOL_ANALYTIC)


class TestGeometryGuards:
    """Infeasible geometry is rejected, not turned into ``sqrt`` of a negative."""

    @pytest.mark.parametrize("cable_length", [0.5, 0.4, 0.0])
    def test_cable_shorter_than_radius_raises(self, cable_length):
        with pytest.raises(ValueError, match="strictly greater than the formation"):
            payload_height(cable_length=cable_length, formation_radius=0.5)

    def test_generator_rejects_infeasible_geometry(self, mjcf_dir):
        with pytest.raises(ValueError, match="strictly greater than the formation"):
            multi_quad_pointmass(
                num_quadrotors=3,
                filename=str(mjcf_dir / "never.xml"),
                cable_length=0.5,
                formation_radius=0.5,
                verbose=False,
            )
        assert not (mjcf_dir / "never.xml").exists()

    def test_model_rejects_infeasible_geometry(self):
        """Three independent re-derivations of the same ring agree it is infeasible.

        Since the M3 payload controller became the model's default,
        ``_init_default_controllers()`` builds a
        :class:`~udaan.flatness.RingTensionSplit` during ``super().__init__()``
        and *its* guard trips first — before the MJCF is ever generated, so the
        generator's own message is no longer what surfaces here. Pin the one
        that does: a looser matcher would accept any geometry ``ValueError`` at
        all, including one raised for an unrelated reason.
        """
        with pytest.raises(
            ValueError, match=r"formation radius 0\.5 must be strictly below every cable length"
        ) as excinfo:
            MultiQuadCsPayloadMujoco(
                render=False, num_quadrotors=3, cable_lengths=0.4, formation_radius=0.5
            )
        assert "0.4" in str(excinfo.value)

    def test_single_quadrotor_ignores_the_radius(self, mjcf_dir):
        """One quadrotor hovers on the axis, so a wide radius is not infeasible."""
        path = mjcf_dir / "m3_single.xml"
        multi_quad_pointmass(
            num_quadrotors=1,
            filename=str(path),
            cable_length=0.4,
            formation_radius=DEFAULT_FORMATION_RADIUS,
            verbose=False,
        )
        np.testing.assert_allclose(_quad_payload_distances(path), 0.4, atol=TOL_ANALYTIC)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="formation_radius must be >= 0"):
            payload_height(formation_radius=-0.1)

    @pytest.mark.parametrize("nQ", [0, -1])
    def test_non_positive_quadrotor_count_raises(self, mjcf_dir, nQ):
        with pytest.raises(ValueError, match="num_quadrotors must be >= 1"):
            multi_quad_pointmass(
                num_quadrotors=nQ, filename=str(mjcf_dir / "never.xml"), verbose=False
            )

    @pytest.mark.parametrize("quad_mass", [0.049, 0.001, 1e-7])
    def test_quad_mass_below_the_floor_raises(self, mjcf_dir, quad_mass):
        """A positive mass is not enough — there is a documented minimum."""
        with pytest.raises(ValueError, match="must be at least"):
            multi_quad_pointmass(
                num_quadrotors=2,
                filename=str(mjcf_dir / "never.xml"),
                quad_mass=quad_mass,
                verbose=False,
            )

    def test_payload_below_the_ground_raises(self, mjcf_dir):
        """A subterranean payload is as fatal as an infeasible triangle."""
        with pytest.raises(ValueError, match="at or below the ground plane"):
            multi_quad_pointmass(
                num_quadrotors=3,
                filename=str(mjcf_dir / "never.xml"),
                cable_length=1.9,
                spawn_height=0.5,
                verbose=False,
            )
        assert not (mjcf_dir / "never.xml").exists()

    def test_inertias_at_construction_raise(self):
        """MuJoCo derives inertia from geometry, so a kwarg would be overwritten."""
        with pytest.raises(ValueError, match="inertias"):
            MultiQuadCsPayloadMujoco(
                render=False, num_quadrotors=3, inertias=np.diag([0.01, 0.01, 0.02])
            )

    def test_inertias_can_be_assigned_after_construction(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        mdl.inertias = np.diag([0.01, 0.01, 0.02])
        np.testing.assert_allclose(mdl.inertias[0], np.diag([0.01, 0.01, 0.02]))
        for body_id in mdl._quad_body_id:
            np.testing.assert_allclose(
                mdl._mjMdl.model.body_inertia[body_id], [0.01, 0.01, 0.02], atol=TOL_ANALYTIC
            )

    @pytest.mark.parametrize("cable_lengths", [[1.0, 1.0000001, 1.0], [1.0, 1.00002, 1.0]])
    def test_near_uniform_parameters_are_not_silently_flattened(self, cable_lengths):
        """The uniformity check is exact — no tolerance to slip a value through."""
        with pytest.raises(ValueError, match="uniform"):
            MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3, cable_lengths=cable_lengths)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cable_lengths": [1.0, 1.2, 1.0]},
            {"masses": [0.9, 1.1, 0.9]},
        ],
    )
    def test_non_uniform_construction_parameters_raise(self, kwargs):
        """One MJCF encodes one ring and one tendon length — say so loudly."""
        with pytest.raises(ValueError, match="uniform"):
            MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3, **kwargs)


class TestDefaultsReachTheModel:
    """The packaged defaults, not stray literals, drive the generated asset."""

    def test_model_parameters_match_defaults(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        np.testing.assert_allclose(mdl.masses, DEFAULT_QUAD_MASS, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(mdl.cable_lengths, DEFAULT_CABLE_LENGTH, atol=TOL_ANALYTIC)
        assert mdl.payload_mass == pytest.approx(DEFAULT_PAYLOAD_MASS)
