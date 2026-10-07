"""Where generated MJCF is written, and whether it can be loaded from there.

Two defects this pins:

1. Generated assets used to be written into ``udaan/models/assets/mjcf``, i.e.
   inside the installed package. A wheel in a system ``site-packages``, a
   container image or a read-only mount cannot be written to, so constructing
   any generated-asset model raised ``PermissionError``; where it did succeed it
   left build artefacts inside the install, which is why six ``.gitignore``
   patterns existed to keep a source checkout clean.
2. Every generated asset opens with ``<include file="scene.xml"/>``, which
   MuJoCo resolves relative to the *including file*. Writing the XML without
   the scene produced a file that could not be loaded from where it was
   written — the symptom users hit through ``udaan generate-xml -o``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

from udaan.utils.assets import multi_quad_pointmass
from udaan.utils.assets.cache import (
    PACKAGED_MJCF_DIR,
    SCENE_XML,
    ensure_generated,
    ensure_scene_beside,
    generated_mjcf_dir,
    resolve_mjcf,
)

pytestmark = pytest.mark.mujoco


class TestCacheLocation:
    def test_the_generated_directory_is_not_inside_the_package(self):
        """The whole point: a read-only install must still be able to generate."""
        generated = generated_mjcf_dir().resolve()
        packaged = PACKAGED_MJCF_DIR.resolve()
        assert generated != packaged
        assert packaged not in generated.parents

    def test_the_env_override_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path / "assets"))
        assert generated_mjcf_dir() == tmp_path / "assets"
        assert (tmp_path / "assets").is_dir()

    def test_the_packaged_directory_holds_only_hand_written_assets(self):
        """No generated (config-hash) file should be shipped or left behind."""
        stale = [
            p.name
            for p in PACKAGED_MJCF_DIR.glob("*.xml")
            if p.name.startswith(
                (
                    "multi_quad_pointmass_",
                    "multi_quad_rigidbody_",
                    "multi_quad_hose_",
                    "quad_flexcable_",
                    "fleet_",
                    "cspayload_fleet_",
                )
            )
        ]
        assert not stale, f"generated assets left inside the package: {stale}"


class TestResolution:
    def test_a_packaged_name_resolves(self):
        assert resolve_mjcf(SCENE_XML) == str(PACKAGED_MJCF_DIR / SCENE_XML)

    def test_a_generated_name_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        (tmp_path / "made_up.xml").write_text("<mujoco/>")
        assert resolve_mjcf("made_up.xml") == str(tmp_path / "made_up.xml")

    def test_a_full_path_is_passed_through(self, tmp_path):
        target = tmp_path / "elsewhere.xml"
        target.write_text("<mujoco/>")
        assert resolve_mjcf(str(target)) == str(target)

    def test_an_unknown_name_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        with pytest.raises(OSError, match="neither"):
            resolve_mjcf("nothing_like_this.xml")


class TestSceneTravelsWithTheAsset:
    def test_a_generated_asset_loads_from_a_fresh_directory(self, tmp_path):
        """The ``generate-xml -o`` failure, reproduced and fixed.

        Before, the written file's ``<include file="scene.xml"/>`` pointed at
        nothing and ``mj_loadXML`` refused it.
        """
        target = tmp_path / "out.xml"
        multi_quad_pointmass(num_quadrotors=2, filename=str(target), verbose=False)
        assert (tmp_path / SCENE_XML).exists(), "scene.xml was not written beside the asset"
        model = mujoco.MjModel.from_xml_path(str(target))
        assert model.nbody > 1

    def test_the_cli_output_loads_from_where_it_was_written(self, tmp_path):
        """End to end through the CLI, since that is where users hit it."""
        target = tmp_path / "cli.xml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "udaan",
                "generate-xml",
                "multi-quad",
                "-n",
                "2",
                "-o",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert mujoco.MjModel.from_xml_path(str(target)).nbody > 1

    def test_it_does_not_clobber_an_existing_scene(self, tmp_path):
        scene = tmp_path / SCENE_XML
        scene.write_text("<mujoco><!-- mine --></mujoco>")
        ensure_scene_beside(tmp_path / "whatever.xml")
        assert "mine" in scene.read_text()


class TestAtomicGeneration:
    def test_the_file_appears_only_once_complete(self, tmp_path, monkeypatch):
        """A half-written file must never be visible under the final name.

        The old exists-then-write let a second process open a partial file
        while the first was still writing it.
        """
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        target = tmp_path / "staged.xml"

        def build(path):
            assert path != str(target), "generate() must write a staging path, not the destination"
            assert not target.exists(), "the destination appeared before the write finished"
            Path(path).write_text("<mujoco/>")

        assert ensure_generated("staged.xml", build) == str(target)
        assert target.read_text() == "<mujoco/>"

    def test_a_failed_build_leaves_nothing_behind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))

        def build(path):
            raise RuntimeError("generator blew up")

        with pytest.raises(RuntimeError, match="blew up"):
            ensure_generated("doomed.xml", build)
        assert not (tmp_path / "doomed.xml").exists()
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".doomed")]
        assert not leftovers, f"staging files left behind: {leftovers}"

    def test_a_cached_file_is_not_rebuilt(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        calls = []

        def build(path):
            calls.append(path)
            Path(path).write_text("<mujoco/>")

        first = ensure_generated("cached.xml", build)
        second = ensure_generated("cached.xml", build)
        assert first == second
        assert len(calls) == 1


#: ``os.geteuid`` does not exist on Windows, and this is evaluated at collection
#: time — ``cache.py`` supports ``win32``, so guarding the *call* is not enough.
_IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

#: POSIX-only permission bits; ``chmod`` on Windows only toggles the read-only
#: flag, so the modes these tests assert are meaningless there.
_HAS_POSIX_MODES = sys.platform != "win32"


class TestReadOnlyInstall:
    @pytest.mark.skipif(_IS_ROOT, reason="root ignores directory permissions")
    def test_generation_survives_an_unwritable_package_directory(self, tmp_path, monkeypatch):
        """The defect stated plainly: this used to need write access to the package."""
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        mode = PACKAGED_MJCF_DIR.stat().st_mode
        PACKAGED_MJCF_DIR.chmod(0o555)
        try:
            from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

            model = MultiQuadCsPayloadMujoco(num_quadrotors=2)
            assert str(tmp_path) in model._mjMdl.full_path
        finally:
            PACKAGED_MJCF_DIR.chmod(mode)


class TestGeneratedFilePermissions:
    """A cache is shared; ``mkstemp``'s ``0o600`` is not.

    ``tempfile.mkstemp`` creates ``0o600`` and ``Path.replace`` preserves it,
    so every cached asset used to land owner-only. That breaks the two ways a
    cache is actually shared: a common ``$UDAAN_ASSET_DIR`` on a multi-user
    machine, and the standard container pattern of warming the cache as root
    during the build and then running as an unprivileged UID — which gets a
    ``PermissionError`` on *read*.
    """

    @pytest.mark.skipif(not _HAS_POSIX_MODES, reason="POSIX permission bits")
    @pytest.mark.skipif(_IS_ROOT, reason="root reads regardless of mode")
    def test_a_generated_asset_is_world_readable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        path = Path(ensure_generated("perms.xml", lambda p: Path(p).write_text("<mujoco/>")))
        assert path.stat().st_mode & 0o004, oct(path.stat().st_mode & 0o777)

    @pytest.mark.skipif(not _HAS_POSIX_MODES, reason="POSIX permission bits")
    @pytest.mark.skipif(_IS_ROOT, reason="root reads regardless of mode")
    def test_the_scene_copied_beside_it_is_too(self, tmp_path, monkeypatch):
        """The scene travels with the asset, so it needs the same reach."""
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        ensure_generated("perms_scene.xml", lambda p: Path(p).write_text("<mujoco/>"))
        scene = tmp_path / SCENE_XML
        assert scene.exists()
        assert scene.stat().st_mode & 0o004, oct(scene.stat().st_mode & 0o777)


class TestAGeneratorThatWritesNothing:
    """A 0-byte cache entry would be permanent, so it is never published.

    ``mkstemp`` has already created the file when the callback runs, so a
    callback that returns without writing leaves a 0-byte XML — and
    ``ensure_generated``'s own ``destination.exists()`` short-circuit then makes
    it permanent: every later construction of the same model finds the cache
    warm and dies in MuJoCo with ``ParseXML: empty file``, with no recovery
    short of deleting the cache by hand. Not reachable through today's six call
    sites, all of which funnel through ``save_to``'s unconditional write.
    """

    def test_an_empty_result_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        with pytest.raises(OSError, match="wrote nothing"):
            ensure_generated("empty.xml", lambda path: None)

    def test_nothing_is_cached_and_nothing_is_left_behind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        with pytest.raises(OSError):
            ensure_generated("empty.xml", lambda path: None)
        assert not (tmp_path / "empty.xml").exists()
        assert not [f for f in tmp_path.iterdir() if f.name.startswith(".empty.xml.")]

    def test_the_next_attempt_still_succeeds(self, tmp_path, monkeypatch):
        """Self-healing is the point: a failed build must not poison the cache."""
        monkeypatch.setenv("UDAAN_ASSET_DIR", str(tmp_path))
        with pytest.raises(OSError):
            ensure_generated("retry.xml", lambda path: None)
        path = ensure_generated("retry.xml", lambda p: Path(p).write_text("<mujoco/>"))
        assert Path(path).read_text() == "<mujoco/>"
