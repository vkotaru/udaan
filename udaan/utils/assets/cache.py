"""Where MJCF assets live, and how generated ones get written safely.

Two directories, and the difference matters:

* :data:`PACKAGED_MJCF_DIR` — the hand-written MJCF that ships inside the
  installed package (``quadrotor_mj.xml``, ``scene.xml``, …). Read-only in
  principle and in practice: a wheel installed into a system ``site-packages``,
  a container image, or a read-only mount cannot be written to at all.
* :func:`generated_mjcf_dir` — a per-user cache for the MJCF the generators
  *emit*. Every fleet, multi-quad, flexcable and hose asset is generated on
  demand from constructor parameters and cached under a config-hash filename.

Those were the same directory until this pass, which meant a plain
``pip install udaan`` followed by ``MultiQuadCsPayloadMujoco(num_quadrotors=4)``
tried to write into ``site-packages``. On a read-only install that is a
``PermissionError`` from the constructor; where it did succeed it left build
artefacts inside the installed package, which is why every generated pattern
had to be listed in ``.gitignore`` to keep a source checkout clean.

The cache location, in order:

1. ``$UDAAN_ASSET_DIR`` if set — an explicit override, used verbatim.
2. The platform user cache: ``$XDG_CACHE_HOME`` or ``~/.cache`` on Linux,
   ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows — then
   ``udaan/mjcf`` under it. This is the ``platformdirs`` layout, hand-rolled so
   the package gains no dependency for twenty lines.

The config-hash naming scheme is unchanged, and so is its guarantee: a filename
encodes every parameter that shapes the XML *and* ``MJCF_GENERATOR_VERSION``,
so a stale file can never be mistaken for a current one.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from ..logging import get_logger

_logger = get_logger(__name__)

__all__ = [
    "PACKAGED_MJCF_DIR",
    "SCENE_XML",
    "ensure_generated",
    "ensure_scene_beside",
    "generated_mjcf_dir",
    "resolve_mjcf",
]

#: Hand-written MJCF shipped inside the package. Treat as read-only.
PACKAGED_MJCF_DIR = Path(__file__).resolve().parents[2] / "models" / "assets" / "mjcf"

#: The shared visual scene every generated asset includes. MuJoCo resolves an
#: ``<include>`` relative to the including file's own directory, so a generated
#: asset only loads from a directory that also holds a copy of this.
SCENE_XML = "scene.xml"


def _read_umask() -> int:
    """The process umask. Reading it requires setting it, so it is restored at once."""
    current = os.umask(0)
    os.umask(current)
    return current


def _user_cache_root() -> Path:
    """Platform user-cache base directory, ``platformdirs``-style."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def generated_mjcf_dir() -> Path:
    """Directory generated MJCF is written to and read from.

    Honours ``$UDAAN_ASSET_DIR``; otherwise the platform user cache. Created on
    demand — this is the only place in the package that makes a directory.
    """
    override = os.environ.get("UDAAN_ASSET_DIR")
    directory = Path(override) if override else _user_cache_root() / "udaan" / "mjcf"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_mjcf(model_path: str | os.PathLike) -> str:
    """Absolute path of an MJCF named by bare filename, packaged or generated.

    An argument that is already a path to an existing file is returned as-is,
    so callers holding a full path (a user's own asset, a test fixture) keep
    working. A bare filename is looked up in the packaged directory first, then
    the generated cache.

    Raises:
        OSError: if the file is in neither location.
    """
    candidate = Path(model_path)
    if candidate.is_absolute() or candidate.parent != Path():
        if candidate.exists():
            return str(candidate)
        raise OSError(f"File {candidate} does not exist")

    packaged = PACKAGED_MJCF_DIR / candidate.name
    if packaged.exists():
        return str(packaged)
    generated = generated_mjcf_dir() / candidate.name
    if generated.exists():
        return str(generated)
    raise OSError(
        f"MJCF {candidate.name!r} is in neither {PACKAGED_MJCF_DIR} nor {generated.parent}"
    )


def ensure_scene_beside(target: str | os.PathLike) -> None:
    """Put a copy of :data:`SCENE_XML` next to ``target``, if not already there.

    Every generated asset opens with ``<include file="scene.xml"/>`` and MuJoCo
    resolves that relative to the including file, so an asset written anywhere
    but the packaged directory cannot be loaded until the scene travels with
    it. That is true of the generated cache and equally of a path a user hands
    to ``udaan generate-xml -o``.
    """
    directory = Path(target).resolve().parent
    destination = directory / SCENE_XML
    if destination.exists():
        return
    source = PACKAGED_MJCF_DIR / SCENE_XML
    if not source.exists():  # pragma: no cover - the packaged scene is always shipped
        _logger.warning("packaged %s is missing; %s will not load", SCENE_XML, target)
        return
    _atomic_copy(source, destination)


def ensure_generated(filename: str, generate: Callable[[str], object]) -> str:
    """Return the path of a cached generated MJCF, writing it if absent.

    Args:
        filename: bare config-hash filename, e.g. ``multi_quad_hose_4l_2q_ab12cd34.xml``.
        generate: called with the *destination path* when the file has to be
            built. It must write that path.

    The write is **atomic**: ``generate`` is pointed at a private temporary file
    in the same directory, which is then ``os.replace``d into place. The old
    exists-then-write was a race — two processes constructing the same model
    concurrently could have one reading a half-written file while the other was
    still writing it. ``os.replace`` is atomic on POSIX and Windows alike, and
    because the filename is a hash of the configuration both processes are
    writing identical bytes, so whichever lands last is correct.

    The staging file is checked non-empty before it is published, because
    ``mkstemp`` has already created it: a ``generate`` that returns without
    writing would otherwise publish a 0-byte XML, and :func:`resolve_mjcf`'s
    ``destination.exists()`` makes that **permanent** — every subsequent
    construction of the same model finds the cache warm and fails in MuJoCo
    with ``ParseXML: empty file``, with nothing short of deleting the cache to
    recover. No caller reaches that today; the check costs one ``stat``.

    Raises:
        OSError: if ``generate`` left nothing at the path it was given.
    """
    directory = generated_mjcf_dir()
    destination = directory / filename
    if destination.exists():
        return str(destination)

    ensure_scene_beside(destination)
    handle, staging = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    os.close(handle)
    try:
        generate(str(staging))
        if Path(staging).stat().st_size == 0:
            raise OSError(f"the generator for {filename!r} wrote nothing; refusing to cache it")
        _make_readable(Path(staging))
        Path(staging).replace(destination)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(staging).unlink()
        raise
    return str(destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy ``source`` onto ``destination`` through a same-directory temporary."""
    handle, staging = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        shutil.copyfile(source, staging)
        _make_readable(Path(staging))
        Path(staging).replace(destination)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(staging).unlink()
        raise


#: Permissions a cached asset is published with: what an ordinary file creation
#: would have given, i.e. ``0o666`` masked by the process umask.
#:
#: ``tempfile.mkstemp`` deliberately creates ``0o600`` and ``Path.replace``
#: preserves it, which is right for a temporary and wrong for a cache. A cache
#: is *shared*: a common ``$UDAAN_ASSET_DIR`` on a multi-user machine, or the
#: standard container pattern of warming the cache during the build as root and
#: then running as an unprivileged UID, both get ``PermissionError`` on **read**
#: at ``0o600``. The umask is read once, at import, rather than per write:
#: reading it means setting it, and doing that repeatedly would race with any
#: other thread creating a file.
_CACHED_FILE_MODE = 0o666 & ~_read_umask()


def _make_readable(path: Path) -> None:
    """Relax a ``mkstemp`` temporary from ``0o600`` to :data:`_CACHED_FILE_MODE`."""
    with contextlib.suppress(OSError):  # pragma: no cover - chmod is a no-op on some filesystems
        path.chmod(_CACHED_FILE_MODE)
