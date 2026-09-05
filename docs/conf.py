"""Sphinx configuration for the Udaan documentation.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# ── Project ───────────────────────────────────────────────────────────
project = "Udaan"
author = "vkotaru"
copyright = f"2023–2026, {author}"

try:
    release = _pkg_version("udaan")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# ── Extensions ────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Google / NumPy docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # HTML math
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",  # render type hints in signatures
    "myst_parser",  # allow Markdown alongside reST
    "sphinx_proof",  # {prf:theorem}, {prf:proof}, {prf:lemma}, ...
    "sphinxcontrib.bibtex",  # {cite}`key` via docs/refs.bib
]

# MyST: enable the math + amsmath extensions so $…$ / $$…$$ and \begin{align}
# work without per-file opt-in.
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "deflist",
    "colon_fence",
]
myst_heading_anchors = 3

# The repo README is formatted with centred HTML blocks (no top-level `# H1`),
# so it starts at `##`. That's intentional; don't warn about it.
suppress_warnings = [
    "myst.header",
]

# Bibliography — a single top-level .bib file for now.
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "signature"
autodoc_class_signature = "separated"

# Mock heavy / GUI / optional runtime deps during autodoc's introspection.
# These can deadlock on headless CI (vpython spawns a browser/WebGL context,
# mujoco + glfw open an OpenGL context at import, etc.) — mocking replaces
# them with stub modules so autodoc can walk udaan without real imports.
autodoc_mock_imports = [
    "bokeh",
    "vpython",
    "mujoco",
    "glfw",
    "pyopengl",
    "OpenGL",
    "imageio",
    "imageio_ffmpeg",
    "autobahn",
    "scipy",
    "matplotlib",
    "optuna",
    "cmaes",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Render ``Attributes:`` sections as inline ``:ivar:`` fields rather than as
# standalone ``py:attribute`` descriptions. Without this, every documented
# attribute is described twice — once by napoleon from the class docstring and
# once by autodoc from the dataclass field / enum member itself — which Sphinx
# reports as "duplicate object description of ...". Affects the flatness
# dataclasses (Flats / Inputs / RefState) and the core.types enums.
napoleon_use_ivar = True

_INTERSPHINX_TARGETS = {
    "python": "https://docs.python.org/3",
    "numpy": "https://numpy.org/doc/stable",
    "scipy": "https://docs.scipy.org/doc/scipy",
}

# Bound the inventory fetch so an unreachable host fails fast instead of hanging
# on the default connect timeout of None. This only bounds the *wait* — it does
# not stop the "failed to reach any of the inventories" warning, and under `-W`
# that warning is still a build failure. Determinism comes from _prefetch()
# below, not from this.
intersphinx_timeout = 5


def _prefetch_inventories(targets, timeout=intersphinx_timeout):
    """Resolve ``objects.inv`` to a local cache, dropping anything unreachable.

    ``sphinx-build -W`` promotes every warning to an error, and intersphinx
    warns — once per mapping entry — when it cannot reach an inventory. Left
    alone, that makes the docs job a coin flip: it passes with network and
    fails without, and it can fail intermittently *with* network on a slow or
    rate-limited host. No timeout setting fixes that; it only decides how long
    the flake takes.

    So the network is taken out of the build. Each inventory is fetched once,
    up front, into ``docs/_build/inventories/``; Sphinx is then pointed at the
    local copy, which it reads with no I/O that can fail. An inventory we
    cannot fetch is dropped from the mapping entirely rather than left in it to
    warn: the cross-references into it degrade to plain literal text, which is
    a cosmetic loss, while the alternative is a red build for a reason that has
    nothing to do with the documentation. Cross-reference *resolution* is not
    what the ``-W`` gate is there to check — broken toctrees, malformed
    docstrings and duplicate object descriptions are, and those are all local.
    """
    import pathlib
    import urllib.error
    import urllib.request

    cache = pathlib.Path(__file__).resolve().parent / "_build" / "inventories"
    mapping = {}
    for name, base in targets.items():
        local = cache / f"{name}.inv"
        if not local.exists():
            try:
                with urllib.request.urlopen(f"{base}/objects.inv", timeout=timeout) as response:
                    payload = response.read()
            except (urllib.error.URLError, OSError, ValueError):
                continue
            cache.mkdir(parents=True, exist_ok=True)
            local.write_bytes(payload)
        mapping[name] = (base, str(local))
    return mapping


intersphinx_mapping = _prefetch_inventories(_INTERSPHINX_TARGETS)

# Markdown source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ── HTML (furo) ───────────────────────────────────────────────────────
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["fonts.css", "sidebar-rtd.css"]
html_js_files = ["expand-sidebar.js"]
html_title = f"Udaan {release}"
html_logo = "_static/logo.png"
html_favicon = "_static/icon.png"

_font_stack = (
    '"Fustat", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)
_mono_stack = (
    '"JetBrains Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo, '
    'Consolas, "Liberation Mono", monospace'
)

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "font-stack": _font_stack,
        "font-stack--monospace": _mono_stack,
    },
    "dark_css_variables": {
        "font-stack": _font_stack,
        "font-stack--monospace": _mono_stack,
    },
}
