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
autosummary_generate = True
# Optional runtime deps that may be missing on the RTD builder.
autodoc_mock_imports = [
    "bokeh",
    "vpython",
    "mujoco",
    "scipy",
    "matplotlib",
    "optuna",
    "cmaes",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Markdown source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ── HTML (furo) ───────────────────────────────────────────────────────
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["fonts.css"]
html_title = f"Udaan {release}"

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
