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
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",  # render type hints in signatures
    "myst_parser",  # allow Markdown alongside reST
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "signature"
autodoc_class_signature = "separated"
autosummary_generate = True
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
html_title = f"Udaan {release}"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
