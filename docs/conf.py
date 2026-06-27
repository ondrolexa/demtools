"""Sphinx configuration for demtools documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------

project = "demtools"
copyright = "2024–2026, Ondrej Lexa"
author = "Ondrej Lexa"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_mdinclude",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = []
exclude_patterns = ["_build", "build", "**.ipynb_checkpoints"]

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

suppress_warnings = ["config.cache"]
