# Configuration file for the Sphinx documentation builder.
#
# For the full list of options, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

package_path = os.path.abspath("../")
sys.path.insert(0, package_path)
os.environ["PYTHONPATH"] = os.pathsep.join(
    (package_path, os.environ.get("PYTHONPATH", ""))
)

# -- Project information -----------------------------------------------------

project = "hinode"
copyright = "2026, Roy T. Smart"
author = "Roy T. Smart"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "jupyter_sphinx",
]
autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autodoc_typehints = "description"

graphviz_output_format = "png"
inheritance_graph_attrs = dict(rankdir="TB")

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sun-data/hinode",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/hinode",
            "icon": "fa-brands fa-python",
        },
    ],
}

# https://github.com/readthedocs/readthedocs.org/issues/2569
master_doc = "index"

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "named_arrays": ("https://named-arrays.readthedocs.io/en/stable/", None),
    "xrtpy": ("https://xrtpy.readthedocs.io/en/stable/", None),
}
