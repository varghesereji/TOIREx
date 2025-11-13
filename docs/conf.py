from pathlib import Path
import sys
sys.path.insert(0, Path("../src"))  # points to src/ so it can find packagename


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TOIREx'
copyright = '2025, Varghese Reji'
author = 'Varghese Reji'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'
html_static_path = ['_static']

# logo file

html_logo = '_static/toirex_logo.png'
extensions = [
    "sphinx.ext.autodoc",      # auto-generate docs from docstrings
    "sphinx.ext.napoleon",     # support NumPy/Google style docstrings
    "sphinx.ext.viewcode",     # link to source code
]
