import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RobustiPy'
copyright = ('2025, Daniel Valdenegro Ibarra, Jiani Yan, Duiyi Dai, and Charles Rahal')
author = 'Daniel Valdenegro Ibarra, Jiani Yan, Duiyi Dai, and Charles Rahal'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_static_path = ['_static']

sys.path.insert(0, os.path.abspath('..'))

extensions = ["sphinx_rtd_theme", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.autosummary"]
html_theme = 'sphinx_rtd_theme'