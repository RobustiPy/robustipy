import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RobustiPy'
copyright = '2024, Daniel Valdenegro Ibarra, Charles Rahal, and Jiani Yan'
author = 'Daniel Valdenegro Ibarra, Charles Rahal, and Jiani Yan'
release = 'v 0.1'

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