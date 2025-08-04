# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Insee-stats-DP'
copyright = '2025, Stuart BENOLIEL, Gaspard FEREY, Julien JAMME'
author = 'Stuart BENOLIEL, Gaspard FEREY, Julien JAMME'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Enables Google and NumPy style
    'sphinx_autodoc_typehints',
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
  "github_url": "https://github.com/InseeFrLab/diff-privacy-poc",
  "use_edit_page_button": False,
}
html_static_path = ['_static']
