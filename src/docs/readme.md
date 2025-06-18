## To rebuild the docs from root:

1. Install dependancies
```bash
pip install sphinx & sphinx_rtd_theme
```

2. quickstart:

```bash
sphinx-quickstart
```

3. Cd to the relevant place and setup the apidoc

```bash
cd ./src & sphinx-apidoc -o docs .
```

4. Edit the `conf.py` file:

```python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = ["sphinx_rtd_theme", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.autosummary"]
html_theme = 'sphinx_rtd_theme'
```

5. Add the rst files that you want to display in the `index.rst` file

6. run `make html`, with output path set in `docs/make.bat`:

```bash
make html
```