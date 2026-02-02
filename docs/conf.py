import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # ajoute le dossier parent pour trouver automl

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # si tu utilises docstrings Google/NumPy style
]

# Si certains imports posent encore problème
autodoc_mock_imports = []

# Optionnel : tu peux mettre ton thème
html_theme = "alabaster"
# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_mock_imports = []

# Thème moderne
html_theme = "furo"
