# conf.py

# -- Path setup --------------------------------------------------------------
# Si vos modules Python (ou votre notebook, si nbsphinx doit le trouver via un import)
# se trouvent dans un répertoire parent ou un autre répertoire, ajoutez ces chemins ici.
# import os
# import sys
# sys.path.insert(0, os.path.abspath('..')) # Si votre notebook est à la racine du projet

# -- Renseignements sur le projet -----------------------------------------------------
project = 'Analyse de Séries Temporelles de Roulements'
copyright = '2025, Votre Nom ou Organisation' # Remplacez par vos informations
author = 'Votre Nom ou Organisation' # Remplacez par vos informations

# La version courte X.Y
version = '0.1'
# La version complète, incluant les tags alpha/beta/rc
release = '0.1.0'


# -- Configuration générale ---------------------------------------------------

# Ajoutez ici les noms des modules d'extension Sphinx, sous forme de chaînes.
# Ils peuvent être des extensions venant avec Sphinx (nommées 'sphinx.ext.*')
# ou vos propres extensions.
extensions = [
    'sphinx.ext.autodoc',      # Inclure la documentation depuis les docstrings
    'sphinx.ext.napoleon',     # Support pour les docstrings Google et NumPy
    'sphinx.ext.viewcode',     # Ajouter des liens vers le code source en surbrillance
    'sphinx.ext.intersphinx',  # Lien vers la documentation d'autres projets
    'sphinx_rtd_theme',      # Thème Read the Docs
    'nbsphinx',                # Pour inclure les notebooks Jupyter
    'sphinx.ext.mathjax',      # Pour le rendu des équations LaTeX (si nécessaire)
    # 'myst_parser',           # Si vous voulez utiliser Markdown à la place de reStructuredText pour certaines pages
]

# Configuration pour nbsphinx
nbsphinx_execute = 'never'  # Options: 'always', 'never', 'auto'. 'never' est plus sûr pour les notebooks longs à exécuter ou avec des dépendances complexes.
nbsphinx_allow_errors = True # Permet de continuer la construction même si une cellule du notebook génère une erreur

# Configuration pour intersphinx : exemple de liens vers la documentation d'autres projets
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', None), # Vérifiez le lien exact pour TF
}

# Ajoutez ici les chemins des templates, relatifs à ce répertoire.
templates_path = ['_templates']

# Le suffixe des fichiers source.
# Vous pouvez spécifier une chaîne ou une liste de chaînes.
# source_suffix = '.rst'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown', # Si vous utilisez myst_parser
}

# Le nom du document maître (sans l'extension .rst)
# Pour Sphinx < 2.0, utilisez master_doc = 'index'
# Pour Sphinx >= 2.0, root_doc est préféré mais master_doc est toujours supporté.
root_doc = 'index' # Ce sera votre fichier docs/index.rst ou docs/index.md

# La langue de la documentation.
language = 'fr'

# Liste des motifs, relatifs au répertoire source, qui correspondent
# à des fichiers et répertoires à ignorer lors de la recherche de fichiers source.
# Cette liste affecte aussi html_static_path et html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options pour la sortie HTML -------------------------------------------------

# Le thème à utiliser pour les pages d'aide HTML et HTML. Voir la documentation
# pour une liste des thèmes intégrés.
#
html_theme = 'sphinx_rtd_theme' # Thème populaire pour Read the Docs

# Ajoutez ici les chemins des fichiers statiques (comme les feuilles de style),
# relatifs à ce répertoire. Ils sont copiés après les fichiers statiques intégrés,
# donc un fichier nommé "default.css" écrasera le "default.css" intégré.
html_static_path = ['_static'] # Créez un répertoire docs/_static si vous avez des fichiers CSS personnalisés ou des images

# Si vous avez un fichier CSS personnalisé (par exemple, docs/_static/custom.css)
# def setup(app):
#     app.add_css_file('custom.css')


# -- Options pour la sortie LaTeX ---------------------------------------------
# (Si vous prévoyez de générer des PDF via LaTeX)

latex_elements = {
    # La taille du papier ('letterpaper' ou 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # La taille de la police ('10pt', '11pt' ou '12pt').
    #
    # 'pointsize': '10pt',

    # Options LaTeX supplémentaires.
    #
    # 'preamble': '',

    # Options pour le style des figures (si supporté par le thème).
    'figure_align': 'htbp',
}

# Documents LaTeX groupés (fichier source, nom du fichier de sortie, titre,
# auteur, classe de document [howto, manual, ou votre propre classe]).
latex_documents = [
    (root_doc, 'AnalyseSeriesTemporellesRoulements.tex', project,
     author, 'manual'),
]


# -- Options pour la sortie manuelle (man pages) ---------------------------------------

# Un document de page manuelle (fichier source, nom, description, auteurs,
# section de page manuelle).
man_pages = [
    (root_doc, 'analyseseriestemporellesroulements', project,
     [author], 1)
]


# -- Options pour la sortie Texinfo ----------------------------------------------

# Documents Texinfo groupés (fichier source, nom du fichier de sortie, titre,
# auteur, nom du répertoire, description, catégorie).
texinfo_documents = [
    (root_doc, 'AnalyseSeriesTemporellesRoulements', project,
     author, 'AnalyseSeriesTemporellesRoulements', "Analyse de séries temporelles de roulements et prédiction de défaillance.",
     'Miscellaneous'),
]

# -- Configuration pour Autodoc --------------------------------------------------
# Pour que autodoc trouve vos modules
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.')) # Si vos modules sont dans le répertoire docs
# sys.path.insert(0, os.path.abspath('..')) # Si vos modules sont à la racine du projet

# Trie les membres par type (par exemple, les méthodes après les attributs)
autodoc_member_order = 'groupwise'

# -- Configuration pour Napoleon (si vous utilisez des docstrings Google/NumPy) ----
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
