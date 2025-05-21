.. _usage:

Utilisation
===========

Pour utiliser ce projet et exécuter le notebook Jupyter ``projet_Time_series.ipynb`` :

1.  **Configuration de l'Environnement :**
    * Assurez-vous que toutes les bibliothèques Python listées dans la section :ref:`setup` sont installées. Si vous utilisez Google Colab, la plupart sont préinstallées, mais ``arch`` pourrait nécessiter une installation (``!pip install arch``).

2.  **Obtention des Données :**
    * Le projet nécessite le fichier ``features_1st_test.csv``.
    * **Vous devez télécharger ce fichier depuis sa source d'origine.** *(Précisez ici la source du fichier pour les utilisateurs.)*

3.  **Chargement des Données dans Google Colab (si applicable) :**
    * Si vous exécutez dans Google Colab :
        * Téléversez ``features_1st_test.csv`` dans votre session Colab (via l'onglet "Fichiers" à gauche, puis "Téléverser").
        * Le chemin ``/content/features_1st_test.csv`` utilisé dans le notebook (Cellule [2]) correspond à cet emplacement.

4.  **Exécution du Notebook :**
    * Ouvrez le notebook dans votre environnement Jupyter.
    * Exécutez les cellules séquentiellement.
    * **Prédiction RUL Principale :** Cellules [1] à [16]. La cellule [13] (boucle de simulation) est conçue pour être exécutée plusieurs fois.
    * **Analyse Statistique :** Cellules [17] à [44].
    * **Modèles d'Apprentissage Profond :** Cellules [45] à [52].

5.  **Interprétation des Résultats :**
    * **Simulation RUL :** Les graphiques de la cellule [13] et le DataFrame ``df`` de la cellule [15] permettent d'analyser les prédictions de RUL.
    * **Analyses ARIMA et Deep Learning :** Ces sections fournissent des métriques d'évaluation et des graphiques pour comparer les approches alternatives.
