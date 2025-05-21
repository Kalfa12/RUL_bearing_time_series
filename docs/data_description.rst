.. _data_description:

Description des Données
=======================

Le principal jeu de données utilisé est ``features_1st_test.csv``.

* **Chargement :** Les données sont chargées dans un DataFrame pandas nommé ``set1``.

    .. code-block:: python

        set1 = pd.read_csv('/content/features_1st_test.csv')

* **Structure :** Le jeu de données contient 97 colonnes. La première colonne, initialement 'Unnamed: 0', représente des horodatages et est renommée ``'time'``. Les colonnes restantes sont diverses caractéristiques statistiques extraites des lectures de capteurs de 4 roulements (B1, B2, B3, B4) sur deux axes (x, y).
    * Exemples de caractéristiques : ``B1_x_mean``, ``B1_x_std``, ``B1_x_skew``, ``B1_x_kurtosis``, ``B1_x_entropy``, ``B1_x_rms``, ``B1_x_max``, ``B1_x_p2p``, etc.

* **Index Temporel :** La colonne ``'time'`` est initialement de type objet mais est utilisée comme index pour les graphiques et les références.

* **Cycles :** Le nombre total de points de données (cycles) est déterminé par la longueur du DataFrame (``last_cycle = int(len(set1))``).
