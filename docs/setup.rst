.. _setup:

Configuration et Installation
=============================

Bibliothèques Utilisées
-----------------------

Le projet s'appuie sur plusieurs bibliothèques Python:

* **numpy :** Pour les opérations numériques.
* **matplotlib.pyplot :** Pour la création de graphiques et la visualisation.
* **pandas :** Pour la manipulation et l'analyse de données.
* **tensorflow :** Pour la construction et l'entraînement des modèles d'apprentissage profond (RNN, LSTM, GRU).
* **scikit-learn :** Pour l'ACP (`sklearn.decomposition.PCA`) et le prétraitement des données (`sklearn.preprocessing.StandardScaler`).
* **scipy :** Pour l'ajustement de courbes (`scipy.optimize.curve_fit`).
* **statsmodels :** Pour les tests statistiques (ADF, KPSS) et la modélisation ARIMA.
* **arch :** Pour le test de stationnarité de Phillips-Perron.
* **seaborn :** Pour des visualisations statistiques améliorées.

Vous pouvez les installer avec pip :

.. code-block:: bash

    pip install numpy matplotlib pandas tensorflow scikit-learn scipy statsmodels arch seaborn
