.. _results:

Résultats
=========

Les résultats de la simulation principale de prédiction de la RUL (via l'indicateur de santé et l'ajustement exponentiel) sont générés de manière itérative dans la cellule [13] du notebook.
Chaque itération de la boucle de simulation produit :

1.  Un **graphique** montrant :
    * L'indicateur de santé (PC1) calculé jusqu'au cycle de prédiction actuel.
    * La courbe exponentielle ajustée aux données récentes de l'indicateur de santé.
    * Le seuil de défaillance.
    * Le point de défaillance prédit (``fail_cycle``).

2.  Une **entrée dans le journal `log`** qui stocke le cycle de prédiction actuel (``prediction_cycle``) et le cycle de défaillance prédit (``fail_cycle``).

Après l'exécution de la boucle de simulation, la cellule [15] traite ce journal ``log`` pour créer un DataFrame pandas ``df``. Ce DataFrame contient :

* ``time`` (cycle) : Le cycle auquel la prédiction a été faite.
* ``prediction`` : Le cycle de défaillance prédit.
* ``isvalid`` : Un booléen indiquant si la prédiction est considérée comme valide (par exemple, si le ``fail_cycle`` prédit est inférieur à la fin de vie connue de l'ensemble de test, codé comme ``< 2156`` dans le notebook).
* ``real`` : Une chaîne de caractères indiquant l'état réel supposé du roulement à ce cycle, basé sur des plages de cycles codées en dur (par exemple, "early", "normal", "suspect", "Inner_race_failure"). Ceci permet une comparaison qualitative des prédictions par rapport aux étapes de défaillance connues.

L'affichage de ``df.head()`` (cellule [16]) et de ``df.tail()`` (implicitement à la fin de la cellule [15]) permet d'inspecter ces résultats tabulaires.

Pour les analyses exploratoires avec ARIMA et les modèles d'apprentissage profond, les résultats sont présentés sous forme de :

* **Modèle ARIMA (Section 5.6) :**
    * Métriques d'évaluation (MSE, RMSE, MAE, R²) pour les prédictions sur l'échelle différenciée et sur l'échelle originale.
    * Graphiques comparant les valeurs prédites aux valeurs réelles.

* **Modèles d'Apprentissage Profond (Section 6.4) :**
    * Graphiques comparant les prédictions des modèles RNN, LSTM et GRU aux valeurs réelles de l'indicateur de santé sur l'ensemble de test.

Ces résultats permettent d'évaluer la performance de l'approche principale de prédiction de RUL et des méthodes exploratoires de prévision de séries temporelles.
