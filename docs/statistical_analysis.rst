.. _statistical_analysis:

Analyse Statistique (Exploratoire) : Modélisation ARIMA de l'Indicateur de Santé
================================================================================

Cette section explore la modélisation de l'Indicateur de Santé (PC1 complet du Roulement 3) avec ARIMA.

Indicateur de Santé pour l'Ensemble des Données
-----------------------------------------------

* L'HI (``degredation_0``) est calculé pour l'ensemble des données du roulement choisi (Cellule [19]) et tracé (Cellule [20]).

Tests de Stationnarité
----------------------

La stationnarité de ``degredation_0['PC1']`` est testée (Cellules [21]-[24]) :

* **Test ADF (``test_ADF``) :** Non-stationnaire (p-value ≈ 0.98).
* **Test PP (``test_PP``) :** Non-stationnaire (p-value ≈ 0.99).
* **Test KPSS (``test_KPSS``) :** Non-stationnaire (p-value ≈ 0.01).

Différenciation et Nouveaux Tests de Stationnarité
--------------------------------------------------

* La série est différenciée une fois : ``degredation_1 = degredation_0['PC1'].diff().dropna()`` (Cellule [25]).
* Les tests sur ``degredation_1`` (Cellules [26]-[28]) indiquent la stationnarité :
    * **ADF :** Stationnaire (p-value ≈ 0.03).
    * **PP :** Stationnaire (p-value ≈ 0.00).
    * **KPSS :** Stationnaire (p-value ≈ 0.10).

Graphiques ACF et PACF
----------------------

* Les graphiques ACF et PACF sont tracés pour ``degredation_0['PC1']`` (Cellule [29]) et ``degredation_1`` (Cellule [30]) pour aider à déterminer les ordres (p, q) du modèle ARIMA.

Modélisation ARIMA
------------------

* La série différenciée ``degredation_1`` est divisée en ensembles d'entraînement et de test (entraînement jusqu'à 1950 cycles, Cellule [32]).
* Un modèle ARIMA(2,0,2) est ajusté aux données d'entraînement (Cellule [33]).

    .. code-block:: python

        model = ARIMA(degredation_1_train, order=(2,0,2)) # (p,d,q)
        model_fit = model.fit()
        print(model_fit.summary())

* Les résidus sont analysés (Cellule [34]). Le test de Ljung-Box sur les résidus (lag=10) donne une p-value d'environ 0.0375, indiquant une possible autocorrélation résiduelle.

Évaluation du Modèle ARIMA
--------------------------

* **Échelle Différenciée :**
    * Les prédictions (``prediction_diff``) sont faites sur l'ensemble de test (Cellule [35]).
    * MSE ≈ 0.0058 (Cellule [37]).
    * Les prédictions sont tracées par rapport aux données réelles différenciées (Cellule [38]).
* **Échelle Originale :**
    * Les prédictions sont ramenées à l'échelle originale (``prediction_original``) par dé-différenciation (Cellules [39]-[40]).
    * Métriques d'évaluation (Cellules [41]-[42]) :
        * MSE ≈ 0.801437
        * RMSE ≈ 0.895230
        * MAE ≈ 0.753982
        * R² ≈ -0.201210 (indiquant une mauvaise performance sur l'échelle originale).
    * Les prédictions à l'échelle originale sont tracées (Cellules [43], [44]).
