.. _deep_learning_models:

Modèles d'Apprentissage Profond (Deep Learning) : Prédiction de l'Indicateur de Santé
===================================================================================================

Cette section explore les RNN (SimpleRNN, LSTM, GRU) pour la prédiction de l'HI.

Préparation des Données : Fenêtrage
-----------------------------------

* La fonction ``fenetre(df, w)`` crée des séquences (fenêtres) pour les modèles RNN (Cellule [45]). Avec ``w=4``, des séquences de 3 pas de temps en entrée (``X``) et la valeur suivante en sortie (``y``) sont créées à partir de ``degredation_0['PC1']``.
* Les données sont divisées en ensembles d'entraînement et de test (Cellule [46]).

Architectures des Modèles
-------------------------

Des modèles SimpleRNN, LSTM et GRU sont définis avec 64 unités dans la couche récurrente et une couche Dense de sortie (Cellules [47]-[49]).
L' ``input_shape`` est définie comme ``(None, 1)`` pour accepter des séquences de longueur variable avec 1 feature.

Entraînement des Modèles
------------------------

* Chaque modèle est compilé avec l'optimiseur 'adam' et la fonction de perte 'mse'.
* L'entraînement est effectué sur ``X_train`` et ``Y_train`` pour 100 époques avec une taille de lot de 32 (Cellule [50]).

Prédiction et Visualisation
---------------------------

* Les prédictions (``Prd``) sont faites sur ``X_test`` pour chaque modèle (Cellule [51]).
* Les prédictions des trois modèles sont tracées par rapport aux données d'entraînement et de test réelles (Cellule [52]).
