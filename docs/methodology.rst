.. _methodology:

Méthodologie : Prédiction de la RUL via Indicateur de Santé et Ajustement Exponentiel
====================================================================================

Cette section détaille le pipeline principal pour la prédiction de la Durée de Vie Utile Résiduelle (RUL) d'un roulement.

Chargement des Données et Prétraitement Initial
-----------------------------------------------

* Le jeu de données ``features_1st_test.csv`` est chargé (Cellule [2]).
* La première colonne est renommée de 'Unnamed: 0' à ``'time'`` (Cellule [3]).
* Une copie du DataFrame est effectuée dans une variable nommée ``features`` (Cellule [5]).

Ingénierie des Caractéristiques : Moyennes Mobiles (Exploratoire)
-----------------------------------------------------------------

* Pour comprendre les tendances, des Moyennes Mobiles Simples (SMA), Cumulatives (CMA) et Exponentielles (EMA) sont calculées et tracées pour la caractéristique ``'B4_x_mean'`` (Cellules [6], [7]).

    .. code-block:: python

        ma = pd.DataFrame()
        ma['B4_x_mean'] = features['B4_x_mean']
        ma['SMA'] = ma['B4_x_mean'].rolling(window=5).mean()
        ma['CMA'] = ma["B4_x_mean"].expanding(min_periods=10).mean()
        ma['EMA'] = ma['B4_x_mean'].ewm(span=40,adjust=False).mean()
        # ...
        ma.plot(x="time", y= ['B4_x_mean','SMA','CMA','EMA'])

Création de l'Indicateur de Santé (HI)
--------------------------------------

Un Indicateur de Santé (HI) est dérivé en utilisant l'Analyse en Composantes Principales (ACP) pour représenter l'état de dégradation du roulement.

* **Fonction : ``health_indicator(bearing_data, use_filter=False)``** (Cellule [8])
    * Prend en entrée les données du roulement.
    * Optionnellement, applique un filtre par Moyenne Mobile Exponentielle (EMA avec ``span=40``) si ``use_filter`` est ``True``.
    * Applique l'ACP. La première composante principale (PC1) est choisie comme HI. Le pourcentage de variance expliqué par PC1 est affiché.
    * PC1 est normalisé (décalé pour commencer à 0).
    * Retourne un DataFrame ``degredation`` avec ``'PC1'`` et ``'cycle'``.

* **Caractéristiques Sélectionnées pour l'HI :** Les caractéristiques ``'max'``, ``'p2p'``, ``'rms'`` sont sélectionnées pour un roulement spécifique (par exemple, Roulement 3 dans la cellule [11]).

    .. code-block:: python

        selected_features = ['max','p2p','rms']
        bearing = 3 # Exemple pour le Roulement 3
        B_x = ["B{}_x_".format(bearing)+i for i in selected_features]

Modèle de Dégradation Exponentielle et Prédiction de la RUL
-----------------------------------------------------------

Un modèle exponentiel est ajusté à l'HI (PC1) pour modéliser sa tendance de dégradation.

* **Fonction : ``fit_exp(df, base=500, print_parameters=False)``** (Cellule [8])
    * Ajuste une fonction exponentielle ``y = a * exp(|b| * x)`` aux ``base`` derniers points de PC1.
    * Utilise ``scipy.optimize.curve_fit`` avec ``p0=[0.01,0.001]`` et ``maxfev=10000``.
    * Retourne les paramètres ajustés ``a`` et ``b``.

* **Fonction : ``predict(X_df, p)``** (Cellule [8])
    * Calcule les valeurs prédites de l'HI avec les paramètres ajustés.

* **Calcul de la RUL :**
    * Un seuil de défaillance (``thres = 2``) est défini pour l'HI.
    * Le cycle de défaillance est prédit par : ``fail_cycle = (np.log(thres/m))/abs(n)``.

Simulation du Cycle de Vie et Enregistrement des Prédictions
------------------------------------------------------------

Une boucle simule la prédiction en temps réel (Cellule [13]).

* **Initialisation :**
    * ``init_cycle = 600`` : Cycles initiaux pour le premier modèle.
    * ``prediction_cycle = init_cycle`` : "Temps" actuel.
    * ``log = [[],[]]`` : Pour stocker ``prediction_cycle`` et ``fail_cycle`` prédit.
* **Boucle de Simulation :**
    * À chaque itération, ``prediction_cycle`` est incrémenté de ``increment_cycle = 25``.
    * Les données jusqu'au ``prediction_cycle`` sont utilisées pour calculer l'HI (``health_indicator`` avec filtre).
    * Le modèle exponentiel est ajusté (``fit_exp`` avec ``base=250`` cycles récents).
    * Le ``fail_cycle`` est prédit.
    * Les résultats sont enregistrés et un graphique de la dégradation actuelle et prédite est généré.

Analyse des Résultats de la Simulation
--------------------------------------

Les prédictions enregistrées sont analysées (Cellule [15]).

* Un DataFrame ``df`` est créé à partir de ``log``.
* Une colonne ``isvalid`` est ajoutée (``fail_cycle < 2156``).
* Une colonne ``real`` est ajoutée avec des statuts réels de roulement (codés en dur) pour comparaison qualitative.
