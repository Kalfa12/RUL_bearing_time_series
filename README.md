# RUL Bearing Time Series - Prédiction de la Durée de Vie Utile Restante

## Introduction
Ce projet se concentre sur l'analyse de données de séries temporelles provenant de capteurs de roulements afin de prédire leur Durée de Vie Utile Résiduelle (RUL). L'objectif principal est de développer des modèles capables d'estimer avec précision combien de temps un roulement continuera de fonctionner avant une défaillance, en se basant sur les lectures de ses capteurs au fil du temps.

## <a href="https://colab.research.google.com/github/Kalfa12/RUL_bearing_time_series/blob/main/projet_Time_series.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Ouvrir dans Colab"/></a>

## Objectif du Projet
L'objectif principal est de prédire la RUL des roulements en :
1.  Calculant un Indicateur de Santé (Health Indicator - HI) à partir des données des capteurs.
2.  Utilisant un modèle d'ajustement exponentiel sur le HI pour prédire la défaillance.
3.  Explorant d'autres techniques de modélisation pour la prédiction du HI et l'estimation de la RUL.

## Jeu de Données
Le principal jeu de données utilisé pour cette analyse est `features_1st_test.csv`. Ce fichier contient des caractéristiques extraites des données des capteurs de roulements. Les données incluent diverses caractéristiques statistiques pour différents roulements (B1, B2, B3, B4) sur les axes x et y, telles que la moyenne, l'écart-type, l'asymétrie, l'aplatissement (kurtosis), l'entropie, la valeur efficace (RMS), le maximum et les valeurs crête-à-crête. Des horodatages sont fournis pour chaque point de données.

## Méthodologie
Le cœur du projet est implémenté dans le notebook `projet_Time_series.ipynb`.

### Pipeline Principal de Prédiction RUL
L'approche principale comprend :
1.  **Chargement et Prétraitement des Données :** Chargement du jeu de données `features_1st_test.csv` et préparation initiale des données. Cela inclut le renommage des colonnes et la définition de l'index temporel.
2.  **Calcul de l'Indicateur de Santé (HI) :** Un Indicateur de Santé est généralement dérivé en utilisant l'Analyse en Composantes Principales (ACP) sur les caractéristiques des capteurs. La première composante principale (PC1) est utilisée comme HI.
3.  **Ajustement Exponentiel pour la RUL :** Un modèle exponentiel est ajusté à l'Indicateur de Santé calculé pour prédire la RUL. Le notebook simule ce processus de prédiction de manière itérative.

### Modèles Exploratoires
Le projet explore également d'autres techniques de modélisation :
* **Modèles d'Apprentissage Profond (Deep Learning) :** Les Réseaux de Neurones Récurrents (RNN), spécifiquement SimpleRNN, LSTM et GRU, sont explorés pour prédire l'Indicateur de Santé.
* **Analyse Statistique :** La modélisation ARIMA est étudiée pour l'Indicateur de Santé (PC1 du Roulement 3).

## Fichiers Principaux
* `projet_Time_series.ipynb` : Jupyter Notebook contenant l'analyse principale, la modélisation et le code de simulation.
* `features_1st_test.csv` : Le jeu de données principal avec les caractéristiques extraites des données des capteurs de roulements.
* `docs/` : Contient les fichiers de documentation, probablement pour Sphinx.
* `.readthedocs.yaml` : Fichier de configuration pour Read the Docs.

## Configuration et Installation
Le projet est développé en Python et utilise plusieurs bibliothèques clés.

1.  **Prérequis :**
    * Python 3
    * Jupyter Notebook ou JupyterLab
2.  **Bibliothèques Utilisées :**
    * Numpy
    * Matplotlib
    * Pandas
    * TensorFlow (pour les modèles d'Apprentissage Profond)
    * Scikit-learn 
    * Statsmodels (pour les modèles ARIMA)
    * NBSphinx (pour la génération de documentation à partir de notebooks, comme suggéré par `conf.py`)
3.  **Installation :**
    Il est recommandé d'utiliser un environnement virtuel.
    ```bash
    python -m venv env
    source env/bin/activate  # Sous Windows, utilisez `env\Scripts\activate`
    ```
    Bien que `docs/requirements.txt` soit actuellement vide, vous installeriez typiquement les dépendances en utilisant :
    ```bash
    pip install -r requirements.txt
    ```
    Installez les bibliothèques nécessaires manuellement si un fichier `requirements.txt` n'est pas renseigné :
    ```bash
    pip install numpy matplotlib pandas tensorflow scikit-learn statsmodels jupyter
    ```

## Utilisation
Pour utiliser ce projet :
1.  **Configurer l'Environnement :** Assurez-vous que toutes les bibliothèques listées dans `docs/setup.rst` (et importées dans le notebook) sont installées.
2.  **Exécuter le Jupyter Notebook :** Lancez le notebook `projet_Time_series.ipynb` pour effectuer le chargement des données, l'analyse, l'entraînement des modèles et les simulations de prédiction RUL.

## Résultats
Les résultats de la simulation principale de prédiction RUL (utilisant l'Indicateur de Santé et l'ajustement exponentiel) sont générés de manière itérative dans le notebook (typiquement dans une cellule comme `[13]` comme mentionné dans `docs/results.rst`). Chaque itération de simulation produit généralement :
* Un graphique montrant :
    * L'Indicateur de Santé (PC1).
    * L'ajustement exponentiel au HI.
    * La RUL réelle.
    * La RUL prédite.
* Des métriques telles que l'Erreur Absolue Moyenne (MAE) et la Racine de l'Erreur Quadratique Moyenne (RMSE) pour la prédiction RUL.

## Documentation
Une documentation supplémentaire peut être trouvée dans le répertoire `/docs`. Ce projet est configuré avec les paramètres de Read the Docs (`.readthedocs.yaml` et `docs/conf.py`), ce qui suggère que Sphinx peut être utilisé pour construire une documentation HTML à partir des fichiers `.rst`.
