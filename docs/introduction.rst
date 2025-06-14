.. _introduction:

Introduction
============

Ce projet se concentre sur l'analyse de données temporelles issues de capteurs de vibrations sur des roulements afin de prédire leur Durée de Vie Utile Restante (RUL - Remaining Useful Life). L'approche principale consiste à extraire un indicateur de santé (HI - Health Indicator) à partir des données des capteurs, puis à utiliser un modèle de DeepLearning pour prévoir la défaillance. Le projet explore également des techniques de modélisation alternatives utilisant des méthodes statistiques (ARIMA) pour la prévision de séries temporelles de l'indicateur de santé.

**Objectifs Clés :**

* Charger et prétraiter les données des capteurs de roulements.
* Créer des caractéristiques (features), y compris diverses moyennes mobiles.
* Développer un Indicateur de Santé (HI) pour représenter l'état de dégradation d'un roulement.
* Simuler le cycle de vie d'un roulement et enregistrer les prédictions.
* Explorer les modèles ARIMA et les Réseaux de Neurones Récurrents (RNN, LSTM, GRU) pour la prédiction de séries temporelles de l'HI.
