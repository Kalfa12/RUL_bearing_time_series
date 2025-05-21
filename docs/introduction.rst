.. _introduction:

Introduction
============

Ce projet se concentre sur l'analyse de données temporelles issues de capteurs de roulements afin de prédire leur Durée de Vie Utile Résiduelle (RUL - Remaining Useful Life). L'approche principale consiste à extraire un indicateur de santé (HI - Health Indicator) à partir des données des capteurs en utilisant l'Analyse en Composantes Principales (ACP), puis à ajuster un modèle de dégradation exponentielle à cet HI pour prévoir la défaillance. Le notebook explore également des techniques de modélisation alternatives utilisant des méthodes statistiques (ARIMA) et l'apprentissage profond (RNN, LSTM, GRU) pour la prévision de séries temporelles de l'indicateur de santé.

**Objectifs Clés :**

* Charger et prétraiter les données des capteurs de roulements.
* Créer des caractéristiques (features), y compris diverses moyennes mobiles.
* Développer un Indicateur de Santé (HI) en utilisant l'ACP pour représenter l'état de dégradation d'un roulement.
* Implémenter un modèle de dégradation exponentielle pour prédire la RUL.
* Simuler le cycle de vie d'un roulement et enregistrer les prédictions.
* Explorer les modèles ARIMA et les Réseaux de Neurones Récurrents (RNN, LSTM, GRU) pour la prédiction de séries temporelles de l'HI.
