# 📊 Système Expert d'Analyse Financière Pro

Une application web professionnelle d'analyse quantitative et de benchmarking financier développée avec Streamlit. Cette plateforme permet aux analystes et investisseurs d'importer des données boursières, de visualiser les performances et de calculer des indicateurs de risque avancés.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3366cc)

## 🚀 Fonctionnalités Principales

### 1. Importation et Traitement des Données
- **Support CSV Flexible** : Importation de fichiers de cours boursiers.
- **Détection Intelligente** : Identification automatique des colonnes de dates et de prix.
- **Gestion des Données Manquantes** :
  - Remplacement par 0.
  - Propagation (Forward Fill).
  - Interpolation Linéaire.

### 2. Visualisation Interactive
- **Évolution des Prix** : Graphiques interactifs comparant l'actif principal aux benchmarks.
- **Rendements Normalisés (Base 100)** : Comparaison directe de la performance relative (Alpha).
- **Matrice de Corrélation** : Heatmap des corrélations (Log Returns) pour l'analyse de diversification.

### 3. Analyse Quantitative Avancée
Calcul automatique des ratios financiers clés (Risk-Adjusted Returns) :
- **Performance** : Rendement Annuel, Volatilité.
- **Ratios de Gestion** : Ratio de Sharpe, Ratio de Treynor, Alpha de Jensen.
- **Risque de Marché** : Beta.
- **Risque de Perte** : Max Drawdown, VaR (Value at Risk 95%), CVaR (Conditional VaR 95%).

## 🛠️ Installation et Démarrage

### Prérequis
 Assurez-vous d'avoir Python installé.

### 1. Cloner le dépôt
```bash
git clone https://github.com/boulifamarwan3-alt/ANALYSE-FINANCI-RE-PRO.git
cd ANALYSE-FINANCI-RE-PRO
```

### 2. Créer un environnement virtuel (recommandé)
**Windows :**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```
*(Si `requirements.txt` n'existe pas, installez manuellement : `pip install streamlit pandas numpy plotly`)*

### 4. Lancer l'application
```bash
streamlit run app.py
```

## 📂 Structure du Projet

- `app.py` : Le fichier principal de l'application Streamlit contenant toute la logique.
- `stocks.csv` : Jeu de données exemple (Actions du MASI).
- `.gitignore` : Fichiers ignorés par Git (venv, pycache, etc.).

## 🎨 Design System
L'application utilise un design system "Light Professional" avec :
- Une palette de couleurs financières (Bleu Royal, Gris Ardoise, Vert Succès).
- Une typographie 'Inter' pour une lisibilité optimale.
- Des composants UI stylisés (Cartes, Métriques, Tableaux).

---
*Développé pour l'analyse financière professionnelle.*
