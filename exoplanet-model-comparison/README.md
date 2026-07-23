# Comparing Machine-Learning Models for Exoplanet Detection and Characterization

A single, self-contained Jupyter notebook that builds a complete small research
project: it simulates transit light curves, trains and compares seven machine-learning
models on two tasks (detecting a transit, and measuring the planet's size), evaluates
every model on both **accuracy** and an objective **interpretability** score, and
validates the best ideas on real NASA Kepler light curves.

The notebook is written to be readable without prior astronomy or machine-learning
background, with each concept explained as it appears.

## Running it

Everything is in `Exoplanet_Model_Comparison.ipynb`. There are no other required files:
all code is inlined and a small sample of real Kepler data is embedded directly in the
notebook. The first cell installs any missing packages.

- **Google Colab / Jupyter:** open the notebook and run all cells. A full run takes a
  few minutes (most of it training the physics-informed model).
- **Local environment (optional):** `pip install -r requirements.txt`, then run the
  notebook. `requirements.txt` is only needed if you would rather manage the environment
  yourself; the notebook installs what it needs on first run regardless.

The committed notebook already contains all rendered outputs, so it can be read as a
finished report on GitHub without running anything.

## What it covers

1. Background: transits, and the Kepler and TESS missions.
2. Simulating labelled light curves with known planet properties.
3. Data preprocessing, shown step by step with figures.
4. Seven models: linear, random forest, XGBoost, MLP, CNN, LSTM, and a
   Kolmogorov-Arnold Network (KAN).
5. Task 1 - Detection (classification): leaderboard, ROC curves, and an
   accuracy-vs-interpretability comparison.
6. Task 2 - Characterization (regression): measuring planet size, including a
   physics-informed KAN that recovers sizes without ever seeing size labels.
7. The single-transit regime, where standard folding does not apply.
8. Validation on real Kepler planets, compared against published values.

## Main finding

Accuracy and interpretability trade off against each other: the most accurate models are
opaque, while readable models give up a little accuracy for transparency. There is no
single best model; the right choice depends on whether raw performance or an explanation
you can trust matters more for the task at hand.
