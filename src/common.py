"""Configuration commune : colonnes, chargement, split temporel.

Utilisé par le notebook d'exploration, l'entraînement (train.py) et l'API.
"""

import pandas as pd

NUM = ["year", "temp_anomaly", "rainfall", "fertilizer", "pesticides"]
CAT = ["crop"]
TARGET = "yield_t_ha"


def load_dataset(path="data/processed/dataset.csv"):
    return pd.read_csv(path)


def temporal_split(df, n_test_years=5):
    """Entraîne sur le passé, teste sur les dernières années.

    Jamais de split aléatoire : on veut évaluer la capacité du modèle à
    prédire des années futures, pas à interpoler entre des années connues.
    """
    cut = df["year"].max() - n_test_years
    return df[df["year"] <= cut], df[df["year"] > cut]
