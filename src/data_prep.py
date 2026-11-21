"""
Nettoyage du dataset AgriPredict.

Le fichier brut `dataset_final_cameroun.csv` est déjà fusionné et nettoyé
(une ligne = culture/année) — il est utilisé tel quel, sans modification des
valeurs. Ce script se contente de :
  1. renommer les colonnes techniques en anglais (cohérent avec common.py /
     train.py / api), SANS traduire les libellés de culture eux-mêmes —
     "Cacao, fèves", "Maïs", "Manioc, frais" restent inchangés, pour que
     l'interface et l'API affichent des noms français reconnaissables,
  2. vérifier l'absence de valeurs manquantes et la plausibilité des rendements,
  3. écrire data/processed/dataset.csv.
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw/dataset_final_cameroun.csv")
OUT = Path("data/processed/dataset.csv")

RENAME = {
    "Annee": "year",
    "Produit": "crop",
    "Rendement_t_ha": "yield_t_ha",
    "Anomalie_temp_C": "temp_anomaly",
    "Pluviometrie_mm_an": "rainfall",
    "Engrais_total_tonnes": "fertilizer",
    "Pesticides_tonnes": "pesticides",
    "Qualite_rendement": "quality_flag",
}


def main():
    df = pd.read_csv(RAW)
    df = df.rename(columns=RENAME)

    cols = ["year", "crop", "yield_t_ha", "temp_anomaly", "rainfall",
            "fertilizer", "pesticides", "quality_flag"]
    df = df[cols].sort_values(["crop", "year"]).reset_index(drop=True)

    n_missing = df.isna().sum().sum()
    if n_missing:
        print(f"Attention : {n_missing} valeurs manquantes restantes.")
    else:
        print("Aucune valeur manquante.")

    print("\nCultures détectées :", sorted(df["crop"].unique().tolist()))

    print("\nLignes par culture :")
    print(df.groupby("crop").size())

    print("\nStatistiques du rendement (t/ha) :")
    print(df.groupby("crop")["yield_t_ha"].describe()[["min", "mean", "max"]])

    hors_bornes = df[(df["yield_t_ha"] <= 0) | (df["yield_t_ha"] > 30)]
    if len(hors_bornes):
        print("\nAttention, rendements hors bornes plausibles (0-30 t/ha) :")
        print(hors_bornes)

    print("\nFlags de qualité (X = donnée compilée/estimée avec prudence) :")
    print(df.groupby(["crop", "quality_flag"]).size())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nFichier écrit : {OUT} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
