"""Entraînement et comparaison de modèles pour AgriPredict.

Compare quatre familles de modèles :
  - baseline_moyenne     : moyenne historique par culture (référence naïve)
  - linear_scratch       : régression linéaire + descente de gradient,
                            implémentée à la main en NumPy (src/scratch_ml.py)
  - linear                : régression linéaire scikit-learn (sert à vérifier
                            que l'implémentation from scratch est correcte :
                            mêmes prédictions attendues à convergence)
  - random_forest / gradient_boosting : modèles scikit-learn non linéaires,
                            pour capturer des relations que la régression
                            linéaire ne peut pas représenter.

Usage : python -m src.train   (depuis la racine du projet)
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.common import NUM, CAT, TARGET, load_dataset, temporal_split
from src.scratch_ml import LinearRegressionScratch

FIG = Path("reports/figures")

SKLEARN_MODELS = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

# Hyperparamètres de la descente de gradient (choisis empiriquement : voir
# reports/figures/gradient_descent_convergence.png pour vérifier la
# convergence, la courbe de coût doit décroître puis se stabiliser).
SCRATCH_PARAMS = dict(learning_rate=0.3, n_iterations=5000, lambda_reg=0.0)


def make_pipe(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def fit_scratch(df):
    return LinearRegressionScratch(NUM, CAT[0], **SCRATCH_PARAMS).fit(df, TARGET)


def scores(y, pred):
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "r2": round(float(r2_score(y, pred)), 4),
    }


def plot_gradient_descent_convergence(scratch_model):
    """Deux vues : échelle log sur toutes les itérations (montre qu'on
    approche bien un minimum) + zoom linéaire sur les 50 premières
    itérations (montre la descente elle-même, très rapide ici vu le peu de
    variables et le learning rate choisi)."""
    FIG.mkdir(parents=True, exist_ok=True)
    cost = scratch_model.cost_history_
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(cost)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("Coût J(θ) (échelle log)")
    axes[0].set_title("Convergence complète")
    axes[0].grid(alpha=0.3)

    n_zoom = min(50, len(cost))
    axes[1].plot(range(n_zoom), cost[:n_zoom], marker="o", markersize=3)
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("Coût J(θ)")
    axes[1].set_title(f"Zoom sur les {n_zoom} premières itérations")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG / "gradient_descent_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure écrite : {FIG / 'gradient_descent_convergence.png'} "
          f"(coût initial {cost[0]:.4f} -> final {cost[-1]:.4f})")


def sanity_check_scratch_vs_sklearn(train, test):
    """Vérifie que le modèle from scratch (lambda=0) retrouve les mêmes
    prédictions que sklearn.LinearRegression sur les mêmes données : preuve
    que la descente de gradient converge vers l'optimum et que
    l'implémentation (coût, gradient, standardisation, one-hot) est correcte.
    """
    scratch = LinearRegressionScratch(NUM, CAT[0], learning_rate=0.3,
                                       n_iterations=5000, lambda_reg=0.0).fit(train, TARGET)
    sklearn_pipe = make_pipe(LinearRegression()).fit(train[NUM + CAT], train[TARGET])

    pred_scratch = scratch.predict(test[NUM + CAT])
    pred_sklearn = sklearn_pipe.predict(test[NUM + CAT])
    max_diff = float(np.max(np.abs(pred_scratch - pred_sklearn)))
    print(f"Vérification from-scratch vs sklearn : écart max sur les prédictions "
          f"= {max_diff:.2e} (doit être proche de 0)\n")
    return max_diff


def main():
    df = load_dataset()
    train, test = temporal_split(df, n_test_years=5)
    print(f"{len(train)} lignes train / {len(test)} lignes test "
          f"(test = années {sorted(test['year'].unique())})\n")

    sanity_check_scratch_vs_sklearn(train, test)

    results = {}
    fitted = {}

    # Baseline naïve : moyenne historique par culture
    pred_base = test["crop"].map(train.groupby("crop")[TARGET].mean())
    results["baseline_moyenne"] = scores(test[TARGET], pred_base)

    # Modèle from scratch (NumPy, descente de gradient)
    scratch_model = fit_scratch(train)
    fitted["linear_scratch"] = scratch_model
    results["linear_scratch"] = scores(test[TARGET], scratch_model.predict(test[NUM + CAT]))
    plot_gradient_descent_convergence(scratch_model)

    # Modèles scikit-learn
    for name, model in SKLEARN_MODELS.items():
        pipe = make_pipe(clone(model)).fit(train[NUM + CAT], train[TARGET])
        fitted[name] = pipe
        results[name] = scores(test[TARGET], pipe.predict(test[NUM + CAT]))

    table = pd.DataFrame(results).T
    print("Comparaison des modèles (test = 5 dernières années) :")
    print(table)

    # Le R2 combiné est trompeur : les 3 cultures ont des échelles de
    # rendement très différentes, donc connaître la culture seule explique
    # une grande partie de la variance totale. On regarde aussi le RMSE et le
    # R2 par culture, sur le meilleur modèle, pour évaluer ce qui est appris
    # au-delà de la simple moyenne par culture.
    best = table.drop(index="baseline_moyenne")["rmse"].idxmin()
    print(f"\nModèle retenu pour le déploiement (RMSE global le plus bas) : {best}\n")

    print("Détail par culture (modèle retenu) :")
    per_crop = {}
    pred_best = fitted[best].predict(test[NUM + CAT])
    test_eval = test.assign(pred=pred_best)
    for crop, sub in test_eval.groupby("crop"):
        per_crop[crop] = scores(sub[TARGET], sub["pred"])
    print(pd.DataFrame(per_crop).T)

    # Ré-entraînement sur toutes les années disponibles avant sauvegarde
    if best == "linear_scratch":
        final = fit_scratch(df)
    else:
        final = make_pipe(clone(SKLEARN_MODELS[best])).fit(df[NUM + CAT], df[TARGET])

    Path("models").mkdir(exist_ok=True)
    joblib.dump(final, "models/model.joblib")

    meta = {
        "best_model": best,
        "metrics": results,
        "metrics_per_crop": per_crop,
        "crops": sorted(df["crop"].unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "num_features": NUM,
        "cat_features": CAT,
        "scratch_params": SCRATCH_PARAMS,
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nFichiers écrits : models/model.joblib, models/meta.json")

    # Importance / interprétation des variables selon le type de modèle retenu
    FIG.mkdir(parents=True, exist_ok=True)
    if isinstance(final, LinearRegressionScratch):
        coefs = final.coef_summary().drop("bias")
        print("\nCoefficients (variables standardisées, donc comparables) :")
        print(coefs.sort_values(key=abs, ascending=False))
        coefs.sort_values().plot.barh(figsize=(6, 4))
        plt.title("Coefficients de la régression linéaire from scratch")
        plt.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIG / "importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure écrite : {FIG / 'importance.png'}")
    else:
        model_step = final.named_steps["model"]
        names = final.named_steps["pre"].get_feature_names_out()
        if hasattr(model_step, "feature_importances_"):
            imp = pd.Series(model_step.feature_importances_, index=names).sort_values(ascending=False)
            print("\nImportance des variables :")
            print(imp)
            imp.sort_values().plot.barh(figsize=(6, 4))
            plt.title(f"Importance des variables ({best})")
        else:
            coefs = pd.Series(model_step.coef_, index=names).sort_values(key=abs, ascending=False)
            print("\nCoefficients :")
            print(coefs)
            coefs.sort_values().plot.barh(figsize=(6, 4))
            plt.title(f"Coefficients ({best})")
            plt.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIG / "importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure écrite : {FIG / 'importance.png'}")


if __name__ == "__main__":
    main()
