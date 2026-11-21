"""
Régression linéaire multiple implémentée from scratch en NumPy.

Contrairement aux modèles scikit-learn utilisés ailleurs dans le projet (qui
résolvent la régression par des méthodes fermées ou du boosting), ce module
réimplémente à la main :
  - la standardisation des variables numériques (moyenne / écart-type),
  - l'encodage one-hot de la variable catégorielle (culture),
  - la fonction de coût (MSE + régularisation L2),
  - la descente de gradient (batch gradient descent) qui minimise ce coût.

Objectif : démontrer la compréhension mathématique du modèle (cf. Cours 1,
semaines 1-3, Machine Learning Specialization — Andrew Ng), pas seulement
appeler une bibliothèque. Les modèles scikit-learn (RandomForest, Gradient
Boosting) restent utilisés en parallèle pour capturer les relations non
linéaires, comme prévu par le planning.

Rappel des formules utilisées :

    Prédiction :        ŷ = X·θ            (X inclut une colonne de biais à 1)

    Coût (MSE + L2) :   J(θ) = 1/(2m) · Σ(ŷᵢ - yᵢ)²  +  λ/(2m) · Σ θⱼ²  (j ≠ 0)

    Gradient :          ∂J/∂θ = 1/m · Xᵀ·(ŷ - y)  +  λ/m · θ  (biais exclu)

    Mise à jour :        θ := θ - α · ∂J/∂θ           (α = taux d'apprentissage)
"""

import numpy as np
import pandas as pd


class StandardScalerScratch:
    """Standardisation (moyenne 0, écart-type 1), implémentée à la main."""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # évite une division par zéro
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def one_hot_encode(values: np.ndarray, categories: list) -> np.ndarray:
    """Encodage one-hot manuel, à partir d'une liste de catégories fixée
    (indispensable pour encoder de la même façon les données d'entraînement
    et les nouvelles données à prédire)."""
    out = np.zeros((len(values), len(categories)), dtype=float)
    for j, cat in enumerate(categories):
        out[:, j] = (values == cat).astype(float)
    return out


def mse_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


class LinearRegressionScratch:
    """Régression linéaire multiple entraînée par descente de gradient batch,
    avec régularisation L2 optionnelle (ridge).

    Prend directement un DataFrame en entrée (comme les pipelines
    scikit-learn utilisés ailleurs dans le projet) pour rester interchangeable
    avec eux dans train.py et l'API.
    """

    def __init__(self, num_features, cat_feature, learning_rate=0.1,
                 n_iterations=3000, lambda_reg=0.0):
        self.num_features = list(num_features)
        self.cat_feature = cat_feature
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.scaler = StandardScalerScratch()
        self.categories_ = None
        self.theta_ = None
        self.cost_history_ = []

    def _design_matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        num = df[self.num_features].to_numpy(dtype=float)
        num_scaled = self.scaler.fit_transform(num) if fit else self.scaler.transform(num)

        if fit:
            self.categories_ = sorted(df[self.cat_feature].unique().tolist())
        onehot = one_hot_encode(df[self.cat_feature].to_numpy(), self.categories_)

        X = np.hstack([num_scaled, onehot])
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])

    def fit(self, df: pd.DataFrame, target_col: str):
        X = self._design_matrix(df, fit=True)
        y = df[target_col].to_numpy(dtype=float)
        m, n = X.shape

        theta = np.zeros(n)
        cost_history = []

        for _ in range(self.n_iterations):
            y_pred = X @ theta
            error = y_pred - y

            reg_grad = np.r_[0.0, theta[1:]] * self.lambda_reg  # biais non régularisé
            grad = (X.T @ error + reg_grad) / m
            theta -= self.learning_rate * grad

            reg_cost = self.lambda_reg * np.sum(theta[1:] ** 2)
            cost = (np.sum(error ** 2) + reg_cost) / (2 * m)
            cost_history.append(cost)

        self.theta_ = theta
        self.cost_history_ = cost_history
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design_matrix(df, fit=False)
        return X @ self.theta_

    def coef_summary(self) -> pd.Series:
        """Coefficients nommés, pour interprétation (variables déjà standardisées,
        donc directement comparables entre elles en valeur absolue)."""
        names = ["bias"] + self.num_features + [f"crop_{c}" for c in self.categories_]
        return pd.Series(self.theta_, index=names)
