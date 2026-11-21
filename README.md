# AgriPredict — Prédiction du rendement agricole au Cameroun

Projet réalisé dans le cadre de la préparation du dossier de candidature au double
diplôme ENSTA Paris (filière IA), été 2026.

## Problématique

Peut-on prédire le rendement (t/ha) du cacao, du maïs et du manioc au Cameroun à partir
du climat (température, pluviométrie) et des intrants agricoles (engrais, pesticides) ?
Au-delà de la prédiction elle-même, l'objectif pédagogique du projet est de démontrer
la compréhension mathématique des modèles utilisés, pas seulement leur utilisation via
une bibliothèque : la régression linéaire est donc **implémentée from scratch en
NumPy** (fonction de coût, descente de gradient, régularisation L2), en parallèle des
modèles scikit-learn utilisés pour la version déployée.

## Données

- Source : dataset FAOSTAT déjà nettoyé et fusionné, utilisé tel quel
  (`data/raw/dataset_final_cameroun.csv`), sans modification des valeurs.
- Période : 1990-2024 (35 années).
- Cultures (libellés d'origine conservés, en français) : **Cacao, fèves** ·
  **Maïs** · **Manioc, frais** — 35 lignes par culture, 105 lignes au total.
- Variables : `year`, `crop`, `yield_t_ha` (cible), `temp_anomaly` (°C), `rainfall` (mm/an),
  `fertilizer` (tonnes), `pesticides` (tonnes), `quality_flag` (flag qualité FAOSTAT, informatif).
- Aucune valeur manquante.
- **Limite de couverture** : `fertilizer`, `pesticides` et `temp_anomaly` sont des valeurs
  nationales identiques pour les 3 cultures la même année (pas de valeur spécifique par
  culture) ; `rainfall` diffère légèrement selon la culture. 4 lignes (maïs, 2021-2024)
  portent un flag qualité "X" (donnée compilée/estimée avec prudence par FAOSTAT).

## Méthode

1. **Nettoyage** (`src/data_prep.py`) : renommage des colonnes techniques en anglais
   (`year`, `crop`, `yield_t_ha`...) pour la cohérence du code, **sans traduire les
   libellés de culture** — ils restent en français partout (dataset, API, interface).
2. **Exploration** (`notebooks/01_eda.ipynb`) : évolution du rendement par culture,
   corrélations, évolution des intrants, importance des variables, prédiction vs réel.
3. **Split temporel** (`src/common.py`) : entraînement sur 1990-2019, test sur 2020-2024
   (5 dernières années, ~15 % des lignes) — jamais de split aléatoire, pour évaluer une
   vraie capacité de prédiction sur des années futures.
4. **Modèle from scratch** (`src/scratch_ml.py`, détaillé dans
   `notebooks/02_modelisation_from_scratch.ipynb`) : régression linéaire multiple
   entraînée par **descente de gradient batch**, codée à la main :
   - standardisation manuelle des variables numériques (moyenne / écart-type),
   - encodage one-hot manuel de la variable `crop`,
   - fonction de coût `J(θ) = MSE/2 + régularisation L2`,
   - mise à jour des paramètres `θ := θ - α·∇J(θ)` répétée sur plusieurs milliers
     d'itérations, avec suivi de la courbe de convergence.
5. **Modèles scikit-learn** (`src/train.py`) : baseline (moyenne historique par culture),
   régression linéaire (sert de référence pour valider le modèle from scratch),
   Random Forest, Gradient Boosting — dans un `Pipeline` scikit-learn (standardisation +
   one-hot encoding appliqués après le split, sans fuite d'information).
6. **Sauvegarde** : le modèle avec le RMSE global le plus bas est ré-entraîné sur toutes
   les années puis sauvegardé (`models/model.joblib`, `models/meta.json`). L'API et
   l'interface restent agnostiques du type de modèle chargé (from scratch ou
   scikit-learn) : les deux exposent la même interface `predict(dataframe)`.

## Le modèle from scratch : implémentation et vérification

L'implémentation (`src/scratch_ml.py`, classe `LinearRegressionScratch`) suit les
formules du cours *Machine Learning Specialization* (Andrew Ng) :

```
Prédiction :   ŷ = X·θ
Coût :         J(θ) = 1/(2m)·Σ(ŷᵢ - yᵢ)²  +  λ/(2m)·Σθⱼ²   (j ≠ biais)
Gradient :     ∇J(θ) = 1/m·Xᵀ·(ŷ - y)  +  λ/m·θ
Mise à jour :  θ := θ - α·∇J(θ)
```

**Vérification de correction** : avec `λ = 0`, le modèle from scratch et
`sklearn.LinearRegression` doivent converger vers le même optimum (la régression
linéaire a une solution unique en l'absence de colinéarité parfaite). Test effectué
(`src/train.py::sanity_check_scratch_vs_sklearn`, reproduit dans le notebook) :

> Écart maximal entre les deux jeux de prédictions sur les données de test :
> de l'ordre de **10⁻¹⁴** (erreur numérique flottante, donc résultats identiques) —
> RMSE = 1.0319 pour les deux.

Cela confirme que la standardisation, l'encodage one-hot, la fonction de coût et la
descente de gradient sont correctement implémentés.

**Convergence** (`reports/figures/gradient_descent_convergence.png`) : le coût passe de
27,6 à 0,95 en moins de 30 itérations (dataset petit, variables standardisées ⇒ surface
de coût bien conditionnée). Le notebook `02_modelisation_from_scratch.ipynb` montre
aussi ce qui se passe avec un taux d'apprentissage trop élevé (divergence du coût), et
teste plusieurs valeurs de régularisation L2 (voir "Limites").

## Résultats

Comparaison des modèles (test = 5 dernières années, toutes cultures confondues) :

| Modèle                     | RMSE   | R²     |
|-----------------------------|-------:|-------:|
| Baseline (moyenne)          | 0.8011 | 0.9813 |
| **Linear scratch (NumPy)**  | 1.0319 | 0.9690 |
| Linear (scikit-learn)       | 1.0319 | 0.9690 |
| Random Forest               | 0.5542 | 0.9911 |
| Gradient Boosting           | 0.6346 | 0.9883 |

**Modèle déployé (API/interface) : Random Forest** (RMSE global le plus bas). Le
modèle from scratch obtient exactement le même score que sa contrepartie
scikit-learn (preuve de correction de l'implémentation) mais un score moins bon que
les modèles non linéaires : c'est cohérent, la relation entre climat/intrants et
rendement n'est pas purement linéaire (voir EDA).

Mais ce R² global est trompeur : les 3 cultures ont des échelles de rendement très
différentes (0,3 t/ha pour le cacao contre 12 t/ha pour le manioc), donc *savoir quelle
culture il s'agit* explique déjà l'essentiel de la variance totale — ce que confirme
l'importance des variables du Random Forest (`crop = Manioc, frais` pèse 92,7 %, loin
devant toute variable climatique). Le détail **par culture** est plus honnête :

| Culture         | RMSE   | R²      |
|------------------|-------:|--------:|
| Cacao, fèves     | 0.0494 | -6.1664 |
| Manioc, frais    | 0.9496 | -0.0703 |
| Maïs             | 0.1308 | -5.3680 |

**Interprétation** : sur seulement 5 années de test, le modèle fait moins bien que la
simple moyenne historique de chaque culture (R² négatif). Le graphique "prédiction vs
réel" (notebook EDA) montre qu'il capte une tendance lente mais rate les variations
d'une année sur l'autre — cohérent avec la faible corrélation entre le climat/les
intrants et le rendement pour le manioc et le maïs. Seul le cacao montre une
corrélation forte avec `fertilizer`/`pesticides`/`year`, mais elle reflète
probablement une tendance commune dans le temps plutôt qu'un lien causal direct.

## Limites

- Peu d'années disponibles (35) et seulement 5 ans de test : les métriques par culture
  sont instables et sensibles au choix de la période de test.
- Régularisation L2 testée (λ ∈ {1, 5, 10, 50}) sur le modèle from scratch : elle
  dégrade systématiquement le RMSE de test au lieu de l'améliorer, signe que le
  sur-apprentissage n'est pas le problème principal ici (peu de variables, modèle
  simple) — on garde donc λ = 0.
- Pas de causalité établie : les corrélations observées (notamment pour le cacao)
  peuvent refléter une tendance de fond commune (amélioration progressive des pratiques
  agricoles) plutôt qu'un effet direct du climat ou des intrants.
- Climat représenté par une moyenne annuelle nationale, sans granularité régionale ni
  saisonnière, alors que les cultures ont des besoins différents selon la période de
  l'année.
- Extrapolation au-delà de 2024 : les modèles à base d'arbres (Random Forest, Gradient
  Boosting) ne prédisent pas de tendance au-delà de la plage vue à l'entraînement ; ils
  renvoient des valeurs proches de la dernière période connue. Ce point n'est pas
  affiché comme avertissement dans l'interface (choix délibéré pour garder une
  démonstration simple), mais reste à garder en tête pour toute prédiction sur une
  année lointaine.
- 4 lignes (maïs, 2021-2024) portent un flag qualité FAOSTAT "X", à interpréter avec
  prudence.

## Interface

L'interface (`app/streamlit_app.py`) est volontairement minimaliste, entièrement en
français, pensée pour une prise en main immédiate et une démonstration en direct :
un formulaire (culture, année, température, pluviométrie, engrais, pesticides), un
bouton "Prédire", un résultat affiché en grand. Les détails techniques (modèle utilisé,
tableau comparatif des RMSE) sont disponibles dans un menu "À propos du modèle"
repliable, pour ne pas surcharger l'écran principal.

## Structure du projet

```
agripredict/
├── data/raw/                          CSV FAOSTAT déjà nettoyé, utilisé tel quel
├── data/processed/dataset.csv         table finale (colonnes renommées, valeurs inchangées)
├── notebooks/
│   ├── 01_eda.ipynb                   exploration des données
│   └── 02_modelisation_from_scratch.ipynb   démonstration mathématique + vérification
├── src/
│   ├── common.py                      colonnes, chargement, split temporel
│   ├── data_prep.py                   nettoyage / renommage des colonnes
│   ├── scratch_ml.py                  régression linéaire from scratch (NumPy)
│   └── train.py                       comparaison des modèles, sauvegarde
├── models/                            model.joblib + meta.json
├── api/main.py                        API FastAPI (/health, /meta, /predict)
├── app/streamlit_app.py               interface utilisateur (français, minimaliste)
├── reports/figures/                   graphiques (EDA, convergence, importance)
├── requirements.txt
└── README.md
```

## Lancer le projet

```bash
pip install -r requirements.txt
python src/data_prep.py && python -m src.train
uvicorn api.main:app --reload      # terminal 1 — API sur http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py # terminal 2 — interface sur http://localhost:8501
```
