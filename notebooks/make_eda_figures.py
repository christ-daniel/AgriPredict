"""Script équivalent au notebook 01_eda.ipynb (utilisé pour générer les figures
depuis ce script, puis recopié dans le notebook)."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset.csv")

# 1. Rendement par culture dans le temps (un axe Y par culture, échelles très différentes)
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
for ax, crop in zip(axes, sorted(df["crop"].unique())):
    sub = df[df["crop"] == crop]
    ax.plot(sub["year"], sub["yield_t_ha"], marker="o", markersize=3)
    ax.set_title(f"{crop} — rendement (t/ha)")
    ax.grid(alpha=0.3)
plt.xlabel("Année")
plt.tight_layout()
plt.savefig(FIG / "rendement_temps.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Corrélations (numériques uniquement)
plt.figure(figsize=(6, 5))
num_cols = ["yield_t_ha", "temp_anomaly", "rainfall", "fertilizer", "pesticides", "year"]
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélations (toutes cultures confondues)")
plt.tight_layout()
plt.savefig(FIG / "correlations.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Intrants dans le temps
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
df.groupby("year")["fertilizer"].mean().plot(ax=axes[0], title="Engrais (tonnes, national)")
df.groupby("year")["pesticides"].mean().plot(ax=axes[1], title="Pesticides (tonnes, national)")
plt.tight_layout()
plt.savefig(FIG / "intrants.png", dpi=150, bbox_inches="tight")
plt.close()

print("3 figures écrites dans", FIG)
print()
print("Corrélation avec yield_t_ha (par culture) :")
for crop in sorted(df["crop"].unique()):
    sub = df[df["crop"] == crop]
    corr = sub[num_cols].corr()["yield_t_ha"].drop("yield_t_ha").sort_values(key=abs, ascending=False)
    print(f"\n{crop}:")
    print(corr)
