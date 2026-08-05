"""EDA for the road accident risk dataset. Saves plots to eda_output/."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split

OUT_DIR = Path("eda_output")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET = "accident_risk"
train = pd.read_csv("datasets/train.csv")

CATEGORICAL = [
    "road_type", "lighting", "weather", "time_of_day",
    "road_signs_present", "public_road", "holiday", "school_season",
]
NUMERIC = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]

# --- basic overview -----------------------------------------------------
print("shape:", train.shape)
print("\ndtypes:\n", train.dtypes)
print("\nmissing values:\n", train.isna().sum()[train.isna().sum() > 0])
print("\nduplicate rows:", train.duplicated().sum())
print("\nnumeric summary:\n", train[NUMERIC + [TARGET]].describe())

# --- target distribution -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(train[TARGET], bins=50, kde=True, ax=ax)
ax.set_title("Distribution of accident_risk")
fig.tight_layout()
fig.savefig(OUT_DIR / "target_distribution.png", dpi=150)
plt.close(fig)

print(f"\n{TARGET} skew: {train[TARGET].skew():.3f}")

# --- numeric feature correlations with target -----------------------------
corr = train[NUMERIC + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (numeric features + target)")
fig.tight_layout()
fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)

print("\ncorrelation with target:\n", corr[TARGET].drop(TARGET).sort_values(ascending=False))

# --- numeric feature distributions + relation to target -------------------
fig, axes = plt.subplots(len(NUMERIC), 2, figsize=(11, 3.2 * len(NUMERIC)))
for i, col in enumerate(NUMERIC):
    sns.histplot(train[col], bins=40, ax=axes[i, 0])
    axes[i, 0].set_title(f"Distribution: {col}")

    sample = train.sample(min(20000, len(train)), random_state=0)
    sns.scatterplot(data=sample, x=col, y=TARGET, alpha=0.15, s=10, ax=axes[i, 1])
    axes[i, 1].set_title(f"{col} vs {TARGET}")
fig.tight_layout()
fig.savefig(OUT_DIR / "numeric_features.png", dpi=150)
plt.close(fig)

# --- categorical features vs target ---------------------------------------
n = len(CATEGORICAL)
ncols = 2
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
axes = axes.flatten()
for i, col in enumerate(CATEGORICAL):
    order = train.groupby(col)[TARGET].mean().sort_values(ascending=False).index
    sns.boxplot(data=train, x=col, y=TARGET, order=order, ax=axes[i])
    axes[i].set_title(f"{TARGET} by {col}")
    axes[i].tick_params(axis="x", rotation=30)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.tight_layout()
fig.savefig(OUT_DIR / "categorical_vs_target.png", dpi=150)
plt.close(fig)

# --- categorical value counts (check imbalance / rare categories) ---------
for col in CATEGORICAL:
    print(f"\n{col} value counts:\n{train[col].value_counts(dropna=False)}")

# --- pairwise interaction: mean target by two categoricals (example) ------
pivot = train.pivot_table(values=TARGET, index="road_type", columns="weather", aggfunc="mean")
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
ax.set_title("Mean accident_risk by road_type x weather")
fig.tight_layout()
fig.savefig(OUT_DIR / "interaction_road_type_weather.png", dpi=150)
plt.close(fig)

# --- feature importance from a quick LightGBM fit --------------------------
features = NUMERIC + CATEGORICAL
fi_train = train.copy()
for col in CATEGORICAL:
    fi_train[col] = fi_train[col].astype("category")

X_train, X_val, y_train, y_val = train_test_split(
    fi_train[features], fi_train[TARGET], test_size=0.2, random_state=0
)
fi_model = lgb.LGBMRegressor(
    objective="regression", n_estimators=2000, learning_rate=0.05, random_state=0
)
fi_model.fit(
    X_train, y_train,
    eval_X=X_val, eval_y=y_val,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    categorical_feature=CATEGORICAL,
)

importance = pd.Series(
    fi_model.feature_importances_, index=features
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
importance.plot.barh(ax=ax)
ax.set_title("LightGBM feature importance (split count)")
ax.set_xlabel("importance")
fig.tight_layout()
fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
plt.close(fig)

print("\nfeature importance:\n", importance.sort_values(ascending=False))

print(f"\nPlots saved to {OUT_DIR.resolve()}")
