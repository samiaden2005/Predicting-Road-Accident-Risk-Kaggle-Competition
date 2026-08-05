"""Optuna hyperparameter search for the LightGBM road-risk model.

Uses 3-fold CV (cheaper than main.py's 5-fold) with early stopping per fold.
Writes the winning params to best_params.json, which main.py picks up
automatically if present.
"""

import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from features import engineer_features

N_TRIALS = 40
N_FOLDS = 3

TARGET = "accident_risk"
CATEGORICAL = [
    "road_type", "lighting", "weather", "time_of_day",
    "road_signs_present", "public_road", "holiday", "school_season",
]

train = pd.read_csv("datasets/train.csv")
train = engineer_features(train)
for col in CATEGORICAL:
    train[col] = train[col].astype("category")

features = [c for c in train.columns if c not in ("id", TARGET)]
X, y = train[features], train[TARGET]

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
cv_splits = list(kf.split(X))


def objective(trial):
    params = dict(
        objective="regression",
        metric="rmse",
        random_state=0,
        n_estimators=2500,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 180),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
    )

    fold_rmses = []
    for train_idx, val_idx in cv_splits:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_X=X_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=CATEGORICAL,
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration_)
        fold_rmses.append(mean_squared_error(y_val, preds) ** 0.5)

    return float(np.mean(fold_rmses))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nbest CV rmse:", study.best_value)
print("best params:", study.best_params)

with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)
print("\nwrote best_params.json")
