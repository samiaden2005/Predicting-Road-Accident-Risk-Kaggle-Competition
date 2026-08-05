"""Blend LightGBM, CatBoost, and Ridge to squeeze past a single-model floor.

All three train on identical 5-fold splits so their out-of-fold predictions
are directly comparable. Blend weights are then chosen to minimize OOF RMSE
via a simplex grid search, and applied to the test predictions.
"""

import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from features import engineer_features, NEW_NUMERIC

TARGET = "accident_risk"
CATEGORICAL = [
    "road_type", "lighting", "weather", "time_of_day",
    "road_signs_present", "public_road", "holiday", "school_season",
]
BASE_NUMERIC = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
NUMERIC = BASE_NUMERIC + NEW_NUMERIC
FEATURES = NUMERIC + CATEGORICAL
N_FOLDS = 5

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
train = engineer_features(train)
test = engineer_features(test)

for col in CATEGORICAL:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

X, y = train[FEATURES], train[TARGET]
X_test = test[FEATURES]

lgb_params = dict(
    objective="regression", metric="rmse",
    learning_rate=0.03, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    n_estimators=5000, random_state=0,
)
if os.path.exists("best_params.json"):
    with open("best_params.json") as f:
        lgb_params.update(json.load(f))
    print("using tuned LightGBM params from best_params.json")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
splits = list(kf.split(X))

oof = {name: np.zeros(len(train)) for name in ("lgb", "cat", "ridge")}
test_preds = {name: np.zeros(len(test)) for name in ("lgb", "cat", "ridge")}

for fold, (train_idx, val_idx) in enumerate(splits):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    lgb_cat = X_train.copy()
    for col in CATEGORICAL:
        lgb_cat[col] = lgb_cat[col].astype("category")
    lgb_val = X_val.copy()
    for col in CATEGORICAL:
        lgb_val[col] = lgb_val[col].astype("category")
    lgb_test = X_test.copy()
    for col in CATEGORICAL:
        lgb_test[col] = lgb_test[col].astype("category")

    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(
        lgb_cat, y_train,
        eval_X=lgb_val, eval_y=y_val,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=CATEGORICAL,
    )
    oof["lgb"][val_idx] = lgb_model.predict(lgb_val, num_iteration=lgb_model.best_iteration_)
    test_preds["lgb"] += lgb_model.predict(lgb_test, num_iteration=lgb_model.best_iteration_) / N_FOLDS

    cat_model = CatBoostRegressor(
        iterations=1500, learning_rate=0.08, depth=6,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=0, verbose=False, early_stopping_rounds=50, thread_count=-1,
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=CATEGORICAL,
    )
    oof["cat"][val_idx] = cat_model.predict(X_val)
    test_preds["cat"] += cat_model.predict(X_test) / N_FOLDS

    ridge_pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("model", Ridge(alpha=1.0, random_state=0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    oof["ridge"][val_idx] = ridge_pipe.predict(X_val)
    test_preds["ridge"] += ridge_pipe.predict(X_test) / N_FOLDS

    print(f"fold {fold} done")

for name in ("lgb", "cat", "ridge"):
    rmse = mean_squared_error(y, oof[name]) ** 0.5
    print(f"{name} OOF rmse: {rmse:.5f}")

# --- find blend weights on OOF predictions (simplex grid search) ----------
best_rmse, best_weights = np.inf, None
step = 0.05
for w_lgb in np.arange(0, 1 + step, step):
    for w_cat in np.arange(0, 1 - w_lgb + step, step):
        w_ridge = 1 - w_lgb - w_cat
        if w_ridge < -1e-9:
            continue
        w_ridge = max(w_ridge, 0)
        blend = w_lgb * oof["lgb"] + w_cat * oof["cat"] + w_ridge * oof["ridge"]
        rmse = mean_squared_error(y, blend) ** 0.5
        if rmse < best_rmse:
            best_rmse, best_weights = rmse, (w_lgb, w_cat, w_ridge)

w_lgb, w_cat, w_ridge = best_weights
print(f"\nbest blend weights: lgb={w_lgb:.2f} cat={w_cat:.2f} ridge={w_ridge:.2f}")
print(f"blended OOF rmse: {best_rmse:.5f}")

final_test_preds = (
    w_lgb * test_preds["lgb"] + w_cat * test_preds["cat"] + w_ridge * test_preds["ridge"]
)
submission = pd.DataFrame({"id": test["id"], TARGET: np.clip(final_test_preds, 0, 1)})
submission.to_csv("submission.csv", index=False)
print("\nwrote submission.csv")
