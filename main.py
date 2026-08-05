import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from features import engineer_features, NEW_NUMERIC

TARGET = "accident_risk"
CATEGORICAL = [
    "road_type", "lighting", "weather", "time_of_day",
    "road_signs_present", "public_road", "holiday", "school_season",
]

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train = engineer_features(train)
test = engineer_features(test)

print("correlation of interaction features with target:")
print(train[NEW_NUMERIC + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False))

for col in CATEGORICAL:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

features = [c for c in train.columns if c not in ("id", TARGET)]
X, y = train[features], train[TARGET]
X_test = test[features]

params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    n_estimators=5000,
    random_state=0,
)

if os.path.exists("best_params.json"):
    with open("best_params.json") as f:
        tuned = json.load(f)
    print("loaded tuned hyperparameters from best_params.json:", tuned)
    params.update(tuned)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof = np.zeros(len(train))
test_preds = np.zeros(len(test))
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        categorical_feature=CATEGORICAL,
    )

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / kf.n_splits
    models.append(model)

    fold_rmse = mean_squared_error(y_val, oof[val_idx]) ** 0.5
    print(f"fold {fold} rmse: {fold_rmse:.5f}")

cv_rmse = mean_squared_error(y, oof) ** 0.5
print(f"overall CV rmse: {cv_rmse:.5f}")

importance = pd.Series(
    np.mean([m.feature_importances_ for m in models], axis=0), index=features
).sort_values(ascending=False)
print("\nfeature importance (avg gain-split count):\n", importance)

submission = pd.DataFrame({"id": test["id"], TARGET: np.clip(test_preds, 0, 1)})
submission.to_csv("submission.csv", index=False)
print("\nwrote submission.csv")
