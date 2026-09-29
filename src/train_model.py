"""
Train and compare four models on the feature table, then save the best one.

Usage
-----
    python src/train_model.py                 # test on the last season in the data
    python src/train_model.py --test-season 2024

Models
------
    baseline        predicted finishing position = starting grid position
    ridge           Ridge regression on standardised features
    random_forest   RandomForestRegressor
    xgboost         XGBRegressor (gradient-boosted trees)

Hyper-parameters for the three learned models are chosen on an *inner*
chronological split: fit on all training seasons but the last, validate on the
last training season, pick the best setting, then refit on the full training
set.  The held-out test season is never used for any choice.

Outputs (in ``models/``)
------------------------
    <name>.pkl                 each fitted sklearn Pipeline (imputer + model)
    best_model.pkl             copy of the best model by test MAE
    metrics.json               every model's train/test metrics, chosen
                               hyper-parameters and split info
    feature_importance.csv     per-model feature importances
    test_predictions.csv       every test-set row with every model's prediction
    hyperparameter_trials.csv  validation MAE of every setting that was tried
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import FEATURE_COLUMNS, TARGET  # noqa: E402
from utils import FEATURES_PATH, IMPORTANCE_PATH, METRICS_PATH, MODELS_DIR, compute_metrics, setup_logging  # noqa: E402

log = setup_logging("train_model")
RANDOM_STATE = 42


def build_models(params: dict[str, dict] | None = None) -> dict[str, Pipeline]:
    """
    The three learned models, each wrapped in a Pipeline whose first step
    fills missing values with the *training-set* median.  Fitting the imputer
    inside the pipeline means test rows never influence the fill values.

    ``params`` optionally overrides the estimator keyword arguments per model
    (this is how the selected hyper-parameters are injected).
    """
    params = params or {}
    ridge_kw = {"alpha": 1.0, **params.get("ridge", {})}
    rf_kw = {
        "n_estimators": 500, "max_depth": 6, "min_samples_leaf": 10, "max_features": 0.5,
        "random_state": RANDOM_STATE, "n_jobs": -1, **params.get("random_forest", {}),
    }
    xgb_kw = {
        "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 10,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
        "objective": "reg:squarederror", "random_state": RANDOM_STATE, "n_jobs": -1,
        **params.get("xgboost", {}),
    }
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(**ridge_kw)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(**rf_kw)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(**xgb_kw)),
            ]
        ),
    }


# Small, deliberately coarse grids.  Finishing positions are noisy, so the
# search is mostly about *how much* to regularise each model.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {"max_depth": [4, 6, 8], "min_samples_leaf": [5, 10, 20]},
    "xgboost": {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05],
        "n_estimators": [150, 300],
        "min_child_weight": [5, 20],
        "reg_lambda": [2.0, 10.0],
    },
}


def select_hyperparameters(train: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Pick each model's hyper-parameters on an inner chronological split of the
    *training* data: fit on every training season except the last, score the
    ranked MAE on that last season, keep the best combination.

    Returns the chosen parameters per model and a long table of every trial.
    """
    val_season = int(train["season"].max())
    inner_train = train[train["season"] < val_season]
    inner_val = train[train["season"] == val_season]
    log.info("Hyper-parameter search: fit on %s, validate on %d",
             sorted(inner_train["season"].unique()), val_season)

    chosen: dict[str, dict] = {}
    trials = []
    for name, grid in PARAM_GRIDS.items():
        best_mae, best_params = float("inf"), {}
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            pipe = build_models({name: params})[name]
            pipe.fit(inner_train[FEATURE_COLUMNS], inner_train[TARGET])
            mae = compute_metrics(inner_val.assign(predicted_score=pipe.predict(inner_val[FEATURE_COLUMNS])))["mae"]
            trials.append({"model": name, **params, "val_mae": mae})
            if mae < best_mae:
                best_mae, best_params = mae, params
        chosen[name] = best_params
        log.info("%-14s best validation MAE %.3f with %s", name, best_mae, best_params)
    return chosen, pd.DataFrame(trials)


def chronological_split(df: pd.DataFrame, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train on every completed race *before* ``test_season``; test on the whole
    of ``test_season``.  Nothing after a test race is ever seen in training.
    """
    df = df.dropna(subset=[TARGET])
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    if train.empty or test.empty:
        raise ValueError(f"Bad split: {len(train)} train rows / {len(test)} test rows for test_season={test_season}")
    return train, test


def feature_importance(name: str, pipe: Pipeline) -> pd.DataFrame:
    """
    Extract a comparable importance per feature from each model type:
    absolute standardised coefficients for Ridge, impurity-based
    ``feature_importances_`` for the tree ensembles.  Each vector is normalised
    to sum to 1 so models can be plotted on the same axis.
    """
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    else:
        raw = np.abs(np.asarray(model.coef_, dtype=float))
    total = raw.sum() if raw.sum() > 0 else 1.0
    return pd.DataFrame({"model": name, "feature": FEATURE_COLUMNS, "importance": raw / total})


def train_and_evaluate(
    df: pd.DataFrame, test_season: int
) -> tuple[dict, dict[str, Pipeline], pd.DataFrame, pd.DataFrame, dict[str, dict], pd.DataFrame]:
    """Select hyper-parameters, fit every model, score both splits, collect predictions + importances."""
    train, test = chronological_split(df, test_season)
    log.info("Train: %d rows, seasons %s | Test: %d rows, season %d",
             len(train), sorted(train["season"].unique()), len(test), test_season)
    chosen, trials = select_hyperparameters(train)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    preds = test.copy()
    importances = []

    # --- baseline: finish where you start -----------------------------------
    for split_name, split in (("train", train), ("test", test)):
        m = compute_metrics(split.assign(predicted_score=split["grid_position"]))
        results.setdefault("baseline", {})[split_name] = m
    preds["pred_baseline"] = test["grid_position"].values
    log.info("baseline       test MAE %.3f  spearman %.3f", results["baseline"]["test"]["mae"], results["baseline"]["test"]["spearman"])

    # --- learned models ------------------------------------------------------
    for name, pipe in build_models(chosen).items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {
            "train": compute_metrics(train.assign(predicted_score=pipe.predict(X_train))),
            "test": compute_metrics(test.assign(predicted_score=pipe.predict(X_test))),
        }
        preds[f"pred_{name}"] = pipe.predict(X_test)
        importances.append(feature_importance(name, pipe))
        log.info("%-14s test MAE %.3f  spearman %.3f  podium-exact %.3f  exact-pos %.3f",
                 name, results[name]["test"]["mae"], results[name]["test"]["spearman"],
                 results[name]["test"]["podium_exact_rate"], results[name]["test"]["exact_position_acc"])

    return results, fitted, preds, pd.concat(importances, ignore_index=True), chosen, trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-season", type=int, default=None, help="season held out for testing (default: last in data)")
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_PATH)
    test_season = args.test_season or int(df.dropna(subset=[TARGET])["season"].max())

    results, fitted, preds, importances, chosen, trials = train_and_evaluate(df, test_season)

    learned = {k: v for k, v in results.items() if k != "baseline"}
    best_name = min(learned, key=lambda k: learned[k]["test"]["mae"])
    log.info("Best model by test MAE: %s", best_name)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipe in fitted.items():
        joblib.dump(pipe, MODELS_DIR / f"{name}.pkl")
    shutil.copyfile(MODELS_DIR / f"{best_name}.pkl", MODELS_DIR / "best_model.pkl")

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seasons": sorted(int(s) for s in df[df["season"] < test_season]["season"].unique()),
        "test_season": test_season,
        "n_train_rows": int((df["season"] < test_season).sum()),
        "n_test_rows": int((df["season"] == test_season).sum()),
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "best_model": best_name,
        "hyperparameters": chosen,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    importances.to_csv(IMPORTANCE_PATH, index=False)
    preds.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    trials.to_csv(MODELS_DIR / "hyperparameter_trials.csv", index=False)

    table = pd.DataFrame({k: v["test"] for k, v in results.items()}).T
    log.info("Test-season (%d) comparison:\n%s", test_season,
             table[["mae", "spearman", "podium_exact_rate", "podium_overlap", "exact_position_acc", "winner_acc"]].round(3).to_string())
    log.info("Saved models + metrics to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
