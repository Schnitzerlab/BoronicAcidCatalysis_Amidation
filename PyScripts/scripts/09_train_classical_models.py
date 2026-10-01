#!/usr/bin/env python3
"""Train the classical model-development grid from one reproducible script.

Supported model-development combinations:
- models: Decision Tree, Random Forest, Gradient Boosting, XGBoost
- tuning: RandomizedSearchCV or Optuna
- feature selection: RFE(3/5/7/10/15), RFECV, embedded importance, Boruta

Reviewer-oriented safeguards:
1. The test set is split once and never used for feature selection or HPO.
2. Feature selection is fitted on training data only.
3. Hyperparameter tuning uses training-set CV only.
4. Configurations are ranked by CV/HPO score, not by test-set performance.
5. Exact selected features, seeds, parameters, train/test IDs, and metrics are saved.
6. Legacy Fukui columns are explicitly excluded.
7. Row-wise CV uses 5-fold KFold without shuffling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from common import ensure_dir, exclude_fukui_columns, write_table

MODELS = ["decision_tree", "random_forest", "gradient_boosting", "xgboost"]
TUNINGS = ["random", "optuna"]
SELECTORS = ["rfe3", "rfe5", "rfe7", "rfe10", "rfe15", "rfecv", "embedded", "boruta"]

METADATA_EXCLUDE = {
    "MoleculeID",
    "Molecule",
    "SMILES",
    "CanonicalSMILES",
    "ValidSMILES",
    "Intermediate",
    "SMILES_int",
    "Set",
    "Position",
    "Position 1",
    "Position 2",
    "Substituent",
    "Substituent 1",
    "Substituent 2",
}


def r2_weighted_high(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = y_true.copy().astype(float)
    weights[weights < 1] = 1
    return r2_score(y_true, y_pred, sample_weight=weights)


WEIGHTED_R2 = make_scorer(r2_weighted_high, greater_is_better=True)


def model_factory(name: str, seed: int):
    if name == "decision_tree":
        return DecisionTreeRegressor(random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed)
    if name == "xgboost":
        return XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
    raise ValueError(name)


def random_space(name: str):
    if name == "decision_tree":
        return {
            "max_depth": [None] + list(range(3, 31, 2)),
            "min_samples_split": [2, 3, 5, 7, 10, 15, 20, 30, 50],
            "min_samples_leaf": [1, 2, 3, 5, 7, 10, 20],
            "max_features": ["auto", "sqrt", "log2", 0.2, 0.3, 0.5, None],
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        }
    if name == "random_forest":
        return {
            "n_estimators": [50, 100, 200, 300, 500, 800, 1000],
            "max_depth": [None] + list(range(3, 21, 2)),
            "min_samples_split": [2, 3, 5, 7, 10, 15, 20, 30, 50],
            "min_samples_leaf": [1, 2, 3, 5, 7, 10, 20],
            "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5, None],
            "bootstrap": [True, False],
        }
    if name == "gradient_boosting":
        return {
            "n_estimators": [50, 100, 200, 400, 600, 1000],
            "learning_rate": np.logspace(-3, -0.5, 10),
            "max_depth": list(range(2, 11)),
            "min_samples_split": [2, 3, 5, 7, 10, 20, 50],
            "min_samples_leaf": [1, 2, 3, 5, 10, 20],
            "max_features": ["sqrt", "log2", 0.2, 0.5, 1.0],
            "subsample": np.linspace(0.5, 1.0, 6),
            "loss": ["squared_error", "absolute_error", "huber", "quantile"],
        }
    if name == "xgboost":
        return {
            "n_estimators": [100, 200, 400, 600, 1000],
            "learning_rate": np.logspace(-3, -0.5, 10),
            "max_depth": [3, 5, 7, 10, 15, 20],
            "min_child_weight": [1, 3, 5, 7, 10, 20],
            "subsample": np.linspace(0.5, 1.0, 6),
            "colsample_bytree": np.linspace(0.3, 1.0, 8),
            "gamma": np.logspace(-3, 1, 10),
            "reg_alpha": np.logspace(-3, 1, 10),
            "reg_lambda": np.logspace(-3, 1, 10),
        }
    raise ValueError(name)


def optuna_params(trial, name: str):
    if name == "decision_tree":
        return {
            "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(3, 31, 2))),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 3, 5, 7, 10, 15, 20, 30, 50]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 3, 5, 7, 10, 20]),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", 0.2, 0.3, 0.5, None]),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 500, 800, 1000]),
            "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(3, 21, 2))),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 3, 5, 7, 10, 15, 20, 30, 50]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 3, 5, 7, 10, 20]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.3, 0.5, None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 400, 600, 1000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 3, 5, 7, 10, 20, 50]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 3, 5, 10, 20]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.5, 1.0]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "loss": trial.suggest_categorical("loss", ["squared_error", "absolute_error", "huber", "quantile"]),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400, 600, 1000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, 15, 20]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 7, 10, 20]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0.001, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
        }
    raise ValueError(name)


def select_features(selector_name, X_train, y_train, model_name, seed, cv_folds):
    base = model_factory(model_name, seed)
    if selector_name.startswith("rfe") and selector_name != "rfecv":
        n = int(selector_name.replace("rfe", ""))
        selector = RFE(base, n_features_to_select=min(n, X_train.shape[1]))
        selector.fit(X_train, y_train)
        selected = X_train.columns[selector.support_].tolist()
    elif selector_name == "rfecv":
        selector = RFECV(
            base,
            step=1,
            cv=KFold(n_splits=cv_folds, shuffle=False),
            scoring=WEIGHTED_R2,
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)
        selected = X_train.columns[selector.support_].tolist()
    elif selector_name == "embedded":
        base.fit(X_train, y_train)
        importances = pd.Series(base.feature_importances_, index=X_train.columns)
        selected = importances[importances > 0].sort_values(ascending=False).index.tolist()
        if not selected:
            selected = importances.sort_values(ascending=False).head(min(10, len(importances))).index.tolist()
    elif selector_name == "boruta":
        from boruta import BorutaPy
        rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        selector = BorutaPy(rf, verbose=0, random_state=seed)
        selector.fit(X_train.values, np.asarray(y_train))
        selected = X_train.columns[selector.support_].tolist()
        if not selected:
            selected = X_train.columns.tolist()
    else:
        raise ValueError(selector_name)
    return selected


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "Weighted_R2": r2_weighted_high(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def run_one(model_name, tuning, selector_name, X_train, X_test, y_train, y_test, train_ids, test_ids, groups_train, args):
    run_dir = ensure_dir(args.output / model_name / tuning / selector_name)
    selected = select_features(selector_name, X_train, y_train, model_name, args.seed, args.cv_folds)
    Xtr = X_train[selected]
    Xte = X_test[selected]

    n_unique_groups = pd.Series(groups_train).nunique() if groups_train is not None else 0
    use_group_cv = groups_train is not None and n_unique_groups >= 2
    # Row-wise datasets use deterministic KFold without shuffling.
    # Grouped datasets use GroupKFold.
    cv_splitter = (
        GroupKFold(n_splits=min(args.cv_folds, int(n_unique_groups)))
        if use_group_cv
        else KFold(n_splits=args.cv_folds, shuffle=False)
    )

    if tuning == "random":
        search = RandomizedSearchCV(
            model_factory(model_name, args.seed),
            param_distributions=random_space(model_name),
            n_iter=args.random_iterations,
            scoring=WEIGHTED_R2,
            cv=cv_splitter,
            n_jobs=-1,
            random_state=args.seed,
            return_train_score=True,
            verbose=args.verbose,
        )
        if use_group_cv:
            search.fit(Xtr, y_train, groups=np.asarray(groups_train))
        else:
            search.fit(Xtr, y_train)
        best_params = search.best_params_
        hpo_score = float(search.best_score_)
        pd.DataFrame(search.cv_results_).to_csv(run_dir / "hpo_trials.csv", index=False)
    else:
        import optuna

        def objective(trial):
            params = optuna_params(trial, model_name)
            model = model_factory(model_name, args.seed)
            model.set_params(**params)
            kwargs = {
                "cv": cv_splitter,
                "scoring": WEIGHTED_R2,
                "n_jobs": -1,
            }
            if use_group_cv:
                kwargs["groups"] = np.asarray(groups_train)
            return float(cross_val_score(model, Xtr, y_train, **kwargs).mean())

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
        )
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=args.progress)
        best_params = study.best_params
        hpo_score = float(study.best_value)
        study.trials_dataframe().to_csv(run_dir / "hpo_trials.csv", index=False)

    final_model = model_factory(model_name, args.seed)
    final_model.set_params(**best_params)
    final_model.fit(Xtr, y_train)

    pred_train = final_model.predict(Xtr)
    pred_test = final_model.predict(Xte)
    train_m = metrics(y_train, pred_train)
    test_m = metrics(y_test, pred_test)

    # Report a second CV evaluation of the selected best model.
    cv_kwargs = {"cv": cv_splitter, "scoring": WEIGHTED_R2, "n_jobs": -1}
    if use_group_cv:
        cv_kwargs["groups"] = np.asarray(groups_train)
    final_cv_scores = cross_val_score(final_model, Xtr, y_train, **cv_kwargs)
    final_cv_mean = float(np.mean(final_cv_scores))
    final_cv_std = float(np.std(final_cv_scores))

    artifact = {
        "model": final_model,
        "features": selected,
        "model_name": model_name,
        "tuning": tuning,
        "selector": selector_name,
        "split_seed": args.seed,
        "target": args.target,
        "best_params": best_params,
        "hpo_cv_weighted_r2": hpo_score,
        "split_mode": args._actual_split_mode,
    }
    joblib.dump(artifact, run_dir / "model.joblib")
    (run_dir / "selected_features.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    (run_dir / "best_parameters.json").write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "MoleculeID": list(train_ids) + list(test_ids),
            "Set": ["Training"] * len(train_ids) + ["Test"] * len(test_ids),
            "Observed conversion (%)": list(y_train) + list(y_test),
            "Predicted conversion (%)": list(pred_train) + list(pred_test),
        }
    )
    pred_df.to_excel(run_dir / "predictions.xlsx", index=False)

    # Reproducible diagnostic figure for the individual run.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_train, pred_train, alpha=0.35, label="Training")
    ax.scatter(y_test, pred_test, alpha=0.65, label="Test")
    lo = min(float(np.min(y_train)), float(np.min(y_test)), float(np.min(pred_train)), float(np.min(pred_test)))
    hi = max(float(np.max(y_train)), float(np.max(y_test)), float(np.max(pred_train)), float(np.max(pred_test)))
    ax.plot([lo, hi], [lo, hi], "--", label="Ideal")
    ax.set_xlabel("Experimental conversion (%)")
    ax.set_ylabel("Predicted conversion (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "observed_vs_predicted.png", dpi=300)
    plt.close(fig)

    row = {
        "model": model_name,
        "tuning": tuning,
        "selector": selector_name,
        "n_features": len(selected),
        "HPO_CV_Weighted_R2": hpo_score,
        **{f"Train_{k}": v for k, v in train_m.items()},
        **{f"Test_{k}": v for k, v in test_m.items()},
        "CV_r2_weighted_high_mean": final_cv_mean,
        "CV_r2_weighted_high_std": final_cv_std,
    }
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Training feature table (.xlsx/.csv)")
    parser.add_argument("--target", default="conversions (%)")
    parser.add_argument("--output", type=Path, default=Path("results/classical_models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["auto", "row", "group"], default="auto", help="auto uses group splitting only when duplicate CanonicalSMILES are present.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-iterations", type=int, default=2000)
    parser.add_argument("--optuna-trials", type=int, default=1000)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--tunings", nargs="+", choices=TUNINGS, default=TUNINGS)
    parser.add_argument("--selectors", nargs="+", choices=SELECTORS, default=SELECTORS)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output)
    df = pd.read_excel(args.input) if args.input.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target!r} not found.")

    df = df.dropna(subset=[args.target]).copy()
    candidate = [c for c in df.columns if c not in METADATA_EXCLUDE | {args.target}]
    candidate = exclude_fukui_columns(candidate)
    numeric = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric feature columns found after exclusions.")

    # Only complete rows enter the classical models.
    cols_for_drop = numeric + [args.target]
    clean_idx = df[cols_for_drop].dropna().index
    df = df.loc[clean_idx].reset_index(drop=True)
    X = df[numeric]
    y = df[args.target]
    ids = df["MoleculeID"].astype(str) if "MoleculeID" in df.columns else df.index.astype(str)

    canonical = df["CanonicalSMILES"].astype(str) if "CanonicalSMILES" in df.columns else None
    has_duplicate_groups = canonical is not None and canonical.duplicated(keep=False).any()
    use_group_split = args.split_mode == "group" or (args.split_mode == "auto" and has_duplicate_groups)
    if use_group_split:
        groups = canonical if canonical is not None else ids
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        id_train, id_test = ids.iloc[train_idx], ids.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        actual_split_mode = "group"
    else:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, ids, test_size=args.test_size, random_state=args.seed
        )
        groups_train = None
        actual_split_mode = "row"

    split_df = pd.DataFrame({"MoleculeID": list(id_train) + list(id_test), "Set": ["Training"] * len(id_train) + ["Test"] * len(id_test)})
    split_df["SplitMode"] = actual_split_mode
    split_df.to_csv(args.output / "train_test_split.csv", index=False)
    args._actual_split_mode = actual_split_mode
    (args.output / "feature_columns.txt").write_text("\n".join(numeric) + "\n", encoding="utf-8")

    summary = []
    for model_name in args.models:
        for tuning in args.tunings:
            for selector in args.selectors:
                print(f"=== {model_name} | {tuning} | {selector} ===")
                try:
                    row = run_one(
                        model_name,
                        tuning,
                        selector,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        id_train,
                        id_test,
                        groups_train,
                        args,
                    )
                except Exception as exc:
                    row = {
                        "model": model_name,
                        "tuning": tuning,
                        "selector": selector,
                        "status": f"FAILED: {type(exc).__name__}: {exc}",
                    }
                    print(row["status"])
                summary.append(row)
                pd.DataFrame(summary).to_csv(args.output / "model_grid_summary.csv", index=False)

    summary_df = pd.DataFrame(summary)
    if "HPO_CV_Weighted_R2" in summary_df.columns:
        summary_df = summary_df.sort_values("HPO_CV_Weighted_R2", ascending=False, na_position="last")
    summary_df.to_excel(args.output / "model_grid_summary.xlsx", index=False)
    print("Model grid complete. Configurations are ordered by training-CV score, not test performance.")


if __name__ == "__main__":
    main()
