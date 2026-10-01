#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the seven TabPFN train/test evaluations used in the project.

The modelling procedure is:

- row-wise 80:20 train/test split
- seeds: 3, 43, 99, 604, 1704, 2005, 2508
- median imputation fitted on the training split only
- RobustScaler(quantile_range=(5, 95)) fitted on the training split only
- TunedTabPFNRegressor with RMSE HPO
- HPO/model random_state equals the respective split seed
- fallback to plain TabPFNRegressor if HPO fails

Repository-only metadata columns are ignored so that the descriptor matrix is
the same as in the original DATA table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_SEEDS = [3, 43, 99, 604, 1704, 2005, 2508]
TARGET = "conversions (%)"
ID_COLUMN = "Molecule"
SMILES_COLUMN = "SMILES"
HPO_METRIC = "rmse"
ROBUST_QUANTILE_RANGE = (5.0, 95.0)

# These repository metadata columns are identifiers rather than model descriptors.
REPOSITORY_METADATA = {
    "MoleculeID", "CanonicalSMILES", "ValidSMILES", "Intermediate", "SMILES_int",
    "Set", "Position", "Position 1", "Position 2", "Substituent",
    "Substituent 1", "Substituent 2",
}


def normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def find_column(df: pd.DataFrame, wanted: str, required: bool = True) -> Optional[str]:
    wanted_norm = normalize_header(wanted)
    matches = [c for c in df.columns if normalize_header(c) == wanted_norm]
    if len(matches) == 1:
        return matches[0]
    if not required and len(matches) == 0:
        return None
    raise ValueError(
        f"Expected exactly one column matching {wanted!r}, found {matches}.\n"
        f"Available columns: {list(df.columns)}"
    )


def extract_mol_number(value: object) -> Optional[int]:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def convert_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str).str.strip().str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = convert_numeric_series(df[col])
    return out


def save_real_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, out_svg: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_pred_train, s=28, facecolor="darkgray", edgecolor="gray",
                linewidth=0.8, alpha=0.35, label="Train")
    plt.scatter(y_test, y_pred_test, s=38, facecolor="tab:blue", edgecolor="navy",
                linewidth=0.8, alpha=0.45, label="Test")
    plt.plot([0, 100], [0, 100], linestyle="--", linewidth=2.2,
             color="darkgreen", label="Ideal")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("Experimental conversion (%)")
    plt.ylabel("Predicted conversion (%)")
    plt.legend(loc="upper left", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()


def save_parity_plot(y_true, y_pred, out_png: Path) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.8)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_residual_hist(y_true, y_pred, out_png: Path) -> None:
    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Predicted - True)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/tabpfn"))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-trials", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.input)

    target_col = find_column(df, TARGET)
    id_col = find_column(df, ID_COLUMN)
    smiles_col = find_column(df, SMILES_COLUMN, required=False)

    excluded = {target_col, id_col} | REPOSITORY_METADATA
    if smiles_col is not None:
        excluded.add(smiles_col)
    feature_columns = [c for c in df.columns if c not in excluded]
    if not feature_columns:
        raise ValueError("No descriptor feature columns found.")

    X_df = numeric_dataframe(df[feature_columns])
    y_series = convert_numeric_series(df[target_col])
    if y_series.isna().any():
        bad_rows = df.loc[y_series.isna(), [id_col, target_col]].copy()
        bad_rows.index = bad_rows.index + 2
        raise ValueError(
            "Target column contains missing/non-numeric values.\n"
            f"Problematic Excel rows:\n{bad_rows.to_string()}"
        )
    y = y_series.to_numpy(dtype=float)

    all_nan = [c for c in feature_columns if X_df[c].isna().all()]
    if all_nan:
        raise ValueError(
            "The following feature columns contain no usable numeric values:\n"
            + "\n".join(f"  - {c}" for c in all_nan)
        )

    mol_ids = df[id_col].astype(str)
    mol_numbers = mol_ids.map(extract_mol_number)
    summary = []

    for seed in args.split_seeds:
        run_dir = args.output / f"split_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== TabPFN seed {seed} ===")

        (X_train_df, X_test_df, y_train, y_test,
         mol_train, mol_test, moln_train, moln_test) = train_test_split(
            X_df, y, mol_ids.values, mol_numbers.values,
            test_size=args.test_size, random_state=seed,
        )

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_df)
        X_test_imp = imputer.transform(X_test_df)

        scaler = RobustScaler(quantile_range=ROBUST_QUANTILE_RANGE)
        X_train = scaler.fit_transform(X_train_imp)
        X_test = scaler.transform(X_test_imp)

        used_hpo = False
        best_params = None
        try:
            from tabpfn_extensions.hpo import TunedTabPFNRegressor
            regressor = TunedTabPFNRegressor(
                n_trials=args.n_trials,
                metric=HPO_METRIC,
                random_state=seed,
                verbose=True,
            )
            regressor.fit(X_train, y_train)
            used_hpo = True
            best_params = getattr(regressor, "best_params_", None)
        except Exception as exc:
            print("WARNING: HPO failed, falling back to plain TabPFN.")
            print(f"Original exception: {exc}")
            from tabpfn import TabPFNRegressor
            regressor = TabPFNRegressor()
            regressor.fit(X_train, y_train)

        y_pred_test = np.asarray(regressor.predict(X_test))
        y_pred_train = np.asarray(regressor.predict(X_train))

        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)

        pd.DataFrame([{
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "Used_HPO": used_hpo,
            "RandomSeed": seed,
            "TestSize": args.test_size,
            "N_Trials": args.n_trials,
        }]).to_csv(run_dir / "metrics.csv", index=False)

        pd.DataFrame({
            "Molecule": mol_test,
            "MolNumber": moln_test,
            "y_true": y_test,
            "y_pred": y_pred_test,
            "residual": y_pred_test - y_test,
        }).to_csv(run_dir / "predictions_test.csv", index=False)

        if best_params is not None:
            pd.DataFrame([best_params]).to_csv(run_dir / "best_hyperparameters.csv", index=False)

        save_real_vs_predicted(y_train, y_pred_train, y_test, y_pred_test,
                               run_dir / "real_vs_predicted.svg")
        save_parity_plot(y_test, y_pred_test, run_dir / "parity_test.png")
        save_residual_hist(y_test, y_pred_test, run_dir / "residuals_test.png")

        summary.append({
            "RandomSeed": seed,
            "Test_R2": r2,
            "Test_RMSE": rmse,
            "Test_MAE": mae,
            "Used_HPO": used_hpo,
            "N_Trials": args.n_trials,
        })
        print(f"Seed {seed}: R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")

    summary_df = pd.DataFrame(summary).sort_values("RandomSeed")
    summary_df.to_csv(args.output / "tabpfn_seed_metrics.csv", index=False)

    stats = []
    for metric in ["Test_R2", "Test_RMSE", "Test_MAE"]:
        vals = summary_df[metric].astype(float)
        stats.append({"Metric": metric, "Mean": vals.mean(), "Std": vals.std(ddof=1),
                      "Min": vals.min(), "Max": vals.max()})
    with pd.ExcelWriter(args.output / "TabPFN_seed_summary.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Per_seed_metrics", index=False)
        pd.DataFrame(stats).to_excel(writer, sheet_name="Summary_mean_std", index=False)

    (args.output / "feature_columns.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )
    print(f"\nCompleted {len(args.split_seeds)} TabPFN evaluation(s) -> {args.output}")


if __name__ == "__main__":
    main()
