#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TabPFN on the full experimental dataset and predict an in-silico library.

The modelling procedure is:

- the full experimental dataset is used for training (no train/test split here)
- median imputation is fitted on the experimental training data
- RobustScaler(quantile_range=(5, 95)) is fitted on the training data
- TunedTabPFNRegressor uses RMSE HPO
- the respective seed is used as random_state
- default seeds: 3, 43, 99, 604, 1704, 2005, 2508
- default HPO trials: 1, as in the supplied production-prediction workflow
- if HPO fails, the script falls back to plain TabPFNRegressor

The output filenames match the supplied production workflow so that step 12 can
aggregate them directly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

DEFAULT_SEEDS = [3, 43, 99, 604, 1704, 2005, 2508]
TARGET = "conversions (%)"
ID_COLUMN = "Molecule"
SMILES_COLUMN = "SMILES"
ROBUST_QUANTILE_RANGE = (5.0, 95.0)
HPO_METRIC = "rmse"

# Repository metadata columns are identifiers rather than model descriptors.
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
        f"Available columns:\n{list(df.columns)}"
    )


def extract_mol_number(mol: str) -> Optional[int]:
    match = re.search(r"(\d+)", str(mol))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TabPFN on full experimental data and predict an in-silico library."
    )
    parser.add_argument("--train-file", "--train_file", required=True, type=Path)
    parser.add_argument("--input", "--insilico-file", "--insilico_file", required=True, type=Path)
    parser.add_argument("--output", "--output-dir", "--output_dir",
                        type=Path, default=Path("predictions/tabpfn_insilico"))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n-trials", "--n_trials", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not args.input.exists():
        raise FileNotFoundError(f"In-silico file not found: {args.input}")

    # ------------------------------------------------------------------
    # Load training data and derive the descriptor feature list.
    # ------------------------------------------------------------------
    df_train = pd.read_excel(args.train_file)
    target_col = find_column(df_train, TARGET)
    train_id_col = find_column(df_train, ID_COLUMN)
    train_smiles_col = find_column(df_train, SMILES_COLUMN, required=False)

    excluded_train = {target_col, train_id_col} | REPOSITORY_METADATA
    if train_smiles_col is not None:
        excluded_train.add(train_smiles_col)
    feature_columns = [c for c in df_train.columns if c not in excluded_train]
    if not feature_columns:
        raise ValueError("No descriptor feature columns found in training file.")

    X_train_df = numeric_dataframe(df_train[feature_columns])
    y_series = convert_numeric_series(df_train[target_col])
    if y_series.isna().any():
        bad_rows = df_train.loc[y_series.isna(), [train_id_col, target_col]].copy()
        bad_rows.index = bad_rows.index + 2
        raise ValueError(
            "Target column contains missing/non-numeric values.\n"
            f"Problematic Excel rows:\n{bad_rows.to_string()}"
        )
    y_train = y_series.to_numpy(dtype=float)

    all_nan_train = [c for c in feature_columns if X_train_df[c].isna().all()]
    if all_nan_train:
        raise ValueError(
            "The following training feature columns contain no usable numeric values:\n"
            + "\n".join(f"  - {c}" for c in all_nan_train)
        )

    # ------------------------------------------------------------------
    # Load in-silico data and align to the exact training feature order.
    # ------------------------------------------------------------------
    df_insilico = pd.read_excel(args.input)
    insilico_id_col = find_column(df_insilico, ID_COLUMN)
    insilico_smiles_col = find_column(df_insilico, SMILES_COLUMN, required=False)

    missing = [c for c in feature_columns if c not in df_insilico.columns]
    if missing:
        raise ValueError(
            "In-silico file is missing descriptor columns required by training:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # Extra repository metadata are allowed; extra descriptor-like columns are not.
    allowed_meta = REPOSITORY_METADATA | {insilico_id_col, target_col}
    if insilico_smiles_col is not None:
        allowed_meta.add(insilico_smiles_col)
    extras = [
        c for c in df_insilico.columns
        if c not in feature_columns and c not in allowed_meta
    ]
    if extras:
        raise ValueError(
            "Unexpected non-training columns in in-silico file:\n"
            + "\n".join(f"  - {c}" for c in extras)
        )

    X_insilico_df = numeric_dataframe(df_insilico[feature_columns])
    all_nan_insilico = [c for c in feature_columns if X_insilico_df[c].isna().all()]
    if all_nan_insilico:
        raise ValueError(
            "The following in-silico feature columns contain no usable numeric values:\n"
            + "\n".join(f"  - {c}" for c in all_nan_insilico)
        )

    mol_ids = df_insilico[insilico_id_col].astype(str)
    mol_numbers = mol_ids.map(extract_mol_number)
    smiles_values = (
        df_insilico[insilico_smiles_col].astype(str).values
        if insilico_smiles_col is not None else None
    )

    train_missing = int(X_train_df.isna().sum().sum())
    insilico_missing = int(X_insilico_df.isna().sum().sum())

    # ------------------------------------------------------------------
    # Preprocessing is fit ONCE on the full experimental training set.
    # ------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_df)
    X_insilico_imp = imputer.transform(X_insilico_df)

    scaler = RobustScaler(quantile_range=ROBUST_QUANTILE_RANGE)
    X_train = scaler.fit_transform(X_train_imp)
    X_insilico = scaler.transform(X_insilico_imp)

    print("========================================================")
    print("TabPFN production prediction")
    print("========================================================")
    print(f"Training file       : {args.train_file}")
    print(f"In-silico file      : {args.input}")
    print(f"Training rows       : {len(df_train)}")
    print(f"In-silico rows      : {len(df_insilico)}")
    print(f"Descriptor features : {len(feature_columns)}")
    print(f"Seeds               : {args.seeds}")
    print(f"HPO trials/seed     : {args.n_trials}")
    print(f"Training missing    : {train_missing}")
    print(f"In-silico missing   : {insilico_missing}")
    print("Feature alignment   : OK")
    print("========================================================")

    (args.output / "feature_columns.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )

    for seed in args.seeds:
        print(f"\n=== Production prediction: seed {seed} ===")
        used_hpo = False
        best_params = None

        try:
            from tabpfn_extensions.hpo import TunedTabPFNRegressor
            model = TunedTabPFNRegressor(
                n_trials=args.n_trials,
                metric=HPO_METRIC,
                random_state=seed,
                verbose=True,
            )
            model.fit(X_train, y_train)
            used_hpo = True
            best_params = getattr(model, "best_params_", None)
        except Exception as exc:
            print("WARNING: HPO failed. Falling back to plain TabPFNRegressor.")
            print(f"Original exception: {exc}")
            from tabpfn import TabPFNRegressor
            model = TabPFNRegressor()
            model.fit(X_train, y_train)

        y_pred = np.asarray(model.predict(X_insilico), dtype=float)

        results = pd.DataFrame({
            "Molecule": mol_ids,
            "MolNumber": mol_numbers,
        })
        if smiles_values is not None:
            results["SMILES"] = smiles_values
        results["Prediction"] = y_pred
        results = results.sort_values("Prediction", ascending=False).reset_index(drop=True)
        results.insert(0, "Rank", np.arange(1, len(results) + 1))

        output_csv = args.output / f"in_silico_predictions_seed_{seed}.csv"
        output_xlsx = args.output / f"in_silico_predictions_seed_{seed}.xlsx"
        results.to_csv(output_csv, index=False)
        results.to_excel(output_xlsx, index=False)

        metadata_file = args.output / f"run_info_seed_{seed}.txt"
        with open(metadata_file, "w", encoding="utf-8") as handle:
            handle.write(f"Training file: {args.train_file}\n")
            handle.write(f"In-silico file: {args.input}\n")
            handle.write(f"Seed: {seed}\n")
            handle.write(f"N trials: {args.n_trials}\n")
            handle.write(f"Used HPO: {used_hpo}\n")
            handle.write(f"Number training rows: {len(df_train)}\n")
            handle.write(f"Number in-silico rows: {len(df_insilico)}\n")
            handle.write(f"Number features: {len(feature_columns)}\n")
            handle.write("Robust scaling: True\n")
            handle.write(f"Robust quantile range: {ROBUST_QUANTILE_RANGE}\n")
            handle.write(f"Missing training values before imputation: {train_missing}\n")
            handle.write(f"Missing in-silico values before imputation: {insilico_missing}\n")
            handle.write(f"Best params: {best_params}\n")

        print(f"CSV output   : {output_csv}")
        print(f"Excel output : {output_xlsx}")

    print(f"\nCompleted {len(args.seeds)} production prediction run(s) -> {args.output}")


if __name__ == "__main__":
    main()
