#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:17:57 2025

@author: tobiasschnitzer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_pkl_using_training_cols.py

Apply a legacy TabPFN .pkl to DATA.xlsx by aligning columns to the exact
feature set (and order) derived from DATA_training.xlsx (Sheet1).

Steps:
1) Read DATA_training.xlsx:Sheet1 -> derive feature columns
2) Read DATA.xlsx:Sheet1 -> reorder/trim to training feature columns
3) Load final_model_tabpfn.pkl -> predict -> write 'conversions (%)'
4) Save DATA_with_predictions.xlsx and an alignment log
"""

import os
import warnings
import numpy as np
import pandas as pd

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False
    import pickle

# columns to exclude from features
EXCLUDE_DEFAULT = {
    "conversions (%)", "conversion (%)", "conversions(%)",
    "yield (%)", "y", "target", "label",
    "id", "ID", "Id", "index",
    "Intermediate", "SMILES_int", "Molecule", "SMILES",
}

MODEL_FILE = "final_model_tabpfn.pkl"
TRAIN_FILE = "DATA_training.xlsx"
APPLY_FILE = "DATA.xlsx"
SHEET = "Sheet1"
OUTPUT_FILE = "DATA_with_predictions.xlsx"
OUTPUT_LOG = "DATA_with_predictions_alignment_log.txt"


def load_df(path: str, sheet: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    if df.empty:
        raise ValueError(f"'{path}' is empty.")
    return df


def derive_training_features(df_train: pd.DataFrame):
    candidates = [c for c in df_train.columns if c not in EXCLUDE_DEFAULT]
    return candidates


def load_model(path: str):
    if _HAS_JOBLIB:
        try:
            return joblib.load(path)
        except Exception as e:
            warnings.warn(f"joblib.load failed ({e}), trying pickle.")
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    print(f"[INFO] Reading training data: {TRAIN_FILE} (sheet={SHEET})")
    df_train = load_df(TRAIN_FILE, SHEET)
    training_feats = derive_training_features(df_train)
    print(f"[INFO] Derived {len(training_feats)} feature columns from training.")

    print(f"[INFO] Reading application data: {APPLY_FILE} (sheet={SHEET})")
    df = load_df(APPLY_FILE, SHEET)

    missing = [c for c in training_feats if c not in df.columns]
    extra = [c for c in df.columns if c not in training_feats and c not in EXCLUDE_DEFAULT]

    if missing:
        raise ValueError(f"Application data is missing training features: {missing}")

    if extra:
        print(f"[ALIGN] Ignoring {len(extra)} extra column(s) not in training:")
        for c in extra:
            print("  -", c)

    X_df = df.reindex(columns=training_feats)
    X_num = X_df.apply(pd.to_numeric, errors="coerce")

    if X_num.shape[1] != len(training_feats):
        raise RuntimeError("Feature alignment error.")

    print(f"[INFO] Loading model: {MODEL_FILE}")
    model = load_model(MODEL_FILE)

    print("[INFO] Predicting…")
    y_pred = model.predict(X_num.values)
    y_pred = np.asarray(y_pred).ravel()

    df_out = df.copy()
    df_out["conversions (%)"] = y_pred
    df_out.to_excel(OUTPUT_FILE, index=False)
    print(f"[OK] Wrote predictions: {OUTPUT_FILE}")

    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("Training feature columns (order preserved):\n")
        for c in training_feats:
            f.write(f"  {c}\n")
        if missing:
            f.write("\nMissing (not filled):\n")
            for c in missing:
                f.write(f"  {c}\n")
        if extra:
            f.write("\nExtra columns ignored:\n")
            for c in extra:
                f.write(f"  {c}\n")
    print(f"[OK] Wrote alignment log: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()