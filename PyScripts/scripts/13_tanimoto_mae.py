#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_tanimoto_mae.py

Reviewer-requested applicability-domain analysis for the TabPFN workflow.

For every held-out test compound in every TabPFN split, this script:
  1. reconstructs the exact train/test split from the master dataset and seed,
  2. calculates the maximum Morgan/Tanimoto similarity to the corresponding
     training set (Morgan radius 2, 2048 bits),
  3. joins the TabPFN prediction from split_<seed>/predictions_test.csv,
  4. calculates signed and absolute prediction errors,
  5. reports Spearman correlation between similarity and absolute error,
  6. reports MAE in predefined similarity bins,
  7. optionally calculates the similarity of named target compounds
     (e.g. FIBA/MIBA/DBA) to each training split, excluding exact self-matches.

The script is deliberately independent of model retraining. It consumes the
outputs of 10_train_tabpfn.py and the same master dataset used in that step.

Example
-------
python scripts/13_tanimoto_mae.py \
    --data data/training/DATA_394_model.xlsx \
    --tabpfn-results results/tabpfn \
    --output results/reviewer/tanimoto

Optional target compounds:
python scripts/13_tanimoto_mae.py \
    --data data/training/DATA_394_model.xlsx \
    --tabpfn-results results/tabpfn \
    --output results/reviewer/tanimoto \
    --targets data/reviewer/target_compounds.xlsx
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


DEFAULT_BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
DEFAULT_LABELS = ["<0.40", "0.40–0.50", "0.50–0.60", "0.60–0.70", "0.70–0.80", "≥0.80"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_header(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def find_column(df: pd.DataFrame, wanted: str, required: bool = True) -> str | None:
    key = normalize_header(wanted)
    matches = [c for c in df.columns if normalize_header(c) == key]
    if len(matches) == 1:
        return matches[0]
    if not required and not matches:
        return None
    raise ValueError(f"Expected exactly one column matching {wanted!r}; found {matches}.")


def canonicalize(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def seed_from_dir(path: Path) -> int:
    m = re.fullmatch(r"split_(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot extract split seed from directory: {path}")
    return int(m.group(1))


def build_master(path: Path, target_name: str) -> pd.DataFrame:
    raw = read_table(path)
    mol_col = find_column(raw, "Molecule")
    smiles_col = find_column(raw, "SMILES")
    target_col = find_column(raw, target_name)

    out = pd.DataFrame({
        "Molecule": raw[mol_col].astype(str),
        "SMILES": raw[smiles_col],
        "Conversion": pd.to_numeric(raw[target_col], errors="coerce"),
    })
    out["Canonical_SMILES"] = out["SMILES"].map(canonicalize)

    bad = out["Canonical_SMILES"].isna() | out["Conversion"].isna()
    if bad.any():
        raise ValueError(
            "Master dataset contains invalid SMILES or non-numeric target values:\n"
            + out.loc[bad, ["Molecule", "SMILES", "Conversion"]].head(30).to_string(index=False)
        )
    if out["Molecule"].duplicated().any():
        dups = out.loc[out["Molecule"].duplicated(keep=False), "Molecule"].tolist()
        raise ValueError(f"Duplicate Molecule identifiers in master dataset: {dups[:30]}")

    struct_dups = out["Canonical_SMILES"].duplicated(keep=False)
    if struct_dups.any():
        examples = out.loc[struct_dups, ["Molecule", "Canonical_SMILES"]]
        raise ValueError(
            "Exact structural duplicates remain in the master dataset. "
            "Remove them before the final applicability-domain analysis.\n"
            + examples.head(40).to_string(index=False)
        )
    return out.reset_index(drop=True)


def make_fingerprints(master: pd.DataFrame, radius: int, fp_size: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps = {}
    for row in master.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.Canonical_SMILES)
        fps[row.Molecule] = generator.GetFingerprint(mol)
    return generator, fps


def maximum_similarity(query_fp, train_fps, train_names):
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
    best = int(np.argmax(sims))
    return float(sims[best]), train_names[best]


def discover_runs(root: Path) -> list[tuple[int, Path]]:
    runs = []
    for run_dir in root.glob("split_*"):
        pred = run_dir / "predictions_test.csv"
        if pred.exists():
            runs.append((seed_from_dir(run_dir), run_dir))
    runs.sort(key=lambda x: x[0])
    if not runs:
        raise FileNotFoundError(f"No split_*/predictions_test.csv files found below {root}")
    return runs


def reconstruct_split(master: pd.DataFrame, seed: int, test_size: float):
    idx = np.arange(len(master))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed
    )
    return master.iloc[train_idx].copy(), master.iloc[test_idx].copy()


def read_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    required = {"Molecule", "y_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    pred = pred[["Molecule", "y_pred"]].copy()
    pred["Molecule"] = pred["Molecule"].astype(str)
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    if pred["y_pred"].isna().any():
        raise ValueError(f"Non-numeric predictions found in {path}")
    if pred["Molecule"].duplicated().any():
        raise ValueError(f"Duplicate Molecule identifiers in {path}")
    return pred


def analyze_targets(target_path: Path, runs, master, fps, generator, test_size):
    targets_raw = read_table(target_path)
    smiles_col = find_column(targets_raw, "SMILES")
    label_col = (
        find_column(targets_raw, "Label", required=False)
        or find_column(targets_raw, "Molecule", required=False)
    )
    if label_col is None:
        labels = pd.Series([f"Target_{i+1}" for i in range(len(targets_raw))])
    else:
        labels = targets_raw[label_col].astype(str)

    targets = pd.DataFrame({"Target": labels, "SMILES": targets_raw[smiles_col]})
    targets["Canonical_SMILES"] = targets["SMILES"].map(canonicalize)
    if targets["Canonical_SMILES"].isna().any():
        raise ValueError("Target file contains invalid SMILES.")

    rows = []
    for seed, _ in runs:
        train_df, _ = reconstruct_split(master, seed, test_size)
        train_rows = list(train_df.itertuples(index=False))
        for target in targets.itertuples(index=False):
            qmol = Chem.MolFromSmiles(target.Canonical_SMILES)
            qfp = generator.GetFingerprint(qmol)

            candidate_fps = []
            candidate_names = []
            excluded = 0
            for row in train_rows:
                if row.Canonical_SMILES == target.Canonical_SMILES:
                    excluded += 1
                    continue
                candidate_fps.append(fps[row.Molecule])
                candidate_names.append(row.Molecule)

            if candidate_fps:
                sim, nearest = maximum_similarity(qfp, candidate_fps, candidate_names)
            else:
                sim, nearest = np.nan, None

            rows.append({
                "Seed": seed,
                "Target": target.Target,
                "SMILES": target.SMILES,
                "Max_Tanimoto_to_Training": sim,
                "Nearest_Training_Molecule": nearest,
                "Exact_Matches_Excluded": excluded,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path,
                        help="Master training table used by 10_train_tabpfn.py.")
    parser.add_argument("--tabpfn-results", required=True, type=Path,
                        help="Folder containing split_<seed>/predictions_test.csv.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="conversions (%)")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--targets", type=Path, default=None,
                        help="Optional XLSX/CSV with SMILES and Label or Molecule.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    master = build_master(args.data, args.target)
    generator, fps = make_fingerprints(master, args.radius, args.fp_size)
    runs = discover_runs(args.tabpfn_results)

    all_rows = []
    split_summary = []

    for seed, run_dir in runs:
        train_df, test_df = reconstruct_split(master, seed, args.test_size)
        pred = read_predictions(run_dir / "predictions_test.csv")

        expected = set(test_df["Molecule"])
        observed = set(pred["Molecule"])
        if expected != observed:
            raise ValueError(
                f"Split {seed}: predictions_test.csv does not match the reconstructed test set.\n"
                f"Missing predictions: {sorted(expected-observed)[:20]}\n"
                f"Unexpected predictions: {sorted(observed-expected)[:20]}"
            )

        test = test_df.merge(pred, on="Molecule", how="left", validate="one_to_one")
        train_names = train_df["Molecule"].tolist()
        train_fps = [fps[m] for m in train_names]
        train_lookup = train_df.set_index("Molecule")

        rows = []
        for row in test.itertuples(index=False):
            sim, nearest = maximum_similarity(fps[row.Molecule], train_fps, train_names)
            nearest_row = train_lookup.loc[nearest]
            signed = float(row.y_pred) - float(row.Conversion)
            rows.append({
                "Seed": seed,
                "Molecule": row.Molecule,
                "SMILES": row.SMILES,
                "Experimental_Conversion": float(row.Conversion),
                "Predicted_Conversion": float(row.y_pred),
                "Signed_Error": signed,
                "Absolute_Error": abs(signed),
                "Max_Tanimoto_to_Training": sim,
                "Nearest_Training_Molecule": nearest,
                "Nearest_Training_SMILES": nearest_row["SMILES"],
                "Nearest_Training_Conversion": float(nearest_row["Conversion"]),
            })

        split_df = pd.DataFrame(rows)
        rho, p = spearmanr(
            split_df["Max_Tanimoto_to_Training"],
            split_df["Absolute_Error"],
        )
        split_summary.append({
            "Seed": seed,
            "N_Train": len(train_df),
            "N_Test": len(split_df),
            "MAE": split_df["Absolute_Error"].mean(),
            "Median_Absolute_Error": split_df["Absolute_Error"].median(),
            "Mean_Max_Tanimoto": split_df["Max_Tanimoto_to_Training"].mean(),
            "Median_Max_Tanimoto": split_df["Max_Tanimoto_to_Training"].median(),
            "Spearman_rho": rho,
            "Spearman_p": p,
        })
        all_rows.append(split_df)

    all_df = pd.concat(all_rows, ignore_index=True)
    split_summary_df = pd.DataFrame(split_summary).sort_values("Seed")

    all_df["Similarity_Bin"] = pd.cut(
        all_df["Max_Tanimoto_to_Training"],
        bins=DEFAULT_BINS,
        labels=DEFAULT_LABELS,
        include_lowest=True,
        right=False,
    )

    per_split_bins = (
        all_df.groupby(["Seed", "Similarity_Bin"], observed=True)
        .agg(
            N=("Absolute_Error", "size"),
            MAE=("Absolute_Error", "mean"),
            SD_Absolute_Error=("Absolute_Error", "std"),
            Median_Absolute_Error=("Absolute_Error", "median"),
            Mean_Similarity=("Max_Tanimoto_to_Training", "mean"),
            Min_Similarity=("Max_Tanimoto_to_Training", "min"),
            Max_Similarity=("Max_Tanimoto_to_Training", "max"),
        )
        .reset_index()
    )

    bin_summary = (
        per_split_bins.groupby("Similarity_Bin", observed=True)
        .agg(
            N_Splits=("Seed", "nunique"),
            Mean_MAE_across_splits=("MAE", "mean"),
            SD_MAE_across_splits=("MAE", "std"),
            Min_split_MAE=("MAE", "min"),
            Max_split_MAE=("MAE", "max"),
            Mean_similarity=("Mean_Similarity", "mean"),
        )
        .reset_index()
    )

    combined_rho, combined_p = spearmanr(
        all_df["Max_Tanimoto_to_Training"],
        all_df["Absolute_Error"],
    )

    overall_df = pd.DataFrame([
        {"Metric": "Number of splits", "Value": len(split_summary_df)},
        {"Metric": "Mean MAE across splits", "Value": split_summary_df["MAE"].mean()},
        {"Metric": "SD MAE across splits", "Value": split_summary_df["MAE"].std(ddof=1)},
        {"Metric": "Mean Spearman rho across splits", "Value": split_summary_df["Spearman_rho"].mean()},
        {"Metric": "SD Spearman rho across splits", "Value": split_summary_df["Spearman_rho"].std(ddof=1)},
        {"Metric": "Combined Spearman rho", "Value": combined_rho},
        {"Metric": "Combined Spearman p-value", "Value": combined_p},
    ])

    target_df = None
    if args.targets is not None:
        target_df = analyze_targets(
            args.targets, runs, master, fps, generator, args.test_size
        )

    xlsx = args.output / "Tanimoto_vs_MAE.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="All_Test_Predictions", index=False)
        split_summary_df.to_excel(writer, sheet_name="Per_Split_Summary", index=False)
        per_split_bins.to_excel(writer, sheet_name="MAE_Bins_Per_Split", index=False)
        bin_summary.to_excel(writer, sheet_name="MAE_Bins_Summary", index=False)
        overall_df.to_excel(writer, sheet_name="Overall_Summary", index=False)
        if target_df is not None:
            target_df.to_excel(writer, sheet_name="Target_Similarity", index=False)

    all_df.to_csv(args.output / "Tanimoto_vs_MAE_all_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        all_df["Max_Tanimoto_to_Training"],
        all_df["Absolute_Error"],
        alpha=0.45,
        s=24,
    )
    ax.set_xlabel("Maximum Morgan/Tanimoto similarity to training set")
    ax.set_ylabel("Absolute prediction error / %-points")
    ax.set_xlim(0, 1.02)
    ax.text(
        0.02, 0.98,
        f"Spearman ρ = {combined_rho:.2f}\np = {combined_p:.3g}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(args.output / "Error_vs_Tanimoto.png", dpi=300)
    plt.close(fig)

    if not bin_summary.empty:
        order = [lab for lab in DEFAULT_LABELS if lab in set(bin_summary["Similarity_Bin"].astype(str))]
        plot_df = bin_summary.copy()
        plot_df["Similarity_Bin"] = plot_df["Similarity_Bin"].astype(str)
        plot_df = plot_df.set_index("Similarity_Bin").reindex(order).dropna(subset=["Mean_MAE_across_splits"]).reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(plot_df))
        ax.bar(
            x,
            plot_df["Mean_MAE_across_splits"],
            yerr=plot_df["SD_MAE_across_splits"],
            capsize=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Similarity_Bin"], rotation=30, ha="right")
        ax.set_xlabel("Maximum Morgan/Tanimoto similarity to training set")
        ax.set_ylabel("MAE / %-points")
        fig.tight_layout()
        fig.savefig(args.output / "MAE_by_Tanimoto_bin.png", dpi=300)
        plt.close(fig)

    if target_df is not None and not target_df.empty:
        target_summary = (
            target_df.groupby("Target")
            .agg(
                Mean_Max_Tanimoto=("Max_Tanimoto_to_Training", "mean"),
                SD_Max_Tanimoto=("Max_Tanimoto_to_Training", "std"),
                Min_Max_Tanimoto=("Max_Tanimoto_to_Training", "min"),
                Max_Max_Tanimoto=("Max_Tanimoto_to_Training", "max"),
            )
            .reset_index()
        )
        target_summary.to_csv(args.output / "Target_similarity_summary.csv", index=False)

    print("\nTanimoto/MAE analysis complete.")
    print(f"Splits analyzed: {len(runs)}")
    print(f"Workbook: {xlsx}")
    print(f"Combined Spearman rho: {combined_rho:.3f} (p={combined_p:.4g})")


if __name__ == "__main__":
    main()
