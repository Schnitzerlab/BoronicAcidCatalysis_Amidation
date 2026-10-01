#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_benchmarking.py

Reviewer-requested baseline benchmarking for the TabPFN catalyst model.

Methods
-------
1. TabPFN (predictions from 10_train_tabpfn.py)
2. 1-nearest-neighbour using Morgan/Tanimoto (radius 2, 2048 bits)
3. Ortho-substitution / ortho-halogen ranking heuristic:
       no ortho substituent = 0
       other ortho substituent = 1
       ortho-F = 2, ortho-Cl = 3, ortho-Br = 4, ortho-I = 5
4. Random ranking baseline

Metrics
-------
For numerical predictors (TabPFN and 1-NN):
    R², MAE, RMSE

For all ranking methods:
    Hits@Top10%, Precision@Top10%, Enrichment Factor@Top10%,
    Top-k success@Top10%, ROC-AUC@Top10%, PR-AUC@Top10%,
    and the same metrics for Top20%.

Random and ortho-heuristic top-k metrics are averaged over repeated random
tie/ranking realizations; their within-split SDs are also saved.

The train/test membership is reconstructed from the same master table, row
order, test size and random seed used by 10_train_tabpfn.py, and is verified
against split_<seed>/predictions_test.csv.

Example
-------
python scripts/14_benchmarking.py \
    --data data/training/DATA_394_model.xlsx \
    --tabpfn-results results/tabpfn \
    --output results/reviewer/benchmarking
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)


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
        raise ValueError("Duplicate Molecule identifiers remain in the master dataset.")
    struct_dups = out["Canonical_SMILES"].duplicated(keep=False)
    if struct_dups.any():
        raise ValueError(
            "Exact structural duplicates remain in the master dataset. "
            "Remove them before final benchmarking.\n"
            + out.loc[struct_dups, ["Molecule", "Canonical_SMILES"]].head(40).to_string(index=False)
        )
    return out.reset_index(drop=True)


def seed_from_dir(path: Path) -> int:
    m = re.fullmatch(r"split_(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot extract seed from {path.name}")
    return int(m.group(1))


def discover_runs(root: Path) -> list[tuple[int, Path]]:
    runs = []
    for run_dir in root.glob("split_*"):
        if (run_dir / "predictions_test.csv").exists():
            runs.append((seed_from_dir(run_dir), run_dir))
    runs.sort(key=lambda x: x[0])
    if not runs:
        raise FileNotFoundError(f"No split_*/predictions_test.csv files found below {root}")
    return runs


def reconstruct_split(master: pd.DataFrame, seed: int, test_size: float):
    idx = np.arange(len(master))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    return master.iloc[train_idx].copy(), master.iloc[test_idx].copy()


def load_tabpfn_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    required = {"Molecule", "y_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    pred = pred[["Molecule", "y_pred"]].copy()
    pred["Molecule"] = pred["Molecule"].astype(str)
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    if pred["y_pred"].isna().any() or pred["Molecule"].duplicated().any():
        raise ValueError(f"Invalid or duplicate predictions in {path}")
    return pred


def ortho_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, 0, 0, 0, 0

    best_score = 0
    flags = {9: 0, 17: 0, 35: 0, 53: 0}

    for b_atom in mol.GetAtoms():
        if b_atom.GetAtomicNum() != 5:
            continue
        b_idx = b_atom.GetIdx()
        for ipso in b_atom.GetNeighbors():
            if ipso.GetAtomicNum() != 6:
                continue
            ipso_idx = ipso.GetIdx()
            for ortho in ipso.GetNeighbors():
                oi = ortho.GetIdx()
                if oi == b_idx or ortho.GetAtomicNum() != 6:
                    continue
                bond_io = mol.GetBondBetweenAtoms(ipso_idx, oi)
                if bond_io is None or not (bond_io.GetIsAromatic() or bond_io.IsInRing()):
                    continue

                for sub in ortho.GetNeighbors():
                    si = sub.GetIdx()
                    if si == ipso_idx:
                        continue
                    bond = mol.GetBondBetweenAtoms(oi, si)
                    if bond is None:
                        continue
                    if bond.GetIsAromatic() or bond.IsInRing():
                        continue

                    z = sub.GetAtomicNum()
                    if z == 53:
                        score = 5
                    elif z == 35:
                        score = 4
                    elif z == 17:
                        score = 3
                    elif z == 9:
                        score = 2
                    else:
                        score = 1
                    best_score = max(best_score, score)
                    if z in flags:
                        flags[z] = 1

    return best_score, flags[9], flags[17], flags[35], flags[53]


def one_nn_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, generator):
    train_fps = [
        generator.GetFingerprint(Chem.MolFromSmiles(s))
        for s in train_df["Canonical_SMILES"]
    ]
    train_y = train_df["Conversion"].to_numpy(float)
    train_names = train_df["Molecule"].tolist()

    preds, sims, names = [], [], []
    for row in test_df.itertuples(index=False):
        fp = generator.GetFingerprint(Chem.MolFromSmiles(row.Canonical_SMILES))
        vals = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        j = int(np.argmax(vals))
        preds.append(float(train_y[j]))
        sims.append(float(vals[j]))
        names.append(train_names[j])
    return np.asarray(preds), np.asarray(sims), names


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
    }


def hit_labels(y_true, fraction):
    n = len(y_true)
    k = max(1, int(math.ceil(fraction * n)))
    labels = np.zeros(n, dtype=int)
    labels[np.argsort(y_true)[-k:]] = 1
    return labels, k


def ranking_metrics(y_true, scores, fraction):
    y_true = np.asarray(y_true, float)
    scores = np.asarray(scores, float)
    labels, k = hit_labels(y_true, fraction)
    predicted_top = np.argsort(scores)[-k:]
    hits = int(labels[predicted_top].sum())
    precision = hits / k
    prevalence = labels.mean()
    ef = precision / prevalence
    success = float(hits > 0)
    roc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan
    pr = average_precision_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan
    return {
        "k": k,
        "Hits": hits,
        "Precision": precision,
        "EF": ef,
        "TopK_Success": success,
        "ROC_AUC": roc,
        "PR_AUC": pr,
    }


def repeated_random_metrics(y_true, fraction, repetitions, rng):
    values = []
    for _ in range(repetitions):
        scores = rng.random(len(y_true))
        values.append(ranking_metrics(y_true, scores, fraction))
    out = {}
    for key in ["Hits", "Precision", "EF", "TopK_Success", "ROC_AUC", "PR_AUC"]:
        arr = np.array([v[key] for v in values], float)
        out[key] = float(arr.mean())
        out[key + "_WithinSplit_SD"] = float(arr.std(ddof=1))
    out["k"] = values[0]["k"]
    return out


def repeated_tie_metrics(y_true, base_scores, fraction, repetitions, rng):
    base_scores = np.asarray(base_scores, float)
    values = []
    # Tiny jitter only breaks ties; it cannot reverse unequal heuristic scores.
    for _ in range(repetitions):
        jitter = rng.random(len(base_scores)) * 1e-9
        values.append(ranking_metrics(y_true, base_scores + jitter, fraction))
    out = {}
    for key in ["Hits", "Precision", "EF", "TopK_Success"]:
        arr = np.array([v[key] for v in values], float)
        out[key] = float(arr.mean())
        out[key + "_WithinSplit_SD"] = float(arr.std(ddof=1))
    labels, k = hit_labels(np.asarray(y_true, float), fraction)
    out["k"] = k
    out["ROC_AUC"] = roc_auc_score(labels, base_scores)
    out["PR_AUC"] = average_precision_score(labels, base_scores)
    out["ROC_AUC_WithinSplit_SD"] = 0.0
    out["PR_AUC_WithinSplit_SD"] = 0.0
    return out


def add_metrics_row(rows, seed, method, n_test, regression, m10, m20):
    row = {
        "Seed": seed,
        "Method": method,
        "N_Test": n_test,
        "R2": regression.get("R2", np.nan),
        "MAE": regression.get("MAE", np.nan),
        "RMSE": regression.get("RMSE", np.nan),
    }
    for label, metrics in [("Top10", m10), ("Top20", m20)]:
        for key, value in metrics.items():
            row[f"{key}_{label}"] = value
    rows.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--tabpfn-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="conversions (%)")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--random-repetitions", type=int, default=5000)
    parser.add_argument("--heuristic-repetitions", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    master = build_master(args.data, args.target)
    runs = discover_runs(args.tabpfn_results)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius, fpSize=args.fp_size
    )
    rng = np.random.default_rng(args.random_seed)

    metric_rows = []
    split_tables = {}

    for seed, run_dir in runs:
        train_df, test_df = reconstruct_split(master, seed, args.test_size)
        pred = load_tabpfn_predictions(run_dir / "predictions_test.csv")

        expected = set(test_df["Molecule"])
        observed = set(pred["Molecule"])
        if expected != observed:
            raise ValueError(
                f"Split {seed}: TabPFN prediction file does not match reconstructed test set."
            )
        test_df = test_df.merge(pred, on="Molecule", how="left", validate="one_to_one")
        y_true = test_df["Conversion"].to_numpy(float)
        tabpfn_pred = test_df["y_pred"].to_numpy(float)

        nn_pred, nn_sim, nn_names = one_nn_predictions(train_df, test_df, generator)

        ortho_score = np.asarray(
            [ortho_features(s)[0] for s in test_df["SMILES"]], dtype=float
        )

        for name, score in [
            ("TabPFN", tabpfn_pred),
            ("1-NN Morgan/Tanimoto", nn_pred),
        ]:
            reg = regression_metrics(y_true, score)
            m10 = ranking_metrics(y_true, score, 0.10)
            m20 = ranking_metrics(y_true, score, 0.20)
            add_metrics_row(metric_rows, seed, name, len(test_df), reg, m10, m20)

        h10 = repeated_tie_metrics(
            y_true, ortho_score, 0.10, args.heuristic_repetitions, rng
        )
        h20 = repeated_tie_metrics(
            y_true, ortho_score, 0.20, args.heuristic_repetitions, rng
        )
        add_metrics_row(
            metric_rows, seed, "Ortho-halogen heuristic", len(test_df), {}, h10, h20
        )

        r10 = repeated_random_metrics(
            y_true, 0.10, args.random_repetitions, rng
        )
        r20 = repeated_random_metrics(
            y_true, 0.20, args.random_repetitions, rng
        )
        add_metrics_row(
            metric_rows, seed, "Random", len(test_df), {}, r10, r20
        )

        split_tables[seed] = pd.DataFrame({
            "Molecule": test_df["Molecule"],
            "SMILES": test_df["SMILES"],
            "Experimental_Conversion": y_true,
            "TabPFN_Prediction": tabpfn_pred,
            "1NN_Prediction": nn_pred,
            "Max_Tanimoto_to_Training": nn_sim,
            "Nearest_Training_Molecule": nn_names,
            "Ortho_Halogen_Score": ortho_score,
        })

    metrics_df = pd.DataFrame(metric_rows).sort_values(["Method", "Seed"])

    summary_rows = []
    metric_columns = [
        "R2", "MAE", "RMSE",
        "Hits_Top10", "Precision_Top10", "EF_Top10", "TopK_Success_Top10",
        "ROC_AUC_Top10", "PR_AUC_Top10",
        "Hits_Top20", "Precision_Top20", "EF_Top20", "TopK_Success_Top20",
        "ROC_AUC_Top20", "PR_AUC_Top20",
    ]
    for method, subset in metrics_df.groupby("Method", sort=False):
        row = {"Method": method, "N_Splits": subset["Seed"].nunique()}
        for metric in metric_columns:
            vals = pd.to_numeric(subset.get(metric), errors="coerce").dropna()
            row[f"{metric}_Mean"] = vals.mean() if len(vals) else np.nan
            row[f"{metric}_SD"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[f"{metric}_Min"] = vals.min() if len(vals) else np.nan
            row[f"{metric}_Max"] = vals.max() if len(vals) else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    out_xlsx = args.output / "Benchmarking.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Per_Split_Metrics", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for seed, table in split_tables.items():
            table.to_excel(writer, sheet_name=f"Split_{seed}", index=False)

    metrics_df.to_csv(args.output / "Benchmarking_per_split.csv", index=False)
    summary_df.to_csv(args.output / "Benchmarking_summary.csv", index=False)

    # Plot enrichment factor for Top 10% and Top 20%.
    plot_df = summary_df.dropna(
        subset=["EF_Top10_Mean", "EF_Top20_Mean"], how="all"
    ).copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(plot_df))
        width = 0.35
        ax.bar(
            x - width/2, plot_df["EF_Top10_Mean"], width,
            yerr=plot_df["EF_Top10_SD"], capsize=4, label="Top 10%"
        )
        ax.bar(
            x + width/2, plot_df["EF_Top20_Mean"], width,
            yerr=plot_df["EF_Top20_SD"], capsize=4, label="Top 20%"
        )
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_ylabel("Enrichment factor")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Method"], rotation=30, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "Benchmarking_enrichment.png", dpi=300)
        plt.close(fig)

    # Plot Precision@k.
    plot_df = summary_df.dropna(
        subset=["Precision_Top10_Mean", "Precision_Top20_Mean"], how="all"
    ).copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(plot_df))
        width = 0.35
        ax.bar(
            x - width/2, plot_df["Precision_Top10_Mean"], width,
            yerr=plot_df["Precision_Top10_SD"], capsize=4, label="Top 10%"
        )
        ax.bar(
            x + width/2, plot_df["Precision_Top20_Mean"], width,
            yerr=plot_df["Precision_Top20_SD"], capsize=4, label="Top 20%"
        )
        ax.set_ylabel("Precision@k")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Method"], rotation=30, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "Benchmarking_precision.png", dpi=300)
        plt.close(fig)

    print("\nBenchmarking complete.")
    print(f"Splits analyzed: {len(runs)}")
    print(f"Workbook: {out_xlsx}")
    cols = [
        "Method", "N_Splits",
        "R2_Mean", "MAE_Mean",
        "Precision_Top10_Mean", "EF_Top10_Mean",
        "Precision_Top20_Mean", "EF_Top20_Mean",
    ]
    print(summary_df[[c for c in cols if c in summary_df.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
