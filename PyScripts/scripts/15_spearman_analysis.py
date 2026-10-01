#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate Spearman rank correlations for the seven TabPFN test splits.

This is the repository-integrated version of the final Spearman analysis used
for the project. It preserves the original analysis logic (Spearman rho for
each split, followed by the mean and sample standard deviation across splits)
but reads the native outputs of ``10_train_tabpfn.py`` directly.

Expected input
--------------
A folder containing one subfolder per TabPFN split, for example::

    results/tabpfn/
        split_3/predictions_test.csv
        split_43/predictions_test.csv
        split_99/predictions_test.csv
        split_604/predictions_test.csv
        split_1704/predictions_test.csv
        split_2005/predictions_test.csv
        split_2508/predictions_test.csv

Each ``predictions_test.csv`` must contain the columns ``y_true`` and ``y_pred``.

Outputs
-------
``Spearman_rank_correlation.xlsx``
    Sheet ``Per_Split``: Spearman rho (and p-value) for each split.
    Sheet ``Summary``: mean rho, sample SD (n-1), and number of splits.

``Spearman_rank_correlation.csv``
    Plain-text copy of the per-split table.

Example
-------
python scripts/15_spearman_analysis.py \
    --input results/tabpfn \
    --output results/reviewer/spearman
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


FILE_NAME = "predictions_test.csv"
DEFAULT_EXPECTED_SEEDS = 7


def seed_from_dir(path: Path) -> int:
    """Extract the numerical seed from a directory named ``split_<seed>``."""
    match = re.fullmatch(r"split_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot extract seed from directory name: {path.name}")
    return int(match.group(1))


def discover_prediction_files(root: Path) -> list[tuple[int, Path]]:
    """Find and sort all ``split_<seed>/predictions_test.csv`` files."""
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")

    runs: list[tuple[int, Path]] = []
    seen_seeds: set[int] = set()

    for split_dir in sorted(root.glob("split_*")):
        if not split_dir.is_dir():
            continue

        match = re.fullmatch(r"split_(\d+)", split_dir.name)
        if match is None:
            continue

        prediction_file = split_dir / FILE_NAME
        if not prediction_file.exists():
            continue

        seed = seed_from_dir(split_dir)
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed detected: {seed}")

        seen_seeds.add(seed)
        runs.append((seed, prediction_file))

    runs.sort(key=lambda item: item[0])

    if not runs:
        raise FileNotFoundError(
            f"No split_*/{FILE_NAME} files found below {root}"
        )

    return runs


def numeric_pair_columns(path: Path) -> tuple[list[float], list[float]]:
    """Read and validate ``y_true`` and ``y_pred`` from one split file."""
    df = pd.read_csv(path)

    for column in ("y_true", "y_pred"):
        if list(df.columns).count(column) != 1:
            raise ValueError(
                f"{path}: column {column!r} is missing or occurs more than once."
            )

    values = df[["y_true", "y_pred"]].copy()
    values["y_true"] = pd.to_numeric(values["y_true"], errors="coerce")
    values["y_pred"] = pd.to_numeric(values["y_pred"], errors="coerce")

    invalid = (
        values["y_true"].isna()
        | values["y_pred"].isna()
        | ~values["y_true"].map(math.isfinite)
        | ~values["y_pred"].map(math.isfinite)
    )

    if invalid.any():
        bad_rows = (invalid[invalid].index + 2).tolist()  # header = row 1
        raise ValueError(
            f"{path}: y_true/y_pred contain missing, non-numeric, or non-finite "
            f"values at CSV rows {bad_rows[:20]}."
        )

    if len(values) < 2:
        raise ValueError(f"{path}: fewer than two y_true/y_pred pairs.")

    if values["y_true"].nunique() < 2:
        raise ValueError(f"{path}: y_true is constant; Spearman rho is undefined.")

    if values["y_pred"].nunique() < 2:
        raise ValueError(f"{path}: y_pred is constant; Spearman rho is undefined.")

    return values["y_true"].tolist(), values["y_pred"].tolist()


def calculate(root: Path, expected_seeds: int) -> pd.DataFrame:
    """Calculate Spearman rho for all detected TabPFN test splits."""
    runs = discover_prediction_files(root)

    if expected_seeds > 0 and len(runs) != expected_seeds:
        raise ValueError(
            f"Expected {expected_seeds} split files, but found {len(runs)} below {root}. "
            "Use --expected-seeds 0 to disable this check."
        )

    rows = []

    for seed, path in runs:
        y_true, y_pred = numeric_pair_columns(path)
        result = spearmanr(y_true, y_pred)
        rho = float(result.statistic)
        p_value = float(result.pvalue)

        if not (math.isfinite(rho) and math.isfinite(p_value)):
            raise ValueError(
                f"Split {seed}: Spearman calculation returned a non-finite value."
            )

        rows.append(
            {
                "Seed": seed,
                "N_Test": len(y_true),
                "Spearman_rho": rho,
                "Spearman_p_value": p_value,
                "Source": str(path),
            }
        )

    return pd.DataFrame(rows).sort_values("Seed").reset_index(drop=True)


def export(results: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    """Write per-split and summary results to XLSX and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    values = results["Spearman_rho"].astype(float).tolist()
    if len(values) < 2:
        raise ValueError(
            "At least two split correlations are required for a sample standard deviation."
        )

    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values)  # sample SD, denominator n-1

    summary = pd.DataFrame(
        [
            {"Metric": "Mean Spearman rho", "Value": mean_value},
            {"Metric": "Sample SD Spearman rho (n-1)", "Value": std_value},
            {"Metric": "Number of splits", "Value": len(values)},
        ]
    )

    xlsx_path = output_dir / "Spearman_rank_correlation.xlsx"
    csv_path = output_dir / "Spearman_rank_correlation.csv"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Per_Split", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    results.to_csv(csv_path, index=False)

    return xlsx_path, csv_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Spearman rank correlation between experimental and predicted "
            "conversion for each TabPFN test split."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Folder containing split_<seed>/predictions_test.csv from step 10.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output folder for the Spearman result files.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=DEFAULT_EXPECTED_SEEDS,
        help="Expected number of splits (default: 7; use 0 to disable the check).",
    )
    args = parser.parse_args()

    results = calculate(args.input, args.expected_seeds)
    xlsx_path, csv_path, summary = export(results, args.output)

    print("=" * 60)
    print("Spearman rank-correlation analysis")
    print("=" * 60)

    for row in results.itertuples(index=False):
        print(
            f"Seed {row.Seed:>4}: rho={row.Spearman_rho:.6f} "
            f"(p={row.Spearman_p_value:.4g}, n={row.N_Test})"
        )

    mean_value = float(summary.loc[summary["Metric"] == "Mean Spearman rho", "Value"].iloc[0])
    std_value = float(summary.loc[summary["Metric"] == "Sample SD Spearman rho (n-1)", "Value"].iloc[0])

    print(
        f"\nMean ± sample SD: {mean_value:.6f} ± {std_value:.6f}"
    )
    print(f"Excel output: {xlsx_path.resolve()}")
    print(f"CSV output  : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
