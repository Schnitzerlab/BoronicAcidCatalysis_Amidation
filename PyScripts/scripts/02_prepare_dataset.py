#!/usr/bin/env python3
"""Create a stable dataset manifest from an Excel/CSV input table.

Every downstream file is keyed by a reproducible internal ``MoleculeID``.
Original columns (including experimental conversion for training data) are kept.
The prepared dataset is written to a stable dataset-specific workspace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem

from common import dataset_dir, dump_json, ensure_dir, read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help="Stable dataset name, e.g. training, mono_library, di_library")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--molecule-column", default="Molecule")
    parser.add_argument(
        "--duplicate-policy",
        choices=["keep", "drop"],
        default="keep",
        help="Training data should normally use keep until duplicate provenance has been audited.",
    )
    args = parser.parse_args()

    df = read_table(args.input)
    if args.smiles_column not in df.columns:
        raise ValueError(f"SMILES column {args.smiles_column!r} not found in {args.input}")

    canonical = []
    valid = []
    for smi in df[args.smiles_column].astype(str):
        mol = Chem.MolFromSmiles(smi)
        valid.append(mol is not None)
        canonical.append(Chem.MolToSmiles(mol, canonical=True) if mol is not None else None)

    df = df.copy()
    df["CanonicalSMILES"] = canonical
    df["ValidSMILES"] = valid
    invalid = df.loc[~df["ValidSMILES"]]
    if not invalid.empty:
        bad_rows = (invalid.index + 2).tolist()  # +2 for Excel header + 1-based rows
        raise ValueError(f"Invalid SMILES detected at source rows: {bad_rows}")

    if args.duplicate_policy == "drop":
        df = df.drop_duplicates(subset=["CanonicalSMILES"], keep="first").reset_index(drop=True)

    # Preserve a human-readable source label while using a stable internal key.
    if args.molecule_column in df.columns:
        source_names = df[args.molecule_column].astype(str)
    else:
        source_names = pd.Series([f"source_{i:04d}" for i in range(1, len(df) + 1)])
        df.insert(0, args.molecule_column, source_names)

    df.insert(0, "MoleculeID", [f"mol_{i:04d}" for i in range(1, len(df) + 1)])
    # Put standardized SMILES column name in place even if source column differed.
    if args.smiles_column != "SMILES":
        df.insert(2, "SMILES", df[args.smiles_column])

    out_dir = ensure_dir(dataset_dir(args.dataset) / "00_manifest")
    write_table(df, out_dir / "manifest.xlsx")
    write_table(df, out_dir / "manifest.csv")

    duplicate_mask = df.duplicated("CanonicalSMILES", keep=False)
    duplicate_df = df.loc[duplicate_mask].copy()
    write_table(duplicate_df, out_dir / "duplicate_structures.csv")

    dump_json(
        {
            "dataset": args.dataset,
            "source_file": str(args.input.resolve()),
            "n_rows": int(len(df)),
            "n_unique_canonical_smiles": int(df["CanonicalSMILES"].nunique()),
            "n_duplicate_rows": int(duplicate_mask.sum()),
            "duplicate_policy": args.duplicate_policy,
            "smiles_column": args.smiles_column,
            "molecule_column": args.molecule_column,
        },
        out_dir / "dataset_info.json",
    )

    print(f"Prepared {len(df)} molecules -> {out_dir / 'manifest.xlsx'}")
    if duplicate_mask.any():
        print(f"Duplicate audit written to {out_dir / 'duplicate_structures.csv'}")


if __name__ == "__main__":
    main()
