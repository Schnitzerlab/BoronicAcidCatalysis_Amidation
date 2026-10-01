#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate TabPFN in-silico predictions across seeds and create ranked outputs.

The script reads the per-seed CSV files produced by ``11_apply_models.py``:
``in_silico_predictions_seed_<SEED>.csv``.

For every molecule it verifies cross-seed consistency, then calculates the mean
predicted conversion, sample standard deviation, standard error of the mean,
and the number of seeds. Results are ranked by the mean prediction.

Outputs:
- an Excel workbook with Summary, All_Seed_Predictions, and Sources sheets;
- a CSV summary;
- optionally, a PDF report containing ranked 2D molecular structures.

Relative paths are resolved from the current working directory, which should
normally be the repository root.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PREDICTION_FOLDER = "tabpfn_insilico_predictions"
DEFAULT_EXPECTED_SEEDS = 7

DEFAULT_OUTPUT_XLSX = "in_silico_averaged_predictions.xlsx"
DEFAULT_OUTPUT_CSV = "in_silico_averaged_predictions.csv"
DEFAULT_OUTPUT_PDF = "insilico_ranked_structures.pdf"

FILE_PATTERN = "in_silico_predictions_seed_*.csv"

REQUIRED_COLUMNS = {
    "Molecule",
    "SMILES",
    "Prediction",
}


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average TabPFN in-silico predictions over random seeds and create "
            "a ranked molecular-structure PDF."
        )
    )

    parser.add_argument(
        "--prediction-folder",
        "--input",
        dest="prediction_folder",
        default=DEFAULT_PREDICTION_FOLDER,
        help=(
            "Folder containing in_silico_predictions_seed_*.csv. "
            f"Default: {DEFAULT_PREDICTION_FOLDER}"
        ),
    )

    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=DEFAULT_EXPECTED_SEEDS,
        help=(
            "Expected number of unique seed files. Default: 7. "
            "Use 0 to accept any number of seeds."
        ),
    )

    parser.add_argument(
        "--output-xlsx",
        default=DEFAULT_OUTPUT_XLSX,
        help=f"Excel output. Default: {DEFAULT_OUTPUT_XLSX}",
    )

    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"CSV output. Default: {DEFAULT_OUTPUT_CSV}",
    )

    parser.add_argument(
        "--output-pdf",
        default=DEFAULT_OUTPUT_PDF,
        help=f"PDF structure report. Default: {DEFAULT_OUTPUT_PDF}",
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only aggregate predictions; do not create the structure PDF.",
    )

    return parser.parse_args()


# =============================================================================
# Path helpers
# =============================================================================

def script_directory() -> Path:
    """Return the directory containing this Python script."""
    return Path(__file__).resolve().parent


def resolve_relative_to_script(path_string: str) -> Path:
    """
    Resolve paths for the consolidated repository workflow.

    Absolute paths are left unchanged. Relative paths are resolved from the
    current working directory (normally the repository root), so commands such
    as ``--prediction-folder predictions/mono_test3`` work when this script is
    stored in ``scripts/``.
    """
    path = Path(path_string).expanduser()

    if path.is_absolute():
        return path

    return Path.cwd() / path


# =============================================================================
# General helpers
# =============================================================================

def extract_seed(path: Path) -> int:
    """
    Extract integer seed from:
        in_silico_predictions_seed_43.csv
    """
    match = re.fullmatch(
        r"in_silico_predictions_seed_(\d+)\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not extract seed number from filename: {path.name}"
        )

    return int(match.group(1))


def extract_mol_number(value: object) -> Optional[int]:
    """
    Extract the first integer from a molecule ID.

    Examples:
        mol_1   -> 1
        mol_403 -> 403
    """
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def robust_numeric(series: pd.Series, column_name: str, source: Path) -> pd.Series:
    """
    Convert a series robustly to numeric.

    Handles:
        12.34
        "12.34"
        "12,34"

    Raises a clear error if conversion fails.
    """
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace("\u00a0", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        cleaned = cleaned.replace(
            {
                "": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "NULL": np.nan,
                "null": np.nan,
            }
        )

        numeric = pd.to_numeric(cleaned, errors="coerce")

    if numeric.isna().any():
        bad_indices = numeric[numeric.isna()].index
        examples = series.loc[bad_indices].head(20)

        raise ValueError(
            f"Missing or non-numeric values in column '{column_name}' "
            f"of file:\n{source}\n\n"
            f"Problematic values:\n{examples.to_string()}"
        )

    return numeric.astype(float)


# =============================================================================
# Prediction file discovery
# =============================================================================

def find_prediction_files(prediction_folder: Path) -> list[Path]:
    """
    Find prediction CSV files.

    rglob() is intentionally used so the script remains robust if another
    subfolder level is added later.

    Only CSV files matching FILE_PATTERN are selected.
    XLSX and run_info files are ignored automatically.
    """
    if not prediction_folder.exists():
        raise FileNotFoundError(
            "Prediction folder not found:\n"
            f"{prediction_folder}\n\n"
            "Expected it next to this Python script."
        )

    if not prediction_folder.is_dir():
        raise NotADirectoryError(
            f"Prediction path is not a directory: {prediction_folder}"
        )

    files = sorted(prediction_folder.rglob(FILE_PATTERN))

    if not files:
        raise RuntimeError(
            f"No files matching '{FILE_PATTERN}' were found below:\n"
            f"{prediction_folder}"
        )

    return files


def map_unique_seed_files(files: list[Path]) -> dict[int, Path]:
    """
    Create a {seed: file} mapping and refuse duplicate files for a seed.

    This prevents one seed from accidentally receiving double weight.
    """
    by_seed: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = {}

    for path in files:
        seed = extract_seed(path)

        if seed in by_seed:
            duplicates.setdefault(seed, [by_seed[seed]]).append(path)
        else:
            by_seed[seed] = path

    if duplicates:
        lines = [
            "Duplicate CSV files found for one or more seeds.",
            "A seed must not be counted more than once.",
            "",
        ]

        for seed, paths in sorted(duplicates.items()):
            lines.append(f"Seed {seed}:")
            for path in paths:
                lines.append(f"  - {path}")

        raise RuntimeError("\n".join(lines))

    return by_seed


# =============================================================================
# Load one prediction CSV
# =============================================================================

def load_prediction_file(path: Path, seed: int) -> pd.DataFrame:
    """
    Load one seed prediction file and normalize it to:

        Molecule
        MolNumber
        SMILES
        Seed_<seed>
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path)

    # Strip accidental whitespace around column names.
    df.columns = [str(c).strip() for c in df.columns]

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise ValueError(
            f"File is missing required columns:\n{path}\n\n"
            f"Missing: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    out = pd.DataFrame(index=df.index)

    # -------------------------------------------------------------------------
    # Molecule
    # -------------------------------------------------------------------------
    out["Molecule"] = df["Molecule"].astype(str).str.strip()

    bad_molecule = (
        out["Molecule"].eq("")
        | out["Molecule"].str.lower().isin({"nan", "none"})
    )

    if bad_molecule.any():
        raise ValueError(
            f"Missing molecule identifiers found in:\n{path}"
        )

    duplicated = out["Molecule"].duplicated(keep=False)

    if duplicated.any():
        duplicate_ids = sorted(out.loc[duplicated, "Molecule"].unique())

        raise ValueError(
            f"Duplicate molecule IDs found in:\n{path}\n\n"
            f"Examples: {duplicate_ids[:20]}"
        )

    # -------------------------------------------------------------------------
    # MolNumber
    # -------------------------------------------------------------------------
    derived_mol_number = out["Molecule"].map(extract_mol_number)

    if "MolNumber" in df.columns:
        supplied_mol_number = pd.to_numeric(
            df["MolNumber"],
            errors="coerce",
        )

        out["MolNumber"] = supplied_mol_number.where(
            supplied_mol_number.notna(),
            derived_mol_number,
        )
    else:
        out["MolNumber"] = derived_mol_number

    # -------------------------------------------------------------------------
    # SMILES
    # -------------------------------------------------------------------------
    out["SMILES"] = df["SMILES"].astype(str).str.strip()

    bad_smiles = (
        out["SMILES"].eq("")
        | out["SMILES"].str.lower().isin({"nan", "none"})
    )

    if bad_smiles.any():
        bad_ids = out.loc[bad_smiles, "Molecule"].tolist()

        raise ValueError(
            f"Missing SMILES in:\n{path}\n\n"
            f"Affected molecules: {bad_ids[:20]}"
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    out[f"Seed_{seed}"] = robust_numeric(
        df["Prediction"],
        column_name="Prediction",
        source=path,
    ).to_numpy()

    return out


# =============================================================================
# Cross-seed validation + aggregation
# =============================================================================

def validate_and_align(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_seed: int,
    current_seed: int,
) -> pd.DataFrame:
    """
    Validate that current and reference contain exactly the same molecules
    and identical SMILES.

    Row order may differ; current is aligned to the reference molecule order.
    """
    reference_ids = set(reference["Molecule"])
    current_ids = set(current["Molecule"])

    missing = sorted(reference_ids - current_ids)
    extra = sorted(current_ids - reference_ids)

    if missing or extra:
        lines = [
            "Molecule mismatch between seed files.",
            f"Reference seed: {reference_seed}",
            f"Current seed:   {current_seed}",
        ]

        if missing:
            lines.append(
                f"\nMissing from seed {current_seed}: {missing[:30]}"
            )

        if extra:
            lines.append(
                f"\nExtra in seed {current_seed}: {extra[:30]}"
            )

        raise ValueError("\n".join(lines))

    # Align current to reference row order.
    aligned = (
        current.set_index("Molecule")
        .loc[reference["Molecule"]]
        .reset_index()
    )

    # Compare SMILES molecule by molecule.
    ref_smiles = reference.set_index("Molecule")["SMILES"]
    cur_smiles = aligned.set_index("Molecule")["SMILES"]

    smiles_difference = ref_smiles != cur_smiles

    if smiles_difference.any():
        bad_molecules = ref_smiles.index[smiles_difference].tolist()

        examples = []
        for molecule in bad_molecules[:20]:
            examples.append(
                f"{molecule}\n"
                f"  seed {reference_seed}: {ref_smiles.loc[molecule]}\n"
                f"  seed {current_seed}: {cur_smiles.loc[molecule]}"
            )

        raise ValueError(
            "SMILES mismatch between seed files.\n\n"
            + "\n".join(examples)
        )

    return aligned


def aggregate_predictions(
    seed_files: dict[int, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate predictions over all seeds.

    Returns
    -------
    summary_df
        Ranked averaged predictions.

    all_seed_df
        Molecule metadata + one prediction column per seed.

    sources_df
        Seed-to-file mapping.
    """
    seed_items = sorted(seed_files.items())

    if not seed_items:
        raise RuntimeError("No seed files supplied for aggregation.")

    reference_seed, reference_path = seed_items[0]
    reference = load_prediction_file(reference_path, reference_seed)

    all_seed_df = reference[
        ["Molecule", "MolNumber", "SMILES", f"Seed_{reference_seed}"]
    ].copy()

    for seed, path in seed_items[1:]:
        current = load_prediction_file(path, seed)

        current = validate_and_align(
            reference=reference,
            current=current,
            reference_seed=reference_seed,
            current_seed=seed,
        )

        all_seed_df[f"Seed_{seed}"] = current[f"Seed_{seed}"].to_numpy()

    prediction_columns = [
        f"Seed_{seed}"
        for seed, _ in seed_items
    ]

    prediction_matrix = all_seed_df[prediction_columns].to_numpy(dtype=float)

    n_seeds = len(prediction_columns)

    average = prediction_matrix.mean(axis=1)

    if n_seeds > 1:
        std_dev = prediction_matrix.std(axis=1, ddof=1)
        std_error = std_dev / math.sqrt(n_seeds)
    else:
        std_dev = np.full(len(all_seed_df), np.nan)
        std_error = np.full(len(all_seed_df), np.nan)

    summary_df = pd.DataFrame(
        {
            "Molecule": all_seed_df["Molecule"],
            "MolNumber": all_seed_df["MolNumber"],
            "SMILES": all_seed_df["SMILES"],
            "Average_Prediction": average,
            "Std_Dev": std_dev,
            "Std_Error": std_error,
            "N_Seeds": n_seeds,
        }
    )

    # Do not silently clip predictions outside 0-100.
    outside_range = (
        (summary_df["Average_Prediction"] < 0)
        | (summary_df["Average_Prediction"] > 100)
    )

    if outside_range.any():
        print(
            "WARNING: "
            f"{int(outside_range.sum())} averaged prediction(s) are outside 0-100 %. "
            "Values are kept unchanged."
        )

    summary_df = (
        summary_df
        .sort_values(
            by="Average_Prediction",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    summary_df.insert(
        0,
        "Rank",
        np.arange(1, len(summary_df) + 1),
    )

    sources_df = pd.DataFrame(
        {
            "Seed": [seed for seed, _ in seed_items],
            "File": [str(path) for _, path in seed_items],
        }
    )

    return summary_df, all_seed_df, sources_df


# =============================================================================
# Output tables
# =============================================================================

def write_tables(
    summary_df: pd.DataFrame,
    all_seed_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    output_xlsx: Path,
    output_csv: Path,
) -> None:
    """
    Write clean averaged table as CSV and one Excel workbook.

    No additional helper files are created.
    """
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        output_xlsx,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        all_seed_df.to_excel(
            writer,
            sheet_name="All_Seed_Predictions",
            index=False,
        )

        sources_df.to_excel(
            writer,
            sheet_name="Sources",
            index=False,
        )


# =============================================================================
# Molecular structure PDF
# =============================================================================

def create_structure_pdf(
    summary_df: pd.DataFrame,
    output_pdf: Path,
) -> list[str]:
    """
    Create an A4 PDF ranked by average predicted conversion.

    Every entry contains:
        rank
        molecule ID
        average prediction ± SEM
        2D molecular structure

    Returns a list of molecules with invalid SMILES.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.graphics import renderPDF
        from reportlab.lib.units import mm

        from svglib.svglib import svg2rlg

    except ImportError as exc:
        raise ImportError(
            "Could not create the structure PDF because a required package "
            "is missing.\n\n"
            "Required packages:\n"
            "  rdkit\n"
            "  reportlab\n"
            "  svglib\n\n"
            f"Original error: {exc}"
        ) from exc

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    def smiles_to_mol(smiles: str):
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            AllChem.Compute2DCoords(mol)

        return mol

    def mol_to_svg(
        mol,
        width: int,
        height: int,
    ) -> str:
        drawer = rdMolDraw2D.MolDraw2DSVG(
            width,
            height,
        )

        opts = drawer.drawOptions()
        opts.useBWAtomPalette()
        opts.fixedBondLength = 30
        opts.minFontSize = 14

        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        return drawer.GetDrawingText()

    c = canvas.Canvas(
        str(output_pdf),
        pagesize=A4,
    )

    page_width, page_height = A4

    # 12 molecules per page, as in the previous structure report.
    n_columns = 3
    n_rows = 4
    molecules_per_page = n_columns * n_rows

    margin = 17 * mm

    usable_width = page_width - 2 * margin
    usable_height = page_height - 2 * margin

    cell_width = usable_width / n_columns
    cell_height = usable_height / n_rows

    svg_width = int(cell_width * 0.90)
    svg_height = int(cell_height * 0.63)

    data = summary_df.reset_index(drop=True)

    number_of_pages = max(
        1,
        math.ceil(len(data) / molecules_per_page),
    )

    invalid_smiles: list[str] = []

    index = 0

    for page_number in range(number_of_pages):

        for row_index in range(n_rows):
            for column_index in range(n_columns):

                if index >= len(data):
                    break

                row = data.iloc[index]

                cell_left = margin + column_index * cell_width
                cell_bottom = (
                    page_height
                    - margin
                    - (row_index + 1) * cell_height
                )

                center_x = cell_left + cell_width / 2

                # Leave room below the structure for two text lines.
                structure_y = cell_bottom + cell_height * 0.32

                molecule = smiles_to_mol(
                    str(row["SMILES"])
                )

                if molecule is not None:

                    svg = mol_to_svg(
                        molecule,
                        svg_width,
                        svg_height,
                    )

                    drawing = svg2rlg(
                        io.BytesIO(
                            svg.encode("utf-8")
                        )
                    )

                    structure_x = (
                        cell_left
                        + (cell_width - svg_width) / 2
                    )

                    renderPDF.draw(
                        drawing,
                        c,
                        structure_x,
                        structure_y,
                    )

                else:
                    invalid_smiles.append(
                        str(row["Molecule"])
                    )

                    c.setFont(
                        "Helvetica",
                        8,
                    )

                    c.drawCentredString(
                        center_x,
                        cell_bottom + cell_height * 0.60,
                        "Invalid SMILES",
                    )

                # -------------------------------------------------------------
                # Text block
                # -------------------------------------------------------------
                text_y = cell_bottom + cell_height * 0.25

                c.setFont(
                    "Helvetica-Bold",
                    8,
                )

                c.drawCentredString(
                    center_x,
                    text_y,
                    f'#{int(row["Rank"])} | {row["Molecule"]}',
                )

                mean_prediction = float(
                    row["Average_Prediction"]
                )

                sem = row["Std_Error"]
                n_seeds = int(row["N_Seeds"])

                if pd.notna(sem):
                    prediction_text = (
                        f"{mean_prediction:.1f}% "
                        f"± {float(sem):.2f} SEM "
                        f"(n={n_seeds})"
                    )
                else:
                    prediction_text = (
                        f"{mean_prediction:.1f}% "
                        f"(n={n_seeds})"
                    )

                c.setFont(
                    "Helvetica",
                    8,
                )

                c.drawCentredString(
                    center_x,
                    text_y - 10,
                    prediction_text,
                )

                index += 1

        # Page number
        c.setFont(
            "Helvetica",
            7,
        )

        c.drawRightString(
            page_width - margin,
            8 * mm,
            f"Page {page_number + 1}/{number_of_pages}",
        )

        c.showPage()

    c.save()

    return invalid_smiles


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    root = script_directory()

    prediction_folder = resolve_relative_to_script(
        args.prediction_folder
    )

    output_xlsx = resolve_relative_to_script(
        args.output_xlsx
    )

    output_csv = resolve_relative_to_script(
        args.output_csv
    )

    output_pdf = resolve_relative_to_script(
        args.output_pdf
    )

    print("============================================================")
    print("TabPFN in-silico prediction analysis")
    print("============================================================")
    print(f"Script folder    : {root}")
    print(f"Prediction folder: {prediction_folder}")
    print()

    # -------------------------------------------------------------------------
    # Locate CSV prediction files
    # -------------------------------------------------------------------------
    prediction_files = find_prediction_files(
        prediction_folder
    )

    seed_files = map_unique_seed_files(
        prediction_files
    )

    found_seeds = sorted(seed_files)

    print(
        f"Found {len(found_seeds)} unique prediction CSV file(s)."
    )

    print(
        "Seeds: "
        + ", ".join(str(seed) for seed in found_seeds)
    )

    for seed in found_seeds:
        print(
            f"  Seed {seed:>5}: {seed_files[seed].name}"
        )

    print()

    # -------------------------------------------------------------------------
    # Number-of-seeds validation
    # -------------------------------------------------------------------------
    if args.expected_seeds > 0:

        if len(found_seeds) != args.expected_seeds:

            raise RuntimeError(
                f"Expected {args.expected_seeds} unique seed files, "
                f"but found {len(found_seeds)}.\n"
                f"Found seeds: {found_seeds}\n\n"
                "If this is intentional, run with:\n"
                "  --expected-seeds 0\n"
                "or specify another expected number."
            )

        print(
            f"Seed count check  : OK ({args.expected_seeds})"
        )

    else:
        print(
            "Seed count check  : disabled"
        )

    print()

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    summary_df, all_seed_df, sources_df = aggregate_predictions(
        seed_files
    )

    print(
        f"Molecules         : {len(summary_df)}"
    )

    print(
        "Cross-seed check  : OK "
        "(same molecules and SMILES)"
    )

    print()

    # -------------------------------------------------------------------------
    # Write average table
    # -------------------------------------------------------------------------
    write_tables(
        summary_df=summary_df,
        all_seed_df=all_seed_df,
        sources_df=sources_df,
        output_xlsx=output_xlsx,
        output_csv=output_csv,
    )

    print(
        f"Excel summary     : {output_xlsx}"
    )

    print(
        f"CSV summary       : {output_csv}"
    )

    # -------------------------------------------------------------------------
    # Structure PDF
    # -------------------------------------------------------------------------
    if args.no_pdf:
        print(
            "Structure PDF     : skipped (--no-pdf)"
        )

    else:
        try:
            invalid_smiles = create_structure_pdf(
                summary_df=summary_df,
                output_pdf=output_pdf,
            )

            print(
                f"Structure PDF     : {output_pdf}"
            )

            if invalid_smiles:
                print(
                    f"WARNING: {len(invalid_smiles)} invalid SMILES."
                )

                for molecule in invalid_smiles[:20]:
                    print(
                        f"  - {molecule}"
                    )

            else:
                print(
                    "SMILES drawing    : OK"
                )

        except ImportError as exc:
            # The averaged prediction outputs are already safe on disk.
            print()
            print(
                "WARNING: Averaging completed successfully, "
                "but the PDF could not be created."
            )
            print(exc)

    print()
    print("============================================================")
    print("Finished.")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("ERROR")
        print("=====")
        print(exc)
        sys.exit(1)
