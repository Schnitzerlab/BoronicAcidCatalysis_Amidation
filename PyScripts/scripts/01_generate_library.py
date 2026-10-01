#!/usr/bin/env python3
"""Generate mono- and disubstituted phenylboronic-acid virtual libraries.

Structures are canonicalized during generation, symmetry-related duplicates
are removed, and the resulting libraries are written with explicit filenames.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from common import ensure_dir, project_root, write_table

SUBSTITUENTS_MONO = {
    "Methyl": "C",
    "Isopropyl": "C(C)C",
    "Phenyl": "C2=CC=CC=C2",
    "Benzoxy": "OCC2=CC=CC=C2",
    "Hydroxy": "O",
    "Methoxy": "OC",
    "Acetate": "OC(=O)C",
    "Acetylamine": "NC(=O)C",
    "Fluoride": "F",
    "Chloride": "Cl",
    "Bromide": "Br",
    "Iodide": "I",
    "Carbaldehyde": "C=O",
    "Methylcarboxylate": "C(=O)OC",
    "Carboxamide": "C(=O)N",
    "Methylthiolate": "SC",
    "Trifluoromethyl": "C(F)(F)F",
    "Cyano": "C#N",
    "Nitro": "[N+](=O)[O-]",
    "Boronic Acid": "B(O)O",
}

# Second substituent needs different ring digits for the embedded phenyl/benzoxy
# fragments used by the original disubstituted-library script.
SUBSTITUENTS_DI_2 = {
    **SUBSTITUENTS_MONO,
    "Phenyl": "C3=CC=CC=C3",
    "Benzoxy": "OCC3=CC=CC=C3",
    "Boronic Acid": "B(O)(O)",
}
SUBSTITUENTS_DI_1 = {**SUBSTITUENTS_MONO, "Boronic Acid": "B(O)(O)"}

DI_TEMPLATES = {
    "OB(O)C1=CC=CC({sub2})=C1{sub1}": (6, 5),
    "OB(O)C1=CC=C({sub2})C=C1{sub1}": (6, 4),
    "OB(O)C1=CC({sub2})=CC=C1{sub1}": (6, 3),
    "OB(O)C1=C({sub2})C=CC=C1{sub1}": (6, 2),
    "OB(O)C1=CC=CC({sub1})=C1{sub2}": (5, 6),
    "OB(O)C1=CC=C({sub2})C({sub1})=C1": (5, 4),
    "OB(O)C1=CC({sub2})=CC({sub1})=C1": (5, 3),
    "OB(O)C1=C({sub2})C=CC({sub1})=C1": (5, 2),
    "OB(O)C1=CC=C({sub1})C=C1{sub2}": (4, 6),
    "OB(O)C1=CC=C({sub1})C({sub2})=C1": (4, 5),
    "OB(O)C1=CC({sub2})=C({sub1})C=C1": (4, 3),
    "OB(O)C1=C({sub2})C=C({sub1})C=C1": (4, 2),
    "OB(O)C1=CC({sub1})=CC=C1{sub2}": (3, 6),
    "OB(O)C1=CC({sub1})=CC({sub2})=C1": (3, 5),
    "OB(O)C1=CC({sub1})=C({sub2})C=C1": (3, 4),
    "OB(O)C1=C({sub2})C({sub1})=CC=C1": (3, 2),
    "OB(O)C1=C({sub1})C=CC=C1{sub2}": (2, 6),
    "OB(O)C1=C({sub1})C=CC({sub2})=C1": (2, 5),
    "OB(O)C1=C({sub1})C=C({sub2})C=C1": (2, 4),
    "OB(O)C1=C({sub1})C({sub2})=CC=C1": (2, 3),
}

MIRROR = {2: 6, 3: 5, 4: 4, 5: 3, 6: 2}


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def normalize_positions(pos1: int, pos2: int) -> tuple[int, int]:
    original = (pos1, pos2)
    mirrored = (MIRROR[pos1], MIRROR[pos2])
    # Standardize symmetry-equivalent positional assignments: prefer the
    # orientation with the smaller minimum position, then smaller sum.
    key_original = (min(original), sum(original))
    key_mirrored = (min(mirrored), sum(mirrored))
    return mirrored if key_mirrored < key_original else original


def generate_mono() -> pd.DataFrame:
    rows = []
    seen = set()
    for name, sub in SUBSTITUENTS_MONO.items():
        smiles_positions = {
            2: f"B(c1c({sub})cccc1)(O)O",
            3: f"B(c1cc({sub})ccc1)(O)O",
            4: f"B(c1ccc({sub})cc1)(O)O",
            5: f"B(c1cccc({sub})c1)(O)O",
            6: f"B(c1c({sub})cccc1)(O)O",
        }
        for pos, smi in smiles_positions.items():
            can = canonical_smiles(smi)
            if can is None or can in seen:
                continue
            seen.add(can)
            rows.append({"Substituent": name, "Position": pos, "SMILES": can})
    df = pd.DataFrame(rows).sort_values(["Substituent", "Position", "SMILES"]).reset_index(drop=True)
    df.insert(0, "Molecule", [f"mono_{i:04d}" for i in range(1, len(df) + 1)])
    return df


def generate_di() -> pd.DataFrame:
    # Deduplicate by canonical structure directly during generation.
    unique: dict[str, dict] = {}
    for (name1, sub1), (name2, sub2) in itertools.product(
        SUBSTITUENTS_DI_1.items(), SUBSTITUENTS_DI_2.items()
    ):
        for template, (pos1, pos2) in DI_TEMPLATES.items():
            smi = template.format(sub1=sub1, sub2=sub2)
            can = canonical_smiles(smi)
            if can is None or can in unique:
                continue
            p1, p2 = normalize_positions(pos1, pos2)
            unique[can] = {
                "SMILES": can,
                "Substituent 1": name1,
                "Substituent 2": name2,
                "Position 1": p1,
                "Position 2": p2,
            }
    df = pd.DataFrame(unique.values())
    df = df.sort_values(
        ["Substituent 1", "Substituent 2", "Position 1", "Position 2", "SMILES"]
    ).reset_index(drop=True)
    df.insert(0, "Molecule", [f"di_{i:04d}" for i in range(1, len(df) + 1)])
    return df


def draw_library(df: pd.DataFrame, out_dir: Path, prefix: str, per_image: int = 100) -> None:
    ensure_dir(out_dir)
    for start in range(0, len(df), per_image):
        part = df.iloc[start : start + per_image]
        mols = [Chem.MolFromSmiles(s) for s in part["SMILES"]]
        legends = part["Molecule"].astype(str).tolist()
        image = Draw.MolsToGridImage(
            mols, molsPerRow=10, subImgSize=(250, 250), legends=legends, useSVG=False
        )
        image.save(out_dir / f"{prefix}_{start // per_image + 1:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mono", "di", "both"], default="both")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "data" / "libraries",
        help="Directory for generated library tables.",
    )
    parser.add_argument("--draw", action="store_true", help="Also write structure-grid PNG files.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    if args.mode in {"mono", "both"}:
        mono = generate_mono()
        write_table(mono, out_dir / "mono_library.xlsx")
        write_table(mono, out_dir / "mono_library.csv")
        if args.draw:
            draw_library(mono, out_dir / "mono_structures", "mono")
        print(f"Mono library: {len(mono)} unique structures -> {out_dir / 'mono_library.xlsx'}")

    if args.mode in {"di", "both"}:
        di = generate_di()
        write_table(di, out_dir / "di_library.xlsx")
        write_table(di, out_dir / "di_library.csv")
        if args.draw:
            draw_library(di, out_dir / "di_structures", "di")
        print(f"Di library: {len(di)} unique structures -> {out_dir / 'di_library.xlsx'}")


if __name__ == "__main__":
    main()
