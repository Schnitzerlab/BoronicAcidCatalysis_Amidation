#!/usr/bin/env python3
"""Generate up to N ETKDGv3 conformers and attempt MMFF94 relaxation.

If MMFF94 parameters are unavailable (as is common for boron-containing
structures in RDKit), the embedded ETKDG geometry is retained and passed to
the subsequent PM3 optimization. This behavior is written to the QC table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock
from tqdm import tqdm

from common import dataset_dir, ensure_dir, load_manifest, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-confs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune-rms", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=0, help="0 lets RDKit use all available threads.")
    parser.add_argument("--max-iters", type=int, default=10000)
    args = parser.parse_args()

    manifest = load_manifest(args.dataset)
    out_dir = ensure_dir(dataset_dir(args.dataset) / "01_conformers")
    rows = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Conformers"):
        mol_id = str(row["MoleculeID"])
        smi = str(row["SMILES"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"MoleculeID": mol_id, "status": "invalid_smiles", "n_embedded": 0})
            continue
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.numThreads = args.threads
        params.pruneRmsThresh = args.prune_rms
        params.randomSeed = args.seed
        params.useSmallRingTorsions = True

        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=args.num_confs, params=params))
        if not conf_ids:
            rows.append({"MoleculeID": mol_id, "status": "embedding_failed", "n_embedded": 0})
            continue

        mmff_available = bool(AllChem.MMFFHasAllMoleculeParams(mol))
        n_converged = 0
        for conf_number, conf_id in enumerate(conf_ids, start=1):
            mmff_status = "parameters_unavailable"
            if mmff_available:
                try:
                    status_code = AllChem.MMFFOptimizeMolecule(
                        mol, confId=conf_id, maxIters=args.max_iters, mmffVariant="MMFF94"
                    )
                    if status_code == 0:
                        mmff_status = "converged"
                        n_converged += 1
                    elif status_code == 1:
                        mmff_status = "max_iterations_reached"
                    else:
                        mmff_status = f"status_{status_code}"
                except Exception as exc:
                    mmff_status = f"error: {exc}"

            # Every successfully embedded conformer is written and subsequently
            # optimized at the PM3 level, even if MMFF94 parameters are unavailable
            # for the boron-containing molecule.
            xyz_path = out_dir / f"{mol_id}__conf_{conf_number:03d}.xyz"
            xyz_path.write_text(MolToXYZBlock(mol, confId=conf_id), encoding="utf-8")
            rows.append(
                {
                    "MoleculeID": mol_id,
                    "Conformer": conf_number,
                    "RDKitConfID": int(conf_id),
                    "MMFF_available": mmff_available,
                    "MMFF_status": mmff_status,
                    "XYZ": xyz_path.name,
                    "status": "ok",
                    "n_embedded": len(conf_ids),
                }
            )
        print(
            f"{mol_id}: embedded {len(conf_ids)} / requested {args.num_confs}; "
            f"MMFF available={mmff_available}, converged={n_converged}"
        )

    write_table(pd.DataFrame(rows), out_dir / "conformer_generation.csv")
    print(f"Conformer files -> {out_dir}")


if __name__ == "__main__":
    main()
