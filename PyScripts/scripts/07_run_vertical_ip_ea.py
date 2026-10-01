#!/usr/bin/env python3
"""Calculate vertical ionization energy (IP) and electron affinity (EA).

Neutral, cation, and anion single-point energies are calculated at the fixed
B3LYP/def2-SVP optimized geometry. Legacy columns labelled as Fukui indices are
intentionally not calculated or written because they were redundant with the
energy differences and are not used in the revised analysis.
"""

from __future__ import annotations

import argparse

import pandas as pd
from rdkit import Chem

from common import (
    HARTREE_TO_EV,
    dataset_dir,
    ensure_dir,
    last_single_point_energy,
    locate_orca,
    orca_terminated_normally,
    read_xyz,
    run_orca_job,
    write_table,
    xyz_block,
)


def multiplicity(atoms, charge: int) -> int:
    pt = Chem.GetPeriodicTable()
    electrons = sum(pt.GetAtomicNumber(symbol) for symbol, *_ in atoms) - charge
    return 1 if electrons % 2 == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--orca", help="Path to ORCA executable. Otherwise ORCA_EXE/PATH is used.")
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--maxcore", type=int, default=2000)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    base = dataset_dir(args.dataset)
    xyz_dir = base / "05_dft_xyz"
    out_dir = ensure_dir(base / "06_vertical_ip_ea")
    orca = locate_orca(args.orca)

    xyz_files = sorted(xyz_dir.glob("mol_*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No DFT geometries found in {xyz_dir}")

    rows = []
    for idx, xyz_path in enumerate(xyz_files, start=1):
        mol_id = xyz_path.stem
        atoms = read_xyz(xyz_path)
        energies = {}
        statuses = {}

        for label, charge in [("neutral", 0), ("cation", 1), ("anion", -1)]:
            mult = multiplicity(atoms, charge)
            inp = out_dir / f"{mol_id}__{label}.inp"
            out = out_dir / f"{mol_id}__{label}.out"
            header = "! B3LYP def2-SVP TightSCF SlowConv\n"
            header += f"%maxcore {args.maxcore}\n"
            if args.nprocs > 1:
                header += f"%pal nprocs {args.nprocs} end\n"
            inp.write_text(
                header + f"* xyz {charge} {mult}\n" + xyz_block(atoms) + "\n*\n",
                encoding="utf-8",
            )

            if out.exists() and not args.rerun and orca_terminated_normally(out):
                status = "existing_success"
            else:
                print(f"[{idx}/{len(xyz_files)}] {mol_id} {label}")
                run_orca_job(orca_exe=orca, input_path=inp, output_path=out, cwd=out_dir)
                status = "success" if out.exists() and orca_terminated_normally(out) else "failed"

            statuses[label] = status
            energies[label] = last_single_point_energy(out) if out.exists() and orca_terminated_normally(out) else None

        e0, ep, em = energies["neutral"], energies["cation"], energies["anion"]
        ip = (ep - e0) * HARTREE_TO_EV if None not in (ep, e0) else None
        ea = (e0 - em) * HARTREE_TO_EV if None not in (e0, em) else None
        rows.append(
            {
                "MoleculeID": mol_id,
                "E_neutral (Eh)": e0,
                "E_cation (Eh)": ep,
                "E_anion (Eh)": em,
                "Ionization Energy (eV)": ip,
                "Electron Affinity (eV)": ea,
                "neutral_status": statuses["neutral"],
                "cation_status": statuses["cation"],
                "anion_status": statuses["anion"],
            }
        )

    write_table(pd.DataFrame(rows), out_dir / "vertical_ip_ea_summary.csv")
    print(f"Vertical IP/EA results -> {out_dir / 'vertical_ip_ea_summary.csv'}")


if __name__ == "__main__":
    main()
