#!/usr/bin/env python3
"""Optimize selected PM3 geometries at B3LYP/def2-SVP with ORCA 6."""

from __future__ import annotations

import argparse

import pandas as pd

from common import (
    dataset_dir,
    ensure_dir,
    extract_last_orca_geometry,
    locate_orca,
    orca_terminated_normally,
    read_xyz,
    run_orca_job,
    write_table,
    write_xyz,
    xyz_block,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--orca", help="Path to ORCA executable. Otherwise ORCA_EXE/PATH is used.")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--maxcore", type=int, default=2000, help="ORCA maxcore in MB per process.")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    base = dataset_dir(args.dataset)
    input_dir = base / "03_pm3_selected"
    out_dir = ensure_dir(base / "04_dft_geometry")
    xyz_dir = ensure_dir(base / "05_dft_xyz")
    orca = locate_orca(args.orca)

    xyz_files = sorted(input_dir.glob("mol_*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No selected PM3 geometries found in {input_dir}")

    rows = []
    for i, xyz_path in enumerate(xyz_files, start=1):
        mol_id = xyz_path.stem
        inp = out_dir / f"{mol_id}.inp"
        out = out_dir / f"{mol_id}.out"

        header = "! B3LYP def2-SVP Opt TightSCF RIJCOSX\n"
        header += f"%maxcore {args.maxcore}\n"
        if args.nprocs > 1:
            header += f"%pal nprocs {args.nprocs} end\n"
        header += "%geom\n  MaxIter 500\nend\n"

        atoms = read_xyz(xyz_path)
        inp.write_text(
            header
            + f"* xyz {args.charge} {args.multiplicity}\n"
            + xyz_block(atoms)
            + "\n*\n",
            encoding="utf-8",
        )

        if out.exists() and not args.rerun and orca_terminated_normally(out):
            rc = 0
            status = "existing_success"
        else:
            print(f"[{i}/{len(xyz_files)}] DFT optimization: {mol_id}")
            rc = run_orca_job(orca_exe=orca, input_path=inp, output_path=out, cwd=out_dir)
            status = "success" if out.exists() and orca_terminated_normally(out) else "failed"

        geometry_written = False
        if out.exists() and orca_terminated_normally(out):
            geometry = extract_last_orca_geometry(out)
            if geometry is not None:
                write_xyz(geometry, xyz_dir / f"{mol_id}.xyz", comment=f"Final B3LYP/def2-SVP geometry; {mol_id}")
                geometry_written = True
            else:
                sidecar = out_dir / f"{mol_id}.xyz"
                if sidecar.exists():
                    (xyz_dir / f"{mol_id}.xyz").write_text(
                        sidecar.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
                    )
                    geometry_written = True

        rows.append(
            {
                "MoleculeID": mol_id,
                "status": status,
                "returncode": rc,
                "FinalXYZ_written": geometry_written,
            }
        )

    write_table(pd.DataFrame(rows), out_dir / "dft_geometry_job_status.csv")
    print(f"DFT outputs -> {out_dir}")
    print(f"Final DFT geometries -> {xyz_dir}")


if __name__ == "__main__":
    main()
