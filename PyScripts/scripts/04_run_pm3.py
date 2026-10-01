#!/usr/bin/env python3
"""Run ORCA PM3 geometry optimizations for all generated conformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    dataset_dir,
    ensure_dir,
    locate_orca,
    orca_terminated_normally,
    read_xyz,
    run_orca_job,
    write_table,
    xyz_block,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--orca", help="Path to ORCA executable. Otherwise ORCA_EXE/PATH is used.")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--rerun", action="store_true", help="Re-run even normally terminated existing jobs.")
    args = parser.parse_args()

    base = dataset_dir(args.dataset)
    input_dir = base / "01_conformers"
    out_dir = ensure_dir(base / "02_pm3")
    orca = locate_orca(args.orca)

    xyz_files = sorted(input_dir.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No conformer XYZ files found in {input_dir}")

    status_rows = []
    for i, xyz_path in enumerate(xyz_files, start=1):
        stem = xyz_path.stem
        inp = out_dir / f"{stem}.inp"
        out = out_dir / f"{stem}.out"

        if out.exists() and not args.rerun and orca_terminated_normally(out):
            status_rows.append({"Job": stem, "status": "existing_success", "returncode": 0})
            print(f"[{i}/{len(xyz_files)}] {stem}: existing successful output, skipped")
            continue

        atoms = read_xyz(xyz_path)
        inp.write_text(
            "! PM3 Opt\n"
            f"* xyz {args.charge} {args.multiplicity}\n"
            f"{xyz_block(atoms)}\n"
            "*\n",
            encoding="utf-8",
        )
        print(f"[{i}/{len(xyz_files)}] PM3: {stem}")
        rc = run_orca_job(orca_exe=orca, input_path=inp, output_path=out, cwd=out_dir)
        normal = out.exists() and orca_terminated_normally(out)
        status_rows.append(
            {"Job": stem, "status": "success" if normal else "failed", "returncode": rc}
        )

    write_table(pd.DataFrame(status_rows), out_dir / "pm3_job_status.csv")
    print(f"PM3 outputs -> {out_dir}")


if __name__ == "__main__":
    main()
