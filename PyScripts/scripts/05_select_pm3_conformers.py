#!/usr/bin/env python3
"""Select the lowest-energy optimized PM3 conformer for each molecule.

For each successfully converged PM3 geometry optimization, the final
ORCA single-point energy is extracted. The lowest-energy conformer of
each molecule is selected for subsequent DFT calculations.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import pandas as pd

from common import (
    HARTREE_TO_KJMOL,
    dataset_dir,
    ensure_dir,
    extract_last_orca_geometry,
    last_single_point_energy,
    optimization_done,
    orca_terminated_normally,
    write_table,
    write_xyz,
)

CONF_RE = re.compile(r"^(mol_\d+)__conf_(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    base = dataset_dir(args.dataset)
    pm3_dir = base / "02_pm3"
    selected_dir = ensure_dir(base / "03_pm3_selected")

    groups = defaultdict(list)
    for out in sorted(pm3_dir.glob("*.out")):
        m = CONF_RE.match(out.stem)
        if m:
            groups[m.group(1)].append((int(m.group(2)), out))

    all_rows = []
    selected_rows = []
    for mol_id, jobs in sorted(groups.items()):
        valid = []
        for conf_num, out in jobs:
            normal = orca_terminated_normally(out)
            opt_done = optimization_done(out)
            energy = last_single_point_energy(out) if normal and opt_done else None
            row = {
                "MoleculeID": mol_id,
                "Conformer": conf_num,
                "Output": out.name,
                "ORCA_normal_termination": normal,
                "Optimization_done": opt_done,
                "Energy_Eh": energy,
            }
            all_rows.append(row)
            if energy is not None:
                valid.append((energy, conf_num, out))

        if not valid:
            selected_rows.append({"MoleculeID": mol_id, "status": "no_valid_pm3_conformer"})
            continue

        valid.sort(key=lambda x: x[0])
        best_energy, best_conf, best_out = valid[0]
        geometry = extract_last_orca_geometry(best_out)
        if geometry is None:
            sidecar = pm3_dir / f"{best_out.stem}.xyz"
            if sidecar.exists():
                # Copy exact sidecar text if ORCA did not print a parseable geometry.
                (selected_dir / f"{mol_id}.xyz").write_text(
                    sidecar.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
                )
            else:
                selected_rows.append(
                    {
                        "MoleculeID": mol_id,
                        "status": "selected_but_geometry_missing",
                        "BestConformer": best_conf,
                        "Energy_Eh": best_energy,
                    }
                )
                continue
        else:
            write_xyz(
                geometry,
                selected_dir / f"{mol_id}.xyz",
                comment=f"Lowest-energy PM3 conformer {best_conf}; E={best_energy:.12f} Eh",
            )

        selected_rows.append(
            {
                "MoleculeID": mol_id,
                "status": "selected",
                "BestConformer": best_conf,
                "Energy_Eh": best_energy,
                "XYZ": f"{mol_id}.xyz",
            }
        )

        # Relative energies for all valid conformers of this molecule.
        for row in all_rows:
            if row["MoleculeID"] == mol_id and row.get("Energy_Eh") is not None:
                row["RelativeEnergy_kJmol"] = (row["Energy_Eh"] - best_energy) * HARTREE_TO_KJMOL
                row["Selected"] = row["Conformer"] == best_conf

    write_table(pd.DataFrame(all_rows), selected_dir / "pm3_conformer_energies.csv")
    write_table(pd.DataFrame(selected_rows), selected_dir / "selected_conformers.csv")
    print(f"Selected PM3 geometries -> {selected_dir}")


if __name__ == "__main__":
    main()
