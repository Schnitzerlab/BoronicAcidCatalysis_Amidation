#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:28:12 2025

Identify the lowest-energy PM3 conformer per molecule and save the optimized geometry.

- Extracts energies from .out files
- Finds lowest-energy conformer for each mol_X
- Copies its corresponding .xyz file to 'OptPM3_Geometry' as mol_X.xyz

Author: Tobias Schnitzer
"""

import os
import re
import shutil
from collections import defaultdict
from tqdm import tqdm

# === Config ===
input_dir = "PM3_Optimized"
output_dir = "OptPM3_Geometry"
energy_pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
end_marker = "*** OPTIMIZATION RUN DONE ***"

os.makedirs(output_dir, exist_ok=True)

# === Step 1: Group all .out files by molecule ===
mol_groups = defaultdict(list)
for file in os.listdir(input_dir):
    if file.endswith(".out") and file.startswith("mol_"):
        base = file.replace(".out", "")
        mol_id = "_".join(base.split("_")[:2])  # e.g., "mol_1"
        mol_groups[mol_id].append(base)

# === Step 2: Extract energies and select lowest-energy conformer ===
summary_log = []

for mol_id, conf_bases in tqdm(mol_groups.items(), desc="Evaluating conformers"):
    energies = {}
    for base in conf_bases:
        out_path = os.path.join(input_dir, base + ".out")
        try:
            with open(out_path, "r") as f:
                content = f.read()
                if end_marker not in content:
                    continue
                match = energy_pattern.search(content)
                if match:
                    energies[base] = float(match.group(1))
        except Exception as e:
            print(f"❌ Error reading {out_path}: {e}")

    if not energies:
        print(f"⚠️ No valid energies found for {mol_id}")
        continue

    # Identify lowest-energy conformer
    best_base = min(energies, key=energies.get)
    best_energy = energies[best_base]

    # Copy .xyz file
    xyz_src = os.path.join(input_dir, best_base + ".xyz")
    xyz_dst = os.path.join(output_dir, mol_id + ".xyz")
    try:
        shutil.copyfile(xyz_src, xyz_dst)
        summary_log.append((mol_id, best_base, best_energy))
    except Exception as e:
        print(f"❌ Failed to copy {xyz_src} to {xyz_dst}: {e}")

# === Step 3: Log summary ===
with open(os.path.join(output_dir, "best_conformers.csv"), "w") as f:
    f.write("Molecule,BestConformer,Energy\n")
    for mol_id, base, energy in summary_log:
        f.write(f"{mol_id},{base},{energy:.10f}\n")

print(f"✅ Done. Best conformers stored in '{output_dir}'")
