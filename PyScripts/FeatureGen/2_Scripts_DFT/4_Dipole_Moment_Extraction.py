#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 19:47:27 2025

@author: tobiasschnitzer

Extraction of Dipole Magnitudes from ORCA Output Files (B3LYP/Def2-SVP)
Author: Tobias Schnitzer
Date: 2025-07-22

This script extracts only the dipole magnitude (Debye) from ORCA output files
in the B3LYP_Optimized folder and writes the results to a CSV file.

Input:
    - ORCA output files (.out) in the B3LYP_Optimized folder
Output:
    - CSV file (Dipole_Moment_B3LYP_def2SVP.csv) with dipole magnitudes

Usage:
    - Place all .out files in B3LYP_Optimized.
    - Run: python3 extract_b3lyp_dipole_magnitude.py
"""

import os
import re
import pandas as pd

# Folder and output file definitions
output_folder = "B3LYP_Optimized"
csv_file = "Dipole_Moment_summary.csv"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

results = []

# Gather and sort .out files
out_files = sorted(
    [f for f in os.listdir(output_folder) if f.endswith(".out")],
    key=lambda x: (int(re.search(r"mol_(\d+)", x).group(1)) if re.search(r"mol_(\d+)", x) else float('inf'))
)

print(f"🚀 Found {len(out_files)} B3LYP/Def2-SVP output files. Starting extraction...\n")

for idx, file in enumerate(out_files, start=1):
    mol_name = file.replace(".out", "")
    orca_output_path = os.path.join(output_folder, file)
    print(f"[{idx}/{len(out_files)}] Extracting dipole magnitude for {mol_name} ...", end=" ")

    with open(orca_output_path, "r") as out_file:
        lines = out_file.readlines()
        success = any("ORCA TERMINATED NORMALLY" in line for line in lines)

    dipole_magnitude = None

    if success:
        for line in lines:
            if "Magnitude (Debye)" in line:
                match = re.search(r"Magnitude \(Debye\)\s+:\s+([-\d\.E]+)", line)
                if match:
                    dipole_magnitude = float(match.group(1))
                    break

        if dipole_magnitude is not None:
            print("✅")
            results.append({
                "Molecule": mol_name,
                "Dipole_Magnitude_Debye": dipole_magnitude
            })
        else:
            print("❌ Dipole magnitude not found.")
    else:
        print("❌ ORCA did not terminate normally.")

# Save results to CSV
if results:
    df = pd.DataFrame(results)
    # Optional: sort by molecule number if present
    try:
        df["SortKey"] = df["Molecule"].str.extract(r"(\d+)").astype(float)
        df = df.sort_values(by="SortKey")
        df = df.drop(columns=["SortKey"])
    except Exception:
        pass
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Results saved to: {os.path.abspath(csv_file)}")
    print(f"Extracted dipole magnitudes for {len(results)} molecules.")
else:
    print("\n⚠️ No dipole magnitudes extracted! Please check your ORCA output files.")
