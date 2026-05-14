#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 20:09:12 2025

@author: tobiasschnitzer

Post-processing script for EA, IP, and Fukui index extraction from
existing ORCA output files in B3LYP_EA_IP_FUKUI.
Author: Tobias Schnitzer
Date: 2025-07-25
"""

import os
import csv

# === SETTINGS ===
OUTPUT_FOLDER = "B3LYP_EA_IP_FUKUI"

def extract_energy(output_file):
    """Extract final single point energy from ORCA output, check normal termination."""
    if not os.path.exists(output_file):
        return None

    converged = False
    energy = None
    try:
        with open(output_file, "r") as f:
            for line in f:
                if "FINAL SINGLE POINT ENERGY" in line:
                    energy = float(line.split()[-1])
                if "ORCA TERMINATED NORMALLY" in line:
                    converged = True
    except Exception:
        return None
    return energy if converged else None

def extract_number(mol_name):
    parts = mol_name.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else float('inf')

# === MAIN LOOP ===

# Identify molecule names by looking for *_neutral.out files
out_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith("_neutral.out")]
mol_names = sorted([f.replace("_neutral.out", "") for f in out_files], key=extract_number)

results = []
print(f"🔍 Extracting data from {len(mol_names)} molecules...")

for mol_name in mol_names:
    neutral_out = os.path.join(OUTPUT_FOLDER, f"{mol_name}_neutral.out")
    cation_out  = os.path.join(OUTPUT_FOLDER, f"{mol_name}_cation.out")
    anion_out   = os.path.join(OUTPUT_FOLDER, f"{mol_name}_anion.out")

    E_neutral = extract_energy(neutral_out)
    E_cation  = extract_energy(cation_out)
    E_anion   = extract_energy(anion_out)

    if None not in [E_neutral, E_cation, E_anion]:
        IP = (E_cation - E_neutral) * 27.2114
        EA = (E_neutral - E_anion) * 27.2114
        Fukui_plus  = E_cation - E_neutral
        Fukui_minus = E_neutral - E_anion
    else:
        IP = EA = Fukui_plus = Fukui_minus = None

    results.append({
        "Molecule": mol_name,
        "E_neutral (Eh)": E_neutral,
        "E_cation (Eh)": E_cation,
        "E_anion (Eh)": E_anion,
        "Ionization Energy (eV)": IP,
        "Electron Affinity (eV)": EA,
        "Fukui+": Fukui_plus,
        "Fukui-": Fukui_minus
    })

# === CSV OUTPUT ===
if results:
    output_csv = "EA_IP_Fukui_summary.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Data extraction complete. Results saved to: {output_csv}")
else:
    print("⚠️ No successful extractions. Please check the output files.")
