#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 20:15:05 2025

@author: tobiasschnitzer

Extract Mulliken, Loewdin, and Mayer charge data for boron atoms
from all ORCA .out files in B3LYP_Optimized
"""

import os
import re
import pandas as pd

# --- Settings ---
OUTPUT_FOLDER = "B3LYP_Optimized"       # Adjust to your calculation folder
CSV_FILE = "Mulliken_Loewdin_Mayer_summary.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results = []

def extract_boron_data(orca_output_path, mol_name):
    """Extract Mulliken, Loewdin, Mayer charge/bond order data for boron from an ORCA .out file."""
    print(f"🚀 Extracting data for {mol_name}...")

    with open(orca_output_path, "r") as out_file:
        lines = out_file.readlines()

    # Check for normal termination
    if not any("ORCA TERMINATED NORMALLY" in line for line in lines):
        print(f"❌ ORCA did not terminate normally for {mol_name}, skipping...")
        return None

    boron_mulliken = boron_loewdin = None
    mayer_data = {"QA": None, "VA": None, "BVA": None, "FA": None}
    boron_atom_number = None

    # Find B atom number in atom list (first occurrence)
    for i, line in enumerate(lines):
        if re.search(r"\s*\d+\s+B\s+", line):
            match = re.match(r"\s*(\d+)\s+B", line)
            if match:
                boron_atom_number = match.group(1)
                print(f"  ➜ Found boron atom number: {boron_atom_number}")
                break

    if boron_atom_number is None:
        print(f"⚠️ No B-atom found in {mol_name}, skipping...")
        return None

    # Extract Mulliken, Loewdin, Mayer values
    for i, line in enumerate(lines):
        if "MULLIKEN ATOMIC CHARGES" in line:
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    break
                match = re.search(rf"\s*{boron_atom_number}\s+B\s+:\s+([-\d\.E]+)", lines[j])
                if match:
                    boron_mulliken = float(match.group(1))
                    break

        if "LOEWDIN ATOMIC CHARGES" in line:
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    break
                match = re.search(rf"\s*{boron_atom_number}\s+B\s+:\s+([-\d\.E]+)", lines[j])
                if match:
                    boron_loewdin = float(match.group(1))
                    break

        if "ATOM       NA         ZA         QA" in line:
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    break
                match = re.search(
                    rf"\s*{boron_atom_number}\s+B\s+[\d\.\-E]+\s+[\d\.\-E]+\s+([-\d\.E]+)\s+([-\d\.E]+)\s+([-\d\.E]+)\s+([-\d\.E]+)",
                    lines[j])
                if match:
                    mayer_data = {
                        "QA": float(match.group(1)),
                        "VA": float(match.group(2)),
                        "BVA": float(match.group(3)),
                        "FA": float(match.group(4))
                    }
                    break

    print(f"✅ Extraction successful for {mol_name}!")
    return {
        "Molecule": mol_name,
        "Mulliken_B_Charge": boron_mulliken,
        "Loewdin_B_Charge": boron_loewdin,
        "Mayer_QA": mayer_data["QA"],
        "Mayer_VA": mayer_data["VA"],
        "Mayer_BVA": mayer_data["BVA"],
        "Mayer_FA": mayer_data["FA"]
    }

# --- Process all .out files ---
out_files = sorted(
    [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".out")],
    key=lambda x: (
        int(re.search(r"mol_(\d+)", x).group(1)) if re.search(r"mol_(\d+)", x) else float('inf'),
        "anion" in x, "neutral" in x, "cation" in x
    )
)

total_files = len(out_files)
print(f"🔹 Extracting boron charge data from {total_files} ORCA output files in '{OUTPUT_FOLDER}'...")

for index, file in enumerate(out_files, start=1):
    mol_name = file.replace(".out", "")
    orca_output_path = os.path.join(OUTPUT_FOLDER, file)
    print(f"🔄 Processing {index}/{total_files}: {mol_name}")
    result = extract_boron_data(orca_output_path, mol_name)
    if result:
        results.append(result)

# --- Save as CSV ---
if results:
    df = pd.DataFrame(results)
    df.sort_values(by=["Molecule"], key=lambda x: x.str.extract(r"(\d+)").astype(float).fillna(0).astype(int).values[:, 0], inplace=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Results saved in: {os.path.abspath(CSV_FILE)}")
    print(f"📊 Extracted boron charge data from {len(results)} molecules")
else:
    print("⚠️ No Mulliken, Loewdin, or Mayer data for boron found! Check ORCA output files.")
