#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 19:59:02 2025

@author: tobiasschnitzer

xtracts HOMO and LUMO energies (in eV) and the HOMO-LUMO gap from an ORCA output file.


"""

import os
import re
import pandas as pd

def extract_homo_lumo(file_path):
    
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
    except Exception as e:
        return None, None, None, False

    homo_energy = None
    lumo_energy = None
    orbital_energies = []
    in_orbital_block = False

    for line in lines:
        if "ORBITAL ENERGIES" in line:
            in_orbital_block = True
            continue
        if in_orbital_block and "*Only the first 10 virtual orbitals were printed." in line:
            break
        if in_orbital_block:
            parts = line.split()
            if len(parts) == 4:
                try:
                    occ = float(parts[1])
                    energy_eV = float(parts[3])
                    orbital_energies.append((occ, energy_eV))
                except ValueError:
                    continue

    for occ, energy_eV in orbital_energies:
        if occ == 2.0000:
            homo_energy = energy_eV
        elif occ == 0.0000 and lumo_energy is None:
            lumo_energy = energy_eV

    homo_lumo_gap = None
    if homo_energy is not None and lumo_energy is not None:
        homo_lumo_gap = lumo_energy - homo_energy
        return homo_energy, lumo_energy, homo_lumo_gap, True
    else:
        return homo_energy, lumo_energy, homo_lumo_gap, False

def process_directory(directory):
    """
    Scans a directory for ORCA output files and extracts HOMO/LUMO values for each molecule.
    Prints progress and extraction status live.

    Args:
        directory (str): Directory containing ORCA .out files.
    """
    data = []
    out_files = [f for f in os.listdir(directory) if f.endswith(".out")]
    total = len(out_files)
    print(f"Found {total} .out files in '{directory}'.")

    for idx, filename in enumerate(sorted(out_files), 1):
        file_path = os.path.join(directory, filename)
        homo, lumo, gap, success = extract_homo_lumo(file_path)
        mol_number = int(re.findall(r'\d+', filename)[0]) if re.search(r'\d+', filename) else float('inf')
        status_str = "OK" if success else "FAILED"
        print(f"[{idx}/{total}] {filename}: Extraction {status_str}")

        data.append({
            "File": filename,
            "Molecule Number": mol_number,
            "HOMO Energy (eV)": homo,
            "LUMO Energy (eV)": lumo,
            "HOMO-LUMO Gap (eV)": gap,
            "Extraction Success": status_str
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by=["Molecule Number"])
    df.drop(columns=["Molecule Number"], inplace=True)
    
    output_csv = os.path.join(os.getcwd(), "HOMO_LUMO_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to: {output_csv}")

if __name__ == "__main__":
    directory = "B3LYP_Optimized"  # set to the folder with your ORCA output files
    process_directory(directory)
