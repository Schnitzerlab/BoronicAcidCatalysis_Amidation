#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 15:28:23 2025

@author: tobiasschnitzer

Parallel EA/IP/Fukui index calculation using B3LYP/def2-SVP single-point ORCA energies.
Author: Tobias Schnitzer
Date: 2025-07-25
"""

import os
import subprocess
import csv
from ase.io import read
from concurrent.futures import ProcessPoolExecutor, as_completed

# === SETTINGS ===
ORCA_PATH = "/Users/tobiasschnitzer/Library/orca_6_0_1/orca"
INPUT_FOLDER = "B3LYP_Optimized_XYZ"
OUTPUT_FOLDER = "B3LYP_EA_IP_FUKUI"
MAX_PARALLEL_JOBS = 4
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ORCA_INPUT_TEMPLATE = """! B3LYP def2-SVP TightSCF SlowConv
%maxcore 2000
* xyz {charge} {multiplicity}
{xyz_coords}
*
"""

def determine_multiplicity(atoms, charge):
    total_electrons = sum(atom.number for atom in atoms) - charge
    return 1 if total_electrons % 2 == 0 else 2

def write_orca_input(filename, atoms, charge):
    multiplicity = determine_multiplicity(atoms, charge)
    xyz_text = "\n".join(
        f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}"
        for atom in atoms
    )
    input_data = ORCA_INPUT_TEMPLATE.format(charge=charge, multiplicity=multiplicity, xyz_coords=xyz_text)
    with open(filename, "w") as f:
        f.write(input_data)

def run_orca(input_file):
    output_file = input_file.replace(".inp", ".out")
    command = f"{ORCA_PATH} {input_file} > {output_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        with open(output_file, "a") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        return False
    return True

def extract_energy(output_file):
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

def process_molecule(xyz_file):
    xyz_path = os.path.join(INPUT_FOLDER, xyz_file)
    mol_name = xyz_file.replace(".xyz", "")
    atoms = read(xyz_path)

    neutral_inp = os.path.join(OUTPUT_FOLDER, f"{mol_name}_neutral.inp")
    cation_inp  = os.path.join(OUTPUT_FOLDER, f"{mol_name}_cation.inp")
    anion_inp   = os.path.join(OUTPUT_FOLDER, f"{mol_name}_anion.inp")

    write_orca_input(neutral_inp, atoms, charge=0)
    write_orca_input(cation_inp, atoms, charge=1)
    write_orca_input(anion_inp, atoms, charge=-1)

    run_orca(neutral_inp)
    run_orca(cation_inp)
    run_orca(anion_inp)

    E_neutral = extract_energy(neutral_inp.replace(".inp", ".out"))
    E_cation  = extract_energy(cation_inp.replace(".inp", ".out"))
    E_anion   = extract_energy(anion_inp.replace(".inp", ".out"))

    if None in [E_neutral, E_cation, E_anion]:
        return {
            "Molecule": mol_name,
            "E_neutral (Eh)": E_neutral,
            "E_cation (Eh)": E_cation,
            "E_anion (Eh)": E_anion,
            "Ionization Energy (eV)": None,
            "Electron Affinity (eV)": None,
            "Fukui+": None,
            "Fukui-": None
        }

    IP = (E_cation - E_neutral) * 27.2114
    EA = (E_neutral - E_anion) * 27.2114
    Fukui_plus  = E_cation - E_neutral
    Fukui_minus = E_neutral - E_anion

    return {
        "Molecule": mol_name,
        "E_neutral (Eh)": E_neutral,
        "E_cation (Eh)": E_cation,
        "E_anion (Eh)": E_anion,
        "Ionization Energy (eV)": IP,
        "Electron Affinity (eV)": EA,
        "Fukui+": Fukui_plus,
        "Fukui-": Fukui_minus
    }

# === MAIN BLOCK ===
if __name__ == "__main__":
    import time
    start = time.time()

    xyz_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xyz")],
        key=lambda x: extract_number(x.replace(".xyz", ""))
    )

    print(f"🔹 Starting EA/IP/Fukui calculations for {len(xyz_files)} molecules using up to {MAX_PARALLEL_JOBS} workers...\n")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        futures = {executor.submit(process_molecule, xyz): xyz for xyz in xyz_files}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"  [{i}/{len(xyz_files)}] {result['Molecule']} ✓")

    if results:
        results.sort(key=lambda x: extract_number(x["Molecule"]))
        output_csv = "EA_IP_Fukui_summary.csv"
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ All calculations finished. Results saved to: {output_csv}")
    else:
        print("⚠️ No successful calculations.")

    print(f"⏱️  Elapsed time: {time.time() - start:.1f} seconds")
