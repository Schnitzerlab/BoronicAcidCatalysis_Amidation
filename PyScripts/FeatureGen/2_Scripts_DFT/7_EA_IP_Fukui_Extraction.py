#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 20:57:06 2025

@author: tobiasschnitzer
Created on Wed Jul 24 2025
Publication-ready script for automated calculation of EA, IP, and Fukui indices
using B3LYP/def2-SVP single-point energies with ORCA, based on provided xyz geometries.

"""

import os
import subprocess
import csv
from ase.io import read

# === SETTINGS ===
ORCA_PATH = "/Users/tobiasschnitzer/Library/orca_6_0_1/orca"  # Adjust if necessary
INPUT_FOLDER = "B3LYP_Optimized_XYZ"
OUTPUT_FOLDER = "B3LYP_EA_IP_FUKUI"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ORCA_INPUT_TEMPLATE = """! B3LYP def2-SVP TightSCF SlowConv
%maxcore 2000
* xyz {charge} {multiplicity}
{xyz_coords}
*
"""

def determine_multiplicity(atoms, charge):
    """Determine multiplicity based on total electron count."""
    total_electrons = sum(atom.number for atom in atoms) - charge
    return 1 if total_electrons % 2 == 0 else 2

def write_orca_input(filename, atoms, charge):
    """Write the ORCA input file with proper multiplicity."""
    multiplicity = determine_multiplicity(atoms, charge)
    xyz_text = "\n".join(
        f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}"
        for atom in atoms
    )
    input_data = ORCA_INPUT_TEMPLATE.format(charge=charge, multiplicity=multiplicity, xyz_coords=xyz_text)
    with open(filename, "w") as f:
        f.write(input_data)

def run_orca(input_file):
    """Run ORCA calculation, capturing output."""
    output_file = input_file.replace(".inp", ".out")
    command = f"{ORCA_PATH} {input_file} > {output_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ORCA crashed for {input_file}. Please check the output!")
        with open(output_file, "a") as f:
            f.write(result.stdout)
            f.write(result.stderr)

def extract_energy(output_file):
    """Extract final single point energy from ORCA output, check normal termination."""
    if not os.path.exists(output_file):
        print(f"⚠️ Output file not found: {output_file}")
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
    except Exception as e:
        print(f"⚠️ Error while reading {output_file}: {e}")
    
    if not converged:
        print(f"⚠️ WARNING: ORCA did not terminate normally for {output_file}!")
        return None
    return energy

def extract_number(mol_name):
    # Tries to sort by final number in mol_x (e.g. mol_1, mol_2, ...)
    parts = mol_name.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else float('inf')

# Main loop
results = []
xyz_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xyz")], key=lambda x: extract_number(x.replace(".xyz", "")))

print(f"🔹 Starting EA/IP/Fukui calculations for {len(xyz_files)} molecules...")

for xyz_file in xyz_files:
    xyz_path = os.path.join(INPUT_FOLDER, xyz_file)
    mol_name = xyz_file.replace(".xyz", "")
    atoms = read(xyz_path)

    neutral_inp = os.path.join(OUTPUT_FOLDER, f"{mol_name}_neutral.inp")
    cation_inp = os.path.join(OUTPUT_FOLDER, f"{mol_name}_cation.inp")
    anion_inp = os.path.join(OUTPUT_FOLDER, f"{mol_name}_anion.inp")

    write_orca_input(neutral_inp, atoms, charge=0)
    write_orca_input(cation_inp, atoms, charge=1)
    write_orca_input(anion_inp, atoms, charge=-1)

    print(f"🚀 Calculating for {mol_name} ...")
    run_orca(neutral_inp)
    run_orca(cation_inp)
    run_orca(anion_inp)

    E_neutral = extract_energy(neutral_inp.replace(".inp", ".out"))
    E_cation = extract_energy(cation_inp.replace(".inp", ".out"))
    E_anion = extract_energy(anion_inp.replace(".inp", ".out"))

    if E_neutral is not None and E_cation is not None and E_anion is not None:
        IP = (E_cation - E_neutral) * 27.2114  # Eh to eV
        EA = (E_neutral - E_anion) * 27.2114
        Fukui_plus = E_cation - E_neutral
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

# Output sorted results as CSV in working directory
if results:
    results.sort(key=lambda x: extract_number(x["Molecule"]))
    output_csv = "EA_IP_Fukui_summary.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Calculations finished. Results saved to {output_csv}")
else:
    print("⚠️ No successful calculations.")

