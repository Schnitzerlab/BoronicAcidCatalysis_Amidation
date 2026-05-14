#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:49:35 2025

@author: tobiasschnitzer

Geometry Optimization of Molecules Using ORCA (B3LYP/Def2-SVP)
Author: Tobias Schnitzer
Date: 2025-02-27

This script performs sequential geometry optimizations for a set of molecular structures (in .xyz format)
using the B3LYP functional with the Def2-SVP basis set via the ORCA quantum chemistry software.
Optimized geometries and output files are saved in a specified output directory.

"""

import os
import subprocess
import csv

# === CONFIGURATION ===
xyz_folder = "OptPM3_Geometry"
output_folder = "B3LYP_Optimized"
orca_path = "/Users/tobiasschnitzer/Library/orca_6_0_1/orca"
os.makedirs(output_folder, exist_ok=True)

# === HELPERS ===
def read_xyz(filepath):
    """Read atomic coordinates from an XYZ file."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    atoms = []
    for line in lines[2:]:
        tokens = line.strip().split()
        if len(tokens) == 4:
            atoms.append(tokens)
    return atoms

def write_orca_input(filename, atoms):
    """Write an ORCA input file for geometry optimization."""
    header = (
        "! B3LYP def2-SVP Opt TightSCF RIJCOSX\n\n"
        "%geom\n"
        "  MaxIter 500\n"
        "end\n\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        f.write("* xyz 0 1\n")
        for atom in atoms:
            f.write("  " + "  ".join(atom) + "\n")
        f.write("*\n")

def run_orca(inp_file, out_file):
    """Run an ORCA calculation."""
    try:
        result = subprocess.run(
            [orca_path, inp_file],
            stdout=open(out_file, "w"),
            stderr=subprocess.STDOUT
        )
        if result.returncode != 0:
            return False
        else:
            return True
    except Exception:
        return False

def check_orca_success(out_file):
    """Check if the ORCA job terminated normally."""
    try:
        with open(out_file, "r") as f:
            content = f.read()
        if "****ORCA TERMINATED NORMALLY****" in content:
            return "success"
        else:
            return "failed"
    except Exception:
        return "failed"

# === MAIN WORKFLOW ===

xyz_files = [f for f in os.listdir(xyz_folder) if f.endswith(".xyz")]
total_jobs = len(xyz_files)
if not xyz_files:
    print(f"No .xyz files found in {xyz_folder}!")
    exit()

print(f"🔹 Generating ORCA input files for {total_jobs} structures...")

# 1. Generate ORCA input files
for idx, xyz_file in enumerate(sorted(xyz_files), start=1):
    name = os.path.splitext(xyz_file)[0]
    inp_path = os.path.join(output_folder, f"{name}.inp")
    atoms = read_xyz(os.path.join(xyz_folder, xyz_file))
    write_orca_input(inp_path, atoms)
    print(f"  [{idx}/{total_jobs}] Input generated: {inp_path}")

print("\n✅ All ORCA input files have been generated successfully.\n")

# 2. Run ORCA calculations and check results
print("🔹 Starting ORCA geometry optimizations...")
job_status = []
for idx, xyz_file in enumerate(sorted(xyz_files), start=1):
    name = os.path.splitext(xyz_file)[0]
    inp_path = os.path.join(output_folder, f"{name}.inp")
    out_path = os.path.join(output_folder, f"{name}.out")
    print(f"  [{idx}/{total_jobs}] Running ORCA for {name}...", end=' ', flush=True)
    success = run_orca(inp_path, out_path)

    status = check_orca_success(out_path)
    job_status.append({'filename': name, 'status': status})
    if status == "success":
        print("✅ Success (terminated normally).")
    else:
        print("❌ Failed (not terminated normally).")

print("\n✅ All calculations finished.\n")

# 3. Create summary and CSV statistics
n_success = sum(1 for entry in job_status if entry['status'] == 'success')
n_failed = len(job_status) - n_success
failed_files = [entry['filename'] for entry in job_status if entry['status'] == 'failed']

with open("Geo_Opt_Overview.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "status"])
    for entry in job_status:
        writer.writerow([entry['filename'], entry['status']])

print(f"🔹 Summary:")
print(f"   Total jobs: {len(job_status)}")
print(f"   Success:    {n_success}")
print(f"   Failed:     {n_failed}")
if failed_files:
    print("   Failed files:")
    for f in failed_files:
        print(f"     - {f}")

print('\n📄 Job statistics have been saved to "Geo_Opt_Overview.csv".')
