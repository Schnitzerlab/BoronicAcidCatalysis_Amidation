#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 20:31:58 2025

@author: tobiasschnitzer
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Geometry Optimization of Molecules Using ORCA (B3LYP/Def2-SVP)
Author: Tobias Schnitzer
Date: 2025-07-25

This script performs geometry optimizations for all .xyz files in a folder using ORCA
and runs up to N jobs in parallel (1 core per job). It is optimized for macOS on Apple Silicon.
"""

import os
import subprocess
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# === CONFIGURATION ===
xyz_folder = "OptPM3_Geometry"
output_folder = "B3LYP_Optimized"
orca_path = "/Users/tobiasschnitzer/Library/orca_6_0_1/orca"
max_parallel_jobs = 4  # Adjust depending on your available cores
os.makedirs(output_folder, exist_ok=True)

# === FUNCTIONS ===

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
        with open(out_file, "w") as out_f:
            result = subprocess.run(
                [orca_path, inp_file],
                stdout=out_f,
                stderr=subprocess.STDOUT
            )
        return result.returncode == 0
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

def run_single_job(xyz_file):
    """Full pipeline for one molecule: input creation, execution, status check."""
    name = os.path.splitext(xyz_file)[0]
    xyz_path = os.path.join(xyz_folder, xyz_file)
    inp_path = os.path.join(output_folder, f"{name}.inp")
    out_path = os.path.join(output_folder, f"{name}.out")
    atoms = read_xyz(xyz_path)
    write_orca_input(inp_path, atoms)
    success = run_orca(inp_path, out_path)
    status = check_orca_success(out_path)
    return {'filename': name, 'status': status}

# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":
    import time
    start_time = time.time()

    xyz_files = [f for f in os.listdir(xyz_folder) if f.endswith(".xyz")]
    total_jobs = len(xyz_files)

    if not xyz_files:
        print(f"No .xyz files found in {xyz_folder}!")
        exit()

    print(f"\n🔹 Found {total_jobs} .xyz file(s) in '{xyz_folder}'.")
    print(f"🔹 ORCA executable: {orca_path}")
    print(f"🔹 Running up to {max_parallel_jobs} jobs in parallel...\n")

    job_status = []
    with ProcessPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = {executor.submit(run_single_job, xyz_file): xyz_file for xyz_file in sorted(xyz_files)}
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            job_status.append(result)
            print(f"  [{idx}/{total_jobs}] {result['filename']}: {result['status']}")

    # === SUMMARY ===
    n_success = sum(1 for entry in job_status if entry['status'] == 'success')
    n_failed = total_jobs - n_success
    failed_files = [entry['filename'] for entry in job_status if entry['status'] == 'failed']

    with open("Geo_Opt_Overview.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "status"])
        for entry in job_status:
            writer.writerow([entry['filename'], entry['status']])

    print("\n🔹 Job Summary:")
    print(f"   Total jobs: {total_jobs}")
    print(f"   Success:    {n_success}")
    print(f"   Failed:     {n_failed}")
    if failed_files:
        print("   Failed files:")
        for f in failed_files:
            print(f"     - {f}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Elapsed time: {elapsed:.1f} seconds")
    print('\n📄 Summary written to "Geo_Opt_Overview.csv".\n✅ All jobs processed.')
