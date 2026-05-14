#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 20:23:07 2025

@author: tobiasschnitzer

Check ORCA Termination Status of Output Files

This script verifies whether all ORCA calculations in a given directory
terminated normally. It scans for the presence of the line:
'****ORCA TERMINATED NORMALLY****' in each .out file and prints a list
of files that did not finish correctly.

Directory scanned: B3LYP_EA_IP_FUKUI
"""

import os

# === Configuration ===
output_folder = "B3LYP_EA_IP_FUKUI"
termination_flag = "****ORCA TERMINATED NORMALLY****"

# === Find all .out files ===
out_files = [f for f in os.listdir(output_folder) if f.endswith(".out")]
if not out_files:
    print(f"No .out files found in '{output_folder}'.")
    exit()

# === Check each file for termination flag ===
failed_files = []
for out_file in sorted(out_files):
    file_path = os.path.join(output_folder, out_file)
    try:
        with open(file_path, "r") as f:
            content = f.read()
        if termination_flag not in content:
            failed_files.append(out_file)
    except Exception as e:
        print(f"⚠️ Error reading {out_file}: {e}")
        failed_files.append(out_file)

# === Report ===
print(f"\n🔍 Checked {len(out_files)} ORCA output files in '{output_folder}'.")

if failed_files:
    print(f"\n❌ {len(failed_files)} file(s) did NOT terminate normally:")
    for f in failed_files:
        print(f"   - {f}")
else:
    print("\n✅ All ORCA jobs terminated normally.")
