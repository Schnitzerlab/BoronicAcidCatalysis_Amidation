#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:13:02 2025

@author: tobiasschnitzer

Extraction of Final Geometries from ORCA Output Files (B3LYP/Def2-SVP)
Author: Tobias Schnitzer
Date: 2025-07-22

This script extracts the final optimized geometry from each ORCA output file
(B3LYP/Def2-SVP) and writes it as a .xyz file to a specified folder.

Input:
    - ORCA output files (.out) in the B3LYP_Optimized folder
Output:
    - Extracted final geometries as .xyz files in the B3LYP_Optimized_XYZ folder

Usage:
    - Ensure OUTPUT_FOLDER and EXTRACTED_FOLDER are set correctly.
    - Run the script: python3 GeoOpt_B3LYP_def2SVP_Extract.py
"""

import os

# Define folders
OUTPUT_FOLDER = "B3LYP_Optimized"         # Folder containing ORCA output files (.out)
EXTRACTED_FOLDER = "B3LYP_Optimized_XYZ"  # Folder to save extracted .xyz files
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

def extract_final_xyz_from_orca_output(out_file_path):
    """
    Extracts the final geometry from an ORCA output file and returns an XYZ-formatted string.
    Returns None if extraction fails.
    """
    try:
        with open(out_file_path, "r") as f:
            lines = f.readlines()
        # Find all indices of "CARTESIAN COORDINATES (ANGSTROEM)"
        indices = [i for i, line in enumerate(lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in line]
        if not indices:
            return None
        last_index = indices[-1]
        # Coordinates start two lines after header
        geometry_lines = []
        for line in lines[last_index + 2:]:
            if line.strip() == "":
                break
            geometry_lines.append(line.strip())
        n_atoms = len(geometry_lines)
        # Remove potential atom index at the beginning, keep: Atom  X  Y  Z
        xyz_block = []
        for l in geometry_lines:
            split = l.split()
            # If the first entry is a number, skip it (ORCA prints index sometimes)
            if split[0].isdigit() and len(split) == 5:
                xyz_block.append(' '.join(split[1:]))
            else:
                xyz_block.append(' '.join(split))
        xyz_content = f"{n_atoms}\nExtracted from {os.path.basename(out_file_path)}\n"
        xyz_content += "\n".join(xyz_block)
        return xyz_content
    except Exception:
        return None

if __name__ == "__main__":
    out_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".out")]
    total_files = len(out_files)

    if not out_files:
        print("❌ No .out files found! Please check the output folder path.")
    else:
        print(f"🚀 Found {total_files} ORCA output files. Starting extraction...\n")

        for idx, out_file in enumerate(out_files, start=1):
            print(f"[{idx}/{total_files}] Processing: {out_file} ...", end=" ")
            out_file_path = os.path.join(OUTPUT_FOLDER, out_file)
            xyz_content = extract_final_xyz_from_orca_output(out_file_path)

            if xyz_content:
                xyz_file = os.path.splitext(out_file)[0] + ".xyz"
                xyz_file_path = os.path.join(EXTRACTED_FOLDER, xyz_file)
                with open(xyz_file_path, "w") as f:
                    f.write(xyz_content)
                print("✅ Extraction successful.")
            else:
                print("❌ Extraction failed.")

        print(f"\n🎯 Extraction complete. XYZ files saved in '{EXTRACTED_FOLDER}'.")
