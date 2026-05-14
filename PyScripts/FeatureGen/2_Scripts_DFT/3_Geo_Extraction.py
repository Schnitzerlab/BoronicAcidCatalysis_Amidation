#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:13:35 2025

@author: tobiasschnitzer

Analysis of B3LYP/Def2-SVP Optimized Geometries:
- Determines B–O and B–C bond lengths
- Computes the dihedral angle between:
    1. The plane defined by boron (B) and its two neighboring oxygen atoms (O)
    2. The plane defined by the carbon atom bound to B and its two neighboring carbon atoms

Author: Tobias Schnitzer
Date: 2025-07-22
"""

import os
import csv
import numpy as np
from ase import Atoms
from ase.io import read

# Input and output folders/files
INPUT_FOLDER = "B3LYP_Optimized_XYZ"  # Folder containing B3LYP/Def2-SVP optimized geometries (.xyz)
OUTPUT_CSV = "geometric_parameters_summary.csv"

results = []

def plane_fit(points):
    """Calculates the normal vector of a plane defined by three or more points."""
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    return normal / np.linalg.norm(normal)

def analyze_geometry(xyz_path):
    """Analyzes bond lengths and the dihedral angle between two defined planes."""
    try:
        atoms = read(xyz_path)
    except Exception as e:
        print(f"❌ Error reading {xyz_path}: {e}")
        return

    data = {"File": os.path.basename(xyz_path)}

    # Find boron atom
    bor_index = next((i for i, atom in enumerate(atoms) if atom.symbol == "B"), None)
    if bor_index is None:
        print(f"⚠️ No B atom found in {xyz_path}, skipping...")
        return

    # Find neighbors of boron (within 2.1 Å)
    neighbors = [i for i in range(len(atoms)) if i != bor_index and atoms.get_distance(bor_index, i) < 2.1]
    o_indices = [i for i in neighbors if atoms[i].symbol == "O"]
    c_indices = [i for i in neighbors if atoms[i].symbol == "C"]

    if len(o_indices) < 2 or len(c_indices) < 1:
        print(f"⚠️ Not enough B neighbors in {xyz_path}, skipping...")
        return

    # Calculate bond lengths
    data["B–C distance (Å)"] = round(atoms.get_distance(bor_index, c_indices[0]), 3)
    data["B–O distance 1 (Å)"] = round(atoms.get_distance(bor_index, o_indices[0]), 3)
    data["B–O distance 2 (Å)"] = round(atoms.get_distance(bor_index, o_indices[1]), 3)

    # Calculate dihedral angle
    dihedral_angle = None
    if len(o_indices) >= 2 and len(c_indices) >= 1:
        o_b_o_plane_positions = np.array([atoms[i].position for i in o_indices] + [atoms[bor_index].position])
        normal_o_b_o = plane_fit(o_b_o_plane_positions)

        # C-plane: central C atom bound to B and its two neighboring C atoms
        central_c_index = c_indices[0]
        c_neighbors = [i for i in range(len(atoms)) if atoms[i].symbol == "C" and i != bor_index and atoms.get_distance(central_c_index, i) < 1.6]
        if len(c_neighbors) < 2:
            print(f"⚠️ Not enough C atoms for second plane in {xyz_path} (at least 2 required).")
        else:
            c_plane_positions = np.array([atoms[i].position for i in c_neighbors] + [atoms[central_c_index].position])
            normal_c_plane = plane_fit(c_plane_positions)

            cos_theta = np.dot(normal_o_b_o, normal_c_plane)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Correct for numerical errors
            dihedral_angle = np.degrees(np.arccos(cos_theta))

            # Ensure dihedral is within [0,90]
            if dihedral_angle > 90:
                dihedral_angle = 180 - dihedral_angle
            if dihedral_angle < 0:
                dihedral_angle = -dihedral_angle

            data["Dihedral O–B–O / C–C–C (°)"] = round(dihedral_angle, 3)
    else:
        print(f"⚠️ Could not compute dihedral for {xyz_path}")
        data["Dihedral O–B–O / C–C–C (°)"] = None

    results.append(data)

def sort_key(entry):
    """Sort results by molecule number from file name, if possible."""
    try:
        filename = entry["File"]
        mol_number = int("".join(filter(str.isdigit, filename)))
        return mol_number
    except ValueError:
        return float('inf')

def main():
    """Main analysis function for all XYZ files in the folder."""
    xyz_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xyz")]
    print(f"🚀 Analyzing {len(xyz_files)} molecules from {INPUT_FOLDER}")

    for idx, xyz_file in enumerate(xyz_files, start=1):
        print(f"[{idx}/{len(xyz_files)}] Analyzing: {xyz_file}")
        analyze_geometry(os.path.join(INPUT_FOLDER, xyz_file))

    if not results:
        print("❌ No results found! Please check your input data.")
        return

    results.sort(key=sort_key)

    # Save results
    fieldnames = sorted({key for res in results for key in res})
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Results saved to {OUTPUT_CSV} ({len(results)} molecules analyzed)")

if __name__ == "__main__":
    main()
