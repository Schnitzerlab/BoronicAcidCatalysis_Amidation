#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:40:08 2025

@author: tobiasschnitzer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:00:02 2025

@author: Tobias Schnitzer
"""

import os
import subprocess

# Directories for PM3 optimizations
INPUT_FOLDER = "Smiles_to_XYZ"
OUTPUT_FOLDER = "PM3_Optimized"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Adjust ORCA path
ORCA_PATH = "/Users/tobiasschnitzer/Library/orca_6_0_1/orca"

def create_orca_pm3_input(xyz_file):
    """Creates an ORCA input file for PM3 optimization."""
    base_name = os.path.splitext(os.path.basename(xyz_file))[0]
    input_file = os.path.join(OUTPUT_FOLDER, f"{base_name}.inp")
    output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}.out")

    xyz_absolute_path = os.path.abspath(os.path.join(INPUT_FOLDER, xyz_file))

    with open(input_file, "w") as f:
        f.write("! PM3 Opt\n")  # ✅ Correct ORCA syntax for semi-empirical calculations
        f.write(f"* xyzfile 0 1 {xyz_absolute_path}\n")

    return input_file, output_file

def run_orca(input_file, output_file, index, total):
    """Runs ORCA for PM3 optimization."""
    print(f"🔄 Running ORCA PM3 optimization ({index}/{total}): {input_file}")
    try:
        subprocess.run([ORCA_PATH, input_file], stdout=open(output_file, "w"), stderr=subprocess.PIPE, check=True)
        print(f"✅ Optimization completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ORCA error for {input_file}: {e}")

if __name__ == "__main__":
    xyz_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xyz")]
    total_files = len(xyz_files)
    print(f"🚀 Starting sequential PM3 optimization for {total_files} molecules...")

    for index, xyz_file in enumerate(xyz_files, start=1):
        input_file, output_file = create_orca_pm3_input(xyz_file)
        run_orca(input_file, output_file, index, total_files)

    print("🎯 All PM3 optimizations successfully completed!")
