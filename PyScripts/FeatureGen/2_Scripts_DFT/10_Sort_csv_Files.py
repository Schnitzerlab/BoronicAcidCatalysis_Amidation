#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:06:49 2025

@author: tobiasschnitzer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organizes CSV files into a subfolder.
Moves all CSV files except 'merged_results.csv' into 'csv_Files/'.
"""

import os
import shutil

# Define the target directory
target_folder = "csv_Files"

# Create the folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# List all CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != "merged_results.csv"]

# Move each file
for file in csv_files:
    source_path = os.path.join('.', file)
    destination_path = os.path.join(target_folder, file)
    shutil.move(source_path, destination_path)
    print(f"✅ Moved: {file} → {target_folder}/")

if not csv_files:
    print("ℹ️ No CSV files to move (other than 'merged_results.csv').")
else:
    print(f"\n📁 All files moved to '{target_folder}/'")
