#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:39:56 2025

@author: tobiasschnitzer
"""

import pandas as pd

# Mapping for transformation
transform_map = {2: 6, 3: 5, 4: 4, 5: 3, 6: 2}

# Load the CSV file
input_file = "filtered_phenylboronic_derivatives.csv"  # Passe den Dateinamen ggf. an
output_csv = "filtered_phenylboronic_derivatives_transformed.csv"
output_xlsx = "filtered_phenylboronic_derivatives_transformed.xlsx"

# Versuch, das richtige Trennzeichen zu erkennen
df = pd.read_csv(input_file, sep=None, engine='python')
df.columns = df.columns.str.strip()  # Entferne unerwartete Leerzeichen

# Debug: Spaltennamen ausgeben
print("Gefundene Spaltennamen:", df.columns.tolist())

# Transform positions
df["Position 1 transf"] = df["Position 1"].map(transform_map)
df["Position 2 transf"] = df["Position 2"].map(transform_map)

# Compute min and sum for both pairs
df["Min Original"] = df[["Position 1", "Position 2"]].min(axis=1)
df["Sum Original"] = df["Position 1"] + df["Position 2"]
df["Min Transf"] = df[["Position 1 transf", "Position 2 transf"]].min(axis=1)
df["Sum Transf"] = df["Position 1 transf"] + df["Position 2 transf"]

# Choose the final positions based on min and sum comparison
def select_final_positions(row):
    if row["Min Original"] < row["Min Transf"]:
        return row["Position 1"], row["Position 2"]
    elif row["Min Transf"] < row["Min Original"]:
        return row["Position 1 transf"], row["Position 2 transf"]
    else:  # Gleichheit der Minimalwerte
        if row["Sum Original"] <= row["Sum Transf"]:
            return row["Position 1"], row["Position 2"]
        else:
            return row["Position 1 transf"], row["Position 2 transf"]

df[["Pos1_final", "Pos2_final"]] = df.apply(select_final_positions, axis=1, result_type="expand")

# Save to CSV
df.to_csv(output_csv, index=False, sep="\t")

# Save to Excel with selected columns
df_filtered = df[["SMILES", "Substituent 1", "Substituent 2", "Pos1_final", "Pos2_final"]]
df_filtered.to_excel(output_xlsx, index=False)

print(f"Dateien gespeichert: {output_csv}, {output_xlsx}")
