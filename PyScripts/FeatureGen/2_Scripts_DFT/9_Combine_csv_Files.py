#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:59:37 2025

Merges multiple descriptor CSVs into one and sorts molecules by ascending mol_xx order.
Adapted for summary files.

"""

import pandas as pd

# === Neue File-Paths ===
dipole_file   = "Dipole_Moment_summary.csv"
geometry_file = "geometric_parameters_summary.csv"
homo_lumo_file = "HOMO_LUMO_summary.csv"
mulliken_file = "Mulliken_Loewdin_Mayer_summary.csv"
pm3_file = "EA_IP_Fukui_summary.csv"
smiles_file = "MolMass_RotBonds_TPSA_summary.csv"

def load_csv(file_path, possible_columns):
    """Loads a CSV file and renames the molecule column if necessary."""
    try:
        df = pd.read_csv(file_path)
        for col in possible_columns:
            if col in df.columns:
                df = df.rename(columns={col: "Molecule"})
                return df
        print(f"⚠️ No recognized molecule column in {file_path}. Available columns: {df.columns.tolist()}")
        return None
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return None

# Load standard CSV files with known molecule columns
df_dipole = load_csv(dipole_file, ["Molecule"])
df_geometry = load_csv(geometry_file, ["File", "Molecule"])
df_homo_lumo = load_csv(homo_lumo_file, ["File", "Molecule"])
df_mulliken = load_csv(mulliken_file, ["Molecule"])
df_pm3 = load_csv(pm3_file, ["Molecule", "Molekül"])  # Abfangen verschiedener Namensvarianten

# Load SMILES/Properties CSV, assign Molecule IDs if missing
try:
    df_smiles = pd.read_csv(smiles_file)
    # Prüfen, ob Spalte "Molecule" existiert, sonst generieren
    if "Molecule" not in df_smiles.columns:
        df_smiles.insert(0, "Molecule", [f"mol_{i+1}" for i in range(len(df_smiles))])
except FileNotFoundError:
    print(f"⚠️ File not found: {smiles_file}")
    df_smiles = None

# Sammle alle geladenen DataFrames
dataframes = [df for df in [df_dipole, df_geometry, df_homo_lumo, df_mulliken, df_pm3, df_smiles] if df is not None]

if not dataframes:
    print("❌ No valid dataframes found. Check file paths and column names.")
    exit()

# Standardisiere Molekülnamen (z. B. "mol_1.xyz" → "mol_1")
def standardize_molecule_name(name):
    return name.split(".")[0].replace("xyz", "").strip()

for df in dataframes:
    if "Molecule" in df.columns:
        df["Molecule"] = df["Molecule"].astype(str).apply(standardize_molecule_name)

# Merge aller DataFrames auf 'Molecule'
df_merged = dataframes[0]
for df in dataframes[1:]:
    df_merged = df_merged.merge(df, on="Molecule", how="outer")

# Bevorzugte Spaltenreihenfolge (angepasst)
preferred_order = [
    "Molecule", "SMILES", "Molecular Weight (g/mol)", "Rotatable Bond Count", "TPSA",
    "Dipole_Magnitude_Debye",
    "Fukui+", "Fukui-", "B–C distance (Å)", "B–O distance 1 (Å)", "B–O distance 2 (Å)",
    "Dihedral O–B–O / C–C–C (°)", "HOMO Energy (eV)", "LUMO Energy (eV)",
    "HOMO-LUMO Gap (eV)", "E_neutral (Eh)", "E_cation (Eh)", "E_anion (Eh)",
    "Ionization Energy (eV)", "Electron Affinity (eV)", "Mulliken_B_Charge",
    "Loewdin_B_Charge", "Mayer_QA", "Mayer_VA", "Mayer_BVA", "Mayer_FA"
]

# Nur vorhandene Spalten in der bevorzugten Reihenfolge behalten
available_columns = [col for col in preferred_order if col in df_merged.columns]

# Numerische und nicht-numerische Spalten trennen
numeric_cols = df_merged[available_columns].select_dtypes(include="number").columns.tolist()
non_numeric_cols = [col for col in available_columns if col not in numeric_cols]

# Mittelwerte für numerische Spalten gruppiert nach Molecule
df_final_numeric = df_merged[["Molecule"] + numeric_cols].groupby("Molecule", as_index=False).mean()

# Nicht-numerische Spalten deduplizieren
non_numeric_cols_cleaned = [col for col in non_numeric_cols if col != "Molecule"]
df_final_non_numeric = df_merged[["Molecule"] + non_numeric_cols_cleaned].drop_duplicates(subset="Molecule")

# Numerisch und nicht-numerisch zusammenführen
df_final = pd.merge(df_final_numeric, df_final_non_numeric, on="Molecule", how="left")

# Endgültige Spaltenreihenfolge
df_final = df_final[[col for col in preferred_order if col in df_final.columns]]

# Sortieren nach mol_xx
def extract_mol_number(name):
    try:
        return int(name.replace("mol_", ""))
    except:
        return float("inf")  # Nicht-standardisierte Namen kommen zuletzt

df_final["__mol_index"] = df_final["Molecule"].apply(extract_mol_number)
df_final = df_final.sort_values("__mol_index").drop(columns="__mol_index").reset_index(drop=True)

# Speichern
output_file = "merged_results.csv"
df_final.to_csv(output_file, index=False)

print(f"✅ Merged file saved: {output_file}")
