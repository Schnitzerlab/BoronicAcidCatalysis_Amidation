#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 23:29:22 2025

@author: tobiasschnitzer
"""

"""
Title: Generation of Non-Redundant Phenylboronic Acid Derivatives
Author: [Your Name]
Date: [Date]
Description:
This script generates derivatives of phenylboronic acid (B(c1ccccc1)(O)O) 
by substituting one hydrogen atom at positions C2, C3, C4, C5, or C6 with 
various functional groups. The generated SMILES codes are stored in a CSV 
and an Excel file, and molecular structures are saved as a PNG image.

Key Features:
1. Generates unique, non-redundant derivatives by eliminating symmetric duplicates.
2. Saves SMILES codes in both `.csv` and `.xlsx` formats.
3. Produces a `.png` file displaying the molecular structures.
4. Outputs the number of generated derivatives in the terminal.
"""
# Required Libraries
from rdkit import Chem
from rdkit.Chem import Draw
import csv
import pandas as pd  # For exporting Excel files



# 1. # Base molecule (Phenylboronic acid) and list of substituents with their correct SMILES representation
base_smiles = "B(c1ccccc1)(O)O"  # Phenylboronic acid
substituents = {
    "Methyl": "C",              # -CH3
    "Isopropyl": "C(C)C",       # -CH(CH3)2 
    "Phenyl": "C2=CC=CC=C2",    # -Ph 
    "Benzoxy": "OCC2=CC=CC=C2", # -OCH2Ph 
    "Hydroxy": "O",             # -OH
    "Methoxy": "OC",            # -OCH3
    "Acetate": "OC(=O)C",        # -OC(=O)CH3
    "Acetylamine": "NC(=O)C",    # -NH–C(=O)CH3
    "Fluoride": "F",             # -F
    "Chloride": "Cl",            # -Cl
    "Bromide": "Br",             # -Br
    "Iodide": "I",               # -I
    "Carbaldehyde": "C=O",       # -CHO (Aldehyd)
    "Methylcarboxylate": "C(=O)OC",  # -C(=O)OCH3
    "Carboxamide": "C(=O)N",     # -C(=O)NH2
    "Methylthiolate": "SC",      # -SCH3
    "Trifluoromethyl": "C(F)(F)F",  # -CF3 
    "Cyano": "C#N",             # -CN
    "Nitro": "[N+](=O)[O-]",    # -NO2
    "Boronic Acid": "B(O)O"       # -B(OH)2
}

# 2. Generate all derivatives (Position 2-6) und filter duplicates
derivatives = []        # List of unique derivates (Name, Position, SMILES)
seen_smiles = set()     # Set to check for duplicatse via canonical SMILES

for name, sub in substituents.items():
    # SMILES patern for substitution at each position C2–C6
    smiles_positions = {
        2: f"B(c1c({sub})cccc1)(O)O",
        3: f"B(c1cc({sub})ccc1)(O)O",
        4: f"B(c1ccc({sub})cc1)(O)O",
        5: f"B(c1cccc({sub})c1)(O)O",
        6: f"B(c1c({sub})cccc1)(O)O"
    }
    for pos, smi in smiles_positions.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # Ungültigen SMILES überspringen
        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        if can_smiles not in seen_smiles:
            seen_smiles.add(can_smiles)
            derivatives.append((name, pos, can_smiles))

# Optional: Sort list of names of substituent and positions
derivatives.sort(key=lambda x: (x[0], x[1]))

# 3. Write SMILES codes in CSV file
csv_filename = "derivatives_phenylboronic_acid.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Substituent", "Position", "SMILES"])
    for name, pos, smi in derivatives:
        writer.writerow([name, pos, smi])

# 4. Safe SMILE codes in Excel file
xlsx_filename = "derivatives_phenylboronic_acid.xlsx"
df = pd.DataFrame(derivatives, columns=["Substituent", "Position", "SMILES"])
df.to_excel(xlsx_filename, index=False, sheet_name="SMILES_Data")

# 5. Generate PNG image with structures
mols = [Chem.MolFromSmiles(smi) for (_, _, smi) in derivatives]
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300))
img.save("derivatives_phenylboronic_acid.png")

# 6. Anzahl der gespeicherten Derivate ausgeben
print(f"Number of generated derivatives: {len(derivatives)}")
print(f"CSV file saved as: {csv_filename}")
print(f"Excel file saved as: {xlsx_filename}")
print("PNG file saved as: derivatives_phenylboronic_acid.png")

