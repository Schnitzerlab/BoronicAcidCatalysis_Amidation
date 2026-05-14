#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:50:39 2025

@author: tobiasschnitzer
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Input and output file paths
input_file = 'SMILES.xlsx'
output_file = 'MolMass_RotBonds_TPSA_summary.csv'

# Load the input Excel file
df = pd.read_excel(input_file)

# Initialize lists to store molecular properties
molecular_weights = []
rotatable_bonds = []
tpsa_values = []

# Iterate over each SMILES string in the column 'SMILES'
for smile in df['SMILES']:
    mol = Chem.MolFromSmiles(smile)
    if mol:
        mw = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
    else:
        # In case of an invalid SMILES, append None
        mw = None
        rot_bonds = None
        tpsa = None

    molecular_weights.append(mw)
    rotatable_bonds.append(rot_bonds)
    tpsa_values.append(tpsa)

# Add the computed properties as new columns to the dataframe
df['Molecular Weight (g/mol)'] = molecular_weights
df['Rotatable Bond Count'] = rotatable_bonds
df['TPSA'] = tpsa_values

# Save the annotated dataframe to a CSV file
df.to_csv(output_file, index=False)

print(f"Annotated SMILES data saved to '{output_file}'")
