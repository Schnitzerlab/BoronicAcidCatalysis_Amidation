#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:54:44 2025

@author: tobiasschnitzer
"""

import pandas as pd
from rdkit import Chem

# Datei mit den ursprünglichen SMILES laden
input_filename = "phenylboronic_derivatives.csv"
output_filename = "filtered_phenylboronic_derivatives.csv"

# CSV-Datei einlesen
df = pd.read_csv(input_filename)

# Set zur Speicherung der einzigartigen kanonischen SMILES
unique_smiles = set()
filtered_data = []

# Durch alle SMILES iterieren und kanonische SMILES generieren
for _, row in df.iterrows():
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        if canonical_smiles not in unique_smiles:
            unique_smiles.add(canonical_smiles)
            filtered_data.append(row)

# Ergebnis als neuen DataFrame speichern
filtered_df = pd.DataFrame(filtered_data)

# Gesäuberte Datei speichern
filtered_df.to_csv(output_filename, index=False)

# Anzahl der eindeutigen Strukturen ausgeben
print(f"Anzahl der eindeutigen Strukturen nach Entfernung der Redundanz: {len(filtered_df)}")
print(f"Gefilterte Datei gespeichert als: {output_filename}")