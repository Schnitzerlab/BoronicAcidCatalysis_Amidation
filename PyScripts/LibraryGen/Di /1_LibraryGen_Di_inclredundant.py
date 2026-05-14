#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:44:18 2025

@author: tobiasschnitzer
"""
import itertools
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import math

# Basisstruktur von Phenylboronsäure
base_smiles = "OB(O)c1ccccc1"

# Mögliche Substituenten und ihre SMILES-Codes
substituents_1 = {
    "Methyl": "C", "Isopropyl": "C(C)C", "Phenyl": "C2=CC=CC=C2", "Benzoxy": "OCC2=CC=CC=C2",
    "Hydroxy": "O", "Methoxy": "OC", "Acetate": "OC(=O)C", "Acetylamine": "NC(=O)C",
    "Fluoride": "F", "Chloride": "Cl", "Bromide": "Br", "Iodide": "I",
    "Carbaldehyde": "C=O", "Methylcarboxylate": "C(=O)OC", "Carboxamide": "C(=O)N",
    "Methylthiolate": "SC", "Trifluoromethyl": "C(F)(F)F", "Cyano": "C#N",
    "Nitro": "[N+](=O)[O-]", "Boronic Acid": "B(O)(O)"
}

substituents_2 = {
    "Methyl": "C", "Isopropyl": "C(C)C", "Phenyl": "C3=CC=CC=C3", "Benzoxy": "OCC3=CC=CC=C3",
    "Hydroxy": "O", "Methoxy": "OC", "Acetate": "OC(=O)C", "Acetylamine": "NC(=O)C",
    "Fluoride": "F", "Chloride": "Cl", "Bromide": "Br", "Iodide": "I",
    "Carbaldehyde": "C=O", "Methylcarboxylate": "C(=O)OC", "Carboxamide": "C(=O)N",
    "Methylthiolate": "SC", "Trifluoromethyl": "C(F)(F)F", "Cyano": "C#N",
    "Nitro": "[N+](=O)[O-]", "Boronic Acid": "B(O)(O)"
}

# Korrekte SMILES-Vorlagen mit definierten Positionen
smiles_templates = {
    "OB(O)C1=CC=CC({sub2})=C1{sub1}": (6, 5),
    "OB(O)C1=CC=C({sub2})C=C1{sub1}": (6, 4),
    "OB(O)C1=CC({sub2})=CC=C1{sub1}": (6, 3),
    "OB(O)C1=C({sub2})C=CC=C1{sub1}": (6, 2),
    "OB(O)C1=CC=CC({sub1})=C1{sub2}": (5, 6),
    "OB(O)C1=CC=C({sub2})C({sub1})=C1": (5, 4),
    "OB(O)C1=CC({sub2})=CC({sub1})=C1": (5, 3),
    "OB(O)C1=C({sub2})C=CC({sub1})=C1": (5, 2),
    "OB(O)C1=CC=C({sub1})C=C1{sub2}": (4, 6),
    "OB(O)C1=CC=C({sub1})C({sub2})=C1": (4, 5),
    "OB(O)C1=CC({sub2})=C({sub1})C=C1": (4, 3),
    "OB(O)C1=C({sub2})C=C({sub1})C=C1": (4, 2),
    "OB(O)C1=CC({sub1})=CC=C1{sub2}": (3, 6),
    "OB(O)C1=CC({sub1})=CC({sub2})=C1": (3, 5),
    "OB(O)C1=CC({sub1})=C({sub2})C=C1": (3, 4),
    "OB(O)C1=C({sub2})C({sub1})=CC=C1": (3, 2),
    "OB(O)C1=C({sub1})C=CC=C1{sub2}": (2, 6),
    "OB(O)C1=C({sub1})C=CC({sub2})=C1": (2, 5),
    "OB(O)C1=C({sub1})C=C({sub2})C=C1": (2, 4),
    "OB(O)C1=C({sub1})C({sub2})=CC=C1": (2, 3)
}

data = []
molecules = []

# Generiere alle Kombinationen von zwei Substituenten (inklusive identischer Paare)
for (name1, sub1), (name2, sub2) in itertools.product(substituents_1.items(), substituents_2.items()):
    for template, (pos1, pos2) in smiles_templates.items():
        smiles = template.format(sub1=sub1, sub2=sub2)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            data.append([smiles, name1, name2, pos1, pos2])
            molecules.append(mol)

# Daten als CSV speichern
csv_filename = "phenylboronic_derivatives.csv"
pd.DataFrame(data, columns=["SMILES", "Substituent 1", "Substituent 2", "Position 1", "Position 2"]).to_csv(csv_filename, index=False)

# Daten als Excel speichern
excel_filename = "phenylboronic_derivatives.xlsx"
pd.DataFrame(data, columns=["SMILES", "Substituent 1", "Substituent 2", "Position 1", "Position 2"]).to_excel(excel_filename, index=False)

# Strukturen als PNG speichern, falls zu viele Moleküle in einem Bild problematisch sind
mols_per_image = 100
num_images = math.ceil(len(molecules) / mols_per_image)

for i in range(num_images):
    subset = molecules[i * mols_per_image:(i + 1) * mols_per_image]
    img = Draw.MolsToGridImage(subset, molsPerRow=10, subImgSize=(300, 300))
    img_filename = f"phenylboronic_derivatives_{i+1}.png"
    if isinstance(img, Image.Image):
        img.save(img_filename)
    print(f"Strukturabbildung gespeichert als: {img_filename}")

# Anzahl der generierten Derivate ausgeben
print(f"Anzahl der generierten Derivate: {len(data)}")
print(f"CSV-Datei gespeichert als: {csv_filename}")
print(f"Excel-Datei gespeichert als: {excel_filename}")
