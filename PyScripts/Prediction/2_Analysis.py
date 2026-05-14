#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:09:26 2025

@author: tobiasschnitzer
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# === Datei laden ===
df = pd.read_excel("DATA_with_predictions.xlsx")

# === Nach "Predicted conversions (%)" sortieren (absteigend) ===
df_sorted = df.sort_values(by="conversions (%)", ascending=False).reset_index(drop=True)

# === Farbskala definieren: Blau (hoch) → Rot (niedrig)
norm = plt.Normalize(df_sorted["Predicted conversions (%)"].min(), df_sorted["Predicted conversions (%)"].max())
cmap = plt.cm.get_cmap('coolwarm')
colors = [cmap(norm(val)) for val in df_sorted["Predicted conversions (%)"]]

# === Top 50 Moleküle extrahieren
top_50 = df_sorted.head(50)

# === Moleküle und Beschriftungen erzeugen
mols = []
legends = []
for i, row in top_50.iterrows():
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol:
        mols.append(mol)
        legends.append(f'{row["Molecule"]}\n{row["Predicted conversions (%)"]:.1f}%')
    else:
        mols.append(None)
        legends.append("Invalid SMILES")

# === Bild mit RDKit zeichnen (SVG oder PNG)
img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(300, 300),
    legends=legends,
    useSVG=False
)

# === Bild anzeigen und speichern
img.save("Top50_Predicted_Structures.png")
img.show()
