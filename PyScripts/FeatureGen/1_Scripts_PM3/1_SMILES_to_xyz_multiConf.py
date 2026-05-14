#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 15:24:43 2025

Conformer generation from SMILES list in Excel.
For each molecule, 20 conformers are generated, MMFF94-optimized,
and saved as separate XYZ files.

Author: Tobias Schnitzer
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock
from tqdm import tqdm

# === Configuration ===
input_excel = "SMILES.xlsx"
smiles_column = "SMILES"
output_dir = "Smiles_to_XYZ"
num_confs = 20
os.makedirs(output_dir, exist_ok=True)

# === Read SMILES ===
df = pd.read_excel(input_excel)
smiles_list = df[smiles_column].dropna().tolist()

# === Conformer generation loop ===
for idx, smi in tqdm(enumerate(smiles_list, start=1), total=len(smiles_list), desc="Generating conformers"):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"⚠️ Invalid SMILES at row {idx}: {smi}")
        continue

    mol = Chem.AddHs(mol)

    # Embed conformers using ETKDGv3
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    params.maxAttempts = 1000
    params.useSmallRingTorsions = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) == 0:
        print(f"❌ No conformers generated for mol_{idx}")
        continue

    # Optimize conformers using MMFF94
    for conf_id in conf_ids:
        try:
            result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=10000)
            if result != 0:
                print(f"⚠️ mol_{idx} conf_{conf_id+1} not fully optimized.")
        except Exception as e:
            print(f"❌ Error optimizing mol_{idx} conf_{conf_id+1}: {e}")

    # Write conformers to XYZ files
    for i, conf_id in enumerate(conf_ids, start=1):
        filename = f"mol_{idx}_conf{i}.xyz"
        xyz_path = os.path.join(output_dir, filename)
        xyz_block = MolToXYZBlock(mol, confId=conf_id)
        with open(xyz_path, "w") as f:
            f.write(xyz_block)

print(f"✅ Finished. All conformers saved in '{output_dir}'")
