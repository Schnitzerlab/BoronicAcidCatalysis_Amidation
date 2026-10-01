#!/usr/bin/env python3
"""Generate position-specific substituent descriptors for arylboronic acids.

For every molecule, the phenyl ring attached to the primary boronic-acid unit
is identified and substituents at positions 2-6 are extracted automatically.
Each unique substituent is represented by a phenyl-capped model, pre-relaxed
with UFF, optimized at the PM3 level with ORCA, and converted into the three
substituent descriptors used by the machine-learning workflow:

- substituent molecular mass (Molmasse)
- summed atomic van-der-Waals sphere volume (VdW)
- buried-volume descriptor (BV)

The resulting descriptors are mapped back to ring positions 2-6 together with
binary substitution indicators. Sterimol parameters and Tolman cone angles are
not calculated because they are not used as model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from common import (
    dataset_dir,
    ensure_dir,
    extract_last_orca_geometry,
    load_manifest,
    locate_orca,
    optimization_done,
    orca_terminated_normally,
    run_orca_job,
    write_table,
    write_xyz,
)

# Values used for an unsubstituted ring position (hydrogen).
H_MASS = 1.008
H_VDW = 7.238
H_BV = 4.693

# Atomic van-der-Waals radii used for the substituent volume descriptor.
# Boron is written explicitly as 1.50 A, corresponding to the default value
# used for elements not listed in the original descriptor calculation.
VDW_RADII = {
    "H": 1.20,
    "B": 1.50,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}


def _find_primary_phenyl_boron(mol: Chem.Mol):
    """Return (boron_idx, ring_set, pos1_idx) for the first aryl-B unit."""
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if len(r) == 6]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "B":
            continue
        b_idx = atom.GetIdx()
        for nbr in atom.GetNeighbors():
            n_idx = nbr.GetIdx()
            if nbr.GetSymbol() != "C":
                continue
            for ring in rings:
                if n_idx in ring and all(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in ring):
                    return b_idx, set(ring), n_idx
    return None


def _ring_order(ring_atoms: set[int], pos1_idx: int, neighbor_idx: int, mol: Chem.Mol) -> list[int]:
    order = [pos1_idx, neighbor_idx]
    prev = pos1_idx
    current = neighbor_idx
    while True:
        candidates = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in ring_atoms and n.GetIdx() != prev
        ]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == pos1_idx:
            break
        order.append(nxt)
        prev, current = current, nxt
    return order


def _standalone_smiles(dummy_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(dummy_smiles)
    if mol is None:
        return None
    dummy = next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy is None:
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(dummy)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Chem.MolToSmiles(out, canonical=True, isomericSmiles=True)


def extract_substituents(smiles: str) -> tuple[list[dict], dict]:
    """Extract substituent fragments and positions from a phenylboronic acid."""
    mol = Chem.MolFromSmiles(str(smiles))
    qc = {"status": "success", "reason": "", "number_substituents": 0}
    if mol is None:
        qc.update(status="failed", reason="invalid_smiles")
        return [], qc

    found = _find_primary_phenyl_boron(mol)
    if found is None:
        qc.update(status="failed", reason="no_supported_phenylboronic_acid_unit")
        return [], qc
    _, ring_atoms, pos1_idx = found

    ring_neighbors = [
        n.GetIdx() for n in mol.GetAtomWithIdx(pos1_idx).GetNeighbors() if n.GetIdx() in ring_atoms
    ]
    if len(ring_neighbors) != 2:
        qc.update(status="failed", reason="ring_numbering_failed")
        return [], qc

    order_a = _ring_order(ring_atoms, pos1_idx, ring_neighbors[0], mol)
    order_b = _ring_order(ring_atoms, pos1_idx, ring_neighbors[1], mol)

    raw = []
    for ring_idx in ring_atoms:
        if ring_idx == pos1_idx:
            continue
        for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
            root = nbr.GetIdx()
            if root in ring_atoms:
                continue

            branch = set()
            stack = [root]
            while stack:
                idx = stack.pop()
                if idx in branch or idx in ring_atoms:
                    continue
                branch.add(idx)
                for n2 in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if n2.GetIdx() not in ring_atoms and n2.GetIdx() not in branch:
                        stack.append(n2.GetIdx())

            # Detach the substituent from the aryl ring before adding the dummy
            # attachment atom. This prevents RDKit from assigning an artificial
            # attachment hydrogen to elements such as iodine and sulfur.
            rw = Chem.RWMol(mol)
            rw.RemoveBond(ring_idx, root)
            dummy_idx = rw.AddAtom(Chem.Atom("*"))
            rw.AddBond(root, dummy_idx, Chem.BondType.SINGLE)
            frag = Chem.MolFragmentToSmiles(
                rw.GetMol(),
                atomsToUse=sorted(branch | {dummy_idx}),
                canonical=True,
                isomericSmiles=True,
            )
            stand = _standalone_smiles(frag)
            pos_a = order_a.index(ring_idx) + 1 if ring_idx in order_a else None
            pos_b = order_b.index(ring_idx) + 1 if ring_idx in order_b else None
            raw.append(
                {
                    "PositionA": pos_a,
                    "PositionB": pos_b,
                    "SubstituentKey": frag,
                    "SubstituentSMILES": stand,
                }
            )

    positions_a = sorted(x["PositionA"] for x in raw if x["PositionA"] is not None)
    positions_b = sorted(x["PositionB"] for x in raw if x["PositionB"] is not None)
    use_a = not positions_b or (positions_a and tuple(positions_a) <= tuple(positions_b))

    out = []
    for item in raw:
        pos = item["PositionA"] if use_a else item["PositionB"]
        if pos is None:
            continue
        out.append(
            {
                "Position": int(pos),
                "SubstituentKey": item["SubstituentKey"],
                "SubstituentSMILES": item["SubstituentSMILES"],
            }
        )
    out.sort(key=lambda x: x["Position"])
    qc["number_substituents"] = len(out)
    return out, qc


def _dummy_and_neighbor(mol: Chem.Mol) -> tuple[int, int]:
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1 or len(dummies[0].GetNeighbors()) != 1:
        raise ValueError("Expected exactly one dummy atom with one neighbor.")
    return dummies[0].GetIdx(), dummies[0].GetNeighbors()[0].GetIdx()


def build_phenyl_capped_model(dummy_smiles: str, seed: int = 42):
    """Build a phenyl-capped substituent and return molecule + sub atom indices."""
    ph = Chem.MolFromSmiles("c1ccccc1[*]")
    sub = Chem.MolFromSmiles(dummy_smiles)
    if ph is None or sub is None:
        raise ValueError(f"Could not parse substituent: {dummy_smiles}")

    for atom in ph.GetAtoms():
        atom.SetProp("_origin_fragment", "phenyl")
    for atom in sub.GetAtoms():
        atom.SetProp("_origin_fragment", "substituent")

    ph_dummy, ph_attach = _dummy_and_neighbor(ph)
    sub_dummy, sub_attach = _dummy_and_neighbor(sub)
    offset = ph.GetNumAtoms()

    rw = Chem.RWMol(Chem.CombineMols(ph, sub))
    rw.AddBond(ph_attach, offset + sub_attach, Chem.BondType.SINGLE)
    for idx in sorted([ph_dummy, offset + sub_dummy], reverse=True):
        rw.RemoveAtom(idx)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    capped = Chem.AddHs(capped)

    # Newly added H atoms inherit the fragment identity of their heavy-atom neighbor.
    for atom in capped.GetAtoms():
        if atom.GetAtomicNum() == 1 and not atom.HasProp("_origin_fragment"):
            nbrs = atom.GetNeighbors()
            if nbrs and nbrs[0].HasProp("_origin_fragment"):
                atom.SetProp("_origin_fragment", nbrs[0].GetProp("_origin_fragment"))

    params = AllChem.ETKDG()
    params.randomSeed = int(seed)
    code = AllChem.EmbedMolecule(capped, params)
    if code != 0:
        raise RuntimeError("RDKit 3D embedding failed")

    uff_status = None
    try:
        uff_status = int(AllChem.UFFOptimizeMolecule(capped))
    except Exception:
        uff_status = -1

    sub_indices = [
        a.GetIdx()
        for a in capped.GetAtoms()
        if a.HasProp("_origin_fragment") and a.GetProp("_origin_fragment") == "substituent"
    ]
    if not sub_indices:
        raise RuntimeError("No substituent atom indices retained after capping")
    return capped, sub_indices, uff_status


def mol_to_xyz_atoms(mol: Chem.Mol):
    conf = mol.GetConformer()
    return [
        (
            atom.GetSymbol(),
            float(conf.GetAtomPosition(atom.GetIdx()).x),
            float(conf.GetAtomPosition(atom.GetIdx()).y),
            float(conf.GetAtomPosition(atom.GetIdx()).z),
        )
        for atom in mol.GetAtoms()
    ]


def group_molecular_mass(standalone_smiles: str | None) -> float | None:
    if not standalone_smiles:
        return None
    mol = Chem.MolFromSmiles(standalone_smiles)
    if mol is None:
        return None
    # Standalone fragment carries one attachment H; remove its mass to obtain
    # the mass of the substituent as bound to the aryl ring.
    return float(Descriptors.MolWt(mol) - 1.007276)


def vdw_volume(atoms: list[tuple[str, float, float, float]]) -> float:
    return float(
        sum((4.0 / 3.0) * math.pi * VDW_RADII.get(symbol, 1.50) ** 3 for symbol, *_ in atoms)
    )


def buried_volume(atoms: list[tuple[str, float, float, float]], radius: float = 3.5) -> float:
    """Calculate the BV descriptor used in the model feature set.

    The definition follows the descriptor-generation protocol: 10,000 random
    points are sampled reproducibly in a sphere of radius 3.5 A around the
    coordinate origin and counted as occupied when within 1.5 A of an atom.
    """
    coords = np.asarray([[x, y, z] for _, x, y, z in atoms], dtype=float)
    rng = np.random.RandomState(42)
    points = rng.uniform(-radius, radius, (10000, 3))
    points = points[np.linalg.norm(points, axis=1) <= radius]
    if len(points) == 0:
        return float("nan")
    occupied = 0
    for point in points:
        if np.any(np.linalg.norm(coords - point, axis=1) < 1.5):
            occupied += 1
    return float(occupied / len(points) * 100.0)


def sub_id(key: str) -> str:
    return "sub_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--orca", help="Path to ORCA executable. Otherwise ORCA_EXE/PATH is used.")
    parser.add_argument("--rerun", action="store_true", help="Re-run existing successful substituent PM3 jobs.")
    parser.add_argument("--seed", type=int, default=42, help="RDKit embedding seed for capped substituents.")
    parser.add_argument(
        "--limit-substituents",
        type=int,
        default=None,
        help="Optional test limit for the number of unique substituents to calculate.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Extract substituents and write capped XYZ files without running ORCA.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.dataset)
    base = dataset_dir(args.dataset)
    out_dir = ensure_dir(base / "07_substituent_features")
    assignment_path = out_dir / "substitution_assignment.xlsx"
    feature_path = out_dir / "substituent_features.xlsx"

    assignment_rows = []
    qc_rows = []
    unique_subs: dict[str, str | None] = {}

    for _, row in manifest.iterrows():
        mol_id = str(row["MoleculeID"])
        subs, qc = extract_substituents(str(row["SMILES"]))
        qc_rows.append({"MoleculeID": mol_id, **qc})
        for sub in subs:
            assignment_rows.append({"MoleculeID": mol_id, **sub})
            unique_subs[sub["SubstituentKey"]] = sub["SubstituentSMILES"]

    assignments = pd.DataFrame(
        assignment_rows,
        columns=["MoleculeID", "Position", "SubstituentKey", "SubstituentSMILES"],
    )
    write_table(assignments, assignment_path)
    write_table(pd.DataFrame(qc_rows), out_dir / "substitution_qc.csv")

    keys = sorted(unique_subs)
    if args.limit_substituents is not None:
        keys = keys[: max(0, args.limit_substituents)]

    cache = ensure_dir(dataset_dir("_substituents"))
    capped_dir = ensure_dir(cache / "01_capped_xyz")
    pm3_dir = ensure_dir(cache / "02_pm3")
    clean_dir = ensure_dir(cache / "03_clean_xyz")
    library_path = cache / "substituent_descriptor_library.xlsx"

    if library_path.exists():
        descriptor_df = pd.read_excel(library_path)
    else:
        descriptor_df = pd.DataFrame(
            columns=[
                "SubstituentID",
                "SubstituentKey",
                "SubstituentSMILES",
                "Molmass",
                "Van der Waals Volume (Å³)",
                "Buried Volume (%)",
                "UFF_status",
                "PM3_status",
            ]
        )
    existing = {
        str(row["SubstituentKey"]): row.to_dict()
        for _, row in descriptor_df.iterrows()
        if pd.notna(row.get("SubstituentKey"))
    }

    orca = None if args.prepare_only else locate_orca(args.orca)
    updated = dict(existing)

    for i, key in enumerate(keys, start=1):
        sid = sub_id(key)
        standalone = unique_subs.get(key)
        previous = existing.get(key)
        if previous and not args.rerun and str(previous.get("PM3_status")) == "success":
            print(f"[{i}/{len(keys)}] {sid}: existing successful descriptor, skipped")
            continue

        try:
            capped, sub_indices, uff_status = build_phenyl_capped_model(key, seed=args.seed)
            xyz_atoms = mol_to_xyz_atoms(capped)
            xyz_path = capped_dir / f"{sid}.xyz"
            write_xyz(xyz_atoms, xyz_path, comment=f"Phenyl-capped substituent {key}")
            (capped_dir / f"{sid}.json").write_text(
                json.dumps(
                    {
                        "SubstituentKey": key,
                        "SubstituentSMILES": standalone,
                        "SubstituentAtomIndices": sub_indices,
                        "UFF_status": uff_status,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            updated[key] = {
                "SubstituentID": sid,
                "SubstituentKey": key,
                "SubstituentSMILES": standalone,
                "Molmass": group_molecular_mass(standalone),
                "Van der Waals Volume (Å³)": np.nan,
                "Buried Volume (%)": np.nan,
                "UFF_status": "failed",
                "PM3_status": f"prepare_failed: {exc}",
            }
            print(f"[{i}/{len(keys)}] {sid}: preparation failed: {exc}")
            continue

        if args.prepare_only:
            updated[key] = {
                "SubstituentID": sid,
                "SubstituentKey": key,
                "SubstituentSMILES": standalone,
                "Molmass": group_molecular_mass(standalone),
                "Van der Waals Volume (Å³)": np.nan,
                "Buried Volume (%)": np.nan,
                "UFF_status": uff_status,
                "PM3_status": "not_run",
            }
            continue

        job_dir = ensure_dir(pm3_dir / sid)
        job_xyz = job_dir / "input.xyz"
        write_xyz(xyz_atoms, job_xyz, comment=f"Phenyl-capped substituent {key}")
        inp = job_dir / "job.inp"
        out = job_dir / "job.out"
        inp.write_text(
            "! PM3 Opt\n"
            "* xyzfile 0 1 input.xyz\n",
            encoding="utf-8",
        )

        print(f"[{i}/{len(keys)}] PM3 substituent: {sid}")
        rc = run_orca_job(orca_exe=orca, input_path=inp, output_path=out, cwd=job_dir)
        success = (
            rc == 0
            and out.exists()
            and orca_terminated_normally(out)
            and optimization_done(out)
        )
        if not success:
            updated[key] = {
                "SubstituentID": sid,
                "SubstituentKey": key,
                "SubstituentSMILES": standalone,
                "Molmass": group_molecular_mass(standalone),
                "Van der Waals Volume (Å³)": np.nan,
                "Buried Volume (%)": np.nan,
                "UFF_status": uff_status,
                "PM3_status": "failed",
            }
            continue

        final_atoms = extract_last_orca_geometry(out)
        if final_atoms is None or max(sub_indices) >= len(final_atoms):
            updated[key] = {
                "SubstituentID": sid,
                "SubstituentKey": key,
                "SubstituentSMILES": standalone,
                "Molmass": group_molecular_mass(standalone),
                "Van der Waals Volume (Å³)": np.nan,
                "Buried Volume (%)": np.nan,
                "UFF_status": uff_status,
                "PM3_status": "geometry_extraction_failed",
            }
            continue

        clean_atoms = [final_atoms[j] for j in sub_indices]
        write_xyz(clean_atoms, clean_dir / f"{sid}.xyz", comment=f"Substituent {key} after PM3")
        updated[key] = {
            "SubstituentID": sid,
            "SubstituentKey": key,
            "SubstituentSMILES": standalone,
            "Molmass": group_molecular_mass(standalone),
            "Van der Waals Volume (Å³)": vdw_volume(clean_atoms),
            "Buried Volume (%)": buried_volume(clean_atoms),
            "UFF_status": uff_status,
            "PM3_status": "success",
        }

    descriptor_df = pd.DataFrame(updated.values())
    if not descriptor_df.empty:
        descriptor_df = descriptor_df.sort_values("SubstituentID").reset_index(drop=True)
    write_table(descriptor_df, library_path)
    used_df = descriptor_df[descriptor_df["SubstituentKey"].astype(str).isin(unique_subs.keys())].copy() if not descriptor_df.empty else descriptor_df
    write_table(used_df, out_dir / "substituent_descriptor_library_used.xlsx")

    lookup = descriptor_df.set_index("SubstituentKey").to_dict(orient="index") if not descriptor_df.empty else {}
    assignments_by_mol = {
        mol_id: part.to_dict(orient="records")
        for mol_id, part in assignments.groupby("MoleculeID")
    } if not assignments.empty else {}

    feature_rows = []
    for _, row in manifest.iterrows():
        mol_id = str(row["MoleculeID"])
        entry = {"MoleculeID": mol_id}
        for pos in range(2, 7):
            entry[f"{pos}_binary"] = 0
            entry[f"Molmasse_{pos}"] = H_MASS
            entry[f"VdW_{pos}"] = H_VDW
            entry[f"BV_{pos}"] = H_BV

        for sub in assignments_by_mol.get(mol_id, []):
            pos = int(sub["Position"])
            if not 2 <= pos <= 6:
                continue
            entry[f"{pos}_binary"] = 1
            desc = lookup.get(str(sub["SubstituentKey"]))
            if desc is None or str(desc.get("PM3_status")) != "success":
                # Do not silently replace a failed/missing substituent calculation by H.
                entry[f"Molmasse_{pos}"] = group_molecular_mass(sub.get("SubstituentSMILES"))
                entry[f"VdW_{pos}"] = np.nan
                entry[f"BV_{pos}"] = np.nan
            else:
                entry[f"Molmasse_{pos}"] = desc.get("Molmass")
                entry[f"VdW_{pos}"] = desc.get("Van der Waals Volume (Å³)")
                entry[f"BV_{pos}"] = desc.get("Buried Volume (%)")
        feature_rows.append(entry)

    features = pd.DataFrame(feature_rows)
    write_table(features, feature_path)
    print(f"Substituent features -> {feature_path}")
    print(f"Unique substituents identified: {len(unique_subs)}")
    if args.prepare_only:
        print("Prepare-only mode: PM3-derived VdW/BV values were not calculated.")


if __name__ == "__main__":
    main()
