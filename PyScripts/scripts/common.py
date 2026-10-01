#!/usr/bin/env python3
"""Shared helpers for the boronic-acid catalyst workflow.

The public scripts in this repository intentionally avoid machine-specific paths.
Dataset-specific intermediate files live under ``work/<dataset_name>/``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

HARTREE_TO_KJMOL = 2625.49962
HARTREE_TO_EV = 27.211386245988

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
ENERGY_RE = re.compile(rf"FINAL SINGLE POINT ENERGY\s+({_FLOAT})")
COORD_RE = re.compile(
    rf"^\s*(?:\d+\s+)?([A-Z][a-z]?)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})(?:\s+.*)?$"
)


def project_root() -> Path:
    """Return repository root assuming this file lives in ``scripts/``."""
    return Path(__file__).resolve().parents[1]


def dataset_dir(dataset: str, root: Path | None = None) -> Path:
    root = root or project_root()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset.strip())
    if not safe:
        raise ValueError("Dataset name must not be empty.")
    return root / "work" / safe


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python")
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(path, index=index)
    elif suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=index)
    else:
        raise ValueError(f"Unsupported output table format: {path}")


def dump_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_manifest(dataset: str, root: Path | None = None) -> pd.DataFrame:
    path = dataset_dir(dataset, root) / "00_manifest" / "manifest.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Run 02_prepare_dataset.py for this dataset first."
        )
    df = pd.read_excel(path)
    required = {"MoleculeID", "SMILES"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return df


def locate_orca(explicit: str | None = None) -> Path:
    """Resolve ORCA executable from --orca, ORCA_EXE, or PATH."""
    candidates = [explicit, os.environ.get("ORCA_EXE"), shutil.which("orca")]
    for candidate in candidates:
        if candidate:
            p = Path(candidate).expanduser().resolve()
            if p.exists() and os.access(p, os.X_OK):
                return p
    raise FileNotFoundError(
        "Could not find the ORCA executable. Provide --orca /path/to/orca, "
        "set ORCA_EXE, or add ORCA to PATH."
    )


def orca_version(orca_exe: Path) -> str | None:
    """Best-effort extraction of ORCA version from executable output."""
    try:
        proc = subprocess.run(
            [str(orca_exe)], capture_output=True, text=True, timeout=20, check=False
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"Program Version\s+([^\s]+)", text)
        if m:
            return m.group(1)
        m = re.search(r"ORCA\s+(?:VERSION|Version)\s*[:=]?\s*([^\s]+)", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def read_xyz(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ file: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in XYZ file: {path}") from exc
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ coordinate line in {path}: {line!r}")
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if len(atoms) != n_atoms:
        raise ValueError(f"XYZ atom count mismatch in {path}: expected {n_atoms}, got {len(atoms)}")
    return atoms


def write_xyz(
    atoms: Sequence[tuple[str, float, float, float]], path: Path, comment: str = ""
) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n{comment}\n")
        for symbol, x, y, z in atoms:
            handle.write(f"{symbol:<3s} {x: .10f} {y: .10f} {z: .10f}\n")


def xyz_block(atoms: Sequence[tuple[str, float, float, float]]) -> str:
    return "\n".join(
        f"{symbol:<3s} {x: .10f} {y: .10f} {z: .10f}"
        for symbol, x, y, z in atoms
    )


def orca_terminated_normally(text_or_path: str | Path) -> bool:
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = text_or_path
    return "ORCA TERMINATED NORMALLY" in text


def optimization_done(text_or_path: str | Path) -> bool:
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = text_or_path
    return "OPTIMIZATION RUN DONE" in text


def last_single_point_energy(text_or_path: str | Path) -> float | None:
    """Return the *last* FINAL SINGLE POINT ENERGY in an ORCA output."""
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = text_or_path
    matches = ENERGY_RE.findall(text)
    return float(matches[-1]) if matches else None


def extract_last_orca_geometry(text_or_path: str | Path):
    """Extract the final Cartesian geometry (Å) printed by ORCA.

    Returns a list of ``(element, x, y, z)`` tuples or ``None``.
    """
    if isinstance(text_or_path, Path):
        lines = text_or_path.read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        lines = text_or_path.splitlines()

    starts = [i for i, line in enumerate(lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in line]
    if not starts:
        return None

    for start in reversed(starts):
        atoms: list[tuple[str, float, float, float]] = []
        started = False
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped or set(stripped) <= {"-"}:
                if started and not stripped:
                    break
                continue
            m = COORD_RE.match(line)
            if m:
                started = True
                atoms.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
            elif started:
                break
        if atoms:
            return atoms
    return None


def run_orca_job(
    *,
    orca_exe: Path,
    input_path: Path,
    output_path: Path,
    cwd: Path,
) -> int:
    """Run an ORCA input safely without shell redirection."""
    ensure_dir(cwd)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as out_handle:
        proc = subprocess.run(
            [str(orca_exe), input_path.name],
            cwd=cwd,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return proc.returncode


def exclude_fukui_columns(columns: Iterable[str]) -> list[str]:
    """Remove legacy Fukui-like columns from a list of feature names."""
    blocked = {
        "fukui+",
        "fukui-",
        "fukui_plus",
        "fukui_minus",
        "fukui plus",
        "fukui minus",
    }
    return [c for c in columns if c.strip().lower() not in blocked]
