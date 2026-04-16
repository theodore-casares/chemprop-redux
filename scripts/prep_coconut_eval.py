"""Prepare COCONUT CSV for chemprop prediction.

Reads data/coconut_csv_lite-04-2026.csv, keeps drug-like rows (Lipinski
violations <= 1), canonicalizes SMILES with RDKit, dedupes, removes any
molecule that appears in the SD1 training set (prevents train/eval
leakage), writes data/coconut_eval.csv with a single `smiles` column.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "coconut_csv_lite-04-2026.csv"
SD1 = ROOT / "data" / "SD1_training_set.csv"
DST = ROOT / "data" / "coconut_eval.csv"

LIPINSKI_MAX = 1


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def main() -> None:
    print(f"reading {SRC.name}")
    df = pd.read_csv(
        SRC,
        usecols=["canonical_smiles", "lipinski_rule_of_five_violations"],
        dtype={"canonical_smiles": "string"},
    )
    n0 = len(df)
    print(f"  {n0:,} rows")

    df = df.dropna(subset=["canonical_smiles"])
    print(f"  {len(df):,} after drop NaN SMILES")

    df = df[df["lipinski_rule_of_five_violations"].fillna(99) <= LIPINSKI_MAX]
    print(f"  {len(df):,} after Lipinski <= {LIPINSKI_MAX}")

    print("canonicalizing...")
    df["smiles"] = df["canonical_smiles"].map(canonicalize)
    df = df.dropna(subset=["smiles"])
    print(f"  {len(df):,} after RDKit parse")

    df = df.drop_duplicates("smiles")
    print(f"  {len(df):,} after dedupe")

    print(f"reading {SD1.name} to remove training-set overlap")
    sd1 = pd.read_csv(SD1, skiprows=1, usecols=["SMILES"], dtype={"SMILES": "string"})
    sd1_canon = {c for c in sd1["SMILES"].map(canonicalize) if c is not None}
    print(f"  {len(sd1_canon):,} SD1 canonical SMILES")
    before = len(df)
    df = df[~df["smiles"].isin(sd1_canon)]
    print(f"  {len(df):,} after SD1 overlap removal (dropped {before - len(df):,})")

    df[["smiles"]].to_csv(DST, index=False)
    print(f"wrote {DST} ({len(df):,} mols, {DST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
