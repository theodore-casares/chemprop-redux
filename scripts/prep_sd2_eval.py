"""Prepare SD2 (Drug Repurposing Hub) for prediction.

Reads data/SD2_raw_prediction_set.csv, canonicalizes SMILES, dedupes,
strips any molecule that appears in SD1 (prevents train/eval leakage),
writes data/sd2_eval.csv with a single `smiles` column.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "SD2_raw_prediction_set.csv"
SD1 = ROOT / "data" / "SD1_training_set.csv"
DST = ROOT / "data" / "sd2_eval.csv"


def canonicalize(smi: str) -> str | None:
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi.strip())
    return Chem.MolToSmiles(m) if m else None


def main() -> None:
    print(f"reading {SRC.name}")
    df = pd.read_csv(SRC, skiprows=1, dtype={"SMILES": "string"})
    print(f"  {len(df):,} rows")

    df = df.dropna(subset=["SMILES"])
    df["smiles"] = df["SMILES"].map(canonicalize)
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
    print(f"wrote {DST} ({len(df):,} mols)")


if __name__ == "__main__":
    main()
