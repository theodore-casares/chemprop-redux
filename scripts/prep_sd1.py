"""Binarize SD1 growth-inhibition screen into active/inactive labels.

Per PROJECT.md: hit cutoff = 1 SD below mean growth. Lower Mean value =
stronger growth inhibition, so the threshold is `mu - sigma`. Mean below
threshold → active (1), otherwise inactive (0).

Output: data/sd1_train.csv with columns [smiles, active].
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "SD1_training_set.csv"
DST = ROOT / "data" / "sd1_train.csv"


def canonicalize(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    m = Chem.MolFromSmiles(s.strip())
    return Chem.MolToSmiles(m) if m else None


def main() -> None:
    print(f"reading {SRC.name}")
    df = pd.read_csv(SRC, skiprows=1)
    print(f"  {len(df):,} rows, cols: {list(df.columns)}")

    df = df.dropna(subset=["SMILES", "Mean"])
    df["smiles"] = df["SMILES"].map(canonicalize)
    df = df.dropna(subset=["smiles"])
    print(f"  {len(df):,} after parse")

    mu = df["Mean"].mean()
    sigma = df["Mean"].std()
    cutoff = mu - sigma
    print(f"  Mean  μ={mu:.4f}  σ={sigma:.4f}  cutoff (μ−σ)={cutoff:.4f}")

    df["active"] = (df["Mean"] < cutoff).astype(int)
    n_act = int(df["active"].sum())
    print(f"  actives: {n_act:,}  inactives: {len(df) - n_act:,}  rate: {n_act / len(df):.3%}")

    df = df.drop_duplicates("smiles")
    print(f"  {len(df):,} after dedupe")

    df[["smiles", "active"]].to_csv(DST, index=False)
    print(f"wrote {DST} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
