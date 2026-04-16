"""RDKit helpers shared by data loaders."""

from __future__ import annotations

from rdkit import Chem


def make_mol(smi: str, keep_h: bool = False, add_h: bool = False) -> Chem.Mol | None:
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
    mol = Chem.MolFromSmiles(smi, params)
    if mol is not None and add_h:
        mol = Chem.AddHs(mol)
    return mol
