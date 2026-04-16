"""Run a trained MoleculeModel on a MoleculeDataLoader."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import MoleculeModel


def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler=None,
) -> List[List[float]]:
    model.eval()
    preds: List[List[float]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
            batch: MoleculeDataset
            mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = (
                batch.batch_graph(),
                batch.features(),
                batch.atom_descriptors(),
                batch.atom_features(),
                batch.bond_features(),
            )
            out = model(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
            out = out.cpu().numpy()
            if scaler is not None:
                out = scaler.inverse_transform(out)
            preds.extend(out.tolist())

    return preds
