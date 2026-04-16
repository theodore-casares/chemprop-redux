"""Minimal TrainArgs / PredictArgs.

The upstream chemprop library uses Tap for CLI parsing with hundreds of
flags. This re-implementation keeps the hyperparameters PROJECT.md
requires and stubs the rest with sensible defaults so the model/data
code can import them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Optional

import torch


@dataclass
class TrainArgs:
    # --- task ---
    dataset_type: str = "classification"
    num_tasks: int = 1
    multiclass_num_classes: int = 3
    loss_function: str = "binary_cross_entropy"
    target_columns: Optional[List[str]] = None
    smiles_columns: Optional[List[str]] = None

    # --- architecture ---
    hidden_size: int = 300
    depth: int = 3
    dropout: float = 0.0
    activation: str = "ReLU"
    ffn_num_layers: int = 2
    ffn_hidden_size: int = 300
    bias: bool = False
    atom_messages: bool = False
    undirected: bool = False
    aggregation: str = "mean"
    aggregation_norm: float = 100.0
    mpn_shared: bool = False

    # --- reactions / multi-mol (unused here) ---
    number_of_molecules: int = 1
    reaction: bool = False
    reaction_solvent: bool = False
    hidden_size_solvent: Optional[int] = None
    bias_solvent: Optional[bool] = None
    depth_solvent: Optional[int] = None

    # --- extra features (200-d RDKit global) ---
    features_only: bool = False
    use_input_features: bool = True
    features_size: int = 200
    features_generator: Optional[List[str]] = field(default_factory=lambda: ["rdkit_2d_normalized"])
    features_path: Optional[List[str]] = None
    features_scaling: bool = True
    atom_descriptors: Optional[str] = None
    atom_descriptors_size: int = 0
    atom_features_size: int = 0
    bond_features_path: Optional[List[str]] = None
    overwrite_default_atom_features: bool = False
    overwrite_default_bond_features: bool = False

    # --- freeze / transfer (unused) ---
    checkpoint_frzn: Optional[str] = None
    freeze_first_only: bool = False
    frzn_ffn_layers: int = 0

    # --- training loop ---
    epochs: int = 30
    batch_size: int = 50
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    warmup_epochs: float = 2.0
    num_lrs: int = 1
    grad_clip: Optional[float] = None
    log_frequency: int = 10
    target_weights: Optional[List[float]] = None
    evidential_regularization: float = 0.0
    train_data_size: int = 0  # populated after split
    class_balance: bool = False
    num_workers: int = 8
    seed: int = 0

    # --- spectra (unused) ---
    spectra_activation: str = "exp"
    train_class_sizes: Optional[List] = None

    # --- device ---
    cuda: bool = field(default_factory=lambda: torch.cuda.is_available())
    device: torch.device = field(init=False)

    # --- I/O ---
    data_path: Optional[str] = None
    separate_val_path: Optional[str] = None
    separate_test_path: Optional[str] = None
    save_dir: Optional[str] = None
    checkpoint_paths: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.cuda:
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "device"}

    def from_dict(self, d: dict, skip_unsettable: bool = False) -> None:
        valid = {f.name for f in fields(self)}
        for k, v in d.items():
            if k in valid:
                setattr(self, k, v)
            elif not skip_unsettable:
                raise KeyError(k)


@dataclass
class PredictArgs(TrainArgs):
    test_path: Optional[str] = None
    preds_path: Optional[str] = None
    fingerprint_type: str = "MPN"


@dataclass
class FingerprintArgs(PredictArgs):
    pass
