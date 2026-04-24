"""Score a SMILES CSV with a trained MoleculeModel checkpoint or an ensemble.

Single model:
  predict.py --checkpoint runs/sd1/model.pt --test_path X.csv --preds_path Y.csv

Ensemble (mean + std over all model.pt under a directory, recursive):
  predict.py --checkpoint_dir runs/sd1_ens --test_path X.csv --preds_path Y.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDatapoint,
    MoleculeDataset,
    StandardScaler,
)
from chemprop.models import MoleculeModel
from chemprop.train import predict


_GEN = None


def _init_worker() -> None:
    global _GEN
    from chemprop.features import get_features_generator
    _GEN = get_features_generator("rdkit_2d_normalized")


def _featurize(smi: str) -> np.ndarray:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return np.zeros(200, dtype=np.float64)
    feats = np.asarray(_GEN(mol), dtype=np.float64)
    return np.where(np.isnan(feats), 0.0, feats)


def load_and_featurize(path: Path, smiles_col: str, n_workers: int) -> list[MoleculeDatapoint]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    smiles = [r[smiles_col] for r in rows]

    if n_workers <= 1:
        _init_worker()
        feats = [_featurize(s) for s in tqdm(smiles, desc="featurize")]
    else:
        with Pool(n_workers, initializer=_init_worker) as pool:
            feats = list(tqdm(pool.imap(_featurize, smiles, chunksize=64),
                              total=len(smiles), desc="featurize"))

    return [MoleculeDatapoint(smiles=[s], features=f) for s, f in zip(smiles, feats)]


def load_scaler(ckpt: dict) -> StandardScaler | None:
    fs = ckpt.get("feature_scaler")
    if fs and fs.get("means") is not None:
        return StandardScaler(
            means=np.asarray(fs["means"]),
            stds=np.asarray(fs["stds"]),
        )
    return None


def run_one(ckpt_path: Path, ds: MoleculeDataset, batch_size: int, num_workers: int) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = TrainArgs()
    args.from_dict(ckpt["args"], skip_unsettable=True)
    args.__post_init__()

    scaler = load_scaler(ckpt)
    # normalize_features works from raw_features each call, so it is safe to
    # re-apply with a different scaler for each ensemble member on the same
    # dataset instance — no re-featurization needed.
    if scaler is not None:
        ds.normalize_features(scaler)

    loader = MoleculeDataLoader(dataset=ds, batch_size=batch_size, num_workers=num_workers)

    model = MoleculeModel(args).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    preds = predict(model=model, data_loader=loader)
    return np.asarray(preds, dtype=float)[:, 0]


def collect_ckpts(root: Path) -> list[Path]:
    paths = sorted(root.rglob("model.pt"))
    if not paths:
        raise SystemExit(f"no model.pt found under {root}")
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="path to a single model.pt")
    g.add_argument("--checkpoint_dir", help="directory containing model.pt files (recursive)")
    p.add_argument("--test_path", required=True)
    p.add_argument("--preds_path", required=True)
    p.add_argument("--smiles_column", default="smiles")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--featurize_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                   help="parallel RDKit featurization workers (1 = serial)")
    cli = p.parse_args()

    if cli.checkpoint:
        ckpts = [Path(cli.checkpoint)]
    else:
        ckpts = collect_ckpts(Path(cli.checkpoint_dir))
    print(f"ensemble size: {len(ckpts)}")
    for c in ckpts:
        print(f"  {c}")

    print(f"\nloading {cli.test_path}  (featurize_workers={cli.featurize_workers})")
    data = load_and_featurize(Path(cli.test_path), cli.smiles_column, cli.featurize_workers)
    print(f"  {len(data):,} datapoints")
    ds = MoleculeDataset(data)

    all_preds = np.zeros((len(data), len(ckpts)), dtype=float)
    for i, cp in enumerate(ckpts):
        print(f"\n[{i + 1}/{len(ckpts)}] scoring with {cp}")
        all_preds[:, i] = run_one(cp, ds, cli.batch_size, cli.num_workers)

    mean_pred = all_preds.mean(axis=1)
    std_pred = all_preds.std(axis=1)

    out = Path(cli.preds_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        w = csv.writer(f)
        if len(ckpts) == 1:
            w.writerow([cli.smiles_column, "pred"])
            for dp, pr in zip(data, mean_pred):
                w.writerow([dp.smiles[0], pr])
        else:
            w.writerow([cli.smiles_column, "pred", "pred_std", "n_models"])
            for dp, pr, sd in zip(data, mean_pred, std_pred):
                w.writerow([dp.smiles[0], pr, sd, len(ckpts)])
    print(f"\nwrote {out} ({len(mean_pred):,} preds)")

    arr = mean_pred
    pct = np.percentile(arr, [1, 25, 50, 75, 90, 95, 99])
    bar = "─" * 42
    print(f"\n{bar}\n prediction score distribution (mean)\n{bar}")
    print(f"  n        : {len(arr):>10,}")
    print(f"  mean     : {arr.mean():>10.4f}")
    print(f"  std      : {arr.std():>10.4f}")
    print(f"  min/max  : {arr.min():>10.4f} / {arr.max():.4f}")
    print(f"  p1/25/50/75/90/95/99: {pct[0]:.3f} / {pct[1]:.3f} / {pct[2]:.3f} / {pct[3]:.3f} / {pct[4]:.3f} / {pct[5]:.3f} / {pct[6]:.3f}")
    for thr in [0.1, 0.2, 0.5, 0.9]:
        n_hit = int((arr >= thr).sum())
        print(f"  score ≥ {thr:>3}: {n_hit:>10,} ({100 * n_hit / len(arr):.2f}%)")
    if len(ckpts) > 1:
        print(f"  per-mol std — mean: {std_pred.mean():.4f}  p95: {np.percentile(std_pred, 95):.4f}  max: {std_pred.max():.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=60, edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(0.2, color="red", ls="--", alpha=0.6, label="0.2 (paper thresh)")
    ax.axvline(0.5, color="orange", ls="--", alpha=0.6, label="0.5")
    ax.set_xlabel("prediction score"); ax.set_ylabel("count (log)")
    title_n = f"{len(ckpts)}-model ensemble mean" if len(ckpts) > 1 else "single model"
    ax.set_title(f"pred distribution  ({len(arr):,} mols, {title_n})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    hist_path = out.with_suffix(".hist.png")
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"wrote {hist_path}")


if __name__ == "__main__":
    main()
