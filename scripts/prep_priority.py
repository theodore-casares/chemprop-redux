"""Filter a preds CSV into a priority set (paper's rule).

Rule (Liu et al. 2023):
    pred >= pred_thresh  AND  max_tanimoto_to_SD1_actives < tan_thresh

Tanimoto against SD1 actives uses Morgan fingerprints r=2, 2048 bits.
SD1 actives are read from sd1_train.csv (rows where active == 1).

Usage:
  prep_priority.py --preds data/sd2_preds.csv --out data/sd2_priority.csv
  prep_priority.py --preds data/coconut_preds_ens.csv \
      --out data/coconut_priority_ens.csv --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


_ACTIVE_FPS: list = []


def mol_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def load_active_fps(sd1_path: Path) -> list:
    fps = []
    with open(sd1_path) as f:
        for r in csv.DictReader(f):
            if int(float(r["active"])) == 1:
                fp = mol_fp(r["smiles"])
                if fp is not None:
                    fps.append(fp)
    if not fps:
        raise SystemExit(f"no active fingerprints in {sd1_path}")
    return fps


def _init_worker(active_fps):
    global _ACTIVE_FPS
    _ACTIVE_FPS = active_fps


def _max_tan(smi: str) -> float:
    fp = mol_fp(smi)
    if fp is None:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(fp, _ACTIVE_FPS)
    return float(max(sims)) if sims else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="preds CSV with columns smiles,pred[,pred_std,n_models]")
    p.add_argument("--out", required=True)
    p.add_argument("--sd1", default="data/sd1_train.csv", help="SD1 binarized training CSV (smiles,active)")
    p.add_argument("--pred_thresh", type=float, default=0.2)
    p.add_argument("--tan_thresh", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=max(1, (__import__('os').cpu_count() or 1) - 1))
    cli = p.parse_args()

    print(f"loading SD1 actives from {cli.sd1}")
    active_fps = load_active_fps(Path(cli.sd1))
    print(f"  {len(active_fps)} active fingerprints")

    # read preds (keep all columns)
    with open(cli.preds) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "smiles" not in fieldnames or "pred" not in fieldnames:
        raise SystemExit(f"{cli.preds} must have smiles,pred columns")
    print(f"loaded {len(rows):,} rows from {cli.preds}")

    # pred filter first (cheap) — shrinks the Tan work
    pred_vals = np.array([float(r["pred"]) for r in rows], dtype=float)
    pred_ok = pred_vals >= cli.pred_thresh
    n_pred_pass = int(pred_ok.sum())
    print(f"  pred >= {cli.pred_thresh}: {n_pred_pass:,} / {len(rows):,}")

    keep_rows = [r for r, ok in zip(rows, pred_ok) if ok]
    smiles_to_score = [r["smiles"] for r in keep_rows]

    print(f"computing max Tanimoto to SD1 actives for {len(smiles_to_score):,} mols  (workers={cli.workers})")
    if cli.workers <= 1:
        _init_worker(active_fps)
        tans = [_max_tan(s) for s in tqdm(smiles_to_score, desc="tanimoto")]
    else:
        with Pool(cli.workers, initializer=_init_worker, initargs=(active_fps,)) as pool:
            tans = list(tqdm(pool.imap(_max_tan, smiles_to_score, chunksize=256),
                             total=len(smiles_to_score), desc="tanimoto"))

    tans_arr = np.asarray(tans, dtype=float)
    tan_ok = tans_arr < cli.tan_thresh
    n_both = int(tan_ok.sum())
    print(f"  tan < {cli.tan_thresh}: {n_both:,} / {len(smiles_to_score):,}")

    out_fields = list(fieldnames)
    if "max_tan_to_sd1_actives" not in out_fields:
        out_fields.append("max_tan_to_sd1_actives")

    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r, t, ok in zip(keep_rows, tans, tan_ok):
            if ok:
                r = {**r, "max_tan_to_sd1_actives": t}
                w.writerow(r)
    print(f"wrote {out_path}  ({n_both:,} priority mols)")


if __name__ == "__main__":
    main()
