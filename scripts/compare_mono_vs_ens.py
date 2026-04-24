"""Mono vs ensemble comparison on a prediction set.

Inputs:
  --mono   : single-model preds CSV (smiles, pred)
  --ens    : ensemble preds CSV (smiles, pred, pred_std, n_models)
  --priority_mono / --priority_ens : optional priority CSVs (post Tan filter)
  --out_json : structured summary dump
  --out_scatter : optional PNG scatter of mono vs ens score

Reports:
  - score correlation (Pearson, Spearman)
  - mean abs diff, p95 abs diff
  - top-K (50, 100, 240) Jaccard overlap on raw scores
  - priority-set Jaccard (if both priority files given)
  - ensemble per-mol std summary
  - count cross-tab on >=0.2 threshold
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_preds(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["smiles"]] = {k: float(v) if k != "smiles" else v for k, v in r.items() if k}
    return out


def read_priority_smiles(path: Path) -> set[str]:
    with open(path) as f:
        return {r["smiles"] for r in csv.DictReader(f)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def topk(scores: np.ndarray, smiles: list[str], k: int) -> set[str]:
    if k > len(scores):
        return set(smiles)
    idx = np.argpartition(-scores, k - 1)[:k]
    return {smiles[i] for i in idx}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mono", required=True)
    p.add_argument("--ens", required=True)
    p.add_argument("--priority_mono", default=None)
    p.add_argument("--priority_ens", default=None)
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_scatter", default=None)
    p.add_argument("--label", default="dataset")
    cli = p.parse_args()

    mono = read_preds(Path(cli.mono))
    ens = read_preds(Path(cli.ens))
    common = sorted(set(mono) & set(ens))
    if not common:
        raise SystemExit("no overlapping smiles between mono and ens")
    print(f"[{cli.label}] mono={len(mono):,}  ens={len(ens):,}  common={len(common):,}")

    smiles = common
    m = np.array([mono[s]["pred"] for s in smiles])
    e = np.array([ens[s]["pred"] for s in smiles])
    sd = np.array([ens[s].get("pred_std", float("nan")) for s in smiles])

    pr = pearsonr(m, e)
    sp = spearmanr(m, e)
    diff = e - m
    abs_diff = np.abs(diff)

    print(f"  Pearson r       : {pr.statistic:.4f}  (p={pr.pvalue:.2e})")
    print(f"  Spearman ρ      : {sp.statistic:.4f}  (p={sp.pvalue:.2e})")
    print(f"  mean |ens-mono| : {abs_diff.mean():.4f}")
    print(f"  p95 |ens-mono|  : {np.percentile(abs_diff, 95):.4f}")
    print(f"  max |ens-mono|  : {abs_diff.max():.4f}")
    print(f"  ens std — mean  : {np.nanmean(sd):.4f}  p95: {np.nanpercentile(sd, 95):.4f}  max: {np.nanmax(sd):.4f}")

    print(f"\n  threshold cross-tab (>= 0.2):")
    m_hi = m >= 0.2
    e_hi = e >= 0.2
    print(f"    mono≥0.2 only: {int((m_hi & ~e_hi).sum())}")
    print(f"    ens≥0.2 only : {int((~m_hi & e_hi).sum())}")
    print(f"    both         : {int((m_hi & e_hi).sum())}")
    print(f"    neither      : {int((~m_hi & ~e_hi).sum())}")

    print(f"\n  top-K Jaccard (mono vs ens raw scores):")
    topk_results = {}
    for k in [50, 100, 240, 500, 1000]:
        if k > len(smiles):
            continue
        j = jaccard(topk(m, smiles, k), topk(e, smiles, k))
        topk_results[k] = j
        print(f"    K={k:>5}: {j:.4f}")

    summary = {
        "label": cli.label,
        "n_common": len(common),
        "pearson_r": float(pr.statistic),
        "spearman_rho": float(sp.statistic),
        "mean_abs_diff": float(abs_diff.mean()),
        "p95_abs_diff": float(np.percentile(abs_diff, 95)),
        "max_abs_diff": float(abs_diff.max()),
        "ens_std_mean": float(np.nanmean(sd)),
        "ens_std_p95": float(np.nanpercentile(sd, 95)),
        "ens_std_max": float(np.nanmax(sd)),
        "thr_0.2": {
            "mono_only": int((m_hi & ~e_hi).sum()),
            "ens_only": int((~m_hi & e_hi).sum()),
            "both": int((m_hi & e_hi).sum()),
            "neither": int((~m_hi & ~e_hi).sum()),
        },
        "topk_jaccard": topk_results,
    }

    if cli.priority_mono and cli.priority_ens:
        pm = read_priority_smiles(Path(cli.priority_mono))
        pe = read_priority_smiles(Path(cli.priority_ens))
        j = jaccard(pm, pe)
        print(f"\n  priority-set Jaccard: {j:.4f}  (mono={len(pm)}  ens={len(pe)}  ∩={len(pm & pe)}  ∪={len(pm | pe)})")
        summary["priority"] = {
            "mono_n": len(pm),
            "ens_n": len(pe),
            "intersection": len(pm & pe),
            "union": len(pm | pe),
            "jaccard": j,
        }

    if cli.out_json:
        Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cli.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {cli.out_json}")

    if cli.out_scatter:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(m, e, s=4, alpha=0.3)
        lim = max(m.max(), e.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
        ax.axvline(0.2, color="red", ls=":", alpha=0.5)
        ax.axhline(0.2, color="red", ls=":", alpha=0.5)
        ax.set_xlabel("single-model score")
        ax.set_ylabel("10-ensemble mean score")
        ax.set_title(f"{cli.label}: mono vs ensemble  (Pearson={pr.statistic:.3f}, Spearman={sp.statistic:.3f}, n={len(smiles):,})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        Path(cli.out_scatter).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cli.out_scatter, dpi=120)
        plt.close(fig)
        print(f"wrote {cli.out_scatter}")


if __name__ == "__main__":
    main()
