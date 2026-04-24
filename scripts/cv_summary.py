"""Aggregate per-seed test metrics across an ensemble directory.

Reads runs/<root>/seed*/test_scores.csv (and test_preds.csv when available)
and prints mean ± std — paper-style report.

Usage:
  cv_summary.py --ens_dir runs/sd1_ens
  cv_summary.py --ens_dir runs/sd1_ens --out runs/sd1_ens/cv_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


def read_scores(path: Path) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["metric"]] = float(r["value"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ens_dir", required=True)
    p.add_argument("--out", default=None, help="optional CSV sink for per-seed rows")
    cli = p.parse_args()

    root = Path(cli.ens_dir)
    runs = sorted(root.glob("seed*/test_scores.csv"))
    if not runs:
        raise SystemExit(f"no seed*/test_scores.csv under {root}")

    per_seed: list[dict] = []
    for sp in runs:
        scores = read_scores(sp)
        seed = sp.parent.name
        per_seed.append({"seed": seed, **scores})

    metrics = [k for k in per_seed[0] if k != "seed"]
    bar = "─" * 48
    print(f"ensemble size: {len(per_seed)} (from {root})")
    print(f"\n{bar}")
    print(f" {'seed':<10}  " + "  ".join(f"{m:>10}" for m in metrics))
    print(bar)
    for row in per_seed:
        print(f" {row['seed']:<10}  " + "  ".join(f"{row[m]:>10.4f}" for m in metrics))
    print(bar)

    print("\nmean ± std over folds:")
    for m in metrics:
        vals = [row[m] for row in per_seed]
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {m:<10}: {mu:.4f} ± {sd:.4f}  (n={len(vals)})")

    if cli.out:
        out_path = Path(cli.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            w = csv.DictWriter(f, fieldnames=["seed", *metrics])
            w.writeheader()
            w.writerows(per_seed)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
