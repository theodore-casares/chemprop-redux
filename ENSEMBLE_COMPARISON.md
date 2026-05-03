# Single-Model vs 10-Model Ensemble

Apples-to-apples comparison: the **single model** in `runs/sd1/` (one 80/10/10 random split, seed 0, 30 epochs) vs the **10-model ensemble** in `runs/sd1_ens/` (seeds 0–9, each its own random split, otherwise identical hyperparameters). Same data, same architecture, same training procedure as the paper.

## 1. Held-Out Test Performance

Each ensemble seed has its own held-out split → mean ± std across 10 splits is directly comparable to the paper's 10-fold CV report.

| Metric | Paper (10 models, 10-fold CV) | Ours single (1 split) | Ours ensemble (10 splits) |
|---|---|---|---|
| AUROC | 0.792 ± 0.042 | 0.825 | **0.7895 ± 0.0311** |
| AUPRC | 0.337 ± 0.088 | 0.380 | **0.3302 ± 0.0673** |
| Accuracy | — | 0.939 | 0.9401 ± 0.0070 |

**Read:** the ensemble mean lands almost exactly on the paper's reported numbers, and the spread is *tighter* than the paper's (σ_AUROC 0.031 vs 0.042). The single-model number (0.825 AUROC) was a lucky split — it sits at the top of the per-seed distribution, ~+1σ above the mean.

Per-seed AUROC range: 0.749 (seed 6) to 0.830 (seed 9). Per-seed AUPRC: 0.196 (seed 1) to 0.435 (seed 4). See `runs/sd1_ens/cv_summary.csv`.

## 2. SD2 (Drug Repurposing Hub) — Inference Comparison

Single model in `data/sd2_preds.csv` vs ensemble mean in `data/sd2_preds_ens.csv`. 5,864 molecules each, same input list.

| | Single | Ensemble |
|---|---|---|
| score ≥ 0.2 | 545 | **688** |
| score ≥ 0.5 | — | 115 |
| score ≥ 0.9 | — | 13 |
| Pearson r (single vs ens score) | — | **0.882** |
| Spearman ρ | — | **0.847** |
| mean \|ens − single\| | — | 0.040 |
| p95 \|ens − single\| | — | 0.149 |
| max \|ens − single\| | — | 0.490 |
| Per-mol ensemble σ — mean | — | 0.056 |
| Per-mol ensemble σ — p95 | — | 0.155 |
| Per-mol ensemble σ — max | — | 0.422 |

**Threshold cross-tab (≥ 0.2):** 426 both, 119 single-only, 262 ensemble-only, 5,057 neither.

**Top-K Jaccard (raw scores, single vs ensemble):**

| K | Jaccard |
|---|---|
| 50 | 0.587 |
| 100 | 0.539 |
| 240 | 0.491 |
| 500 | 0.550 |
| 1000 | 0.582 |

Both models agree on roughly half of any top-K cut. The remaining half is genuinely different chemistry being prioritized.

### Priority filter (paper rule: score ≥ 0.2 AND max Tanimoto to SD1 actives < 0.3)

| | Single | Ensemble |
|---|---|---|
| Priority count | **240** (matched paper exactly) | **314** |
| Priority Jaccard (single ∩ ens / single ∪ ens) | — | **0.365** (148 ∩ / 406 ∪) |

**Important correction to FINDINGS.md.** The single-model "exact 240 = paper" match was a *coincidence of the seed-0 split*, not a true reproduction. Of our 240 priority molecules, only 148 (62%) survive the ensemble filter. Of the ensemble's 314, only 148 (47%) were also flagged by the single model. The paper's 240 number is not stable to the choice of model instantiation.

## 3. COCONUT — Novel-Chemistry Inference Comparison

*Pending — ensemble inference on 502,874 mols still running. Will fill once `data/coconut_preds_ens.csv` is written.*

Planned report: same metrics as §2 (Pearson/Spearman, threshold cross-tab, top-K Jaccard, priority Jaccard, ensemble-σ distribution + high-σ "uncertain" mols).

## 4. What the ensemble buys

1. **Calibrated mean.** Single-split AUROC can land anywhere in [0.75, 0.83]; 10-model mean is centered on 0.79 with tight σ. Ensemble removes split-luck from the headline number.
2. **Per-molecule uncertainty (σ).** Single model emits a bare score with no uncertainty; ensemble σ flags disagreement between models. SD2 mean σ = 0.056 with a long tail (max 0.42) — highest-σ mols are the ones any single-model decision is most likely to flip on.
3. **Different priority set.** ~half of the single-model's top picks are not the ensemble's top picks. For a virtual-screen handoff to a wet lab, the difference is real (148/240 = 62% recall).

## 5. What it does not buy

1. **Shape of the score distribution is similar.** Score ranks are very similar (Spearman 0.85 on SD2). For exploratory analysis where rank order matters more than absolute calibration, single model is most of the way there.
2. **No new actives discovered "for free."** Ensemble surfaces *more* hits at score ≥ 0.2 (688 vs 545) but the extras may be noise — they did not survive any held-out validation in this work.

## 6. Reproduction commands

```
# train ensemble (10 × ~4 min on M5 MPS)
bash scripts/train_ensemble.sh

# CV-style mean ± std over per-seed test scores
.venv/bin/python scripts/cv_summary.py --ens_dir runs/sd1_ens --out runs/sd1_ens/cv_summary.csv

# ensemble inference (mean + per-mol std)
.venv/bin/python scripts/predict.py --checkpoint_dir runs/sd1_ens \
  --test_path data/sd2_eval.csv --preds_path data/sd2_preds_ens.csv

# priority filter on ensemble preds
.venv/bin/python scripts/prep_priority.py \
  --preds data/sd2_preds_ens.csv --out data/sd2_priority_ens.csv

# single vs ensemble comparison
.venv/bin/python scripts/compare_mono_vs_ens.py \
  --mono data/sd2_preds.csv --ens data/sd2_preds_ens.csv \
  --priority_mono data/sd2_priority.csv --priority_ens data/sd2_priority_ens.csv \
  --out_json runs/sd1_ens/sd2_compare.json \
  --out_scatter runs/sd1_ens/sd2_compare_scatter.png --label SD2
```

## 7. Artifacts

- `runs/sd1_ens/seed{0..9}/model.pt` — 10 ensemble checkpoints
- `runs/sd1_ens/cv_summary.csv` — per-seed + mean ± std
- `runs/sd1_ens/sd2_compare.json` — structured SD2 comparison
- `runs/sd1_ens/sd2_compare_scatter.png` — single-vs-ensemble scatter
- `data/sd2_preds_ens.csv`, `data/sd2_priority_ens.csv` — ensemble SD2 outputs
- `data/coconut_preds_ens.csv`, `data/coconut_priority_ens.csv` — *pending*
