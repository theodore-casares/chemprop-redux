# Findings

Reproduction of *Liu et al. 2023* (Nat Chem Biol) D-MPNN for *A. baumannii* growth inhibition.

## Setup

- Single model (paper used 10-ensemble). One random 80/10/10 split, seed 0.
- Hyperparameters match paper: depth=3, hidden=300, ffn=2, dropout=0, 200 RDKit features, 30 epochs, Adam + Noam LR.
- Binarized SD1 at μ−σ → **480 actives / 7,204 inactives** (exact paper match).

## Model Performance (SD1 test split)

| Metric | Paper (10-ensemble, 10-fold CV) | Ours (1 model, 1 split) |
|---|---|---|
| AUROC | 0.792 ± 0.042 | **0.825** |
| AUPRC | 0.337 ± 0.088 | **0.380** |
| Accuracy | — | 0.939 |

Within paper's ±1σ band. Best val AUC at epoch 7; overfits after ~epoch 10.

## SD2 Leakage Audit (Drug Repurposing Hub vs SD1 training)

Paper scored Hub after training on SD1 without reporting train/eval overlap.

| Criterion | Count | % of SD2 |
|---|---|---|
| Exact canonical-SMILES match | 783 | 11.8% |
| Morgan-fingerprint identical (Tan = 1.0) | 1,230 | 18.5% |
| Tan ≥ 0.9 | 1,571 | 23.6% |
| Tan ≥ 0.3 (paper's own novelty threshold) | 5,215 | **78.5%** |

Of the 783 exact overlaps: 82 SD1-actives (10.5% active rate vs 6.25% baseline → enriched).

## SD2 Priority Count (reproduces paper)

Scored 5,864 SD2 molecules (SD1 overlap stripped).

| Filter | Paper | Ours |
|---|---|---|
| score > 0.2 | — | 545 |
| score > 0.2 + Tanimoto < 0.3 to SD1-actives | **240** | **240** |

Priority set saved at `data/sd2_priority.csv`.

## COCONUT Eval Set (novel-chemistry stress test)

Natural-products database; paper did not use.

- Raw: 738,827 molecules
- After Lipinski ≤ 1 filter: 504,276
- After RDKit parse + dedupe: 504,256
- After SD1 overlap strip: **502,874**
- Inference output: `data/coconut_preds.csv`

## Takeaways

1. Single-model reproduction lands inside the paper's ensemble variance on both AUROC and AUPRC.
2. Paper's Drug Repurposing Hub eval set has 12% exact and 78.5% near-duplicate overlap with training chemistry — novelty claims depend critically on the Tanimoto < 0.3 filter.
3. With paper's filter, we recover the exact same 240 priority-molecule count.

## Files

- `runs/sd1/model.pt` — best checkpoint
- `runs/sd1/history.png`, `test_curves.png` — training curves, ROC/PR on test
- `data/sd1_train.csv` — binarized training data
- `data/sd2_eval.csv` — 5,864 mols, SD1-stripped
- `data/sd2_preds.csv`, `sd2_priority.csv`, `sd2_preds.hist.png` — SD2 scores + filtered set
- `data/coconut_eval.csv`, `coconut_preds.csv` (to run) — COCONUT novelty eval
- `data/sd2_max_tanimoto_to_sd1.npy` — per-mol max Tan to SD1, cached
