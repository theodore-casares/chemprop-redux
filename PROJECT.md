# Project Brief

## Paper Overview

Goal: find new antibiotic chemotypes with novel mechanisms of action (low pre-existing resistance).

### Screen
- 7,684 small molecules screened at 50 µM vs *A. baumannii* ATCC 17978 in LB medium.
- Library: 2,341 off-patent drugs + 5,343 synthetic chemicals (Broad Institute).
- Hit cutoff: 1 SD below mean growth → 480 active, 7,204 inactive.

### Model + Prediction
- Ensemble of 10 RDKit-augmented message-passing models.
- Applied to Drug Repurposing Hub (6,680 molecules).
- Threshold: prediction score > 0.2.
- Filter: Tanimoto nearest-neighbor similarity < 0.3 to training actives (force novel chemotypes).
- Result: 240 priority molecules (then lab-tested; not relevant to us).

### Methods (key details)
- SMILES → molecular graph via RDKit. Atoms + bonds extracted.
- Molecule-level features: message-passing learned representation **concatenated with 200 RDKit global features** (fixes MPNN weakness on global properties of larger molecules).
- Split: 80/10/10 train/val/test, random.
- Training: 30 epochs, eval on val each epoch.
- Hyperparameters:
  - message-passing steps = 3
  - hidden size = 300
  - feed-forward layers = 2
  - dropout = 0
- Tanimoto similarity for chemical-space comparison train ↔ predict.

---

## Assignment

Reimplement recent ML conference paper. Not just run it — probe why it works, whether claims hold, where it breaks.

Must satisfy:
1. Evaluate on dataset(s) **not** used in paper.
2. Non-trivial extension / stress test.

### Component A — Claim Verification
Pick ≥2 quantitative claims. Rigorously test them. Writeup section "What the Paper Doesn't Tell You":
- Where results match paper, where they diverge.
- ≥1 stress test: distribution shift, degenerate input, edge case.
- Best explanation for divergence.

### Component B — Hypothesis-Driven Ablations
Before experiments: submit ≥2 written hypotheses about which architecture components are most critical and why (in Intermediate Report, discuss w/ TAs at check-in #2).
Final ablation eval on:
- Do experiments actually test stated hypotheses.
- Honest engagement w/ results against predictions.
- Quality of explanation *why* components matter.

---

## Distilled Plan

- **Reimplement model**: skip 10-model ensemble (compute/time). One model enough for insight.
- **New eval dataset**: paper uses Drug Repurposing Hub. We use marine natural products from COCONUT (https://coconut.naturalproducts.net/) — openly accessible, structurally diverse, drug-relevant.
- **Stress test**: pick 2 paper claims, check if they hold.
- **Hypothesis-driven ablations**: predict 2 critical components up front, test in final report.
- Optional: train on molecules inhibiting different bacteria, eval on COCONUT or Drug Repurposing Hub.

---

## Implementation Details (from chemprop_abaucin repo)

Core: directed message passing neural network (D-MPNN) on molecular graphs.

### Pipeline
`SMILES → RDKit Mol → MolGraph → Message Passing → Readout → FFN → Prediction`

### 1. Graph construction (`features/featurization.py`)
- Atom features (~133 dims): atomic number, degree, formal charge, chirality, H count, hybridization, aromaticity, mass (one-hot).
- Bond features (14 dims): bond type, conjugation, ring membership, stereo.
- Graphs batched into `BatchMolGraph` for efficient computation.

### 2. Message passing (`models/mpn.py` — `MPNEncoder`)
- Directed: messages carried on bonds, not atoms.
- K depth iterations (K = message-passing steps from paper hyperparams above):
  - Aggregate messages from neighboring bonds.
  - Apply learned weight matrix `W_h` + dropout + activation.
- After K rounds: atom hidden states = original atom features combined with aggregated incoming messages via `W_o`.
- Readout: sum atom hidden states per molecule → fixed-size molecular embedding.

### 3. Prediction head (`models/model.py` — `MoleculeModel`)
- MPNN embedding concatenated with global features (see Methods above for the `rdkit_2d_normalized` descriptors used by Abaucin paper).
- FFN → output.
- Supports binary classification (BCE), regression (MSE), multiclass, spectra.

### 4. Training (`train/run_training.py`)
- Optimizer: Adam.
- LR schedule: Noam (warmup + exponential decay).
- Splits: random / scaffold-based / k-fold CV.
- Gradient clipping, target scaling via `StandardScaler`, multi-task w/ missing-target masking.

### Entry points
- `chemprop_train` — trains (supports ensembling/CV).
- `chemprop_predict` — loads checkpoint + scalers, runs inference.

---

## Links
- Chemprop: https://github.com/chemprop/chemprop
- Paper code: https://github.com/GaryLiu152/chemprop_abaucin
- Chemprop website: http://chemprop.csail.mit.edu
- COCONUT: https://coconut.naturalproducts.net/
- Slides: https://docs.google.com/presentation/d/14pbd9LTXzfPSJHyXYkfLxnK8Q80LhVnjImg8a3WqCRM/edit?usp=sharing
