# Chemistry Filter — Post-hoc Triage for Priority Lists

Context: top COCONUT / SD2 priority hits mix OOD artifacts (diazonium) and PAINS-like shortcuts (organometallics). See `SANITY_CHECK.md`. A chemistry filter is needed before any of these lists is treated as a real shortlist.

Three tiers, from least to most aggressive.

## Tier 1 — Minimal

Recommended starting point.

- **RDKit `FilterCatalog`** with PAINS_A + PAINS_B + PAINS_C (Baell & Holloway 2010). Catches diazonium, rhodanines, quinones, reactive Michael acceptors, toxoflavins.
- **Organometallic blacklist** — reject any molecule containing atoms in: `{Ti, Zn, Sn, Hg, Pb, As, Cd, Ag, Pt, Ru, Os, Fe, Cu, Ni}`.
- **Structural hygiene** — drop radicals, isolated explicit-H atoms, mols with >1 disconnected component.

## Tier 2 — Standard (recommended, paper-analog)

Tier 1 plus:

- **Brenk filter** (~105 structural alerts: aldehydes, epoxides, aziridines, peroxides, reactive phosphorus, thiocarbonyls, β-carbonyl quaternary N).
- **NIH filter** (~105 FDA-flagged alerts: reactive leaving groups, azo, nitroaromatic in specific contexts).
- **QED ≥ 0.3** — Bickerton composite drug-likeness. For COCONUT, column already present in source CSV; for SD2 compute via `rdkit.Chem.QED.qed`.

Closest automated analog to the paper's manual curation step. Expect ~60-80% reduction in priority counts.

## Tier 3 — Aggressive

Tier 2 plus:

- **ZINC** `not-for-drug` filter set.
- **Property bounds** (beyond Lipinski-1 already applied to COCONUT):
  - MW 150–500
  - LogP 0–5
  - rotatable bonds ≤ 10
  - HBA ≤ 10, HBD ≤ 5
- **Additional rejects**: >1 metal-like atom, >2 aromatic S/Se, >1 halogenated methyl group, any perhaloalkyl chain length >3.

Use only when downstream assay throughput is tight and over-filtering is preferable to false positives.

## Recommendation

**Tier 2** — PAINS + Brenk + NIH + QED ≥ 0.3 + organometallic blacklist. Matches the paper's described manual curation most closely while remaining reproducible.

## Implementation

Single script: `scripts/filter_priority.py`.

- Input: any `*_priority.csv` (cols: `smiles, pred, max_tan_to_sd1_actives`).
- Output: `*_priority_filtered.csv` with added columns `pains_flag, brenk_flag, nih_flag, qed, n_metals, passed`.
- `--tier {1,2,3}` flag.

Re-run on both `data/sd2_priority.csv` and `data/coconut_priority.csv` to produce triaged shortlists for downstream analysis.
