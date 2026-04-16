# Sanity Check — Top COCONUT Hits

COCONUT virtual-screen top-10 dominated by diazonium `[N+]#N` compounds. Asked: legitimate learned feature or spurious?

## Reactive-group enrichment in SD1 training vs COCONUT priority set

| Group | SD1 actives (n=480) | SD1 inactives (n=7,204) | Enrich (act/inact) | COCONUT priority (n=13,015) |
|---|---|---|---|---|
| diazonium `[N+]#N` | 0 (0.00%) | 0 (0.00%) | — | **37 (0.28%)** |
| azide | 3 (0.62%) | 26 (0.36%) | 1.73× | 0 |
| nitro | 10 (2.08%) | 93 (1.29%) | 1.61× | 38 (0.29%) |
| isocyanide | 0 | 0 | — | 5 (0.04%) |
| organometal (Ti/Zn/Sn/Hg) | 4 (0.83%) | 3 (0.04%) | **20.0×** | 10 (0.08%) |
| α,β-unsat carbonyl (Michael) | 6 (1.25%) | 151 (2.10%) | 0.60× | 431 (3.31%) |

## Findings

1. **Diazonium is out-of-distribution, not learned.** Zero across all 7,684 SD1 molecules. Model never saw this group during training. The 37 diazonium compounds dominating COCONUT top hits score high purely from their RDKit 200-d descriptor signature projecting into a high-confidence region by chance — a distribution-shift artifact, not a biological signal.

2. **Organometallic is a legitimate shortcut.** 20× enriched in SD1 actives (4/480 vs 3/7,204). Model correctly weights it high. But these are toxic / assay-artifact chemistry, not viable antibiotics — the original screen likely flagged them as "active" because they kill indiscriminately.

3. **Nitro / azide** enrichments weak (~1.6×). Minor contribution to rankings.

4. **Michael acceptors** depleted in actives (0.60×). The 431 appearing in COCONUT priority are model-confident calls unrelated to reactive-group shortcut.

## Implication

Our priority set mixes two failure modes:
- OOD artifacts (e.g. diazonium) — model hallucinating confidence on unseen chemistry.
- PAINS-like shortcuts (organometallics) — model learned a real but non-useful signal.

Paper acknowledged this implicitly: they applied a post-hoc chemistry filter ("molecules with known antibiotic features, prior reported activity, or possible nonspecific membrane activity") before retaining 2 of 240. Same filter required here.

## Files

- `data/coconut_priority.csv` — 13,015 priority molecules, unfiltered for chemistry
- `data/sd2_priority.csv` — 240 SD2 priority, same concern applies
