# Deep learning-guided discovery of an antibiotic targeting *Acinetobacter baumannii*

**Authors:** Gary Liu, Denise B. Catacutan, Khushi Rathod, Kyle Swanson, Wengong Jin, Jody C. Mohammed, Anush Chiappino-Pepe, Saad A. Syed, Meghan Fragis, Kenneth Rachwalski, Jakob Magolan, Michael G. Surette, Brian K. Coombes, Tommi Jaakkola, Regina Barzilay, James J. Collins, Jonathan M. Stokes

**Published:** *Nature Chemical Biology*, Volume 19, November 2023, pp. 1342–1350
**DOI:** https://doi.org/10.1038/s41589-023-01349-8
**Received:** 25 March 2022 | **Accepted:** 25 April 2023 | **Published online:** 25 May 2023

---

## Abstract

*Acinetobacter baumannii* is a nosocomial Gram-negative pathogen that often displays multidrug resistance. Discovering new antibiotics against *A. baumannii* has proven challenging through conventional screening approaches. Fortunately, machine learning methods allow for the rapid exploration of chemical space, increasing the probability of discovering new antibacterial molecules. Here we screened ~7,500 molecules for those that inhibited the growth of *A. baumannii* in vitro. We trained a neural network with this growth inhibition dataset and performed in silico predictions for structurally new molecules with activity against *A. baumannii*. Through this approach, we discovered abaucin, an antibacterial compound with narrow-spectrum activity against *A. baumannii*. Further investigations revealed that abaucin perturbs lipoprotein trafficking through a mechanism involving LolE. Moreover, abaucin could control an *A. baumannii* infection in a mouse wound model. This work highlights the utility of machine learning in antibiotic discovery and describes a promising lead with targeted activity against a challenging Gram-negative pathogen.

---

## Introduction

*A. baumannii* is a nosocomial Gram-negative pathogen that often displays multidrug resistance due to its robust outer membrane and its ability to acquire and retain extracellular DNA that frequently encodes antibiotic resistance genes. It can survive for prolonged durations on surfaces and is resistant to desiccation. Discovering fundamentally new antibiotics against *A. baumannii* has proven challenging through conventional screening approaches. Most new antibiotics that achieve clinical use are analogs of existing classes. While structural analogs may satisfy short-term clinical needs, their long-term efficacy is inherently limited due to the high prevalence of existing resistance determinants.

Ideally, new antibiotic discovery efforts should focus on identifying new chemotypes with mechanisms of action that are unique relative to existing antibiotics. Such compounds are likely to have prolonged utility, given that the probability of pre-existing clinical resistance is low.

Machine learning methods allow for the rapid exploration of vast chemical/sequence spaces in silico. Typical high-throughput screening programs are limited to testing a few million molecules at the largest scales. Contemporary algorithmic approaches can assess hundreds of millions to billions of molecules for antibacterial properties. For example, Stokes et al. applied a message-passing neural network trained on growth inhibition of lab strain *E. coli* to discover new broad-spectrum antibacterial compounds. Ma et al. applied natural language processing neural network models to predict broad-spectrum antimicrobial peptides encoded in the human gut microbiome.

Beyond discovering structurally and functionally new antibiotics, a largely unmet need exists for narrow-spectrum therapies that target specific bacterial species. Such antibiotics are beneficial for two reasons:

1. The rate at which resistance to narrow-spectrum agents would disseminate is likely lower than conventional broad-spectrum agents, because narrow-spectrum drugs do not impose a universal selective pressure that favors the wide propagation of resistance determinants.
2. Narrow-spectrum antibiotics would not disrupt the ecology of the microbiota during treatment. Dysbiosis has been associated with infectious diseases, inflammatory bowel diseases, metabolic diseases, neuropsychiatric disorders, and cancer. *Clostridioides difficile* infections are prime examples of opportunistic infections resulting from antibiotic-induced dysbiosis, causing upwards of 224,000 infections in hospitalized patients and 13,000 deaths in the US alone in 2017.

---

## Results

### Machine learning-guided discovery of abaucin

The authors applied a message-passing deep neural network to discover new antibiotics against *A. baumannii*. They first screened a diverse collection of 7,684 small molecules at 50 µM for those that inhibited the growth of *A. baumannii* ATCC 17978 in Lysogeny Broth (LB) medium. This chemical collection consisted of:

- **2,341** off-patent drugs
- **5,343** synthetic chemicals

Using a conventional hit cutoff of one standard deviation below the mean growth of the entire dataset resulted in **480 molecules** defined as 'active' and **7,204** defined as 'inactive'.

#### Model Architecture

The directed message-passing neural network translates the graph structure of a molecule into a continuous vector. The model:

1. Iteratively exchanges information of local chemistry between adjacent atoms and bonds in a series of 'message-passing' steps
2. After a defined number of steps, sums vector representations of local chemical regions into a single continuous vector
3. Supplements this learned vector with fixed molecular features computed using RDKit
4. Feeds the combined vector into a feed-forward neural network that predicts antibacterial properties

The model was optimized using an ensemble of ten classifiers. Final performance:

- **Area under precision-recall curve:** 0.337 ± 0.088
- **Area under ROC curve:** 0.792 ± 0.042

The baseline model without RDKit features performed worse (AUPRC 0.266 ± 0.070, AUROC 0.756 ± 0.050), highlighting the importance of computable molecular features.

#### Predictions and Validation

The ensemble was applied to the Drug Repurposing Hub (6,680 molecules). Using a prediction score threshold of >0.2 and Tanimoto nearest neighbor similarity of <0.3 to training set 'actives', 240 priority molecules were identified and tested at 50 µM.

Results:

- **9 of 240** priority molecules showed >80% growth inhibition
- **0 of 240** lowest-scoring molecules showed activity (confirming discriminatory utility)
- **40 of 240** highest-scoring molecules (without Tanimoto filtering) showed activity

Structure-based filtering removed molecules with known antibiotic features, prior reported activity, or possible nonspecific membrane activity, retaining two molecules:

- **RS102895** (a CCR2-selective chemokine receptor antagonist): MIC ~2 µg/ml
- **Serdemetan** (an HDM2 antagonist): MIC ~32 µg/ml

RS102895 was renamed **abaucin** for its activity against *A. baumannii*.

#### Abaucin characteristics

- Modest bactericidal activity in LB medium
- Upon removal after 6 h treatment, regrowth occurred with increasing lag as concentration increased
- **No discernible activity in nutrient-deplete PBS**, suggesting the target is a biological process maximally active during growth and division
- Not membrane-active (which retains efficacy in nutrient-deplete conditions)

### Abaucin has a narrow spectrum of activity

Abaucin was tested against diverse clinical isolates:

| Pathogen | # Strains | Activity |
|----------|-----------|----------|
| *A. baumannii* (clinical) | 41 | Overcame all intrinsic and acquired resistance |
| Carbapenem-resistant *Enterobacteriaceae* | 24 | No activity up to 20× MIC |
| *P. aeruginosa* | 24 | No activity up to 20× MIC |
| *S. aureus* | 14 | No activity up to 20× MIC |

#### Commensal species

Panels tested:

- **34** diverse human gut commensals
- **19** diverse human skin commensals

Ampicillin (128 µg/ml) and ciprofloxacin (2 µg/ml) displayed broad activity across commensals. Abaucin largely avoided growth inhibition even at 20× MIC, with bona fide inhibition only against *Bifidobacterium breve* and *Bifidobacterium longum* (above *A. baumannii* MIC). Since *Bifidobacterium* is Gram-positive and phylogenetically divergent from *Acinetobacter*, this lower potency activity is likely through an unrelated mechanism.

#### Structure-activity relationship (abaucin analogs)

| Compound | E. coli | P. aeruginosa | S. aureus | A. baumannii |
|----------|---------|---------------|-----------|--------------|
| Abaucin (1) | >128 µg/ml | >128 µg/ml | >128 µg/ml | **2 µg/ml** |
| Analog 2 (no CF3) | >128 | >128 | >128 | 8 |
| Analog 3 (F-substituted) | >128 | >128 | >128 | 64 |
| Analog 4 (Cl-substituted) | >128 | >128 | >128 | 128 |

### Abaucin inhibits lipoprotein trafficking in *A. baumannii*

#### Resistant mutant selection

Abaucin-resistant clones isolated on solid media at 4–5 µg/ml did not display cross-resistance to diverse antibiotics. Whole-genome sequencing of four independent isolates revealed mutations in or upstream of **LolE**, an essential inner membrane protein involved in lipoprotein trafficking:

- **Y394F** and **upstream G→A** mutation: 4-fold abaucin resistance
- **A362T** (two independent isolates): 16-fold abaucin resistance

Frequency of resistance in vitro: **10⁻⁸–10⁻⁷**, consistent with antibiotics targeting a single protein.

*A. baumannii* LolE position A362 is homologous to *E. coli* LolE position I365, which resides near the acyl chains of the nascent lipoprotein during transport. Structural prediction using RoseTTAFold confirmed the spatial proximity.

#### Transcriptomic support

RNA-seq of *A. baumannii* treated with 5× MIC abaucin showed:

- Downregulation of aerobic electron transport chain genes
- Downregulation of transmembrane ion transport genes
- Consistent with activation of the Cpx two-component envelope stress response, which monitors lipoprotein trafficking from the inner to outer membrane in Gram-negative bacteria

#### CRISPRi and qPCR validation

- **LolE knockdown** (via inducible CRISPRi, three sgRNAs): 4- to 8-fold decrease in abaucin MIC vs. empty vector
- **qPCR** on resistant mutants: the intergenic G→A mutant showed ~4-fold increased *lolE* expression, paralleling its 4-fold resistance (multicopy suppression-mediated resistance)

#### Phenotypic confirmation

Fluorescence microscopy (DAPI for DNA, FM4-64 for envelope) showed that abaucin-treated *A. baumannii* displayed:

- Increased cell swelling
- Loss of intracellular nucleoid condensation

These phenotypes are consistent with prior work on lipoprotein transport inhibition in *E. coli*.

### Abaucin suppresses *A. baumannii* in a wound infection model

**Model:** neutropenic C57BL/6 mice, dorsal wound infection with ~6.5 × 10⁶ CFU *A. baumannii* ATCC 17978.

**Treatment:** Glaxal Base Moisturizing Cream + vehicle (1.65% DMSO) or abaucin (4% w/v), applied at 2, 3, 4, 6, 10, 21, and 24 h postinfection.

**Results at 25 h postinfection:**

| Group | Bacterial load |
|-------|---------------|
| Vehicle | ~6.9 × 10⁸ CFU/g (significant inflammation) |
| Abaucin 4% | ~4.0 × 10⁷ CFU/g (markedly less inflammation, near pre-treatment levels) |
| Pre-treatment control | similar to abaucin group |

Statistical significance: vehicle vs. abaucin p = 0.0039; pre-Tx vs. vehicle p = 0.0034; pre-Tx vs. abaucin not significant (p = 0.0704).

---

## Discussion

Key contributions:

1. **Machine learning validated for narrow-spectrum antibiotic discovery** targeting a challenging Gram-negative pathogen
2. **Lipoprotein trafficking** is a highly sought-after antibiotic target not yet perturbed by clinical drugs
3. Narrow-spectrum activity explained in part by divergence of the *A. baumannii* Lol system: most Gram-negatives have asymmetric LolC/LolD/LolE, while *A. baumannii* has a symmetric complex with LolD and two copies of LolE (LolF), without LolC

**Opportunities for future work:**

- Larger training datasets and prediction libraries (hundreds of millions to billions of compounds)
- Multi-property optimization models (growth inhibition + mammalian cell toxicity)
- Medicinal chemistry optimization of abaucin for enhanced in vivo activity

---

## Key methods summary

### Training data acquisition

*A. baumannii* ATCC 17978 grown overnight in LB, diluted 1/10,000, 99 µl dispensed to 96-well plates. 1 µl of 5 mM molecule stock added (final 50 µM), duplicate. Plates incubated 16 h at 37 °C, read at OD600. Actives defined as growth ≥1σ below dataset mean.

### Model (Chemprop)

- Directed message-passing neural network
- Hyperparameters: 3 message-passing steps, hidden size 300, 2 feed-forward layers, dropout 0
- Augmented with 200 RDKit molecule-level features
- Ensemble of 10 models, each trained on unique 80/10/10 split, 30 epochs
- 10-fold cross-validation

### Structural analysis

Morgan fingerprints (radius 2, 2048 bits) via RDKit. Tanimoto similarity for chemical comparison. t-SNE with Jaccard distance via scikit-learn.

### Mechanism studies

- Suppressor mutant evolution on solid LB + abaucin (2–10 µg/ml); Illumina MiSeq + Breseq alignment to reference
- RNA-seq via kallisto + DESeq2 (cutoff log2FC ≥ 1.5, p_adj ≤ 0.01); GO enrichment via EcoCyc Pathway Tools
- CRISPRi via pFD152 plasmid, three sgRNAs, aTc induction
- qPCR with *rpoD*, *gltA*, *gyrB* housekeeping normalization
- RoseTTAFold structural prediction via Robetta; alignment via Maestro and ChimeraX

### Animal model

6–8 week old female C57BL/6N mice, cyclophosphamide-induced neutropenia (150 mg/kg day T-4, 100 mg/kg day T-1). Tape-stripped dorsal abrasion (~2 cm²), infected with 6.5 × 10⁶ CFU. Treatments in Glaxal Base at multiple timepoints over 24 h. Wound tissue homogenized and plated on LB + chloramphenicol.

---

## Data and code availability

- **GenBank:** OP677864–OP677867 (abaucin-resistant mutant genomes)
- **GEO:** GSE214305 (RNA-seq datasets)
- **Code:** https://github.com/chemprop/chemprop (Chemprop); https://github.com/GaryLiu152/chemprop_abaucin (paper-specific snapshot)

---

## Funding

David Braley Centre for Antibiotic Discovery; Weston Family Foundation; Audacious Project; C3.ai Digital Transformation Institute; Abdul Latif Jameel Clinic for Machine Learning in Health; DTRA DOMANE program; DARPA Accelerated Molecular Discovery program; CIHR (FRN-156361, FRN-148713); Genome Canada GAPP (OGI-146); McMaster Faculty of Health Sciences; Boris Family; Marshall Scholarship; DOE BER (DE-FG02-02ER63445).

## Competing interests

J.M.S. is cofounder and scientific director of Phare Bio. J.J.C. is cofounder and scientific advisory board chair of Phare Bio and Enbiotix.

## Correspondence

- James J. Collins: jimjc@mit.edu
- Jonathan M. Stokes: stokesjm@mcmaster.ca
