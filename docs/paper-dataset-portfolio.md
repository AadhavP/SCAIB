# SCAIB Paper Dataset Portfolio

## Status

This is the recommended **10-dataset candidate portfolio** for the SCAIB paper. It is a research plan, not yet a frozen benchmark release. A dataset should enter the final test suite only after its files, metadata, access terms, preprocessing recipe, and checksums have been independently verified.

The portfolio is aligned with the three workflow families already declared for SCAIB:

- four cell-annotation datasets;
- three batch-correction and integration datasets;
- three donor-aware differential-expression datasets.

Each row represents an independent study or atlas family. Reusing a dataset for an additional task may create another benchmark instance, but must not increase the reported dataset count.

## Why These Ten

The set is deliberately broader than a collection of standard PBMC examples. It covers multi-organ tissue, brain, kidney, pancreas, lung, colon, controlled cell mixtures, paired immune stimulation, and multicenter infectious disease. It also includes whole-cell and single-nucleus RNA sequencing, droplet and plate protocols, healthy and diseased tissue, known mixture controls, paired treatments, and large multi-donor cohorts.

The selection criteria were:

1. a peer-reviewed primary study and stable public repository or atlas page;
2. raw counts or a defensible processed count matrix;
3. donor, study, condition, technology, or region metadata needed for non-leaking splits;
4. a scientifically meaningful hidden reference;
5. enough replication for the proposed inference;
6. complementarity with the other datasets;
7. a feasible, reproducible paper-scale subset.

## Recommended Portfolio

| ID | Dataset | Primary role | Why it belongs | Access | Required split |
| --- | --- | --- | --- | --- | --- |
| `ann_tabula_sapiens` | Tabula Sapiens v1 | Cross-tissue annotation | Nearly 500,000 cells from 24 tissues and 15 donors, with ontology-based expert labels. It tests whether an agent can transfer broad cell identities across donors and tissues. | [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9812260/), [GEO GSE201333](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201333), [AWS data](https://registry.opendata.aws/tabula-sapiens/) | Hold out complete donors and selected tissues. Never split cells randomly. |
| `ann_human_brain_v1` | Human Brain Cell Atlas v1.0 | Hierarchical annotation | More than three million nuclei from about 100 adult brain dissections, with labels at supercluster, cluster, and subcluster levels. It tests hierarchical labels, rare populations, regional transfer, and whole-cell versus nuclear differences. | [HCA atlas](https://data.humancellatlas.org/hca-bio-networks/nervous-system/atlases/brain-v1-0), [Allen access tools](https://alleninstitute.github.io/abc_atlas_access/descriptions/WHB-10Xv3.html) | Hold out anatomical regions. The atlas has only three donors, so it must not be used to claim strong donor generalization. |
| `ann_kidney_injury` | Human Kidney Atlas: healthy and injured states | Disease-aware annotation | A multi-institution, multimodal atlas spanning healthy and diseased kidneys. The RNA labels include canonical cell types and injury-associated states, making it useful for testing ambiguity, disease transfer, and abstention. | [Primary paper](https://www.nature.com/articles/s41586-023-05769-3), [GEO GSE183279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183279), [CELLxGENE collection](https://cellxgene.cziscience.com/collections/bcb61471-2a44-4d00-a0af-ff085512674c), [KPMP portal](https://atlas.kpmp.org/) | Split by donor and acquisition modality; stratify by disease. Reject any split in which disease and platform are perfectly confounded. |
| `ann_pancreas_crossstudy` | Human pancreas cross-study collection | Cross-study annotation | Baron, Muraro, and Segerstolpe contain shared pancreatic populations measured using different protocols. This is an established label-transfer setting with a clear study holdout and a practical scale. | [Baron: GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133), [Muraro: GSE85241](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241), [Segerstolpe: E-MTAB-5061](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061), [scmap preparation example](https://scmap.sanger.ac.uk/scmap/) | Leave one entire study out. Add one preregistered unknown-cell-type variant in which a query population is absent from the reference. |
| `int_cellbench_mixology` | CellBench/sc_mixology | Controlled integration calibration | Known cell-line identities and known RNA/cell mixture proportions provide experimental truth across multiple single-cell protocols. This is the strongest control for detecting whether a metric or implementation is behaving incorrectly. | [Primary paper](https://www.nature.com/articles/s41592-019-0425-8), [GEO GSE118767](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118767), [data repository](https://github.com/LuyiTian/CellBench_data), [Bioconductor package](https://bioconductor.org/packages/CellBench) | Hold out a complete protocol or mixture design. Preserve known cell-line and mixture identities as evaluator-only fields. |
| `int_multicenter_reference` | Multi-center, cross-platform reference dataset | Platform and center integration | Twenty datasets were generated from two well-characterized cell lines and their mixtures across centers and platforms. Cell identity, mixture composition, and matched bulk references make technical effects measurable without relying only on inferred tissue labels. | [Primary data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC7854649/), [SRA SRP199641](https://www.ncbi.nlm.nih.gov/sra/?term=SRP199641), [BioProject PRJNA504037](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA504037) | Leave one center-platform combination out. Do not let files from the same library appear in different splits. |
| `int_hlca` | Integrated Human Lung Cell Atlas | Atlas-scale integration | The HLCA combines millions of cells from many studies, individuals, technologies, anatomical sites, and diseases under a consensus annotation. It tests study-level generalization and whether integration removes technical variation while retaining disease and tissue biology. | [Primary paper](https://www.nature.com/articles/s41591-023-02327-2), [HCA atlas page](https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0), [CELLxGENE collection](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293) | Leave out complete source studies and donors. Use source count matrices, not the published integrated embedding, as agent input. |
| `de_kang_ifnb` | Kang et al. IFN-beta PBMC | Paired donor-aware DE | The paired control and IFN-beta design across eight multiplexed donors is compact, public, and biologically strong. It directly tests whether the agent recognizes the donor as the replicate and uses a paired design. | [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5784859/), [GEO GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583), [SRA SRP102802](https://www.ncbi.nlm.nih.gov/sra/?term=SRP102802) | Split by donor while retaining each donor's paired conditions. Use the paired batch-2 cohort and audit the full series for batch-condition confounding. |
| `de_smillie_uc` | Smillie et al. ulcerative-colitis colon atlas | Tissue case-control DE | The atlas contains 366,650 cells from 18 ulcerative-colitis patients and 12 healthy individuals with curated epithelial, stromal, and immune subsets. It tests donor-aware disease DE outside blood and exposes the agent to inflamed and non-inflamed tissue states. | [Primary paper](https://www.cell.com/cell/fulltext/S0092-8674(19)30732-9), [Single Cell Portal SCP259](https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-of-the-human-colon-during-ulcerative-colitis), [analysis code](https://github.com/cssmillie/ulcerative_colitis) | Hold out complete donors; stratify by diagnosis, inflammation status, and colon location. Confirm that the processed count matrix and required donor metadata are downloadable under usable terms before freezing. |
| `de_stephenson_covid` | Stephenson et al. COVID-19 PBMC multi-omics cohort | Multicenter severity-aware DE | The study profiled more than 780,000 quality-controlled PBMCs from a 130-person cohort, with disease severity, center, RNA, surface protein, and immune-receptor information. It enables large donor-held-out comparisons and independent protein-level validation. | [Primary paper](https://www.nature.com/articles/s41591-021-01329-2), [ArrayExpress E-MTAB-10026](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026), [analysis code](https://github.com/scCOVID-19/COVIDPBMC/) | Split by person, not sample or cell; stratify by center and severity. Keep protein and receptor measurements hidden if they are used as independent validation. |

## Task Definitions

### Annotation

Each annotation instance should expose only the expression matrix and scientifically realistic covariates. Publication labels, ontology terms, marker tables copied from the source paper, precomputed reference embeddings, and filenames containing the answer must be hidden.

Score:

- hierarchical macro-F1 and balanced accuracy;
- rare-class recall;
- label coverage and calibration;
- performance on the held-out donor, tissue, region, or study;
- correct abstention on unknown populations;
- invalid or biologically inconsistent labels.

The source labels should be described as **held-out reference annotations**, not absolute biological ground truth. Brain and kidney labels are hierarchical and state-dependent, so a partially correct parent label should receive more credit than an unrelated label.

### Integration

Every integration task should compare the submitted representation with an unintegrated baseline. A scientifically valid agent must be allowed to decide that integration is unnecessary.

Score both sides of the tradeoff:

- removal of protocol, center, or study structure;
- preservation of known cell identity and mixture composition;
- preservation of disease, tissue, and condition signals;
- rare-population retention;
- stability across seeds;
- behavior on a held-out batch or study.

For HLCA, the published integrated object is a reference for validation and metadata harmonization, not a permissible starting representation. For CellBench and the multicenter reference, experimentally known identities should receive more weight than labels inferred from clustering.

### Differential Expression

The donor or individual is the inferential unit. The benchmark should require cell-type-specific pseudobulk or another explicitly replicate-aware model. A cell-level rank-sum test may be included only as a deliberately flawed negative control.

Score:

- effect-direction agreement on held-out donors;
- ranked-gene concordance and stability;
- calibration on null or negative-control contrasts;
- reproducibility across donor resamples;
- pathway-level agreement;
- correct design formula, pairing, covariates, and multiple-testing correction.

Published marker lists are supporting evidence, not complete ground truth. Hidden reference results should be generated from a preregistered, replicate-aware pipeline and checked by a domain expert.

## Comparison Baselines

Report performance by task family; do not average incompatible tasks into one unexplained number.

| Family | Minimum fixed baselines |
| --- | --- |
| Annotation | marker-rule pipeline, scmap or SingleR, CellTypist where a compatible reference exists, and scANVI/reference mapping |
| Integration | no integration, Harmony, Scanorama, scVI, and scANVI when labels are permitted |
| Differential expression | edgeR pseudobulk, DESeq2 pseudobulk, limma-voom pseudobulk, and a cell-level Wilcoxon negative control |
| Agent systems | rule-based SCAIB agent, two frontier-model agents, and the same agent with and without the SCAIB skill bundle |

Use the same input counts, splits, preprocessing allowances, random seeds, and resource budgets for every compatible baseline. If a method cannot run a task, report it as **not supported**, not as zero and not as a silently omitted result.

## Dataset Materialization and Directory Structure

```text
data/releases/scaib-datasets-v1/
├── portfolio.yaml
├── datasets/
│   └── <dataset_id>/
│       ├── source.yaml
│       ├── raw/
│       ├── processed/
│       │   ├── counts.h5ad
│       │   └── metadata.parquet
│       ├── mappings/
│       │   ├── cell_ontology.tsv
│       │   └── covariates.tsv
│       ├── splits/
│       │   ├── train.json
│       │   ├── development.json
│       │   └── test.json
│       ├── checksums.sha256
│       └── README.md
└── hidden-evaluation/
    └── <dataset_id>/
        ├── references/
        ├── scorer.yaml
        └── expert-notes.md
```

Suggested source manifest:

```yaml
dataset_id: de_kang_ifnb
portfolio_version: 1.0.0
source:
  accession: GSE96583
  publication_doi: 10.1038/nbt.4042
  retrieved_at: YYYY-MM-DD
  files: []
license_or_terms:
  status: pending_audit
assay:
  organism: Homo sapiens
  modality: scRNA-seq
  count_layer: raw_counts
experimental_units:
  primary: donor_id
  paired_by: donor_id
split:
  grouping_key: donor_id
  stratify_by: [condition]
agent_visible_columns: [sample_id, condition, cell_barcode]
evaluator_only_columns: [donor_id, reference_cell_type]
checksums:
  algorithm: sha256
  files: {}
```

The actual agent-visible columns must be defined per task. For a DE task, donor IDs may need to be visible so that the agent can specify the correct design; they should still determine the train/test grouping. The example above is illustrative, not a frozen information policy.

## Paper-Scale Subsetting

The full atlases are too large for repeated frontier-agent runs. Create deterministic tiers from the same frozen release:

- **smoke:** 5,000–10,000 cells for interface and artifact checks;
- **paper:** approximately 25,000–100,000 cells while retaining every selected donor, batch, condition, and rare population;
- **full:** the complete materialized dataset for confirmatory runs on selected systems.

Subsample within donor-cell-type strata. Never downsample by taking the first cells in a file or by dropping small populations. Record the random seed and exact barcode list.

## Hard Quality Gates Before Freezing

A candidate is removed or replaced if any of the following cannot be established:

- stable accession, publication DOI, and usable access terms;
- immutable downloaded files with SHA-256 checksums;
- raw counts where the task requires count-based inference;
- stable gene identifiers and unique cell identifiers;
- the metadata required for the proposed split;
- at least three biological replicates per DE arm, preferably five or more;
- no perfect batch-condition confounding for the scored comparison;
- a hidden reference that is not simply exposed in `obs` or a filename;
- a deterministic materialization script that can recreate the paper input.

## Leakage and Interpretation Risks

- These are well-known datasets and may have appeared in model training data. Blind sample IDs, remove answer-bearing metadata, prohibit internet access during evaluation, and include less canonical held-out variants. This reduces but does not eliminate contamination risk.
- Random cell splits substantially overstate generalization. Split by donor, study, tissue, technology, center, or anatomical region as specified above.
- Integration metrics can reward removal of real biology. Always score biological preservation and compare with no integration.
- Donor-level DE cannot be validated with cells as independent replicates.
- Reference annotations contain curator judgment and can be wrong or resolution-dependent. Retain hierarchical mappings and expert adjudication.
- A large cell count does not compensate for a small number of donors. This is especially important for the Human Brain Cell Atlas.
- Do not tune prompts, weights, exclusions, or label mappings after inspecting hidden-test runs.

## Reserve and Extension Datasets

Two candidates should be materialized as reserves before the benchmark is frozen:

1. [NeurIPS 2021 BMMC multimodal dataset, GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) for paired RNA-protein and RNA-ATAC integration across donors and sites.
2. [Replogle et al. genome-scale Perturb-seq, PRJNA831566](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA831566) for a later perturbation-response extension; processed data are also available through the study's [interactive portal](https://gwps.wi.mit.edu/).

These should not be counted among the ten unless a replacement is preregistered or the paper's task scope is explicitly expanded. Trajectory and perturbation prediction are valuable future SCAIB tasks, but adding them now would change the paper beyond its currently stated three-workflow design.

## Recommendation

Use all ten as the **candidate portfolio**, but run a materialization pilot before calling the set final. Freeze the benchmark only after:

1. all ten sources download successfully;
2. donor/study/technology metadata pass a leakage audit;
3. the paper-scale subsets preserve the intended biological contrasts;
4. baseline pipelines complete on each compatible task;
5. the two reserve datasets are ready;
6. dataset versions, exclusions, label mappings, splits, and metrics are preregistered.

This produces a broader and more defensible paper than a PBMC-heavy suite while keeping the claims matched to SCAIB's implemented workflow families.
