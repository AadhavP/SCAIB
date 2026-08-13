# SCAIB Paper Dataset Shortlist

## Status

This folder contains a paper-facing shortlist of high-quality public single-cell
datasets for comparing SCAIB against existing benchmarking methods. It is not a
frozen benchmark release and does not contain downloaded data.

Download entrypoints are listed in [downloads.md](downloads.md). Source-quality
notes are in [evidence.md](evidence.md), and the structured shortlist is in
[datasets.yaml](datasets.yaml).

The current recommendation is to use all **10** datasets below as the candidate
portfolio. If access, licensing, metadata, or split feasibility fails during
materialization, the paper can still proceed with a defensible **8-dataset
minimum** by removing the weakest candidates before preregistration.

## Selection Principles

- Use independent studies or atlas families, not repeated tasks on the same
  dataset, when counting datasets.
- Cover annotation, integration, and donor-aware differential expression.
- Prefer datasets with primary publications, stable accessions, raw counts or
  defensible count matrices, donor/study/platform metadata, and hidden
  evaluator-only references.
- Avoid random cell splits. Split by donor, study, tissue, region, center,
  platform, or condition as appropriate.
- Report scores by task family. Do not average annotation, integration, and
  differential-expression results into one unexplained number.

## Recommended 10-Dataset Portfolio

| ID | Dataset | Family | Why it belongs | Required split |
| --- | --- | --- | --- | --- |
| `ann_tabula_sapiens` | Tabula Sapiens v1 | Annotation | Broad human multi-organ atlas with expert labels, multiple donors, and tissue diversity. Useful for cross-tissue and donor-held-out annotation. | Hold out complete donors and selected tissues. |
| `ann_human_brain_v1` | Human Brain Cell Atlas v1.0 | Annotation | Large whole-brain single-nucleus atlas with hierarchical labels across regions. Tests regional transfer, rare populations, and hierarchical scoring. | Hold out anatomical regions; avoid donor-generalization claims because donor count is small. |
| `ann_kidney_injury` | Human Kidney Atlas healthy/injured states | Annotation | Multimodal healthy and diseased kidney atlas with canonical and injury-associated states. Tests disease-aware annotation and abstention. | Split by donor and acquisition modality, stratified by disease. |
| `ann_pancreas_crossstudy` | Human pancreas cross-study collection | Annotation | Baron, Muraro, and Segerstolpe provide an established cross-study label-transfer setting with shared pancreatic populations and protocol variation. | Leave one complete study out. |
| `int_cellbench_mixology` | CellBench/sc_mixology | Integration | Controlled cell-line and RNA/cell mixture truth across protocols. Strong calibration set for integration metrics and implementation sanity checks. | Hold out complete protocol or mixture design. |
| `int_multicenter_reference` | Multi-center cross-platform reference | Integration | Twenty datasets from two reference cell lines and mixtures across centers/platforms. Directly tests center and platform correction. | Leave out one center-platform combination; keep libraries intact. |
| `int_hlca` | Integrated Human Lung Cell Atlas | Integration | Atlas-scale lung dataset with study, donor, technology, anatomy, disease, and consensus labels. Tests study-held-out integration at realistic scale. | Leave out complete source studies and donors. |
| `de_kang_ifnb` | Kang et al. IFN-beta PBMC | Differential expression | Compact paired stimulation design across multiplexed donors. Tests whether agents model donor as the inferential unit. | Split by donor while preserving paired conditions. |
| `de_smillie_uc` | Smillie et al. ulcerative-colitis colon atlas | Differential expression | Large colon disease atlas with epithelial, stromal, and immune subsets. Tests tissue case-control DE outside blood. | Hold out donors; stratify by diagnosis, inflammation status, and colon location. |
| `de_stephenson_covid` | Stephenson et al. COVID-19 PBMC cohort | Differential expression | Large multicenter PBMC multi-omics cohort with severity labels and independent protein/receptor measurements. | Split by person, stratified by center and severity. |

## Baseline Comparisons

Minimum compatible baselines by family:

| Family | Fixed baselines |
| --- | --- |
| Annotation | Marker-rule pipeline, SingleR or scmap, CellTypist where compatible, scANVI/reference mapping where labels are permitted. |
| Integration | No integration, Harmony, Scanorama, scVI, scANVI where labels are permitted. |
| Differential expression | edgeR pseudobulk, DESeq2 pseudobulk, limma-voom pseudobulk, and cell-level Wilcoxon as a negative control. |
| Agent systems | Rule-based SCAIB agent, at least two frontier-model agents, and the same agent with and without the SCAIB skill bundle. |

If a method cannot run a task, report it as `not_supported`; do not score it as
zero and do not silently omit it.

## Hard Gates Before Freezing

Before using any dataset in the final paper benchmark release, verify:

- downloadable files and immutable checksums;
- access terms and redistribution constraints;
- raw counts or count-equivalent matrix availability;
- unique cell identifiers and stable gene identifiers;
- donor/study/platform/condition metadata needed for the declared split;
- no perfect batch-condition confounding for the scored comparison;
- hidden reference fields are removed from agent-visible inputs;
- deterministic smoke, paper, and full subset definitions.
