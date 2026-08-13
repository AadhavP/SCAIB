# Dataset Evidence Notes

Retrieval date: 2026-08-11

This file records the paper-readiness evidence for the `paper-datasets`
shortlist. It verifies the existence and task fit of sources; it does not
certify that the files have been downloaded, checksummed, or legally cleared for
redistribution.

## Evidence Legend

- **Source verified:** primary paper, public accession, atlas page, or analysis
  repository was located.
- **Split feasible:** available metadata appears sufficient for a non-random
  paper split, pending materialization audit.
- **Freeze blocker:** issue that must be resolved before this candidate becomes
  part of a frozen benchmark release.

## Annotation Candidates

### `ann_tabula_sapiens` - Tabula Sapiens v1

- Source verified: GEO `GSE201333` is public and describes Tabula Sapiens as a
  single-cell transcriptomic atlas of 24 human tissues and organs. The GEO record
  points to processed supplementary files and a public AWS bucket.
- Source verified: AWS Open Data describes Tabula Sapiens v1 as nearly 500,000
  cells from 24 organs of 15 normal human subjects, with a later v2 release also
  available.
- Paper role: cross-tissue cell annotation with donor and tissue holdout.
- Split feasible: donor and tissue are expected split keys; do not randomly
  split cells.
- Baseline fit: marker rules, SingleR/scmap, CellTypist, and scANVI-style
  reference mapping are compatible.
- Freeze blockers: audit data-use terms, choose v1 versus v2 explicitly, and
  verify which fields are safe for agent-visible metadata.
- Sources:
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201333
  - https://registry.opendata.aws/tabula-sapiens/
  - https://www.science.org/doi/10.1126/science.abl4896

### `ann_human_brain_v1` - Human Brain Cell Atlas v1.0

- Source verified: HCA describes the atlas as an adult whole-brain
  single-nucleus survey across forebrain, midbrain, and hindbrain, with over
  three million nuclei and hierarchical labels.
- Source verified: Allen Brain Cell Atlas access documentation lists expression
  matrices, cell/gene metadata, and taxonomy resources in an AWS public dataset.
- Paper role: hierarchical region-held-out annotation.
- Split feasible: anatomical region is the primary holdout key; donor is a
  secondary grouping variable.
- Baseline fit: marker rules, SingleR/scmap, and reference mapping are
  compatible, but hierarchical scoring is needed.
- Freeze blockers: CC BY-NC 4.0 terms need redistribution audit; the donor count
  is small, so claims must be regional/hierarchical rather than donor
  generalization.
- Sources:
  - https://data.humancellatlas.org/hca-bio-networks/nervous-system/atlases/brain-v1-0
  - https://alleninstitute.github.io/abc_atlas_access/descriptions/WHB-10Xv3.html
  - https://www.science.org/doi/10.1126/science.add7046

### `ann_kidney_injury` - Human Kidney Atlas healthy and injured states

- Source verified: Nature primary article is available for the healthy/injured
  kidney atlas.
- Source verified: GEO `GSE183279` is public, lists 199 samples, and links
  multiple subseries including scRNA-seq, snRNA-seq, SNARE-seq2, Slide-seq2,
  Visium, and other modalities.
- Paper role: disease-aware kidney cell-state annotation with abstention and
  state ambiguity.
- Split feasible: donor, modality, and disease status are required split keys.
- Baseline fit: marker rules, SingleR/CellTypist, and scANVI-style mapping are
  compatible after modality-specific preprocessing is defined.
- Freeze blockers: reject any split where disease and platform/modality are
  perfectly confounded; define which modalities are in scope for the first paper.
- Sources:
  - https://www.nature.com/articles/s41586-023-05769-3
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183279
  - https://atlas.kpmp.org/
  - https://cellxgene.cziscience.com/collections/bcb61471-2a44-4d00-a0af-ff085512674c

### `ann_pancreas_crossstudy` - Human pancreas cross-study collection

- Source verified: Baron `GSE84133` is public and covers single-cell pancreatic
  islets from four human donors, with raw data in SRA and processed CSV.
- Source verified: Muraro `GSE85241` is public and covers CEL-seq2 pancreatic
  cells from four donors, with raw SRA and processed CSV.
- Source pending: Segerstolpe `E-MTAB-5061` is an accepted ArrayExpress access
  path but needs a materialization pass to confirm current file names and
  metadata.
- Paper role: leave-one-study-out annotation and label-transfer.
- Split feasible: source study is the main split key; harmonized cell-type
  vocabulary must be frozen before scoring.
- Baseline fit: scmap is especially relevant; SingleR, marker rules, and scANVI
  are compatible.
- Freeze blockers: harmonize labels across studies and preregister one
  unknown-population variant only if the absent-label policy is fixed in advance.
- Sources:
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241
  - https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061
  - https://scmap.sanger.ac.uk/scmap/

## Integration Candidates

### `int_cellbench_mixology` - CellBench/sc_mixology

- Source verified: Nature Methods primary article describes 14 datasets from
  single cells and RNA/cell mixtures across droplet and plate scRNA-seq
  protocols, with known mixture controls.
- Source verified: GEO `GSE118767` is public and contains multiple subseries for
  cell mixture and protocol comparisons.
- Source verified: the processed data and analysis repository resolves to
  `LuyiTian/sc_mixology`.
- Paper role: controlled integration calibration and metric sanity check.
- Split feasible: hold out protocol or mixture design; keep known cell-line and
  mixture truth evaluator-only.
- Baseline fit: no integration, Harmony, Scanorama, scVI, and scANVI.
- Freeze blockers: treat as technical calibration rather than tissue biology;
  verify current processed object availability.
- Sources:
  - https://www.nature.com/articles/s41592-019-0425-8
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118767
  - https://github.com/LuyiTian/CellBench_data
  - https://bioconductor.org/packages/CellBench

### `int_multicenter_reference` - Multi-center cross-platform reference

- Source verified: Scientific Data article describes 20 scRNA-seq datasets from
  two biologically distinct cell lines, generated across multiple platforms and
  four sequencing centers.
- Source verified: PubMed, SRA `SRP199641`, BioProject `PRJNA504037`, and a
  Springer Nature figshare collection provide stable access paths.
- Paper role: center-platform integration and batch-correction benchmark.
- Split feasible: center-platform combination is the main holdout key; libraries
  must stay intact within one split.
- Baseline fit: no integration, Harmony, Scanorama, scVI, and related
  batch-correction methods.
- Freeze blockers: verify file-to-library mapping and matched bulk/reference
  labels before scoring.
- Sources:
  - https://www.nature.com/articles/s41597-021-00809-x
  - https://pubmed.ncbi.nlm.nih.gov/33531477/
  - https://www.ncbi.nlm.nih.gov/sra/?term=SRP199641
  - https://www.ncbi.nlm.nih.gov/bioproject/PRJNA504037
  - https://springernature.figshare.com/collections/A_Multi-center_Cross-platform_Single-cell_RNA_Sequencing_Reference_Dataset/5213468

### `int_hlca` - Integrated Human Lung Cell Atlas

- Source verified: Nature Medicine primary article describes HLCA as an atlas
  combining 49 respiratory datasets and over 2.4 million cells from 486
  individuals.
- Source verified: HCA atlas page describes core/full HLCA, raw count access in
  downloadable objects, consensus labels, and CC BY 4.0 terms.
- Paper role: study-held-out atlas-scale integration.
- Split feasible: source study and donor are the required split keys; disease
  and anatomical annotations should be preserved for biology-retention metrics.
- Baseline fit: no integration, Harmony, Scanorama, scVI, and scANVI.
- Freeze blockers: agent input must start from count matrices, not the published
  integrated embedding; deterministic subsetting is mandatory because of scale.
- Sources:
  - https://www.nature.com/articles/s41591-023-02327-2
  - https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0
  - https://github.com/LungCellAtlas/HLCA
  - https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293

## Differential-Expression Candidates

### `de_kang_ifnb` - Kang et al. IFN-beta PBMC

- Source verified: GEO `GSE96583` is public and describes demuxlet-based
  multiplexed droplet scRNA-seq with approximately 15k PBMCs used to study
  IFN-beta response.
- Source verified: GEO links SRA `SRP102802`, raw matrix files, and processed
  supplementary files.
- Paper role: compact paired donor-aware stimulation DE.
- Split feasible: donor must be the inferential unit; preserve paired stimulated
  and control conditions in each donor-level split.
- Baseline fit: edgeR pseudobulk, DESeq2 pseudobulk, limma-voom pseudobulk, and
  cell-level Wilcoxon negative control.
- Freeze blockers: explicitly use the paired stimulation cohort; audit batch
  labels and donor-condition structure before freezing.
- Sources:
  - https://www.nature.com/articles/nbt.4042
  - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583
  - https://www.ncbi.nlm.nih.gov/sra/?term=SRP102802

### `de_smillie_uc` - Smillie et al. ulcerative-colitis colon atlas

- Source verified: Single Cell Portal `SCP259` reports 365,492 total cells and
  21,784 genes for the UC colon atlas, with DOI and study summary.
- Source verified: analysis code repository lists required metadata, epithelial,
  stromal, and immune matrix files, plus Seurat objects.
- Paper role: tissue case-control DE outside blood.
- Split feasible: donor holdout with diagnosis, inflammation status, and colon
  location stratification.
- Baseline fit: edgeR pseudobulk, DESeq2 pseudobulk, limma-voom pseudobulk, and
  cell-level Wilcoxon negative control.
- Freeze blockers: verify current download availability and terms; balance
  diagnosis, inflammation, and location so DE does not become a confounding test.
- Sources:
  - https://doi.org/10.1016/j.cell.2019.06.029
  - https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-of-the-human-colon-during-ulcerative-colitis
  - https://github.com/cssmillie/ulcerative_colitis

### `de_stephenson_covid` - Stephenson et al. COVID-19 PBMC multi-omics cohort

- Source verified: Nature Medicine article describes single-cell multi-omics
  analysis of COVID-19 immune response and provides data/code availability.
- Source verified: the article lists the COVID-19 cell atlas portal, downloadable
  H5AD object, ArrayExpress `E-MTAB-10026`, and analysis scripts.
- Source verified: HCA project page lists ArrayExpress `E-MTAB-10026`, EGA
  `EGAS00001005465`, and CC BY 4.0/HCA data policy.
- Paper role: multicenter severity-aware donor-held-out PBMC DE with protein and
  receptor measurements as hidden validation.
- Split feasible: person is the grouping key; center and severity must be
  stratification keys.
- Baseline fit: edgeR pseudobulk, DESeq2 pseudobulk, limma-voom pseudobulk, and
  cell-level Wilcoxon negative control.
- Freeze blockers: audit controlled-access components and deidentification;
  verify that center and severity are not too confounded for the chosen contrast.
- Sources:
  - https://www.nature.com/articles/s41591-021-01329-2
  - https://www.ebi.ac.uk/biostudies/studies/E-MTAB-10026
  - https://explore.data.humancellatlas.org/projects/b963bd4b-4bc1-4404-8425-69d74bc636b8
  - https://github.com/scCOVID-19/COVIDPBMC/
  - https://covid19cellatlas.org/

## Recommended Use In The Paper

Use the complete 10-dataset candidate set for the paper's benchmark inventory
figure and materialization pilot. Freeze the final benchmark only after the
download/checksum/access audit. If the final release needs to shrink, preserve
at least:

- three annotation datasets;
- two integration datasets;
- three donor-aware differential-expression datasets.

The two weakest candidates to drop first, if materialization pressure is high,
are `ann_pancreas_crossstudy` because label harmonization is labor-intensive and
`ann_human_brain_v1` if the non-commercial license or large-file handling blocks
paper execution. Do not drop `de_kang_ifnb`, because it is the cleanest compact
paired DE vertical slice.

