# Where To Download The Paper Datasets

Retrieval date: 2026-08-11

This file gives the clearest current download entrypoint for each candidate
dataset. Use these links to materialize local data later; do not commit the
downloaded files to this repository.

## Recommended Local Layout

```text
data/raw/paper-datasets/
  ann_tabula_sapiens/
  ann_human_brain_v1/
  ann_kidney_injury/
  ann_pancreas_crossstudy/
  int_cellbench_mixology/
  int_multicenter_reference/
  int_hlca/
  de_kang_ifnb/
  de_smillie_uc/
  de_stephenson_covid/
```

For each dataset, save:

- `source-page.html` or a text note with the exact source page and access date;
- downloaded raw/processed files;
- `checksums.sha256`;
- a short `README.md` naming which files are agent-visible versus evaluator-only.

## Direct Download Index

Direct links below are the first URLs to try when materializing the benchmark
set. Landing pages are still recorded because they are the citation and license
source of truth.

| ID | Direct/file endpoint | Landing/provenance page | Prefer these files first | Notes |
| --- | --- | --- | --- | --- |
| `ann_tabula_sapiens` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE201nnn/GSE201333/suppl/GSE201333_RAW.tar>; `s3://czb-tabula-sapiens/` | GEO `GSE201333`: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201333>; AWS registry: <https://registry.opendata.aws/tabula-sapiens/> | `GSE201333_RAW.tar` for processed CSV/H5AD; AWS bucket listing for browsable project data. | GEO says processed data are supplementary files and raw fastq is not provided there. AWS notes raw fastq access can require a data-use agreement. |
| `ann_human_brain_v1` | `s3://allen-brain-cell-atlas/expression_matrices/WHB-10Xv3/20240330/`; `s3://allen-brain-cell-atlas/metadata/WHB-10Xv3/20241115/`; `s3://allen-brain-cell-atlas/mapmycells/WHB-10Xv3/20240831/` | Allen Brain Cell Atlas: <https://alleninstitute.github.io/abc_atlas_access/descriptions/WHB-10Xv3.html>; HCA atlas page: <https://data.humancellatlas.org/hca-bio-networks/nervous-system/atlases/brain-v1-0> | Metadata first, then a paper-scale subset of expression matrices. | Large dataset. Do not start by downloading the full 70 GB expression matrix set. License is CC BY-NC 4.0. |
| `ann_kidney_injury` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183279/suppl/GSE183279_RAW.tar> | GEO `GSE183279`: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183279>; KPMP portal: <https://atlas.kpmp.org/>; CELLxGENE collection: <https://cellxgene.cziscience.com/collections/bcb61471-2a44-4d00-a0af-ff085512674c> | `GSE183279_RAW.tar` plus the scRNA/snRNA subseries needed for the first paper task. | GEO is the most scriptable entrypoint; KPMP/CELLxGENE are useful for atlas objects and metadata inspection. |
| `ann_pancreas_crossstudy` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84133/suppl/GSE84133_RAW.tar>; <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/GSE85241/suppl/GSE85241_cellsystems_dataset_4donors_updated.csv.gz>; <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/GSE85241/suppl/GSE85241_cel-seq_barcodes.csv.gz>; <https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-5061> | Baron GEO `GSE84133`: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133>; Muraro GEO `GSE85241`: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241>; Segerstolpe BioStudies `E-MTAB-5061`: <https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061> | Baron raw TAR, Muraro updated donor matrix and barcodes, Segerstolpe matrix/metadata from BioStudies. | Materialize each source study separately, then harmonize labels. Keep study identity for leave-one-study-out splits. |
| `int_cellbench_mixology` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/GSE118767/suppl/GSE118767_RAW.tar>; <https://github.com/LuyiTian/CellBench_data> | GEO `GSE118767`: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118767>; CellBench package: <https://bioconductor.org/packages/CellBench> | `GSE118767_RAW.tar`; processed SingleCellExperiment objects from the GitHub/Bioconductor path. | The processed objects are often the most convenient starting point for integration metric calibration. |
| `int_multicenter_reference` | Figshare collection files: <https://springernature.figshare.com/collections/A_Multi-center_Cross-platform_Single-cell_RNA_Sequencing_Reference_Dataset/5213468>; SRA run selector via `SRP199641` | SRA `SRP199641`: <https://www.ncbi.nlm.nih.gov/sra/?term=SRP199641>; BioProject `PRJNA504037`: <https://www.ncbi.nlm.nih.gov/bioproject/PRJNA504037> | Figshare processed gene-count matrices and metadata first; SRA FASTQ only if remapping is required. | Figshare collection explicitly lists processed gene-count matrices and FASTQ metadata. |
| `int_hlca` | HCA atlas downloads: <https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0>; CELLxGENE collection: <https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293> | HLCA GitHub: <https://github.com/LungCellAtlas/HLCA> | HCA/CELLxGENE `.h5ad` or `.rds` objects; raw counts in `adata.raw.X` or `seurat_object@assays$RNA@counts`. | Do not use the published integrated embedding as agent input. It can be evaluator-only reference material. |
| `de_kang_ifnb` | <https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_RAW.tar> | GEO `GSE96583`: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583>; SRA `SRP102802`: <https://www.ncbi.nlm.nih.gov/sra/?term=SRP102802> | `GSE96583_RAW.tar`, especially batch 2 control/stim matrices and metadata. | Use the paired IFN-beta batch for the DE task. Preserve donor pairing. |
| `de_smillie_uc` | Single Cell Portal download tab: <https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-of-the-human-colon-during-ulcerative-colitis>; filename inventory: <https://github.com/cssmillie/ulcerative_colitis> | Same as direct endpoint. | Portal downloads for `all.meta2.txt`, epithelial/stromal/immune matrices, and optional Seurat discovery-cohort objects described in the repo. | The GitHub README lists expected matrix filenames; the portal is the canonical data source. |
| `de_stephenson_covid` | HCA project download/export page: <https://explore.data.humancellatlas.org/projects/b963bd4b-4bc1-4404-8425-69d74bc636b8>; BioStudies API: <https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-10026> | BioStudies page: <https://www.ebi.ac.uk/biostudies/studies/E-MTAB-10026>; analysis repo: <https://github.com/scCOVID-19/COVIDPBMC/> | HCA `.h5ad`, `.csv`, and `.xlsx` files; ArrayExpress accession for processed data. | HCA lists access as granted. Audit controlled-access components before using receptor/protein data. |

## Scriptable Source Hints

Use these only after confirming access terms and required files. The commands
show location discovery or single-file downloads; they intentionally do not
materialize the full benchmark portfolio.

```bash
mkdir -p \
  data/raw/paper-datasets/ann_tabula_sapiens \
  data/raw/paper-datasets/ann_kidney_injury \
  data/raw/paper-datasets/ann_pancreas_crossstudy \
  data/raw/paper-datasets/int_cellbench_mixology \
  data/raw/paper-datasets/de_kang_ifnb

# Direct GEO supplementary downloads
curl -L -o data/raw/paper-datasets/ann_tabula_sapiens/GSE201333_RAW.tar \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE201nnn/GSE201333/suppl/GSE201333_RAW.tar
curl -L -o data/raw/paper-datasets/ann_kidney_injury/GSE183279_RAW.tar \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE183nnn/GSE183279/suppl/GSE183279_RAW.tar
curl -L -o data/raw/paper-datasets/ann_pancreas_crossstudy/GSE84133_RAW.tar \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84133/suppl/GSE84133_RAW.tar
curl -L -o data/raw/paper-datasets/ann_pancreas_crossstudy/GSE85241_cellsystems_dataset_4donors_updated.csv.gz \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/GSE85241/suppl/GSE85241_cellsystems_dataset_4donors_updated.csv.gz
curl -L -o data/raw/paper-datasets/ann_pancreas_crossstudy/GSE85241_cel-seq_barcodes.csv.gz \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/GSE85241/suppl/GSE85241_cel-seq_barcodes.csv.gz
curl -L -o data/raw/paper-datasets/int_cellbench_mixology/GSE118767_RAW.tar \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/GSE118767/suppl/GSE118767_RAW.tar
curl -L -o data/raw/paper-datasets/de_kang_ifnb/GSE96583_RAW.tar \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/GSE96583_RAW.tar

# Tabula Sapiens AWS bucket listing
aws s3 ls --no-sign-request s3://czb-tabula-sapiens/

# Human Brain Cell Atlas public AWS directories
aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/expression_matrices/WHB-10Xv3/20240330/
aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/metadata/WHB-10Xv3/20241115/
aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/mapmycells/WHB-10Xv3/20240831/

# EBI BioStudies metadata APIs
curl -L https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-5061
curl -L https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-10026
```

When using GEO, keep the accession page alongside the direct FTP URL in the
dataset-specific `README.md`; the page is the better citation and access-date
record.

## Materialization Warnings

- Do not download full atlas-scale files until the split and subset recipe is
  frozen.
- Do not expose cell-type labels, marker tables, published embeddings, donor IDs
  used only for splitting, or hidden reference results to agents unless the task
  explicitly requires them.
- For disease and DE datasets, audit whether diagnosis, center, batch, and
  technology are confounded before running any baseline.
- Save checksums immediately after download and record access dates in each
  dataset directory.
