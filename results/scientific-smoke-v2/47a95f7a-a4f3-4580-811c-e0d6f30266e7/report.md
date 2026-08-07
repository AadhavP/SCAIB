# Scientific Benchmark Report: PBMC Cell-Type Annotation

- Benchmark: pbmc-cell-annotation
- Run: 47a95f7a-a4f3-4580-811c-e0d6f30266e7
- Dataset: scanpy.datasets.pbmc68k_reduced (100 cells x 765 genes)
- Pipeline: pbmc_default

## Objective metrics

| Metric | Status | Value | Evidence |
| --- | --- | ---: | --- |
| Adjusted Rand index (annotation_ari) | succeeded | 0.2777835656588808 | obs.bulk_labels; obs.louvain |
| Normalized mutual information (annotation_nmi) | succeeded | 0.5968761067224851 | obs.bulk_labels; obs.louvain |
| Annotation silhouette (annotation_silhouette) | succeeded | 0.48887532763183117 | obsm embedding; obs.louvain |

**Final score:** 0.45451166667106574

## Executed trajectory

| Step | Operation | Status | Wall time (s) |
| ---: | --- | --- | ---: |
| 1 | qc_filter | succeeded | 0.034 |
| 2 | normalize | succeeded | 0.047 |
| 3 | select_hvg | succeeded | 0.034 |
| 4 | pca | succeeded | 0.180 |
