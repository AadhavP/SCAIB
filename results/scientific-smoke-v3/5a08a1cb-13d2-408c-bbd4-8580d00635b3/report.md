# Scientific Benchmark Report: PBMC Cell-Type Annotation

- Benchmark: pbmc-cell-annotation
- Run: 5a08a1cb-13d2-408c-bbd4-8580d00635b3
- Dataset: scanpy.datasets.pbmc68k_reduced (100 cells x 765 genes)
- Pipeline: pbmc_default

## Objective metrics

| Metric | Status | Value | Evidence |
| --- | --- | ---: | --- |
| Adjusted Rand index (annotation_ari) | succeeded | 0.29338967571087093 | obs.bulk_labels; obs.louvain |
| Normalized mutual information (annotation_nmi) | succeeded | 0.6019061467023405 | obs.bulk_labels; obs.louvain |
| Annotation silhouette (annotation_silhouette) | succeeded | 0.5182103756815195 | obsm embedding; obs.louvain |

**Final score:** 0.47116873269824366

## Executed trajectory

| Step | Operation | Status | Wall time (s) |
| ---: | --- | --- | ---: |
| 1 | qc_filter | succeeded | 0.029 |
| 2 | normalize | succeeded | 0.040 |
| 3 | select_hvg | succeeded | 0.021 |
| 4 | pca | succeeded | 0.169 |
