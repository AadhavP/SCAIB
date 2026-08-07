# Scientific Benchmark Report: PBMC Cell-Type Annotation

- Benchmark: pbmc-cell-annotation
- Run: 92a2aa84-263a-4b24-a773-e3b8ea8b19a2
- Dataset: scanpy.datasets.pbmc68k_reduced (100 cells x 765 genes)
- Pipeline: pbmc_default

## Objective metrics

| Metric | Status | Value | Evidence |
| --- | --- | ---: | --- |
| Adjusted Rand index (annotation_ari) | succeeded | 0.2777835656588808 | obs.bulk_labels; obs.louvain |
| Normalized mutual information (annotation_nmi) | succeeded | 0.5968761067224851 | obs.bulk_labels; obs.louvain |
| Annotation silhouette (annotation_silhouette) | succeeded | 0.5658256709575653 | obsm embedding; obs.louvain |

**Final score:** 0.4801617811129771

## Executed trajectory

| Step | Operation | Status | Wall time (s) |
| ---: | --- | --- | ---: |
| 1 | qc_filter | succeeded | 0.035 |
| 2 | normalize | succeeded | 0.045 |
| 3 | select_hvg | failed | 0.010 |

## Errors

- step 3 (select_hvg): ValueError: Bin edges must be unique: Index([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan],
      dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg
