# Agent Evaluation Report

- Agent: rule-based
- Benchmark: pbmc-cell-annotation
- Run: 2ba259ea-2049-4055-9b2f-bb24850f151e
- Episode: dc2991f5-0cc4-45ba-98ef-c88055a6d6dc

## Decision summary

| Step | Decision | Method | Execution | Local reward |
| ---: | --- | --- | --- | ---: |
| 1 | quality_control | qc_filter | succeeded | 0.7 |
| 2 | normalize | normalize | succeeded | 0.6999999995834332 |
| 3 | marker-genes | marker-genes | failed | - |

## Scores

- Local decision score: 0.6999999997917166
- Final pipeline score: 0.48856986235348854

| Metric | Status | Score |
| --- | --- | ---: |
| Adjusted Rand index (annotation_ari) | succeeded | 0.3387795931977326 |
| Normalized mutual information (annotation_nmi) | succeeded | 0.603882830173753 |
| Annotation silhouette (annotation_silhouette) | succeeded | 0.5575244650244713 |
