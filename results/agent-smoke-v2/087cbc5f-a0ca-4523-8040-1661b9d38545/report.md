# Agent Evaluation Report

- Agent: rule-based
- Benchmark: pbmc-cell-annotation
- Run: 087cbc5f-a0ca-4523-8040-1661b9d38545
- Episode: ffe0da02-384a-4919-b782-7ae2875aade1

## Decision summary

| Step | Decision | Method | Execution | Local reward |
| ---: | --- | --- | --- | ---: |
| 1 | quality_control | qc_filter | succeeded | 0.7 |
| 2 | normalize | normalize | succeeded | 0.6999999995834332 |

## Scores

- Local decision score: 0.6999999997917166
- Final pipeline score: 0.48856986235348854

| Metric | Status | Score |
| --- | --- | ---: |
| Adjusted Rand index (annotation_ari) | succeeded | 0.3387795931977326 |
| Normalized mutual information (annotation_nmi) | succeeded | 0.603882830173753 |
| Annotation silhouette (annotation_silhouette) | succeeded | 0.5575244650244713 |
