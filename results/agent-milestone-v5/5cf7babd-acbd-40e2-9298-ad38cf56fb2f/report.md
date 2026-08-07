# Agent Evaluation Report

- Agent: rule-based
- Benchmark: pbmc-cell-annotation
- Run: 5cf7babd-acbd-40e2-9298-ad38cf56fb2f
- Episode: 90685714-b473-4a7c-9c00-c6e1dc326c17

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

## Evaluation dimensions

- Scientific outcome score: 0.0625
- Decision score: 1.0
- Method score: 1.0
- Trajectory score: 0.859375
- Benchmark score: 0.3296875
- Score formula: `0.70*scientific_outcome + 0.10*decision + 0.10*method + 0.10*trajectory`

### Applicability matrix

| Metric | Eligible | Structural exclusion | Reason |
| --- | --- | --- | --- |
| cell_annotation.macro_f1@1.0 | True | False | eligible |
| cell_annotation.mcc@1.0 | True | False | eligible |
| cell_annotation.rare_recall@1.0 | True | False | eligible |
| cell_annotation.balanced_accuracy@1.0 | True | False | eligible |
| cell_annotation.accuracy@1.0 | True | False | eligible |
| cell_annotation.ece@1.0 | True | False | eligible |

### Versioned metric results

| Metric | Status | Raw | Normalized |
| --- | --- | ---: | ---: |
| cell_annotation.macro_f1@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.mcc@1.0 | computed | 0.0 | 0.5 |
| cell_annotation.rare_recall@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.balanced_accuracy@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.accuracy@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.ece@1.0 | computed | - | - |
