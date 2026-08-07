# Agent Evaluation Report

- Agent: rule-based
- Benchmark: pbmc-cell-annotation
- Run: cefedfd5-de0e-4b0e-a370-ac5a80cbdffd
- Episode: 917ca81e-d7d7-45e3-9dbc-c6d625fa968c

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

- Scientific outcome score: 0.125
- Decision score: 1.0
- Method score: 1.0
- Decision quality multiplier: 0.6944444444444445
- Trajectory score: 0.9125
- Global agent score: 0.07921006944444445
- Score formula: `scientific_outcome * decision_score * method_selection_score * trajectory_score`

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
| cell_annotation.ece@1.0 | failed | - | 0.0 |

### Decision timeline

| Decision | Method | Appropriateness | Parameters | Execution | Overall |
| --- | --- | ---: | ---: | ---: | ---: |
| decision-1 | qc_filter | 1.000 | 1.000 | 0.083 | 0.694 |
| decision-2 | normalize | 1.000 | 1.000 | 0.083 | 0.694 |

### Local decision rewards

| Decision | Category | Value | Formula |
| --- | --- | ---: | --- |
| decision-1 | qc_strategy | 0.7 | decision_local_reward |
| decision-2 | normalization | 0.6999999995834332 | decision_local_reward |

### Trajectory analysis

- Decision efficiency: 1.0
- Decision consistency: 1.0
- Adaptation ability: 1.0
- Counterproductive-action signal: 0.42500000000000004
- Short-term gain signal: 0.7
- Long-term damage signal: 0.575
- Good: all recorded action submissions followed the protocol, no contradictory transformations were detected
- Bad: local reward exceeded the final outcome, indicating a counterproductive signal
- Recommended improvement: validate local improvements against downstream scientific outcomes
