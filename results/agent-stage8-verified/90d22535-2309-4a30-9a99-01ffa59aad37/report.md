# Agent Evaluation Report

- Agent: rule-based
- Benchmark: pbmc-cell-annotation
- Run: 90d22535-2309-4a30-9a99-01ffa59aad37
- Episode: 98ecf40f-d234-422d-81ab-08b69123b43b

## Decision summary

| Step | Decision | Method | Execution | Local reward |
| ---: | --- | --- | --- | ---: |
| 1 | quality_control | qc_filter | succeeded | 0.7 |
| 2 | normalize | normalize | succeeded | 0.7000000002600546 |

## Scores

- Local decision score: 0.7000000001300273
- Final pipeline score: 0.5141417557598752

| Metric | Status | Score |
| --- | --- | ---: |
| Adjusted Rand index (annotation_ari) | succeeded | 0.39130591114527297 |
| Normalized mutual information (annotation_nmi) | succeeded | 0.6170523295042291 |
| Annotation silhouette (annotation_silhouette) | succeeded | 0.5539922975003719 |

## Evaluation dimensions

- Scientific outcome score: 0.0
- Biology score: 0.0
- Technical score: None
- Robustness score: 1.0
- Decision score: 1.0
- Method score: 1.0
- Decision quality multiplier: 0.7236025216939351
- Trajectory score: 0.8076923076923077
- Final agent score: 0.0
- Global agent score: 0.0
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
| clustering.ari@1.0 | True | False | eligible |
| batch_integration.iLISI@1.0 | False | True | structural requirements unavailable: metadata:batch |
| batch_integration.graph_connectivity@1.0 | False | True | structural requirements unavailable: metadata:batch |

### Versioned metric results

| Metric | Status | Raw | Normalized |
| --- | --- | ---: | ---: |
| cell_annotation.macro_f1@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.mcc@1.0 | computed | 0.0 | 0.5 |
| cell_annotation.rare_recall@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.balanced_accuracy@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.accuracy@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.ece@1.0 | failed | - | 0.0 |
| clustering.ari@1.0 | computed | 0.39130591114527297 | 0.6956529555726365 |
| batch_integration.iLISI@1.0 | structurally_ineligible | - | - |
| batch_integration.graph_connectivity@1.0 | structurally_ineligible | - | - |

### Decision timeline

| Decision | Method | Appropriateness | Parameters | Execution | Overall |
| --- | --- | ---: | ---: | ---: | ---: |
| decision-1 | qc_filter | 1.000 | 1.000 | 0.171 | 0.724 |
| decision-2 | normalize | 1.000 | 1.000 | 0.171 | 0.724 |

### Local decision rewards

| Decision | Category | Value | Formula |
| --- | --- | ---: | --- |
| decision-1 | qc_strategy | 0.7 | decision_local_reward |
| decision-2 | normalization | 0.7000000002600546 | decision_local_reward |

### Trajectory analysis

- Method exploration score: 0.07692307692307693
- Alternative coverage: 0.07692307692307693
- Decision regret: 0.0
- Decision efficiency: 1.0
- Decision consistency: 1.0
- Adaptation ability: 1.0
- Counterproductive-action signal: 0.2999999997399454
- Short-term gain signal: 0.7000000002600546
- Long-term damage signal: 0.7000000002600546
- Good: all recorded action submissions followed the protocol, no contradictory transformations were detected
- Bad: local reward exceeded the final outcome, indicating a counterproductive signal
- Recommended improvement: validate local improvements against downstream scientific outcomes

### Scientific domain scores

| Domain | Score | Included | Excluded | Failed |
| --- | ---: | --- | --- | --- |
| biology | 0.0 | clustering.ari, cell_annotation.macro_f1, cell_annotation.rare_recall | - | - |
| technical | - | - | batch_integration.iLISI, batch_integration.graph_connectivity | - |
| robustness | 1.0 | robustness.seed_stability | - | - |

### Robustness

- Seeds: 0
- Seed stability: 1.0
- Clustering pairwise ARI: None
- Prediction agreement: None
