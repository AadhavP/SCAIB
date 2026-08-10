# Agent Evaluation Report

- Agent: mock
- Benchmark: pbmc-batch-correction
- Run: e28e1ab5-c9ce-46d0-bfe8-0660875f29d6
- Episode: 586e814c-9a69-4d22-a984-706b36e364a4
- Agent type: llm_tool_agent
- Model: z-ai/glm-5.2

## Decision summary

| Step | Decision | Method | Execution | Local reward |
| ---: | --- | --- | --- | ---: |
| 1 | - | normalize | failed | - |

## Scores

- Local decision score: unavailable
- Final pipeline score: unavailable

| Metric | Status | Score |
| --- | --- | ---: |
| Batch silhouette (batch_silhouette) | unavailable | - |
| Cell-type silhouette (cell_type_silhouette) | unavailable | - |

## Evaluation dimensions

- Scientific outcome score: 0.0
- Biology score: 0.0
- Technical score: None
- Robustness score: 1.0
- Decision score: 0.5
- Method score: 0.6666666666666666
- Decision quality multiplier: 0.2835386047986406
- Trajectory score: 0.7124999999999999
- Final agent score: 0.0
- Global agent score: 0.0
- Score formula: `scientific_outcome * decision_score * method_selection_score * trajectory_score`

### Applicability matrix

| Metric | Eligible | Structural exclusion | Reason |
| --- | --- | --- | --- |
| cell_annotation.macro_f1@1.0 | True | False | eligible |
| cell_annotation.mcc@1.0 | True | False | eligible |
| cell_annotation.balanced_accuracy@1.0 | True | False | eligible |
| cell_annotation.rare_recall@1.0 | True | False | eligible |
| cell_annotation.accuracy@1.0 | True | False | eligible |
| clustering.ari@1.0 | True | False | eligible |
| batch_integration.iLISI@1.0 | False | True | structural requirements unavailable: metadata:batch |
| batch_integration.graph_connectivity@1.0 | False | True | structural requirements unavailable: metadata:batch |

### Versioned metric results

| Metric | Status | Raw | Normalized |
| --- | --- | ---: | ---: |
| cell_annotation.macro_f1@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.mcc@1.0 | computed | 0.0 | 0.5 |
| cell_annotation.balanced_accuracy@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.rare_recall@1.0 | computed | 0.0 | 0.0 |
| cell_annotation.accuracy@1.0 | computed | 0.0 | 0.0 |
| clustering.ari@1.0 | computed | 0.4147795455021274 | 0.7073897727510637 |
| batch_integration.iLISI@1.0 | structurally_ineligible | - | - |
| batch_integration.graph_connectivity@1.0 | structurally_ineligible | - | - |

### Decision timeline

| Decision | Method | Appropriateness | Parameters | Execution | Overall |
| --- | --- | ---: | ---: | ---: | ---: |
| decision-1 | - | 0.500 | 1.000 | 0.201 | 0.567 |

### Local decision rewards

| Decision | Category | Value | Formula |
| --- | --- | ---: | --- |
| decision-1 | normalization | 0.0 | decision_local_reward |

### Trajectory analysis

- Method exploration score: 1.0
- Alternative coverage: 1.0
- Decision regret: 0.0
- Decision efficiency: 1.0
- Decision consistency: 1.0
- Adaptation ability: 1.0
- Counterproductive-action signal: 1.0
- Short-term gain signal: 0.0
- Long-term damage signal: 0.0
- Good: no contradictory transformations were detected
- Bad: none recorded
- Recommended improvement: none recorded

### Scientific domain scores

| Domain | Score | Included | Excluded | Failed |
| --- | ---: | --- | --- | --- |
| biology | 0.0 | clustering.ari, cell_annotation.macro_f1, cell_annotation.rare_recall | - | - |
| technical | - | - | batch_integration.iLISI, batch_integration.graph_connectivity | - |
| robustness | 1.0 | robustness.seed_stability | - | - |

### Robustness

- Seeds: 42
- Seed stability: 1.0
- Clustering pairwise ARI: None
- Prediction agreement: None
