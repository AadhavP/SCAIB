# Scientific metric and agent evaluation

The evaluation layer separates four observables that should not be collapsed into one opaque reward:

- scientific outcome: versioned metric definitions, raw values, normalized values, applicability, and metric groups;
- decision quality: whether each submitted action was permitted, applicable, executed, and supported by artifacts;
- method quality: whether the selected method produced valid downstream evidence;
- trajectory quality: deterministic protocol, dependency, artifact, efficiency, and outcome-alignment dimensions.

Metric applicability is decided before computation. Missing benchmark structure, such as reference labels or a required batch variable, produces `structurally_ineligible` and removes that metric from the group denominator. Missing candidate evidence, such as confidence values for calibration, produces an eligible candidate failure with the metric's declared failure score. Candidate-specific weights are never renormalized.

The PBMC primary group uses the weighted mean:

`0.35*macro_f1 + 0.25*mcc + 0.20*rare_recall + 0.10*balanced_accuracy + 0.10*accuracy`

The calibration group is reported separately and does not contribute to the primary scientific outcome. The global agent score is described below.

Every computed metric records its implementation backend and metric version. Candidate prediction tables are materialized as evaluator-owned artifacts and are not added to the agent's episode observations. Biological reference labels remain hidden from the agent.

This milestone intentionally does not perform counterfactual evaluation or infer causal claims from trajectories. It evaluates only recorded actions, environment outcomes, artifacts, resource measurements, and declared benchmark requirements.

## Stage 8 metric engine and benchmark profiles

The generic metric API lives under `agent_evals.evaluation.metrics`. It exposes a strict `ScientificMetric` contract, a registry, applicability evidence, raw/native values, normalized values, implementation versions, and adapter-backed annotation, clustering, integration, embedding, and differential-expression metrics. The catalog reuses the existing open-source backends; optional scIB-style metrics remain optional and are never silently substituted when required structure is absent.

Benchmark profiles are versioned YAML contracts under `configs/metrics`. They freeze metric membership and weights by domain. Domain scores use a weighted geometric mean: required candidate failures remain zero-valued, while structurally ineligible optional metrics are excluded with an explicit reason. Domain scores then combine into the scientific outcome using the profile's frozen domain weights.

The Stage 8 report also records seed robustness (when replicate artifacts are available), method exploration, alternative coverage, unnecessary retries, decision regret, and reference baselines (`scanpy_default`, `seurat_reference`, `random_agent`, and evaluator-only `oracle_agent`). OpenHands execution, counterfactual rollouts, LLM judges, and frontend work remain outside this milestone.

## Stage 7 decision intelligence

Each action is also represented as a structured decision cascade. The cascade exposes a scientific decision category, intent and hypothesis fields, chosen method, chosen parameters, evidence references, alternatives, confidence, expected effects, and downstream dependencies. These are explicit agent outputs; private chain-of-thought is not collected or scored.

The decision ontology is reusable across benchmarks. It includes data loading, QC strategy, normalization, feature selection, dimensionality reduction, integration, clustering, annotation, differential expression, and interpretation. A benchmark can override allowed methods, expected inputs, alternatives, evaluator metrics, and parameter ranges under `decision_evaluation`.

Method selection is decomposed into appropriateness, parameter quality, and execution quality. Local decision rewards use category-specific declared formulas. Trajectory intelligence reports efficiency, consistency, adaptation, and an observable local-reward/final-outcome gap. That gap is a diagnostic signal only; it is not a causal or counterfactual claim.

The global agent score is multiplicative:

`scientific_outcome * decision_quality * trajectory_quality`

where `decision_quality = decision_score * method_selection_score`. This prevents a high scientific outcome from masking weak observable decision and trajectory quality.
