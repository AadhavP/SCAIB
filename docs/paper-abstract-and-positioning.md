# SCAIB: Decision-Centric Evaluation of Autonomous Agents for Single-Cell Analysis

## Abstract

Large language model agents are increasingly capable of executing single-cell RNA-sequencing analyses, yet their evaluation remains fragmented. Method-centered benchmarks rigorously score isolated scientific outputs, whereas emerging agent benchmarks emphasize task completion, executable workflows, or recovery of terminal biological conclusions. These approaches reveal whether an agent succeeds, but provide limited standardized evidence about whether its observable intermediate decisions were scientifically appropriate, reproducible, and aligned with downstream outcomes. We present SCAIB, an extensible evaluation framework for autonomous agents in computational single-cell biology. SCAIB represents each run as a replayable trajectory linking dataset observations, scientific decisions, method and parameter selections, executions, resource use, and validated artifacts. A declarative benchmark specification freezes task definitions, agent-visible information, allowed actions, artifact contracts, hidden references, metric applicability, scoring profiles, and resource budgets. Evaluation is decomposed into stage-specific scientific performance, decision and method quality, trajectory quality, and task-specific terminal biological utility. Raw and normalized metrics remain visible, structurally ineligible metrics are distinguished from missing or malformed outputs, and frozen aggregation rules prevent candidate-specific reweighting. A provider-neutral harness enables matched comparison of models, agent scaffolds, and skill configurations under shared execution conditions. The current implementation provides a Scanpy-backed PBMC testbed and versioned evaluation profiles for cell annotation, integration, and differential expression, with annotation serving as the principal complete vertical slice. By making observable scientific decisions and their alignment with validated intermediate and terminal outcomes first-class benchmark objects, SCAIB complements endpoint-focused evaluations and provides a reproducible basis for diagnosing, comparing, and improving autonomous single-cell analysis systems.

## State of the Field

As of August 2026, the field has advanced beyond biological question answering. A recent survey identifies 58 foundation and agentic single-cell models, while emphasizing that evaluation remains fragmented across modalities, datasets, and notions of interpretability and trustworthiness ([LLM4Cell](https://aclanthology.org/2026.acl-long.1942/)).

| Direction | Representative work | What it does | Remaining opening for SCAIB |
| --- | --- | --- | --- |
| Method benchmarking | [scIB](https://www.nature.com/articles/s41592-021-01336-8), [Open Problems](https://www.nature.com/articles/s41587-025-02694-w) | Provides rigorous, task-specific metrics and reproducible comparison of single-cell algorithms | Evaluates methods and outputs, not the complete agent, its decisions, or its trajectory |
| Specialized agents | [CellAgent](https://arxiv.org/abs/2407.09811), [CASSIA](https://www.nature.com/articles/s41467-025-67084-x) | Automates workflows or specialized tasks such as annotation using multi-agent systems | Primarily evaluates a particular system rather than offering a neutral evaluation layer |
| Broad single-cell agent evaluation | [Benchmarking LLM-based agents for single-cell omics analysis](https://link.springer.com/article/10.1186/s13059-026-03998-z) | Compares multiple models and frameworks on 50 tasks using 18 planning, execution, retrieval, and outcome metrics | The authors identify agent decision-making as a remaining black box; per-task cost and latency are also not fully captured |
| Deterministic local evaluation | [scBench](https://arxiv.org/abs/2602.09063) | Offers 394 verifiable problems across six platforms and seven task categories, generally beginning immediately before an analysis step | Provides strong deterministic endpoint grading, but most problems evaluate a local result through structured pass/fail output |
| Long-horizon scientific evaluation | [GeneBench-Pro](https://cdn.openai.com/pdf/21938268-21af-442f-af93-3b2249afb241/genebench-pro.pdf), [scBench-Long](https://latch.bio/scbench-long) | Tests dependent statistical decisions and recovery of terminal scientific conclusions; scBench-Long adds hidden trajectory rubrics | Intermediate reasoning is largely inferred diagnostically around an endpoint score rather than represented as a reusable, structured scientific decision object |

## How SCAIB Fits into the Field

The strongest defensible positioning is:

> SCAIB bridges objective single-cell method benchmarking and long-horizon agent evaluation by jointly scoring validated scientific artifacts, observable method and parameter choices, trajectory behavior, and independently evaluated terminal biological outcomes under a common, provider-neutral harness.

The important distinctions are:

- **Decisions are first-class objects.** SCAIB records explicit, observable scientific decisions--not private chain-of-thought--including the selected method, parameters, evidence, alternatives, dependencies, execution status, and resulting artifacts.

- **Intermediate and terminal quality remain separate.** Stage-level metrics can identify a poor integration or QC decision even when the final answer happens to be acceptable. Conversely, scientifically reasonable local decisions do not excuse an incorrect terminal conclusion.

- **The score cannot hide weak reasoning.** The implemented global formulation couples scientific outcome, decision quality, and trajectory quality multiplicatively. Frozen applicability and weighting rules prevent an agent from benefiting by omitting difficult outputs.

- **The evaluation target is the complete submitted system.** Scores belong to a declared combination of model, scaffold, tools, skills, runtime, and resource budget--not to a model name in isolation.

- **Failures are localized without making unsupported causal claims.** Current trajectories support diagnostic localization and decision-outcome alignment. True causal attribution will require controlled ablations or the planned counterfactual replay system.

## Claims to Avoid in the Current Draft

SCAIB should not be described as the first single-cell agent benchmark, the first long-horizon benchmark, or as already outperforming the state of the art. Existing literature rules out the first two claims, and the repository does not yet contain the publication experiments required for the third.

The current metric architecture and provider-neutral harness are substantial, but real frontier-agent comparisons, broader datasets, completed integration and differential-expression evaluations, human baselines, and counterfactual experiments remain unfinished. The scAutoML concept should therefore remain a separate paper, as proposed in the meeting notes.

## Quantitative Results Sentence to Add Before Submission

The final abstract should include a quantitative results sentence after the publication experiments are complete. A suitable structure is:

> Across **[N] agents**, **[M] datasets**, and **[R] repeated runs**, SCAIB identified **[X] classes of decision-level failure** that were not distinguishable from terminal scores alone, while revealing significant differences in robustness, efficiency, and scientific decision quality under matched execution budgets.

This sentence should only be completed once the corresponding experiments have been run.

## Sources Reviewed

### Project Sources

- `C:\Users\tdech\Downloads\Agents 4 Single Cell Team Meeting Notes.pdf`
- `C:\Users\tdech\.codex\attachments\317a07ab-a2c4-4912-99dd-46dc8c3cc843\pasted-text.txt`
- `docs/metric-evaluation.md`
- `docs/agent-harness.md`
- `docs/evaluation.md`
- `examples/benchmarks/pbmc-cell-annotation.yaml`

### Related Literature

- Dip et al. [LLM4Cell: Taxonomy and Evaluation of LLM and Agentic Models for Single-Cell Biology](https://aclanthology.org/2026.acl-long.1942/).
- Luecken et al. [Benchmarking atlas-level data integration in single-cell genomics](https://www.nature.com/articles/s41592-021-01336-8).
- Luecken et al. [Defining and benchmarking open problems in single-cell analysis](https://www.nature.com/articles/s41587-025-02694-w).
- Xiao et al. [CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis](https://arxiv.org/abs/2407.09811).
- Ghazanfar et al. [CASSIA: A multi-agent large language model for automated and interpretable cell annotation](https://www.nature.com/articles/s41467-025-67084-x).
- Liu et al. [Benchmarking LLM-based agents for single-cell omics analysis](https://link.springer.com/article/10.1186/s13059-026-03998-z).
- Workman et al. [scBench: Evaluating AI Agents on Single-Cell RNA-seq Analysis](https://arxiv.org/abs/2602.09063).
- Li and Ho. [GeneBench-Pro: Evaluating Multistage Statistical Reasoning in Genomics, Quantitative Biology, and Translational Biomedicine](https://cdn.openai.com/pdf/21938268-21af-442f-af93-3b2249afb241/genebench-pro.pdf).
- Diks et al. [scBench-Long: Verifiable Benchmarking of Long-Horizon Single-Cell Biology](https://latch.bio/scbench-long).
