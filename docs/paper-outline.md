# SCAIB Paper Outline

## Working Title

**SCAIB: Decision-Centric Evaluation of Autonomous Agents for Single-Cell Analysis**

## Main Idea

Current benchmarks mainly ask whether an agent completed a task or produced the correct final result. SCAIB also evaluates whether the agent made scientifically valid decisions along the way.

SCAIB scores:

1. final scientific outcome;
2. method and parameter choices;
3. decision trajectory;
4. reliability across repeated runs;
5. runtime and cost.

## Paper Sections

### 1. Introduction

- Single-cell analysis requires many connected scientific decisions.
- AI agents can now execute these workflows, but they remain unreliable.
- Final-answer scoring does not explain why an agent succeeded or failed.
- SCAIB connects decisions, intermediate artifacts, and final outcomes.

### 2. Related Work

Briefly compare SCAIB with:

- single-cell method benchmarks such as scIB and Open Problems;
- single-cell agents such as CellAgent and CASSIA;
- agent benchmarks such as scBench, scBench-Long, and GeneBench-Pro.

Do not claim that SCAIB is the first single-cell agent benchmark. Its difference is structured decision evaluation within a framework-neutral, replayable environment.

### 3. SCAIB Framework

Describe:

- declarative benchmark definitions;
- provider-neutral agent interface;
- scientific environment and tools;
- structured decisions and trajectories;
- validated scientific artifacts;
- stage-level and terminal metrics;
- reproducible reports.

### 4. Benchmark Design

The first release should focus on three workflow families:

1. cell annotation;
2. batch correction and integration;
3. donor-aware differential expression.

Each benchmark instance should define:

- dataset and checksum;
- scientific question;
- information visible to the agent;
- hidden reference data;
- allowed tools and actions;
- required output artifacts;
- metrics and scoring rules;
- runtime and token budget;
- random seeds.

Use train, development, and test splits separated by donor, study, tissue, or sequencing technology. Do not randomly split cells from the same donor across evaluation sets.

### 5. Experiments

Compare at least:

- fixed Scanpy pipeline;
- rule-based agent;
- two frontier-model agent systems;
- the same agent with and without the single-cell skill bundle;
- human expert performance on a smaller subset.

Minimum target:

| Item | Minimum |
| --- | ---: |
| Workflow families | 3 |
| Independent datasets | 10 |
| Agent configurations | 4 |
| Repeated runs | 3 per task |
| Expert-reviewed trajectories | 15 |

## Required Validation

A traditional model ablation is not necessary because this paper introduces a benchmark, not a new agent architecture. Instead, validate the benchmark in four ways.

### 1. Endpoint-Only Versus SCAIB Scoring

Score each run using:

- final outcome only;
- the complete SCAIB scorecard.

Show cases where similar final results came from decisions of different scientific quality.

### 2. Controlled Scientific Errors

Create workflows with known mistakes:

- excessive QC removes rare cells;
- unnecessary integration;
- integration removes biological signal;
- poor clustering resolution;
- forced annotation of an unknown population;
- differential expression ignores donor replication.

Test whether SCAIB lowers the correct score component and identifies the correct failure stage.

### 3. Expert Agreement

Have at least two single-cell experts score a subset of decisions. Compare their judgments with SCAIB's decision scores.

### 4. Reliability and Cost

Report:

- variation across repeated runs;
- completion and failure rates;
- invalid artifact rate;
- runtime;
- token and tool use;
- estimated monetary cost.

## Main Figures

### Figure 1: Framework overview

```text
Dataset -> Agent decision -> Tool execution -> Artifact -> Metrics -> Final outcome
```

### Figure 2: Benchmark inventory

A matrix showing datasets, task families, workflow stages, ground-truth type, and number of instances.

### Figure 3: Agent performance

A heatmap comparing systems across benchmark tasks.

### Figure 4: Outcome Versus Decision Quality

A scatter plot with final scientific performance on one axis and decision quality on the other.

### Figure 5: Failure Detection

A heatmap showing which SCAIB score component responds to each controlled scientific error.

### Figure 6: Reliability and Cost

Repeated-run distributions and a scientific-quality versus cost Pareto plot.

## Main Tables

1. Comparison with related benchmarks
2. Dataset and task inventory
3. Multidimensional agent leaderboard
4. Expert agreement and failure-detection results

## What Must Be Finished

- [ ] Finalize the annotation benchmark.
- [ ] Complete integration and differential-expression benchmarks.
- [x] Select a 10-dataset candidate portfolio.
- [ ] Materialize, audit, and freeze all 10 datasets and two reserves.
- [ ] Freeze metrics, artifact requirements, and data splits.
- [ ] Complete real GPT and Claude execution.
- [ ] Run agents with and without the skill bundle.
- [ ] Add controlled scientific failure cases.
- [ ] Obtain expert review of selected trajectories.
- [ ] Run at least three repetitions per task.
- [ ] Generate the six main figures.
- [ ] Add quantitative results to the abstract.

## Recommended Order

1. Select datasets.
2. Define the scientific question and ground truth for each dataset.
3. Freeze required artifacts and metrics.
4. Validate the benchmark with controlled errors.
5. Complete real-agent execution.
6. Run a small pilot.
7. Lock the benchmark.
8. Run the full experiment matrix.
9. Generate figures and write the results.

The next immediate task is materializing and auditing the 10-dataset candidate
portfolio: four annotation datasets, three integration datasets, and three
donor-aware differential-expression datasets. The portfolio and split rules are
defined in [SCAIB Paper Dataset Portfolio](paper-dataset-portfolio.md).
