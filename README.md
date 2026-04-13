# SART / LayerABS Artifact

Code and experiment launcher for the paper.

## Paper Information

**Title**  
SART: Sign-Absolute Reformulation Theory for Binary Variable Reduction in Neural Network Verification

**Venue**  
Accepted at **OOPSLA 2026**

**Authors**  
Jin Xu, Miaomiao Zhang, and Bowen Du

**Affiliation**  
Tongji University, China

**DOI**  
[10.1145/3798237](https://doi.org/10.1145/3798237)

**Abstract**

> Complete formal verification of neural networks is crucial for their deployment in safety-critical domains. A key bottleneck stems from encoding complexity: traditional methods assign one binary variable per unstable ReLU neuron. We propose the Sign-Absolute Reformulation Theory (SART), which fundamentally breaks the conventional one-to-one mapping between unstable neurons and binary variables by establishing formal reducibility criteria. This allows for finer-grained modeling, where each unstable neuron corresponds on average to fewer than one binary variable, thereby reducing verification complexity at its source. Based on SART, we derive a theoretical lower bound on the number of binary variables required for complete verification and, under the assumption that P != NP, prove that variables in the final layer can be compressed by 50%, while the number of variables in intermediate layers cannot be further reduced. To overcome the apparent "last-layer-only" limitation, we recast verification as a sequential process and, crucially, show that the gain lifts to the entire network: LayerABS, a SART-based progressive tightening verifier, iteratively treats intermediate layers as temporary final layers and propagates tight bounds that shrink the global search space and binary-variable counts. Furthermore, we reveal a structural law influencing verification complexity: when the signs of weights of unstable neurons satisfy numerical symmetry, with positive and negative weights equal or differing by at most one, the worst-case verification complexity achieves the theoretical optimum, offering theoretical guidance for the design of verification-friendly architectures. As a general-purpose underlying encoding, the value of SART is independent of specific algorithms. To comprehensively evaluate its effectiveness, we first evaluate the abstraction-free SART encoding, and then integrate it with abstraction techniques to construct the complete verifier LayerABS and its incomplete variant Incomplete-LayerABS. Across benchmarks, our methods surpass state-of-the-art baselines, validating SART's practical impact.

**Supplementary Material**

- [SART_Supplementary_Material.pdf](SART_Supplementary_Material.pdf)

This repository has been cleaned into a single active maintenance path. The main LayerABS families, their paper-facing controllers, the benchmark/model variants, the paper presets, and the result summarization/export tools are all kept in-tree. Historical copies, one-off branches, and cached runtime artifacts were removed from the active layout.

## What This Repository Contains

- the refactored LayerABS family controllers and shared algorithm code under `sart/layerabs/`
- baseline verifiers and related scripts under `sart/verify/`
- benchmark models under `sart/models/`
- benchmark property files under `sart/*_properties/` and `sart/mnist_vnnlib/`
- conversion and preparation utilities under `sart/conversion_tools/`
- a root launcher, `run_experiment.py`, that can list scripts, run family controllers, run named paper presets, summarize completed runs, and export paper-table artifacts
- tests under `tests/`

External baseline frameworks such as PRIMA, Marabou, MIPVerify, and beta-CROWN are intentionally not integrated into the unified launcher. The in-tree paper presets focus on the repository's own LayerABS-family methods.

## Setup

### Python Environment

Recommended baseline:

- Python `3.9+`
- dependencies from `requirements.txt`
- a working Gurobi installation and license for the MILP-based paths

Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Conda instead of `venv`, use your existing environment workflow and install the same requirements there.

### What Requires Gurobi

Gurobi is required for:

- `LayerABS_abstract_milp`
- `LayerABS_standard_milp`
- `LayerABS_incomplete_layerabs`
- the complete/stats/time-limit family paths that invoke MILP refinement or complete solving

Some other verifier scripts under `sart/verify/` also depend on Gurobi or other external packages.

## Repository Layout

```text
.
├── README.md
├── run_experiment.py
├── paper_presets.py
├── paper_results.py
├── experiment_catalog.py
├── layerabs_naming.py
├── sart/
│   ├── models/
│   ├── mnist_properties/
│   ├── cifar_properties/
│   ├── acasxu_properties/
│   ├── vnncomp_eran_properties/
│   ├── mnist_vnnlib/
│   ├── conversion_tools/
│   ├── verify/
│   ├── result/
│   └── layerabs/
│       ├── LayerABS_*.py
│       ├── default_profiles/
│       ├── family_wrappers/
│       ├── layerabs_core/
│       ├── layerabs_variants/
│       └── support/
├── docs/
│   ├── code_classification.md
│   ├── layerabs_families.md
│   └── paper_tables/
└── tests/
```

Practical navigation rule:

- start from the root `LayerABS_*.py` files when you want the clean paper-facing entrypoint
- use `default_profiles/` when you want the repository's default benchmark/network wrapper for a family
- use `family_wrappers/` when you want another benchmark or network in an existing family
- edit `layerabs_core/` when you need shared algorithm logic
- edit `layerabs_variants/` when you only need to change family configuration, benchmark coverage, or default parameters

More detail for the `layerabs/` subtree is in [sart/layerabs/README.md](sart/layerabs/README.md).

## Method Families

The active LayerABS-side methods now follow the paper semantics directly.

### Abstraction-Enabled Complete Methods

- `LayerABS_abstract_sart.py`: abstraction-enabled complete `LayerABS(SART)`
- `LayerABS_abstract_milp.py`: abstraction-enabled complete `LayerABS(MILP)`

These are the two complete methods in the paper where abstraction is present and the final complete stage uses either SART encoding or MILP encoding.

### Incomplete Method

- `LayerABS_incomplete_layerabs.py`: `Incomplete-LayerABS`

This controller follows the paper definition directly:

- Stage 1: abstract interpretation filter
- Stage 2: hybrid abstract SART encoding
- no Stage 3 complete fallback

The controller exposes the paper's Stage 2 depth parameter `k` through `--k-layers`. The current default is `k=2`.

### No-Abstraction Baselines

- `LayerABS_puresart.py`: `PureSART`
- `LayerABS_standard_milp.py`: `Standard MILP`

These are the no-abstraction baseline families.

### Statistics Families

- `LayerABS_abstract_sart_stats.py`
- `LayerABS_abstract_milp_stats.py`
- `LayerABS_puresart_stats.py`
- `LayerABS_standard_milp_stats.py`

These families are used for runtime, unstable-neuron, and related statistics experiments.

### Time-Limit Family

- `LayerABS_abstract_sart_timelimit.py`

This is the abstraction-enabled SART family for the time-limit experiments.

## Root Controllers vs. Default Profiles

The repository now keeps only model-neutral controllers at the root of `layerabs/`.

### Root Controllers

| Script | Role |
| --- | --- |
| `LayerABS_abstract_sart.py` | paper-facing controller for abstraction-enabled complete `LayerABS(SART)` |
| `LayerABS_abstract_milp.py` | paper-facing controller for abstraction-enabled complete `LayerABS(MILP)` |
| `LayerABS_incomplete_layerabs.py` | paper-facing controller for `Incomplete-LayerABS` |
| `LayerABS_puresart.py` | paper-facing controller for `PureSART` |
| `LayerABS_standard_milp.py` | paper-facing controller for `Standard MILP` |
| `LayerABS_abstract_sart_stats.py` | stats controller for abstraction-enabled `LayerABS(SART)` |
| `LayerABS_abstract_milp_stats.py` | stats controller for abstraction-enabled `LayerABS(MILP)` |
| `LayerABS_puresart_stats.py` | stats controller for `PureSART` |
| `LayerABS_standard_milp_stats.py` | stats controller for `Standard MILP` |
| `LayerABS_abstract_sart_timelimit.py` | time-limit controller for abstraction-enabled `LayerABS(SART)` |

### Default Profiles

The benchmark/network-suffixed paper-default wrappers live under `sart/layerabs/default_profiles/`.

Current default profiles are:

- `LayerABS_abstract_sart_mnist_10x80.py`
- `LayerABS_abstract_milp_mnist_5x50.py`
- `LayerABS_incomplete_layerabs_mnist_10x80.py`
- `LayerABS_puresart_mnist_10x80.py`
- `LayerABS_standard_milp_mnist_10x80.py`
- `LayerABS_abstract_sart_stats_mnist_6x100.py`
- `LayerABS_abstract_milp_stats_mnist_6x100.py`
- `LayerABS_puresart_stats_vnncomp_6x100.py`
- `LayerABS_standard_milp_stats_vnncomp_6x100.py`
- `LayerABS_abstract_sart_timelimit_mnist_10x80.py`

Use the root controller when you want to select the variant explicitly. Use a default-profile script only when you want the repository's default benchmark/network wrapper or a stable paper-default path.

## Variant Naming

Variant names are now cleaner and paper-facing. Typical examples:

- `mnist_5x50`
- `mnist_5x80`
- `mnist_6x100`
- `mnist_9x100`
- `mnist_10x80`
- `mnist_9x200`
- `vnncomp_6x100`
- `vnncomp_9x100`
- `cifar10_5x50`
- `cifar10_6x80`

VNN-COMP note:

- the paper's ERAN `5x100` and `8x100` correspond to the code variants `vnncomp_6x100` and `vnncomp_9x100`
- the code counts the output layer in the architecture label, while the paper names those models without counting the output layer

## Quick Start

### List Available Scripts

```bash
python run_experiment.py --list
python run_experiment.py --list --filter LayerABS_
```

### List LayerABS Entrypoints and Families

```bash
python run_experiment.py --list-layerabs
python run_experiment.py --list-layerabs-families
```

Useful filters:

```bash
python run_experiment.py --list-layerabs --family abstract_sart
python run_experiment.py --list-layerabs --bucket family_wrappers/abstract_sart
python run_experiment.py --list-layerabs --content-group ablation
```

### Run a Family Controller

```bash
python run_experiment.py --script LayerABS_abstract_sart --script-arg=--variant --script-arg=mnist_5x50
python run_experiment.py --script LayerABS_abstract_milp --script-arg=--variant --script-arg=mnist_5x50
python run_experiment.py --script LayerABS_puresart --script-arg=--variant --script-arg=mnist_10x80
python run_experiment.py --script LayerABS_standard_milp --script-arg=--variant --script-arg=mnist_10x80
```

### Run Incomplete-LayerABS with an Explicit `k`

```bash
python run_experiment.py --script LayerABS_incomplete_layerabs \
  --script-arg=--variant --script-arg=mnist_5x50 \
  --script-arg=--k-layers --script-arg=3
```

### Preview a Resolved Run Without Executing It

```bash
python run_experiment.py --script LayerABS_abstract_sart --script-arg=--variant --script-arg=vnncomp_6x100 --dry-run
python run_experiment.py --paper-preset table8_vnncomp_complete --dry-run
```

### Run a Verify Script

```bash
python run_experiment.py --script sart/verify/mnist_new_5x50/deeppoly_mnist_new_5x50.py
```

## Launcher Features

The root launcher, `run_experiment.py`, is the normal way to interact with the repository.

It supports:

- listing all runnable scripts
- listing LayerABS scripts and families
- filtering by family, benchmark, content group, or bucket
- running a script by path, normalized name, or experiment id
- running named paper presets
- summarizing completed paper presets
- exporting paper-table summaries in bulk

Common commands:

```bash
python run_experiment.py --help
python run_experiment.py --list-paper-presets
python run_experiment.py --paper-preset table4_layerabs_complete --dry-run
python run_experiment.py --summarize-paper-preset table4_layerabs_complete
python run_experiment.py --summarize-paper-preset table4_layerabs_complete --summary-format json
python run_experiment.py --export-paper-tables
```

The plain `--list` output distinguishes:

- `layerabs_family_controller`
- `layerabs_default_profile`
- `thin_wrapper`
- `verify_script`

The `--list-layerabs` output also includes the paper-facing `paper_role`.

The `--list-layerabs-families` output shows both:

- `path=` for the canonical default profile
- `controller=` for the recommended root family controller

## Paper Presets

The repository includes named presets for the paper's internal LayerABS-family experiments.

### Current Presets

- `table2_no_abstraction`
- `table3_vnncomp_hard_cases`
- `table4_layerabs_complete`
- `table5_fallback_frequency`
- `table6_incomplete_mnist`
- `table7_sart_vs_milp_ablation`
- `table8_vnncomp_complete`
- `table9_incomplete_vnncomp_k3`
- `table10_incomplete_vnncomp_k_sweep`
- `table11_incomplete_vnncomp_ldsa_k3`
- `table12_incomplete_vnncomp_ldsa_k_sweep`

### Coverage Labels

- `supported`: directly runnable from the current public controller path
- `partial`: the in-repo LayerABS side is runnable, but the full paper table depends on excluded external baselines or settings
- `unsupported`: documented from the paper, but the required runtime mode is not exposed in the current public controller path

### Examples

```bash
python run_experiment.py --list-paper-presets
python run_experiment.py --paper-preset table3 --dry-run
python run_experiment.py --paper-preset table6 --dry-run
python run_experiment.py --paper-preset table10 --dry-run
```

Important preset notes:

- `table3_vnncomp_hard_cases` is supported and encodes the paper's seven hard VNN-COMP property IDs with the paper timeout
- `table6_incomplete_mnist` and `table9_incomplete_vnncomp_k3` apply the paper's `30s` per-solve Gurobi fairness limit through `--mip-time-limit 30`
- `table11` and `table12` remain unsupported because the current public path does not expose the LDSA alternation mode used in those tables

The preset definitions live in [paper_presets.py](paper_presets.py).

## Result Summaries and Paper Table Export

The repository can summarize completed preset runs and export them in machine-readable formats.

### Summarize a Single Preset

```bash
python run_experiment.py --summarize-paper-preset table4
python run_experiment.py --summarize-paper-preset table6 --summary-format json
python run_experiment.py --summarize-paper-preset table7 --summary-format tsv
```

Available summary formats:

- `text`
- `json`
- `tsv`

You can also write the summary directly to a file:

```bash
python run_experiment.py --summarize-paper-preset table4 --summary-format tsv --summary-output table4.tsv
```

### Export All Current Paper Tables

```bash
python run_experiment.py --export-paper-tables
python run_experiment.py --export-paper-tables docs/paper_tables
python run_experiment.py --export-paper-tables --filter table6
```

This export produces:

- one `json` file per preset
- one `tsv` file per preset
- `index.json`
- `index.tsv`

The default export directory is [docs/paper_tables/](docs/paper_tables/).

The result summarizer lives in [paper_results.py](paper_results.py).

### Table 5 Note

`table5_fallback_frequency` now depends on `StageOutcome:` markers written by the verification runner. If the existing logs were generated before those markers were added, the summary will report `stage_metrics=missing`. In that case, rerun the preset before summarizing:

```bash
python run_experiment.py --paper-preset table5
python run_experiment.py --summarize-paper-preset table5
```

## Outputs and Artifacts

Runtime outputs are written under [sart/result/](sart/result/).

Typical locations:

- `sart/result/log/`: per-run logs
- `sart/result/original_result/`: textual result files written by the experiment scripts
- `docs/paper_tables/`: exported preset summaries in `json` and `tsv`

The launcher's preset summary mode matches the newest compatible local artifacts for each preset run. For most families it prefers a matching `result + log` timestamp pair to avoid mixing results across methods.

## Main Code Paths

### `sart/layerabs/layerabs_core/`

This directory holds the shared implementation.

Important modules include:

- `layerabs_abstract_sart_family_propagation.py`
- `layerabs_abstract_milp_family_propagation.py`
- `layerabs_incomplete_family_propagation.py`
- `layerabs_no_abstraction_propagation.py`
- `layerabs_abstract_sart_stats_family_propagation.py`
- `layerabs_abstract_milp_stats_family_propagation.py`
- `layerabs_abstract_sart_timelimit_family_propagation.py`
- `layerabs_verification_runners.py`
- `layerabs_solver_helpers.py`
- `layerabs_shared_helpers.py`
- `layerabs_stage_helpers.py`
- `layerabs_runtime_helpers.py`
- `layerabs_stats_family_helpers.py`
- `layerabs_logging_helpers.py`
- `layerabs_process_helpers.py`
- `layerabs_parallel_task_helpers.py`
- `layerabs_parallel_args_helpers.py`

### `sart/layerabs/layerabs_variants/`

This directory defines which benchmark/network variants each family supports. It is the normal place to edit:

- variant names
- default deltas
- model/property bindings
- special paper slices such as the Table 3 hard-case run

### `sart/layerabs/family_wrappers/`

This directory holds thin wrappers for additional benchmark/network variants under each family. These files should remain small and should not carry duplicated algorithm logic.

### `sart/verify/`

This directory holds the other verifier entrypoints that are not part of the LayerABS family-controller layer.

## Reproducibility Notes

- active methods are kept in the main tree
- historical copies, one-off branches, archived runtime dumps, `specialized/`, `legacy/`, and `non_code_assets/` were removed from the active repository structure
- `__pycache__/` and runtime `fdl/` snapshots are ignored and should not be committed
- the active mainline now uses paper-facing method names consistently: `abstract_sart`, `abstract_milp`, `incomplete_layerabs`, `puresart`, `standard_milp`

## IDE Notes

### PyCharm

If PyCharm still shows red import underlines:

1. make sure the project uses the same interpreter/environment that you use to run experiments
2. mark the repository root as a source root if needed
3. invalidate IDE caches if stale analysis remains

The active LayerABS code path was refactored to reduce import-time side effects, so remaining IDE red lines are now much more likely to be interpreter or cache issues rather than broken local structure.

## Additional Documentation

- [sart/layerabs/README.md](sart/layerabs/README.md): layout and entrypoint map for the LayerABS code tree
- [docs/layerabs_families.md](docs/layerabs_families.md): family guide and refactor layout
- [docs/code_classification.md](docs/code_classification.md): code classification and directory roles
- [docs/paper_tables/README.md](docs/paper_tables/README.md): exported paper-table artifacts

## License

This repository is released under the [MIT License](LICENSE).
