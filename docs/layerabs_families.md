# LayerABS Family Guide

This document describes the active LayerABS-family code path after the repository cleanup. It is intentionally limited to the maintained paper-facing families and their current runtime structure.

Use the root launcher as the single entrypoint:

```bash
python run_experiment.py --list-layerabs-families
python run_experiment.py --list-layerabs
python run_experiment.py --list-paper-presets
```

## Active Families

### Main Complete Families

| Family | Controller | Paper Meaning |
| --- | --- | --- |
| `abstract_sart` | `LayerABS_abstract_sart.py` | abstraction-enabled complete `LayerABS(SART)` |
| `abstract_milp` | `LayerABS_abstract_milp.py` | abstraction-enabled complete `LayerABS(MILP)` |

### Incomplete Family

| Family | Controller | Paper Meaning |
| --- | --- | --- |
| `incomplete_layerabs` | `LayerABS_incomplete_layerabs.py` | `Incomplete-LayerABS`, i.e. Stage 1 + Stage 2 without Stage 3 fallback |

`Incomplete-LayerABS` exposes the Stage 2 depth parameter `k` through `--k-layers`. The current default is `k=2`.

### No-Abstraction Baselines

| Family | Controller | Paper Meaning |
| --- | --- | --- |
| `puresart` | `LayerABS_puresart.py` | `PureSART` |
| `standard_milp` | `LayerABS_standard_milp.py` | `Standard MILP` |

### Statistics Families

| Family | Controller | Paper Meaning |
| --- | --- | --- |
| `abstract_sart_stats` | `LayerABS_abstract_sart_stats.py` | statistics branch for abstraction-enabled `LayerABS(SART)` |
| `abstract_milp_stats` | `LayerABS_abstract_milp_stats.py` | statistics branch for abstraction-enabled `LayerABS(MILP)` |
| `puresart_stats` | `LayerABS_puresart_stats.py` | statistics branch for `PureSART` |
| `standard_milp_stats` | `LayerABS_standard_milp_stats.py` | statistics branch for `Standard MILP` |

### Timelimit Family

| Family | Controller | Paper Meaning |
| --- | --- | --- |
| `abstract_sart_timelimit` | `LayerABS_abstract_sart_timelimit.py` | time-limit branch for abstraction-enabled `LayerABS(SART)` |

## Physical Layout

The LayerABS code path is split into four main layers:

- root `LayerABS_*.py`
  paper-facing family controllers
- `default_profiles/`
  benchmark/network-specific default wrappers such as `LayerABS_abstract_sart_mnist_10x80.py`
- `family_wrappers/<family>/`
  thin wrappers for additional benchmark/network variants
- `layerabs_core/`
  shared implementation code

Configuration lives in `layerabs_variants/`. Small helpers live in `support/`.

## Default Profiles

The default-profile layer currently includes:

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

Use a root controller when you want to choose the variant yourself. Use a default profile when you want the repository's built-in benchmark/network default.

## Variant Naming

Common active variants include:

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

VNN-COMP naming note:

- the paper's ERAN `5x100` and `8x100` correspond to `vnncomp_6x100` and `vnncomp_9x100` in code
- the code counts the output layer in the architecture label

## Shared Core Modules

Key active modules under `layerabs_core/` are:

- `layerabs_abstract_sart_family_propagation.py`
- `layerabs_abstract_milp_family_propagation.py`
- `layerabs_incomplete_family_propagation.py`
- `layerabs_no_abstraction_propagation.py`
- `layerabs_abstract_sart_stats_family_propagation.py`
- `layerabs_abstract_milp_stats_family_propagation.py`
- `layerabs_abstract_sart_timelimit_family_propagation.py`
- `layerabs_verification_runners.py`
- `layerabs_solver_helpers.py`
- `layerabs_runtime_helpers.py`
- `layerabs_stage_helpers.py`
- `layerabs_stats_family_helpers.py`
- `layerabs_shared_helpers.py`
- `layerabs_parallel_task_helpers.py`
- `layerabs_parallel_args_helpers.py`

These modules are implementation code. The paper-facing entrypoints are the root controllers.

## Paper Presets

Paper presets relevant to the in-repo methods are:

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

Coverage labels:

- `supported`
- `partial`
- `unsupported`

Important current notes:

- `table3_vnncomp_hard_cases` is runnable and pins the seven hard-case properties from the paper
- `table6` and `table9` apply the paper's `30s` per-solve Gurobi fairness limit
- `table11` and `table12` remain unsupported because LDSA mode is not exposed in the public controller path
- `table5` stage-level fallback metrics require logs generated with `StageOutcome:` markers

## Maintenance Rules

1. If you are changing algorithm logic, start in `layerabs_core/`.
2. If you are only adding or modifying a benchmark/network variant, start in `layerabs_variants/` and `family_wrappers/`.
3. Keep root controllers thin. They should select variants, expose CLI arguments, and call shared logic.
4. Keep default profiles thin. They should bind one benchmark/network default, not duplicate algorithm code.
5. Use `run_experiment.py --list-layerabs-families` and `--list-paper-presets` to discover the current public experiment surface before adding anything new.
