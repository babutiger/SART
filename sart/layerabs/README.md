# Symbolic Compute Code Layout

`layerabs/` is the active LayerABS code tree.

## Physical Layout

- root `LayerABS_*.py`
  model-neutral family controllers
- `default_profiles/`
  paper-default benchmark/network wrappers
- `family_wrappers/<family>/`
  thin wrappers for additional benchmark/network variants
- `layerabs_core/`
  shared implementation modules
- `layerabs_variants/`
  family-specific variant tables
- `support/`
  small utility modules

Historical branches, one-off copies, and archived runtime assets were removed from the active tree.

## Where To Start

- use the root `LayerABS_*.py` files when you want the paper-facing controller
- use `default_profiles/` when you want the built-in benchmark/network default
- use `family_wrappers/` when you want another benchmark/network variant
- use `layerabs_core/` when you need shared algorithm logic
- use `layerabs_variants/` when you need to change variant coverage or defaults

## Root Controllers

The current root controllers are:

- `LayerABS_abstract_sart.py`
- `LayerABS_abstract_milp.py`
- `LayerABS_incomplete_layerabs.py`
- `LayerABS_puresart.py`
- `LayerABS_standard_milp.py`
- `LayerABS_abstract_sart_stats.py`
- `LayerABS_abstract_milp_stats.py`
- `LayerABS_puresart_stats.py`
- `LayerABS_standard_milp_stats.py`
- `LayerABS_abstract_sart_timelimit.py`

These are the preferred public entrypoints.

## Default Profiles

The current default-profile wrappers are:

- `default_profiles/LayerABS_abstract_sart_mnist_10x80.py`
- `default_profiles/LayerABS_abstract_milp_mnist_5x50.py`
- `default_profiles/LayerABS_incomplete_layerabs_mnist_10x80.py`
- `default_profiles/LayerABS_puresart_mnist_10x80.py`
- `default_profiles/LayerABS_standard_milp_mnist_10x80.py`
- `default_profiles/LayerABS_abstract_sart_stats_mnist_6x100.py`
- `default_profiles/LayerABS_abstract_milp_stats_mnist_6x100.py`
- `default_profiles/LayerABS_puresart_stats_vnncomp_6x100.py`
- `default_profiles/LayerABS_standard_milp_stats_vnncomp_6x100.py`
- `default_profiles/LayerABS_abstract_sart_timelimit_mnist_10x80.py`

## Important Notes

- `Incomplete-LayerABS` is exposed explicitly through `LayerABS_incomplete_layerabs.py`
- the default `k` for `Incomplete-LayerABS` is `2`
- VNN-COMP variants use the code labels `vnncomp_6x100` and `vnncomp_9x100`
- the root launcher understands both public controller names and normalized experiment ids

## Useful Commands

```bash
python run_experiment.py --list
python run_experiment.py --list-layerabs
python run_experiment.py --list-layerabs-families
python run_experiment.py --list-paper-presets
python run_experiment.py --script LayerABS_abstract_sart --script-arg=--variant --script-arg=mnist_5x50 --dry-run
python run_experiment.py --script LayerABS_incomplete_layerabs --script-arg=--variant --script-arg=mnist_5x50 --script-arg=--k-layers --script-arg=3 --dry-run
python run_experiment.py --paper-preset table8_vnncomp_complete --dry-run
python run_experiment.py --summarize-paper-preset table4_layerabs_complete
python run_experiment.py --export-paper-tables
```

The launcher output distinguishes:

- `layerabs_family_controller`
- `layerabs_default_profile`
- `thin_wrapper`
- `verify_script`

The family listing also shows:

- `path=` for the canonical default profile
- `controller=` for the recommended root controller

See the repository root [README.md](../../README.md) for the full project overview.
