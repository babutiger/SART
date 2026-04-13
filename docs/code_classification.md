# Code Classification

`sart/layerabs/` is classified by function, not by historical script lineage.

## Directory Layers

- root `LayerABS_*.py`
  paper-facing family controllers
- `default_profiles/`
  built-in benchmark/network defaults for those families
- `family_wrappers/<family>/`
  thin wrappers for additional benchmark/network variants
- `layerabs_core/`
  shared implementation logic
- `layerabs_variants/`
  family-specific variant tables
- `support/`
  small utility modules

Only the active paper method path is kept in-tree.

## Entrypoint Roles

`run_experiment.py --list` and `--list-layerabs` expose the public role labels.

Main roles are:

- `layerabs_family_controller`
- `layerabs_default_profile`
- `thin_wrapper`
- `verify_script`

For LayerABS entrypoints, the launcher also reports:

- `paper_role`
- `content`
- `bucket`
- `role`

## Content Groups

The active LayerABS families are grouped as:

- `main_complete`
  abstraction-enabled complete methods
- `ablation`
  internal method comparisons such as `abstract_milp`, `puresart`, and `standard_milp`
- `measurement`
  statistics-oriented families
- `timelimit`
  time-limit experiments

## Typical Examples

### Family Controllers

- `LayerABS_abstract_sart.py`
- `LayerABS_abstract_milp.py`
- `LayerABS_incomplete_layerabs.py`
- `LayerABS_puresart.py`
- `LayerABS_standard_milp.py`

### Default Profiles

- `default_profiles/LayerABS_abstract_sart_mnist_10x80.py`
- `default_profiles/LayerABS_abstract_milp_mnist_5x50.py`
- `default_profiles/LayerABS_incomplete_layerabs_mnist_10x80.py`
- `default_profiles/LayerABS_puresart_mnist_10x80.py`
- `default_profiles/LayerABS_standard_milp_mnist_10x80.py`

### Wrappers

- `family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_5x50.py`
- `family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_6x100.py`
- `family_wrappers/abstract_milp/LayerABS_abstract_milp_vnncomp_9x100.py`
- `family_wrappers/puresart/LayerABS_puresart_mnist_5x50.py`

### Shared Core

- `layerabs_core/layerabs_abstract_sart_family_propagation.py`
- `layerabs_core/layerabs_abstract_milp_family_propagation.py`
- `layerabs_core/layerabs_incomplete_family_propagation.py`
- `layerabs_core/layerabs_no_abstraction_propagation.py`

## Recommended Editing Rules

- If you are changing shared algorithm behavior, start in `layerabs_core/`.
- If you are only changing benchmark coverage or default parameters, start in `layerabs_variants/`.
- If you only need a new benchmark/network entrypoint, add or edit a thin wrapper under `family_wrappers/<family>/`.
- Do not turn a new experiment into another copied root script if it can be expressed as a controller + variant + wrapper combination.
- Recover deleted historical branches from version history rather than rebuilding ad hoc backup directories in-tree.
