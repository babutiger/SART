# Conversion Tools

This directory contains data-preparation and format-conversion utilities that are
not part of the main verification runtime.

Typical uses:

- convert repository property files into ERAN-style CSV inputs
- convert repository property files into VNNLIB files
- generate Marabou-style text constraints
- inspect `.pth` checkpoints and generate property files or ground-truth labels

Current scripts:

- `evaluate_checkpoint_ground_truth.py`
- `generate_properties_from_checkpoint.py`
- `convert_property_to_eran_csv.py`
- `convert_property_folder_to_eran_csv.py`
- `convert_property_to_vnnlib.py`
- `convert_property_folder_to_vnnlib.py`
- `convert_property_to_marabou_txt.py`
- `convert_property_folder_to_marabou_txt.py`
- `export_vnnlib_paths_to_csv_relative.py`
- `export_vnnlib_paths_to_csv_absolute.py`

These scripts are maintenance utilities. They are not imported by the active
LayerABS controller path.

Every conversion script now exposes a small command-line interface. Run
`python <script> --help` for usage details instead of editing hard-coded example
paths inside the file.
