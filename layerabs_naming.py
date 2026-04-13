from __future__ import annotations

from pathlib import Path


KNOWN_LAYERABS_FAMILIES = (
    "complete_timelimit_plus4layer",
    "incomplete_layerabs",
    "final_output_complete",
    "abstract_milp_stats",
    "abstract_sart_timelimit",
    "puresart",
    "standard_milp",
    "puresart_stats",
    "standard_milp_stats",
    "abstract_milp",
    "abstract_sart_stats",
    "complete_copy_abs",
    "complete_copy",
    "abstract_sart",
)


LEGACY_FAMILY_ALIASES = {
    "abstract_sart": "abstract_sart",
    "complete": "abstract_sart",
    "all_complete": "abstract_sart",
    "abstract_milp": "abstract_milp",
    "complete_milp": "abstract_milp",
    "all_complete_milp_code": "abstract_milp",
    "puresart": "puresart",
    "puresart_complete": "puresart",
    "all_complete_ablation_factor_no_abstract_mycode": "puresart",
    "standard_milp": "standard_milp",
    "puremilp_complete": "standard_milp",
    "all_complete_ablation_factor_no_abstract_milp_code": "standard_milp",
    "abstract_sart_stats": "abstract_sart_stats",
    "complete_stats": "abstract_sart_stats",
    "all_complete_num_time_count": "abstract_sart_stats",
    "abstract_milp_stats": "abstract_milp_stats",
    "complete_stats_milp": "abstract_milp_stats",
    "all_complete_milp_code_num_time_count": "abstract_milp_stats",
    "all_complete_ablation_factor_no_abstract_mycode_num_time_count": "puresart_stats",
    "standard_milp_stats": "standard_milp_stats",
    "puremilp_stats": "standard_milp_stats",
    "all_complete_ablation_factor_no_abstract_milp_code_num_time_count": "standard_milp_stats",
    "vnncomp_eran": "abstract_sart",
    "all_complete_eran": "abstract_sart",
    "vnncomp_eran_milp": "abstract_milp",
    "all_complete_eran_milp_code": "abstract_milp",
    "abstract_sart_timelimit": "abstract_sart_timelimit",
    "complete_timelimit": "abstract_sart_timelimit",
    "all_complete_full_complete_excluded_add_timelimit": "abstract_sart_timelimit",
    "all_complete_full_complete_excluded_add_timelimit_plus_4layer": "complete_timelimit_plus4layer",
    "finalout_complete": "final_output_complete",
    "all_complete_copy": "complete_copy",
    "all_complete_copy_abs": "complete_copy_abs",
}


LEGACY_SCRIPT_PATH_ALIASES = (
    (
        "sart/layerabs/LayerABS_mnist_10x80_all_complete.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_sart_mnist_10x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_10x80_all_complete_CIFAR10_5x50.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_cifar10_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_10x80_all_complete_CIFAR10_6x80.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_5x50_all_complete.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_5x80_all_complete.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_5x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_6x100_all_complete.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_9x100_all_complete.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_9x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete/LayerABS_mnist_9x200_all_complete.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_9x200.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_5x50_all_complete_MILP_code.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_milp_mnist_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code/LayerABS_mnist_5x50_all_complete_MILP_code_10x80.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_mnist_10x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code/LayerABS_mnist_5x50_all_complete_MILP_code_5x80.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_mnist_5x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code/LayerABS_mnist_5x50_all_complete_MILP_code_6x100.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code/LayerABS_mnist_5x50_all_complete_MILP_code_9x100.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_mnist_9x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code/LayerABS_mnist_5x50_all_complete_MILP_code_9x200.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_mnist_9x200.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode.py",
        "sart/layerabs/default_profiles/LayerABS_puresart_mnist_10x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_CIFAR10_5x50.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_cifar10_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_CIFAR10_6x80.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_5x50.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_mnist_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_5x80.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_mnist_5x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_6x100.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_9x100.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_mnist_9x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_9x200.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_mnist_9x200.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_VNNCOMP_6x100.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_VNNCOMP_9x100.py",
        "sart/layerabs/family_wrappers/puresart/LayerABS_puresart_vnncomp_9x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code.py",
        "sart/layerabs/default_profiles/LayerABS_standard_milp_mnist_10x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_CIFAR10_5x50.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_cifar10_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_CIFAR10_6x80.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_5x50.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_mnist_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_5x80.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_mnist_5x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_6x100.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_9x100.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_mnist_9x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_9x200.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_mnist_9x200.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_VNNCOMP_6X100.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_VNNCOMP_9X100.py",
        "sart/layerabs/family_wrappers/standard_milp/LayerABS_standard_milp_vnncomp_9x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_6x100_all_complete_num_time_count.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_sart_stats_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_num_time_count/LayerABS_mnist_6x100_all_complete_num_time_count_CIFAR10_6X80.py",
        "sart/layerabs/family_wrappers/abstract_sart_stats/LayerABS_abstract_sart_stats_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_num_time_count/LayerABS_mnist_6x100_all_complete_num_time_count_VNNCOMP_6X100.py",
        "sart/layerabs/family_wrappers/abstract_sart_stats/LayerABS_abstract_sart_stats_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_5x50_all_complete_MILP_code_6x100_num_time_count.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_milp_stats_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code_num_time_count/LayerABS_mnist_5x50_all_complete_MILP_code_6x100_num_time_count_CIFAR10_6X80.py",
        "sart/layerabs/family_wrappers/abstract_milp_stats/LayerABS_abstract_milp_stats_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_milp_code_num_time_count/LayerABS_mnist_5x50_all_complete_MILP_code_6x100_num_time_count_VNNCOMP_6X100.py",
        "sart/layerabs/family_wrappers/abstract_milp_stats/LayerABS_abstract_milp_stats_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_VNNCOMP_6x100_num_time_count.py",
        "sart/layerabs/default_profiles/LayerABS_puresart_stats_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode_num_time_count/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_VNNCOMP_6x100_num_time_count_CIFAR10_6X80.py",
        "sart/layerabs/family_wrappers/puresart_stats/LayerABS_puresart_stats_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_mycode_num_time_count/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_mycode_VNNCOMP_6x100_num_time_count_MNIST6X100.py",
        "sart/layerabs/family_wrappers/puresart_stats/LayerABS_puresart_stats_mnist_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_VNNCOMP_6X100_num_time_count.py",
        "sart/layerabs/default_profiles/LayerABS_standard_milp_stats_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code_num_time_count/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_VNNCOMP_6X100_num_time_count_CIFAR10_6X80.py",
        "sart/layerabs/family_wrappers/standard_milp_stats/LayerABS_standard_milp_stats_cifar10_6x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_ablation_factor_no_abstract_milp_code_num_time_count/LayerABS_mnist_10x80_all_complete_ablation_factor_NO_abstract_MILP_code_VNNCOMP_6X100_num_time_count_MNIST6X100.py",
        "sart/layerabs/family_wrappers/standard_milp_stats/LayerABS_standard_milp_stats_mnist_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_6x100_all_complete_ERAN_6X100_VNNCOMP.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_eran/LayerABS_mnist_6x100_all_complete_ERAN_6X200_VNNCOMP.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_6x200.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_eran/LayerABS_mnist_6x100_all_complete_ERAN_9X100_VNNCOMP.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_9x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_6x100_all_complete_ERAN_MILP_code_6X100_VNNCOMP.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_eran_milp_code/LayerABS_mnist_6x100_all_complete_ERAN_MILP_code_9X100_VNNCOMP.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_vnncomp_9x100.py",
    ),
    (
        "sart/layerabs/LayerABS_mnist_10x80_all_complete_full_complete_excluded_add_timelimit.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_sart_timelimit_mnist_10x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_full_complete_excluded_add_timelimit/LayerABS_mnist_5x50_all_complete_full_complete_excluded_add_timelimit.py",
        "sart/layerabs/family_wrappers/abstract_sart_timelimit/LayerABS_abstract_sart_timelimit_mnist_5x50.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_full_complete_excluded_add_timelimit/LayerABS_mnist_5x80_all_complete_full_complete_excluded_add_timelimit.py",
        "sart/layerabs/family_wrappers/abstract_sart_timelimit/LayerABS_abstract_sart_timelimit_mnist_5x80.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_full_complete_excluded_add_timelimit/LayerABS_mnist_6x100_all_complete_full_complete_excluded_add_timelimit.py",
        "sart/layerabs/family_wrappers/abstract_sart_timelimit/LayerABS_abstract_sart_timelimit_mnist_6x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_full_complete_excluded_add_timelimit/LayerABS_mnist_9x100_all_complete_full_complete_excluded_add_timelimit.py",
        "sart/layerabs/family_wrappers/abstract_sart_timelimit/LayerABS_abstract_sart_timelimit_mnist_9x100.py",
    ),
    (
        "sart/layerabs/family_wrappers/all_complete_full_complete_excluded_add_timelimit/LayerABS_mnist_9x200_all_complete_full_complete_excluded_add_timelimit.py",
        "sart/layerabs/family_wrappers/abstract_sart_timelimit/LayerABS_abstract_sart_timelimit_mnist_9x200.py",
    ),
    (
        "sart/layerabs/specialized/LayerABS_mnist_5x50_all_complete_full_complete_excluded_add_timelimit_+4layer.py",
        "sart/layerabs/specialized/LayerABS_complete_timelimit_plus4layer_mnist_5x50.py",
    ),
    (
        "sart/layerabs/specialized/LayerABS_mnist_10x80_finalout_complete.py",
        "sart/layerabs/specialized/LayerABS_final_output_complete_mnist_10x80.py",
    ),
    (
        "sart/layerabs/legacy/LayerABS_mnist_10x80_all_complete_copy.py",
        "sart/layerabs/legacy/LayerABS_complete_copy_mnist_10x80.py",
    ),
    (
        "sart/layerabs/legacy/LayerABS_mnist_10x80_all_complete_copy_abs.py",
        "sart/layerabs/legacy/LayerABS_complete_copy_abs_mnist_10x80.py",
    ),
)


LEGACY_SCRIPT_ALIASES: dict[str, str] = {}
for old_rel, new_rel in LEGACY_SCRIPT_PATH_ALIASES:
    old_key = old_rel.removesuffix(".py")
    new_key = new_rel.removesuffix(".py")
    LEGACY_SCRIPT_ALIASES[old_key] = new_key
    LEGACY_SCRIPT_ALIASES[Path(old_key).name] = new_key


MOVED_ROOT_DEFAULT_PROFILES = (
    (
        "sart/layerabs/LayerABS_complete_mnist_10x80.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_sart_mnist_10x80.py",
    ),
    (
        "sart/layerabs/LayerABS_incomplete_layerabs_mnist_10x80.py",
        "sart/layerabs/default_profiles/LayerABS_incomplete_layerabs_mnist_10x80.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_milp_mnist_5x50.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_milp_mnist_5x50.py",
    ),
    (
        "sart/layerabs/LayerABS_puresart_complete_mnist_10x80.py",
        "sart/layerabs/default_profiles/LayerABS_puresart_mnist_10x80.py",
    ),
    (
        "sart/layerabs/LayerABS_puremilp_complete_mnist_10x80.py",
        "sart/layerabs/default_profiles/LayerABS_standard_milp_mnist_10x80.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_stats_mnist_6x100.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_sart_stats_mnist_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_stats_milp_mnist_6x100.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_milp_stats_mnist_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_puresart_stats_vnncomp_6x100.py",
        "sart/layerabs/default_profiles/LayerABS_puresart_stats_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_puremilp_stats_vnncomp_6x100.py",
        "sart/layerabs/default_profiles/LayerABS_standard_milp_stats_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_timelimit_mnist_10x80.py",
        "sart/layerabs/default_profiles/LayerABS_abstract_sart_timelimit_mnist_10x80.py",
    ),
)

for old_rel, new_rel in MOVED_ROOT_DEFAULT_PROFILES:
    LEGACY_SCRIPT_ALIASES[old_rel.removesuffix(".py")] = new_rel.removesuffix(".py")


REMOVED_ROOT_VNNCOMP_ENTRYPOINTS = (
    (
        "sart/layerabs/LayerABS_vnncomp_eran.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_vnncomp_eran_6x100.py",
        "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_vnncomp_eran_milp.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_vnncomp_6x100.py",
    ),
    (
        "sart/layerabs/LayerABS_vnncomp_eran_milp_6x100.py",
        "sart/layerabs/family_wrappers/abstract_milp/LayerABS_abstract_milp_vnncomp_6x100.py",
    ),
)

for old_rel, new_rel in REMOVED_ROOT_VNNCOMP_ENTRYPOINTS:
    old_key = old_rel.removesuffix(".py")
    new_key = new_rel.removesuffix(".py")
    LEGACY_SCRIPT_ALIASES[old_key] = new_key
    LEGACY_SCRIPT_ALIASES[Path(old_key).name] = new_key


MOVED_ROOT_CONTROLLERS = (
    (
        "sart/layerabs/LayerABS_complete.py",
        "sart/layerabs/LayerABS_abstract_sart.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_milp.py",
        "sart/layerabs/LayerABS_abstract_milp.py",
    ),
    (
        "sart/layerabs/LayerABS_puresart_complete.py",
        "sart/layerabs/LayerABS_puresart.py",
    ),
    (
        "sart/layerabs/LayerABS_puremilp_complete.py",
        "sart/layerabs/LayerABS_standard_milp.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_stats.py",
        "sart/layerabs/LayerABS_abstract_sart_stats.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_stats_milp.py",
        "sart/layerabs/LayerABS_abstract_milp_stats.py",
    ),
    (
        "sart/layerabs/LayerABS_puremilp_stats.py",
        "sart/layerabs/LayerABS_standard_milp_stats.py",
    ),
    (
        "sart/layerabs/LayerABS_complete_timelimit.py",
        "sart/layerabs/LayerABS_abstract_sart_timelimit.py",
    ),
)

for old_rel, new_rel in MOVED_ROOT_CONTROLLERS:
    old_key = old_rel.removesuffix(".py")
    new_key = new_rel.removesuffix(".py")
    LEGACY_SCRIPT_ALIASES[old_key] = new_key
    LEGACY_SCRIPT_ALIASES[Path(old_key).name] = new_key


_MOVED_FAMILY_WRAPPER_PREFIXES = (
    ("complete", "abstract_sart", "LayerABS_complete_", "LayerABS_abstract_sart_"),
    ("complete_milp", "abstract_milp", "LayerABS_complete_milp_", "LayerABS_abstract_milp_"),
    ("puresart_complete", "puresart", "LayerABS_puresart_complete_", "LayerABS_puresart_"),
    ("puremilp_complete", "standard_milp", "LayerABS_puremilp_complete_", "LayerABS_standard_milp_"),
    ("complete_stats", "abstract_sart_stats", "LayerABS_complete_stats_", "LayerABS_abstract_sart_stats_"),
    ("complete_stats_milp", "abstract_milp_stats", "LayerABS_complete_stats_milp_", "LayerABS_abstract_milp_stats_"),
    ("puremilp_stats", "standard_milp_stats", "LayerABS_puremilp_stats_", "LayerABS_standard_milp_stats_"),
    ("complete_timelimit", "abstract_sart_timelimit", "LayerABS_complete_timelimit_", "LayerABS_abstract_sart_timelimit_"),
)


def _build_moved_family_wrapper_aliases() -> tuple[tuple[str, str], ...]:
    repo_root = Path(__file__).resolve().parent
    wrappers_root = (
        repo_root
        / "sart"
        / "layerabs"
        / "family_wrappers"
    )
    aliases: list[tuple[str, str]] = []
    for old_dir, new_dir, old_prefix, new_prefix in _MOVED_FAMILY_WRAPPER_PREFIXES:
        new_dir_path = wrappers_root / new_dir
        if not new_dir_path.exists():
            continue
        for new_path in sorted(new_dir_path.glob("LayerABS*.py")):
            new_rel = new_path.relative_to(repo_root).as_posix()
            old_rel = new_rel.replace(
                f"family_wrappers/{new_dir}/",
                f"family_wrappers/{old_dir}/",
                1,
            ).replace(new_prefix, old_prefix, 1)
            aliases.append((old_rel, new_rel))
    return tuple(aliases)


for old_rel, new_rel in _build_moved_family_wrapper_aliases():
    old_key = old_rel.removesuffix(".py")
    new_key = new_rel.removesuffix(".py")
    LEGACY_SCRIPT_ALIASES[old_key] = new_key
    LEGACY_SCRIPT_ALIASES[Path(old_key).name] = new_key
