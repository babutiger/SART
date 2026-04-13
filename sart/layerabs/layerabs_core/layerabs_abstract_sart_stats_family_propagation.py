"""Shared propagation implementation for the `abstract_sart_stats` family."""

import time

import numpy as np

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)

from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    generate_variable_bounds,
    terminate_managed_child_processes,
)
from sart.layerabs.layerabs_core.layerabs_stage_helpers import (
    build_layer_sizes,
)
from sart.layerabs.layerabs_core.layerabs_stats_family_helpers import (
    extend_property_stats_variable_bounds,
    extend_recent_stats_variable_bounds,
    propagate_first_hidden_stats_layer,
)
from sart.layerabs.layerabs_core.layerabs_stage_runner_helpers import (
    run_first_hidden_stage_with_builder,
    run_hidden_stage_with_builder,
    run_output_stage_with_builder,
    run_property_stage_with_builder,
)
from sart.layerabs.layerabs_core.layerabs_parallel_task_helpers import (
    run_parallel_first_hidden_layer_task,
    run_parallel_first_hidden_layer_task_complete,
    run_parallel_hidden_layer_task,
    run_parallel_hidden_layer_task_complete,
    run_parallel_output_layer_task,
    run_parallel_output_layer_task_complete,
    run_parallel_property_layer_task,
    run_parallel_property_layer_task_complete,
)
from sart.layerabs.layerabs_core.layerabs_logging_helpers import (
    redirect_stdout_to_timestamped_log,
)
from sart.layerabs.layerabs_core.layerabs_process_helpers import (
    get_managed_child_process_list,
)
from sart.layerabs.layerabs_core.layerabs_parallel_args_helpers import (
    build_regular_property_layer_task_args,
    build_regular_first_hidden_layer_task_args,
    build_regular_hidden_layer_task_args,
    build_regular_output_layer_task_args,
)
from sart.layerabs.layerabs_core.layerabs_runtime_helpers import (
    append_stage_snapshots,
    finalize_propagation_run,
)
from sart.layerabs.layerabs_core.layerabs_stats_helpers import (
    log_stage_execution_time,
    print_unstable_neuron_summary,
)

__all__ = [
    "run_refinement_propagation",
    "run_complete_propagation",
    "run_refinement_stage_propagation",
    "run_complete_stage_propagation",
    "terminate_family_child_processes",
    "style_time",
    "terminate_all_child_processes",
]

MAX_PARALLEL_WORKERS = 28


def _get_child_process_ids():
    return get_managed_child_process_list(__name__)


def _finalize_abstract_sart_stats_run(
    start_time_all,
    final_output_layer_interval,
    four_dimensional_list,
    save_2mip,
):
    final_output_layer_interval, temp_file_path, save_2mip = finalize_propagation_run(
        start_time_all=start_time_all,
        interval_label="final_output_layer_interval",
        interval_values=final_output_layer_interval,
        layerabs_state=four_dimensional_list,
        snapshot_filename="four_dimensional_list.pkl",
        mip_snapshots=save_2mip,
    )
    print_unstable_neuron_summary(four_dimensional_list)
    return final_output_layer_interval, temp_file_path, save_2mip


style_time = redirect_stdout_to_timestamped_log(__file__)

# Apply the input perturbation and clamp bounds to [0, 1].
# Return one lower/upper bound pair per input dimension.

ABSTRACT_SART_STATS_REFINEMENT_RUNNERS = {
    "output": run_parallel_output_layer_task,
    "property": run_parallel_property_layer_task,
    "first_hidden": run_parallel_first_hidden_layer_task,
    "hidden": run_parallel_hidden_layer_task,
}

ABSTRACT_SART_STATS_COMPLETE_RUNNERS = {
    "output": run_parallel_output_layer_task_complete,
    "property": run_parallel_property_layer_task_complete,
    "first_hidden": run_parallel_first_hidden_layer_task_complete,
    "hidden": run_parallel_hidden_layer_task_complete,
}


def _run_output_layer_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    biases,
    hidden_sizes,
    output_layer,
    four_dimensional_list,
    save_deeppoly,
    layer_sizes,
    variable_bounds_list,
    weights,
    final_weights,
    hidden_layers_activate,
    l_mip_num,
):
    return run_output_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=four_dimensional_list,
        build_extended_variable_bounds=lambda: extend_recent_stats_variable_bounds(
            variable_bounds_list,
            layer_index,
            four_dimensional_list,
            weights,
            final_weights,
        ),
        build_task_args=lambda input_layer_symbols, variable_bounds_list_add: (
            build_regular_output_layer_task_args(
                _get_child_process_ids(),
                layer_index,
                weight_matrix,
                input_layer_symbols,
                biases,
                output_layer,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list_add,
                hidden_layers_activate,
                [],
                l_mip_num,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_property_layer_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    final_weights,
    final_biases,
    output_layer,
    final_output_layer,
    four_dimensional_list,
    save_deeppoly,
    layer_sizes,
    variable_bounds_list,
    weights,
    output_size,
    final_output_layer_interval,
    l_mip_num,
):
    return run_property_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        layerabs_state=four_dimensional_list,
        property_layer_interval=final_output_layer_interval,
        build_extended_variable_bounds=lambda: extend_property_stats_variable_bounds(
            variable_bounds_list,
            layer_index,
            four_dimensional_list,
            weights,
            final_weights,
        ),
        build_task_args=lambda variable_bounds_list_add: (
            build_regular_property_layer_task_args(
                _get_child_process_ids(),
                layer_index,
                weight_matrix,
                final_weights,
                final_biases,
                output_layer,
                final_output_layer,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list_add,
                output_size,
                [],
                l_mip_num,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_first_hidden_parallel_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    biases,
    hidden_sizes,
    hidden_layers,
    four_dimensional_list,
    save_deeppoly,
    layer_sizes,
    variable_bounds_list,
    hidden_layers_activate,
    l_mip_num,
):
    return run_first_hidden_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=four_dimensional_list,
        build_task_args=lambda input_layer_symbols: (
            build_regular_first_hidden_layer_task_args(
                _get_child_process_ids(),
                layer_index,
                weight_matrix,
                input_layer_symbols,
                biases,
                hidden_layers,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list,
                hidden_layers_activate,
                [],
                [],
                l_mip_num,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_hidden_parallel_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    biases,
    hidden_sizes,
    hidden_layers,
    four_dimensional_list,
    save_deeppoly,
    layer_sizes,
    variable_bounds_list,
    weights,
    final_weights,
    hidden_layers_activate,
    l_mip_num,
):
    return run_hidden_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=four_dimensional_list,
        build_extended_variable_bounds=lambda: extend_recent_stats_variable_bounds(
            variable_bounds_list,
            layer_index,
            four_dimensional_list,
            weights,
            final_weights,
        ),
        build_task_args=lambda input_layer_symbols, variable_bounds_list_add: (
            build_regular_hidden_layer_task_args(
                _get_child_process_ids(),
                layer_index,
                weight_matrix,
                input_layer_symbols,
                biases,
                hidden_layers,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list_add,
                hidden_layers_activate,
                [],
                [],
                l_mip_num,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_abstract_sart_stats_propagation(
    weights,
    biases,
    input_size,
    hidden_sizes,
    output_size,
    final_weights,
    final_biases,
    hidden_layers,
    hidden_layers_activate,
    output_layer,
    final_output_layer,
    four_dimensional_list,
    input_layer,
    bound_pair,
    save_deeppoly,
    l_mip_num,
    stage_runners,
):
    final_output_layer_interval = [None] * (output_size - 1)
    save_2mip = []
    layer_sizes = build_layer_sizes(hidden_sizes, output_size)
    variable_bounds_list = generate_variable_bounds(
        input_layer,
        bound_pair,
        hidden_layers,
        hidden_layers_activate,
        output_layer,
        final_output_layer,
    )

    start_time_all = time.time()
    for layer_index in range(len(weights) + 1):
        if layer_index != len(weights):
            weight_matrix = np.array(weights[layer_index])
        else:
            weight_matrix = final_weights

        start_time = time.time()

        if layer_index == 0:
            print(f"Starting first layer, Layer: {layer_index}")
            save_mip_layer_before, save_mip_layer_after = propagate_first_hidden_stats_layer(
                layer_index,
                weight_matrix,
                biases,
                hidden_layers,
                hidden_layers_activate,
                four_dimensional_list,
                save_deeppoly,
                input_size=input_size,
            )
            append_stage_snapshots(
                save_2mip,
                save_mip_layer_before,
                save_mip_layer_after,
            )
            log_stage_execution_time(start_time)
            continue

        if layer_index == len(weights) - 1:
            print(f"Starting output layer, Layer: {layer_index}")
            save_mip_layer_before = _run_output_layer_stage(
                stage_runners["output"],
                layer_index,
                weight_matrix,
                biases,
                hidden_sizes,
                output_layer,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list,
                weights,
                final_weights,
                hidden_layers_activate,
                l_mip_num,
            )
            append_stage_snapshots(save_2mip, save_mip_layer_before)
            log_stage_execution_time(start_time)
            continue

        if layer_index == len(weights):
            print(f"Starting final output layer, Layer: {layer_index}")
            save_mip_layer_before = _run_property_layer_stage(
                stage_runners["property"],
                layer_index,
                weight_matrix,
                final_weights,
                final_biases,
                output_layer,
                final_output_layer,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list,
                weights,
                output_size,
                final_output_layer_interval,
                l_mip_num,
            )
            append_stage_snapshots(save_2mip, save_mip_layer_before)
            log_stage_execution_time(start_time)
            continue

        print(f"Starting hidden layer, Layer: {layer_index}")
        if layer_index == 1:
            save_mip_layer_before, save_mip_layer_after = _run_first_hidden_parallel_stage(
                stage_runners["first_hidden"],
                layer_index,
                weight_matrix,
                biases,
                hidden_sizes,
                hidden_layers,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list,
                hidden_layers_activate,
                l_mip_num,
            )
        else:
            save_mip_layer_before, save_mip_layer_after = _run_hidden_parallel_stage(
                stage_runners["hidden"],
                layer_index,
                weight_matrix,
                biases,
                hidden_sizes,
                hidden_layers,
                four_dimensional_list,
                save_deeppoly,
                layer_sizes,
                variable_bounds_list,
                weights,
                final_weights,
                hidden_layers_activate,
                l_mip_num,
            )
        append_stage_snapshots(
            save_2mip,
            save_mip_layer_before,
            save_mip_layer_after,
        )
        log_stage_execution_time(start_time)

    return _finalize_abstract_sart_stats_run(
        start_time_all,
        final_output_layer_interval,
        four_dimensional_list,
        save_2mip,
    )


# Shared layermip propagation for the refinement-stage abstract_sart_stats pass.
def run_refinement_stage_propagation(
    weights,
    biases,
    input_size,
    hidden_sizes,
    output_size,
    final_weights,
    final_biases,
    hidden_layers,
    hidden_layers_activate,
    output_layer,
    final_output_layer,
    four_dimensional_list,
    input_layer,
    bound_pair,
    save_deeppoly,
    l_mip_num,
):
    return _run_abstract_sart_stats_propagation(
        weights,
        biases,
        input_size,
        hidden_sizes,
        output_size,
        final_weights,
        final_biases,
        hidden_layers,
        hidden_layers_activate,
        output_layer,
        final_output_layer,
        four_dimensional_list,
        input_layer,
        bound_pair,
        save_deeppoly,
        l_mip_num,
        stage_runners=ABSTRACT_SART_STATS_REFINEMENT_RUNNERS,
    )


# Shared layermip propagation for the complete-stage abstract_sart_stats pass.
def run_complete_stage_propagation(
    weights,
    biases,
    input_size,
    hidden_sizes,
    output_size,
    final_weights,
    final_biases,
    hidden_layers,
    hidden_layers_activate,
    output_layer,
    final_output_layer,
    four_dimensional_list,
    input_layer,
    bound_pair,
    save_deeppoly,
    l_mip_num,
):
    return _run_abstract_sart_stats_propagation(
        weights,
        biases,
        input_size,
        hidden_sizes,
        output_size,
        final_weights,
        final_biases,
        hidden_layers,
        hidden_layers_activate,
        output_layer,
        final_output_layer,
        four_dimensional_list,
        input_layer,
        bound_pair,
        save_deeppoly,
        l_mip_num,
        stage_runners=ABSTRACT_SART_STATS_COMPLETE_RUNNERS,
    )

# Child-process cleanup and final result checks.
def terminate_all_child_processes():
    terminate_managed_child_processes(_get_child_process_ids())


run_refinement_propagation = run_refinement_stage_propagation
run_complete_propagation = run_complete_stage_propagation
terminate_family_child_processes = terminate_all_child_processes
COMPLETE_STATS_REFINEMENT_RUNNERS = ABSTRACT_SART_STATS_REFINEMENT_RUNNERS
COMPLETE_STATS_COMPLETE_RUNNERS = ABSTRACT_SART_STATS_COMPLETE_RUNNERS


_LEGACY_ALIASES = {
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip": (
        run_refinement_stage_propagation
    ),
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip_complete": (
        run_complete_stage_propagation
    ),
}


def __getattr__(name):
    """Resolve legacy entrypoint aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
