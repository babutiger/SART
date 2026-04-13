"""Shared propagation implementation for the `abstract_milp_stats` family."""

import time

import numpy as np

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    generate_variable_bounds,
    terminate_managed_child_processes,
)
from sart.layerabs.layerabs_core.layerabs_abstract_milp_stats_task_helpers import (
    ABSTRACT_MILP_STATS_COMPLETE_RUNNERS,
    ABSTRACT_MILP_STATS_REFINEMENT_RUNNERS,
    get_child_process_ids,
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
from sart.layerabs.layerabs_core.layerabs_parallel_args_helpers import (
    build_stats_milp_first_hidden_layer_task_args,
    build_stats_milp_hidden_layer_task_args,
    build_stats_milp_output_layer_task_args,
    build_stats_milp_property_layer_task_args,
)
from sart.layerabs.layerabs_core.layerabs_logging_helpers import (
    redirect_stdout_to_timestamped_log,
)
from sart.layerabs.layerabs_core.layerabs_runtime_helpers import (
    append_stage_snapshots,
    finalize_propagation_run,
)
from sart.layerabs.layerabs_core.layerabs_stats_helpers import (
    log_stage_execution_time,
    print_unstable_neuron_summary,
)

MAX_PARALLEL_WORKERS = 28

__all__ = [
    "run_refinement_stage_propagation",
    "run_complete_stage_propagation",
    "run_refinement_propagation",
    "run_complete_propagation",
    "terminate_family_child_processes",
    "style_time",
    "terminate_all_child_processes",
]

style_time = redirect_stdout_to_timestamped_log(__file__)


def _finalize_stats_milp_run(
    start_time_all,
    property_layer_interval,
    layerabs_state,
    mip_snapshots,
    include_unstable_summary=False,
):
    property_layer_interval, temp_file_path, mip_snapshots = finalize_propagation_run(
        start_time_all=start_time_all,
        interval_label="property_layer_interval",
        interval_values=property_layer_interval,
        layerabs_state=layerabs_state,
        snapshot_filename="layerabs_state.pkl",
        mip_snapshots=mip_snapshots,
    )

    if include_unstable_summary:
        print_unstable_neuron_summary(layerabs_state)

    terminate_all_child_processes()
    return property_layer_interval, temp_file_path, mip_snapshots


def _run_stats_milp_propagation(
    network_weights,
    network_biases,
    input_size,
    hidden_sizes,
    output_size,
    property_layer_weights,
    property_layer_biases,
    hidden_layer_symbols,
    activated_hidden_layer_symbols,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    input_layer_symbols,
    bound_pair,
    deeppoly_bounds,
    mip_backtrack_depth,
    stage_runners,
    include_unstable_summary=False,
    log_non_hidden_stage_times=False,
):
    property_layer_interval = [None] * (output_size - 1)
    mip_snapshots = []
    layer_sizes = build_layer_sizes(hidden_sizes, output_size)
    variable_bounds = generate_variable_bounds(
        input_layer_symbols,
        bound_pair,
        hidden_layer_symbols,
        activated_hidden_layer_symbols,
        output_layer_symbols,
        property_layer_symbols,
    )

    start_time_all = time.time()
    for layer_index in range(len(network_weights) + 1):
        if layer_index != len(network_weights):
            weight_matrix = np.array(network_weights[layer_index])
        else:
            weight_matrix = property_layer_weights

        start_time = time.time()

        if layer_index == 0:
            print(f"Starting first layer, Layer: {layer_index}")
            previous_layer_symbols = collect_state_slot_symbols(
                layerabs_state,
                layer_index,
                input_size,
                0,
            )
            (
                save_mip_layer_before,
                save_mip_layer_after,
            ) = propagate_first_hidden_stats_layer(
                layer_index,
                weight_matrix,
                network_biases,
                hidden_layer_symbols,
                activated_hidden_layer_symbols,
                layerabs_state,
                deeppoly_bounds,
                previous_layer_symbols=previous_layer_symbols,
            )
            append_stage_snapshots(
                mip_snapshots,
                save_mip_layer_before,
                save_mip_layer_after,
            )

            if log_non_hidden_stage_times:
                log_stage_execution_time(start_time)

        elif layer_index == len(network_weights) - 1:
            print(f"Starting output layer, Layer: {layer_index}")
            save_mip_layer_before = _run_output_layer_stage(
                stage_runners["output"],
                layer_index,
                weight_matrix,
                network_biases,
                hidden_sizes,
                variable_bounds,
                network_weights,
                property_layer_weights,
                layerabs_state,
                output_layer_symbols,
                deeppoly_bounds,
                layer_sizes,
                activated_hidden_layer_symbols,
                mip_backtrack_depth,
            )
            append_stage_snapshots(mip_snapshots, save_mip_layer_before)

            if log_non_hidden_stage_times:
                log_stage_execution_time(start_time)

        elif layer_index == len(network_weights):
            print(f"Starting final output layer, Layer: {layer_index}")
            save_mip_layer_before = _run_property_layer_stage(
                stage_runners["property"],
                layer_index,
                weight_matrix,
                variable_bounds,
                network_weights,
                property_layer_weights,
                property_layer_biases,
                layerabs_state,
                output_layer_symbols,
                property_layer_symbols,
                deeppoly_bounds,
                layer_sizes,
                output_size,
                mip_backtrack_depth,
                property_layer_interval,
            )
            append_stage_snapshots(mip_snapshots, save_mip_layer_before)

            if log_non_hidden_stage_times:
                log_stage_execution_time(start_time)

        else:
            print(f"Starting hidden layer, Layer: {layer_index}")

            if layer_index == 1:
                (
                    save_mip_layer_before,
                    save_mip_layer_after,
                ) = _run_first_hidden_parallel_stage(
                    stage_runners["first_hidden"],
                    layer_index,
                    weight_matrix,
                    network_biases,
                    hidden_sizes,
                    layerabs_state,
                    hidden_layer_symbols,
                    deeppoly_bounds,
                    layer_sizes,
                    variable_bounds,
                    activated_hidden_layer_symbols,
                    mip_backtrack_depth,
                )
            else:
                (
                    save_mip_layer_before,
                    save_mip_layer_after,
                ) = _run_hidden_parallel_stage(
                    stage_runners["hidden"],
                    layer_index,
                    weight_matrix,
                    network_biases,
                    hidden_sizes,
                    variable_bounds,
                    network_weights,
                    property_layer_weights,
                    layerabs_state,
                    hidden_layer_symbols,
                    deeppoly_bounds,
                    layer_sizes,
                    activated_hidden_layer_symbols,
                    mip_backtrack_depth,
                )

            append_stage_snapshots(
                mip_snapshots,
                save_mip_layer_before,
                save_mip_layer_after,
            )
            log_stage_execution_time(start_time)

    return _finalize_stats_milp_run(
        start_time_all,
        property_layer_interval,
        layerabs_state,
        mip_snapshots,
        include_unstable_summary=include_unstable_summary,
    )
def _run_output_layer_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    network_biases,
    hidden_sizes,
    variable_bounds,
    network_weights,
    property_layer_weights,
    layerabs_state,
    output_layer_symbols,
    deeppoly_bounds,
    layer_sizes,
    activated_hidden_layer_symbols,
    mip_backtrack_depth,
):
    return run_output_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=layerabs_state,
        build_extended_variable_bounds=lambda: extend_recent_stats_variable_bounds(
            variable_bounds,
            layer_index,
            layerabs_state,
            network_weights,
            property_layer_weights,
        ),
        build_task_args=lambda previous_layer_symbols, extended_variable_bounds: (
            build_stats_milp_output_layer_task_args(
                layer_index,
                weight_matrix,
                previous_layer_symbols,
                network_biases,
                output_layer_symbols,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                extended_variable_bounds,
                activated_hidden_layer_symbols,
                [],
                mip_backtrack_depth,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_property_layer_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    variable_bounds,
    network_weights,
    property_layer_weights,
    property_layer_biases,
    layerabs_state,
    output_layer_symbols,
    property_layer_symbols,
    deeppoly_bounds,
    layer_sizes,
    output_size,
    mip_backtrack_depth,
    property_layer_interval,
):
    return run_property_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        layerabs_state=layerabs_state,
        property_layer_interval=property_layer_interval,
        build_extended_variable_bounds=lambda: extend_property_stats_variable_bounds(
            variable_bounds,
            layer_index,
            layerabs_state,
            network_weights,
            property_layer_weights,
        ),
        build_task_args=lambda extended_variable_bounds: (
            build_stats_milp_property_layer_task_args(
                layer_index,
                weight_matrix,
                property_layer_weights,
                property_layer_biases,
                output_layer_symbols,
                property_layer_symbols,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                extended_variable_bounds,
                output_size,
                [],
                mip_backtrack_depth,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_first_hidden_parallel_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    network_biases,
    hidden_sizes,
    layerabs_state,
    hidden_layer_symbols,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds,
    activated_hidden_layer_symbols,
    mip_backtrack_depth,
):
    return run_first_hidden_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=layerabs_state,
        build_task_args=lambda previous_layer_symbols: (
            build_stats_milp_first_hidden_layer_task_args(
                layer_index,
                weight_matrix,
                previous_layer_symbols,
                network_biases,
                hidden_layer_symbols,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                variable_bounds,
                activated_hidden_layer_symbols,
                [],
                [],
                mip_backtrack_depth,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def _run_hidden_parallel_stage(
    stage_runner,
    layer_index,
    weight_matrix,
    network_biases,
    hidden_sizes,
    variable_bounds,
    network_weights,
    property_layer_weights,
    layerabs_state,
    hidden_layer_symbols,
    deeppoly_bounds,
    layer_sizes,
    activated_hidden_layer_symbols,
    mip_backtrack_depth,
):
    return run_hidden_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=layerabs_state,
        build_extended_variable_bounds=lambda: extend_recent_stats_variable_bounds(
            variable_bounds,
            layer_index,
            layerabs_state,
            network_weights,
            property_layer_weights,
        ),
        build_task_args=lambda previous_layer_symbols, extended_variable_bounds: (
            build_stats_milp_hidden_layer_task_args(
                layer_index,
                weight_matrix,
                previous_layer_symbols,
                network_biases,
                hidden_layer_symbols,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                extended_variable_bounds,
                activated_hidden_layer_symbols,
                [],
                [],
                mip_backtrack_depth,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def run_refinement_stage_propagation(
    network_weights,
    network_biases,
    input_size,
    hidden_sizes,
    output_size,
    property_layer_weights,
    property_layer_biases,
    hidden_layer_symbols,
    activated_hidden_layer_symbols,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    input_layer_symbols,
    bound_pair,
    deeppoly_bounds,
    mip_backtrack_depth,
):
    """Run the refinement-stage `abstract_milp_stats` propagation pass."""
    return _run_stats_milp_propagation(
        network_weights,
        network_biases,
        input_size,
        hidden_sizes,
        output_size,
        property_layer_weights,
        property_layer_biases,
        hidden_layer_symbols,
        activated_hidden_layer_symbols,
        output_layer_symbols,
        property_layer_symbols,
        layerabs_state,
        input_layer_symbols,
        bound_pair,
        deeppoly_bounds,
        mip_backtrack_depth,
        ABSTRACT_MILP_STATS_REFINEMENT_RUNNERS,
        include_unstable_summary=True,
        log_non_hidden_stage_times=True,
    )


def run_complete_stage_propagation(
    network_weights,
    network_biases,
    input_size,
    hidden_sizes,
    output_size,
    property_layer_weights,
    property_layer_biases,
    hidden_layer_symbols,
    activated_hidden_layer_symbols,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    input_layer_symbols,
    bound_pair,
    deeppoly_bounds,
    mip_backtrack_depth,
):
    """Run the complete-stage `abstract_milp_stats` propagation pass."""
    return _run_stats_milp_propagation(
        network_weights,
        network_biases,
        input_size,
        hidden_sizes,
        output_size,
        property_layer_weights,
        property_layer_biases,
        hidden_layer_symbols,
        activated_hidden_layer_symbols,
        output_layer_symbols,
        property_layer_symbols,
        layerabs_state,
        input_layer_symbols,
        bound_pair,
        deeppoly_bounds,
        mip_backtrack_depth,
        ABSTRACT_MILP_STATS_COMPLETE_RUNNERS,
    )


def terminate_all_child_processes():
    """Terminate worker processes tracked by this module."""
    terminate_managed_child_processes(get_child_process_ids())


run_refinement_propagation = run_refinement_stage_propagation
run_complete_propagation = run_complete_stage_propagation
terminate_family_child_processes = terminate_all_child_processes


_LEGACY_TASK_HELPER_NAMES = {
    "jisuan_function1_lp_sym_lmipnum1": "compute_first_hidden_layer_neuron",
    "jisuan_function2_lp_sym_lmipnum1": "compute_hidden_layer_neuron",
    "jisuan_function_output_lp_sym_lmipnum1": "compute_output_layer_neuron",
    "jisuan_function_finaloutput_lp_sym_lmipnum1": (
        "compute_property_layer_neuron"
    ),
    "jisuan_function1_lp_sym_lmipnum1_layermip_complete": (
        "compute_first_hidden_layer_neuron_complete"
    ),
    "jisuan_function2_lp_sym_lmipnum1_layermip_complete": (
        "compute_hidden_layer_neuron_complete"
    ),
    "jisuan_function_output_lp_sym_lmipnum1_layermip_complete": (
        "compute_output_layer_neuron_complete"
    ),
    "jisuan_function_finaloutput_lp_sym_lmipnum1_layermip_complete": (
        "compute_property_layer_neuron_complete"
    ),
    "parallel_task1_lp_sym_lmipnum1": (
        "run_parallel_first_hidden_layer_task_stats_milp"
    ),
    "parallel_task2_lp_sym_lmipnum1": (
        "run_parallel_hidden_layer_task_stats_milp"
    ),
    "parallel_task_output_lp_sym_lmipnum1": (
        "run_parallel_output_layer_task_stats_milp"
    ),
    "parallel_task_finaloutput_lp_sym_lmipnum1": (
        "run_parallel_property_layer_task_stats_milp"
    ),
    "parallel_task1_lp_sym_lmipnum1_layermip_complete": (
        "run_parallel_first_hidden_layer_task_complete_stats_milp"
    ),
    "parallel_task2_lp_sym_lmipnum1_layermip_complete": (
        "run_parallel_hidden_layer_task_complete_stats_milp"
    ),
    "parallel_task_output_lp_sym_lmipnum1_layermip_complete": (
        "run_parallel_output_layer_task_complete_stats_milp"
    ),
    "parallel_task_finaloutput_lp_sym_lmipnum1_layermip_complete": (
        "run_parallel_property_layer_task_complete_stats_milp"
    ),
}

_LEGACY_LOCAL_ALIASES = {
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip": (
        run_refinement_stage_propagation
    ),
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip_complete": (
        run_complete_stage_propagation
    ),
}


def __getattr__(name):
    """Resolve legacy compatibility aliases for archived scripts."""
    if name in _LEGACY_LOCAL_ALIASES:
        return resolve_legacy_alias(__name__, name, _LEGACY_LOCAL_ALIASES)

    helper_name = _LEGACY_TASK_HELPER_NAMES.get(name)
    if helper_name is not None:
        from sart.layerabs.layerabs_core import (
            layerabs_abstract_milp_stats_task_helpers as stats_milp_task_helpers,
        )

        return getattr(stats_milp_task_helpers, helper_name)

    return resolve_legacy_alias(__name__, name, {})
