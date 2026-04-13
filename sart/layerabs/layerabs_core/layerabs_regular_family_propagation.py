"""Shared propagation helpers for regular LayerABS family runners."""

from __future__ import annotations

import copy
import time

import numpy as np

from sart.layerabs.support.longge3 import (
    relu3,
    relu3_deeppoly_low,
    relu3_deeppoly_up,
)
from sart.layerabs.layerabs_core.layerabs_parallel_args_helpers import (
    build_regular_first_hidden_layer_task_args,
    build_regular_hidden_layer_task_args,
    build_regular_output_layer_task_args,
    build_regular_property_layer_task_args,
)
from sart.layerabs.layerabs_core.layerabs_runtime_helpers import (
    append_stage_snapshots,
    finalize_propagation_run,
    log_stage_execution_time,
    run_parallel_tasks,
)
from sart.layerabs.layerabs_core.layerabs_stage_helpers import (
    build_layer_sizes,
    collect_state_slot_symbols,
    merge_hidden_layer_results,
    merge_output_layer_results,
    merge_property_layer_results,
)
from sart.layerabs.layerabs_core.layerabs_stage_runner_helpers import (
    run_first_hidden_stage_with_builder,
    run_hidden_stage_with_builder,
    run_output_stage_with_builder,
    run_property_stage_with_builder,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    add_or_update_inner_list,
    copy_array_values,
    generate_variable_bounds,
    longe_np,
)


MAX_PARALLEL_WORKERS = 28


def finalize_regular_run(
    start_time_all,
    final_output_layer_interval,
    layerabs_state,
    mip_snapshots,
    cleanup_callback=None,
):
    final_output_layer_interval, temp_file_path, mip_snapshots = finalize_propagation_run(
        start_time_all=start_time_all,
        interval_label="final_output_layer_interval",
        interval_values=final_output_layer_interval,
        layerabs_state=layerabs_state,
        snapshot_filename="four_dimensional_list.pkl",
        mip_snapshots=mip_snapshots,
    )
    if cleanup_callback is not None:
        cleanup_callback()
    return final_output_layer_interval, temp_file_path, mip_snapshots


def propagate_first_hidden_layer_from_inputs(
    layer_index,
    layer_weights,
    network_biases,
    input_size,
    hidden_layer_symbols,
    activated_hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
):
    input_layer_symbols = [
        layerabs_state[layer_index][input_index][0][0]
        for input_index in range(input_size)
    ]
    pre_activation_mip_bounds = []
    post_activation_mip_bounds = []

    for neuron_index in range(len(layer_weights)):
        pre_activation_bounds = []
        post_activation_bounds = []
        neuron_output = longe_np(
            layer_weights[neuron_index],
            input_layer_symbols,
            network_biases[layer_index][neuron_index][0],
        )

        pre_activation_lower_bound = 0
        pre_activation_upper_bound = 0
        post_activation_lower_bound = 0
        post_activation_upper_bound = 0

        for input_index in range(len(layer_weights[neuron_index])):
            if float(layer_weights[neuron_index][input_index]) <= 0:
                pre_activation_lower_bound += (
                    float(layer_weights[neuron_index][input_index])
                    * layerabs_state[layer_index][input_index][0][4]
                )
            else:
                pre_activation_lower_bound += (
                    float(layer_weights[neuron_index][input_index])
                    * layerabs_state[layer_index][input_index][0][3]
                )
        pre_activation_lower_bound += float(np.array(network_biases[layer_index][neuron_index]))

        for input_index in range(len(layer_weights[neuron_index])):
            if float(layer_weights[neuron_index][input_index]) <= 0:
                pre_activation_upper_bound += (
                    float(layer_weights[neuron_index][input_index])
                    * layerabs_state[layer_index][input_index][0][3]
                )
            else:
                pre_activation_upper_bound += (
                    float(layer_weights[neuron_index][input_index])
                    * layerabs_state[layer_index][input_index][0][4]
                )
        pre_activation_upper_bound += float(np.array(network_biases[layer_index][neuron_index]))

        pre_activation_symbol = str(hidden_layer_symbols[layer_index][neuron_index])
        pre_activation_equation = pre_activation_symbol + "==" + neuron_output
        layerabs_state[layer_index + 1][neuron_index][0][0] = pre_activation_symbol
        layerabs_state[layer_index + 1][neuron_index][0][1] = neuron_output
        layerabs_state[layer_index + 1][neuron_index][0][2] = pre_activation_equation
        layerabs_state[layer_index + 1][neuron_index][0][3] = pre_activation_lower_bound
        layerabs_state[layer_index + 1][neuron_index][0][4] = pre_activation_upper_bound
        layerabs_state[layer_index + 1][neuron_index][0][5] = pre_activation_symbol + ">=" + neuron_output
        layerabs_state[layer_index + 1][neuron_index][0][6] = pre_activation_symbol + "<=" + neuron_output
        layerabs_state[layer_index + 1][neuron_index][0][7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
        layerabs_state[layer_index + 1][neuron_index][0][8] = deeppoly_bounds[layer_index * 2][neuron_index][1]

        pre_activation_bounds.append(pre_activation_lower_bound)
        pre_activation_bounds.append(pre_activation_upper_bound)
        pre_activation_mip_bounds.append(pre_activation_bounds)

        relu_output = relu3(
            pre_activation_symbol,
            pre_activation_lower_bound,
            pre_activation_upper_bound,
        )
        relu_lower_form = relu3_deeppoly_low(
            pre_activation_symbol,
            pre_activation_lower_bound,
            pre_activation_upper_bound,
        )
        relu_upper_form = relu3_deeppoly_up(
            pre_activation_symbol,
            pre_activation_lower_bound,
            pre_activation_upper_bound,
        )

        if pre_activation_lower_bound > 0:
            post_activation_lower_bound = pre_activation_lower_bound
            post_activation_upper_bound = pre_activation_upper_bound
        if pre_activation_upper_bound < 0:
            post_activation_lower_bound = 0
            post_activation_upper_bound = 0
        if pre_activation_lower_bound <= 0 and pre_activation_upper_bound >= 0:
            post_activation_lower_bound = 0
            post_activation_upper_bound = pre_activation_upper_bound

        post_activation_symbol = str(activated_hidden_layer_symbols[layer_index][neuron_index])
        post_activation_equation = post_activation_symbol + "==" + relu_output
        layerabs_state[layer_index + 1][neuron_index][1][0] = post_activation_symbol
        layerabs_state[layer_index + 1][neuron_index][1][1] = relu_output
        layerabs_state[layer_index + 1][neuron_index][1][2] = post_activation_equation
        layerabs_state[layer_index + 1][neuron_index][1][3] = post_activation_lower_bound
        layerabs_state[layer_index + 1][neuron_index][1][4] = post_activation_upper_bound
        layerabs_state[layer_index + 1][neuron_index][1][5] = post_activation_symbol + ">=" + relu_lower_form
        layerabs_state[layer_index + 1][neuron_index][1][6] = post_activation_symbol + "<=" + relu_upper_form
        layerabs_state[layer_index + 1][neuron_index][1][7] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][0]
        layerabs_state[layer_index + 1][neuron_index][1][8] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][1]

        post_activation_bounds.append(post_activation_lower_bound)
        post_activation_bounds.append(post_activation_upper_bound)
        post_activation_mip_bounds.append(post_activation_bounds)

    return pre_activation_mip_bounds, post_activation_mip_bounds

def append_layer_slot_bounds(variable_bounds_list_add, layerabs_state, layer_index, neuron_index, slot_index):
    layer_slot = layerabs_state[layer_index + 1][neuron_index][slot_index]
    variable_bounds_entry = [layer_slot[0], layer_slot[3], layer_slot[4]]
    return add_or_update_inner_list(variable_bounds_list_add, variable_bounds_entry)


def extend_recent_variable_bounds(
    variable_bounds_list,
    layerabs_state,
    network_weights,
    property_layer_weights,
    layer_index,
):
    variable_bounds_list_add = copy.deepcopy(variable_bounds_list)
    if layer_index < 3:
        return variable_bounds_list_add

    total_weight_layers = len(network_weights)
    for history_layer_index in range(layer_index - 3, layer_index):
        if history_layer_index != total_weight_layers:
            history_layer_weights = np.array(network_weights[history_layer_index])
        else:
            history_layer_weights = property_layer_weights

        slot_index = 0 if history_layer_index == layer_index - 1 else 1
        for neuron_index in range(len(history_layer_weights)):
            variable_bounds_list_add = append_layer_slot_bounds(
                variable_bounds_list_add,
                layerabs_state,
                history_layer_index,
                neuron_index,
                slot_index,
            )

    return variable_bounds_list_add


def extend_property_layer_variable_bounds(
    variable_bounds_list,
    layerabs_state,
    network_weights,
    property_layer_weights,
    layer_index,
):
    variable_bounds_list_add = copy.deepcopy(variable_bounds_list)
    if layer_index < 3:
        return variable_bounds_list_add

    total_weight_layers = len(network_weights)
    for history_layer_index in range(layer_index - 3, layer_index):
        if (
            history_layer_index != total_weight_layers
            and history_layer_index != total_weight_layers - 1
        ):
            history_layer_weights = np.array(network_weights[history_layer_index])
            slot_index = 1
            history_layer_width = len(history_layer_weights)
        else:
            history_layer_weights = property_layer_weights
            slot_index = 0
            history_layer_width = len(history_layer_weights) + 1

        for neuron_index in range(history_layer_width):
            variable_bounds_list_add = append_layer_slot_bounds(
                variable_bounds_list_add,
                layerabs_state,
                history_layer_index,
                neuron_index,
                slot_index,
            )

    return variable_bounds_list_add


def run_output_layer_stage(
    child_process_ids,
    stage_runner,
    layer_index,
    layer_weights,
    network_biases,
    hidden_sizes,
    output_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds_list,
    network_weights,
    property_layer_weights,
    activated_hidden_layer_symbols,
    mip_backtrack_depth,
):
    return run_output_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=layerabs_state,
        build_extended_variable_bounds=lambda: extend_recent_variable_bounds(
            variable_bounds_list,
            layerabs_state,
            network_weights,
            property_layer_weights,
            layer_index,
        ),
        build_task_args=lambda previous_layer_symbols, extended_variable_bounds: (
            build_regular_output_layer_task_args(
                child_process_ids,
                layer_index,
                layer_weights,
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


def run_property_layer_stage(
    child_process_ids,
    stage_runner,
    layer_index,
    layer_weights,
    property_layer_weights,
    property_layer_biases,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds_list,
    network_weights,
    output_size,
    final_output_layer_interval,
    mip_backtrack_depth,
):
    return run_property_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        layerabs_state=layerabs_state,
        property_layer_interval=final_output_layer_interval,
        build_extended_variable_bounds=lambda: extend_property_layer_variable_bounds(
            variable_bounds_list,
            layerabs_state,
            network_weights,
            property_layer_weights,
            layer_index,
        ),
        build_task_args=lambda extended_variable_bounds: (
            build_regular_property_layer_task_args(
                child_process_ids,
                layer_index,
                layer_weights,
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


def run_first_hidden_parallel_stage(
    child_process_ids,
    stage_runner,
    layer_index,
    layer_weights,
    network_biases,
    hidden_sizes,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds_list,
    activated_hidden_layer_symbols,
    mip_backtrack_depth,
):
    return run_first_hidden_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=layerabs_state,
        build_task_args=lambda previous_layer_symbols: (
            build_regular_first_hidden_layer_task_args(
                child_process_ids,
                layer_index,
                layer_weights,
                previous_layer_symbols,
                network_biases,
                hidden_layer_symbols,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                variable_bounds_list,
                activated_hidden_layer_symbols,
                [],
                [],
                mip_backtrack_depth,
            )
        ),
        max_parallel_workers=MAX_PARALLEL_WORKERS,
    )


def run_hidden_parallel_stage(
    child_process_ids,
    stage_runner,
    layer_index,
    layer_weights,
    network_biases,
    hidden_sizes,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds_list,
    network_weights,
    property_layer_weights,
    activated_hidden_layer_symbols,
    mip_backtrack_depth,
):
    return run_hidden_stage_with_builder(
        stage_runner,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=layerabs_state,
        build_extended_variable_bounds=lambda: extend_recent_variable_bounds(
            variable_bounds_list,
            layerabs_state,
            network_weights,
            property_layer_weights,
            layer_index,
        ),
        build_task_args=lambda previous_layer_symbols, extended_variable_bounds: (
            build_regular_hidden_layer_task_args(
                child_process_ids,
                layer_index,
                layer_weights,
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


def run_regular_propagation(
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
    layerabs_state,
    input_layer,
    bound_pair,
    deeppoly_bounds,
    mip_backtrack_depth,
    stage_runners,
    child_process_ids,
    cleanup_callback=None,
):
    final_output_layer_interval = [None] * (output_size - 1)
    mip_snapshots = []
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
            layer_weights = np.array(weights[layer_index])
        else:
            layer_weights = final_weights

        start_time = time.time()

        if layer_index == 0:
            print(f"Starting first layer, Layer: {layer_index}")
            pre_activation_mip_bounds, post_activation_mip_bounds = propagate_first_hidden_layer_from_inputs(
                layer_index,
                layer_weights,
                biases,
                input_size,
                hidden_layers,
                hidden_layers_activate,
                layerabs_state,
                deeppoly_bounds,
            )
            append_stage_snapshots(
                mip_snapshots,
                pre_activation_mip_bounds,
                post_activation_mip_bounds,
            )
            log_stage_execution_time(start_time)
            continue

        if layer_index == len(weights) - 1:
            print(f"Starting output layer, Layer: {layer_index}")
            pre_activation_mip_bounds = run_output_layer_stage(
                child_process_ids,
                stage_runners["output"],
                layer_index,
                layer_weights,
                biases,
                hidden_sizes,
                output_layer,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                variable_bounds_list,
                weights,
                final_weights,
                hidden_layers_activate,
                mip_backtrack_depth,
            )
            append_stage_snapshots(mip_snapshots, pre_activation_mip_bounds)
            log_stage_execution_time(start_time)
            continue

        if layer_index == len(weights):
            print(f"Starting final output layer, Layer: {layer_index}")
            pre_activation_mip_bounds = run_property_layer_stage(
                child_process_ids,
                stage_runners["property"],
                layer_index,
                layer_weights,
                final_weights,
                final_biases,
                output_layer,
                final_output_layer,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                variable_bounds_list,
                weights,
                output_size,
                final_output_layer_interval,
                mip_backtrack_depth,
            )
            append_stage_snapshots(mip_snapshots, pre_activation_mip_bounds)
            log_stage_execution_time(start_time)
            continue

        print(f"Starting hidden layer, Layer: {layer_index}")
        if layer_index == 1:
            pre_activation_mip_bounds, post_activation_mip_bounds = run_first_hidden_parallel_stage(
                child_process_ids,
                stage_runners["first_hidden"],
                layer_index,
                layer_weights,
                biases,
                hidden_sizes,
                hidden_layers,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                variable_bounds_list,
                hidden_layers_activate,
                mip_backtrack_depth,
            )
        else:
            pre_activation_mip_bounds, post_activation_mip_bounds = run_hidden_parallel_stage(
                child_process_ids,
                stage_runners["hidden"],
                layer_index,
                layer_weights,
                biases,
                hidden_sizes,
                hidden_layers,
                layerabs_state,
                deeppoly_bounds,
                layer_sizes,
                variable_bounds_list,
                weights,
                final_weights,
                hidden_layers_activate,
                mip_backtrack_depth,
            )

        append_stage_snapshots(
            mip_snapshots,
            pre_activation_mip_bounds,
            post_activation_mip_bounds,
        )
        log_stage_execution_time(start_time)

    return finalize_regular_run(
        start_time_all,
        final_output_layer_interval,
        layerabs_state,
        mip_snapshots,
        cleanup_callback=cleanup_callback,
    )
