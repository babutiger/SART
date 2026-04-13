"""Shared stage-level helpers for LayerABS family propagation runners."""

from __future__ import annotations

import copy
import numpy as np

from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    add_or_update_inner_list,
    copy_array_values,
    longe_np,
)
from sart.layerabs.layerabs_core.layerabs_runtime_helpers import (
    run_parallel_tasks,
)
from sart.layerabs.support.longge3 import (
    relu3,
    relu3_deeppoly_low,
    relu3_deeppoly_up,
)


def build_layer_sizes(hidden_sizes, output_size):
    """Return the per-stage width sequence including output and property layers."""
    return list(hidden_sizes) + [output_size, output_size - 1]


def collect_state_slot_symbols(layerabs_state, state_layer_index, neuron_count, state_slot):
    """Collect the symbolic identifiers from one LayerABS state slot."""
    return [
        layerabs_state[state_layer_index][neuron_index][state_slot][0]
        for neuron_index in range(neuron_count)
    ]


def append_state_slot_bounds(
    extended_variable_bounds,
    layerabs_state,
    state_layer_index,
    state_slot,
    neuron_count,
):
    """Append one LayerABS state slot's bound triples into the variable-bound list."""
    for neuron_index in range(neuron_count):
        variable_bounds_entry = [
            layerabs_state[state_layer_index][neuron_index][state_slot][0],
            layerabs_state[state_layer_index][neuron_index][state_slot][3],
            layerabs_state[state_layer_index][neuron_index][state_slot][4],
        ]
        extended_variable_bounds = add_or_update_inner_list(
            extended_variable_bounds,
            variable_bounds_entry,
        )
    return extended_variable_bounds


def merge_output_layer_results(
    results,
    layerabs_state,
    state_layer_index,
    save_mip_layer_before,
):
    """Write output-layer results back to LayerABS state and MIP snapshots."""
    for neuron_index, result in sorted(results, key=lambda item: item[0]):
        pre_activation_state, pre_activation_bounds = result
        copy_array_values(
            pre_activation_state,
            layerabs_state[state_layer_index][neuron_index][0],
        )
        save_mip_layer_before.append(pre_activation_bounds)


def merge_property_layer_results(
    results,
    layerabs_state,
    state_layer_index,
    property_layer_interval,
    save_mip_layer_before,
):
    """Write property-layer results back to LayerABS state and interval outputs."""
    for neuron_index, result in sorted(results, key=lambda item: item[0]):
        (
            pre_activation_state,
            property_layer_interval_entry,
            pre_activation_bounds,
        ) = result
        copy_array_values(
            pre_activation_state,
            layerabs_state[state_layer_index][neuron_index][0],
        )
        property_layer_interval[neuron_index] = property_layer_interval_entry[
            neuron_index
        ]
        save_mip_layer_before.append(pre_activation_bounds)


def merge_hidden_layer_results(
    results,
    layerabs_state,
    state_layer_index,
    save_mip_layer_before,
    save_mip_layer_after,
):
    """Write hidden-layer pre/post-activation results back to LayerABS state."""
    for neuron_index, result in sorted(results, key=lambda item: item[0]):
        (
            pre_activation_state,
            post_activation_state,
            pre_activation_bounds,
            post_activation_bounds,
        ) = result
        copy_array_values(
            pre_activation_state,
            layerabs_state[state_layer_index][neuron_index][0],
        )
        copy_array_values(
            post_activation_state,
            layerabs_state[state_layer_index][neuron_index][1],
        )
        save_mip_layer_before.append(pre_activation_bounds)
        save_mip_layer_after.append(post_activation_bounds)


def execute_output_stage_tasks(
    task_runner,
    task_args,
    layerabs_state,
    state_layer_index,
    max_parallel_workers,
):
    """Run output-layer tasks in parallel and merge their results."""
    save_mip_layer_before = []
    results = run_parallel_tasks(task_runner, task_args, max_parallel_workers)
    merge_output_layer_results(
        results,
        layerabs_state,
        state_layer_index,
        save_mip_layer_before,
    )
    return save_mip_layer_before


def execute_property_stage_tasks(
    task_runner,
    task_args,
    layerabs_state,
    state_layer_index,
    property_layer_interval,
    max_parallel_workers,
):
    """Run property-layer tasks in parallel and merge their results."""
    save_mip_layer_before = []
    results = run_parallel_tasks(task_runner, task_args, max_parallel_workers)
    merge_property_layer_results(
        results,
        layerabs_state,
        state_layer_index,
        property_layer_interval,
        save_mip_layer_before,
    )
    return save_mip_layer_before


def execute_hidden_stage_tasks(
    task_runner,
    task_args,
    layerabs_state,
    state_layer_index,
    max_parallel_workers,
):
    """Run hidden-layer tasks in parallel and merge pre/post-activation results."""
    save_mip_layer_before = []
    save_mip_layer_after = []
    results = run_parallel_tasks(task_runner, task_args, max_parallel_workers)
    merge_hidden_layer_results(
        results,
        layerabs_state,
        state_layer_index,
        save_mip_layer_before,
        save_mip_layer_after,
    )
    return save_mip_layer_before, save_mip_layer_after


def extend_recent_variable_bounds_window(
    variable_bounds,
    layer_index,
    resolve_history_width,
    append_slot_bounds,
):
    """Extend variable bounds using the recent three-stage LayerABS window."""
    extended_variable_bounds = copy.deepcopy(variable_bounds)
    if layer_index < 3:
        return extended_variable_bounds

    for history_layer_index in range(layer_index - 3, layer_index):
        history_layer_width = resolve_history_width(history_layer_index)
        state_slot = 0 if history_layer_index == layer_index - 1 else 1
        extended_variable_bounds = append_slot_bounds(
            extended_variable_bounds,
            history_layer_index,
            state_slot,
            history_layer_width,
        )
    return extended_variable_bounds


def extend_property_layer_variable_bounds_window(
    variable_bounds,
    layer_index,
    resolve_property_history_spec,
    append_slot_bounds,
):
    """Extend variable bounds for the final property layer backtracking window."""
    extended_variable_bounds = copy.deepcopy(variable_bounds)
    if layer_index < 3:
        return extended_variable_bounds

    for history_layer_index in range(layer_index - 3, layer_index):
        state_slot, history_layer_width = resolve_property_history_spec(
            history_layer_index
        )
        extended_variable_bounds = append_slot_bounds(
            extended_variable_bounds,
            history_layer_index,
            state_slot,
            history_layer_width,
        )
    return extended_variable_bounds


def propagate_first_hidden_layer_from_inputs(
    layer_index,
    layer_weights,
    network_biases,
    pre_activation_symbols,
    post_activation_symbols,
    layerabs_state,
    deeppoly_bounds,
    previous_layer_symbols,
):
    """Propagate the input layer into the first hidden layer and record ReLU state."""
    save_mip_layer_before = []
    save_mip_layer_after = []

    for neuron_index in range(len(layer_weights)):
        pre_activation_bounds = []
        post_activation_bounds = []

        neuron_output = longe_np(
            layer_weights[neuron_index],
            previous_layer_symbols,
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
        pre_activation_lower_bound += float(
            np.array(network_biases[layer_index][neuron_index])
        )

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
        pre_activation_upper_bound += float(
            np.array(network_biases[layer_index][neuron_index])
        )

        pre_activation_symbol = str(pre_activation_symbols[neuron_index])
        pre_activation_equation = pre_activation_symbol + "==" + neuron_output
        layerabs_state[layer_index + 1][neuron_index][0][0] = pre_activation_symbol
        layerabs_state[layer_index + 1][neuron_index][0][1] = neuron_output
        layerabs_state[layer_index + 1][neuron_index][0][2] = pre_activation_equation
        layerabs_state[layer_index + 1][neuron_index][0][3] = (
            pre_activation_lower_bound
        )
        layerabs_state[layer_index + 1][neuron_index][0][4] = (
            pre_activation_upper_bound
        )
        layerabs_state[layer_index + 1][neuron_index][0][5] = (
            pre_activation_symbol + ">=" + neuron_output
        )
        layerabs_state[layer_index + 1][neuron_index][0][6] = (
            pre_activation_symbol + "<=" + neuron_output
        )
        layerabs_state[layer_index + 1][neuron_index][0][7] = deeppoly_bounds[
            layer_index * 2
        ][neuron_index][0]
        layerabs_state[layer_index + 1][neuron_index][0][8] = deeppoly_bounds[
            layer_index * 2
        ][neuron_index][1]

        pre_activation_bounds.append(pre_activation_lower_bound)
        pre_activation_bounds.append(pre_activation_upper_bound)
        save_mip_layer_before.append(pre_activation_bounds)

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

        post_activation_symbol = str(post_activation_symbols[neuron_index])
        post_activation_equation = post_activation_symbol + "==" + relu_output
        layerabs_state[layer_index + 1][neuron_index][1][0] = post_activation_symbol
        layerabs_state[layer_index + 1][neuron_index][1][1] = relu_output
        layerabs_state[layer_index + 1][neuron_index][1][2] = post_activation_equation
        layerabs_state[layer_index + 1][neuron_index][1][3] = (
            post_activation_lower_bound
        )
        layerabs_state[layer_index + 1][neuron_index][1][4] = (
            post_activation_upper_bound
        )
        layerabs_state[layer_index + 1][neuron_index][1][5] = (
            post_activation_symbol + ">=" + relu_lower_form
        )
        layerabs_state[layer_index + 1][neuron_index][1][6] = (
            post_activation_symbol + "<=" + relu_upper_form
        )
        layerabs_state[layer_index + 1][neuron_index][1][7] = deeppoly_bounds[
            layer_index * 2 + 1
        ][neuron_index][0]
        layerabs_state[layer_index + 1][neuron_index][1][8] = deeppoly_bounds[
            layer_index * 2 + 1
        ][neuron_index][1]

        post_activation_bounds.append(post_activation_lower_bound)
        post_activation_bounds.append(post_activation_upper_bound)
        save_mip_layer_after.append(post_activation_bounds)

    return save_mip_layer_before, save_mip_layer_after
