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
    build_no_abstraction_property_layer_task_args,
    build_no_abstraction_first_hidden_layer_task_args,
    build_no_abstraction_hidden_layer_task_args,
    build_no_abstraction_output_layer_task_args,
)
from sart.layerabs.layerabs_core.layerabs_parallel_task_helpers import (
    run_parallel_first_hidden_layer_task_complete_no_abstraction,
    run_parallel_hidden_layer_task_complete_no_abstraction,
    run_parallel_output_layer_task_complete_no_abstraction,
    run_parallel_property_layer_task_complete_no_abstraction,
)
from sart.layerabs.layerabs_core.layerabs_runtime_helpers import (
    append_stage_snapshots,
    finalize_propagation_run,
    log_stage_execution_time,
)
from sart.layerabs.layerabs_core.layerabs_stage_runner_helpers import (
    run_first_hidden_stage_with_builder,
    run_hidden_stage_with_builder,
    run_output_stage_with_builder,
    run_property_stage_with_builder,
)
from sart.layerabs.layerabs_core.layerabs_stage_helpers import (
    build_layer_sizes,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    add_or_update_inner_list,
    generate_variable_bounds,
    longe_np,
)
from sart.layerabs.layerabs_core.layerabs_stats_helpers import (
    print_unstable_neuron_summary,
)


MAX_PARALLEL_CORES = 28

def _extend_recent_variable_bounds(
    variable_bounds_list,
    *,
    four_dimensional_list,
    weights,
    final_weights,
    layer_index,
):
    variable_bounds_list_add = copy.deepcopy(variable_bounds_list)
    if layer_index < 3:
        return variable_bounds_list_add

    for history_layer in range(layer_index - 3, layer_index):
        if history_layer != len(weights):
            weight_matrix_r = np.array(weights[history_layer])
        else:
            weight_matrix_r = final_weights

        state_index = 0 if history_layer == layer_index - 1 else 1
        for neuron_index in range(len(weight_matrix_r)):
            variable_bounds_list_add = add_or_update_inner_list(
                variable_bounds_list_add,
                [
                    four_dimensional_list[history_layer + 1][neuron_index][state_index][0],
                    four_dimensional_list[history_layer + 1][neuron_index][state_index][3],
                    four_dimensional_list[history_layer + 1][neuron_index][state_index][4],
                ],
            )

    return variable_bounds_list_add


def _extend_final_layer_variable_bounds(
    variable_bounds_list,
    *,
    four_dimensional_list,
    weights,
    final_weights,
    layer_index,
):
    variable_bounds_list_add = copy.deepcopy(variable_bounds_list)
    if layer_index < 3:
        return variable_bounds_list_add

    for history_layer in range(layer_index - 3, layer_index):
        if history_layer != len(weights) and history_layer != len(weights) - 1:
            weight_matrix_r = np.array(weights[history_layer])
            for neuron_index in range(len(weight_matrix_r)):
                variable_bounds_list_add = add_or_update_inner_list(
                    variable_bounds_list_add,
                    [
                        four_dimensional_list[history_layer + 1][neuron_index][1][0],
                        four_dimensional_list[history_layer + 1][neuron_index][1][3],
                        four_dimensional_list[history_layer + 1][neuron_index][1][4],
                    ],
                )
        else:
            for neuron_index in range(len(final_weights) + 1):
                variable_bounds_list_add = add_or_update_inner_list(
                    variable_bounds_list_add,
                    [
                        four_dimensional_list[history_layer + 1][neuron_index][0][0],
                        four_dimensional_list[history_layer + 1][neuron_index][0][3],
                        four_dimensional_list[history_layer + 1][neuron_index][0][4],
                    ],
                )

    return variable_bounds_list_add


def _run_first_weight_layer(
    *,
    weight_matrix,
    biases,
    input_size,
    hidden_layers,
    hidden_layers_activate,
    four_dimensional_list,
    save_2mip,
):
    input_layer_list = [
        four_dimensional_list[0][input_index][0][0]
        for input_index in range(input_size)
    ]

    save_mip_layer_before = []
    save_mip_layer_after = []

    for neuron_index in range(len(weight_matrix)):
        neuron_output = longe_np(
            weight_matrix[neuron_index],
            input_layer_list,
            biases[0][neuron_index][0],
        )

        temp_lower_before = 0
        temp_upper_before = 0
        temp_lower_after = 0
        temp_upper_after = 0

        for input_index in range(len(weight_matrix[neuron_index])):
            weight = float(weight_matrix[neuron_index][input_index])
            if weight <= 0:
                temp_lower_before += (
                    weight * four_dimensional_list[0][input_index][0][4]
                )
                temp_upper_before += (
                    weight * four_dimensional_list[0][input_index][0][3]
                )
            else:
                temp_lower_before += (
                    weight * four_dimensional_list[0][input_index][0][3]
                )
                temp_upper_before += (
                    weight * four_dimensional_list[0][input_index][0][4]
                )

        temp_lower_before += float(np.array(biases[0][neuron_index]))
        temp_upper_before += float(np.array(biases[0][neuron_index]))

        four_dimensional_list[1][neuron_index][0][0] = str(hidden_layers[0][neuron_index])
        four_dimensional_list[1][neuron_index][0][1] = neuron_output
        four_dimensional_list[1][neuron_index][0][2] = (
            str(hidden_layers[0][neuron_index]) + "==" + neuron_output
        )
        four_dimensional_list[1][neuron_index][0][3] = temp_lower_before
        four_dimensional_list[1][neuron_index][0][4] = temp_upper_before
        four_dimensional_list[1][neuron_index][0][5] = (
            str(hidden_layers[0][neuron_index]) + ">=" + neuron_output
        )
        four_dimensional_list[1][neuron_index][0][6] = (
            str(hidden_layers[0][neuron_index]) + "<=" + neuron_output
        )

        save_mip_layer_before.append([temp_lower_before, temp_upper_before])

        output_layer_relu = relu3(
            str(hidden_layers[0][neuron_index]),
            temp_lower_before,
            temp_upper_before,
        )
        output_layer_relu_deeppoly_low = relu3_deeppoly_low(
            str(hidden_layers[0][neuron_index]),
            temp_lower_before,
            temp_upper_before,
        )
        output_layer_relu_deeppoly_up = relu3_deeppoly_up(
            str(hidden_layers[0][neuron_index]),
            temp_lower_before,
            temp_upper_before,
        )

        if temp_lower_before > 0:
            temp_lower_after = temp_lower_before
            temp_upper_after = temp_upper_before
        if temp_upper_before < 0:
            temp_lower_after = 0
            temp_upper_after = 0
        if temp_lower_before <= 0 and temp_upper_before >= 0:
            temp_lower_after = 0
            temp_upper_after = temp_upper_before

        four_dimensional_list[1][neuron_index][1][0] = str(
            hidden_layers_activate[0][neuron_index]
        )
        four_dimensional_list[1][neuron_index][1][1] = output_layer_relu
        four_dimensional_list[1][neuron_index][1][2] = (
            str(hidden_layers_activate[0][neuron_index]) + "==" + output_layer_relu
        )
        four_dimensional_list[1][neuron_index][1][3] = temp_lower_after
        four_dimensional_list[1][neuron_index][1][4] = temp_upper_after
        four_dimensional_list[1][neuron_index][1][5] = (
            str(hidden_layers_activate[0][neuron_index])
            + ">="
            + output_layer_relu_deeppoly_low
        )
        four_dimensional_list[1][neuron_index][1][6] = (
            str(hidden_layers_activate[0][neuron_index])
            + "<="
            + output_layer_relu_deeppoly_up
        )

        save_mip_layer_after.append([temp_lower_after, temp_upper_after])

    append_stage_snapshots(save_2mip, save_mip_layer_before, save_mip_layer_after)


def _run_output_layer(
    *,
    child_process_ids,
    layer_index,
    weight_matrix,
    weights,
    biases,
    hidden_sizes,
    final_weights,
    output_layer,
    hidden_layers_activate,
    four_dimensional_list,
    effective_layer_sizes,
    variable_bounds_list,
    save_2mip,
):
    save_mip_layer_before = run_output_stage_with_builder(
        run_parallel_output_layer_task_complete_no_abstraction,
        layer_index=layer_index,
        previous_layer_width=hidden_sizes[layer_index - 1],
        layerabs_state=four_dimensional_list,
        build_extended_variable_bounds=lambda: _extend_recent_variable_bounds(
            variable_bounds_list,
            four_dimensional_list=four_dimensional_list,
            weights=weights,
            final_weights=final_weights,
            layer_index=layer_index,
        ),
        build_task_args=lambda input_layer_list, variable_bounds_list_add: (
            build_no_abstraction_output_layer_task_args(
                child_process_ids,
                layer_index,
                weight_matrix,
                input_layer_list,
                biases,
                output_layer,
                four_dimensional_list,
                effective_layer_sizes,
                variable_bounds_list_add,
                hidden_layers_activate,
                [],
            )
        ),
        max_parallel_workers=MAX_PARALLEL_CORES,
    )
    append_stage_snapshots(save_2mip, save_mip_layer_before)


def _run_final_output_layer(
    *,
    child_process_ids,
    layer_index,
    weight_matrix,
    weights,
    final_weights,
    final_biases,
    output_layer,
    final_output_layer,
    four_dimensional_list,
    effective_layer_sizes,
    variable_bounds_list,
    output_size,
    final_output_layer_interval,
    save_2mip,
):
    save_mip_layer_before = run_property_stage_with_builder(
        run_parallel_property_layer_task_complete_no_abstraction,
        layer_index=layer_index,
        layerabs_state=four_dimensional_list,
        property_layer_interval=final_output_layer_interval,
        build_extended_variable_bounds=lambda: _extend_final_layer_variable_bounds(
            variable_bounds_list,
            four_dimensional_list=four_dimensional_list,
            weights=weights,
            final_weights=final_weights,
            layer_index=layer_index,
        ),
        build_task_args=lambda variable_bounds_list_add: (
            build_no_abstraction_property_layer_task_args(
                child_process_ids,
                layer_index,
                weight_matrix,
                final_weights,
                final_biases,
                output_layer,
                final_output_layer,
                four_dimensional_list,
                effective_layer_sizes,
                variable_bounds_list_add,
                output_size,
                [],
            )
        ),
        max_parallel_workers=MAX_PARALLEL_CORES,
    )
    append_stage_snapshots(save_2mip, save_mip_layer_before)


def _run_hidden_layer(
    *,
    child_process_ids,
    layer_index,
    weight_matrix,
    weights,
    biases,
    input_size,
    hidden_sizes,
    hidden_layers,
    hidden_layers_activate,
    final_weights,
    four_dimensional_list,
    effective_layer_sizes,
    variable_bounds_list,
    save_2mip,
):
    if layer_index == 1:
        save_mip_layer_before, save_mip_layer_after = run_first_hidden_stage_with_builder(
            run_parallel_first_hidden_layer_task_complete_no_abstraction,
            layer_index=layer_index,
            previous_layer_width=hidden_sizes[layer_index - 1],
            layerabs_state=four_dimensional_list,
            build_task_args=lambda input_layer_list: (
                build_no_abstraction_first_hidden_layer_task_args(
                    child_process_ids,
                    layer_index,
                    weight_matrix,
                    input_layer_list,
                    biases,
                    hidden_layers,
                    four_dimensional_list,
                    effective_layer_sizes,
                    variable_bounds_list,
                    hidden_layers_activate,
                    [],
                    [],
                )
            ),
            max_parallel_workers=MAX_PARALLEL_CORES,
        )
    else:
        save_mip_layer_before, save_mip_layer_after = run_hidden_stage_with_builder(
            run_parallel_hidden_layer_task_complete_no_abstraction,
            layer_index=layer_index,
            previous_layer_width=hidden_sizes[layer_index - 1],
            layerabs_state=four_dimensional_list,
            build_extended_variable_bounds=lambda: _extend_recent_variable_bounds(
                variable_bounds_list,
                four_dimensional_list=four_dimensional_list,
                weights=weights,
                final_weights=final_weights,
                layer_index=layer_index,
            ),
            build_task_args=lambda input_layer_list, variable_bounds_list_add: (
                build_no_abstraction_hidden_layer_task_args(
                    child_process_ids,
                    layer_index,
                    weight_matrix,
                    input_layer_list,
                    biases,
                    hidden_layers,
                    four_dimensional_list,
                    effective_layer_sizes,
                    variable_bounds_list_add,
                    hidden_layers_activate,
                    [],
                    [],
                )
            ),
            max_parallel_workers=MAX_PARALLEL_CORES,
        )

    append_stage_snapshots(save_2mip, save_mip_layer_before, save_mip_layer_after)

def run_no_abstraction_forward_propagation(
    *,
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
    child_process_ids,
    emit_stats=False,
    print_nonhidden_layer_timing=False,
):
    final_output_layer_interval = [None] * (output_size - 1)
    save_2mip = []
    effective_layer_sizes = build_layer_sizes(hidden_sizes, output_size)
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
            print(f"Starting first layer,  Layer: {layer_index}")
            _run_first_weight_layer(
                weight_matrix=weight_matrix,
                biases=biases,
                input_size=input_size,
                hidden_layers=hidden_layers,
                hidden_layers_activate=hidden_layers_activate,
                four_dimensional_list=four_dimensional_list,
                save_2mip=save_2mip,
            )
            if print_nonhidden_layer_timing:
                log_stage_execution_time(start_time)
        elif layer_index == len(weights) - 1:
            print(f"Starting output layer, Layer: {layer_index}")
            _run_output_layer(
                child_process_ids=child_process_ids,
                layer_index=layer_index,
                weight_matrix=weight_matrix,
                weights=weights,
                biases=biases,
                hidden_sizes=hidden_sizes,
                final_weights=final_weights,
                output_layer=output_layer,
                hidden_layers_activate=hidden_layers_activate,
                four_dimensional_list=four_dimensional_list,
                effective_layer_sizes=effective_layer_sizes,
                variable_bounds_list=variable_bounds_list,
                save_2mip=save_2mip,
            )
            if print_nonhidden_layer_timing:
                log_stage_execution_time(start_time)
        elif layer_index == len(weights):
            print(f"Starting final output layer, Layer: {layer_index}")
            _run_final_output_layer(
                child_process_ids=child_process_ids,
                layer_index=layer_index,
                weight_matrix=weight_matrix,
                weights=weights,
                final_weights=final_weights,
                final_biases=final_biases,
                output_layer=output_layer,
                final_output_layer=final_output_layer,
                four_dimensional_list=four_dimensional_list,
                effective_layer_sizes=effective_layer_sizes,
                variable_bounds_list=variable_bounds_list,
                output_size=output_size,
                final_output_layer_interval=final_output_layer_interval,
                save_2mip=save_2mip,
            )
            if print_nonhidden_layer_timing:
                log_stage_execution_time(start_time)
        else:
            print(f"Starting hidden layer, Layer: {layer_index}")
            _run_hidden_layer(
                child_process_ids=child_process_ids,
                layer_index=layer_index,
                weight_matrix=weight_matrix,
                weights=weights,
                biases=biases,
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                hidden_layers=hidden_layers,
                hidden_layers_activate=hidden_layers_activate,
                final_weights=final_weights,
                four_dimensional_list=four_dimensional_list,
                effective_layer_sizes=effective_layer_sizes,
                variable_bounds_list=variable_bounds_list,
                save_2mip=save_2mip,
            )
            log_stage_execution_time(start_time)

    final_output_layer_interval, temp_file_path, save_2mip = finalize_propagation_run(
        start_time_all=start_time_all,
        interval_label="final_output_layer_interval",
        interval_values=final_output_layer_interval,
        layerabs_state=four_dimensional_list,
        snapshot_filename="four_dimensional_list.pkl",
        mip_snapshots=save_2mip,
    )
    if emit_stats:
        print_unstable_neuron_summary(four_dimensional_list)

    return final_output_layer_interval, temp_file_path, save_2mip
