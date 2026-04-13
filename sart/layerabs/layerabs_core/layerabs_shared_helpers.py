from __future__ import annotations

import os
import signal

import sympy as sp
from sart.layerabs.support.longge3 import add, mul
from sart.layerabs.support.read_nnet import (
    read_nnet_file,
)
from sart.layerabs.support.read_property import (
    input_bound_pair,
    read_property,
)


def add_or_update_inner_list(lst, new_element):
    """Update an inner list by key, or append it when the key is new."""
    first_element = new_element[0]

    for inner_list_index, inner_list in enumerate(lst):
        if inner_list[0] == first_element:
            lst[inner_list_index] = new_element
            return lst

    lst.append(new_element)
    return lst


def copy_array_values(original_array, target_array):
    if len(target_array) < len(original_array):
        raise ValueError("Target array is not long enough to hold all values.")

    for value_index in range(len(original_array)):
        target_array[value_index] = original_array[value_index]


def generate_variable_bounds(
    input_layer_symbols,
    bound_pair,
    hidden_layer_symbols,
    activated_hidden_layer_symbols,
    output_layer_symbols,
    property_layer_symbols,
):
    if len(input_layer_symbols) != len(bound_pair):
        raise ValueError("input_layer_symbols and bound_pair must have the same length.")

    variable_bounds = []
    for input_index in range(len(input_layer_symbols)):
        variable_bounds.append(
            [str(input_layer_symbols[input_index]), bound_pair[input_index][0], bound_pair[input_index][1]]
        )

    if len(hidden_layer_symbols) != len(activated_hidden_layer_symbols):
        raise ValueError(
            "hidden_layer_symbols and activated_hidden_layer_symbols must have the same number of layers."
        )

    for hidden_layer_index in range(len(hidden_layer_symbols)):
        if len(hidden_layer_symbols[hidden_layer_index]) != len(
            activated_hidden_layer_symbols[hidden_layer_index]
        ):
            raise ValueError(
                f"Layer {hidden_layer_index + 1} must expose the same number of hidden and activated symbols."
            )
        for neuron_index in range(len(hidden_layer_symbols[hidden_layer_index])):
            variable_bounds.append([str(hidden_layer_symbols[hidden_layer_index][neuron_index]), None, None])
        for neuron_index in range(len(activated_hidden_layer_symbols[hidden_layer_index])):
            variable_bounds.append(
                [str(activated_hidden_layer_symbols[hidden_layer_index][neuron_index]), None, None]
            )

    for var in output_layer_symbols:
        variable_bounds.append([str(var), None, None])

    for var in property_layer_symbols:
        variable_bounds.append([str(var), None, None])

    return variable_bounds


def store_data_by_4dlist(nnet, property_path, delta_num):
    network_weights, network_biases, input_size, hidden_sizes, output_size = read_nnet_file(nnet)
    input_pixel_list, property_layer_weights, property_layer_biases = read_property(property_path)

    final_output_size = output_size - 1
    bound_pair = input_bound_pair(input_pixel_list, delta_num)

    input_layer_symbols = [sp.symbols(f"x0_{input_index}") for input_index in range(input_size)]
    hidden_layer_symbols = [
        [
            sp.symbols("x{0}_{1}".format(hidden_layer_index + 1, neuron_index))
            for neuron_index in range(hidden_sizes[hidden_layer_index])
        ]
        for hidden_layer_index in range(len(hidden_sizes))
    ]
    activated_hidden_layer_symbols = [
        [
            sp.symbols("x{0}_{1}_a".format(hidden_layer_index + 1, neuron_index))
            for neuron_index in range(hidden_sizes[hidden_layer_index])
        ]
        for hidden_layer_index in range(len(hidden_sizes))
    ]
    output_layer_symbols = [
        sp.symbols(f"x{len(hidden_sizes) + 1}_{output_neuron_index}")
        for output_neuron_index in range(output_size)
    ]
    property_layer_symbols = [
        sp.symbols(f"x{len(hidden_sizes) + 2}_{property_neuron_index}")
        for property_neuron_index in range(final_output_size)
    ]

    dim1_size = len(hidden_sizes) + 3
    dim2_size = input_size
    dim3_size = 2
    dim4_size = 9

    layerabs_state = [
        [
            [[None for _ in range(dim4_size)] for _ in range(dim3_size)]
            for _ in range(dim2_size)
        ]
        for _ in range(dim1_size)
    ]

    for input_index in range(dim2_size):
        for state_slot in range(dim3_size):
            layerabs_state[0][input_index][state_slot][0] = str(input_layer_symbols[input_index])
            layerabs_state[0][input_index][state_slot][1] = str(input_layer_symbols[input_index])
            layerabs_state[0][input_index][state_slot][2] = str(input_layer_symbols[input_index])
            layerabs_state[0][input_index][state_slot][3] = bound_pair[input_index][0]
            layerabs_state[0][input_index][state_slot][4] = bound_pair[input_index][1]
            layerabs_state[0][input_index][state_slot][5] = str(input_layer_symbols[input_index])
            layerabs_state[0][input_index][state_slot][6] = str(input_layer_symbols[input_index])
            layerabs_state[0][input_index][state_slot][7] = str(input_layer_symbols[input_index])
            layerabs_state[0][input_index][state_slot][8] = str(input_layer_symbols[input_index])

    return (
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
    )


def create_fdl_directory():
    folder_name = "fdl"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        os.chmod(folder_name, 0o777)
    return folder_name


def extract_values1(layerabs_state, layer_index, backtrack_depth, neuron_index, layer_sizes):
    """Extract equality constraints for one target neuron within the backtrack window."""
    extracted_values = [layerabs_state[layer_index + 1][neuron_index][0][2]]

    for current_layer in range(layer_index, layer_index - backtrack_depth, -1):
        if current_layer < len(layerabs_state):
            for source_neuron_index in range(layer_sizes[current_layer - 1]):
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    for current_layer in range(layer_index - backtrack_depth, 0, -1):
        for source_neuron_index in range(layer_sizes[current_layer - 1]):
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    return extracted_values


def extract_values2(layerabs_state, layer_index, backtrack_depth, neuron_index, layer_sizes):
    """Extract equality and older DeepPoly constraints for one target neuron."""
    extracted_values = [layerabs_state[layer_index + 1][neuron_index][0][2]]

    for current_layer in range(layer_index, layer_index - backtrack_depth, -1):
        if current_layer < len(layerabs_state):
            for source_neuron_index in range(layer_sizes[current_layer - 1]):
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    for current_layer in range(layer_index - backtrack_depth, 0, -1):
        for source_neuron_index in range(layer_sizes[current_layer - 1]):
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][5])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][6])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][5])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][6])

    return extracted_values


def extract_values3(layerabs_state, layer_index, backtrack_depth, neuron_index, layer_sizes):
    """Extract property-layer constraints plus the backtracked DeepPoly context."""
    extracted_values = [layerabs_state[layer_index + 1][neuron_index][0][2]]

    for output_neuron_index in range(layer_sizes[-2]):
        extracted_values.append(layerabs_state[layer_index][output_neuron_index][0][2])

    for current_layer in range(layer_index - 1, layer_index - backtrack_depth - 1, -1):
        if current_layer < len(layerabs_state):
            for source_neuron_index in range(layer_sizes[current_layer - 1]):
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    for current_layer in range(layer_index - backtrack_depth - 1, 0, -1):
        for source_neuron_index in range(layer_sizes[current_layer - 1]):
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][5])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][6])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][5])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][6])

    return extracted_values


def extract_values1_complete(layerabs_state, layer_index, backtrack_depth, neuron_index, layer_sizes):
    """Extract complete-pass equality constraints for one target neuron."""
    extracted_values = [layerabs_state[layer_index + 1][neuron_index][0][2]]

    for current_layer in range(layer_index, layer_index - backtrack_depth, -1):
        if current_layer < len(layerabs_state):
            for source_neuron_index in range(layer_sizes[current_layer - 1]):
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    for current_layer in range(layer_index - backtrack_depth, 0, -1):
        for source_neuron_index in range(layer_sizes[current_layer - 1]):
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    return extracted_values


def extract_values2_complete(layerabs_state, layer_index, backtrack_depth, neuron_index, layer_sizes):
    """Extract complete-pass constraints while keeping older equality forms."""
    extracted_values = [layerabs_state[layer_index + 1][neuron_index][0][2]]

    for current_layer in range(layer_index, layer_index - backtrack_depth, -1):
        if current_layer < len(layerabs_state):
            for source_neuron_index in range(layer_sizes[current_layer - 1]):
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    for current_layer in range(layer_index - backtrack_depth, 0, -1):
        for source_neuron_index in range(layer_sizes[current_layer - 1]):
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    return extracted_values


def extract_values3_complete(layerabs_state, layer_index, backtrack_depth, neuron_index, layer_sizes):
    """Extract complete-pass property-layer constraints plus the backtracked context."""
    extracted_values = [layerabs_state[layer_index + 1][neuron_index][0][2]]

    for output_neuron_index in range(layer_sizes[-2]):
        extracted_values.append(layerabs_state[layer_index][output_neuron_index][0][2])

    for current_layer in range(layer_index - 1, layer_index - backtrack_depth - 1, -1):
        if current_layer < len(layerabs_state):
            for source_neuron_index in range(layer_sizes[current_layer - 1]):
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][2])
                extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][2])

    for current_layer in range(layer_index - backtrack_depth - 1, 0, -1):
        for source_neuron_index in range(layer_sizes[current_layer - 1]):
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][5])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][0][6])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][5])
            extracted_values.append(layerabs_state[current_layer][source_neuron_index][1][6])

    return extracted_values


def longe_np(weight_matrix_lg, input_list_lg, bias_lg):
    neuron_output_longge = [""]
    for idx, ii in enumerate(weight_matrix_lg):
        vvv = str(input_list_lg[idx])
        if neuron_output_longge[0] == "":
            neuron_output_longge = mul(str(ii), vvv)
        else:
            neuron_output_longge = add(neuron_output_longge, mul(str(ii), vvv))

    neuron_output_longge = add(neuron_output_longge, str(bias_lg))
    return neuron_output_longge


def check_negative_upper_bound(array):
    for sub_array in array:
        if sub_array[-1] >= 0:
            return False
    return True


def terminate_managed_child_processes(child_process_ids):
    for pid in child_process_ids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    child_process_ids[:] = []


def replace_tuple_element(tup, index, new_value):
    if index < 0 or index >= len(tup):
        raise IndexError("Tuple index is out of range.")

    temp_list = list(tup)
    temp_list[index] = new_value
    return tuple(temp_list)
