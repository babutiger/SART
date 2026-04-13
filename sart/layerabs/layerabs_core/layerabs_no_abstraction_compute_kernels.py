from __future__ import annotations

from sart.layerabs.support.longge3 import (
    relu3,
    relu3_deeppoly_low,
    relu3_deeppoly_up,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    longe_np,
)
from sart.layerabs.layerabs_core.layerabs_solver_helpers import (
    optimize_with_bounds,
)


def _initialize_symbolic_state(layer_slot, symbol_name, neuron_output):
    state = [None] * 9
    symbolic_equality = symbol_name + "==" + neuron_output
    symbolic_lower_form = symbol_name + "==" + neuron_output
    symbolic_upper_form = symbol_name + "==" + neuron_output

    layer_slot[0] = symbol_name
    layer_slot[1] = neuron_output
    layer_slot[2] = symbolic_equality
    layer_slot[5] = symbolic_lower_form
    layer_slot[6] = symbolic_upper_form

    state[0] = symbol_name
    state[1] = neuron_output
    state[2] = symbolic_equality
    state[5] = symbolic_lower_form
    state[6] = symbolic_upper_form
    return state


def _store_interval_bounds(layer_slot, state, lower_bound, upper_bound):
    layer_slot[3] = lower_bound
    layer_slot[4] = upper_bound
    state[3] = lower_bound
    state[4] = upper_bound
    return [lower_bound, upper_bound]


def _resolve_relu_interval(lower_bound, upper_bound):
    if lower_bound > 0:
        return lower_bound, upper_bound
    if upper_bound < 0:
        return 0, 0
    return 0, upper_bound


def compute_first_hidden_layer_neuron_no_abstraction_generic(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    layer_sizes,
    variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    _unused_post_activation_mip_bounds,
    extract_values1_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    neuron_output = longe_np(
        layer_weights[neuron_index],
        input_layer_symbols,
        network_biases[layer_index][neuron_index][0],
    )
    pre_activation_slot = layerabs_state[layer_index + 1][neuron_index][0]
    pre_activation_symbol = str(hidden_layer_symbols[layer_index][neuron_index])
    pre_activation_state = _initialize_symbolic_state(
        pre_activation_slot,
        pre_activation_symbol,
        neuron_output,
    )

    constraints = extract_values1_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
    bound_result = optimize_bounds_fn(constraints, pre_activation_symbol, variable_bounds)
    pre_activation_bounds = _store_interval_bounds(
        pre_activation_slot,
        pre_activation_state,
        bound_result[0],
        bound_result[1],
    )

    relu_output = relu3(
        pre_activation_symbol,
        pre_activation_slot[3],
        pre_activation_slot[4],
    )
    post_activation_lower_bound, post_activation_upper_bound = _resolve_relu_interval(
        pre_activation_slot[3],
        pre_activation_slot[4],
    )

    post_activation_symbol = str(activated_hidden_layer_symbols[layer_index][neuron_index])
    post_activation_slot = layerabs_state[layer_index + 1][neuron_index][1]
    post_activation_state = [None] * 9
    activated_symbolic_equality = post_activation_symbol + "==" + relu_output
    post_activation_slot[0] = post_activation_symbol
    post_activation_slot[1] = relu_output
    post_activation_slot[2] = activated_symbolic_equality
    post_activation_slot[3] = post_activation_lower_bound
    post_activation_slot[4] = post_activation_upper_bound
    post_activation_state[0] = post_activation_symbol
    post_activation_state[1] = relu_output
    post_activation_state[2] = activated_symbolic_equality
    post_activation_state[3] = post_activation_lower_bound
    post_activation_state[4] = post_activation_upper_bound

    relu_lower_form = relu3_deeppoly_low(
        pre_activation_symbol,
        pre_activation_slot[3],
        pre_activation_slot[4],
    )
    relu_upper_form = relu3_deeppoly_up(
        pre_activation_symbol,
        pre_activation_slot[3],
        pre_activation_slot[4],
    )
    activated_symbolic_lower_form = post_activation_symbol + ">=" + relu_lower_form
    activated_symbolic_upper_form = post_activation_symbol + "<=" + relu_upper_form
    post_activation_slot[5] = activated_symbolic_lower_form
    post_activation_slot[6] = activated_symbolic_upper_form
    post_activation_state[5] = activated_symbolic_lower_form
    post_activation_state[6] = activated_symbolic_upper_form

    post_activation_bounds = [post_activation_lower_bound, post_activation_upper_bound]
    return pre_activation_state, post_activation_state, pre_activation_bounds, post_activation_bounds


def compute_hidden_layer_neuron_no_abstraction_generic(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    _unused_post_activation_mip_bounds,
    extract_values1_fn,
    extract_values2_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    neuron_output = longe_np(
        layer_weights[neuron_index],
        input_layer_symbols,
        network_biases[layer_index][neuron_index][0],
    )
    pre_activation_slot = layerabs_state[layer_index + 1][neuron_index][0]
    pre_activation_symbol = str(hidden_layer_symbols[layer_index][neuron_index])
    pre_activation_state = _initialize_symbolic_state(
        pre_activation_slot,
        pre_activation_symbol,
        neuron_output,
    )

    if layer_index == 2:
        constraints = extract_values1_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
    else:
        constraints = extract_values2_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)

    bound_result = optimize_bounds_fn(
        constraints,
        pre_activation_symbol,
        extended_variable_bounds,
    )
    pre_activation_bounds = _store_interval_bounds(
        pre_activation_slot,
        pre_activation_state,
        bound_result[0],
        bound_result[1],
    )

    relu_output = relu3(
        pre_activation_symbol,
        pre_activation_slot[3],
        pre_activation_slot[4],
    )
    post_activation_lower_bound, post_activation_upper_bound = _resolve_relu_interval(
        pre_activation_slot[3],
        pre_activation_slot[4],
    )

    post_activation_symbol = str(activated_hidden_layer_symbols[layer_index][neuron_index])
    post_activation_slot = layerabs_state[layer_index + 1][neuron_index][1]
    post_activation_state = [None] * 9
    activated_symbolic_equality = post_activation_symbol + "==" + relu_output
    post_activation_slot[0] = post_activation_symbol
    post_activation_slot[1] = relu_output
    post_activation_slot[2] = activated_symbolic_equality
    post_activation_slot[3] = post_activation_lower_bound
    post_activation_slot[4] = post_activation_upper_bound
    post_activation_state[0] = post_activation_symbol
    post_activation_state[1] = relu_output
    post_activation_state[2] = activated_symbolic_equality
    post_activation_state[3] = post_activation_lower_bound
    post_activation_state[4] = post_activation_upper_bound

    relu_lower_form = relu3_deeppoly_low(
        pre_activation_symbol,
        pre_activation_slot[3],
        pre_activation_slot[4],
    )
    relu_upper_form = relu3_deeppoly_up(
        pre_activation_symbol,
        pre_activation_slot[3],
        pre_activation_slot[4],
    )
    activated_symbolic_lower_form = post_activation_symbol + ">=" + relu_lower_form
    activated_symbolic_upper_form = post_activation_symbol + "<=" + relu_upper_form
    post_activation_slot[5] = activated_symbolic_lower_form
    post_activation_slot[6] = activated_symbolic_upper_form
    post_activation_state[5] = activated_symbolic_lower_form
    post_activation_state[6] = activated_symbolic_upper_form

    post_activation_bounds = [post_activation_lower_bound, post_activation_upper_bound]
    return pre_activation_state, post_activation_state, pre_activation_bounds, post_activation_bounds


def compute_output_layer_neuron_no_abstraction_generic(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    layer_sizes,
    extended_variable_bounds,
    _unused_activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    extract_values2_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    neuron_output = longe_np(
        layer_weights[neuron_index],
        input_layer_symbols,
        network_biases[layer_index][neuron_index][0],
    )
    pre_activation_slot = layerabs_state[layer_index + 1][neuron_index][0]
    pre_activation_symbol = str(output_layer_symbols[neuron_index])
    pre_activation_state = _initialize_symbolic_state(
        pre_activation_slot,
        pre_activation_symbol,
        neuron_output,
    )

    constraints = extract_values2_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
    bound_result = optimize_bounds_fn(
        constraints,
        pre_activation_symbol,
        extended_variable_bounds,
    )
    pre_activation_bounds = _store_interval_bounds(
        pre_activation_slot,
        pre_activation_state,
        bound_result[0],
        bound_result[1],
    )
    return pre_activation_state, pre_activation_bounds


def compute_property_layer_neuron_no_abstraction_generic(
    layer_index,
    neuron_index,
    property_layer_weights,
    property_layer_biases,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    layer_sizes,
    extended_variable_bounds,
    output_size,
    _unused_pre_activation_mip_bounds,
    extract_values3_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    neuron_output = longe_np(
        property_layer_weights[neuron_index],
        output_layer_symbols,
        property_layer_biases[neuron_index][0],
    )
    pre_activation_slot = layerabs_state[layer_index + 1][neuron_index][0]
    pre_activation_symbol = str(property_layer_symbols[neuron_index])
    pre_activation_state = _initialize_symbolic_state(
        pre_activation_slot,
        pre_activation_symbol,
        neuron_output,
    )

    constraints = extract_values3_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
    bound_result = optimize_bounds_fn(
        constraints,
        pre_activation_symbol,
        extended_variable_bounds,
    )
    pre_activation_bounds = _store_interval_bounds(
        pre_activation_slot,
        pre_activation_state,
        bound_result[0],
        bound_result[1],
    )

    property_layer_interval_entry = [None] * (output_size - 1)
    property_layer_interval_entry[neuron_index] = [
        pre_activation_state[3],
        pre_activation_state[4],
    ]
    return pre_activation_state, property_layer_interval_entry, pre_activation_bounds
