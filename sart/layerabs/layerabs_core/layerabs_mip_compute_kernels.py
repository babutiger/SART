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


def compute_first_hidden_layer_neuron_generic(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    _unused_post_activation_mip_bounds,
    mip_backtrack_depth,
    extract_values1_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    del mip_backtrack_depth
    neuron_output = longe_np(
        layer_weights[neuron_index],
        input_layer_symbols,
        network_biases[layer_index][neuron_index][0],
    )

    pre_activation_state = [None] * 9
    post_activation_state = [None] * 9
    pre_activation_bounds = []
    post_activation_bounds = []

    pre_activation_symbol = str(hidden_layer_symbols[layer_index][neuron_index])
    layerabs_state[layer_index + 1][neuron_index][0][0] = pre_activation_symbol
    layerabs_state[layer_index + 1][neuron_index][0][1] = neuron_output
    pre_activation_state[0] = pre_activation_symbol
    pre_activation_state[1] = neuron_output

    symbolic_equality = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][2] = symbolic_equality
    pre_activation_state[2] = symbolic_equality

    symbolic_lower_form = pre_activation_symbol + "==" + neuron_output
    symbolic_upper_form = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][5] = symbolic_lower_form
    layerabs_state[layer_index + 1][neuron_index][0][6] = symbolic_upper_form
    pre_activation_state[5] = symbolic_lower_form
    pre_activation_state[6] = symbolic_upper_form

    layerabs_state[layer_index + 1][neuron_index][0][7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
    layerabs_state[layer_index + 1][neuron_index][0][8] = deeppoly_bounds[layer_index * 2][neuron_index][1]
    pre_activation_state[7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
    pre_activation_state[8] = deeppoly_bounds[layer_index * 2][neuron_index][1]

    if (
        layerabs_state[layer_index + 1][neuron_index][0][7] >= 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] > 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    elif (
        layerabs_state[layer_index + 1][neuron_index][0][7] < 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] <= 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    else:
        constraints = extract_values1_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
        objective = pre_activation_symbol
        bound_result = optimize_bounds_fn(constraints, objective, variable_bounds)
        layerabs_state[layer_index + 1][neuron_index][0][3] = bound_result[0]
        layerabs_state[layer_index + 1][neuron_index][0][4] = bound_result[1]
        pre_activation_state[3] = bound_result[0]
        pre_activation_state[4] = bound_result[1]
        pre_activation_bounds.append(bound_result[0])
        pre_activation_bounds.append(bound_result[1])

    relu_output = relu3(
        pre_activation_symbol,
        layerabs_state[layer_index + 1][neuron_index][0][3],
        layerabs_state[layer_index + 1][neuron_index][0][4],
    )
    post_activation_lower_bound = 0
    post_activation_upper_bound = 0

    if layerabs_state[layer_index + 1][neuron_index][0][3] > 0:
        post_activation_lower_bound = layerabs_state[layer_index + 1][neuron_index][0][3]
        post_activation_upper_bound = layerabs_state[layer_index + 1][neuron_index][0][4]
    if layerabs_state[layer_index + 1][neuron_index][0][4] < 0:
        post_activation_lower_bound = 0
        post_activation_upper_bound = 0
    if (
        layerabs_state[layer_index + 1][neuron_index][0][3] <= 0
        and layerabs_state[layer_index + 1][neuron_index][0][4] >= 0
    ):
        post_activation_lower_bound = 0
        post_activation_upper_bound = layerabs_state[layer_index + 1][neuron_index][0][4]

    post_activation_symbol = str(activated_hidden_layer_symbols[layer_index][neuron_index])
    activated_symbolic_equality = post_activation_symbol + "==" + relu_output
    layerabs_state[layer_index + 1][neuron_index][1][0] = post_activation_symbol
    layerabs_state[layer_index + 1][neuron_index][1][1] = relu_output
    layerabs_state[layer_index + 1][neuron_index][1][2] = activated_symbolic_equality
    layerabs_state[layer_index + 1][neuron_index][1][3] = post_activation_lower_bound
    layerabs_state[layer_index + 1][neuron_index][1][4] = post_activation_upper_bound
    post_activation_state[0] = post_activation_symbol
    post_activation_state[1] = relu_output
    post_activation_state[2] = activated_symbolic_equality
    post_activation_state[3] = post_activation_lower_bound
    post_activation_state[4] = post_activation_upper_bound

    relu_lower_form = relu3_deeppoly_low(
        pre_activation_symbol,
        layerabs_state[layer_index + 1][neuron_index][0][3],
        layerabs_state[layer_index + 1][neuron_index][0][4],
    )
    relu_upper_form = relu3_deeppoly_up(
        pre_activation_symbol,
        layerabs_state[layer_index + 1][neuron_index][0][3],
        layerabs_state[layer_index + 1][neuron_index][0][4],
    )
    activated_symbolic_lower_form = post_activation_symbol + ">=" + relu_lower_form
    activated_symbolic_upper_form = post_activation_symbol + "<=" + relu_upper_form
    layerabs_state[layer_index + 1][neuron_index][1][5] = activated_symbolic_lower_form
    layerabs_state[layer_index + 1][neuron_index][1][6] = activated_symbolic_upper_form
    post_activation_state[5] = activated_symbolic_lower_form
    post_activation_state[6] = activated_symbolic_upper_form

    layerabs_state[layer_index + 1][neuron_index][1][7] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][0]
    layerabs_state[layer_index + 1][neuron_index][1][8] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][1]
    post_activation_state[7] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][0]
    post_activation_state[8] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][1]
    post_activation_bounds.append(post_activation_lower_bound)
    post_activation_bounds.append(post_activation_upper_bound)
    return pre_activation_state, post_activation_state, pre_activation_bounds, post_activation_bounds


def compute_hidden_layer_neuron_generic(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    _unused_post_activation_mip_bounds,
    mip_backtrack_depth,
    extract_values1_fn,
    extract_values2_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    neuron_output = longe_np(
        layer_weights[neuron_index],
        input_layer_symbols,
        network_biases[layer_index][neuron_index][0],
    )

    pre_activation_state = [None] * 9
    post_activation_state = [None] * 9
    pre_activation_bounds = []
    post_activation_bounds = []

    pre_activation_symbol = str(hidden_layer_symbols[layer_index][neuron_index])
    symbolic_equality = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][0] = pre_activation_symbol
    layerabs_state[layer_index + 1][neuron_index][0][1] = neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][2] = symbolic_equality
    pre_activation_state[0] = pre_activation_symbol
    pre_activation_state[1] = neuron_output
    pre_activation_state[2] = symbolic_equality

    symbolic_lower_form = pre_activation_symbol + "==" + neuron_output
    symbolic_upper_form = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][5] = symbolic_lower_form
    layerabs_state[layer_index + 1][neuron_index][0][6] = symbolic_upper_form
    pre_activation_state[5] = symbolic_lower_form
    pre_activation_state[6] = symbolic_upper_form

    layerabs_state[layer_index + 1][neuron_index][0][7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
    layerabs_state[layer_index + 1][neuron_index][0][8] = deeppoly_bounds[layer_index * 2][neuron_index][1]
    pre_activation_state[7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
    pre_activation_state[8] = deeppoly_bounds[layer_index * 2][neuron_index][1]

    if (
        layerabs_state[layer_index + 1][neuron_index][0][7] >= 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] > 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    elif (
        layerabs_state[layer_index + 1][neuron_index][0][7] < 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] <= 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    else:
        if mip_backtrack_depth >= 3:
            if layer_index in range(2, mip_backtrack_depth + 1):
                constraints = extract_values1_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
            else:
                constraints = extract_values2_fn(
                    layerabs_state,
                    layer_index,
                    mip_backtrack_depth,
                    neuron_index,
                    layer_sizes,
                )
        else:
            if layer_index == 2:
                constraints = extract_values1_fn(layerabs_state, layer_index, 0, neuron_index, layer_sizes)
            else:
                constraints = extract_values2_fn(
                    layerabs_state,
                    layer_index,
                    mip_backtrack_depth,
                    neuron_index,
                    layer_sizes,
                )
        bound_result = optimize_bounds_fn(
            constraints,
            pre_activation_symbol,
            extended_variable_bounds,
        )
        layerabs_state[layer_index + 1][neuron_index][0][3] = bound_result[0]
        layerabs_state[layer_index + 1][neuron_index][0][4] = bound_result[1]
        pre_activation_state[3] = bound_result[0]
        pre_activation_state[4] = bound_result[1]
        pre_activation_bounds.append(bound_result[0])
        pre_activation_bounds.append(bound_result[1])

    relu_output = relu3(
        pre_activation_symbol,
        layerabs_state[layer_index + 1][neuron_index][0][3],
        layerabs_state[layer_index + 1][neuron_index][0][4],
    )
    post_activation_lower_bound = 0
    post_activation_upper_bound = 0

    if layerabs_state[layer_index + 1][neuron_index][0][3] > 0:
        post_activation_lower_bound = layerabs_state[layer_index + 1][neuron_index][0][3]
        post_activation_upper_bound = layerabs_state[layer_index + 1][neuron_index][0][4]
    if layerabs_state[layer_index + 1][neuron_index][0][4] < 0:
        post_activation_lower_bound = 0
        post_activation_upper_bound = 0
    if (
        layerabs_state[layer_index + 1][neuron_index][0][3] <= 0
        and layerabs_state[layer_index + 1][neuron_index][0][4] >= 0
    ):
        post_activation_lower_bound = 0
        post_activation_upper_bound = layerabs_state[layer_index + 1][neuron_index][0][4]

    post_activation_symbol = str(activated_hidden_layer_symbols[layer_index][neuron_index])
    activated_symbolic_equality = post_activation_symbol + "==" + relu_output
    layerabs_state[layer_index + 1][neuron_index][1][0] = post_activation_symbol
    layerabs_state[layer_index + 1][neuron_index][1][1] = relu_output
    layerabs_state[layer_index + 1][neuron_index][1][2] = activated_symbolic_equality
    layerabs_state[layer_index + 1][neuron_index][1][3] = post_activation_lower_bound
    layerabs_state[layer_index + 1][neuron_index][1][4] = post_activation_upper_bound
    post_activation_state[0] = post_activation_symbol
    post_activation_state[1] = relu_output
    post_activation_state[2] = activated_symbolic_equality
    post_activation_state[3] = post_activation_lower_bound
    post_activation_state[4] = post_activation_upper_bound

    relu_lower_form = relu3_deeppoly_low(
        pre_activation_symbol,
        layerabs_state[layer_index + 1][neuron_index][0][3],
        layerabs_state[layer_index + 1][neuron_index][0][4],
    )
    relu_upper_form = relu3_deeppoly_up(
        pre_activation_symbol,
        layerabs_state[layer_index + 1][neuron_index][0][3],
        layerabs_state[layer_index + 1][neuron_index][0][4],
    )
    activated_symbolic_lower_form = post_activation_symbol + ">=" + relu_lower_form
    activated_symbolic_upper_form = post_activation_symbol + "<=" + relu_upper_form
    layerabs_state[layer_index + 1][neuron_index][1][5] = activated_symbolic_lower_form
    layerabs_state[layer_index + 1][neuron_index][1][6] = activated_symbolic_upper_form
    post_activation_state[5] = activated_symbolic_lower_form
    post_activation_state[6] = activated_symbolic_upper_form

    layerabs_state[layer_index + 1][neuron_index][1][7] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][0]
    layerabs_state[layer_index + 1][neuron_index][1][8] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][1]
    post_activation_state[7] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][0]
    post_activation_state[8] = deeppoly_bounds[layer_index * 2 + 1][neuron_index][1]
    post_activation_bounds.append(post_activation_lower_bound)
    post_activation_bounds.append(post_activation_upper_bound)
    return pre_activation_state, post_activation_state, pre_activation_bounds, post_activation_bounds


def compute_output_layer_neuron_generic(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    mip_backtrack_depth,
    extract_values2_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    del activated_hidden_layer_symbols
    neuron_output = longe_np(
        layer_weights[neuron_index],
        input_layer_symbols,
        network_biases[layer_index][neuron_index][0],
    )
    pre_activation_state = [None] * 9
    pre_activation_bounds = []

    pre_activation_symbol = str(output_layer_symbols[neuron_index])
    symbolic_equality = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][0] = pre_activation_symbol
    layerabs_state[layer_index + 1][neuron_index][0][1] = neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][2] = symbolic_equality
    pre_activation_state[0] = pre_activation_symbol
    pre_activation_state[1] = neuron_output
    pre_activation_state[2] = symbolic_equality

    symbolic_lower_form = pre_activation_symbol + "==" + neuron_output
    symbolic_upper_form = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][5] = symbolic_lower_form
    layerabs_state[layer_index + 1][neuron_index][0][6] = symbolic_upper_form
    pre_activation_state[5] = symbolic_lower_form
    pre_activation_state[6] = symbolic_upper_form

    layerabs_state[layer_index + 1][neuron_index][0][7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
    layerabs_state[layer_index + 1][neuron_index][0][8] = deeppoly_bounds[layer_index * 2][neuron_index][1]
    pre_activation_state[7] = deeppoly_bounds[layer_index * 2][neuron_index][0]
    pre_activation_state[8] = deeppoly_bounds[layer_index * 2][neuron_index][1]

    if (
        layerabs_state[layer_index + 1][neuron_index][0][7] >= 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] > 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    elif (
        layerabs_state[layer_index + 1][neuron_index][0][7] < 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] <= 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    else:
        constraints = extract_values2_fn(
            layerabs_state,
            layer_index,
            mip_backtrack_depth,
            neuron_index,
            layer_sizes,
        )
        bound_result = optimize_bounds_fn(
            constraints,
            pre_activation_symbol,
            extended_variable_bounds,
        )
        layerabs_state[layer_index + 1][neuron_index][0][3] = bound_result[0]
        layerabs_state[layer_index + 1][neuron_index][0][4] = bound_result[1]
        pre_activation_state[3] = bound_result[0]
        pre_activation_state[4] = bound_result[1]
        pre_activation_bounds.append(bound_result[0])
        pre_activation_bounds.append(bound_result[1])

    return pre_activation_state, pre_activation_bounds


def compute_property_layer_neuron_generic(
    layer_index,
    neuron_index,
    property_layer_weights,
    property_layer_biases,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    output_size,
    _unused_pre_activation_mip_bounds,
    mip_backtrack_depth,
    extract_values3_fn,
    optimize_bounds_fn=optimize_with_bounds,
):
    neuron_output = longe_np(
        property_layer_weights[neuron_index],
        output_layer_symbols,
        property_layer_biases[neuron_index][0],
    )
    pre_activation_state = [None] * 9
    pre_activation_bounds = []

    pre_activation_symbol = str(property_layer_symbols[neuron_index])
    symbolic_equality = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][0] = pre_activation_symbol
    layerabs_state[layer_index + 1][neuron_index][0][1] = neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][2] = symbolic_equality
    pre_activation_state[0] = pre_activation_symbol
    pre_activation_state[1] = neuron_output
    pre_activation_state[2] = symbolic_equality

    symbolic_lower_form = pre_activation_symbol + "==" + neuron_output
    symbolic_upper_form = pre_activation_symbol + "==" + neuron_output
    layerabs_state[layer_index + 1][neuron_index][0][5] = symbolic_lower_form
    layerabs_state[layer_index + 1][neuron_index][0][6] = symbolic_upper_form
    pre_activation_state[5] = symbolic_lower_form
    pre_activation_state[6] = symbolic_upper_form

    layerabs_state[layer_index + 1][neuron_index][0][7] = deeppoly_bounds[layer_index * 2 - 1][neuron_index][0]
    layerabs_state[layer_index + 1][neuron_index][0][8] = deeppoly_bounds[layer_index * 2 - 1][neuron_index][1]
    pre_activation_state[7] = deeppoly_bounds[layer_index * 2 - 1][neuron_index][0]
    pre_activation_state[8] = deeppoly_bounds[layer_index * 2 - 1][neuron_index][1]

    if (
        layerabs_state[layer_index + 1][neuron_index][0][7] >= 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] > 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    elif (
        layerabs_state[layer_index + 1][neuron_index][0][7] < 0
        and layerabs_state[layer_index + 1][neuron_index][0][8] <= 0
    ):
        layerabs_state[layer_index + 1][neuron_index][0][3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        layerabs_state[layer_index + 1][neuron_index][0][4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_state[3] = layerabs_state[layer_index + 1][neuron_index][0][7]
        pre_activation_state[4] = layerabs_state[layer_index + 1][neuron_index][0][8]
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][7])
        pre_activation_bounds.append(layerabs_state[layer_index + 1][neuron_index][0][8])
    else:
        constraints = extract_values3_fn(
            layerabs_state,
            layer_index,
            mip_backtrack_depth,
            neuron_index,
            layer_sizes,
        )
        bound_result = optimize_bounds_fn(
            constraints,
            pre_activation_symbol,
            extended_variable_bounds,
        )
        layerabs_state[layer_index + 1][neuron_index][0][3] = bound_result[0]
        layerabs_state[layer_index + 1][neuron_index][0][4] = bound_result[1]
        pre_activation_state[3] = bound_result[0]
        pre_activation_state[4] = bound_result[1]
        pre_activation_bounds.append(bound_result[0])
        pre_activation_bounds.append(bound_result[1])

    property_layer_interval_entry = [None] * (output_size - 1)
    property_layer_interval_entry[neuron_index] = [
        pre_activation_state[3],
        pre_activation_state[4],
    ]
    return pre_activation_state, property_layer_interval_entry, pre_activation_bounds
