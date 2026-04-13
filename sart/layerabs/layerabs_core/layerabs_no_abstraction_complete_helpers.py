from __future__ import annotations

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)
from sart.layerabs.layerabs_core.layerabs_no_abstraction_compute_kernels import (
    compute_first_hidden_layer_neuron_no_abstraction_generic,
    compute_hidden_layer_neuron_no_abstraction_generic,
    compute_output_layer_neuron_no_abstraction_generic,
    compute_property_layer_neuron_no_abstraction_generic,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    extract_values1_complete,
    extract_values2_complete,
    extract_values3_complete,
)
from sart.layerabs.layerabs_core.layerabs_solver_helpers import (
    optimize_with_bounds,
)

__all__ = [
    "compute_first_hidden_layer_neuron_complete_no_abstraction",
    "compute_hidden_layer_neuron_complete_no_abstraction",
    "compute_output_layer_neuron_complete_no_abstraction",
    "compute_property_layer_neuron_complete_no_abstraction",
]


def compute_first_hidden_layer_neuron_complete_no_abstraction(
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
    pre_activation_mip_bounds,
    post_activation_mip_bounds,
    optimize_bounds_fn=optimize_with_bounds,
):
    return compute_first_hidden_layer_neuron_no_abstraction_generic(
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
        pre_activation_mip_bounds,
        post_activation_mip_bounds,
        extract_values1_fn=extract_values1_complete,
        optimize_bounds_fn=optimize_bounds_fn,
    )


def compute_hidden_layer_neuron_complete_no_abstraction(
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
    pre_activation_mip_bounds,
    post_activation_mip_bounds,
    optimize_bounds_fn=optimize_with_bounds,
):
    return compute_hidden_layer_neuron_no_abstraction_generic(
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
        pre_activation_mip_bounds,
        post_activation_mip_bounds,
        extract_values1_fn=extract_values1_complete,
        extract_values2_fn=extract_values2_complete,
        optimize_bounds_fn=optimize_bounds_fn,
    )


def compute_output_layer_neuron_complete_no_abstraction(
    layer_index,
    neuron_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    optimize_bounds_fn=optimize_with_bounds,
):
    return compute_output_layer_neuron_no_abstraction_generic(
        layer_index,
        neuron_index,
        layer_weights,
        input_layer_symbols,
        network_biases,
        output_layer_symbols,
        layerabs_state,
        layer_sizes,
        extended_variable_bounds,
        activated_hidden_layer_symbols,
        pre_activation_mip_bounds,
        extract_values2_fn=extract_values2_complete,
        optimize_bounds_fn=optimize_bounds_fn,
    )


def compute_property_layer_neuron_complete_no_abstraction(
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
    pre_activation_mip_bounds,
    optimize_bounds_fn=optimize_with_bounds,
):
    return compute_property_layer_neuron_no_abstraction_generic(
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
        pre_activation_mip_bounds,
        extract_values3_fn=extract_values3_complete,
        optimize_bounds_fn=optimize_bounds_fn,
    )


_LEGACY_ALIASES = {
    "jisuan_function1_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        compute_first_hidden_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function2_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        compute_hidden_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function_output_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        compute_output_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function_finaloutput_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        compute_property_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function1_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        compute_first_hidden_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function2_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        compute_hidden_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function_output_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        compute_output_layer_neuron_complete_no_abstraction
    ),
    "jisuan_function_finaloutput_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        compute_property_layer_neuron_complete_no_abstraction
    ),
}


def __getattr__(name):
    """Resolve legacy helper aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
