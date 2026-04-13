from __future__ import annotations

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)

__all__ = [
    "build_regular_output_layer_task_args",
    "build_regular_property_layer_task_args",
    "build_regular_first_hidden_layer_task_args",
    "build_regular_hidden_layer_task_args",
    "build_no_abstraction_output_layer_task_args",
    "build_no_abstraction_property_layer_task_args",
    "build_no_abstraction_first_hidden_layer_task_args",
    "build_no_abstraction_hidden_layer_task_args",
    "build_stats_milp_output_layer_task_args",
    "build_stats_milp_property_layer_task_args",
    "build_stats_milp_first_hidden_layer_task_args",
    "build_stats_milp_hidden_layer_task_args",
]


def _build_indexed_task_args(task_width, prefix_values, suffix_values):
    return [
        (*prefix_values, neuron_index, *suffix_values)
        for neuron_index in range(task_width)
    ]


def build_regular_output_layer_task_args(
    child_process_ids,
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            output_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            extended_variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_regular_property_layer_task_args(
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
    pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
            property_layer_weights,
            property_layer_biases,
            output_layer_symbols,
            property_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            extended_variable_bounds,
            output_size,
            pre_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_regular_first_hidden_layer_task_args(
    child_process_ids,
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    post_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            hidden_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
            post_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_regular_hidden_layer_task_args(
    child_process_ids,
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    post_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            hidden_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            extended_variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
            post_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_no_abstraction_output_layer_task_args(
    child_process_ids,
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            output_layer_symbols,
            layerabs_state,
            layer_sizes,
            extended_variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
        ),
    )


def build_no_abstraction_property_layer_task_args(
    child_process_ids,
    layer_index,
    layer_weights,
    property_layer_weights,
    property_layer_biases,
    output_layer_symbols,
    property_layer_symbols,
    layerabs_state,
    layer_sizes,
    extended_variable_bounds,
    output_size,
    pre_activation_mip_bounds,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
            property_layer_weights,
            property_layer_biases,
            output_layer_symbols,
            property_layer_symbols,
            layerabs_state,
            layer_sizes,
            extended_variable_bounds,
            output_size,
            pre_activation_mip_bounds,
        ),
    )


def build_no_abstraction_first_hidden_layer_task_args(
    child_process_ids,
    layer_index,
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
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
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
        ),
    )


def build_no_abstraction_hidden_layer_task_args(
    child_process_ids,
    layer_index,
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
):
    return _build_indexed_task_args(
        len(layer_weights),
        (child_process_ids, layer_index),
        (
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
        ),
    )


def build_stats_milp_output_layer_task_args(
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (layer_index,),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            output_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            extended_variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_stats_milp_property_layer_task_args(
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
    pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (layer_index,),
        (
            property_layer_weights,
            property_layer_biases,
            output_layer_symbols,
            property_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            extended_variable_bounds,
            output_size,
            pre_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_stats_milp_first_hidden_layer_task_args(
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    post_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (layer_index,),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            hidden_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
            post_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )


def build_stats_milp_hidden_layer_task_args(
    layer_index,
    layer_weights,
    input_layer_symbols,
    network_biases,
    hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    pre_activation_mip_bounds,
    post_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _build_indexed_task_args(
        len(layer_weights),
        (layer_index,),
        (
            layer_weights,
            input_layer_symbols,
            network_biases,
            hidden_layer_symbols,
            layerabs_state,
            deeppoly_bounds,
            layer_sizes,
            extended_variable_bounds,
            activated_hidden_layer_symbols,
            pre_activation_mip_bounds,
            post_activation_mip_bounds,
            mip_backtrack_depth,
        ),
    )

_LEGACY_ALIASES = {
    "build_regular_output_task_args": build_regular_output_layer_task_args,
    "build_regular_finaloutput_task_args": (
        build_regular_property_layer_task_args
    ),
    "build_regular_hidden_first_task_args": (
        build_regular_first_hidden_layer_task_args
    ),
    "build_regular_hidden_later_task_args": (
        build_regular_hidden_layer_task_args
    ),
    "build_no_abstraction_output_task_args": (
        build_no_abstraction_output_layer_task_args
    ),
    "build_no_abstraction_finaloutput_task_args": (
        build_no_abstraction_property_layer_task_args
    ),
    "build_no_abstraction_hidden_first_task_args": (
        build_no_abstraction_first_hidden_layer_task_args
    ),
    "build_no_abstraction_hidden_later_task_args": (
        build_no_abstraction_hidden_layer_task_args
    ),
    "build_stats_milp_output_task_args": build_stats_milp_output_layer_task_args,
    "build_stats_milp_finaloutput_task_args": (
        build_stats_milp_property_layer_task_args
    ),
    "build_stats_milp_hidden_first_task_args": (
        build_stats_milp_first_hidden_layer_task_args
    ),
    "build_stats_milp_hidden_later_task_args": (
        build_stats_milp_hidden_layer_task_args
    ),
    "build_ablation_output_task_args": (
        build_no_abstraction_output_layer_task_args
    ),
    "build_ablation_finaloutput_task_args": (
        build_no_abstraction_property_layer_task_args
    ),
    "build_ablation_hidden_first_task_args": (
        build_no_abstraction_first_hidden_layer_task_args
    ),
    "build_ablation_hidden_later_task_args": (
        build_no_abstraction_hidden_layer_task_args
    ),
}


def __getattr__(name):
    """Resolve legacy task-argument builder aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
