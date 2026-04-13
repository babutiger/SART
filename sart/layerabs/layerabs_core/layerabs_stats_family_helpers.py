"""Shared helper functions for `abstract_sart_stats` and `abstract_milp_stats`."""

from __future__ import annotations

from sart.layerabs.layerabs_core.layerabs_stage_helpers import (
    append_state_slot_bounds,
    collect_state_slot_symbols,
    extend_property_layer_variable_bounds_window,
    extend_recent_variable_bounds_window,
    propagate_first_hidden_layer_from_inputs as propagate_first_hidden_layer_stage,
)

__all__ = [
    "extend_property_stats_variable_bounds",
    "extend_recent_stats_variable_bounds",
    "propagate_first_hidden_stats_layer",
]


def _append_stats_slot_bounds(
    extended_variable_bounds,
    layerabs_state,
    history_layer_index,
    state_slot,
    history_layer_width,
):
    return append_state_slot_bounds(
        extended_variable_bounds,
        layerabs_state,
        history_layer_index + 1,
        state_slot,
        history_layer_width,
    )


def extend_recent_stats_variable_bounds(
    variable_bounds,
    layer_index,
    layerabs_state,
    network_weights,
    property_layer_weights,
):
    """Extend variable bounds with the recent three-stage stats backtracking window."""

    def resolve_history_width(history_layer_index):
        if history_layer_index != len(network_weights):
            return len(network_weights[history_layer_index])
        return len(property_layer_weights)

    return extend_recent_variable_bounds_window(
        variable_bounds,
        layer_index,
        resolve_history_width,
        lambda extended_variable_bounds, history_layer_index, state_slot, history_layer_width: (
            _append_stats_slot_bounds(
                extended_variable_bounds,
                layerabs_state,
                history_layer_index,
                state_slot,
                history_layer_width,
            )
        ),
    )


def extend_property_stats_variable_bounds(
    variable_bounds,
    layer_index,
    layerabs_state,
    network_weights,
    property_layer_weights,
):
    """Extend variable bounds for the stats property layer backtracking window."""

    def resolve_property_history_spec(history_layer_index):
        if (
            history_layer_index != len(network_weights)
            and history_layer_index != len(network_weights) - 1
        ):
            return 1, len(network_weights[history_layer_index])
        return 0, len(property_layer_weights) + 1

    return extend_property_layer_variable_bounds_window(
        variable_bounds,
        layer_index,
        resolve_property_history_spec,
        lambda extended_variable_bounds, history_layer_index, state_slot, history_layer_width: (
            _append_stats_slot_bounds(
                extended_variable_bounds,
                layerabs_state,
                history_layer_index,
                state_slot,
                history_layer_width,
            )
        ),
    )


def propagate_first_hidden_stats_layer(
    layer_index,
    layer_weights,
    network_biases,
    hidden_layer_symbols,
    activated_hidden_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    *,
    previous_layer_symbols=None,
    input_size=None,
):
    """Propagate the first hidden layer for stats families with optional input lookup."""
    if previous_layer_symbols is None:
        if input_size is None:
            raise ValueError(
                "input_size must be provided when previous_layer_symbols is omitted."
            )
        previous_layer_symbols = collect_state_slot_symbols(
            layerabs_state,
            layer_index,
            input_size,
            0,
        )

    return propagate_first_hidden_layer_stage(
        layer_index,
        layer_weights,
        network_biases,
        hidden_layer_symbols[layer_index],
        activated_hidden_layer_symbols[layer_index],
        layerabs_state,
        deeppoly_bounds,
        previous_layer_symbols,
    )
