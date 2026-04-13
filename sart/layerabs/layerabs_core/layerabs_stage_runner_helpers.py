"""Shared stage-runner skeletons for LayerABS family propagation modules."""

from __future__ import annotations

from sart.layerabs.layerabs_core.layerabs_stage_helpers import (
    collect_state_slot_symbols,
    execute_hidden_stage_tasks,
    execute_output_stage_tasks,
    execute_property_stage_tasks,
)

__all__ = [
    "run_first_hidden_stage_with_builder",
    "run_hidden_stage_with_builder",
    "run_output_stage_with_builder",
    "run_property_stage_with_builder",
]


def run_output_stage_with_builder(
    stage_runner,
    *,
    layer_index,
    previous_layer_width,
    layerabs_state,
    build_extended_variable_bounds,
    build_task_args,
    max_parallel_workers,
):
    previous_layer_symbols = collect_state_slot_symbols(
        layerabs_state,
        layer_index,
        previous_layer_width,
        1,
    )
    extended_variable_bounds = build_extended_variable_bounds()
    output_layer_task_args = build_task_args(
        previous_layer_symbols,
        extended_variable_bounds,
    )
    return execute_output_stage_tasks(
        stage_runner,
        output_layer_task_args,
        layerabs_state,
        layer_index + 1,
        max_parallel_workers,
    )


def run_property_stage_with_builder(
    stage_runner,
    *,
    layer_index,
    layerabs_state,
    property_layer_interval,
    build_extended_variable_bounds,
    build_task_args,
    max_parallel_workers,
):
    extended_variable_bounds = build_extended_variable_bounds()
    property_layer_task_args = build_task_args(extended_variable_bounds)
    return execute_property_stage_tasks(
        stage_runner,
        property_layer_task_args,
        layerabs_state,
        layer_index + 1,
        property_layer_interval,
        max_parallel_workers,
    )


def run_first_hidden_stage_with_builder(
    stage_runner,
    *,
    layer_index,
    previous_layer_width,
    layerabs_state,
    build_task_args,
    max_parallel_workers,
):
    previous_layer_symbols = collect_state_slot_symbols(
        layerabs_state,
        layer_index,
        previous_layer_width,
        1,
    )
    hidden_layer_task_args = build_task_args(previous_layer_symbols)
    return execute_hidden_stage_tasks(
        stage_runner,
        hidden_layer_task_args,
        layerabs_state,
        layer_index + 1,
        max_parallel_workers,
    )


def run_hidden_stage_with_builder(
    stage_runner,
    *,
    layer_index,
    previous_layer_width,
    layerabs_state,
    build_extended_variable_bounds,
    build_task_args,
    max_parallel_workers,
):
    previous_layer_symbols = collect_state_slot_symbols(
        layerabs_state,
        layer_index,
        previous_layer_width,
        1,
    )
    extended_variable_bounds = build_extended_variable_bounds()
    hidden_layer_task_args = build_task_args(
        previous_layer_symbols,
        extended_variable_bounds,
    )
    return execute_hidden_stage_tasks(
        stage_runner,
        hidden_layer_task_args,
        layerabs_state,
        layer_index + 1,
        max_parallel_workers,
    )
