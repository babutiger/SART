"""Shared task helpers for the `abstract_milp_stats` family."""

import os

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)
from sart.layerabs.layerabs_core.layerabs_lmipnum1_helpers import (
    compute_first_hidden_layer_neuron as compute_first_hidden_layer_neuron_base,
    compute_hidden_layer_neuron as compute_hidden_layer_neuron_base,
    compute_output_layer_neuron as compute_output_layer_neuron_base,
    compute_property_layer_neuron as compute_property_layer_neuron_base,
)
from sart.layerabs.layerabs_core.layerabs_layermip_complete_helpers import (
    compute_first_hidden_layer_neuron_complete as compute_first_hidden_layer_neuron_complete_base,
    compute_hidden_layer_neuron_complete as compute_hidden_layer_neuron_complete_base,
    compute_output_layer_neuron_complete as compute_output_layer_neuron_complete_base,
    compute_property_layer_neuron_complete as compute_property_layer_neuron_complete_base,
)
from sart.layerabs.layerabs_core.layerabs_solver_helpers import (
    SignConsistencyTerminationCallback,
    add_abs_constraints,
    add_constraints_to_bound_models,
    build_bound_models,
    optimize_bound_models,
)
from sart.layerabs.layerabs_core.layerabs_process_helpers import (
    get_managed_child_process_list,
)

__all__ = [
    "ABSTRACT_MILP_STATS_COMPLETE_RUNNERS",
    "ABSTRACT_MILP_STATS_REFINEMENT_RUNNERS",
    "STATS_MILP_COMPLETE_RUNNERS",
    "STATS_MILP_REFINEMENT_RUNNERS",
    "get_child_process_ids",
    "compute_first_hidden_layer_neuron",
    "compute_first_hidden_layer_neuron_complete",
    "compute_hidden_layer_neuron",
    "compute_hidden_layer_neuron_complete",
    "compute_output_layer_neuron",
    "compute_output_layer_neuron_complete",
    "compute_property_layer_neuron",
    "compute_property_layer_neuron_complete",
    "optimize_with_bounds",
    "run_parallel_first_hidden_layer_task_complete_stats_milp",
    "run_parallel_first_hidden_layer_task_stats_milp",
    "run_parallel_hidden_layer_task_complete_stats_milp",
    "run_parallel_hidden_layer_task_stats_milp",
    "run_parallel_output_layer_task_complete_stats_milp",
    "run_parallel_output_layer_task_stats_milp",
    "run_parallel_property_layer_task_complete_stats_milp",
    "run_parallel_property_layer_task_stats_milp",
]


def get_child_process_ids():
    return get_managed_child_process_list(__name__)


def optimize_with_bounds(constraints, objective, variable_bounds):
    callback = SignConsistencyTerminationCallback()
    model_min, model_max, variables_min, variables_max = build_bound_models(
        variable_bounds
    )
    add_constraints_to_bound_models(
        constraints,
        model_min,
        model_max,
        variables_min,
        variables_max,
        abs_var_suffix="_Abs",
        manual_abs_constraint_fn=add_abs_constraints,
    )
    return optimize_bound_models(
        model_min,
        model_max,
        variables_min,
        variables_max,
        objective,
        callback=callback,
        should_terminate=lambda: callback.terminate_solution,
    )


def _run_stats_milp_compute(base_compute_fn, *args):
    return base_compute_fn(*args, optimize_bounds_fn=optimize_with_bounds)


def compute_first_hidden_layer_neuron(
    layer_index,
    neuron_index,
    layer_weights,
    previous_layer_symbols,
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
):
    return _run_stats_milp_compute(
        compute_first_hidden_layer_neuron_base,
        layer_index,
        neuron_index,
        layer_weights,
        previous_layer_symbols,
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
    )


def compute_hidden_layer_neuron(
    layer_index,
    neuron_index,
    layer_weights,
    previous_layer_symbols,
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
):
    return _run_stats_milp_compute(
        compute_hidden_layer_neuron_base,
        layer_index,
        neuron_index,
        layer_weights,
        previous_layer_symbols,
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
    )


def compute_output_layer_neuron(
    layer_index,
    neuron_index,
    layer_weights,
    previous_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _run_stats_milp_compute(
        compute_output_layer_neuron_base,
        layer_index,
        neuron_index,
        layer_weights,
        previous_layer_symbols,
        network_biases,
        output_layer_symbols,
        layerabs_state,
        deeppoly_bounds,
        layer_sizes,
        extended_variable_bounds,
        activated_hidden_layer_symbols,
        _unused_pre_activation_mip_bounds,
        mip_backtrack_depth,
    )


def compute_property_layer_neuron(
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
    property_layer_width,
    _unused_pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _run_stats_milp_compute(
        compute_property_layer_neuron_base,
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
        property_layer_width,
        _unused_pre_activation_mip_bounds,
        mip_backtrack_depth,
    )


def compute_first_hidden_layer_neuron_complete(
    layer_index,
    neuron_index,
    layer_weights,
    previous_layer_symbols,
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
):
    return _run_stats_milp_compute(
        compute_first_hidden_layer_neuron_complete_base,
        layer_index,
        neuron_index,
        layer_weights,
        previous_layer_symbols,
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
    )


def compute_hidden_layer_neuron_complete(
    layer_index,
    neuron_index,
    layer_weights,
    previous_layer_symbols,
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
):
    return _run_stats_milp_compute(
        compute_hidden_layer_neuron_complete_base,
        layer_index,
        neuron_index,
        layer_weights,
        previous_layer_symbols,
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
    )


def compute_output_layer_neuron_complete(
    layer_index,
    neuron_index,
    layer_weights,
    previous_layer_symbols,
    network_biases,
    output_layer_symbols,
    layerabs_state,
    deeppoly_bounds,
    layer_sizes,
    extended_variable_bounds,
    activated_hidden_layer_symbols,
    _unused_pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _run_stats_milp_compute(
        compute_output_layer_neuron_complete_base,
        layer_index,
        neuron_index,
        layer_weights,
        previous_layer_symbols,
        network_biases,
        output_layer_symbols,
        layerabs_state,
        deeppoly_bounds,
        layer_sizes,
        extended_variable_bounds,
        activated_hidden_layer_symbols,
        _unused_pre_activation_mip_bounds,
        mip_backtrack_depth,
    )


def compute_property_layer_neuron_complete(
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
    property_layer_width,
    _unused_pre_activation_mip_bounds,
    mip_backtrack_depth,
):
    return _run_stats_milp_compute(
        compute_property_layer_neuron_complete_base,
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
        property_layer_width,
        _unused_pre_activation_mip_bounds,
        mip_backtrack_depth,
    )


def _return_indexed_task_result(neuron_index, result):
    if result is None:
        raise ValueError(f"Received None for result with index {neuron_index}")
    return neuron_index, result


def _run_indexed_stats_milp_task(args, compute_fn):
    neuron_index = args[1]
    get_child_process_ids().append(os.getpid())
    result = compute_fn(*args)
    return _return_indexed_task_result(neuron_index, result)


def run_parallel_first_hidden_layer_task_stats_milp(args):
    return _run_indexed_stats_milp_task(args, compute_first_hidden_layer_neuron)


def run_parallel_hidden_layer_task_stats_milp(args):
    return _run_indexed_stats_milp_task(args, compute_hidden_layer_neuron)


def run_parallel_output_layer_task_stats_milp(args):
    return _run_indexed_stats_milp_task(args, compute_output_layer_neuron)


def run_parallel_property_layer_task_stats_milp(args):
    return _run_indexed_stats_milp_task(args, compute_property_layer_neuron)


def run_parallel_first_hidden_layer_task_complete_stats_milp(args):
    return _run_indexed_stats_milp_task(
        args,
        compute_first_hidden_layer_neuron_complete,
    )


def run_parallel_hidden_layer_task_complete_stats_milp(args):
    return _run_indexed_stats_milp_task(args, compute_hidden_layer_neuron_complete)


def run_parallel_output_layer_task_complete_stats_milp(args):
    return _run_indexed_stats_milp_task(args, compute_output_layer_neuron_complete)


def run_parallel_property_layer_task_complete_stats_milp(args):
    return _run_indexed_stats_milp_task(
        args,
        compute_property_layer_neuron_complete,
    )


ABSTRACT_MILP_STATS_REFINEMENT_RUNNERS = {
    "output": run_parallel_output_layer_task_stats_milp,
    "property": run_parallel_property_layer_task_stats_milp,
    "first_hidden": run_parallel_first_hidden_layer_task_stats_milp,
    "hidden": run_parallel_hidden_layer_task_stats_milp,
}

ABSTRACT_MILP_STATS_COMPLETE_RUNNERS = {
    "output": run_parallel_output_layer_task_complete_stats_milp,
    "property": run_parallel_property_layer_task_complete_stats_milp,
    "first_hidden": run_parallel_first_hidden_layer_task_complete_stats_milp,
    "hidden": run_parallel_hidden_layer_task_complete_stats_milp,
}


def __getattr__(name):
    if name == "child_process_ids":
        return get_child_process_ids()
    return resolve_legacy_alias(__name__, name, {})
STATS_MILP_REFINEMENT_RUNNERS = ABSTRACT_MILP_STATS_REFINEMENT_RUNNERS
STATS_MILP_COMPLETE_RUNNERS = ABSTRACT_MILP_STATS_COMPLETE_RUNNERS
