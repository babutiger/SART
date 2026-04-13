"""Shared propagation entrypoints for the abstraction-enabled LayerABS(SART) family."""

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)

from sart.layerabs.layerabs_core.layerabs_logging_helpers import (
    redirect_stdout_to_timestamped_log,
)
from sart.layerabs.layerabs_core.layerabs_process_helpers import (
    get_managed_child_process_list,
)
from sart.layerabs.layerabs_core.layerabs_parallel_task_helpers import (
    run_parallel_first_hidden_layer_task,
    run_parallel_first_hidden_layer_task_complete,
    run_parallel_hidden_layer_task,
    run_parallel_hidden_layer_task_complete,
    run_parallel_output_layer_task,
    run_parallel_output_layer_task_complete,
    run_parallel_property_layer_task,
    run_parallel_property_layer_task_complete,
)
from sart.layerabs.layerabs_core.layerabs_regular_family_propagation import (
    run_regular_propagation,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    terminate_managed_child_processes,
)

__all__ = [
    "run_refinement_propagation",
    "run_complete_propagation",
    "run_refinement_stage_propagation",
    "run_complete_stage_propagation",
    "terminate_family_child_processes",
    "style_time",
    "terminate_all_child_processes",
]

style_time = redirect_stdout_to_timestamped_log(__file__)


def _get_child_process_ids():
    return get_managed_child_process_list(__name__)


COMPLETE_REFINEMENT_RUNNERS = {
    "output": run_parallel_output_layer_task,
    "property": run_parallel_property_layer_task,
    "first_hidden": run_parallel_first_hidden_layer_task,
    "hidden": run_parallel_hidden_layer_task,
}

COMPLETE_COMPLETE_RUNNERS = {
    "output": run_parallel_output_layer_task_complete,
    "property": run_parallel_property_layer_task_complete,
    "first_hidden": run_parallel_first_hidden_layer_task_complete,
    "hidden": run_parallel_hidden_layer_task_complete,
}


def run_refinement_stage_propagation(
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
    save_deeppoly,
    l_mip_num,
):
    return run_regular_propagation(
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
        save_deeppoly,
        l_mip_num,
        stage_runners=COMPLETE_REFINEMENT_RUNNERS,
        child_process_ids=_get_child_process_ids(),
        cleanup_callback=terminate_all_child_processes,
    )


def run_complete_stage_propagation(
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
    save_deeppoly,
    l_mip_num,
):
    return run_regular_propagation(
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
        save_deeppoly,
        l_mip_num,
        stage_runners=COMPLETE_COMPLETE_RUNNERS,
        child_process_ids=_get_child_process_ids(),
        cleanup_callback=terminate_all_child_processes,
    )


def terminate_all_child_processes():
    terminate_managed_child_processes(_get_child_process_ids())


run_refinement_propagation = run_refinement_stage_propagation
run_complete_propagation = run_complete_stage_propagation
terminate_family_child_processes = terminate_all_child_processes


_LEGACY_ALIASES = {
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip": (
        run_refinement_stage_propagation
    ),
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip_complete": (
        run_complete_stage_propagation
    ),
}


def __getattr__(name):
    """Resolve legacy entrypoint aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
