"""Shared propagation entrypoints for the `complete_timelimit` family."""

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)

from sart.layerabs.layerabs_core.layerabs_parallel_task_helpers import (
    run_parallel_first_hidden_layer_task,
    run_parallel_hidden_layer_task,
    run_parallel_output_layer_task,
    run_parallel_property_layer_task,
)
from sart.layerabs.layerabs_core.layerabs_logging_helpers import (
    redirect_stdout_to_timestamped_log,
)
from sart.layerabs.layerabs_core.layerabs_process_helpers import (
    get_managed_child_process_list,
)
from sart.layerabs.layerabs_core.layerabs_regular_family_propagation import (
    run_regular_propagation,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    terminate_managed_child_processes,
)

__all__ = [
    "run_timelimit_propagation",
    "run_timelimit_stage_propagation",
    "terminate_family_child_processes",
    "style_time",
    "terminate_all_child_processes",
]

style_time = redirect_stdout_to_timestamped_log(__file__)


def _get_child_process_ids():
    return get_managed_child_process_list(__name__)


TIMELIMIT_STAGE_RUNNERS = {
    "output": run_parallel_output_layer_task,
    "property": run_parallel_property_layer_task,
    "first_hidden": run_parallel_first_hidden_layer_task,
    "hidden": run_parallel_hidden_layer_task,
}


def run_timelimit_stage_propagation(
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
        stage_runners=TIMELIMIT_STAGE_RUNNERS,
        child_process_ids=_get_child_process_ids(),
    )


def terminate_all_child_processes():
    terminate_managed_child_processes(_get_child_process_ids())


run_timelimit_propagation = run_timelimit_stage_propagation
terminate_family_child_processes = terminate_all_child_processes


_LEGACY_ALIASES = {
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip": (
        run_timelimit_stage_propagation
    ),
}


def __getattr__(name):
    """Resolve legacy entrypoint aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
