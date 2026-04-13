"""Shared ProcessPool task wrappers for the LayerABS propagation families."""

from __future__ import annotations

import os

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)
from sart.layerabs.layerabs_core.layerabs_no_abstraction_complete_helpers import (
    compute_first_hidden_layer_neuron_complete_no_abstraction,
    compute_hidden_layer_neuron_complete_no_abstraction,
    compute_output_layer_neuron_complete_no_abstraction,
    compute_property_layer_neuron_complete_no_abstraction,
)
from sart.layerabs.layerabs_core.layerabs_layermip_complete_helpers import (
    compute_first_hidden_layer_neuron_complete,
    compute_hidden_layer_neuron_complete,
    compute_output_layer_neuron_complete,
    compute_property_layer_neuron_complete,
)
from sart.layerabs.layerabs_core.layerabs_lmipnum1_helpers import (
    compute_first_hidden_layer_neuron,
    compute_hidden_layer_neuron,
    compute_output_layer_neuron,
    compute_property_layer_neuron,
)

__all__ = [
    "run_parallel_first_hidden_layer_task",
    "run_parallel_hidden_layer_task",
    "run_parallel_output_layer_task",
    "run_parallel_property_layer_task",
    "run_parallel_first_hidden_layer_task_complete",
    "run_parallel_hidden_layer_task_complete",
    "run_parallel_output_layer_task_complete",
    "run_parallel_property_layer_task_complete",
    "run_parallel_first_hidden_layer_task_complete_no_abstraction",
    "run_parallel_hidden_layer_task_complete_no_abstraction",
    "run_parallel_output_layer_task_complete_no_abstraction",
    "run_parallel_property_layer_task_complete_no_abstraction",
]


def _register_child_process(child_process_ids):
    child_process_ids.append(os.getpid())


def _return_indexed_result(neuron_index, result):
    if result is None:
        raise ValueError(f"Received None for result with index {neuron_index}")
    return neuron_index, result


def _run_indexed_parallel_task(args, compute_fn):
    child_process_ids = args[0]
    neuron_index = args[2]
    _register_child_process(child_process_ids)
    result = compute_fn(*args[1:])
    return _return_indexed_result(neuron_index, result)


def run_parallel_first_hidden_layer_task(args):
    return _run_indexed_parallel_task(args, compute_first_hidden_layer_neuron)


def run_parallel_hidden_layer_task(args):
    return _run_indexed_parallel_task(args, compute_hidden_layer_neuron)


def run_parallel_output_layer_task(args):
    return _run_indexed_parallel_task(args, compute_output_layer_neuron)


def run_parallel_property_layer_task(args):
    return _run_indexed_parallel_task(args, compute_property_layer_neuron)


def run_parallel_first_hidden_layer_task_complete(args):
    return _run_indexed_parallel_task(args, compute_first_hidden_layer_neuron_complete)


def run_parallel_hidden_layer_task_complete(args):
    return _run_indexed_parallel_task(args, compute_hidden_layer_neuron_complete)


def run_parallel_output_layer_task_complete(args):
    return _run_indexed_parallel_task(args, compute_output_layer_neuron_complete)


def run_parallel_property_layer_task_complete(args):
    return _run_indexed_parallel_task(args, compute_property_layer_neuron_complete)


def run_parallel_first_hidden_layer_task_complete_no_abstraction(args):
    return _run_indexed_parallel_task(
        args,
        compute_first_hidden_layer_neuron_complete_no_abstraction,
    )


def run_parallel_hidden_layer_task_complete_no_abstraction(args):
    return _run_indexed_parallel_task(
        args,
        compute_hidden_layer_neuron_complete_no_abstraction,
    )


def run_parallel_output_layer_task_complete_no_abstraction(args):
    return _run_indexed_parallel_task(
        args,
        compute_output_layer_neuron_complete_no_abstraction,
    )


def run_parallel_property_layer_task_complete_no_abstraction(args):
    return _run_indexed_parallel_task(
        args,
        compute_property_layer_neuron_complete_no_abstraction,
    )


_LEGACY_ALIASES = {
    "parallel_task1_lp_sym_lmipnum1": run_parallel_first_hidden_layer_task,
    "parallel_task2_lp_sym_lmipnum1": run_parallel_hidden_layer_task,
    "parallel_task_output_lp_sym_lmipnum1": run_parallel_output_layer_task,
    "parallel_task_finaloutput_lp_sym_lmipnum1": (
        run_parallel_property_layer_task
    ),
    "parallel_task1_lp_sym_lmipnum1_layermip_complete": (
        run_parallel_first_hidden_layer_task_complete
    ),
    "parallel_task2_lp_sym_lmipnum1_layermip_complete": (
        run_parallel_hidden_layer_task_complete
    ),
    "parallel_task_output_lp_sym_lmipnum1_layermip_complete": (
        run_parallel_output_layer_task_complete
    ),
    "parallel_task_finaloutput_lp_sym_lmipnum1_layermip_complete": (
        run_parallel_property_layer_task_complete
    ),
    "parallel_task1_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        run_parallel_first_hidden_layer_task_complete_no_abstraction
    ),
    "parallel_task2_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        run_parallel_hidden_layer_task_complete_no_abstraction
    ),
    "parallel_task_output_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        run_parallel_output_layer_task_complete_no_abstraction
    ),
    "parallel_task_finaloutput_lp_sym_lmipnum1_layermip_complete_no_abstraction": (
        run_parallel_property_layer_task_complete_no_abstraction
    ),
    "parallel_task1_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        run_parallel_first_hidden_layer_task_complete_no_abstraction
    ),
    "parallel_task2_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        run_parallel_hidden_layer_task_complete_no_abstraction
    ),
    "parallel_task_output_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        run_parallel_output_layer_task_complete_no_abstraction
    ),
    "parallel_task_finaloutput_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        run_parallel_property_layer_task_complete_no_abstraction
    ),
}


def __getattr__(name):
    """Resolve legacy task-wrapper aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
