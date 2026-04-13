from __future__ import annotations

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)
from sart.layerabs.layerabs_core.layerabs_mip_compute_kernels import (
    compute_first_hidden_layer_neuron_generic,
    compute_hidden_layer_neuron_generic,
    compute_output_layer_neuron_generic,
    compute_property_layer_neuron_generic,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    extract_values1,
    extract_values2,
    extract_values3,
)
from sart.layerabs.layerabs_core.layerabs_solver_helpers import (
    optimize_with_bounds,
)

__all__ = [
    "compute_first_hidden_layer_neuron",
    "compute_hidden_layer_neuron",
    "compute_output_layer_neuron",
    "compute_property_layer_neuron",
]


def compute_first_hidden_layer_neuron(*args, optimize_bounds_fn=optimize_with_bounds):
    return compute_first_hidden_layer_neuron_generic(
        *args,
        extract_values1_fn=extract_values1,
        optimize_bounds_fn=optimize_bounds_fn,
    )


def compute_hidden_layer_neuron(*args, optimize_bounds_fn=optimize_with_bounds):
    return compute_hidden_layer_neuron_generic(
        *args,
        extract_values1_fn=extract_values1,
        extract_values2_fn=extract_values2,
        optimize_bounds_fn=optimize_bounds_fn,
    )


def compute_output_layer_neuron(*args, optimize_bounds_fn=optimize_with_bounds):
    return compute_output_layer_neuron_generic(
        *args,
        extract_values2_fn=extract_values2,
        optimize_bounds_fn=optimize_bounds_fn,
    )


def compute_property_layer_neuron(*args, optimize_bounds_fn=optimize_with_bounds):
    return compute_property_layer_neuron_generic(
        *args,
        extract_values3_fn=extract_values3,
        optimize_bounds_fn=optimize_bounds_fn,
    )


_LEGACY_ALIASES = {
    "jisuan_function1_lp_sym_lmipnum1": compute_first_hidden_layer_neuron,
    "jisuan_function2_lp_sym_lmipnum1": compute_hidden_layer_neuron,
    "jisuan_function_output_lp_sym_lmipnum1": compute_output_layer_neuron,
    "jisuan_function_finaloutput_lp_sym_lmipnum1": (
        compute_property_layer_neuron
    ),
}


def __getattr__(name):
    """Resolve legacy helper aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)
