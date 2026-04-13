# Canonical PureSART stats baseline entrypoint.

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
    resolve_legacy_alias,
)
from sart.layerabs.layerabs_core.layerabs_logging_helpers import (
    redirect_stdout_to_timestamped_log,
)
from sart.layerabs.layerabs_core.layerabs_process_helpers import (
    get_managed_child_process_list,
)
from sart.layerabs.layerabs_core.layerabs_no_abstraction_propagation import (
    run_no_abstraction_forward_propagation,
)
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    terminate_managed_child_processes,
)
from sart.layerabs.layerabs_core.layerabs_verification_runners import (
    run_layerabs_verification,
)
from sart.layerabs.layerabs_variants.puresart_stats_variants import (
    get_variant_config,
)
from sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80 import (
    network,
)

style_time = redirect_stdout_to_timestamped_log(__file__)


def set_style_time(new_style_time):
    global style_time
    style_time = new_style_time


def _get_child_process_ids():
    return get_managed_child_process_list(__name__)


def forward_propagation_puresart_stats_baseline(
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
):
    return run_no_abstraction_forward_propagation(
        weights=weights,
        biases=biases,
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        final_weights=final_weights,
        final_biases=final_biases,
        hidden_layers=hidden_layers,
        hidden_layers_activate=hidden_layers_activate,
        output_layer=output_layer,
        final_output_layer=final_output_layer,
        four_dimensional_list=four_dimensional_list,
        input_layer=input_layer,
        bound_pair=bound_pair,
        child_process_ids=_get_child_process_ids(),
        emit_stats=True,
        print_nonhidden_layer_timing=True,
    )


def terminate_all_child_processes():
    terminate_managed_child_processes(_get_child_process_ids())


def run_puresart_stats_verification(
    d,
    variant_name="vnncomp_6x100",
):
    config = get_variant_config(variant_name)
    if config.recursion_limit is not None:
        sys.setrecursionlimit(config.recursion_limit)
    run_layerabs_verification(
        config=config,
        d=d,
        style_time=style_time,
        network_cls=network,
        forward_propagation_fn=forward_propagation_puresart_stats_baseline,
        terminate_child_processes=terminate_all_child_processes,
        average_divisor=config.reported_amount,
        reported_total=config.reported_amount,
        print_property_header=True,
    )


def run_configured_puresart_stats_variant(variant_name, d=None):
    config = get_variant_config(variant_name)
    if d is None:
        d = config.default_delta
    run_puresart_stats_verification(d, variant_name=variant_name)

forward_propagation_puresart_stats = forward_propagation_puresart_stats_baseline

_LEGACY_ALIASES = {
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        forward_propagation_puresart_stats_baseline
    ),
    "test_robustness_number_sym_merge_deeppoly_lp_sym_2_ablation_factor_NO_abstract": (
        run_puresart_stats_verification
    ),
    "run_configured_ablation_variant": run_configured_puresart_stats_variant,
}


def __getattr__(name):
    """Resolve legacy no-abstraction aliases for archived scripts."""
    return resolve_legacy_alias(__name__, name, _LEGACY_ALIASES)


if __name__ == "__main__":
    run_puresart_stats_verification(
        0.019,
        variant_name="vnncomp_6x100",
    )
