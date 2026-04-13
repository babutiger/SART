from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from sart.layerabs.layerabs_core.layerabs_abstract_milp_stats_family_propagation import (
    run_complete_propagation as run_abstract_milp_stats_complete_propagation,
    run_refinement_propagation as run_abstract_milp_stats_refinement_propagation,
    style_time,
    terminate_family_child_processes as terminate_abstract_milp_stats_child_processes,
)
from sart.layerabs.layerabs_core.layerabs_verification_runners import (
    run_layerabs_refinement_verification,
)
from sart.layerabs.layerabs_variants.abstract_milp_stats_variants import (
    get_variant_config,
)

def run_abstract_milp_stats_verification(d, variant_name="mnist_6x100"):
    config = get_variant_config(variant_name)
    run_layerabs_refinement_verification(
        config=config,
        d=d,
        style_time=style_time,
        refinement_forward_fn=run_abstract_milp_stats_refinement_propagation,
        complete_forward_fn=run_abstract_milp_stats_complete_propagation,
        terminate_child_processes=terminate_abstract_milp_stats_child_processes,
    )


def run_configured_abstract_milp_stats_variant(variant_name, d=None):
    config = get_variant_config(variant_name)
    if d is None:
        d = config.default_delta
    run_abstract_milp_stats_verification(d, variant_name=variant_name)


run_complete_stats_milp_verification = run_abstract_milp_stats_verification
run_configured_complete_stats_milp_variant = run_configured_abstract_milp_stats_variant
run_configured_milp_variant = run_configured_abstract_milp_stats_variant
test_robustness_number_sym_merge_deeppoly_lp_sym_2 = run_abstract_milp_stats_verification


if __name__ == "__main__":
    run_abstract_milp_stats_verification(0.019, variant_name="mnist_6x100")
