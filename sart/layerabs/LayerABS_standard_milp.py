from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from sart.layerabs.default_profiles.LayerABS_standard_milp_mnist_10x80 import (
    forward_propagation_standard_milp,
    run_configured_standard_milp_variant,
    run_standard_milp_verification,
    set_style_time,
)
from sart.layerabs.layerabs_core.layerabs_logging_helpers import (
    redirect_stdout_to_timestamped_log,
)
from sart.layerabs.support.layerabs_controller_helpers import (
    run_variant_controller,
)


DEFAULT_VARIANT = "mnist_10x80"
set_style_time(redirect_stdout_to_timestamped_log(__file__))
run_configured_variant = run_configured_standard_milp_variant

_LEGACY_ALIASES = {
    "forward_propagation_sym_deeppoly_lp_sym_lmipnum1_layermip_complete_ablation_factor": (
        forward_propagation_standard_milp
    ),
    "test_robustness_number_sym_merge_deeppoly_lp_sym_2_ablation_factor_NO_abstract": (
        run_standard_milp_verification
    ),
    "run_configured_ablation_variant": run_configured_standard_milp_variant,
}


def __getattr__(name):
    if name in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None):
    run_variant_controller(
        description=(
            "Run the standard-MILP baseline controller with any configured "
            "benchmark/network variant."
        ),
        default_variant=DEFAULT_VARIANT,
        run_configured_variant=run_configured_standard_milp_variant,
        argv=argv,
    )


if __name__ == "__main__":
    main()
