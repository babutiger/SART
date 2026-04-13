from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from sart.layerabs.default_profiles.LayerABS_incomplete_layerabs_mnist_10x80 import (
    run_incomplete_layerabs_verification,
    run_configured_incomplete_layerabs_variant,
    test_robustness_number_sym_merge_deeppoly_lp_sym_2,
)
from sart.layerabs.support.layerabs_controller_helpers import (
    run_variant_controller,
)


DEFAULT_VARIANT = "mnist_10x80"
run_configured_variant = run_configured_incomplete_layerabs_variant


def _add_k_layers_argument(parser):
    parser.add_argument(
        "--k-layers",
        type=int,
        default=2,
        help="Number of final layers encoded completely in Stage 2. Default: 2.",
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        default=None,
        help="Optional per-solve Gurobi TimeLimit in seconds for fairness studies.",
    )


def _extra_kwargs(args):
    return {
        "k_layers": args.k_layers,
        "mip_time_limit": args.mip_time_limit,
    }


def main(argv=None):
    run_variant_controller(
        description=(
            "Run the paper's Incomplete-LayerABS controller with Stage 1 + "
            "Stage 2 only."
        ),
        default_variant=DEFAULT_VARIANT,
        run_configured_variant=run_configured_incomplete_layerabs_variant,
        argv=argv,
        add_arguments=_add_k_layers_argument,
        extra_kwargs_builder=_extra_kwargs,
    )


if __name__ == "__main__":
    main()
