from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from sart.layerabs.layerabs_core.layerabs_incomplete_family_propagation import (
    run_incomplete_propagation,
    style_time,
    terminate_family_child_processes,
)
from sart.layerabs.layerabs_core.layerabs_solver_context import (
    override_solver_time_limit_seconds,
)
from sart.layerabs.layerabs_core.layerabs_verification_runners import (
    run_layerabs_incomplete_verification as run_incomplete_layerabs_runner,
)
from sart.layerabs.layerabs_variants.incomplete_layerabs_variants import (
    get_variant_config,
)


def run_incomplete_layerabs_verification(
    d,
    variant_name="mnist_10x80",
    k_layers=2,
    mip_time_limit=None,
):
    config = get_variant_config(variant_name)
    if config.recursion_limit is not None:
        sys.setrecursionlimit(config.recursion_limit)
    with override_solver_time_limit_seconds(mip_time_limit):
        run_incomplete_layerabs_runner(
            config=config,
            d=d,
            style_time=style_time,
            refinement_forward_fn=run_incomplete_propagation,
            terminate_child_processes=terminate_family_child_processes,
            k_layers=k_layers,
        )


def run_configured_incomplete_layerabs_variant(
    variant_name,
    d=None,
    k_layers=2,
    mip_time_limit=None,
):
    config = get_variant_config(variant_name)
    if d is None:
        d = config.default_delta
    run_incomplete_layerabs_verification(
        d,
        variant_name=variant_name,
        k_layers=k_layers,
        mip_time_limit=mip_time_limit,
    )


test_robustness_number_sym_merge_deeppoly_lp_sym_2 = (
    run_incomplete_layerabs_verification
)
run_configured_incomplete_variant = run_configured_incomplete_layerabs_variant


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper's Incomplete-LayerABS verifier "
            "(Stage 1 + Stage 2, without the full-complete fallback)."
        )
    )
    parser.add_argument("--variant", default="mnist_10x80")
    parser.add_argument("--delta", type=float, default=None)
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
    args = parser.parse_args(argv)
    run_configured_incomplete_layerabs_variant(
        args.variant,
        d=args.delta,
        k_layers=args.k_layers,
        mip_time_limit=args.mip_time_limit,
    )


if __name__ == "__main__":
    main()
