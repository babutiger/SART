from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from sart.layerabs.default_profiles.LayerABS_abstract_sart_timelimit_mnist_10x80 import (
    run_abstract_sart_timelimit_verification,
    run_configured_abstract_sart_timelimit_variant,
    test_robustness_number_sym_merge_deeppoly_lp_sym_2,
)
from sart.layerabs.support.layerabs_controller_helpers import (
    run_variant_controller,
)


DEFAULT_VARIANT = "mnist_10x80"
run_configured_variant = run_configured_abstract_sart_timelimit_variant


def main(argv=None):
    run_variant_controller(
        description=(
            "Run the abstraction-enabled LayerABS(SART) time-limit controller "
            "with any configured benchmark/network variant."
        ),
        default_variant=DEFAULT_VARIANT,
        run_configured_variant=run_configured_abstract_sart_timelimit_variant,
        argv=argv,
    )


if __name__ == "__main__":
    main()
