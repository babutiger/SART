from sart.layerabs.LayerABS_standard_milp_stats import (
    run_configured_standard_milp_stats_variant,
)


def run_standard_milp_stats_experiment(d=0.019):
    run_configured_standard_milp_stats_variant("mnist_6x100", d)


if __name__ == "__main__":
    run_standard_milp_stats_experiment(0.019)
