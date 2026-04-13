from sart.layerabs.LayerABS_standard_milp import (
    run_configured_standard_milp_variant,
)


def run_standard_milp_experiment(d=0.018):
    run_configured_standard_milp_variant("mnist_9x200", d)


if __name__ == "__main__":
    run_standard_milp_experiment(0.018)
