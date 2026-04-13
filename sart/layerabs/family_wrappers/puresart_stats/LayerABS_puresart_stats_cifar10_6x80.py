from sart.layerabs.LayerABS_puresart_stats import (
    run_configured_puresart_stats_variant,
)


def run_puresart_stats_experiment(d=0.004):
    run_configured_puresart_stats_variant("cifar10_6x80", d)


if __name__ == "__main__":
    run_puresart_stats_experiment(0.004)
