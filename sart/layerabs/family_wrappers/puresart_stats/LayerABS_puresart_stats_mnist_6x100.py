from sart.layerabs.LayerABS_puresart_stats import (
    run_configured_puresart_stats_variant,
)


def run_puresart_stats_experiment(d=0.019):
    run_configured_puresart_stats_variant("mnist_6x100", d)


if __name__ == "__main__":
    run_puresart_stats_experiment(0.019)
