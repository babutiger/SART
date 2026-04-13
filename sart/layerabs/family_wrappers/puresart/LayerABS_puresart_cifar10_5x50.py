from sart.layerabs.LayerABS_puresart import (
    run_configured_puresart_variant,
)


def run_puresart_experiment(d=0.003):
    run_configured_puresart_variant("cifar10_5x50", d)


if __name__ == "__main__":
    run_puresart_experiment(0.003)
