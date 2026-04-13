from sart.layerabs.LayerABS_incomplete_layerabs import (
    run_configured_incomplete_variant,
)


def test_robustness_number_sym_merge_deeppoly_lp_sym_2(d=0.004, k_layers=2):
    run_configured_incomplete_variant("cifar10_6x80", d=d, k_layers=k_layers)


if __name__ == "__main__":
    test_robustness_number_sym_merge_deeppoly_lp_sym_2()
