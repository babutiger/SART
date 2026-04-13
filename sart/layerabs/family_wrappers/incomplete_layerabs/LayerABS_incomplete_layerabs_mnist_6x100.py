from sart.layerabs.LayerABS_incomplete_layerabs import (
    run_configured_incomplete_variant,
)


def test_robustness_number_sym_merge_deeppoly_lp_sym_2(d=0.019, k_layers=2):
    run_configured_incomplete_variant("mnist_6x100", d=d, k_layers=k_layers)


if __name__ == "__main__":
    test_robustness_number_sym_merge_deeppoly_lp_sym_2()
