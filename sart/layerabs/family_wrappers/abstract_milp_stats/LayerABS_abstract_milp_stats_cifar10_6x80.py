from sart.layerabs.LayerABS_abstract_milp_stats import (
    run_configured_abstract_milp_stats_variant,
)


def test_robustness_number_sym_merge_deeppoly_lp_sym_2(d=0.004):
    run_configured_abstract_milp_stats_variant("cifar10_6x80", d)


if __name__ == "__main__":
    test_robustness_number_sym_merge_deeppoly_lp_sym_2(0.004)
