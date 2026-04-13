from sart.layerabs.LayerABS_abstract_milp import run_configured_abstract_milp_variant


def test_robustness_number_sym_merge_deeppoly_lp_sym_2(d=0.018):
    run_configured_abstract_milp_variant("mnist_9x200", d)


if __name__ == "__main__":
    test_robustness_number_sym_merge_deeppoly_lp_sym_2(0.018)
