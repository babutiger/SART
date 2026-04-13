import unittest

from sart.layerabs.layerabs_variants.abstract_milp_stats_variants import (
    get_variant_config as get_abstract_milp_stats_variant,
)
from sart.layerabs.layerabs_variants.abstract_sart_stats_variants import (
    get_variant_config as get_abstract_sart_stats_variant,
)
from sart.layerabs.layerabs_variants.abstract_sart_timelimit_variants import (
    get_variant_config as get_abstract_sart_timelimit_variant,
)
from sart.layerabs.layerabs_variants.puresart_stats_variants import (
    get_variant_config as get_puresart_stats_variant,
)
from sart.layerabs.layerabs_variants.standard_milp_stats_variants import (
    get_variant_config as get_standard_milp_stats_variant,
)


class LayerABSVariantAliasTests(unittest.TestCase):
    def test_abstract_sart_stats_legacy_alias(self):
        config = get_abstract_sart_stats_variant("mnist_6x100_num_time_count")
        self.assertEqual(config.name, "mnist_6x100")

    def test_abstract_milp_stats_legacy_alias(self):
        config = get_abstract_milp_stats_variant("vnncomp_6x100_num_time_count")
        self.assertEqual(config.name, "vnncomp_6x100")

    def test_puresart_stats_legacy_alias(self):
        config = get_puresart_stats_variant("cifar10_6x80_num_time_count")
        self.assertEqual(config.name, "cifar10_6x80")

    def test_standard_milp_stats_legacy_alias(self):
        config = get_standard_milp_stats_variant("mnist_6x100_num_time_count")
        self.assertEqual(config.name, "mnist_6x100")

    def test_abstract_sart_timelimit_legacy_alias(self):
        config = get_abstract_sart_timelimit_variant("mnist_10x80_timelimit")
        self.assertEqual(config.name, "mnist_10x80")


if __name__ == "__main__":
    unittest.main()
