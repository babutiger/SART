from __future__ import annotations

import unittest
from pathlib import Path

from experiment_catalog import build_layerabs_catalog, build_layerabs_families


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentCatalogTests(unittest.TestCase):
    def test_layerabs_catalog_exposes_paper_role_for_key_entrypoints(self):
        catalog = build_layerabs_catalog(REPO_ROOT)
        by_path = {
            exp.path.relative_to(REPO_ROOT).as_posix(): exp
            for exp in catalog
        }

        self.assertEqual(
            by_path[
                "sart/layerabs/default_profiles/LayerABS_abstract_milp_mnist_5x50.py"
            ].paper_role,
            "layerabs_with_abstraction_milp",
        )
        self.assertEqual(
            by_path[
                "sart/layerabs/default_profiles/LayerABS_incomplete_layerabs_mnist_10x80.py"
            ].paper_role,
            "incomplete_layerabs",
        )

    def test_layerabs_families_expose_canonical_paper_role(self):
        catalog = build_layerabs_catalog(REPO_ROOT)
        families = {
            family.family: family
            for family in build_layerabs_families(catalog)
        }

        self.assertEqual(
            families["abstract_sart"].paper_role,
            "layerabs_with_abstraction_sart",
        )
        self.assertEqual(
            families["puresart"].paper_role,
            "puresart_no_abstraction",
        )
        self.assertEqual(
            families["incomplete_layerabs"].paper_role,
            "incomplete_layerabs",
        )
        self.assertEqual(
            families["abstract_sart"].controller_path.relative_to(REPO_ROOT).as_posix(),
            "sart/layerabs/LayerABS_abstract_sart.py",
        )
        self.assertEqual(
            families["incomplete_layerabs"].controller_path.relative_to(REPO_ROOT).as_posix(),
            "sart/layerabs/LayerABS_incomplete_layerabs.py",
        )
        self.assertNotIn("vnncomp_eran", families)
        self.assertNotIn("vnncomp_eran_milp", families)

    def test_model_neutral_family_controllers_are_not_catalogued_as_experiments(self):
        catalog = build_layerabs_catalog(REPO_ROOT)
        catalog_paths = {
            exp.path.relative_to(REPO_ROOT).as_posix()
            for exp in catalog
        }

        self.assertNotIn(
            "sart/layerabs/LayerABS_abstract_sart.py",
            catalog_paths,
        )
        self.assertNotIn(
            "sart/layerabs/LayerABS_incomplete_layerabs.py",
            catalog_paths,
        )
        self.assertNotIn(
            "sart/layerabs/LayerABS_abstract_sart_stats.py",
            catalog_paths,
        )


if __name__ == "__main__":
    unittest.main()
