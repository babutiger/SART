from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiment_catalog import build_layerabs_catalog, build_layerabs_families
from run_experiment import (
    describe_script,
    print_layerabs_families,
    resolve_script,
    sort_script_listings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunExperimentListingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_layerabs_catalog(REPO_ROOT)
        self.families = build_layerabs_families(self.catalog)

    def test_model_neutral_controller_is_listed_as_family_controller(self):
        script = describe_script(
            REPO_ROOT
            / "sart/layerabs/LayerABS_abstract_sart.py",
            layerabs_catalog=self.catalog,
            layerabs_families=self.families,
        )

        self.assertEqual(script.kind, "layerabs_family_controller")
        self.assertEqual(script.family, "abstract_sart")
        self.assertEqual(script.paper_role, "layerabs_with_abstraction_sart")

    def test_default_profile_root_script_is_listed_separately(self):
        script = describe_script(
            REPO_ROOT
            / "sart/layerabs/default_profiles/LayerABS_abstract_sart_mnist_10x80.py",
            layerabs_catalog=self.catalog,
            layerabs_families=self.families,
        )

        self.assertEqual(script.kind, "layerabs_default_profile")
        self.assertEqual(script.family, "abstract_sart")
        self.assertEqual(script.benchmark, "mnist")
        self.assertEqual(script.network, "10x80")

    def test_verify_script_keeps_generic_verify_kind(self):
        script = describe_script(
            REPO_ROOT
            / "sart/verify/mnist_new_10x80/alpha_crown_mnist_new_10x80.py",
            layerabs_catalog=self.catalog,
            layerabs_families=self.families,
        )

        self.assertEqual(script.kind, "verify_script")

    def test_sort_script_listings_prioritizes_controllers(self):
        scripts = [
            describe_script(
                REPO_ROOT
                / "sart/layerabs/default_profiles/LayerABS_abstract_sart_mnist_10x80.py",
                layerabs_catalog=self.catalog,
                layerabs_families=self.families,
            ),
            describe_script(
                REPO_ROOT
                / "sart/layerabs/LayerABS_abstract_sart.py",
                layerabs_catalog=self.catalog,
                layerabs_families=self.families,
            ),
            describe_script(
                REPO_ROOT
                / "sart/verify/mnist_new_10x80/alpha_crown_mnist_new_10x80.py",
                layerabs_catalog=self.catalog,
                layerabs_families=self.families,
            ),
        ]

        ordered = sort_script_listings(scripts)

        self.assertEqual(ordered[0].kind, "layerabs_family_controller")
        self.assertEqual(ordered[1].kind, "layerabs_default_profile")
        self.assertEqual(ordered[2].kind, "verify_script")

    def test_family_listing_prints_controller_path(self):
        buffer = io.StringIO()
        filtered_catalog = [
            exp for exp in self.catalog if exp.family == "abstract_sart"
        ]

        with redirect_stdout(buffer):
            print_layerabs_families(
                filtered_catalog,
                canonical_catalog=self.catalog,
            )

        line = buffer.getvalue().strip()
        self.assertIn("controller=sart/layerabs/LayerABS_abstract_sart.py", line)
        self.assertIn(
            "path=sart/layerabs/default_profiles/LayerABS_abstract_sart_mnist_10x80.py",
            line,
        )

    def test_legacy_wrapper_name_resolves_to_renamed_wrapper(self):
        available = sorted(
            path
            for path in REPO_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        )

        resolved = resolve_script("LayerABS_complete_mnist_5x50", available)

        self.assertEqual(
            resolved.relative_to(REPO_ROOT).as_posix(),
            "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_mnist_5x50.py",
        )

    def test_legacy_vnncomp_eran_name_resolves_to_abstract_sart_wrapper(self):
        available = sorted(
            path
            for path in REPO_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        )

        resolved = resolve_script("LayerABS_vnncomp_eran_6x100", available)

        self.assertEqual(
            resolved.relative_to(REPO_ROOT).as_posix(),
            "sart/layerabs/family_wrappers/abstract_sart/LayerABS_abstract_sart_vnncomp_6x100.py",
        )

    def test_legacy_symbolic_computation_path_rewrites_to_new_tree(self):
        available = sorted(
            path
            for path in REPO_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        )

        resolved = resolve_script(
            "Symbolic_Computation/symbolic_compute_code/LayerABS_abstract_sart.py",
            available,
        )

        self.assertEqual(
            resolved.relative_to(REPO_ROOT).as_posix(),
            "sart/layerabs/LayerABS_abstract_sart.py",
        )


if __name__ == "__main__":
    unittest.main()
