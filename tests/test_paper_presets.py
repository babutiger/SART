from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from paper_presets import (
    build_paper_presets,
    filter_paper_presets,
    resolve_paper_preset,
)
from run_experiment import discover_scripts, print_paper_presets, run_paper_preset


REPO_ROOT = Path(__file__).resolve().parents[1]


class PaperPresetTests(unittest.TestCase):
    def test_table8_preset_points_to_supported_vnncomp_complete_runs(self):
        preset = resolve_paper_preset("table8")

        self.assertEqual(preset.name, "table8_vnncomp_complete")
        self.assertEqual(preset.coverage, "supported")
        self.assertEqual(len(preset.runs), 2)
        self.assertEqual(
            tuple(run.variant for run in preset.runs),
            ("vnncomp_6x100", "vnncomp_9x100"),
        )

    def test_table11_preset_is_documented_as_unsupported(self):
        preset = resolve_paper_preset("table11")

        self.assertEqual(preset.name, "table11_incomplete_vnncomp_ldsa_k3")
        self.assertEqual(preset.coverage, "unsupported")
        self.assertEqual(preset.runs, ())

    def test_table3_preset_now_maps_to_explicit_hard_case_variants(self):
        preset = resolve_paper_preset("table3")

        self.assertEqual(preset.coverage, "supported")
        self.assertEqual(len(preset.runs), 2)
        self.assertEqual(
            tuple(run.variant for run in preset.runs),
            (
                "paper_table3_vnncomp_6x100_hard_cases",
                "paper_table3_vnncomp_6x100_hard_cases",
            ),
        )

    def test_filter_paper_presets_matches_by_table_label(self):
        presets = build_paper_presets()

        filtered = filter_paper_presets(presets, "table 10")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "table10_incomplete_vnncomp_k_sweep")

    def test_print_paper_presets_includes_coverage_and_table(self):
        preset = resolve_paper_preset("table4")
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_paper_presets((preset,))

        output = buffer.getvalue()
        self.assertIn("table=Table 4", output)
        self.assertIn("coverage=supported", output)
        self.assertIn("title=LayerABS verification on different MNIST models", output)

    def test_run_paper_preset_dry_run_expands_into_multiple_controller_runs(self):
        preset = resolve_paper_preset("table10")
        available = discover_scripts()
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            status = run_paper_preset(
                preset,
                available=available,
                dry_run=True,
            )

        output = buffer.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("paper_preset=table10_incomplete_vnncomp_k_sweep", output)
        self.assertIn("[1/10]", output)
        self.assertIn("LayerABS_incomplete_layerabs", output)
        self.assertIn("--k-layers 5", output)

    def test_table6_preset_applies_paper_fairness_mip_limit(self):
        preset = resolve_paper_preset("table6")

        self.assertTrue(all(run.mip_time_limit == 30 for run in preset.runs))
        self.assertTrue(all("--mip-time-limit" in " ".join(run.script_args()) for run in preset.runs))

    def test_run_paper_preset_rejects_unsupported_entries(self):
        preset = resolve_paper_preset("table12")

        with self.assertRaises(SystemExit):
            run_paper_preset(
                preset,
                available=discover_scripts(),
                dry_run=True,
            )


if __name__ == "__main__":
    unittest.main()
