from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from paper_presets import PaperPresetRun
from paper_results import summarize_paper_run
from run_experiment import main as run_experiment_main


REPO_ROOT = Path(__file__).resolve().parents[1]


class PaperResultsTests(unittest.TestCase):
    def test_incomplete_summary_uses_log_status_to_classify_timeout(self):
        run = PaperPresetRun(
            script="LayerABS_incomplete_layerabs",
            label="Incomplete-LayerABS on mnist_5x50 with k=3",
            variant="mnist_5x50",
            k_layers=3,
            mip_time_limit=30,
        )
        timestamp = "2026-04-13 14:00:00"

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            result_dir = temp_dir / "sart" / "result" / "original_result"
            log_dir = temp_dir / "sart" / "result" / "log"
            result_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)

            (result_dir / f"mnist_new_5x50_incomplete_layerabs_number_result_k_3_delta_0.018_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- is__verify : True",
                        "mnist_property_0 -- time : 10.0",
                        "mnist_property_1 -- is__verify : 0",
                        "mnist_property_1 -- time : 30.0",
                        "delta : 0.018",
                        "k_layers : 3",
                        "number_sum : 1",
                        "time_sum : 40.0",
                        "time_average : 20.0",
                        "time_max : 30.0",
                    ]
                ),
                encoding="utf-8",
            )
            (log_dir / f"layerabs_incomplete_family_propagation_log_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- Verified",
                        "mnist_property_0 -- StageOutcome: stage1_safe",
                        "mnist_property_1 -- Time out",
                        "mnist_property_1 -- UnVerified, Time out",
                        "mnist_property_1 -- StageOutcome: stage2_timeout",
                        "number_sum: 1",
                        "time_sum: 40.0",
                        "time_average: 20.0",
                        "time_max: 30.0",
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_paper_run(run, repo_root=temp_dir)

        self.assertEqual(summary.verified, 1)
        self.assertEqual(summary.timeout, 1)
        self.assertEqual(summary.unknown, 0)
        self.assertEqual(summary.unsafe, 0)
        self.assertEqual(summary.unverified, 0)
        self.assertEqual(summary.unresolved, 0)
        self.assertEqual(summary.expected_property_count, 100)
        self.assertEqual(summary.number_sum, 1)
        self.assertEqual(summary.time_average, 20.0)
        self.assertEqual(summary.stage_outcome_counts["stage1_safe"], 1)
        self.assertEqual(summary.stage_outcome_counts["stage2_timeout"], 1)
        self.assertEqual(summary.stage_outcome_average_times["stage1_safe"], 10.0)
        self.assertEqual(summary.stage_outcome_average_times["stage2_timeout"], 30.0)
        self.assertIsNotNone(summary.result_path)
        self.assertIsNotNone(summary.log_path)

    def test_result_only_summary_is_limited_to_incomplete_controller(self):
        run = PaperPresetRun(
            script="LayerABS_incomplete_layerabs",
            label="Incomplete-LayerABS on mnist_10x80 with k=2",
            variant="mnist_10x80",
            k_layers=2,
        )
        timestamp = "2026-04-13 15:00:00"

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            result_dir = temp_dir / "sart" / "result" / "original_result"
            result_dir.mkdir(parents=True)

            (result_dir / f"mnist_new_10x80_incomplete_layerabs_number_result_k_2_delta_0.015_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- is__verify : 1",
                        "mnist_property_0 -- time : 12.0",
                        "mnist_property_1 -- is__verify : 0",
                        "mnist_property_1 -- time : 18.0",
                        "delta : 0.015",
                        "k_layers : 2",
                        "number_sum : 1",
                        "time_sum : 30.0",
                        "time_average : 15.0",
                        "time_max : 18.0",
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_paper_run(run, repo_root=temp_dir)

        self.assertEqual(summary.verified, 1)
        self.assertEqual(summary.timeout, 0)
        self.assertEqual(summary.unknown, 0)
        self.assertEqual(summary.unsafe, 0)
        self.assertEqual(summary.unverified, 0)
        self.assertEqual(summary.unresolved, 1)
        self.assertIsNotNone(summary.result_path)
        self.assertIsNone(summary.log_path)

    def test_run_experiment_can_print_paper_preset_summary(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            status = run_experiment_main(
                ["--summarize-paper-preset", "table6"]
            )

        output = buffer.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("paper_preset_summary=table6_incomplete_mnist", output)
        self.assertIn("coverage=partial", output)
        self.assertIn("aggregate", output)

    def test_run_experiment_can_emit_json_summary(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            status = run_experiment_main(
                ["--summarize-paper-preset", "table6", "--summary-format", "json"]
            )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(payload["name"], "table6_incomplete_mnist")
        self.assertEqual(payload["coverage"], "partial")
        self.assertIn("runs", payload)
        self.assertIn("aggregate", payload)

    def test_run_experiment_can_emit_tsv_summary_to_file(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_path = Path(temp_dir_str) / "table6.tsv"
            status = run_experiment_main(
                [
                    "--summarize-paper-preset",
                    "table6",
                    "--summary-format",
                    "tsv",
                    "--summary-output",
                    str(output_path),
                ]
            )

            contents = output_path.read_text(encoding="utf-8")

        self.assertEqual(status, 0)
        self.assertIn("kind", contents.splitlines()[0])
        self.assertIn("aggregate", contents)

    def test_run_experiment_can_export_paper_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            output_dir = Path(temp_dir_str) / "paper_tables"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = run_experiment_main(
                    [
                        "--export-paper-tables",
                        str(output_dir),
                        "--filter",
                        "table6",
                    ]
                )

            index_json = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            exported_json = json.loads(
                (output_dir / "table6_incomplete_mnist.json").read_text(encoding="utf-8")
            )
            exported_tsv = (output_dir / "table6_incomplete_mnist.tsv").read_text(
                encoding="utf-8"
            )

        self.assertEqual(status, 0)
        self.assertEqual(len(index_json["presets"]), 1)
        self.assertEqual(index_json["presets"][0]["name"], "table6_incomplete_mnist")
        self.assertEqual(exported_json["name"], "table6_incomplete_mnist")
        self.assertIn("kind", exported_tsv.splitlines()[0])
        self.assertIn("written=", buffer.getvalue())

    def test_stage_metrics_are_printed_when_stage_outcomes_exist(self):
        run = PaperPresetRun(
            script="LayerABS_incomplete_layerabs",
            label="Incomplete-LayerABS on mnist_5x50 with k=3",
            variant="mnist_5x50",
            k_layers=3,
        )
        timestamp = "2026-04-13 16:00:00"

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            result_dir = temp_dir / "sart" / "result" / "original_result"
            log_dir = temp_dir / "sart" / "result" / "log"
            result_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)

            (result_dir / f"mnist_new_5x50_incomplete_layerabs_number_result_k_3_delta_0.018_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- is__verify : True",
                        "mnist_property_0 -- time : 8.0",
                        "mnist_property_1 -- is__verify : 0",
                        "mnist_property_1 -- time : 25.0",
                        "delta : 0.018",
                        "k_layers : 3",
                        "number_sum : 1",
                        "time_sum : 33.0",
                        "time_average : 16.5",
                        "time_max : 25.0",
                    ]
                ),
                encoding="utf-8",
            )
            (log_dir / f"layerabs_incomplete_family_propagation_log_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- Verified",
                        "mnist_property_0 -- StageOutcome: stage1_safe",
                        "mnist_property_1 -- Unknown, Incomplete",
                        "mnist_property_1 -- StageOutcome: stage2_unknown",
                    ]
                ),
                encoding="utf-8",
            )

            from paper_presets import PaperPreset
            from paper_results import print_paper_preset_summary

            preset = PaperPreset(
                name="synthetic_table5",
                table="Table 5",
                section="6.2",
                title="Synthetic fallback summary",
                coverage="supported",
                description="Synthetic test preset.",
                runs=(run,),
            )
            buffer = io.StringIO()
            print_paper_preset_summary(
                preset,
                repo_root=temp_dir,
                emit_line=lambda text: print(text, file=buffer),
            )

        output = buffer.getvalue()
        self.assertIn("stage_metrics", output)
        self.assertIn("stage1_safe=1", output)
        self.assertIn("stage2_unknown=1", output)
        self.assertIn("stage1_safe_time_average=8.0", output)
        self.assertIn("expected_properties=100", output)

    def test_table5_metrics_are_printed_for_table5_presets(self):
        run = PaperPresetRun(
            script="LayerABS_incomplete_layerabs",
            label="Incomplete-LayerABS on mnist_5x50 with k=3",
            variant="mnist_5x50",
            k_layers=3,
        )
        timestamp = "2026-04-13 17:00:00"

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            result_dir = temp_dir / "sart" / "result" / "original_result"
            log_dir = temp_dir / "sart" / "result" / "log"
            result_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)

            (result_dir / f"mnist_new_5x50_incomplete_layerabs_number_result_k_3_delta_0.018_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- is__verify : True",
                        "mnist_property_0 -- time : 10.0",
                        "delta : 0.018",
                        "k_layers : 3",
                        "number_sum : 1",
                        "time_sum : 10.0",
                        "time_average : 10.0",
                        "time_max : 10.0",
                    ]
                ),
                encoding="utf-8",
            )
            (log_dir / f"layerabs_incomplete_family_propagation_log_{timestamp}.txt").write_text(
                "\n".join(
                    [
                        "mnist_property_0 -- Verified",
                        "mnist_property_0 -- StageOutcome: stage1_safe",
                    ]
                ),
                encoding="utf-8",
            )

            from paper_presets import PaperPreset
            from paper_results import print_paper_preset_summary

            preset = PaperPreset(
                name="table5_fallback_frequency",
                table="Table 5",
                section="6.2",
                title="Synthetic fallback summary",
                coverage="supported",
                description="Synthetic test preset.",
                runs=(run,),
            )
            buffer = io.StringIO()
            print_paper_preset_summary(
                preset,
                repo_root=temp_dir,
                emit_line=lambda text: print(text, file=buffer),
            )

        output = buffer.getvalue()
        self.assertIn("table5_metrics", output)
        self.assertIn("stage1_safe_pct=1.0", output)
        self.assertIn("stage1_safe_time_average=10.0", output)


if __name__ == "__main__":
    unittest.main()
