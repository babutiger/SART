from __future__ import annotations

import importlib
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_IMPORT_SAFE_FILES = [
    REPO_ROOT
    / "sart/layerabs/LayerABS_abstract_sart.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_abstract_sart_mnist_10x80.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_abstract_milp.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_abstract_milp_mnist_5x50.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_abstract_sart_stats.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_abstract_sart_stats_mnist_6x100.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_abstract_milp_stats.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_abstract_milp_stats_mnist_6x100.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_abstract_sart_timelimit.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_abstract_sart_timelimit_mnist_10x80.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_incomplete_layerabs.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_incomplete_layerabs_mnist_10x80.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_puresart.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_puresart_mnist_10x80.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_standard_milp.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_standard_milp_mnist_10x80.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_puresart_stats.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_puresart_stats_vnncomp_6x100.py",
    REPO_ROOT
    / "sart/layerabs/LayerABS_standard_milp_stats.py",
    REPO_ROOT
    / "sart/layerabs/default_profiles/LayerABS_standard_milp_stats_vnncomp_6x100.py",
    REPO_ROOT
    / "sart/layerabs/layerabs_core/layerabs_abstract_sart_family_propagation.py",
    REPO_ROOT
    / "sart/layerabs/layerabs_core/layerabs_abstract_milp_family_propagation.py",
    REPO_ROOT
    / "sart/layerabs/layerabs_core/layerabs_abstract_sart_stats_family_propagation.py",
    REPO_ROOT
    / "sart/layerabs/layerabs_core/layerabs_abstract_milp_stats_family_propagation.py",
    REPO_ROOT
    / "sart/layerabs/layerabs_core/layerabs_abstract_sart_timelimit_family_propagation.py",
    REPO_ROOT
    / "sart/layerabs/layerabs_core/layerabs_incomplete_family_propagation.py",
    REPO_ROOT
    / "sart/layerabs/support/read_property.py",
    REPO_ROOT
    / "sart/layerabs/support/layerabs_controller_helpers.py",
]

BANNED_IMPORT_SIDE_EFFECT_PATTERNS = [
    "multiprocessing.Manager().list()",
    "child_process_ids = multiprocessing.Manager().list()",
    "class Logger(",
    "sys.stdout = Logger(",
    "style_time = time.strftime(",
    "add_parent_dirs_to_sys_path()",
]

VISIBLE_ENGLISH_ONLY_FILES = [
    REPO_ROOT / "sart/layerabs/layerabs_core/layerabs_shared_helpers.py",
    REPO_ROOT / "sart/layerabs/support/contain_abs.py",
    REPO_ROOT / "sart/layerabs/support/exnum.py",
    REPO_ROOT / "sart/layerabs/support/extract_filename_from_path.py",
    REPO_ROOT / "sart/layerabs/support/isconstant.py",
    REPO_ROOT / "sart/layerabs/support/lstr2str.py",
    REPO_ROOT / "sart/layerabs/support/replace2sub.py",
    REPO_ROOT / "sart/layerabs/support/timeout.py",
]

ARCHIVED_CHINESE_ALLOWED_PATTERNS = (
    "archive_*_impl.py",
)


class DummyStreamLogger:
    def __init__(self, filename, stream):
        self.filename = filename
        self.stream = stream

    def write(self, message):
        self.stream.write(message)

    def flush(self):
        self.stream.flush()


class LayerAbsLazyImportTests(unittest.TestCase):
    def test_active_files_do_not_embed_eager_import_side_effect_patterns(self):
        for path in ACTIVE_IMPORT_SAFE_FILES:
            source = path.read_text(encoding="utf-8")
            for pattern in BANNED_IMPORT_SIDE_EFFECT_PATTERNS:
                self.assertNotIn(pattern, source, f"{path} still contains {pattern!r}")

    def test_visible_helper_files_do_not_contain_chinese_text(self):
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        for path in VISIBLE_ENGLISH_ONLY_FILES:
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(
                chinese_pattern.search(source),
                f"{path} still contains Chinese text",
            )

    def test_active_python_sources_do_not_contain_chinese_text(self):
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        roots = [
            REPO_ROOT / "sart/layerabs",
            REPO_ROOT / "sart/verify",
        ]

        for root in roots:
            for path in root.rglob("*.py"):
                if any(path.match(f"**/{pattern}") for pattern in ARCHIVED_CHINESE_ALLOWED_PATTERNS):
                    continue
                source = path.read_text(encoding="utf-8")
                self.assertIsNone(
                    chinese_pattern.search(source),
                    f"{path} still contains Chinese text",
                )

    def test_logging_helper_is_lazy_until_materialized(self):
        from sart.layerabs.layerabs_core import (
            layerabs_logging_helpers as logging_helpers,
        )

        original_stdout = sys.stdout
        handle = logging_helpers.redirect_stdout_to_timestamped_log(
            "fake_runner.py",
            stream=original_stdout,
        )
        self.assertIs(sys.stdout, original_stdout)
        self.assertFalse(handle.is_configured)

        with mock.patch.object(logging_helpers, "StreamLogger", DummyStreamLogger), mock.patch.object(
            logging_helpers.os,
            "makedirs",
        ) as mocked_makedirs, mock.patch.object(
            logging_helpers.time,
            "strftime",
            return_value="2026-04-12 00:00:00",
        ):
            try:
                formatted = f"{handle}"
                self.assertEqual(formatted, "2026-04-12 00:00:00")
                self.assertTrue(handle.is_configured)
                self.assertIsInstance(sys.stdout, DummyStreamLogger)
                mocked_makedirs.assert_called_once()
            finally:
                sys.stdout = original_stdout

    def test_process_helper_creates_manager_only_on_first_access(self):
        from sart.layerabs.layerabs_core import (
            layerabs_process_helpers as process_helpers,
        )

        process_helpers._PROCESS_REGISTRY.clear()
        fake_manager = mock.Mock()
        fake_child_list = []
        fake_manager.list.return_value = fake_child_list

        with mock.patch.object(process_helpers.multiprocessing, "Manager", return_value=fake_manager) as mocked_manager:
            child_process_ids = process_helpers.get_managed_child_process_list("lazy-test")
            child_process_ids_again = process_helpers.get_managed_child_process_list("lazy-test")

        self.assertIs(child_process_ids, fake_child_list)
        self.assertIs(child_process_ids_again, fake_child_list)
        mocked_manager.assert_called_once()
        fake_manager.list.assert_called_once()
        process_helpers._PROCESS_REGISTRY.clear()

    def test_compat_helper_raises_standard_module_style_attribute_error(self):
        from sart.layerabs.layerabs_core.layerabs_compat_helpers import (
            resolve_legacy_alias,
        )

        self.assertEqual(
            resolve_legacy_alias("fake.module", "old_name", {"old_name": 123}),
            123,
        )
        with self.assertRaisesRegex(
            AttributeError,
            r"module 'fake\.module' has no attribute 'missing'",
        ):
            resolve_legacy_alias("fake.module", "missing", {"old_name": 123})

    def test_support_read_nnet_parses_minimal_network(self):
        from sart.layerabs.support.read_nnet import (
            read_nnet_file,
        )

        nnet_contents = "\n".join(
            [
                "// comment",
                "// comment",
                "2,2,1,",
                "0,",
                "0,",
                "0,",
                "0,",
                "0,",
                "1.0,2.0,",
                "3.0,4.0,",
                "0.1,",
                "0.2,",
                "5.0,6.0,",
                "0.3,",
            ]
        )

        with tempfile.NamedTemporaryFile("w", suffix=".nnet", delete=False) as file_handle:
            file_handle.write(nnet_contents)
            temp_path = file_handle.name

        try:
            weights, biases, input_size, hidden_sizes, output_size = read_nnet_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        self.assertEqual(weights, [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]])
        self.assertEqual(biases, [[[0.1], [0.2]], [[0.3]]])
        self.assertEqual(input_size, 2)
        self.assertEqual(hidden_sizes, [2])
        self.assertEqual(output_size, 1)

    def test_support_read_property_parses_minimal_property(self):
        from sart.layerabs.support.read_property import (
            input_bound_pair,
            read_property,
        )

        property_contents = "\n".join(
            [
                "0.1",
                "0.9",
                "1 -1 0",
                "-2 3 1",
            ]
        )

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as file_handle:
            file_handle.write(property_contents)
            temp_path = file_handle.name

        try:
            input_pixels, property_weights, property_biases = read_property(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        self.assertEqual(input_pixels, [0.1, 0.9])
        self.assertEqual(property_weights, [[1, -1], [-2, 3]])
        self.assertEqual(property_biases, [[0], [1]])
        self.assertEqual(
            input_bound_pair(input_pixels, 0.2),
            [[0, 0.30000000000000004], [0.7, 1]],
        )

    def test_support_add_sys_path_is_idempotent_for_explicit_path(self):
        from sart.layerabs.support.add_sys_path import (
            add_parent_dirs_to_sys_path,
        )

        original_sys_path = list(sys.path)
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "a" / "b"
            nested_path.mkdir(parents=True, exist_ok=True)
            try:
                added_paths = add_parent_dirs_to_sys_path(nested_path)
                added_paths_again = add_parent_dirs_to_sys_path(nested_path)
            finally:
                sys.path[:] = original_sys_path

        self.assertGreaterEqual(len(added_paths), 3)
        self.assertEqual(added_paths_again, [])

    def test_support_modules_import_without_touching_stdout(self):
        original_stdout = sys.stdout
        original_sys_path = list(sys.path)
        module_names = [
            "sart.layerabs.support.read_nnet",
            "sart.layerabs.support.read_property",
        ]

        for module_name in module_names:
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            self.assertIs(sys.stdout, original_stdout)
            self.assertEqual(sys.path, original_sys_path)
            self.assertIsNotNone(module)

    def test_active_core_modules_do_not_resolve_dunder_attributes(self):
        module_names = [
            "sart.layerabs.layerabs_core.layerabs_abstract_sart_family_propagation",
        ]

        gurobi_stub = types.ModuleType("gurobipy")
        gurobi_stub.GRB = types.SimpleNamespace(
            Callback=types.SimpleNamespace(MIP=0, MIP_OBJBST=1, MIP_OBJBND=2),
            CONTINUOUS=0,
            INFINITY=float("inf"),
        )
        gurobi_stub.Model = object

        with mock.patch.dict(sys.modules, {"gurobipy": gurobi_stub}, clear=False):
            for module_name in module_names:
                sys.modules.pop(module_name, None)
                module = importlib.import_module(module_name)
                with self.assertRaises(AttributeError):
                    getattr(module, "__path__")

    def test_active_core_family_modules_import_with_stubbed_gurobi(self):
        original_stdout = sys.stdout
        module_names = [
            "sart.layerabs.LayerABS_abstract_sart",
            "sart.layerabs.LayerABS_abstract_milp",
            "sart.layerabs.LayerABS_abstract_sart_stats",
            "sart.layerabs.LayerABS_abstract_milp_stats",
            "sart.layerabs.LayerABS_abstract_sart_timelimit",
            "sart.layerabs.LayerABS_incomplete_layerabs",
            "sart.layerabs.LayerABS_puresart",
            "sart.layerabs.LayerABS_standard_milp",
            "sart.layerabs.LayerABS_puresart_stats",
            "sart.layerabs.LayerABS_standard_milp_stats",
            "sart.layerabs.layerabs_core.layerabs_abstract_sart_family_propagation",
            "sart.layerabs.layerabs_core.layerabs_abstract_milp_family_propagation",
            "sart.layerabs.layerabs_core.layerabs_abstract_sart_timelimit_family_propagation",
            "sart.layerabs.layerabs_core.layerabs_incomplete_family_propagation",
            "sart.layerabs.layerabs_core.layerabs_abstract_sart_stats_family_propagation",
            "sart.layerabs.layerabs_core.layerabs_abstract_milp_stats_family_propagation",
        ]

        gurobi_stub = types.ModuleType("gurobipy")
        gurobi_stub.GRB = types.SimpleNamespace(
            Callback=types.SimpleNamespace(MIP=0, MIP_OBJBST=1, MIP_OBJBND=2),
            CONTINUOUS=0,
            INFINITY=float("inf"),
        )
        gurobi_stub.Model = object

        with mock.patch.dict(sys.modules, {"gurobipy": gurobi_stub}, clear=False):
            for module_name in module_names:
                sys.modules.pop(module_name, None)
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)
                self.assertIs(sys.stdout, original_stdout)


if __name__ == "__main__":
    unittest.main()
