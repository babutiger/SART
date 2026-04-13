#!/usr/bin/env python3
"""Run experiment scripts from the repository root.

This launcher fixes two recurring issues in the original codebase:
1. Many scripts assume they are launched from their own directory.
2. Different scripts rely on either the repo root or sart/
   being present on ``sys.path``.

Use this file instead of calling experiment scripts directly.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

from experiment_catalog import (
    build_layerabs_catalog,
    build_layerabs_families,
    is_layerabs_family_controller,
)
from paper_presets import (
    build_paper_presets,
    filter_paper_presets,
    resolve_paper_preset,
)
from paper_results import print_paper_preset_summary
from paper_results import (
    build_paper_preset_summary_payload,
    export_paper_preset_summaries,
    render_paper_preset_summary_json,
    render_paper_preset_summary_tsv,
    summarize_paper_preset,
)
from layerabs_naming import (
    KNOWN_LAYERABS_FAMILIES,
    LEGACY_FAMILY_ALIASES,
    LEGACY_SCRIPT_ALIASES,
)

REPO_ROOT = Path(__file__).resolve().parent
SART_ROOT = REPO_ROOT / "sart"
LAYERABS_ROOT = SART_ROOT / "layerabs"


@dataclass(frozen=True)
class ScriptListing:
    path: Path
    kind: str
    family: str = ""
    paper_role: str = ""
    benchmark: str = ""
    network: str = ""
    role: str = ""


_SCRIPT_KIND_ORDER = {
    "layerabs_family_controller": 0,
    "layerabs_default_profile": 1,
    "thin_wrapper": 2,
    "family_variant": 3,
    "specialized_standalone": 4,
    "standalone_experiment": 5,
    "legacy_copy": 6,
    "archived_implementation": 7,
    "verify_script": 8,
    "script": 9,
}


def discover_scripts() -> list[Path]:
    scripts: list[Path] = []

    for path in sorted((SART_ROOT / "verify").rglob("*.py")):
        if "__pycache__" not in path.parts:
            scripts.append(path)

    for path in sorted(LAYERABS_ROOT.rglob("LayerABS*.py")):
        if "__pycache__" in path.parts:
            continue
        scripts.append(path)

    return scripts


def format_script(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def format_repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _controller_family_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("LayerABS_"):
        return stem[len("LayerABS_") :].lower().replace("+", "_plus_")
    return stem.lower().replace("+", "_plus_")


def describe_script(
    path: Path,
    *,
    layerabs_catalog,
    layerabs_families,
) -> ScriptListing:
    layerabs_by_path = {exp.path.resolve(): exp for exp in layerabs_catalog}
    family_by_name = {family.family: family for family in layerabs_families}
    resolved_path = path.resolve()

    layerabs_exp = layerabs_by_path.get(resolved_path)
    if layerabs_exp is not None:
        kind = layerabs_exp.role
        if layerabs_exp.bucket == "default_profiles":
            kind = "layerabs_default_profile"
        return ScriptListing(
            path=path,
            kind=kind,
            family=layerabs_exp.family,
            paper_role=layerabs_exp.paper_role,
            benchmark=layerabs_exp.benchmark,
            network=layerabs_exp.network,
            role=layerabs_exp.role,
        )

    if is_layerabs_family_controller(path, repo_root=REPO_ROOT):
        family_name = _controller_family_name(path)
        family = family_by_name.get(family_name)
        return ScriptListing(
            path=path,
            kind="layerabs_family_controller",
            family=family_name,
            paper_role="" if family is None else family.paper_role,
        )

    try:
        path.relative_to(SART_ROOT / "verify")
    except ValueError:
        return ScriptListing(path=path, kind="script")

    return ScriptListing(path=path, kind="verify_script")


def format_script_listing(script: ScriptListing) -> str:
    fields = [script.kind]
    if script.family:
        fields.append(f"family={script.family}")
    if script.paper_role:
        fields.append(f"paper_role={script.paper_role}")
    if script.benchmark:
        fields.append(f"benchmark={script.benchmark}")
    if script.network:
        fields.append(f"network={script.network}")
    if script.role and script.kind != script.role:
        fields.append(f"role={script.role}")
    fields.append(f"path={format_script(script.path)}")
    return "\t".join(fields)


def sort_script_listings(items: list[ScriptListing]) -> list[ScriptListing]:
    return sorted(
        items,
        key=lambda item: (
            _SCRIPT_KIND_ORDER.get(item.kind, 99),
            item.family,
            item.benchmark,
            item.network,
            format_script(item.path),
        ),
    )


def _rewrite_legacy_experiment_query(query: str) -> str:
    parts = query.split("/")
    if not parts:
        return query
    if parts[0] == "layerabs" and len(parts) > 1:
        parts[1] = LEGACY_FAMILY_ALIASES.get(parts[1].lower(), parts[1])
        return "/".join(parts)
    parts[0] = LEGACY_FAMILY_ALIASES.get(parts[0].lower(), parts[0])
    return "/".join(parts)


def _rewrite_legacy_script_query(query: str) -> str:
    normalized = query.replace("\\", "/")
    normalized = normalized.replace(
        "Symbolic_Computation/symbolic_compute_code",
        "sart/layerabs",
    )
    normalized = normalized.replace("Symbolic_Computation/", "sart/")
    normalized = normalized.replace("symbolic_compute_code/", "layerabs/")
    return normalized


def resolve_script(query: str, available: list[Path]) -> Path:
    rewritten_query = _rewrite_legacy_script_query(query)
    normalized = rewritten_query.removesuffix(".py")
    direct = REPO_ROOT / f"{normalized}.py"
    if direct.exists():
        return direct.resolve()

    direct = REPO_ROOT / rewritten_query
    if direct.exists():
        return direct.resolve()

    exact_matches = [
        path
        for path in available
        if format_script(path).removesuffix(".py") == normalized
        or path.stem == normalized
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        names = "\n".join(f"  - {format_script(path)}" for path in exact_matches)
        raise SystemExit(f"Ambiguous script name '{query}'. Matches:\n{names}")

    alias_target = LEGACY_SCRIPT_ALIASES.get(normalized)
    if alias_target is not None:
        return resolve_script(alias_target, available)

    raise SystemExit(f"Unknown script: {query}")


def resolve_experiment(query: str, catalog) -> Path:
    rewritten_query = _rewrite_legacy_experiment_query(query)

    exact_matches = [exp.path for exp in catalog if exp.experiment_id == rewritten_query]
    if len(exact_matches) == 1:
        return exact_matches[0]

    suffix_matches = [
        exp.path
        for exp in catalog
        if exp.experiment_id.endswith(rewritten_query)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        names = "\n".join(
            f"  - {exp.experiment_id}"
            for exp in catalog
            if exp.experiment_id.endswith(rewritten_query)
        )
        raise SystemExit(f"Ambiguous experiment id '{query}'. Matches:\n{names}")

    raise SystemExit(f"Unknown experiment id: {query}")


def filter_layerabs_catalog(
    catalog,
    family: str,
    benchmark: str,
    content_group: str,
    bucket: str,
    text_filter: str,
):
    items = catalog
    if family:
        family_query = LEGACY_FAMILY_ALIASES.get(family.lower(), family.lower())
        if family_query in KNOWN_LAYERABS_FAMILIES:
            items = [exp for exp in items if exp.family.lower() == family_query]
        else:
            items = [exp for exp in items if family_query in exp.family.lower()]
    if benchmark:
        items = [exp for exp in items if benchmark.lower() == exp.benchmark.lower()]
    if content_group:
        items = [
            exp
            for exp in items
            if content_group.lower() == exp.content_group.lower()
        ]
    if bucket:
        bucket_query = bucket.lower().strip("/")
        items = [
            exp
            for exp in items
            if exp.bucket.lower() == bucket_query
            or exp.bucket.lower().startswith(f"{bucket_query}/")
        ]
    if text_filter:
        text_filter = text_filter.lower()
        items = [
            exp
            for exp in items
            if text_filter in exp.experiment_id.lower()
            or text_filter in format_script(exp.path).lower()
            or text_filter in exp.entrypoint.lower()
            or text_filter in exp.role.lower()
            or text_filter in exp.paper_role.lower()
            or text_filter in exp.content_group.lower()
            or text_filter in exp.bucket.lower()
        ]
    return items


def print_layerabs_catalog(catalog) -> None:
    for exp in catalog:
        marker = "*" if exp.canonical else "-"
        emit_line(
            f"{marker} {exp.experiment_id}\t"
            f"paper_role={exp.paper_role}\t"
            f"content={exp.content_group}\t"
            f"bucket={exp.bucket}\t"
            f"family={exp.family}\t"
            f"benchmark={exp.benchmark}\t"
            f"network={exp.network}\t"
            f"role={exp.role}\t"
            f"entry={exp.entrypoint}\t"
            f"path={format_script(exp.path)}"
        )


def print_layerabs_families(catalog, canonical_catalog) -> None:
    for family in build_layerabs_families(catalog, canonical_catalog=canonical_catalog):
        fields = [
            family.family,
            f"paper_role={family.paper_role}",
            f"status={family.status}",
            f"count={family.size}",
            f"benchmarks={','.join(family.benchmarks)}",
            f"networks={','.join(family.networks)}",
            f"canonical={family.canonical_id}",
            f"path={format_script(family.canonical_path)}",
        ]
        if family.controller_path is not None:
            fields.append(f"controller={format_script(family.controller_path)}")
        emit_line(
            "\t".join(fields)
        )


def print_paper_presets(presets) -> None:
    for preset in presets:
        fields = [
            preset.name,
            f"table={preset.table}",
            f"section={preset.section}",
            f"coverage={preset.coverage}",
            f"runs={len(preset.runs)}",
            f"title={preset.title}",
            f"description={preset.description}",
        ]
        if preset.notes:
            fields.append(f"notes={len(preset.notes)}")
        emit_line("\t".join(fields))


def emit_line(text: str) -> None:
    try:
        print(text)
    except BrokenPipeError as exc:
        raise SystemExit(0) from exc


def ensure_runtime_dirs() -> None:
    for path in [
        SART_ROOT / "result",
        SART_ROOT / "result" / "log",
        SART_ROOT / "result" / "original_result",
        SART_ROOT / "gurobi_model",
        SART_ROOT / "sources",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_execution_cwd(script_path: Path) -> Path:
    try:
        script_path.relative_to(LAYERABS_ROOT)
    except ValueError:
        return script_path.parent
    return LAYERABS_ROOT


def run_script(
    script_path: Path,
    dry_run: bool,
    script_args: list[str] | None = None,
) -> int:
    script_dir = script_path.parent
    execution_cwd = get_execution_cwd(script_path)
    ensure_runtime_dirs()
    script_args = script_args or []

    print(f"Selected script: {format_script(script_path)}")
    print(f"Execution cwd:   {execution_cwd.relative_to(REPO_ROOT).as_posix()}")
    if script_args:
        print(f"Script args:     {' '.join(script_args)}")
    if dry_run:
        return 0

    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    original_argv = list(sys.argv)
    try:
        os.chdir(execution_cwd)
        for extra_path in [
            str(REPO_ROOT),
            str(SART_ROOT),
            str(LAYERABS_ROOT),
            str(script_dir),
        ]:
            if extra_path not in sys.path:
                sys.path.insert(0, extra_path)
        sys.argv = [str(script_path)] + script_args
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        sys.argv[:] = original_argv


def run_paper_preset(
    preset,
    *,
    available: list[Path],
    dry_run: bool,
) -> int:
    emit_line(
        "\t".join(
            [
                f"paper_preset={preset.name}",
                f"table={preset.table}",
                f"section={preset.section}",
                f"coverage={preset.coverage}",
                f"runs={len(preset.runs)}",
                f"title={preset.title}",
            ]
        )
    )
    emit_line(f"description={preset.description}")
    for note in preset.notes:
        emit_line(f"note={note}")

    if preset.coverage == "unsupported":
        raise SystemExit(
            f"Paper preset '{preset.name}' is documented but not runnable yet."
        )

    if not preset.runs:
        raise SystemExit(
            f"Paper preset '{preset.name}' has no executable runs."
        )

    total = len(preset.runs)
    for index, step in enumerate(preset.runs, start=1):
        emit_line(
            f"[{index}/{total}]\tlabel={step.label}\tscript={step.script}\targs={' '.join(step.script_args())}"
        )
        target = resolve_script(step.script, available)
        status = run_script(
            target,
            dry_run=dry_run,
            script_args=list(step.script_args()),
        )
        if status != 0:
            return status
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="List and run verification experiments from the repository root.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment scripts.",
    )
    parser.add_argument(
        "--list-layerabs",
        action="store_true",
        help="List LayerABS experiments with normalized experiment ids.",
    )
    parser.add_argument(
        "--list-layerabs-families",
        action="store_true",
        help="List inferred LayerABS experiment families and their canonical scripts.",
    )
    parser.add_argument(
        "--list-paper-presets",
        action="store_true",
        help="List paper-facing experiment presets built from the local artifact.",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Filter listed scripts by substring.",
    )
    parser.add_argument(
        "--family",
        default="",
        help="Filter LayerABS listings by inferred family key.",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        help="Filter LayerABS listings by benchmark tag: mnist, cifar10, or vnncomp.",
    )
    parser.add_argument(
        "--content-group",
        default="",
        help="Filter LayerABS listings by content group: main_complete, ablation, measurement, eran_vnncomp, timelimit, final_output, or legacy.",
    )
    parser.add_argument(
        "--bucket",
        default="",
        help="Filter LayerABS listings by physical bucket such as root_entrypoints, family_wrappers/abstract_sart, specialized, or legacy.",
    )
    parser.add_argument(
        "--script",
        help="Relative script path or unique script stem.",
    )
    parser.add_argument(
        "--experiment",
        help="Normalized LayerABS experiment id from --list-layerabs.",
    )
    parser.add_argument(
        "--paper-preset",
        help="Named paper experiment preset from --list-paper-presets.",
    )
    parser.add_argument(
        "--summarize-paper-preset",
        help="Summarize the latest local artifacts for a paper preset without rerunning it.",
    )
    parser.add_argument(
        "--export-paper-tables",
        nargs="?",
        const="docs/paper_tables",
        default="",
        help="Export paper preset summaries as JSON/TSV files. Defaults to docs/paper_tables when no path is provided.",
    )
    parser.add_argument(
        "--summary-format",
        choices=("text", "json", "tsv"),
        default="text",
        help="Output format for --summarize-paper-preset. Default: text.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional file path for --summarize-paper-preset output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the selected script without executing it.",
    )
    parser.add_argument(
        "--script-arg",
        dest="script_args",
        action="append",
        default=[],
        help="Forward one argument to the target script. Repeat for multiple script arguments.",
    )
    args = parser.parse_args(argv)

    available = discover_scripts()
    layerabs_catalog = build_layerabs_catalog(REPO_ROOT)
    paper_presets = build_paper_presets()
    if args.list:
        layerabs_families = build_layerabs_families(layerabs_catalog)
        items: list[ScriptListing] = []
        for path in available:
            listing = describe_script(
                path,
                layerabs_catalog=layerabs_catalog,
                layerabs_families=layerabs_families,
            )
            items.append(listing)
        for item in sort_script_listings(items):
            line = format_script_listing(item)
            if args.filter.lower() in line.lower():
                emit_line(line)
        return 0

    if args.list_layerabs:
        print_layerabs_catalog(
            filter_layerabs_catalog(
                layerabs_catalog,
                family=args.family,
                benchmark=args.benchmark,
                content_group=args.content_group,
                bucket=args.bucket,
                text_filter=args.filter,
            )
        )
        return 0

    if args.list_layerabs_families:
        print_layerabs_families(
            filter_layerabs_catalog(
                layerabs_catalog,
                family=args.family,
                benchmark=args.benchmark,
                content_group=args.content_group,
                bucket=args.bucket,
                text_filter=args.filter,
            ),
            canonical_catalog=layerabs_catalog,
        )
        return 0

    if args.list_paper_presets:
        print_paper_presets(
            filter_paper_presets(
                paper_presets,
                args.filter,
            )
        )
        return 0

    if (
        not args.script
        and not args.experiment
        and not args.paper_preset
        and not args.summarize_paper_preset
        and not args.export_paper_tables
    ):
        parser.error(
            "one of --list, --list-layerabs, --list-layerabs-families, --list-paper-presets, --script, --experiment, --paper-preset, --summarize-paper-preset, or --export-paper-tables is required"
        )

    selected_target_flags = sum(
        bool(value)
        for value in (
            args.script,
            args.experiment,
            args.paper_preset,
            args.summarize_paper_preset,
            args.export_paper_tables,
        )
    )
    if selected_target_flags > 1:
        parser.error(
            "--script, --experiment, --paper-preset, --summarize-paper-preset, and --export-paper-tables cannot be used together"
        )

    if (args.paper_preset or args.summarize_paper_preset or args.export_paper_tables) and args.script_args:
        parser.error(
            "--script-arg cannot be combined with --paper-preset, --summarize-paper-preset, or --export-paper-tables"
        )

    if (args.summary_format != "text" or args.summary_output) and not args.summarize_paper_preset:
        parser.error(
            "--summary-format and --summary-output require --summarize-paper-preset"
        )

    if args.paper_preset:
        preset = resolve_paper_preset(args.paper_preset)
        return run_paper_preset(
            preset,
            available=available,
            dry_run=args.dry_run,
        )

    if args.summarize_paper_preset:
        preset = resolve_paper_preset(args.summarize_paper_preset)
        if args.summary_format == "text":
            if args.summary_output:
                lines: list[str] = []
                print_paper_preset_summary(
                    preset,
                    repo_root=REPO_ROOT,
                    emit_line=lines.append,
                )
                Path(args.summary_output).write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
            else:
                print_paper_preset_summary(
                    preset,
                    repo_root=REPO_ROOT,
                    emit_line=emit_line,
                )
        else:
            summaries = summarize_paper_preset(preset, repo_root=REPO_ROOT)
            payload = build_paper_preset_summary_payload(preset, summaries)
            if args.summary_format == "json":
                rendered = render_paper_preset_summary_json(payload)
            else:
                rendered = render_paper_preset_summary_tsv(payload)
            if args.summary_output:
                Path(args.summary_output).write_text(rendered + "\n", encoding="utf-8")
            else:
                emit_line(rendered)
        return 0

    if args.export_paper_tables:
        selected_presets = filter_paper_presets(paper_presets, args.filter)
        output_dir = (REPO_ROOT / args.export_paper_tables).resolve()
        written_paths = export_paper_preset_summaries(
            selected_presets,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
        )
        emit_line(
            f"exported_paper_tables={len(selected_presets)}\toutput_dir={format_repo_relative_path(output_dir)}"
        )
        for path in sorted(written_paths):
            emit_line(f"written={format_repo_relative_path(path)}")
        return 0

    if args.experiment:
        target = resolve_experiment(args.experiment, layerabs_catalog)
    else:
        target = resolve_script(args.script, available)
    return run_script(
        target,
        args.dry_run,
        script_args=args.script_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
