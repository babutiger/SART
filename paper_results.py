from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from paper_presets import PaperPreset, PaperPresetRun


REPO_ROOT = Path(__file__).resolve().parent
RESULT_DIR = REPO_ROOT / "sart" / "result" / "original_result"
LOG_DIR = REPO_ROOT / "sart" / "result" / "log"

_TIMESTAMP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.txt$")
_RESULT_FLAG_RE = re.compile(r"^(?P<label>.+?) -- is__verify : (?P<value>.+)$")
_RESULT_TIME_RE = re.compile(r"^(?P<label>.+?) -- time : (?P<value>.+)$")
_SUMMARY_RE = re.compile(r"^(?P<key>[A-Za-z_]+)\s*:\s*(?P<value>.+)$")
_LOG_STATUS_RE = re.compile(r"^(?P<label>.+?) -- (?P<status>.+)$")
_STAGE_OUTCOME_RE = re.compile(r"^(?P<label>.+?) -- StageOutcome: (?P<outcome>.+)$")

_CONTROLLER_VARIANT_MODULES = {
    "LayerABS_abstract_sart": (
        "sart.layerabs.layerabs_variants.abstract_sart_variants"
    ),
    "LayerABS_abstract_milp": (
        "sart.layerabs.layerabs_variants.abstract_milp_variants"
    ),
    "LayerABS_incomplete_layerabs": (
        "sart.layerabs.layerabs_variants.incomplete_layerabs_variants"
    ),
    "LayerABS_puresart": (
        "sart.layerabs.layerabs_variants.puresart_variants"
    ),
    "LayerABS_standard_milp": (
        "sart.layerabs.layerabs_variants.standard_milp_variants"
    ),
    "LayerABS_abstract_sart_stats": (
        "sart.layerabs.layerabs_variants.abstract_sart_stats_variants"
    ),
    "LayerABS_abstract_milp_stats": (
        "sart.layerabs.layerabs_variants.abstract_milp_stats_variants"
    ),
    "LayerABS_puresart_stats": (
        "sart.layerabs.layerabs_variants.puresart_stats_variants"
    ),
    "LayerABS_standard_milp_stats": (
        "sart.layerabs.layerabs_variants.standard_milp_stats_variants"
    ),
    "LayerABS_abstract_sart_timelimit": (
        "sart.layerabs.layerabs_variants.abstract_sart_timelimit_variants"
    ),
}

_CONTROLLER_LOG_PREFIXES = {
    "LayerABS_abstract_sart": ("layerabs_complete_family_propagation",),
    "LayerABS_abstract_milp": ("layerabs_complete_milp_family_propagation",),
    "LayerABS_incomplete_layerabs": ("layerabs_incomplete_family_propagation",),
    "LayerABS_puresart": (
        "LayerABS_puresart",
        "LayerABS_puresart_complete_mnist_10x80",
    ),
    "LayerABS_standard_milp": (
        "LayerABS_standard_milp",
        "LayerABS_puremilp_complete_mnist_10x80",
    ),
    "LayerABS_abstract_sart_stats": ("layerabs_complete_stats_family_propagation",),
    "LayerABS_abstract_milp_stats": (
        "layerabs_complete_stats_milp_family_propagation",
    ),
    "LayerABS_puresart_stats": (
        "LayerABS_puresart_stats",
        "LayerABS_puresart_stats_vnncomp_6x100",
    ),
    "LayerABS_standard_milp_stats": (
        "LayerABS_standard_milp_stats",
        "LayerABS_puremilp_stats_vnncomp_6x100",
    ),
    "LayerABS_abstract_sart_timelimit": (
        "layerabs_complete_timelimit_family_propagation",
    ),
}

_USES_K_LAYER_RESULT_NAME = {"LayerABS_incomplete_layerabs"}
_RESULT_ONLY_SAFE_CONTROLLERS = {"LayerABS_incomplete_layerabs"}

_LOG_STATUS_MAP = {
    "Verified": "verified",
    "Time out": "timeout",
    "UnVerified, Time out": "timeout",
    "Unknown, Incomplete": "unknown",
    "UnSafe, Complete": "unsafe",
    "UnVerified": "unverified",
    "UnVerified, Continue Verify...": "continuing",
}


@dataclass(frozen=True)
class PaperRunArtifactMatch:
    result_path: Path | None
    log_path: Path | None
    timestamp: str | None
    result_prefix: str
    delta: float
    k_layers: int | None


@dataclass(frozen=True)
class PaperRunSummary:
    run: PaperPresetRun
    result_prefix: str
    delta: float
    k_layers: int | None
    result_path: Path | None
    log_path: Path | None
    timestamp: str | None
    verified: int
    timeout: int
    unknown: int
    unsafe: int
    unverified: int
    unresolved: int
    property_count: int
    expected_property_count: int
    number_sum: int | float | None
    time_sum: float | None
    time_average: float | None
    time_max: float | None
    stage_outcome_counts: dict[str, int]
    stage_outcome_average_times: dict[str, float]


def _parse_timestamp(path: Path) -> str | None:
    match = _TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    return match.group(1)


def _parse_scalar(value: str):
    text = value.strip()
    if text in {"True", "False"}:
        return text == "True"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _is_verified_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true"}


def _load_variant_config(run: PaperPresetRun):
    if run.variant is None:
        raise ValueError(f"Paper preset run '{run.label}' is missing a variant key")
    module_name = _CONTROLLER_VARIANT_MODULES.get(run.script)
    if module_name is None:
        known = ", ".join(sorted(_CONTROLLER_VARIANT_MODULES))
        raise KeyError(
            f"Unsupported paper-result controller '{run.script}'. Known controllers: {known}"
        )
    module = import_module(module_name)
    return module.get_variant_config(run.variant)


def _result_filename(
    *,
    result_prefix: str,
    delta: float,
    timestamp: str,
    k_layers: int | None,
    uses_k_layer_suffix: bool,
) -> str:
    if uses_k_layer_suffix and k_layers is not None:
        return f"{result_prefix}_k_{k_layers}_delta_{delta}_{timestamp}.txt"
    return f"{result_prefix}_delta_{delta}_{timestamp}.txt"


def _build_result_glob(
    *,
    result_prefix: str,
    delta: float,
    k_layers: int | None,
    uses_k_layer_suffix: bool,
) -> str:
    if uses_k_layer_suffix and k_layers is not None:
        return f"{result_prefix}_k_{k_layers}_delta_{delta}_*.txt"
    return f"{result_prefix}_delta_{delta}_*.txt"


def _sorted_artifact_paths(paths: list[Path]) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: (_parse_timestamp(path) or "", path.name),
        reverse=True,
    )


def _candidate_log_paths(log_dir: Path, script: str) -> list[Path]:
    prefixes = _CONTROLLER_LOG_PREFIXES.get(script, ())
    candidates: dict[str, Path] = {}
    for prefix in prefixes:
        for path in log_dir.glob(f"{prefix}_log_*.txt"):
            candidates[path.name] = path
    return _sorted_artifact_paths(list(candidates.values()))


def resolve_paper_run_artifacts(
    run: PaperPresetRun,
    *,
    repo_root: Path = REPO_ROOT,
) -> PaperRunArtifactMatch:
    config = _load_variant_config(run)
    delta = run.delta if run.delta is not None else config.default_delta
    uses_k_layer_suffix = run.script in _USES_K_LAYER_RESULT_NAME
    result_dir = repo_root / "sart" / "result" / "original_result"
    log_dir = repo_root / "sart" / "result" / "log"

    log_candidates = _candidate_log_paths(log_dir, run.script)
    for log_path in log_candidates:
        timestamp = _parse_timestamp(log_path)
        if timestamp is None:
            continue
        result_path = result_dir / _result_filename(
            result_prefix=config.result_prefix,
            delta=delta,
            timestamp=timestamp,
            k_layers=run.k_layers,
            uses_k_layer_suffix=uses_k_layer_suffix,
        )
        if result_path.exists():
            return PaperRunArtifactMatch(
                result_path=result_path,
                log_path=log_path,
                timestamp=timestamp,
                result_prefix=config.result_prefix,
                delta=delta,
                k_layers=run.k_layers,
            )

    result_candidates = _sorted_artifact_paths(
        list(
            result_dir.glob(
                _build_result_glob(
                    result_prefix=config.result_prefix,
                    delta=delta,
                    k_layers=run.k_layers,
                    uses_k_layer_suffix=uses_k_layer_suffix,
                )
            )
        )
    )
    if result_candidates and run.script in _RESULT_ONLY_SAFE_CONTROLLERS:
        result_path = result_candidates[0]
        timestamp = _parse_timestamp(result_path)
        log_path = None
        if timestamp is not None:
            matching_logs = [
                candidate
                for candidate in log_candidates
                if _parse_timestamp(candidate) == timestamp
            ]
            if matching_logs:
                log_path = matching_logs[0]
        return PaperRunArtifactMatch(
            result_path=result_path,
            log_path=log_path,
            timestamp=timestamp,
            result_prefix=config.result_prefix,
            delta=delta,
            k_layers=run.k_layers,
        )

    return PaperRunArtifactMatch(
        result_path=None,
        log_path=None,
        timestamp=None,
        result_prefix=config.result_prefix,
        delta=delta,
        k_layers=run.k_layers,
    )


def parse_result_file(result_path: Path) -> dict[str, object]:
    property_flags: dict[str, object] = {}
    property_times: dict[str, float] = {}
    summary: dict[str, object] = {}

    for raw_line in result_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        flag_match = _RESULT_FLAG_RE.match(line)
        if flag_match is not None:
            property_flags[flag_match.group("label")] = _parse_scalar(
                flag_match.group("value")
            )
            continue
        time_match = _RESULT_TIME_RE.match(line)
        if time_match is not None:
            property_times[time_match.group("label")] = float(time_match.group("value"))
            continue
        summary_match = _SUMMARY_RE.match(line)
        if summary_match is not None:
            summary[summary_match.group("key")] = _parse_scalar(
                summary_match.group("value")
            )

    summary["property_flags"] = property_flags
    summary["property_times"] = property_times
    return summary


def parse_log_file(log_path: Path) -> dict[str, object]:
    property_status: dict[str, str] = {}
    stage_outcomes: dict[str, str] = {}
    summary: dict[str, object] = {}

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stage_match = _STAGE_OUTCOME_RE.match(line)
        if stage_match is not None:
            stage_outcomes[stage_match.group("label")] = stage_match.group("outcome")
            continue
        status_match = _LOG_STATUS_RE.match(line)
        if status_match is not None:
            normalized = _LOG_STATUS_MAP.get(status_match.group("status"))
            if normalized is not None:
                property_status[status_match.group("label")] = normalized
                continue
        summary_match = _SUMMARY_RE.match(line)
        if summary_match is not None:
            summary[summary_match.group("key")] = _parse_scalar(
                summary_match.group("value")
            )

    summary["property_status"] = property_status
    summary["stage_outcomes"] = stage_outcomes
    return summary


def summarize_paper_run(
    run: PaperPresetRun,
    *,
    repo_root: Path = REPO_ROOT,
) -> PaperRunSummary:
    artifact_match = resolve_paper_run_artifacts(run, repo_root=repo_root)
    config = _load_variant_config(run)

    result_summary = (
        parse_result_file(artifact_match.result_path)
        if artifact_match.result_path is not None
        else {}
    )
    log_summary = (
        parse_log_file(artifact_match.log_path)
        if artifact_match.log_path is not None
        else {}
    )

    property_flags = dict(result_summary.get("property_flags", {}))
    property_times = dict(result_summary.get("property_times", {}))
    property_status = dict(log_summary.get("property_status", {}))
    stage_outcomes = dict(log_summary.get("stage_outcomes", {}))

    for label, flag in property_flags.items():
        if label in property_status:
            continue
        property_status[label] = "verified" if _is_verified_flag(flag) else "unresolved"

    verified = sum(status == "verified" for status in property_status.values())
    timeout = sum(status == "timeout" for status in property_status.values())
    unknown = sum(status == "unknown" for status in property_status.values())
    unsafe = sum(status == "unsafe" for status in property_status.values())
    unverified = sum(status == "unverified" for status in property_status.values())
    unresolved = sum(
        status in {"continuing", "unresolved"} for status in property_status.values()
    )

    metrics_source = result_summary if result_summary else log_summary
    number_sum = metrics_source.get("number_sum")
    time_sum = metrics_source.get("time_sum")
    time_average = metrics_source.get("time_average")
    time_max = metrics_source.get("time_max")

    stage_outcome_counts: dict[str, int] = {}
    stage_outcome_average_times: dict[str, float] = {}
    for outcome in stage_outcomes.values():
        stage_outcome_counts[outcome] = stage_outcome_counts.get(outcome, 0) + 1

    for outcome in sorted(stage_outcome_counts):
        times = [
            property_times[label]
            for label, recorded_outcome in stage_outcomes.items()
            if recorded_outcome == outcome and label in property_times
        ]
        if times:
            stage_outcome_average_times[outcome] = sum(times) / len(times)

    return PaperRunSummary(
        run=run,
        result_prefix=artifact_match.result_prefix,
        delta=artifact_match.delta,
        k_layers=artifact_match.k_layers,
        result_path=artifact_match.result_path,
        log_path=artifact_match.log_path,
        timestamp=artifact_match.timestamp,
        verified=verified,
        timeout=timeout,
        unknown=unknown,
        unsafe=unsafe,
        unverified=unverified,
        unresolved=unresolved,
        property_count=len(property_status),
        expected_property_count=config.report_total(),
        number_sum=number_sum,
        time_sum=None if time_sum is None else float(time_sum),
        time_average=None if time_average is None else float(time_average),
        time_max=None if time_max is None else float(time_max),
        stage_outcome_counts=stage_outcome_counts,
        stage_outcome_average_times=stage_outcome_average_times,
    )


def summarize_paper_preset(
    preset: PaperPreset,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[PaperRunSummary, ...]:
    return tuple(
        summarize_paper_run(run, repo_root=repo_root)
        for run in preset.runs
    )


def _flatten_stage_metrics(summary: PaperRunSummary) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for outcome, count in summary.stage_outcome_counts.items():
        flattened[outcome] = count
    for outcome, average in summary.stage_outcome_average_times.items():
        flattened[f"{outcome}_time_average"] = average
    if summary.expected_property_count:
        for outcome, count in summary.stage_outcome_counts.items():
            flattened[f"{outcome}_pct"] = round(
                count / summary.expected_property_count * 100.0,
                1,
            )
    return flattened


def build_paper_preset_summary_payload(
    preset: PaperPreset,
    summaries: tuple[PaperRunSummary, ...],
) -> dict[str, object]:
    aggregate = {
        "verified": sum(summary.verified for summary in summaries),
        "timeout": sum(summary.timeout for summary in summaries),
        "unknown": sum(summary.unknown for summary in summaries),
        "unsafe": sum(summary.unsafe for summary in summaries),
        "unverified": sum(summary.unverified for summary in summaries),
        "unresolved": sum(summary.unresolved for summary in summaries),
    }

    runs = []
    for summary in summaries:
        run_payload: dict[str, object] = {
            "label": summary.run.label,
            "script": summary.run.script,
            "variant": summary.run.variant,
            "delta": summary.delta,
            "k_layers": summary.k_layers,
            "mip_time_limit": summary.run.mip_time_limit,
            "verified": summary.verified,
            "timeout": summary.timeout,
            "unknown": summary.unknown,
            "unsafe": summary.unsafe,
            "unverified": summary.unverified,
            "unresolved": summary.unresolved,
            "properties": summary.property_count,
            "expected_properties": summary.expected_property_count,
            "number_sum": summary.number_sum,
            "time_sum": summary.time_sum,
            "time_average": summary.time_average,
            "time_max": summary.time_max,
            "result": None
            if summary.result_path is None
            else summary.result_path.relative_to(REPO_ROOT).as_posix(),
            "log": None
            if summary.log_path is None
            else summary.log_path.relative_to(REPO_ROOT).as_posix(),
        }
        run_payload.update(_flatten_stage_metrics(summary))
        runs.append(run_payload)

    payload: dict[str, object] = {
        "name": preset.name,
        "table": preset.table,
        "section": preset.section,
        "coverage": preset.coverage,
        "title": preset.title,
        "description": preset.description,
        "notes": list(preset.notes),
        "runs": runs,
        "aggregate": aggregate,
    }
    if preset.name == "table5_fallback_frequency" and not any(
        summary.stage_outcome_counts for summary in summaries
    ):
        payload["stage_metrics_missing_reason"] = (
            "local artifacts were generated before StageOutcome logging was added; "
            "rerun the preset to populate Table 5 fallback frequencies"
        )
    return payload


def render_paper_preset_summary_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def render_paper_preset_summary_tsv(payload: dict[str, object]) -> str:
    rows = list(payload.get("runs", []))
    aggregate = dict(payload.get("aggregate", {}))
    aggregate["kind"] = "aggregate"
    rows_with_aggregate = [
        {"kind": "run", **row}
        for row in rows
    ] + [aggregate]

    headers: list[str] = []
    for row in rows_with_aggregate:
        for key in row:
            if key not in headers:
                headers.append(key)

    def _normalize(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    lines = ["\t".join(headers)]
    for row in rows_with_aggregate:
        lines.append("\t".join(_normalize(row.get(header)) for header in headers))
    return "\n".join(lines)


def export_paper_preset_summaries(
    presets: tuple[PaperPreset, ...],
    *,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    index_rows: list[dict[str, object]] = []

    for preset in presets:
        summaries = summarize_paper_preset(preset, repo_root=repo_root)
        payload = build_paper_preset_summary_payload(preset, summaries)

        json_path = output_dir / f"{preset.name}.json"
        tsv_path = output_dir / f"{preset.name}.tsv"
        json_path.write_text(
            render_paper_preset_summary_json(payload) + "\n",
            encoding="utf-8",
        )
        tsv_path.write_text(
            render_paper_preset_summary_tsv(payload) + "\n",
            encoding="utf-8",
        )
        written_paths.extend([json_path, tsv_path])
        index_rows.append(
            {
                "name": preset.name,
                "table": preset.table,
                "section": preset.section,
                "coverage": preset.coverage,
                "json": json_path.name,
                "tsv": tsv_path.name,
                "runs": len(payload.get("runs", [])),
            }
        )

    index_payload = {"presets": index_rows}
    index_json_path = output_dir / "index.json"
    index_tsv_path = output_dir / "index.tsv"
    index_json_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    headers = ["name", "table", "section", "coverage", "runs", "json", "tsv"]
    index_lines = ["\t".join(headers)]
    for row in index_rows:
        index_lines.append("\t".join(str(row.get(header, "")) for header in headers))
    index_tsv_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    written_paths.extend([index_json_path, index_tsv_path])
    return written_paths


def print_paper_preset_summary(
    preset: PaperPreset,
    *,
    repo_root: Path = REPO_ROOT,
    emit_line: Callable[[str], None] = print,
) -> tuple[PaperRunSummary, ...]:
    summaries = summarize_paper_preset(preset, repo_root=repo_root)
    emit_line(
        "\t".join(
            [
                f"paper_preset_summary={preset.name}",
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

    if not summaries:
        emit_line("runs=0")
        return summaries

    for summary in summaries:
        fields = [
            f"label={summary.run.label}",
            f"script={summary.run.script}",
            f"variant={summary.run.variant}",
            f"delta={summary.delta}",
        ]
        if summary.k_layers is not None:
            fields.append(f"k_layers={summary.k_layers}")
        if summary.run.mip_time_limit is not None:
            fields.append(f"mip_time_limit={summary.run.mip_time_limit}")
        fields.extend(
            [
                f"verified={summary.verified}",
                f"timeout={summary.timeout}",
                f"unknown={summary.unknown}",
                f"unsafe={summary.unsafe}",
                f"unverified={summary.unverified}",
                f"unresolved={summary.unresolved}",
                f"properties={summary.property_count}",
                f"expected_properties={summary.expected_property_count}",
            ]
        )
        if summary.number_sum is not None:
            fields.append(f"number_sum={summary.number_sum}")
        if summary.time_average is not None:
            fields.append(f"time_average={summary.time_average}")
        if summary.time_max is not None:
            fields.append(f"time_max={summary.time_max}")
        fields.append(
            "result="
            + (
                "missing"
                if summary.result_path is None
                else summary.result_path.relative_to(repo_root).as_posix()
            )
        )
        fields.append(
            "log="
            + (
                "missing"
                if summary.log_path is None
                else summary.log_path.relative_to(repo_root).as_posix()
            )
        )
        emit_line("\t".join(fields))
        if summary.stage_outcome_counts:
            stage_fields = ["stage_metrics", f"label={summary.run.label}"]
            for outcome in (
                "stage1_safe",
                "stage2_safe",
                "stage2_timeout",
                "stage2_unknown",
                "stage3_safe",
                "stage3_unsafe",
                "stage3_timeout",
            ):
                if outcome in summary.stage_outcome_counts:
                    stage_fields.append(f"{outcome}={summary.stage_outcome_counts[outcome]}")
                if outcome in summary.stage_outcome_average_times:
                    stage_fields.append(
                        f"{outcome}_time_average={summary.stage_outcome_average_times[outcome]}"
                    )
            emit_line("\t".join(stage_fields))
            if preset.name == "table5_fallback_frequency" and summary.expected_property_count:
                table5_fields = ["table5_metrics", f"label={summary.run.label}"]
                for outcome in (
                    "stage1_safe",
                    "stage2_safe",
                    "stage2_timeout",
                    "stage3_safe",
                    "stage3_unsafe",
                    "stage3_timeout",
                ):
                    if outcome in summary.stage_outcome_counts:
                        percentage = (
                            summary.stage_outcome_counts[outcome]
                            / summary.expected_property_count
                            * 100.0
                        )
                        table5_fields.append(f"{outcome}_pct={percentage:.1f}")
                    if outcome in summary.stage_outcome_average_times:
                        table5_fields.append(
                            f"{outcome}_time_average={summary.stage_outcome_average_times[outcome]}"
                        )
                emit_line("\t".join(table5_fields))

    total_verified = sum(summary.verified for summary in summaries)
    total_timeout = sum(summary.timeout for summary in summaries)
    total_unknown = sum(summary.unknown for summary in summaries)
    total_unsafe = sum(summary.unsafe for summary in summaries)
    total_unverified = sum(summary.unverified for summary in summaries)
    total_unresolved = sum(summary.unresolved for summary in summaries)
    if preset.name == "table5_fallback_frequency" and not any(
        summary.stage_outcome_counts for summary in summaries
    ):
        emit_line(
            "stage_metrics=missing\t"
            "reason=local artifacts were generated before StageOutcome logging was added; rerun the preset to populate Table 5 fallback frequencies"
        )
    emit_line(
        "\t".join(
            [
                "aggregate",
                f"verified={total_verified}",
                f"timeout={total_timeout}",
                f"unknown={total_unknown}",
                f"unsafe={total_unsafe}",
                f"unverified={total_unverified}",
                f"unresolved={total_unresolved}",
            ]
        )
    )
    return summaries
