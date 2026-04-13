from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from layerabs_naming import KNOWN_LAYERABS_FAMILIES, LEGACY_FAMILY_ALIASES


_ARCH_RE = re.compile(r"(?i)\d+x\d+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_BENCHMARK_TOKENS = ("mnist", "cifar10", "vnncomp")
_KNOWN_FAMILY_PREFIXES = tuple(
    sorted(KNOWN_LAYERABS_FAMILIES, key=len, reverse=True)
)
_FAMILY_PAPER_ROLES = {
    "abstract_sart": "layerabs_with_abstraction_sart",
    "incomplete_layerabs": "incomplete_layerabs",
    "abstract_milp": "layerabs_with_abstraction_milp",
    "puresart": "puresart_no_abstraction",
    "standard_milp": "standard_milp_no_abstraction",
    "abstract_sart_stats": "layerabs_with_abstraction_sart_stats",
    "abstract_milp_stats": "layerabs_with_abstraction_milp_stats",
    "puresart_stats": "puresart_stats",
    "standard_milp_stats": "standard_milp_stats",
    "abstract_sart_timelimit": "timelimit_branch",
}


@dataclass(frozen=True)
class LayerABSExperiment:
    path: Path
    experiment_id: str
    family: str
    paper_role: str
    content_group: str
    bucket: str
    benchmark: str
    network: str
    entrypoint: str
    status: str
    role: str
    canonical: bool


@dataclass(frozen=True)
class LayerABSFamily:
    family: str
    canonical_id: str
    canonical_path: Path
    controller_path: Path | None
    paper_role: str
    status: str
    benchmarks: tuple[str, ...]
    networks: tuple[str, ...]
    size: int


def _slug(text: str) -> str:
    return _NON_ALNUM_RE.sub("-", text.lower()).strip("-")


def _normalize_legacy_family(stem: str) -> str:
    name = stem.lower().replace("+", "_plus_")
    if name.startswith("layerabs_"):
        name = name[len("layerabs_") :]

    name = _ARCH_RE.sub("_", name)
    for token in ["mnist", "cifar10", "cifar", "vnncomp"]:
        name = name.replace(token, "_")

    family = re.sub(r"_+", "_", name).strip("_")
    family = family or "standalone"
    return LEGACY_FAMILY_ALIASES.get(family, family)


def _normalize_family(path: Path, base: Path) -> str:
    relative_path = path.relative_to(base)
    if len(relative_path.parts) >= 3 and relative_path.parts[0] == "family_wrappers":
        family = relative_path.parts[1].lower()
        return LEGACY_FAMILY_ALIASES.get(family, family)

    stem = path.stem
    name = stem[len("LayerABS_") :] if stem.startswith("LayerABS_") else stem
    normalized_name = name.lower().replace("+", "_plus_")
    for family in _KNOWN_FAMILY_PREFIXES:
        if normalized_name == family or normalized_name.startswith(f"{family}_"):
            return family

    return _normalize_legacy_family(stem)


def _infer_benchmark(stem: str) -> str:
    name = stem.lower()
    positions = {
        token: name.rfind(token)
        for token in _BENCHMARK_TOKENS
        if name.rfind(token) >= 0
    }
    if positions:
        return max(positions.items(), key=lambda item: item[1])[0]
    if "cifar" in name or name.startswith("layerabs_cifar_"):
        return "cifar10"
    return "mnist"


def _infer_network(stem: str) -> str:
    matches = _ARCH_RE.findall(stem)
    if not matches:
        return "unknown"
    return matches[-1].lower()


def _infer_status(stem: str) -> str:
    name = stem.lower()
    if name.endswith("_legacy_impl"):
        return "legacy"
    if "copy" in name:
        return "legacy"
    return "active"


def _infer_content_group(family: str, status: str) -> str:
    if status == "legacy":
        return "legacy"
    if "timelimit" in family:
        return "timelimit"
    if "stats" in family:
        return "measurement"
    if "eran" in family or family.startswith("vnncomp_"):
        return "eran_vnncomp"
    if family in {"abstract_milp", "puresart", "standard_milp"}:
        return "ablation"
    return "main_complete"


def _canonical_sort_key(base: Path, path: Path) -> tuple[int, int, int, int, str]:
    parent = path.relative_to(base).parent
    bucket_penalty = (
        0
        if parent in {Path("."), Path("default_profiles")}
        else 1
    )
    stem = path.stem.lower()
    benchmark_markers = sum(token in stem for token in ["cifar10", "vnncomp"])
    arch_count = len(_ARCH_RE.findall(path.stem))
    status_penalty = 1 if _infer_status(path.stem) != "active" else 0
    return (status_penalty, bucket_penalty, benchmark_markers, arch_count, stem)


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _extract_entrypoint(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return "unknown"

    for node in tree.body:
        if not isinstance(node, ast.If) or not _is_main_guard(node):
            continue

        for stmt in reversed(node.body):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Name):
                    return func.id
                return ast.unparse(func)
        return "main_guard_without_call"

    return "no_main_guard"


def _imports_layerabs_module(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.split(".")[-1].startswith("LayerABS_")
    )


def _is_thin_wrapper(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False

    if not any(_imports_layerabs_module(node) for node in tree.body):
        return False

    for node in tree.body:
        if _imports_layerabs_module(node):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If) and _is_main_guard(node):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        return False

    return True


def _infer_role(
    base: Path,
    path: Path,
    status: str,
    canonical: bool,
    family_size: int,
) -> str:
    if path.stem.lower().endswith("_legacy_impl"):
        return "archived_implementation"
    if status == "legacy":
        return "legacy_copy"
    relative_path = path.relative_to(base)
    if relative_path.parts and relative_path.parts[0] == "default_profiles":
        return "default_profile"
    if relative_path.parts and relative_path.parts[0] == "family_wrappers":
        return "thin_wrapper" if _is_thin_wrapper(path) else "family_variant"
    if _is_thin_wrapper(path):
        return "thin_wrapper"
    if family_size == 1:
        return "standalone_experiment"
    if canonical:
        return "canonical_implementation"
    return "family_variant"


def _infer_bucket(base: Path, path: Path) -> str:
    parent = path.relative_to(base).parent
    if parent == Path("."):
        return "root_entrypoints"
    return parent.as_posix()


def _infer_paper_role(family: str, status: str, role: str) -> str:
    if family in _FAMILY_PAPER_ROLES:
        return _FAMILY_PAPER_ROLES[family]
    if role == "archived_implementation":
        return "archived_historical_snapshot"
    if status == "legacy":
        return "historical_legacy_branch"
    return "family_experiment_branch"


def _is_family_controller(path: Path, base: Path) -> bool:
    relative_path = path.relative_to(base)
    if relative_path.parent != Path("."):
        return False
    stem = path.stem
    if not stem.startswith("LayerABS_"):
        return False
    normalized_name = stem[len("LayerABS_") :].lower().replace("+", "_plus_")
    return normalized_name in KNOWN_LAYERABS_FAMILIES


def is_layerabs_family_controller(
    path: Path,
    repo_root: Path | None = None,
    base: Path | None = None,
) -> bool:
    if base is None:
        if repo_root is None:
            raise ValueError("Either repo_root or base must be provided.")
        base = repo_root / "sart" / "layerabs"
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        return False
    return _is_family_controller(resolved_path, resolved_base)


def _infer_symbolic_code_base(catalog: list[LayerABSExperiment]) -> Path | None:
    for exp in catalog:
        for parent in exp.path.parents:
            if parent.name == "layerabs":
                return parent
    return None


def build_layerabs_catalog(repo_root: Path) -> list[LayerABSExperiment]:
    base = repo_root / "sart" / "layerabs"
    paths = sorted(
        path
        for path in base.rglob("LayerABS*.py")
        if "__pycache__" not in path.parts
        and not _is_family_controller(path, base)
    )

    raw_rows = []
    family_groups: dict[str, list[Path]] = defaultdict(list)

    for path in paths:
        family = _normalize_family(path, base)
        status = _infer_status(path.stem)
        family_groups[family].append(path)
        raw_rows.append(
            {
                "path": path,
                "family": family,
                "content_group": _infer_content_group(family, status),
                "bucket": _infer_bucket(base, path),
                "benchmark": _infer_benchmark(path.stem),
                "network": _infer_network(path.stem),
                "entrypoint": _extract_entrypoint(path),
                "status": status,
            }
        )

    canonical_by_family = {
        family: min(group, key=lambda candidate: _canonical_sort_key(base, candidate))
        for family, group in family_groups.items()
    }

    id_counts: dict[str, int] = defaultdict(int)
    experiments: list[LayerABSExperiment] = []
    for row in raw_rows:
        path = row["path"]
        family = row["family"]
        canonical = path == canonical_by_family[family]
        role = _infer_role(
            base,
            path,
            row["status"],
            canonical=canonical,
            family_size=len(family_groups[family]),
        )
        experiment_id = (
            f"layerabs/{family}/{row['benchmark']}/{row['network']}"
        )
        id_counts[experiment_id] += 1
        experiments.append(
            LayerABSExperiment(
                path=path,
                experiment_id=experiment_id,
                family=family,
                paper_role=_infer_paper_role(family, row["status"], role),
                content_group=row["content_group"],
                bucket=row["bucket"],
                benchmark=row["benchmark"],
                network=row["network"],
                entrypoint=row["entrypoint"],
                status=row["status"],
                role=role,
                canonical=canonical,
            )
        )

    if any(count > 1 for count in id_counts.values()):
        updated: list[LayerABSExperiment] = []
        seen: dict[str, int] = defaultdict(int)
        for exp in experiments:
            seen[exp.experiment_id] += 1
            experiment_id = exp.experiment_id
            if id_counts[experiment_id] > 1:
                experiment_id = f"{experiment_id}/{_slug(exp.path.stem)}"
            updated.append(
                LayerABSExperiment(
                    path=exp.path,
                    experiment_id=experiment_id,
                    family=exp.family,
                    paper_role=exp.paper_role,
                    content_group=exp.content_group,
                    bucket=exp.bucket,
                    benchmark=exp.benchmark,
                    network=exp.network,
                    entrypoint=exp.entrypoint,
                    status=exp.status,
                    role=exp.role,
                    canonical=exp.canonical,
                )
            )
        experiments = updated

    return sorted(
        experiments,
        key=lambda exp: (
            exp.family,
            exp.benchmark,
            exp.network,
            exp.path.name,
        ),
    )


def build_layerabs_families(
    catalog: list[LayerABSExperiment],
    canonical_catalog: list[LayerABSExperiment] | None = None,
) -> list[LayerABSFamily]:
    groups: dict[str, list[LayerABSExperiment]] = defaultdict(list)
    for exp in catalog:
        groups[exp.family].append(exp)

    if canonical_catalog is None:
        canonical_catalog = catalog
    canonical_by_family = {
        exp.family: exp
        for exp in canonical_catalog
        if exp.canonical
    }
    symbolic_code_base = _infer_symbolic_code_base(canonical_catalog) or _infer_symbolic_code_base(catalog)

    families: list[LayerABSFamily] = []
    status_order = {"active": 0, "legacy": 1}
    for family, experiments in groups.items():
        visible_experiments = [
            exp for exp in experiments if exp.role != "archived_implementation"
        ]
        if not visible_experiments:
            visible_experiments = experiments
        canonical = canonical_by_family.get(family)
        if canonical is None:
            canonical = next(
                (exp for exp in experiments if exp.canonical),
                experiments[0],
            )
        controller_path = None
        if symbolic_code_base is not None:
            candidate = symbolic_code_base / f"LayerABS_{family}.py"
            if candidate.exists():
                controller_path = candidate
        families.append(
            LayerABSFamily(
                family=family,
                canonical_id=canonical.experiment_id,
                canonical_path=canonical.path,
                controller_path=controller_path,
                paper_role=canonical.paper_role,
                status=canonical.status,
                benchmarks=tuple(
                    sorted({exp.benchmark for exp in visible_experiments})
                ),
                networks=tuple(
                    sorted({exp.network for exp in visible_experiments})
                ),
                size=len(visible_experiments),
            )
        )

    return sorted(
        families,
        key=lambda item: (
            status_order.get(item.status, 99),
            item.family,
        ),
    )
