from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperPresetRun:
    script: str
    label: str
    variant: str | None = None
    delta: float | None = None
    k_layers: int | None = None
    mip_time_limit: float | None = None

    def script_args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self.variant is not None:
            args.extend(["--variant", self.variant])
        if self.delta is not None:
            args.extend(["--delta", str(self.delta)])
        if self.k_layers is not None:
            args.extend(["--k-layers", str(self.k_layers)])
        if self.mip_time_limit is not None:
            args.extend(["--mip-time-limit", str(self.mip_time_limit)])
        return tuple(args)


@dataclass(frozen=True)
class PaperPreset:
    name: str
    table: str
    section: str
    title: str
    coverage: str
    description: str
    runs: tuple[PaperPresetRun, ...] = ()
    notes: tuple[str, ...] = ()


_MNIST_MAIN_VARIANTS = (
    "mnist_5x50",
    "mnist_5x80",
    "mnist_6x100",
    "mnist_9x100",
    "mnist_10x80",
    "mnist_9x200",
)

_NO_ABSTRACTION_VARIANTS = (
    "mnist_5x50",
    "mnist_5x80",
    "mnist_6x100",
    "mnist_9x100",
    "mnist_10x80",
    "mnist_9x200",
    "vnncomp_6x100",
    "vnncomp_9x100",
    "cifar10_5x50",
    "cifar10_6x80",
)

_VNNCOMP_COMPLETE_VARIANTS = (
    "vnncomp_6x100",
    "vnncomp_9x100",
)

_VNNCOMP_PAPER_NAME_NOTE = (
    "The paper's VNN-COMP ERAN 5x100 and 8x100 correspond to the code variants "
    "`vnncomp_6x100` and `vnncomp_9x100`, because the code counts the output "
    "layer in the architecture label."
)


def _controller_runs(
    *,
    script: str,
    variants: tuple[str, ...],
    label_prefix: str,
    k_layers: int | None = None,
    mip_time_limit: float | None = None,
) -> tuple[PaperPresetRun, ...]:
    return tuple(
        PaperPresetRun(
            script=script,
            label=f"{label_prefix} on {variant}",
            variant=variant,
            k_layers=k_layers,
            mip_time_limit=mip_time_limit,
        )
        for variant in variants
    )


def _incomplete_k_sweep_runs(
    *,
    variants: tuple[str, ...],
    ks: tuple[int, ...],
) -> tuple[PaperPresetRun, ...]:
    runs: list[PaperPresetRun] = []
    for variant in variants:
        for k_layers in ks:
            runs.append(
                PaperPresetRun(
                    script="LayerABS_incomplete_layerabs",
                    label=f"Incomplete-LayerABS on {variant} with k={k_layers}",
                    variant=variant,
                    k_layers=k_layers,
                )
            )
    return tuple(runs)


PRESET_ALIASES = {
    "table2": "table2_no_abstraction",
    "table3": "table3_vnncomp_hard_cases",
    "table4": "table4_layerabs_complete",
    "table5": "table5_fallback_frequency",
    "table6": "table6_incomplete_mnist",
    "table7": "table7_sart_vs_milp_ablation",
    "table8": "table8_vnncomp_complete",
    "table9": "table9_incomplete_vnncomp_k3",
    "table10": "table10_incomplete_vnncomp_k_sweep",
    "table11": "table11_incomplete_vnncomp_ldsa_k3",
    "table12": "table12_incomplete_vnncomp_ldsa_k_sweep",
}


PAPER_PRESETS: tuple[PaperPreset, ...] = (
    PaperPreset(
        name="table2_no_abstraction",
        table="Table 2",
        section="6.1",
        title="Pure SART vs. Standard MILP",
        coverage="partial",
        description=(
            "No-abstraction encoding comparison across MNIST, VNN-COMP, and CIFAR-10."
        ),
        runs=
        _controller_runs(
            script="LayerABS_puresart",
            variants=_NO_ABSTRACTION_VARIANTS,
            label_prefix="PureSART",
        )
        + _controller_runs(
            script="LayerABS_standard_milp",
            variants=_NO_ABSTRACTION_VARIANTS,
            label_prefix="Standard MILP",
        ),
        notes=(
            "Reruns the repo's current no-abstraction controllers only; external baselines are intentionally out of scope.",
            "Some historical no-abstraction variant tables still carry hard-case-specific property subsets or timeout overrides, so this preset is a launcher-level approximation rather than a byte-for-byte table reproducer.",
            _VNNCOMP_PAPER_NAME_NOTE,
        ),
    ),
    PaperPreset(
        name="table3_vnncomp_hard_cases",
        table="Table 3",
        section="6.2",
        title="VNN-COMP near-timeout hard cases",
        coverage="supported",
        description=(
            "Near-timeout VNN-COMP 5x100 hard-case slice for PureSART vs. Standard MILP."
        ),
        runs=(
            PaperPresetRun(
                script="LayerABS_puresart",
                label="PureSART on the Table 3 VNN-COMP hard-case slice",
                variant="paper_table3_vnncomp_6x100_hard_cases",
            ),
            PaperPresetRun(
                script="LayerABS_standard_milp",
                label="Standard MILP on the Table 3 VNN-COMP hard-case slice",
                variant="paper_table3_vnncomp_6x100_hard_cases",
            ),
        ),
        notes=(
            "This preset encodes the paper's hard-case property IDs 4, 6, 5, 42, 44, 46, and 83 with a 20000s timeout.",
            _VNNCOMP_PAPER_NAME_NOTE,
        ),
    ),
    PaperPreset(
        name="table4_layerabs_complete",
        table="Table 4",
        section="6.2",
        title="LayerABS verification on different MNIST models",
        coverage="supported",
        description=(
            "Abstraction-enabled LayerABS(SART) complete verification across the main MNIST models."
        ),
        runs=_controller_runs(
            script="LayerABS_abstract_sart",
            variants=_MNIST_MAIN_VARIANTS,
            label_prefix="LayerABS(SART)",
        ),
    ),
    PaperPreset(
        name="table5_fallback_frequency",
        table="Table 5",
        section="6.2",
        title="LayerABS fallback frequency",
        coverage="supported",
        description=(
            "Same LayerABS(SART) MNIST runs used to report Stage 1/2/3 fallback frequency."
        ),
        runs=_controller_runs(
            script="LayerABS_abstract_sart",
            variants=_MNIST_MAIN_VARIANTS,
            label_prefix="LayerABS(SART)",
        ),
        notes=(
            "This preset reproduces the underlying verification runs; `--summarize-paper-preset table5` now aggregates stage-level fallback markers when the logs contain `StageOutcome:` entries.",
        ),
    ),
    PaperPreset(
        name="table6_incomplete_mnist",
        table="Table 6",
        section="6.3",
        title="Incomplete-LayerABS on MNIST",
        coverage="partial",
        description=(
            "MNIST incomplete-verification runs for Incomplete-LayerABS with k=3."
        ),
        runs=_controller_runs(
            script="LayerABS_incomplete_layerabs",
            variants=_MNIST_MAIN_VARIANTS,
            label_prefix="Incomplete-LayerABS",
            k_layers=3,
            mip_time_limit=30,
        ),
        notes=(
            "This preset covers the Incomplete-LayerABS side only; PRIMA remains intentionally outside the unified framework.",
            "This preset applies the paper's 30s per-solve Gurobi fairness limit through `--mip-time-limit 30`.",
        ),
    ),
    PaperPreset(
        name="table7_sart_vs_milp_ablation",
        table="Table 7",
        section="6.4",
        title="SART vs. MILP encoding inside LayerABS",
        coverage="supported",
        description=(
            "LayerABS(SART) vs. LayerABS(MILP) across the main MNIST complete-verification models."
        ),
        runs=
        _controller_runs(
            script="LayerABS_abstract_sart",
            variants=_MNIST_MAIN_VARIANTS,
            label_prefix="LayerABS(SART)",
        )
        + _controller_runs(
            script="LayerABS_abstract_milp",
            variants=_MNIST_MAIN_VARIANTS,
            label_prefix="LayerABS(MILP)",
        ),
    ),
    PaperPreset(
        name="table8_vnncomp_complete",
        table="Table 8",
        section="C.1",
        title="LayerABS on VNN-COMP ERAN",
        coverage="supported",
        description=(
            "Abstraction-enabled LayerABS(SART) complete verification on the VNN-COMP ERAN models."
        ),
        runs=_controller_runs(
            script="LayerABS_abstract_sart",
            variants=_VNNCOMP_COMPLETE_VARIANTS,
            label_prefix="LayerABS(SART)",
        ),
        notes=(_VNNCOMP_PAPER_NAME_NOTE,),
    ),
    PaperPreset(
        name="table9_incomplete_vnncomp_k3",
        table="Table 9",
        section="C.2",
        title="Incomplete-LayerABS vs. PRIMA on VNN-COMP ERAN",
        coverage="partial",
        description=(
            "Incomplete-LayerABS internal runs on VNN-COMP ERAN with k=3."
        ),
        runs=_controller_runs(
            script="LayerABS_incomplete_layerabs",
            variants=_VNNCOMP_COMPLETE_VARIANTS,
            label_prefix="Incomplete-LayerABS",
            k_layers=3,
            mip_time_limit=30,
        ),
        notes=(
            "This preset covers the Incomplete-LayerABS side only; PRIMA remains intentionally outside the unified framework.",
            "This preset applies the paper's 30s per-solve Gurobi fairness limit through `--mip-time-limit 30`.",
            _VNNCOMP_PAPER_NAME_NOTE,
        ),
    ),
    PaperPreset(
        name="table10_incomplete_vnncomp_k_sweep",
        table="Table 10",
        section="C.2",
        title="Incomplete-LayerABS k sweep on VNN-COMP ERAN",
        coverage="supported",
        description=(
            "Hyperparameter sweep for Incomplete-LayerABS on VNN-COMP ERAN with k in {1,2,3,4,5}."
        ),
        runs=_incomplete_k_sweep_runs(
            variants=_VNNCOMP_COMPLETE_VARIANTS,
            ks=(1, 2, 3, 4, 5),
        ),
        notes=(_VNNCOMP_PAPER_NAME_NOTE,),
    ),
    PaperPreset(
        name="table11_incomplete_vnncomp_ldsa_k3",
        table="Table 11",
        section="C.3",
        title="Incomplete-LayerABS vs. PRIMA on VNN-COMP ERAN under LDSA",
        coverage="unsupported",
        description=(
            "LDSA-mode comparison on VNN-COMP ERAN with k=3."
        ),
        notes=(
            "The current public controller path exposes the paper's single-interaction setting, not the LDSA alternation mode used in Table 11.",
            _VNNCOMP_PAPER_NAME_NOTE,
        ),
    ),
    PaperPreset(
        name="table12_incomplete_vnncomp_ldsa_k_sweep",
        table="Table 12",
        section="C.3",
        title="Incomplete-LayerABS LDSA k sweep on VNN-COMP ERAN",
        coverage="unsupported",
        description=(
            "LDSA-mode k sweep for Incomplete-LayerABS on VNN-COMP ERAN."
        ),
        notes=(
            "The current public controller path exposes the paper's single-interaction setting, not the LDSA alternation mode used in Table 12.",
            _VNNCOMP_PAPER_NAME_NOTE,
        ),
    ),
)


def build_paper_presets() -> tuple[PaperPreset, ...]:
    return PAPER_PRESETS


def resolve_paper_preset_name(name: str) -> str:
    normalized = name.lower().strip()
    return PRESET_ALIASES.get(normalized, normalized)


def resolve_paper_preset(name: str) -> PaperPreset:
    normalized = resolve_paper_preset_name(name)
    for preset in PAPER_PRESETS:
        if preset.name == normalized:
            return preset
    known = ", ".join(preset.name for preset in PAPER_PRESETS)
    raise KeyError(f"Unknown paper preset '{name}'. Known presets: {known}")


def filter_paper_presets(
    presets: tuple[PaperPreset, ...],
    text_filter: str,
) -> tuple[PaperPreset, ...]:
    if not text_filter:
        return presets
    text_filter = text_filter.lower()
    return tuple(
        preset
        for preset in presets
        if text_filter in preset.name
        or text_filter in preset.table.lower()
        or text_filter in preset.section.lower()
        or text_filter in preset.title.lower()
        or text_filter in preset.coverage.lower()
        or text_filter in preset.description.lower()
        or any(text_filter in note.lower() for note in preset.notes)
    )
