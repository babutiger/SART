from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardMILPStatsVariant:
    name: str
    model_path: str
    property_template: str
    result_prefix: str
    default_delta: float
    range_start: int
    range_stop: int
    reported_amount: int
    maximum_time_threshold: int = 2000

    def build_property_list(self) -> list[str]:
        return [
            self.property_template.format(index=index)
            for index in range(self.range_start, self.range_stop)
        ]


VARIANTS: dict[str, StandardMILPStatsVariant] = {
    "vnncomp_6x100": StandardMILPStatsVariant(
        name="vnncomp_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_number_result",
        default_delta=0.019,
        range_start=18,
        range_stop=19,
        reported_amount=19,
    ),
    "cifar10_6x80": StandardMILPStatsVariant(
        name="cifar10_6x80",
        model_path="../models/cifar_new_6x80/cifar_net_new_6x80.nnet",
        property_template="../../sart/cifar_properties/cifar_properties_6x80/cifar_property_{index}.txt",
        result_prefix="cifar_net_new_6x80_sym_number_result",
        default_delta=0.004,
        range_start=0,
        range_stop=73,
        reported_amount=73,
    ),
    "mnist_6x100": StandardMILPStatsVariant(
        name="mnist_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.019,
        range_start=80,
        range_stop=81,
        reported_amount=81,
    ),
}

LEGACY_VARIANT_ALIASES = {
    "mnist_6x100_num_time_count": "mnist_6x100",
    "cifar10_6x80_num_time_count": "cifar10_6x80",
    "vnncomp_6x100_num_time_count": "vnncomp_6x100",
}


def get_variant_config(name: str) -> StandardMILPStatsVariant:
    normalized_name = LEGACY_VARIANT_ALIASES.get(name, name)
    try:
        return VARIANTS[normalized_name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown standard MILP stats variant '{name}'. Known variants: {known}"
        ) from exc
