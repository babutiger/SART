from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AbstractMILPStatsVariant:
    name: str
    network_module: str
    model_path: str
    property_template: str
    result_prefix: str
    default_delta: float
    range_start: int
    range_stop: int
    reported_amount: int
    maximum_time_threshold: int = 2000
    l_mip_num: int = 2
    average_divisor: Optional[int] = None
    show_running_average: bool = False

    def build_property_list(self) -> list[str]:
        return [
            self.property_template.format(index=index)
            for index in range(self.range_start, self.range_stop)
        ]

    def report_total(self) -> int:
        return self.reported_amount

    def resolved_average_divisor(self) -> int:
        return self.average_divisor or self.reported_amount


VARIANTS: dict[str, AbstractMILPStatsVariant] = {
    "mnist_6x100": AbstractMILPStatsVariant(
        name="mnist_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.019,
        range_start=89,
        range_stop=90,
        reported_amount=90,
    ),
    "cifar10_6x80": AbstractMILPStatsVariant(
        name="cifar10_6x80",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/cifar_new_6x80/cifar_net_new_6x80.nnet",
        property_template="../../sart/cifar_properties/cifar_properties_6x80/cifar_property_{index}.txt",
        result_prefix="cifar_net_new_6x80_sym_number_result",
        default_delta=0.004,
        range_start=0,
        range_stop=100,
        reported_amount=100,
    ),
    "vnncomp_6x100": AbstractMILPStatsVariant(
        name="vnncomp_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_number_result",
        default_delta=0.019,
        range_start=18,
        range_stop=100,
        reported_amount=100,
    ),
}

LEGACY_VARIANT_ALIASES = {
    "mnist_6x100_num_time_count": "mnist_6x100",
    "cifar10_6x80_num_time_count": "cifar10_6x80",
    "vnncomp_6x100_num_time_count": "vnncomp_6x100",
}


def get_variant_config(name: str) -> AbstractMILPStatsVariant:
    normalized_name = LEGACY_VARIANT_ALIASES.get(name, name)
    try:
        return VARIANTS[normalized_name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown abstraction-enabled LayerABS(MILP) stats variant '{name}'. Known variants: {known}"
        ) from exc
