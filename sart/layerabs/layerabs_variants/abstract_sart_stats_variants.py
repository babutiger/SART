from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AbstractSARTStatsVariant:
    name: str
    network_module: str
    model_path: str
    property_template: str
    result_prefix: str
    default_delta: float
    property_count: Optional[int] = None
    property_indices: Optional[Tuple[int, ...]] = None
    maximum_time_threshold: int = 2000
    l_mip_num: int = 2
    average_divisor: Optional[int] = None

    def build_property_list(self) -> list[str]:
        if self.property_indices is not None:
            indices = self.property_indices
        elif self.property_count is not None:
            indices = range(self.property_count)
        else:
            raise ValueError(f"Variant '{self.name}' is missing property indices/count")

        return [self.property_template.format(index=index) for index in indices]

    def report_total(self) -> int:
        if self.property_count is not None:
            return self.property_count
        if self.property_indices is not None:
            return len(self.property_indices)
        raise ValueError(f"Variant '{self.name}' is missing property indices/count")

    def resolved_average_divisor(self) -> int:
        return self.average_divisor or self.report_total()


VARIANTS: dict[str, AbstractSARTStatsVariant] = {
    "mnist_6x100": AbstractSARTStatsVariant(
        name="mnist_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.019,
        property_indices=(38,),
        average_divisor=39,
    ),
    "cifar10_6x80": AbstractSARTStatsVariant(
        name="cifar10_6x80",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.004,
        property_count=100,
    ),
    "vnncomp_6x100": AbstractSARTStatsVariant(
        name="vnncomp_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_number_result",
        default_delta=0.019,
        property_indices=(
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
            88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99,
        ),
        average_divisor=100,
    ),
}

LEGACY_VARIANT_ALIASES = {
    "mnist_6x100_num_time_count": "mnist_6x100",
    "cifar10_6x80_num_time_count": "cifar10_6x80",
    "vnncomp_6x100_num_time_count": "vnncomp_6x100",
}


def get_variant_config(name: str) -> AbstractSARTStatsVariant:
    normalized_name = LEGACY_VARIANT_ALIASES.get(name, name)
    try:
        return VARIANTS[normalized_name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown abstraction-enabled LayerABS(SART) stats variant '{name}'. Known variants: {known}"
        ) from exc
