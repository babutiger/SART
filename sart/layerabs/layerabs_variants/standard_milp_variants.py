from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class StandardMILPVariant:
    name: str
    model_path: str
    property_template: str
    result_prefix: str
    default_delta: float
    property_count: Optional[int] = None
    property_indices: Optional[Tuple[int, ...]] = None
    maximum_time_threshold: int = 2000
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


VARIANTS: dict[str, StandardMILPVariant] = {
    "mnist_10x80": StandardMILPVariant(
        name="mnist_10x80",
        model_path="../models/mnist_new_10x80/mnist_net_new_10x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_10x80/mnist_property_{index}.txt",
        result_prefix="mnist_new_10x80_sym_number_result",
        default_delta=0.015,
        property_indices=(5,),
        maximum_time_threshold=200000,
        average_divisor=1,
    ),
    "mnist_5x50": StandardMILPVariant(
        name="mnist_5x50",
        model_path="../models/mnist_new_5x50/mnist_net_new_5x50.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x50/mnist_property_{index}.txt",
        result_prefix="mnist_new_5x50_sym_number_result",
        default_delta=0.018,
        property_count=100,
    ),
    "mnist_5x80": StandardMILPVariant(
        name="mnist_5x80",
        model_path="../models/mnist_new_5x80/mnist_net_new_5x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x80/mnist_property_{index}.txt",
        result_prefix="mnist_new_5x80_sym_number_result",
        default_delta=0.019,
        property_count=100,
    ),
    "mnist_6x100": StandardMILPVariant(
        name="mnist_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.019,
        property_count=100,
    ),
    "mnist_9x100": StandardMILPVariant(
        name="mnist_9x100",
        model_path="../models/mnist_new_9x100/mnist_net_new_9x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_9x100_sym_number_result",
        default_delta=0.018,
        property_count=100,
    ),
    "mnist_9x200": StandardMILPVariant(
        name="mnist_9x200",
        model_path="../models/mnist_new_9x200/mnist_net_new_9x200.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x200/mnist_property_{index}.txt",
        result_prefix="mnist_new_9x200_sym_number_result",
        default_delta=0.018,
        property_count=100,
    ),
    "cifar10_5x50": StandardMILPVariant(
        name="cifar10_5x50",
        model_path="../models/cifar_new_5x50/cifar_net_new_5x50.nnet",
        property_template="../../sart/cifar_properties/cifar_properties_5x50/cifar_property_{index}.txt",
        result_prefix="cifar_net_new_5x50_sym_number_result",
        default_delta=0.01,
        property_count=100,
    ),
    "cifar10_6x80": StandardMILPVariant(
        name="cifar10_6x80",
        model_path="../models/cifar_new_6x80/cifar_net_new_6x80.nnet",
        property_template="../../sart/cifar_properties/cifar_properties_6x80/cifar_property_{index}.txt",
        result_prefix="cifar_net_new_6x80_sym_number_result",
        default_delta=0.004,
        property_count=100,
    ),
    "vnncomp_6x100": StandardMILPVariant(
        name="vnncomp_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_number_result",
        default_delta=0.019,
        property_indices=(5,),
        maximum_time_threshold=20000,
        average_divisor=1,
    ),
    "paper_table3_vnncomp_6x100_hard_cases": StandardMILPVariant(
        name="paper_table3_vnncomp_6x100_hard_cases",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_table3_hard_cases_number_result",
        default_delta=0.019,
        property_indices=(4, 6, 5, 42, 44, 46, 83),
        maximum_time_threshold=20000,
        average_divisor=7,
    ),
    "vnncomp_9x100": StandardMILPVariant(
        name="vnncomp_9x100",
        model_path="../models/VNNCOMP_ERAN/mnist_9_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_9x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_9x100_number_result",
        default_delta=0.018,
        property_count=100,
    ),
}


def get_variant_config(name: str) -> StandardMILPVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown standard MILP variant '{name}'. Known variants: {known}"
        ) from exc
