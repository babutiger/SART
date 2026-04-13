from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AbstractSARTVariant:
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
    recursion_limit: Optional[int] = None

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


VARIANTS: dict[str, AbstractSARTVariant] = {
    "mnist_10x80": AbstractSARTVariant(
        name="mnist_10x80",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_10x80/mnist_net_new_10x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_10x80/mnist_property_{index}.txt",
        result_prefix="mnist_new_10x80_sym_number_result",
        default_delta=0.015,
        property_count=100,
    ),
    "mnist_5x50": AbstractSARTVariant(
        name="mnist_5x50",
        network_module="sart.verify.mnist_new_5x50.deeppoly_mnist_new_5x50",
        model_path="../models/mnist_new_5x50/mnist_net_new_5x50.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x50/mnist_property_{index}.txt",
        result_prefix="mnist_new_5x50_sym_number_result",
        default_delta=0.018,
        property_count=100,
    ),
    "mnist_5x80": AbstractSARTVariant(
        name="mnist_5x80",
        network_module="sart.verify.mnist_new_5x80.deeppoly_mnist_new_5x80",
        model_path="../models/mnist_new_5x80/mnist_net_new_5x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x80/mnist_property_{index}.txt",
        result_prefix="mnist_new_5x80_sym_number_result",
        default_delta=0.019,
        property_count=100,
    ),
    "mnist_6x100": AbstractSARTVariant(
        name="mnist_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.019,
        property_count=100,
    ),
    "mnist_9x100": AbstractSARTVariant(
        name="mnist_9x100",
        network_module="sart.verify.mnist_new_9x100.deeppoly_mnist_new_9x100",
        model_path="../models/mnist_new_9x100/mnist_net_new_9x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_9x100_sym_number_result",
        default_delta=0.018,
        property_count=100,
    ),
    "mnist_9x200": AbstractSARTVariant(
        name="mnist_9x200",
        network_module="sart.verify.mnist_new_9x200.deeppoly_mnist_new_9x200",
        model_path="../models/mnist_new_9x200/mnist_net_new_9x200.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x200/mnist_property_{index}.txt",
        result_prefix="mnist_new_9x200_sym_number_result",
        default_delta=0.018,
        property_count=100,
        l_mip_num=3,
    ),
    "cifar10_5x50": AbstractSARTVariant(
        name="cifar10_5x50",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/cifar_new_5x50/cifar_net_new_5x50.nnet",
        property_template="../../sart/cifar_properties/cifar_properties_5x50/cifar_property_{index}.txt",
        result_prefix="cifar_net_new_5x50_sym_number_result",
        default_delta=0.010,
        property_count=100,
        recursion_limit=100000,
    ),
    "cifar10_6x80": AbstractSARTVariant(
        name="cifar10_6x80",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/cifar_new_6x80/cifar_net_new_6x80.nnet",
        property_template="../../sart/cifar_properties/cifar_properties_6x80/cifar_property_{index}.txt",
        result_prefix="cifar_net_new_6x80_sym_number_result",
        default_delta=0.004,
        property_count=100,
        recursion_limit=100000,
    ),
    "vnncomp_6x100": AbstractSARTVariant(
        name="vnncomp_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_number_result",
        default_delta=0.019,
        property_count=100,
    ),
    "vnncomp_6x200": AbstractSARTVariant(
        name="vnncomp_6x200",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_200_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x200/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x200_number_result",
        default_delta=0.019,
        property_count=100,
    ),
    "vnncomp_9x100": AbstractSARTVariant(
        name="vnncomp_9x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_9_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_9x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_9x100_number_result",
        default_delta=0.018,
        property_count=100,
    ),
}


def get_variant_config(name: str) -> AbstractSARTVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown abstraction-enabled LayerABS(SART) variant '{name}'. Known variants: {known}"
        ) from exc
