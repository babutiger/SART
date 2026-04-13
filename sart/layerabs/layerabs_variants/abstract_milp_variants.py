from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AbstractMILPVariant:
    name: str
    network_module: str
    model_path: str
    property_template: str
    result_prefix: str
    default_delta: float
    property_count: int = 100
    maximum_time_threshold: int = 2000
    l_mip_num: int = 2
    average_divisor: Optional[int] = None
    show_running_average: bool = False

    def build_property_list(self) -> list[str]:
        return [
            self.property_template.format(index=index)
            for index in range(self.property_count)
        ]

    def report_total(self) -> int:
        return self.property_count

    def resolved_average_divisor(self) -> int:
        return self.average_divisor or self.property_count


VARIANTS: dict[str, AbstractMILPVariant] = {
    "mnist_5x50": AbstractMILPVariant(
        name="mnist_5x50",
        network_module="sart.verify.mnist_new_5x50.deeppoly_mnist_new_5x50",
        model_path="../models/mnist_new_5x50/mnist_net_new_5x50.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x50/mnist_property_{index}.txt",
        result_prefix="mnist_new_5x50_sym_number_result",
        default_delta=0.018,
    ),
    "mnist_10x80": AbstractMILPVariant(
        name="mnist_10x80",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_10x80/mnist_net_new_10x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_10x80/mnist_property_{index}.txt",
        result_prefix="mnist_new_10x80_sym_number_result",
        default_delta=0.015,
    ),
    "mnist_5x80": AbstractMILPVariant(
        name="mnist_5x80",
        network_module="sart.verify.mnist_new_5x80.deeppoly_mnist_new_5x80",
        model_path="../models/mnist_new_5x80/mnist_net_new_5x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x80/mnist_property_{index}.txt",
        result_prefix="mnist_new_5x80_sym_number_result",
        default_delta=0.019,
    ),
    "mnist_6x100": AbstractMILPVariant(
        name="mnist_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_6x100_sym_number_result",
        default_delta=0.019,
    ),
    "mnist_9x100": AbstractMILPVariant(
        name="mnist_9x100",
        network_module="sart.verify.mnist_new_9x100.deeppoly_mnist_new_9x100",
        model_path="../models/mnist_new_9x100/mnist_net_new_9x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x100/mnist_property_{index}.txt",
        result_prefix="mnist_new_9x100_sym_number_result",
        default_delta=0.018,
    ),
    "mnist_9x200": AbstractMILPVariant(
        name="mnist_9x200",
        network_module="sart.verify.mnist_new_9x200.deeppoly_mnist_new_9x200",
        model_path="../models/mnist_new_9x200/mnist_net_new_9x200.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x200/mnist_property_{index}.txt",
        result_prefix="mnist_new_9x200_sym_number_result",
        default_delta=0.018,
        l_mip_num=3,
    ),
    "vnncomp_6x100": AbstractMILPVariant(
        name="vnncomp_6x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_6_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_6x100_number_result",
        default_delta=0.019,
    ),
    "vnncomp_9x100": AbstractMILPVariant(
        name="vnncomp_9x100",
        network_module="sart.verify.mnist_new_6x100.deeppoly_mnist_new_6x100",
        model_path="../models/VNNCOMP_ERAN/mnist_9_100_nat_flat.nnet",
        property_template="../../sart/vnncomp_eran_properties/vnncomp_eran_mnist_properties_9x100/mnist_property_{index}.txt",
        result_prefix="vnncomp_eran_mnist_properties_9x100_number_result",
        default_delta=0.018,
    ),
}


def get_variant_config(name: str) -> AbstractMILPVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown abstraction-enabled LayerABS(MILP) variant '{name}'. Known variants: {known}"
        ) from exc
