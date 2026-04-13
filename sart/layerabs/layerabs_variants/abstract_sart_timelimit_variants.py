from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AbstractSARTTimelimitVariant:
    name: str
    network_module: str
    model_path: str
    property_template: str
    default_delta: float
    property_count: Optional[int] = None
    property_indices: Optional[Tuple[int, ...]] = None
    maximum_time_threshold: int = 2000
    l_mip_num: int = 2
    average_divisor: Optional[int] = None
    show_running_average: bool = True

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


VARIANTS: dict[str, AbstractSARTTimelimitVariant] = {
    "mnist_10x80": AbstractSARTTimelimitVariant(
        name="mnist_10x80",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_10x80/mnist_net_new_10x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_10x80/mnist_property_{index}.txt",
        default_delta=0.015,
        property_count=100,
    ),
    "mnist_5x50": AbstractSARTTimelimitVariant(
        name="mnist_5x50",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_5x50/mnist_net_new_5x50.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x50/mnist_property_{index}.txt",
        default_delta=0.018,
        property_count=100,
    ),
    "mnist_5x80": AbstractSARTTimelimitVariant(
        name="mnist_5x80",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_5x80/mnist_net_new_5x80.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_5x80/mnist_property_{index}.txt",
        default_delta=0.019,
        property_count=100,
    ),
    "mnist_6x100": AbstractSARTTimelimitVariant(
        name="mnist_6x100",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_6x100/mnist_net_new_6x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_6x100/mnist_property_{index}.txt",
        default_delta=0.019,
        property_indices=tuple(range(69, 100)),
        average_divisor=100,
    ),
    "mnist_9x100": AbstractSARTTimelimitVariant(
        name="mnist_9x100",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_9x100/mnist_net_new_9x100.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x100/mnist_property_{index}.txt",
        default_delta=0.018,
        property_count=100,
    ),
    "mnist_9x200": AbstractSARTTimelimitVariant(
        name="mnist_9x200",
        network_module="sart.verify.mnist_new_10x80.deeppoly_mnist_new_10x80",
        model_path="../models/mnist_new_9x200/mnist_net_new_9x200.nnet",
        property_template="../../sart/mnist_properties/mnist_properties_9x200/mnist_property_{index}.txt",
        default_delta=0.018,
        property_count=100,
    ),
}

LEGACY_VARIANT_ALIASES = {
    "mnist_10x80_timelimit": "mnist_10x80",
    "mnist_5x50_timelimit": "mnist_5x50",
    "mnist_5x80_timelimit": "mnist_5x80",
    "mnist_6x100_timelimit": "mnist_6x100",
    "mnist_9x100_timelimit": "mnist_9x100",
    "mnist_9x200_timelimit": "mnist_9x200",
}


def get_variant_config(name: str) -> AbstractSARTTimelimitVariant:
    normalized_name = LEGACY_VARIANT_ALIASES.get(name, name)
    try:
        return VARIANTS[normalized_name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANTS))
        raise KeyError(
            f"Unknown abstraction-enabled LayerABS(SART) timelimit variant '{name}'. Known variants: {known}"
        ) from exc
