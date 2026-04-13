"""Property-file parsing helpers used by LayerABS experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["input_bound_pair", "read_property"]


def read_property(file_path: str | Path):
    """Parse a property file into input pixels, property weights, and biases."""
    input_pixel_list = []
    property_layer_weights = []
    property_layer_biases = []

    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            values = line.strip().split()
            if not values:
                continue

            if all(value.replace(".", "", 1).isdigit() for value in values):
                input_pixel_list.append(float(line))
            else:
                property_layer_weights.append(list(map(int, values[:-1])))
                property_layer_biases.append([int(values[-1])])

    return input_pixel_list, property_layer_weights, property_layer_biases


def input_bound_pair(input_pixels, delta):
    """Clamp per-pixel perturbation bounds to the [0, 1] input range."""
    property_lb_ub_list = []
    for pixel_value in input_pixels:
        lower_upper_pair = [pixel_value - delta, pixel_value + delta]
        if lower_upper_pair[0] < 0:
            lower_upper_pair[0] = 0
        if lower_upper_pair[1] > 1:
            lower_upper_pair[1] = 1
        property_lb_ub_list.append(lower_upper_pair)
    return property_lb_ub_list


def _main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a LayerABS property file.")
    parser.add_argument("property_file", help="Path to the property file to parse.")
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Optional perturbation size used to print clamped input bounds.",
    )
    args = parser.parse_args()

    input_pixel_list, property_layer_weights, property_layer_biases = read_property(
        args.property_file
    )
    print(f"inputs={input_pixel_list}")
    print(f"weights={property_layer_weights}")
    print(f"biases={property_layer_biases}")
    if args.delta is not None:
        print(f"bounds={input_bound_pair(input_pixel_list, args.delta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
