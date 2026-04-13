"""Read `.nnet` files without import-time side effects."""

from __future__ import annotations

from pathlib import Path

__all__ = ["read_nnet_file"]


def read_nnet_file(file_path):
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file_handle:
        lines = file_handle.readlines()

    model_structure = [
        int(value)
        for value in lines[2].strip().split(",")
        if value.strip().isdigit()
    ]
    input_size = model_structure[0]
    hidden_sizes = model_structure[1:-1]
    output_size = model_structure[-1]

    weights = []
    network_biases = []
    index = 8

    for hidden_size in hidden_sizes:
        weight_lines = [
            list(map(float, line.strip().split(",")[:-1]))
            for line in lines[index : index + hidden_size]
            if line.strip()
        ]
        bias_lines = [
            list(map(float, line.strip().split(",")[:-1]))
            for line in lines[
                index + hidden_size : index + hidden_size + hidden_size
            ]
            if line.strip()
        ]
        weights.append(weight_lines)
        network_biases.append(bias_lines)
        index += hidden_size * 2

    output_weight_lines = [
        list(map(float, line.strip().split(",")[:-1]))
        for line in lines[index : index + output_size]
        if line.strip()
    ]
    output_bias_lines = [
        list(map(float, line.strip().split(",")[:-1]))
        for line in lines[index + output_size : index + output_size + output_size]
        if line.strip()
    ]
    weights.append(output_weight_lines)
    network_biases.append(output_bias_lines)

    return weights, network_biases, input_size, hidden_sizes, output_size


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Parse and print one .nnet file.")
    parser.add_argument("file", help="Path to the .nnet file.")
    args = parser.parse_args()
    pprint(read_nnet_file(args.file))
