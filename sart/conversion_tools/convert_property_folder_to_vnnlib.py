"""Batch-convert repository property files into VNNLIB format."""

import argparse
import os


def batch_txt_to_vnnlib(input_dir, output_dir=None, epsilon=0.1):
    """Convert every property file in a directory into VNNLIB format."""
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        input_txt_path = os.path.join(input_dir, filename)
        output_filename = filename.replace(".txt", f"_eps_{epsilon}.vnnlib")
        output_vnnlib_path = os.path.join(output_dir, output_filename)

        with open(input_txt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        pixels = [float(value.strip()) for value in lines[:784]]
        classification_line = lines[784].strip().split()
        label = classification_line.index("-1")

        with open(output_vnnlib_path, "w", encoding="utf-8") as file:
            file.write(f"; Image label: {label}, Epsilon: {epsilon}\n\n")

            for input_index in range(784):
                file.write(f"(declare-const X_{input_index} Real)\n")

            file.write("\n")

            for output_index in range(10):
                file.write(f"(declare-const Y_{output_index} Real)\n")

            file.write("\n; Input constraints:\n")

            for input_index, pixel in enumerate(pixels):
                lower_bound = max(0, pixel - epsilon)
                upper_bound = min(1, pixel + epsilon)
                file.write(f"(assert (<= X_{input_index} {upper_bound}))\n")
                file.write(f"(assert (>= X_{input_index} {lower_bound}))\n\n")

            file.write("\n; Output constraints:\n")
            file.write("(assert (or\n")
            for output_index in range(10):
                if output_index != label:
                    file.write(f"    (and (>= Y_{output_index} Y_{label}))\n")
            file.write("))\n")

    print(f"Successfully converted property files into {output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch-convert repository property files into VNNLIB format.",
    )
    parser.add_argument("input_dir", help="Directory containing input property files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated VNNLIB files. Defaults to the input directory.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Perturbation radius used for the input interval constraints.",
    )
    return parser.parse_args()


def main():
    """Run the folder-level VNNLIB conversion."""
    args = parse_args()
    batch_txt_to_vnnlib(args.input_dir, output_dir=args.output_dir, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
