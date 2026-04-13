"""Convert a folder of property files into Marabou-style text constraints."""

import argparse
import os


def process_file(input_file, output_file, epsilon):
    """Convert one repository property file into Marabou-style constraints."""
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file,
        "w",
        encoding="utf-8",
    ) as outfile:
        lines = infile.readlines()

        output_width = 0
        correct_label = -1

        for line in lines:
            elements = line.split()
            if len(elements) > 1:
                output_width += 1
                if correct_label == -1 and "-1" in elements:
                    correct_label = elements.index("-1")

        for input_index, line in enumerate(lines):
            elements = line.split()
            if len(elements) != 1:
                continue

            value = float(elements[0])
            lower_bound = max(0, value - epsilon)
            upper_bound = min(1, value + epsilon)
            outfile.write(f"x{input_index} >= {lower_bound}\n")
            outfile.write(f"x{input_index} <= {upper_bound}\n")

        if correct_label != -1:
            for output_index in range(output_width + 1):
                if output_index != correct_label:
                    outfile.write(f"+y{output_index} -y{correct_label} <= 0\n")


def process_folder(input_folder, output_folder, epsilon=0.015):
    """Convert every property file in a directory into Marabou-style text."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        process_file(input_file, output_file, epsilon)

    print(f"Successfully converted property files into {output_folder}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a folder of property files into Marabou-style text constraints.",
    )
    parser.add_argument("input_dir", help="Directory containing input property files.")
    parser.add_argument("output_dir", help="Directory for converted Marabou text files.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.015,
        help="Perturbation radius used for the input interval constraints.",
    )
    return parser.parse_args()


def main():
    """Run the folder-level Marabou text conversion."""
    args = parse_args()
    process_folder(args.input_dir, args.output_dir, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
