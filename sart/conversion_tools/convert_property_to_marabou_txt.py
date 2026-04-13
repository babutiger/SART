"""Convert one property file into a Marabou-style text constraint file."""

import argparse


def process_file(input_file, output_file, epsilon=0.026):
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

    print(f"Successfully wrote {output_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert one property file into a Marabou-style text constraint file.",
    )
    parser.add_argument("input_file", help="Path to the input property file.")
    parser.add_argument("output_file", help="Path to the output Marabou text file.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.026,
        help="Perturbation radius used for the input interval constraints.",
    )
    return parser.parse_args()


def main():
    """Run the single-file Marabou text conversion."""
    args = parse_args()
    process_file(args.input_file, args.output_file, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
