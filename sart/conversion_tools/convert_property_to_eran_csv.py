"""Convert one repository property file into an ERAN-style CSV row."""

import argparse
import csv


def convert_property_file(file_path):
    """Read one property file and return the flattened ERAN CSV row."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    label_row_index = None
    for row_index, line in enumerate(lines):
        if len(line.strip().split()) > 1:
            label_row_index = row_index
            break

    if label_row_index is None:
        raise ValueError(f"{file_path} does not contain a label row.")

    required_lines = label_row_index + 1
    if len(lines) < required_lines:
        raise ValueError(f"{file_path} does not contain the required {required_lines} lines.")

    data_matrix = [
        list(map(float, lines[row_index].strip().split()))
        for row_index in range(label_row_index)
    ]
    label_data = list(map(int, lines[label_row_index].strip().split()))
    label = label_data.index(-1)

    csv_row = [label]
    for input_row in data_matrix:
        csv_row.append(input_row[0])
    return csv_row


def write_csv_row(csv_row, output_csv_path):
    """Write one converted ERAN row to disk."""
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_row)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert one repository property file into an ERAN-style CSV row.",
    )
    parser.add_argument("input_file", help="Path to the input property file.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    parser.add_argument(
        "--print-row",
        action="store_true",
        help="Print the converted CSV row to stdout before writing the file.",
    )
    return parser.parse_args()


def main():
    """Run the single-file ERAN CSV conversion."""
    args = parse_args()
    csv_row = convert_property_file(args.input_file)
    if args.print_row:
        print(csv_row)
    write_csv_row(csv_row, args.output_csv)
    print(f"Successfully wrote data to {args.output_csv}")


if __name__ == "__main__":
    main()
