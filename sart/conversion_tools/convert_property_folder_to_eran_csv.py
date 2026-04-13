"""Convert a folder of repository property files into one ERAN-style CSV."""

import argparse
import csv
import os
import re


def process_file(file_path):
    """Read one property file and return the converted CSV row."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    label_row_index = None
    for row_index, line in enumerate(lines):
        if len(line.strip().split()) > 1:
            label_row_index = row_index
            break

    if label_row_index is None:
        raise ValueError(f"{file_path} does not contain a row with more than one element.")

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


def process_folder(folder_path, output_csv):
    """Scan a folder and write every property file into one CSV file."""
    txt_files = sorted(
        [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".txt")],
        key=lambda file_name: int(re.search(r"\d+", file_name).group()),
    )

    all_rows = []
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        all_rows.append(process_file(file_path))

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in all_rows:
            csv_writer.writerow(row)

    print(f"Successfully wrote all converted rows to {output_csv}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a folder of repository property files into one ERAN-style CSV.",
    )
    parser.add_argument("input_dir", help="Directory containing input property files.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    return parser.parse_args()


def main():
    """Run the folder-level ERAN CSV conversion."""
    args = parse_args()
    process_folder(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
