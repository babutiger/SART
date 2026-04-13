"""Write sorted absolute VNNLIB paths from one folder into a CSV file."""

import argparse
import csv
import os


def extract_property_number(filename):
    """Extract the property id from a VNNLIB filename for sorting."""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")
    return int(parts[2])


def save_vnnlib_paths_to_csv(input_dir, output_csv_path):
    """Save sorted absolute VNNLIB file paths from a directory into a CSV file."""
    vnnlib_files = sorted(
        [
            os.path.abspath(os.path.join(input_dir, filename))
            for filename in os.listdir(input_dir)
            if filename.endswith(".vnnlib")
        ],
        key=lambda path: extract_property_number(os.path.basename(path)),
    )

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        for path in vnnlib_files:
            csv_writer.writerow([path])

    print(f"Successfully wrote {output_csv_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Write sorted absolute VNNLIB paths from one folder into a CSV file.",
    )
    parser.add_argument("input_dir", help="Directory containing VNNLIB files.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    return parser.parse_args()


def main():
    """Run the absolute-path VNNLIB CSV export."""
    args = parse_args()
    save_vnnlib_paths_to_csv(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
