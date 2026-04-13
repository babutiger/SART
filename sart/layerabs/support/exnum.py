import re


def extract_numbers(input_str):
    match = re.match(r'x(\d+)_(\d+)', input_str)
    if not match:
        raise ValueError("Input string does not match the expected pattern")

    x = match.group(1)
    y = match.group(2)

    z = '1' if 'activate' in input_str or 'a' in input_str else '0'

    numlist = [int(x), int(y), int(z)]

    return numlist



if __name__== '__main__':
    print(extract_numbers('x1_0'))  # Expected: [1, 0, 0]
    print(extract_numbers('x1_0_activate'))  # Expected: [1, 0, 1]
    print(extract_numbers('x1_1'))  # Expected: [1, 1, 0]
    print(extract_numbers('x1_1_activate'))  # Expected: [1, 1, 1]
    print(extract_numbers('x1_783'))  # Expected: [1, 783, 0]
    print(extract_numbers('x100_18880'))  # Expected: [100, 18880, 0]
    print(extract_numbers('x10_180_a'))  # Expected: [10, 180, 1]
