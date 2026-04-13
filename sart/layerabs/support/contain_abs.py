def contains_abs(s):
    return "Abs" in s


if __name__ == "__main__":
    test_string = "This is an Abs example."
    print(contains_abs(test_string))  # Expected: True

    test_string2 = "This is an example."
    print(contains_abs(test_string2))  # Expected: False
