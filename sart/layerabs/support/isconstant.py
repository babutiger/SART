def is_constant(s):
    try:
        int(s)
        return True
    except ValueError:
        pass

    try:
        float(s)
        return True
    except ValueError:
        pass

    return False


if __name__== '__main__':
    print(is_constant("123"))  # Expected: True
    print(is_constant("123.456"))  # Expected: True
    print(is_constant("abc"))  # Expected: False
    print(is_constant("1e-3"))  # Expected: True
    print(is_constant("-42"))  # Expected: True
    print(is_constant(" "))  # Expected: False
    print(is_constant(""))  # Expected: False
