def extract_filename_from_path(path):
    """
    Extract the filename stem from a path-like string.
    """
    last_slash_index = path.rfind('/')

    last_dot_txt_index = -4

    if last_slash_index != -1 and last_dot_txt_index != -1:
        filename = path[last_slash_index + 1:last_dot_txt_index]
    else:
        filename = path

    return filename


if __name__ == "__main__":
    path = "../../sart/mnist_properties/mnist_properties_10x80/mnist_property_0.txt"
    filename = extract_filename_from_path(path)
    print(filename)  # Expected: mnist_property_0
