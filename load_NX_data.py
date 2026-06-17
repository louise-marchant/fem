import numpy as np

def load_NX_file(filepath):
    # Open with utf-8-sig to delete \ufeff, then load data
    with open(filepath, mode="r", encoding="utf-8-sig") as f:
        data = np.genfromtxt(
            f,
            delimiter=",",
            comments="%",      # Handle the % lines
            invalid_raise=False # Skip lines it fails to parse as numbers
        )

    # Filter out rows that are entirely NaN (like the original text lines)
    data = data[~np.isnan(data).any(axis=1)]

    iterations = data[:, 0]
    displacement = data[:, 1]

    return iterations, displacement