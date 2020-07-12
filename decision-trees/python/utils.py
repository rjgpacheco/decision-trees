import numpy as np


def is_left(x, boundary):
    if not isinstance(x, np.ndarray):
        x = np.ndarray(x)
    return (x <= boundary).astype(int)


def is_right(x, boundary):
    if not isinstance(x, np.ndarray):
        x = np.ndarray(x)
    return (x > boundary).astype(int)
