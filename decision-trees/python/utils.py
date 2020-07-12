import numpy as np


def to_array(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def is_left(x, boundary):
    return (to_array(x) <= boundary).astype(int)


def is_right(x, boundary):
    return (to_array(x) > boundary).astype(int)
