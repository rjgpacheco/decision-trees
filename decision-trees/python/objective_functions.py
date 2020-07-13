"""
Implement decision functions for decision trees
"""

import numpy as np
from utils import is_left, is_right, to_array


def gini_impurity(y):
    categories, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()  # Class probabilities
    return 1 - np.multiply(p, p).sum()  # Final Gini calculation


def gini_impurity_split(x, y, boundary):
    y = to_array(y)
    left = is_left(x, boundary)
    right = is_right(x, boundary)

    gini_l = gini_impurity(y[left])
    gini_r = gini_impurity(y[right])

    gini_split = gini_l * len(left) + gini_r * len(right)
    gini_split = gini_split / len(y)
    return gini_split


def information_gain():
    return None
