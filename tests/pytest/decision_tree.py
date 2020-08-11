import sys
import pytest
from sklearn import datasets
import numpy as np

sys.path.append("../../decision-trees/python")

from decision_trees import DecisionNode

node = DecisionNode()

iris = datasets.load_iris()

X = iris["data"]
y = iris["target"]
y = (y == 0).astype(int)

x0 = X[:, 0]

node.fit(x0, y, indexes=np.ndarray([]))

print(node)
