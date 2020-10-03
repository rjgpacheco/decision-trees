import sys
import pytest
from sklearn import datasets
import numpy as np

sys.path.append("../../decision-trees/python")

from decision_trees import DecisionNode
from decision_trees import DecisionTree

iris = datasets.load_iris()

X = iris["data"]
y = iris["target"]
y = (y == 0).astype(int)

node = DecisionNode()
node.fit(X, y, indexes=np.ndarray([]))