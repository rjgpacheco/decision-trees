import sys
import pytest
from sklearn import datasets
import numpy as np

sys.path.append("../../decision-trees/python")

from decision_trees import DecisionNode
from decision_trees import DecisionTree
from data import MachineLearningData

node = DecisionNode()

iris = datasets.load_iris()

X = iris["data"]
y = iris["target"]
y = (y == 0).astype(int)


data = MachineLearningData(X=X, y=y)

# Tree

tree = DecisionTree().fit_from_data(data)
print(tree)