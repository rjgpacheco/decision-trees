import json  # NOTE: this is here just so that the str() method is pretty

import numpy as np


class DecisionNode:
    """
    Decision node for a numerical feature
    """

    def __init__(
        self,
        boundary: float = None,
        name: str = None,
        decision: float = None,
        indexes=None,
        left=None,
        right=None,
        parent=None,
    ):
        self.set_boundary(boundary)
        self.set_decision(decision)
        self.set_name(name)
        self.set_indexes(indexes)
        self.set_left(left)
        self.set_right(right)
        self.set_parent(parent)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        dict_form = {
            "left": self.left.__dict__,
            "right": self.right.__dict__,
            "boundary": self.boundary,
            "name": self.name,
            "decision": self.decision,
        }
        return dict_form

    def to_dict(self):

        left = None if self.left is None else self.left.__dict__
        right = None if self.right is None else self.right.__dict__

        dict_form = {
            "left": left,
            "right": right,
            "boundary": self.boundary,
            "name": self.name,
            "decision": self.decision,
        }
        return dict_form

    def from_dict(self, dictionary):
        if dictionary is None:
            return None
        newNode = DecisionNode()
        newNode.left = dictionary["left"]

        return newNode

    def is_leaf(self):
        """
        True if no child nodes are present.
        """
        # NOTE: this is ignoring the existence of a boundary
        return not (self.left or self.right)

    def set_boundary(self, boundary):
        """
        Sets a decision boundary and initializes left and right child nodes.
        """

        if boundary is None:
            self.boundary = None
            self.right = None
            self.left = None
        else:
            self.boundary = boundary
            self.right = DecisionNode()
            self.left = DecisionNode()

    def set_decision(self, decision):
        self.decision = decision

    def set_name(self, name):
        self.name = name

    def get_indexes(self):
        return self.indexes

    def set_indexes(self, indexes):
        self.indexes = indexes

    def set_right(self, right):
        self.right = right

    def set_left(self, left):
        self.left = left

    def set_parent(self, parent):
        self.set_parent = parent

    def __is_left(self, x, boundary):
        return x <= self.boundary

    def __is_right(self, x, boundary):
        return not self.__is_left(x, boundary)

    def traverse(self, x):
        """
        Traverse a decision node.
        """
        if self.is_leaf():
            return decision

        if self.__is_left(x, self.boundary):
            return self.left.traverse(x)
        else:
            return self.right.traverse(x)


class DecisionTree:
    """
    Decision tree for a numerical feature
    """

    def __init__(self):
        self.root = None

    def __is_left(self, x, decision):
        return

    def __fit_node(self, x, y):
        boundary = x.mean()
        decision = y.mean()

        return None

    def fit(self, X, y):
        self.root = DecisionNode()

        return self
