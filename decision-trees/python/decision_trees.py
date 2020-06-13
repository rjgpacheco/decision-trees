import json  # NOTE: this is here just so that the str() method is pretty
import numpy as np

MIN_NODE_INSTANCES = 1
NODE_MAX_DEPTH = 3


class DecisionNode:
    """
    Decision node for a numerical feature
    """

    def __init__(
        self,
        boundary: float = None,
        name: str = None,
        decision: float = None,
        indexes=[],
        left=None,
        right=None,
        parent=None,
        depth=0,
    ):

        self.boundary = boundary
        self.decision = decision
        self.name = name
        self.indexes = indexes
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = depth
        self.type = "ordinal"

    def __repr__(self):
        # return "node"
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        dict_form = {
            "name": self.name,
            "decision": self.decision,
            "boundary": self.boundary,
            "depth": self.depth,
            "type": self.type,
            "indexes": self.indexes.tolist(),
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
            #  "parent": None if self.parent is None else self.parent.to_dict(),
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

    def is_left(self, x):
        if not isinstance(x, np.ndarray):
            x = np.ndarray(x)
        return (x <= self.boundary).astype(int)

    def is_right(self, x):
        if not isinstance(x, np.ndarray):
            x = np.ndarray(x)
        return (x > self.boundary).astype(int)

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

    def fit_node(self, x, y, indexes):
        self.indexes = indexes
        if x is None or y is None:
            raise ValueError("x and y must not be None")

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        self.boundary = x.mean()  # TODO: This is the "hello world" of decisions
        self.decision = y.mean()  # TODO: This is the "hello world" of decisions

        if len(x) <= MIN_NODE_INSTANCES:
            raise Warning("Only one instance supplied to fit_node")
            self.decision = y.mean()  # TODO: This is the "hello world" of decisions
            return

        if self.depth >= NODE_MAX_DEPTH:
            raise Warning(f"Max depth of {NODE_MAX_DEPTH} reached")
            return

        index_left = self.is_left(x)
        index_right = self.is_right(x)

        decision_left = y[index_left].mean()
        decision_right = y[index_right].mean()

        self.left = DecisionNode(
            indexes=index_left,
            decision=decision_left,
            parent=self,
            depth=self.depth + 1,
        )

        self.right = DecisionNode(
            indexes=index_right,
            decision=decision_right,
            parent=self,
            depth=self.depth + 1,
        )

    @property
    def indexes(self):
        return self._indexes

    @indexes.setter
    def indexes(self, value):
        if isinstance(value, np.ndarray):
            self._indexes = value
        else:
            self._indexes = np.ndarray(value)

    @indexes.deleter
    def indexes(self):
        del self._indexes


class DecisionTree:
    """
    Decision tree for a numerical feature
    """

    def __init__(self):
        self.root = None

    def fit(self, X, y):
        self.root = DecisionNode()

        return self
