import json  # NOTE: this is here just so that the str() method is pretty
import warnings

import numpy as np

from utils import is_left, is_right
from uuid import uuid4

MIN_NODE_INSTANCES = 1
NODE_MAX_DEPTH = 3

import logging

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename="decision_trees.log", level=logging.INFO, format=FORMAT)


class DecisionNode:
    """
    Decision node for a numerical feature
    """

    logging.getLogger(__name__)

    def __init__(
        self,
        boundary: float = None,
        node_id: str = None,
        decision: float = None,
        indexes=None,
        left=None,
        right=None,
        parent=None,
        depth=0,
    ):

        self.boundary = boundary
        self.decision = decision
        self.feature_index_global = None
        self.feature_index_local = None
        self.indexes = indexes  #  TODO: move this over to training utilities
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = depth
        self.type = "ordinal"
        self.node_id = uuid4().hex
        self.class_counts = {}
        self.class_scores = {}

        logging.info(f"{self.node_id} Initializing node ")

    def __repr__(self):
        # return "node"
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        dict_form = {
            "decision": self.decision,
            "class_scores": self.class_scores,
            "boundary": self.boundary,
            "depth": self.depth,
            "type": self.type,
            "node_id": self.node_id,
            "class_counts": self.class_counts,
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
        }
        return dict_form

    def from_dict(self, dictionary):
        if dictionary is None:
            return None
        new_node = DecisionNode()
        new_node.left = dictionary["left"]
        return new_node

    def is_leaf(self):
        """
        True if no child nodes are present.
        """
        # NOTE: this is ignoring the existence of a boundary
        return not (self.left or self.right)

    def is_left(self, x):
        return is_left(x, self.boundary)

    def is_right(self, x):
        return is_right(x, self.boundary)

    def traverse(self, x):
        """
        Traverse a decision node.
        """
        if self.is_leaf():
            return self.decision

        if self.is_left(x):
            return self.left.traverse(x)
        else:
            return self.right.traverse(x)

    def fit(self, x, y, recursive=False, indexes=None):

        logging.info(f"{self.node_id} Calling fit()")

        if x is None or y is None:
            raise ValueError("x and y must not be None")

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        categories, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()  # Class probabilities

        categories = [str(x) for x in categories]
        counts = [int(x) for x in counts]

        self.class_scores = dict(zip(categories, probabilities))
        self.class_counts = dict(zip(categories, counts))

        # Stopping conditions
        if len(categories) <= 1:
            logging.warning(f"Pure node, stopping training")
            self.decision = y.mean()
            return self

        if self.depth >= NODE_MAX_DEPTH:
            logging.warning(f"Max depth of {NODE_MAX_DEPTH} reached")
            return None

        if len(x) <= MIN_NODE_INSTANCES:
            logging.warning("Only one instance supplied to fit()")
            self.decision = y.mean()  # TODO: This is the "hello world" of decisions
            return self

        # Actual node training
        self.indexes = indexes

        logging.info(f"{self.node_id} self.class_counts={self.class_scores}")
        logging.info(f"{self.node_id} self.class_scores={self.class_counts}")

        # Set boundary and decision
        self.boundary = x.mean()  # TODO: This is the "hello world" of decisions
        self.decision = y.mean()  # TODO: This is the "hello world" of decisions

        if recursive:
            self.fit_children(x, y, recursive=recursive, indexes=indexes)

        return self

    def fit_children(self, x, y, recursive=False, indexes=None):
        index_left = self.is_left(x)
        index_right = self.is_right(x)

        self.left = DecisionNode(parent=self, depth=self.depth + 1,).fit(
            x=x[index_left], y=y[index_left], recursive=recursive, indexes=indexes
        )

        self.right = DecisionNode(parent=self, depth=self.depth + 1,).fit(
            x=x[index_right], y=y[index_right], recursive=recursive, indexes=indexes
        )
        return self

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
