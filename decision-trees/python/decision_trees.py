import json  # NOTE: this is here just so that the str() method is pretty
import warnings
import random

import numpy as np

from data import MachineLearningData
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
        decision_index=None,
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
        self.decision_index = decision_index

        logging.info(f"{self.node_id} Initializing node ")

    def __repr__(self):
        # return "node"
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        return {
            "boundary": self.boundary,
            "decision": self.decision,
            # "feature_index_global": self.None,
            # "feature_index_local": self.None,
            "indexes": []
            if self.indexes is None
            else self.indexes.astype(
                int
            ).tolist(),  #  TODO: move this over to training utilities,
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
            "depth": self.depth,
            "type": self.type,
            "node_id": self.node_id,
            "class_counts": self.class_counts,
            "class_scores": self.class_scores,
            "decision_index": self.decision_index,
        }

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

    def is_left(self, X):
        return is_left(
            self.get_feature_from_matrix(X, self.decision_index), self.boundary
        )

    def is_right(self, X):
        return is_right(
            self.get_feature_from_matrix(X, self.decision_index), self.boundary
        )

    def traverse(self, X):
        """
        Traverse a decision node.
        """

        x = self.get_feature_from_matrix(X, self.decision_index)

        if self.is_leaf():
            return self.decision

        if self.is_left(x):
            return self.left.traverse(X)
        else:
            return self.right.traverse(X)

    def fit(self, data: MachineLearningData, recursive=False, indexes=None):
        self._fit(
            X=data.X, y=data.y, schema=data.schema, recursive=recursive, indexes=indexes
        )
        return self

    def _fit(self, X, y, schema=None, recursive=False, indexes=None):

        logging.info(f"{self.node_id} Calling fit()")

        if X is None or y is None:
            raise ValueError("x and y must not be None")

        if self.get_n_instances(X) != len(y):
            raise ValueError("x and y must have same length")

        categories, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()  # Class probabilities

        categories = [str(_) for _ in categories]
        counts = [int(_) for _ in counts]

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

        if self.get_n_instances(X) <= MIN_NODE_INSTANCES:
            logging.warning("Only one instance supplied to fit()")
            self.decision = y.mean()  # TODO: This is the "hello world" of decisions
            return self

        # Actual node training
        self.indexes = indexes

        logging.info(f"{self.node_id} self.class_counts={self.class_scores}")
        logging.info(f"{self.node_id} self.class_scores={self.class_counts}")

        # Set boundary and decision
        self.decision_index, self.boundary, self.decision = self.get_best_split(X, y)

        if recursive:
            self.fit_children(X, y, recursive=recursive, indexes=indexes)

        return self

    def fit_children(self, X, y, recursive=False, indexes=None):
        index_left = self.is_left(X)
        index_right = self.is_right(X)

        self.left = DecisionNode(parent=self, depth=self.depth + 1,)._fit(
            X=self.get_matrix_by_index(X, index_left),
            y=y[index_left],
            recursive=recursive,
            indexes=index_left,
        )

        self.right = DecisionNode(parent=self, depth=self.depth + 1,)._fit(
            X=self.get_matrix_by_index(X, index_right),
            y=y[index_right],
            recursive=recursive,
            indexes=index_right,
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

    def get_feature_from_matrix(self, X, decision_feature_index):
        """
        Get feature of feature matrix. This funcion only exists to prevent future bugs.
        """
        return X[:, decision_feature_index]

    def get_n_features(self, X):
        """
        Get number of feature in matrix. This funcion only exists to prevent future bugs.
        """
        return X.shape[1]

    def get_n_instances(self, X):
        """
        Get number of instances in matrix. This funcion only exists to prevent future bugs.
        """
        return X.shape[0]

    def get_matrix_by_index(self, X, indexes):
        """
        Get instances by index vector. This funcion only exists to prevent future bugs.
        """
        return X[indexes, :]

    def get_best_split(self, X, y):
        index = random.randint(
            0, self.get_n_features(X) - 1
        )  # TODO: This will be a terribly complicated method...
        x = self.get_feature_from_matrix(X, index)
        boundary = x.mean()  # TODO: This is the "hello world" of decisions
        decision = y.mean()  # TODO: This is the "hello world" of decisions
        return index, boundary, decision


class DecisionTree:
    """
    Decision tree for a numerical feature
    """

    def __init__(self):
        self.root = DecisionNode()

    def fit(self, X, y):
        self.root._fit(X, y, recursive=True)
        return self

    def fit_from_data(self, data: MachineLearningData):
        self.root.fit(data, recursive=True)
        return self

    def score(self, x):
        return self.root.traverse(x)

    def __repr__(self):
        # return "node"
        return str(self.root)
