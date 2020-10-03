from logger import logger
from decision_node import DecisionNode

class DecisionTree:
    """
    Decision tree for a numerical feature
    """

    def __init__(self):
        logger.warning("Initializing DecisionTree")
        self.root = DecisionNode()

    def fit(self, X, y):
        logger.warning("Calling DecisionTree.fit()")
        self.root.fit(X, y, recursive=True)
        return self

    def score(self, x):
        logger.warning("Calling DecisionTree.score()")
        return self.root.traverse(x)

    def __repr__(self):
        # return "node"
        return str(self.root)
