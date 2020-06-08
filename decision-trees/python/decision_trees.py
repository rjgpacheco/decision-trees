class DecisionNode:
    """
    Decision node for a numerical feature
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.boundary = None
        self.name = None
        self.decision = None

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
        self.boundary = boundary
        self.right = DecisionNode()
        self.left = DecisionNode()

    def set_decision(self, decision):
        if not (self.is_leaf()):
            raise Exception("Setting a decision on a non leaf node")
        self.decision = decision

    def to_str(self, str):
        raise NotImplementedError()
        return ""

    def from_str(self, str):
        raise NotImplementedError()
        return ""

    def traverse(self, x):
        """
        Traverse a decision node.
        """
        if self.is_leaf():
            return decision

        if x <= self.boundary:
            return self.left.traverse(x)
        else:
            return self.right.traverse(x)


class DecisionTree:
    """
    Decision tree for a numerical feature
    """

    def __init__(self):
        self.root = None
