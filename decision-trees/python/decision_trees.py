import json  # NOTE: this is here just so that the str() method is pretty


class DecisionNode:
    """
    Decision node for a numerical feature
    """

    def __init__(
        self, boundary: float = None, name: str = None, decision: float = None,
    ):
        self.set_boundary(boundary)
        self.set_decision(decision)

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
