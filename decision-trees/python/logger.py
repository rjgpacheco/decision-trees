import logging

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename="decision_trees.log", level=logging.INFO, format=FORMAT)
logger = logging.getLogger("DecisionTrees")
