import logging

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename="/home/ricardo/repos/decision-trees/decision_trees.log", level=logging.INFO, format=FORMAT)
logger = logging.getLogger("DecisionTrees")
