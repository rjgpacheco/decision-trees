import logging
import json

import pandas as pd
import numpy as np


class MachineLearningData:

    logging.getLogger(__name__)

    def __init__(self, X, y=None, schema=None, row_id=None):

        if X is None:
            raise ValueError("X must not be None")

        if schema is None:
            logging.warn("schema is None")

        self.X = X
        self._y = y
        self._schema = schema
        self._row_id = row_id

    def __repr__(self):
        # return "node"
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        return {
            "X": pd.DataFrame(self.X).to_csv(index=False, header=False),
            "y": pd.DataFrame(self.y).to_csv(index=False, header=False),
            "schema": self.schema,
        }

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if value is None:
            logging.warn("y is None")
        else:
            if isinstance(value, np.ndarray):
                pass
            else:
                value = np.ndarray(value)
            if len(value) != self.get_n_instances():
                raise ValueError("X and y must have same length")
        self._y = value

    @y.deleter
    def y(self):
        del self._y

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, value):
        self._schema = value

    @schema.deleter
    def schema(self):
        del self._schema

    def get_feature(self, index):
        """
        Get feature of feature matrix. This funcion only exists to prevent future bugs.
        """
        return self.X[:, index]

    def get_instances(self, indexes):
        """
        Get instances by index vector. This funcion only exists to prevent future bugs.
        """
        return self.X[indexes, :]

    def get_n_features(self):
        """
        Get number of feature in matrix. This funcion only exists to prevent future bugs.
        """
        return self.X.shape[1]

    def get_n_instances(self):
        """
        Get number of instances in matrix. This funcion only exists to prevent future bugs.
        """
        return self.X.shape[0]
