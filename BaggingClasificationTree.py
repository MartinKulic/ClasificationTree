from _thread import lock
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from joblib.memory import NotMemorizedFunc
from numpy import dtype, ndarray
from pandas import DataFrame

from ClasificationTree import ClasificationTree


class BagingClasificationTree:

    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.tree = [ClasificationTree() for _ in range(num_trees)]
        #np.random.seed(200)

    def fit(self, X:pd.DataFrame, y: np.ndarray):
        self.classes = np.sort(np.unique(y))

        with ThreadPoolExecutor(max_workers=10) as executor:
            for tree in self.tree:
                executor.submit(self.fit_task, tree, X, y) #self.fit_task(tree, X, y)

    def fit_task(self, tree: ClasificationTree, X: DataFrame, y: ndarray[tuple[Any, ...], dtype[Any]]):
        this_X = [None] * X.shape[0]
        this_y = [None] * X.shape[0]
        # vyber s opakovanim z X,y
        for i in range(X.shape[0]):
            rand_idx = np.random.randint(X.shape[0])
            this_X[i] = X.iloc[rand_idx, :]
            this_y[i] = y[rand_idx]

        this_X_df = pd.DataFrame(this_X, columns=X.columns)
        tree.fit(this_X_df, np.array(this_y))

    def predict(self, X):
        predictions = []
        mutex = Lock()

        with ThreadPoolExecutor(max_workers=10) as executor:
            for tree in self.tree:
                executor.submit(self.method_name, tree, X, predictions, mutex)#self.method_name(tree, X, predictions, mutex)

        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    def method_name(self, tree: ClasificationTree, X, predictions: list[Any], mutex: lock):
        prediction = tree.predict(X)
        with mutex:
            predictions.append(prediction)

    @staticmethod
    def prepareX (x: pd.DataFrame) -> pd.DataFrame:
        return ClasificationTree.prepareX(x)