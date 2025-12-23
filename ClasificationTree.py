import numpy as np

from Node import Node


class ClasificationTree:
    def __init__(self):
        self.root : Node = None

    def fit(self, X : np.array, y : np.array):
        self.X = X  # data
        self.y = y  # truths

        self.classes = np.unique(y).sort()







    def predict(self, x: np.array):
        curNode = self.root

        while curNode.hasNext():
            curNode = curNode.classifie(x)

        return curNode.res_class

    def giny(self):
        pass
