class ClasificationTree:
    def __init__(self):
        self.root = None

    def fit(self, X, y):
        self.X = X  # data
        self.y = y  # truths




    def predict(self, x):
        curNode = self.root

        while curNode.hasNext():
            curNode = curNode.classifie(x)

        return curNode.res_class