class Node:
    def __init__(self, comp = None, res_class = None):
        self.comp = comp
        self.true_child = None
        self.false_child = None
        self.res_class = res_class

    def classifie(self, measurement):
        if self.comp(measurement):
            return self.true_child
        else:
            return self.false_child

    def hasNext(self):
        return (self.comp is not None) and ((self.true_child is not None) or (self.false_child is not None))
