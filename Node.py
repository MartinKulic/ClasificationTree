class Node:
    def __init__(self, comp = None, res_class = None, cond : str = ""):
        self.comp = comp
        self.true_child : Node = None
        self.false_child : Node = None
        self.res_class = res_class

        self.cond = cond

    def classifie(self, measurement) -> "Node":
        if self.comp(measurement):
            return self.true_child
        else:
            return self.false_child

    def hasNext(self):
        return (self.comp is not None) and ((self.true_child is not None) or (self.false_child is not None))

    def __str__(self):
        return f"<| {self.cond} = {self.res_class}|>"
