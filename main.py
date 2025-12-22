from ClasificationTree import ClasificationTree
from Node import Node

x = [10, True, False]

cTree = ClasificationTree()
cTree.root = Node(lambda v : v[0] < 50)

cTree.root.true_child = Node(lambda v : v[1], "les then 50")
cTree.root.false_child = Node(lambda v : v[1], "more then 50")

cTree.root.true_child.true_child = Node(None, "les then 50 and true")
cTree.root.true_child.false_child = Node(lambda v : v[2] , "les then 50 and false")

vys = cTree.predict(x)
print(vys)
