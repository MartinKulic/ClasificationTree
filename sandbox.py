from xml.etree.ElementPath import xpath_tokenizer_re

import numpy as np

from ClasificationTree import ClasificationTree, SplitInfo
from Node import Node
import pandas as pd

x = [10, True, False]

tTree = ClasificationTree()
tTree.root = Node(lambda v : v[0] < 50)

tTree.root.true_child = Node(lambda v : v[1], "les then 50")
tTree.root.false_child = Node(lambda v : v[1], "more then 50")

tTree.root.true_child.true_child = Node(None, "les then 50 and true")
tTree.root.true_child.false_child = Node(lambda v : v[2], "les then 50 and false")

vys = tTree.predict(x)

print(vys)


raw = pd.read_csv('data.csv')
print(raw.isna().sum())

# NEZABUDNI ROZDELIT NA TEST A TRAIN

y = raw["price_class"]
x_raw = raw.drop(columns=["price_class"])

x = ClasificationTree.prepareX(x_raw)

# TEST EXAMPLE


d = {"LovesPopcorn":[True, True, False, False, True, True, False],
     "LovesSoda":[True, False, True, True, True, False, False],
     "Age":[7,12,18,35,38,50,83]}
x = pd.DataFrame.from_dict(d)
y = pd.Series([False, False, True, True, True, False, False])
y = y.map({True: "Loves Cool as Ice", False: "Does not love Coll as Ice"})

cTree = ClasificationTree()

testCounts = np.zeros([2,2])

testCounts[0,0] = 3
testCounts[0,1] = 1
testCounts[1,0] = 0
testCounts[1,1] = 3

print (cTree.impuruty(testCounts))

si = SplitInfo(condition=lambda x: x[1]<3 , impurity=0.0)
c = [[0,1,2,3,4,5],
     [1,2,3,4,5,6],
     [2,3,4,5,6,7],
     [3,4,5,6,7,8], ]
c = np.array(c)
mask = si.condition(c.T)
print(c[mask])

test = np.array([0,0,0])
print(np.argmax(test))

x = ClasificationTree.prepareX(x)
cTree.fit(x, y)

res = cTree.levelOrder(cTree.root)
for level in res:
    for val in level:
        print(val, end=' ')
    print()


nt = [True, True, 15]
print(cTree.predict(nt))









