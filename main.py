import numpy as np
import pandas as pd
from ClasificationTree import ClasificationTree

raw = pd.read_csv('data.csv')
print(raw.isna().sum())

#ROZDELIT NA TEST A TRAIN
train=raw.sample(frac=0.8,random_state=200)
test=raw.drop(train.index)

y_test = test["price_class"]
x_test = ClasificationTree.prepareX(test.drop(columns=["price_class"]))

y_train = train["price_class"]
x_train_raw = train.drop(columns=["price_class"])

x_train = ClasificationTree.prepareX(x_train_raw)


cTree = ClasificationTree()
print("Training", end="")
cTree.fit(x_train, y_train)
print(" DONE")

confusion_matrix = np.zeros([3,3])
class_map = {}
for ind, clas in enumerate(cTree.classes):
    class_map[clas] = ind
print("Testing", end="")
for x_t,y_t in zip(x_test.to_numpy(), y_test.to_numpy()):
    pred_y = cTree.predict(x_t)
    confusion_matrix[class_map[y_t],class_map[pred_y]] += 1
print(" DONE")
print(confusion_matrix)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train.to_numpy(), y_train.to_numpy())

confusion_matrix_sklearn = np.zeros([3,3])
for x_t,y_t in zip(x_test.to_numpy(), y_test.to_numpy()):
    pred_y = clf.predict(x_t.reshape(1, -1))
    confusion_matrix_sklearn[class_map[y_t],class_map[pred_y[0]]] += 1
print(confusion_matrix_sklearn)

