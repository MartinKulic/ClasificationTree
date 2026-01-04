from concurrent.futures import ThreadPoolExecutor
from random import Random
from threading import Lock

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier

import lib
from BaggingClasificationTree import BagingClasificationTree
from ClasificationTree import ClasificationTree

from sklearn.tree import DecisionTreeClassifier

raw = pd.read_csv('../data.csv')

def doCompare(rs, lck:Lock, fil, iter):
    print(f"Doing {iter}")
    # ROZDELIT NA TEST A TRAIN
    train=raw.sample(frac=0.8,random_state=rs)
    test=raw.drop(train.index)

    y_test = test["price_class"]
    x_test = ClasificationTree.prepareX(test.drop(columns=["price_class"]))

    y_train = train["price_class"]
    x_train_raw = train.drop(columns=["price_class"])

    x_train = ClasificationTree.prepareX(x_train_raw)

    # My tree
    cTree = BagingClasificationTree(NTree)
    #print("Training", end="")
    cTree.fit(x_train, y_train.to_numpy())
    #print(" DONE")

    confusion_matrix = np.zeros([3,3])
    class_map = {}
    for ind, clas in enumerate(cTree.classes):
        class_map[clas] = ind

    #print("Testing", end="")
    for x_t,y_t in zip(x_test.to_numpy(), y_test.to_numpy()):
        pred_y = cTree.predict(x_t)
        confusion_matrix[class_map[y_t],class_map[pred_y]] += 1
    #print(f" DONE {iter}")
    #print(cTree.classes)
    print(confusion_matrix)
    my_acc = lib.calculate_accuracy(confusion_matrix)
    _,_,_, my_avg_recall, my_abg_precision, my_avg_f1 = lib.calculate_clasewise_metrics(confusion_matrix)

    # Scikit tree
    clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=NTree)
    clf.fit(x_train, y_train.to_numpy())

    confusion_matrix_sklearn = np.zeros([3,3])
    predictions = clf.predict(x_test)
    for pred_y, y_t in zip(predictions, y_test.to_numpy()):
        confusion_matrix_sklearn[class_map[y_t], class_map[pred_y]] += 1

    print(confusion_matrix_sklearn)
    scikit_acc = lib.calculate_accuracy(confusion_matrix_sklearn)
    _,_,_, scikit_avg_recall, scikit_avg_precision, scikit_avg_f1 = lib.calculate_clasewise_metrics(confusion_matrix_sklearn)

    out_list = [rs, my_acc, my_avg_recall, my_abg_precision, my_avg_f1, scikit_acc, scikit_avg_recall, scikit_avg_precision, scikit_avg_f1]
    with lck:
        write_to_file(out_list, fil)
    print(f" DONE {iter}")
    return


def write_to_file(dta, fle):
    for i in range(0, 5):
        fle.write(f"{dta[i]};")
    fle.write(";")
    for i in range(5, len(dta)):
        fle.write(f"{dta[i]};")
    fle.write("\n")




out_list = []
threads = []
lock = Lock()
random = Random()
NTree = 25

fil = open("../out.csv", 'a')
fil.write(";My;;;;;Scikit;;;;;\n"
            "seed;Accuracy;Average_recall;Average_precision;Average_f1;;Accuracy;Average_recall;Average_precision;Average_f1;\n")

with ThreadPoolExecutor(max_workers=10) as executor:
    for i in range(5000):
        executor.submit(doCompare, random.randint(0,9999999), lock, fil, i)

# for i in range(2):
#         doCompare(i, lock, fil, i)