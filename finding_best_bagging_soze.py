from concurrent.futures.thread import ThreadPoolExecutor
from threading import Lock

import pandas as pd
import numpy as np

import lib
from ClasificationTree import ClasificationTree
from BaggingClasificationTree import BagingClasificationTree


def replication(X_tr, X_ts, y_tr, y_ts, split_rnd, num_of_trees, mutex, out_file, n):
    print(f"Starting tn{num_of_trees} {n}")
    #train
    tree = BagingClasificationTree(num_of_trees)
    tree.fit(X_tr, y_tr)

    confusion_matrix = np.zeros([tree.classes.size, tree.classes.size])
    class_map = {}
    for ind, clas in enumerate(tree.classes):
        class_map[clas] = ind
    for x_t,y_t in zip(X_ts.to_numpy(), y_ts.to_numpy()):
        pred_y = tree.predict(x_t)
        confusion_matrix[class_map[y_t],class_map[pred_y]] += 1

    my_acc = lib.calculate_accuracy(confusion_matrix)
    _,_,_, my_avg_recall, my_abg_precision, my_avg_f1 = lib.calculate_clasewise_metrics(confusion_matrix)
    print(my_acc, my_avg_recall, my_abg_precision, my_avg_f1)

    #write
    with mutex:
        out_file.write(f'{num_of_trees};{split_rnd};{my_acc};{my_avg_recall};{my_abg_precision};{my_avg_f1};\n')
    print(f"DONE tn{num_of_trees} {n} acc{my_acc}")

SIZE_OF_RAND_REPEATS = 15

xs_tr = []
ys_tr = []
xs_ts = []
ys_ts = []

rnds = np.random.randint(0, 99999999, SIZE_OF_RAND_REPEATS)
raw = pd.read_csv('data.csv')

for rnd in rnds:
    train=raw.sample(frac=0.8,random_state=rnd)

    test=raw.drop(train.index)

    y_test = test["price_class"]
    x_test = ClasificationTree.prepareX(test.drop(columns=["price_class"]))

    y_train = train["price_class"]
    x_train_raw = train.drop(columns=["price_class"])

    x_train = ClasificationTree.prepareX(x_train_raw)

    xs_tr.append(x_train)
    ys_tr.append(y_train)
    xs_ts.append(x_test)
    ys_ts.append(y_test)

mut = Lock()
out_file = open('out.csv', 'w')
out_file.write("num_of_trees;split_rnd;accuracy;recall;precision;f1;\n")

for tree_num in range(1, 500):
    print(f"\n=== START tree_num {tree_num} ===")

    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(SIZE_OF_RAND_REPEATS):
            executor.submit(replication, xs_tr[i], xs_ts[i], ys_tr[i], ys_ts[i], rnds[i], tree_num, mut, out_file, i)

    print(f"=== END tree_num {tree_num} ===")
