from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

import lib
from ClasificationTree import ClasificationTree
from BaggingClasificationTree import BagingClasificationTree


def replication(X_tr, X_ts, y_tr, y_ts, split_rnd, num_of_trees, n):
    print(f"Starting tn{num_of_trees} {n}")
    #train
    tree = BagingClasificationTree(num_of_trees)
    tree.fit(X_tr, y_tr)

    confusion_matrix = np.zeros([tree.classes.size, tree.classes.size])
    class_map = {}
    for ind, clas in enumerate(tree.classes):
        class_map[clas] = ind
    for x_t,y_t in zip(X_ts.to_numpy(), y_ts):
        pred_y = tree.predict(x_t)
        confusion_matrix[class_map[y_t], class_map[pred_y]] += 1

    my_acc = lib.calculate_accuracy(confusion_matrix)
    _,_,_, my_avg_recall, my_abg_precision, my_avg_f1 = lib.calculate_clasewise_metrics(confusion_matrix)
    print(my_acc, my_avg_recall, my_abg_precision, my_avg_f1)

    print(f"DONE tn{num_of_trees} {n} acc{my_acc}")
    return (
        num_of_trees,
        split_rnd,
        my_acc,
        my_avg_recall,
        my_abg_precision,
        my_avg_f1
    )

SIZE_OF_RAND_REPEATS = 50

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
    ys_tr.append(y_train.to_numpy())
    xs_ts.append(x_test)
    ys_ts.append(y_test.to_numpy())

with open('out.csv', 'w', buffering=1) as out_file:  # line-buffered
    out_file.write("num_of_trees;split_rnd;accuracy;recall;precision;f1;\n")

    with ProcessPoolExecutor() as executor:
        futures = []

        for i in range(SIZE_OF_RAND_REPEATS):
            futures.append(
                executor.submit(
                    replication,
                    xs_tr[i], xs_ts[i],
                    ys_tr[i], ys_ts[i],
                    rnds[i],
                    1,
                    i
                )
            )

        for future in as_completed(futures):
            (
                num_of_trees,
                split_rnd,
                acc,
                avg_recall,
                avg_precision,
                avg_f1
            ) = future.result()

            out_file.write(
                f"{num_of_trees};{split_rnd};{acc};{avg_recall};{avg_precision};{avg_f1};\n"
            )

    for tree_num in range(5, 500,5):
        print(f"\n=== START tree_num {tree_num} ===")

        with ProcessPoolExecutor() as executor:
            futures = []

            for i in range(SIZE_OF_RAND_REPEATS):
                futures.append(
                    executor.submit(
                        replication,
                        xs_tr[i], xs_ts[i],
                        ys_tr[i], ys_ts[i],
                        rnds[i],
                        tree_num,
                        i
                    )
                )

            for future in as_completed(futures):
                (
                    num_of_trees,
                    split_rnd,
                    acc,
                    avg_recall,
                    avg_precision,
                    avg_f1
                ) = future.result()

                out_file.write(
                    f"{num_of_trees};{split_rnd};{acc};{avg_recall};{avg_precision};{avg_f1};\n"
                )

        print(f"=== END tree_num {tree_num} ===")
