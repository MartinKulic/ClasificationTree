import numpy as np


def calculate_accuracy(matrix):
    count_all = matrix.sum().sum()
    count_corrext = matrix.diagonal().sum()

    acc = count_corrext / count_all
    return acc

def calculate_clasewise_metrics(matrix):
    num_class = matrix.shape[0]

    recall = np.zeros(num_class)
    precision = np.zeros(num_class)
    f1 = np.zeros(num_class)

    for i in range(num_class):
        TP = matrix[i, i]
        FN = matrix[i, :].sum() - TP
        FP = matrix[:, i].sum() - TP

        recall[i] = TP / (TP + FN)
        precision[i] = TP / (TP + FP)
        f1[i] = (2 * precision[i] * recall[i])/ (precision[i] + recall[i])

    avg_recall = np.mean(recall)
    avg_precision = np.mean(precision)
    avg_f1 = np.mean(f1)

    return recall, precision, f1, avg_recall, avg_precision, avg_f1