from collections import namedtuple
from queue import Queue
from sys import float_info
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_bool_dtype

from Node import Node


class SplitInfo:
    true_label = None
    false_l = None

    condition: Callable[..., bool]
    impurity: float

    cond_txt : str

    def __init__(self, condition: Callable[..., bool] = None, impurity: float = None, true_label=None,
                 false_label=None, text:str = ""):
        self.condition = condition
        self.impurity = impurity
        self.true_label = true_label
        self.false_label = false_label
        self.cond_txt = text

class ClasificationTree:

    class TrainData(NamedTuple):
        node : Node
        x_sub_df : pd.DataFrame
        y_sub : np.array



    def __init__(self):
        self.root : Node = Node()


    def fit(self, X : pd.DataFrame, y : np.array):
        # self.atributes = X  # data
        # self.labels = y  # truths

        self.classes = np.sort(np.unique(y))

        queue = Queue()
        queue.put(self.TrainData(self.root, X, y))

        while(not queue.empty()):

            trainData = queue.get()

            if trainData.x_sub_df.size < 3:
                continue

            conditionInfo = self.findBestCondition(trainData.x_sub_df, trainData.y_sub)

            if (conditionInfo.condition is None):
                continue

            trainData.node.comp = conditionInfo.condition
            trainData.node.cond = conditionInfo.cond_txt
            trainData.node.true_child = Node(res_class=conditionInfo.true_label)
            trainData.node.false_child = Node(res_class=conditionInfo.false_label)

            true_child_mask = conditionInfo.condition(trainData.x_sub_df.to_numpy().T)
            queue.put(self.TrainData(trainData.node.true_child, trainData.x_sub_df[true_child_mask], trainData.y_sub[true_child_mask]))
            false_child_mask = np.logical_not(true_child_mask)
            queue.put(self.TrainData(trainData.node.false_child, trainData.x_sub_df[false_child_mask], trainData.y_sub[false_child_mask]))

            trainData.node.res_class = f"{trainData.node.res_class} T:{np.sum(true_child_mask)} F:{np.sum(false_child_mask)}"



    def findBestCondition(self, x : pd.DataFrame, y) -> SplitInfo:

        best_condition = SplitInfo(impurity = float_info.max)

        for i in range(x.shape[1]):
            colDatas = x.iloc[:, i]
            cur_condition = None

            if is_bool_dtype(colDatas):
                cur_condition = self.findBestConditionBoolean(i, colDatas, y)
            else:
                cur_condition = self.findBestConditionNumeric(i, colDatas, y)


            if(cur_condition.impurity < best_condition.impurity):
                best_condition = cur_condition
                best_condition.cond_txt = f"{x.columns[i] }: {best_condition.cond_txt}"

        return best_condition


    def findBestConditionBoolean(self, colIndex, x, y) -> SplitInfo:

        counts = np.zeros([2, self.classes.size])

        true_mask = x
        false_mask = np.logical_not(true_mask)

        true_labels = y[true_mask]
        false_labels = y[false_mask]

        true_unique, true_counts = np.unique(true_labels, return_counts=True)
        false_unique, false_counts = np.unique(false_labels, return_counts=True)

        if ( true_counts.size == 0 ) or ( false_counts.size == 0 ):
            return SplitInfo(impurity = float_info.max)

        true_label = true_unique[np.argmax(true_counts)]
        false_label = false_unique[np.argmax(false_counts)]
        if ( true_label == false_label ):
            return SplitInfo(impurity = float_info.max)

        # takto lebo broadcasting a inak by mi uz tak trvalo moc dloho
        helper = 0
        for i in true_counts:
            counts[0, helper] = i
            helper += 1
        helper = 0
        for i in false_counts:
            counts[1, helper] = i
            helper += 1
        #

        impurity = self.impuruty(counts)

        splitInfo = SplitInfo(condition=(lambda df: df[colIndex]), impurity=impurity, true_label=true_label, false_label=false_label)
        return splitInfo




    def findBestConditionNumeric(self, colIndex, x, y) -> (Callable[[...], bool] , float):

        if (np.unique(y).size < 2):
            return SplitInfo(impurity = float_info.max)
            #raise Exception(f"Cannot find split for {colIndex} - num of unique values is less than 2")

        sorted_x = np.sort(x)
        borders_x = []
        for i in range(sorted_x.size-1):
            border = (sorted_x[i] + sorted_x[i+1])/2
            borders_x.append(border)

        best_condition = SplitInfo(impurity = float_info.max)
        for i in borders_x:
            mask = x < i
            cur_condition = self.findBestConditionBoolean(colIndex, mask, y)

            if(cur_condition.impurity < best_condition.impurity):
                best_condition = cur_condition
                h=i #WTF this caused so much painnnn
                best_condition.condition = lambda df: df[colIndex] < h
                best_condition.cond_txt = f"< {i}"

        return best_condition



    def predict(self, x: np.array):
        curNode = self.root

        while curNode.hasNext():
            curNode = curNode.classifie(x)

        return curNode.res_class

    '''
        weighted gini impurity
        :param counts: 1.dim = napr. 0 - condition true; 1 - condition false; 2.dim = count in classes
    '''
    def impuruty(self, counts: np.array):
        numInNodes = np.sum(counts, axis=1)
        probabilities = np.divide(counts.T, numInNodes).T
        probabilitiesSqrt = np.power(probabilities, 2)
        suma = np.sum(probabilitiesSqrt, axis=1)

        gini = 1-suma #np.array([1-suma[0], 1-suma[1]])

        weights = numInNodes/np.sum(numInNodes)
        weighted_ginis = weights * gini
        weighted_gini = np.sum(weighted_ginis)
        return weighted_gini

    '''
    gini impurity
    '''
    def impurutyOneNode(self, counts : np.array):
        # 1 - sum(probabilityClass_i^2)
        numOfAll = np.sum(counts)
        probabilities = np.divide(counts, numOfAll)
        probabilitiesSqrt = np.power(probabilities, 2)
        suma = np.sum(probabilitiesSqrt)
        return 1 - suma

    @staticmethod
    def prepareX(raw_x: pd.DataFrame) -> pd.DataFrame:
        non_numeric_cols = raw_x.select_dtypes(include=["object"]).columns
        print(non_numeric_cols)

        x_encoded = pd.get_dummies(raw_x, columns=non_numeric_cols, drop_first=False)
        return x_encoded

    '''
    From https://www.geeksforgeeks.org/dsa/level-order-tree-traversal/
    '''
    def levelOrder(self, root : Node):
        if root is None:
            return []

        # Create an empty queue for level order traversal
        q = []
        res = []

        # Enqueue Root
        q.append(root)
        curr_level = 0

        while q:
            len_q = len(q)
            res.append([])

            for _ in range(len_q):
                # Add front of queue and remove it from queue
                node = q.pop(0)
                res[curr_level].append(node)

                # Enqueue left child
                if node.true_child is not None:
                    q.append(node.true_child)

                # Enqueue right child
                if node.false_child is not None:
                    q.append(node.false_child)
            curr_level += 1
        return res




