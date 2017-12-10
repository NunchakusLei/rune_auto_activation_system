import numpy as np
import cv2
from common_func import *


class predExtractor:
    """
    This class further extract information based on
    the result of recognizer apply with sepecific
    rules as constrains
    """
    def __init__(self, predictions, probabilities, boxes):
        """ Initializer """
        self.preds = predictions
        self.probs = probabilities
        self.perplexity = self.get_correlation_m
        self.boxes = boxes

    def dump_preds(self):
        """ dump predictions if perplexity is high """
        self.preds = np.zeros((self.preds.shape), dtype=self.preds.dtype)

    def zero_removal(self):
        for i in range(len(self.preds.flatten())):
            self.probs[i, 0] = -np.float('inf')
            self.preds[i] = np.argmax(self.probs[i,:])

    def get_pred(self):
        self.preds = np.argmax(self.probs, axis=1)
        return self.preds

    def number_redundancy_removal3(self):
        duplicates_num, duplicates = self.find_duplicates()
        missed = self.find_missed_number()

        def fresh(row):
            for col in range(10):
                if col not in missed:
                    self.probs[row, col] = -np.float('inf')

        while max(duplicates_num)>1:
            sorted_p = np.sort(self.probs)
            differences = sorted_p[:,9] - sorted_p[:,8]
            freshed_rows = []

            for item in duplicates:
                if item!='x' and duplicates.count(item)>1:

                    # get all the item's index
                    indexs = []
                    while item in duplicates:
                        i = duplicates.index(item)
                        indexs.append(i)
                        duplicates[i] = "x"

                    # find max prob for this redundancy item
                    max_p = None
                    col = item
                    for row in indexs:
                        if max_p is None:
                            max_p = differences[row]
                        if differences[row] < max_p:
                            fresh(row)
                            freshed_rows.append(row)
                        else:
                            max_p = differences[row]

                    for row in indexs:
                        if differences[row] < max_p:
                            fresh(row)
                            freshed_rows.append(row)

            self.get_pred()
            # reset freshed rows' preds
            for row in freshed_rows:
                self.preds[row] = 0
            duplicates_num, duplicates = self.find_duplicates()
            missed = self.find_missed_number()

            # print "pred:"
            # print self.preds
            # print "duplicates:"
            # print duplicates
            # print "duplicates_num:"
            # print duplicates_num
            # print self.probs
            # raw_input("----")

        missed = self.find_missed_number()
        # print missed
        preds_list = list(self.preds.flatten())
        if 0 in preds_list:
            self.preds[preds_list.index(0)] = missed[0]
        # self.zero_removal()


    def number_redundancy_removal2(self):
        duplicates_num, duplicates = self.find_duplicates()
        while max(duplicates_num)>1:
            sorted_p = np.sort(self.probs)
            differences = sorted_p[:,9] - sorted_p[:,8]

            for item in duplicates:
                if item!='x' and duplicates.count(item)>1:
                    indexs = []
                    # print "item: ", item

                    # get all the item's index
                    while item in duplicates:
                        i = duplicates.index(item)
                        indexs.append(i)
                        duplicates[i] = "x"

                    # find max prob for this redundancy item
                    max_p = None
                    col = item
                    for row in indexs:
                        if max_p is None:
                            max_p = differences[row]
                        if differences[row] < max_p:
                            self.probs[row, col] = -np.float('inf')
                        else:
                            max_p = differences[row]


                    for row in indexs:
                        if differences[row] < max_p:
                            self.probs[row, col] = -np.float('inf')

            self.get_pred()
            duplicates_num, duplicates = self.find_duplicates()

            # print "pred:"
            # print self.preds
            # print "duplicates:"
            # print duplicates
            # print "duplicates_num:"
            # print duplicates_num
            # print self.probs
            # raw_input("----")



    def number_redundancy_removal(self):
        duplicates_num, duplicates = self.find_duplicates()
        while max(duplicates_num)>1:
            for item in duplicates:
                if item!='x' and duplicates.count(item)>1:
                    indexs = []
                    # print "item: ", item

                    # get all the item's index
                    while item in duplicates:
                        i = duplicates.index(item)
                        indexs.append(i)
                        duplicates[i] = "x"

                    # find max prob for this redundancy item
                    max_p = None
                    col = item
                    for row in indexs:
                        if max_p is None:
                            max_p = self.probs[row, col]
                        if self.probs[row, col] < max_p:
                            self.probs[row, col] = -np.float('inf')
                        else:
                            max_p = self.probs[row, col]


                    for row in indexs:
                        if self.probs[row, col] < max_p:
                            self.probs[row, col] = -np.float('inf')

            self.get_pred()
            duplicates_num, duplicates = self.find_duplicates()

            # print "duplicates:"
            # print duplicates
            # print "duplicates_num:"
            # print duplicates_num
            # print self.probs
            # raw_input("----")

            # duplicates_num, duplicates = self.find_duplicates()

    def find_missed_number(self):
        contains = list(self.preds.flatten())
        missed = []
        for i in range(1,10):
            if i not in contains:
                missed.append(i)
        return missed

    def find_duplicates(self):
        duplicates = []
        duplicates_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(self.preds.flatten())):
            duplicates.append(self.preds[i])
            duplicates_num[self.preds[i]] += 1

        # for item in duplicates:
        #     if duplicates.count(item)<2:
        #         duplicates.pop(duplicates.index(item))
        return duplicates_num, duplicates

    def get_perplexity(self):
        """
        This method get the average normalized value
        from correlation matrix
        """
        m = self.get_correlation_m(self.probs)
        return np.average(m)

    def get_correlation_m(self,matrix=None):
        """
        This function calculate the correlation matrix
        """
        if matrix is None:
            matrix=self.probs
        # correlation matrix calculation
        matrix2 = np.dot(matrix, np.transpose(matrix))
        normed = np.linalg.norm(matrix, axis=1)
        normed2 = np.dot(normed.reshape((-1,1)), normed.reshape((1,-1)))
        correlation = matrix2 / normed2
        return correlation

    def draw_pred(self, target_img):
        """
        This function draw predicted number in target_img
        """
        i = 0
        for box in self.boxes:
            draw_pred_point = (int(sort_box_points(box)[1][0]),int(sort_box_points(box)[1][1]))
            cv2.putText(target_img,
                        str(self.preds[i]), draw_pred_point,
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        draw_number_color,
                        5)
            i += 1
