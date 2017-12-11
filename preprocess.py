#!/usr/bin/env python
import cv2
import argparse
import sys, os, numpy as np
import scipy
from common_func import *
from manual_labelling import try_open_video_file
from mnist_deep_estimator import numberRecognizer
from prediction_selection import predExtractor


DEBUG = True

# ############################# #
# Grid searching module (Start) #
# ############################# #

""" rules for cell refine """
single_cell_area_ratio_lower = 0.00443
single_cell_area_ratio_upper = 0.0105
cell_weight_height_ratio = 16/28.0
cell_weight_height_ratio_toleration = 0.12


"""
This function get rid of redundancy number boxes
"""
def cell_redundancy_removal(contours, rects, number_boxes):
    bad_box_indexs = list()
    dist_toleration = 10

    for rect_i in range(len(rects)):
        if rect_i not in bad_box_indexs:
            for rect_j in range(rect_i+1, len(rects)):
                rect_i_center_x = rects[rect_i][0][0]
                rect_i_center_y = rects[rect_i][0][1]
                rect_j_center_x = rects[rect_j][0][0]
                rect_j_center_y = rects[rect_j][0][1]

                dist_x = abs(rect_i_center_x - rect_j_center_x)
                dist_y = abs(rect_i_center_y - rect_j_center_y)

                dist_ij = dist_x**2 + dist_y**2

                if dist_ij < dist_toleration**2:
                    rect_i_area = rects[rect_i][1][0] * rects[rect_i][1][1]
                    rect_j_area = rects[rect_j][1][0] * rects[rect_j][1][1]

                    if rect_i_area > rect_j_area:
                        bad_box_indexs.append(rect_j)
                    else:
                        bad_box_indexs.append(rect_i)

    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(len(number_boxes)):
        if i not in bad_box_indexs:
            good_contours.append(contours[i])
            good_rects.append(rects[i])
            good_boxes.append(number_boxes[i])

    return good_contours, good_rects, good_boxes, bad_box_indexs



"""
This function get rid of outlier number boxes
"""
def cell_outlier_removal(contours, rects, number_boxes):
    dist_list = [0.0] * len(rects)

    for rect_i in range(len(rects)):
        for rect_j in range(rect_i+1,len(rects)):
            rect_i_center_x = rects[rect_i][0][0]
            rect_i_center_y = rects[rect_i][0][1]
            rect_j_center_x = rects[rect_j][0][0]
            rect_j_center_y = rects[rect_j][0][1]

            dist_x = abs(rect_i_center_x - rect_j_center_x)
            dist_y = abs(rect_i_center_y - rect_j_center_y)

            dist_ij = dist_x**2 + dist_y**2

            dist_list[rect_i] += dist_ij
            dist_list[rect_j] += dist_ij

    # print min(dist_list)

    bad_box_indexs = list()
    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(min(9, len(rects))):
        current_min_index = dist_list.index(min(dist_list))

        bad_box_indexs.append(dist_list.pop(current_min_index))
        good_contours.append(contours.pop(current_min_index))
        good_rects.append(rects.pop(current_min_index))
        good_boxes.append(number_boxes.pop(current_min_index))

    return good_contours, good_rects, good_boxes, bad_box_indexs



def cell_refining(src, contours):
    """
    This function refine cells based on the contours and rules
    """
    good_contours = []
    boxes = []
    rects = []
    # src_pts = []
    # central_mass_point = [0,0]
    for contour in contours:
        # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
        rect = cv2.minAreaRect(contour)
        # box = cv2.cv.BoxPoints(rect)
        box = cv2.boxPoints(rect) #for OpenCV 3.x

        if rect[1][0]*rect[1][1]>src.shape[0]*src.shape[1]*single_cell_area_ratio_lower and\
           rect[1][0]*rect[1][1]<src.shape[0]*src.shape[1]*single_cell_area_ratio_upper and\
           min(rect[1])/max(rect[1])<(cell_weight_height_ratio+cell_weight_height_ratio_toleration) and\
           min(rect[1])/max(rect[1])>(cell_weight_height_ratio-cell_weight_height_ratio_toleration): #and\
            # cv2.contourArea(contour)/(rect[1][0]*rect[1][1])>0.8:
            # if abs(rect[2])<10 or abs(rect[2])>86:
            good_contours.append(contour)
            boxes.append(box)
            rects.append(rect)

    #         src_pts.append((int(rect[0][0]),int(rect[0][1])))
    #         central_mass_point[0] += rect[0][0]
    #         central_mass_point[1] += rect[0][1]
    #
    # if len(boxes)!=0:
    #     central_mass_point[0] = central_mass_point[0]/len(boxes)
    #     central_mass_point[1] = central_mass_point[1]/len(boxes)
    return good_contours, rects, boxes, contours



def grid_searching_preprocess(src):
    """
    This function perform based image processing techniques
    targeting grid searching
    """
    img = src.copy()
    # apply basic image processing techniques
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # get gray scale image
    # """
    img = cv2.blur(gray, (5,5))                  # blur image

    img = (img.astype(np.float64))**2
    # cv2.imshow("test", normalize(img))
    img = cv2.GaussianBlur(img, (3,3), 0)        # blur image again

    img = normalize(img)

    img = cv2.Canny(img, 400, 200, apertureSize=3, L2gradient=True) # find edges

    img = cv2.dilate(img, np.ones((3,3),np.uint8), iterations=3)    # enlarge edges

    """

    img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    # img = cv2.Canny(img, 100, 100, apertureSize=3, L2gradient=True)
    # img = cv2.blur(gray, (5,5))                  # blur image
    # """


    return img, gray



def grid_searching(src, pre, gray):
    # get contours
    im2, contours, hierarchy = cv2.findContours(pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # rules based contour analysis (select good countour and convert them to rectagles)
    contours, rects, boxes, _ = cell_refining(src, contours)
    contours, rects, boxes, _ = cell_redundancy_removal(contours, rects, boxes)
    contours, rects, boxes, _ = cell_outlier_removal(contours, rects, boxes)

    # draw boxes
    boxes_int = []
    for box in boxes:
        box_int = np.int0(box)
        boxes_int.append(box_int)
    cv2.drawContours(src,boxes_int,-1,(0,0,255),2)

    return rects, boxes

# ----------------------------- #
# Grid searching module  (End)  #
# ----------------------------- #





# ################################# #
# Number recognition module (Start) #
# ################################# #

def region_of_interest(img, box, side_size=32):
    """
    This function extract the ROI and transform it perspectively
    params img: the original image
        type: np.array (image)
    params box: the 4 points of the ROI box
        type: 2d np.array
    return dst: the ROI transformed perspectively
        type: np.array (image)
    """
    pts1 = sort_box_points(box)#np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[0,side_size],[side_size,side_size],[side_size,0],])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # if DEBUG:
    #     print "M: \n", M
    dst = cv2.warpPerspective(img,M,(side_size,side_size))

    return dst



def get_rim_mask(size, b_ratio=0.1):
    """
    This function generates a mask of rim
    """
    center = (size[0]/2, size[1]/2)
    mask = np.ones((size))
    for row in range(size[0]):
        for col in range(size[1]):
            position = max(abs(row - center[0])/float(center[0]), \
                   abs(col - center[1])/float(center[1]))
            if position > (1-b_ratio):
                # (1-b_ratio) / position
                mask[row, col] = (1-b_ratio) / position
    return mask



def number_recognizing_cell_preprocess(src):
    """
    This function prepare data for feeding prediction algorithm
    """
    # image Processing
    img = src.copy()
    img = 255 - img                     # inverse

    # rim removal
    mask = get_rim_mask(src.shape, 0.35)
    img = normalize(img * (mask**10))

    # data preparation
    feed_data = img.reshape((1, src.shape[0] * src.shape[1])) / 255.

    return feed_data, img



def number_recognizing_preparation(gray, rects, boxes):
    # extracting ROI
    side_size = 28

    dst = None
    pred_data = None
    for box in boxes:
        roi = region_of_interest(gray, box, side_size=side_size)        # get ROI
        one_pred_data, roi = number_recognizing_cell_preprocess(roi)    # preprocessing on each ROI

        # prepare display for human debuging
        if dst is None:
            dst = roi
        else:
            dst = np.hstack((dst, roi))

        # prepare prediction data
        if pred_data is None:
            pred_data = one_pred_data
        else:
            pred_data = np.vstack((pred_data, one_pred_data))

    return pred_data, dst


def number_recognizing_prediction(src, pred_data, boxes, recognizer):
    # perfom CNN number recognition
    prediction, probabilities = recognizer.predict(pred_data)

    # apply constrain rule
    post_extractor = predExtractor(prediction, probabilities, boxes)
    perplexity = post_extractor.get_perplexity()

    # post_extractor.draw_pred(src)
    post_extractor.zero_removal()
    duplicates_num, duplicates = post_extractor.find_duplicates()
    # print "duplicates:"
    # print duplicates
    # print "duplicates_num:"
    # print duplicates_num
    # post_extractor.number_redundancy_removal2()

    if perplexity > 0.25:
        post_extractor.dump_preds()

    # print prediction
    if perplexity <= 0.25:
        # post_extractor.number_redundancy_removal2()
        if len(boxes)<10:
            post_extractor.number_redundancy_removal3()

        # print post_extractor.find_missed_number()

        post_extractor.draw_pred(src)

    # print prediction
    # print probabilities
    # sorted_p = np.sort(probabilities)
    # print sorted_p[:,9] - sorted_p[:,8]

    return list(post_extractor.preds.flatten())

# --------------------------------- #
# Number recognition module  (End)  #
# --------------------------------- #
