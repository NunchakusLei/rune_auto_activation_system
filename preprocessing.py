#!/usr/bin/env python
import cv2
import argparse
import sys, os, numpy as np
import scipy
from common_func import *
from manual_labelling import try_open_video_file
from mnist_deep_estimator import numberRecognizer
from prediction_selection import *


DEBUG = True

single_cell_area_ratio_lower = 0.00443
single_cell_area_ratio_upper = 0.0105
cell_weight_height_ratio = 16/28.0
cell_weight_height_ratio_toleration = 0.12

recognizer = numberRecognizer()

# def draw_box(img, points, color):
#     """
#     Draw a box
#     """
#     # cv2.line(img, (0,0), (100,100), color, 5)
#     for i in range(len(points)):
#         cv2.line(img, tuple(points[i]), tuple(points[(i+1)%(len(points))]), color, 10)



# def general_number_extractor(src_img):
#     # convert source iamge to gray scale and resize
#     gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
#     gray = scipy.misc.imresize(gray, [50, 80])
#
#     # blur
#     # gray = cv2.medianBlur(gray,13)
#     blur = cv2.GaussianBlur(gray,(5,5),0)
#
#     # threshold
#     # ret, gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                                  cv2.THRESH_BINARY, 15, 3)
#     # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
#     # bit wise inverse
#     # gray = cv2.bitwise_not(gray)
#
#     ######################
#     kernel = np.ones([3, 3], np.uint8)
#     gray = cv2.dilate(gray, kernel, iterations = 1)
#
#     """
#     # Mask used to flood filling.
#     # Notice the size needs to be 2 pixels than the image.
#     h, w = gray.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
#     # print(mask)
#     # print(gray)
#
#     # Floodfill from point (0, 0)
#     # cv2.floodFill(gray, mask, (50,0),255);
#
#     # print(gray[49][0])
#
#     gray = cv2.erode(gray, kernel, iterations = 1)
#     """
#
#     return gray



# def rim_extractor(src_img, color):
#     # select color that draw by us
#     defined_color = np.array(list(color))
#     gray = cv2.inRange(src_img, defined_color, defined_color)
#
#     # resize
#     gray = scipy.misc.imresize(gray, [50, 80])
#
#     # threshold
#     ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
#     # enlarge rim a little bit
#     kernel = np.ones([5, 5], np.uint8)
#     gray = cv2.dilate(gray, kernel, iterations = 1)
#
#
#     return gray


def sort_box_points(box):
    """
    This function will sort box points in TL->BL->BR->TR order
    params box: 4 points of the box
        type: 2d np.array
    return out: 4 sorted points of the box
        tyep: 2d np.array
    """
    out = box.copy()
    xs, ys = [], []
    for i in range(4):
        xs.append(box[i][0])
        ys.append(box[i][1])
    xs.sort()
    ys.sort()

    for i in range(4):
        if (box[i][0] in xs[:2]) and (box[i][1] in ys[:2]):
            out[0] = box[i]
        elif (box[i][0] in xs[:2]) and (box[i][1] in ys[2:]):
            out[1] = box[i]
        elif (box[i][0] in xs[2:]) and (box[i][1] in ys[2:]):
            out[2] = box[i]
        elif (box[i][0] in xs[2:]) and (box[i][1] in ys[:2]):
            out[3] = box[i]

    return out


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



# """
# This function will extract roi after the number boxes have been found
# """
# def preprocess_for_number_recognition(src_img, rects, number_boxes):
#     global draw_number_box_color
#     number_boxes_regions_list = list()
#     box_index = 0
#
#     for box in number_boxes:
#
#         # prepare for extracting process
#         general_number_temp = general_number_extractor(region_of_interest(src_img, box))
#         draw_box(src_img, box, draw_number_box_color) # draw the rim
#         rim_temp = rim_extractor(region_of_interest(src_img, box), draw_number_box_color)
#
#         box_center = rects[box_index][0]
#         cv2.circle(src_img, (int(round(box_center[0])), int(round(box_center[1]))), 1, (0,0,255), 5)
#
#
#         # extracting
#         roi_temp = rim_temp + general_number_temp
#
#         kernel = np.ones([3, 3], np.uint8)
#         extracted_result = cv2.erode(roi_temp, kernel, iterations = 1)
#
#         number_boxes_regions_list.append(extracted_result)
#
#         # update loop variable
#         box_index += 1
#
#     return number_boxes_regions_list


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


def number_recognizing_preprocess(src):
    # image Processing
    img = src.copy()
    img = 255 - img                     # inverse
    mask = get_rim_mask(src.shape, 0.35)
    img = normalize(img * (mask**10))   # rim removal
    # """
    # img = 255 - img # inverse
    # img = cv2.erode(img, np.ones((3,3),np.uint8), iterations=1)
    # img = normalize((img.astype(np.float32))**3)
    # threshold
    # ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret,img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    # img = 255 - img # inverse
    # """
    # data preparation
    test_data = img.reshape((1, src.shape[0] * src.shape[1])) / 255.


    return test_data, img



"""
This function get rid of redundancy number boxes
"""
def filter_redundancy_boxes(contours, rects, number_boxes):
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
def filter_outlier_boxes(contours, rects, number_boxes):
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



def preprocessing(src):
    img = src.copy()
    # apply basic image processing techniques
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # get gray scale image
    img = gray
    img = cv2.blur(img, (5,5))                  # blur image
    # cv2.imshow("pre_frame.png", img)
    # cv2.waitKey()

    # """
    img = (img.astype(np.float64))**2
    img = cv2.GaussianBlur(img, (3,3), 0)       # blur image again

    # print np.max(img), np.min(img)
    # print img.max(), img.min()
    img_sharp_gray = normalize(img).copy()
    img = img_sharp_gray
    # """
    """
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    img = cv2.Canny(img, 100, 100, apertureSize=3, L2gradient=True)
    """
    img = cv2.Canny(img, 400, 200, apertureSize=3, L2gradient=True)
    # img = cv2.Canny(img, 500, 200)              # find edge

    # img = cv2.erode(img, np.ones((3,3),np.uint8), iterations=1)
    img = cv2.dilate(img, np.ones((3,3),np.uint8), iterations=3)

    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(src, contours, -1, (255,0,0), 3)


    boxes = []
    rects = []
    src_pts = []
    central_mass_point = [0,0]
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
            boxes.append(box)
            rects.append(rect)
            #cv2.circle(src,(int(rect[0][0]),int(rect[0][1])),8,(0,255,0),-2)

            # raw_input()
            # print rect[1][0]*rect[1][1]
            # print cv2.contourArea(contour)/(rect[1][0]*rect[1][1])
            # print min(rect[1])/max(rect[1])
            # print rect

            src_pts.append((int(rect[0][0]),int(rect[0][1])))
            central_mass_point[0] += rect[0][0]
            central_mass_point[1] += rect[0][1]

    if len(boxes)!=0:
        central_mass_point[0] = central_mass_point[0]/len(boxes)
        central_mass_point[1] = central_mass_point[1]/len(boxes)

    contours, rects, boxes, _ = filter_redundancy_boxes(contours, rects, boxes)
    contours, rects, boxes, _ = filter_outlier_boxes(contours, rects, boxes)


    if len(boxes)>9:
        # cv2.circle(src,(int(central_mass_point[0]),int(central_mass_point[1])),18,(155,255,0),-2)
        """
        pair_src_pts, pair_dst_pts = [], []
        dst_pts = [(0,0),(0,370),(0,740),(220,0),(220,370),(220,740),(440,0),(440,370),(440,470)]
        for i in range(len(src_pts)):
            for j in range(len(dst_pts)):
                pair_src_pts.append(src_pts[i])
                pair_dst_pts.append(dst_pts[j])
        dst_pts_H = np.float32(pair_dst_pts).reshape(-1,1,2)
        src_pts_H = np.float32(pair_src_pts).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts_H, dst_pts_H, cv2.RANSAC, 1.0)
        matchesMask = mask.ravel().tolist()

        dst_pts = np.float32(dst_pts).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(dst_pts,M)
        det_size = dst.astype(np.uint8).shape
        dst = dst.astype(np.uint8).reshape(det_size[0],det_size[2])
        for mask_i in range(len(matchesMask)):
            if matchesMask[mask_i]==1:
                cv2.circle(src,pair_src_pts[mask_i],8,(255,255,0),-2)
        src = cv2.polylines(src,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # print list()
        # print len(boxes)
        print np.array(matchesMask).reshape(len(src_pts),len(dst_pts))
        """

    # draw boxes
    boxes_int = []
    for box in boxes:
        box_int = np.int0(box)
        boxes_int.append(box_int)
    cv2.drawContours(src,boxes_int,-1,(0,0,255),2)


    # extracting ROI
    side_size = 28
    if len(boxes)>1:
        dst = None
        pred_data = None
        for box in boxes:
        # for i in range(1):
        #     box = boxes[0]
            roi = region_of_interest(gray, box, side_size=side_size)
            one_pred_data, roi = number_recognizing_preprocess(roi)

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

        # perfom prediction
        prediction, probabilities = recognizer.predict(pred_data)

        # apply constrain rule
        post_extractor = predExtractor(prediction, probabilities, boxes)
        perplexity = post_extractor.get_perplexity()
        # print post_extractor.get_pred()
        # post_extractor.number_redundancy_removal()

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
        # print prediction.dtype


        # print prediction
        if perplexity <= 0.25:
            # post_extractor.number_redundancy_removal2()
            if len(boxes)<10:
                post_extractor.number_redundancy_removal3()

            # print post_extractor.find_missed_number()

            post_extractor.draw_pred(src)

        cv2.imshow("perspective", dst)
        # cv2.imwrite("ROI.png", dst)

        # print prediction
        # print probabilities
        # sorted_p = np.sort(probabilities)
        # print sorted_p[:,9] - sorted_p[:,8]




    # print len(contours)
    # """
    return img



def decode_src_and_feed_preprocessing(video_src, fps=24, src_type='image'):
    # set up option variables
    key = None
    if src_type=='image':
        wait_key_time = 0
    else:
        wait_key_time = 1000//fps

    # loop to read each frame
    last_frame = None
    while True:
        ret, frame = video_src.read()
        if not ret or key==ord('q'):
            break
        elif key==ord('s'):
            cv2.imwrite("pre_frame.png", last_frame)
        elif key==ord('p'):
            if wait_key_time==0:
                wait_key_time = 1000//fps
            else:
                wait_key_time = 0

        # print 480./(frame.shape)[1]
        # frame = cv2.resize(frame, None, None, 480./frame.shape[1], 480./frame.shape[1])
        last_frame = frame.copy()
        img = preprocessing(frame) # do the preprocessing
        cv2.imshow("preprocessing visualize", frame)
        key = cv2.waitKey(wait_key_time) & 0xFF
    return



def parse_user_input():
    """
    Parse the command line input argument

    usage: preprocessing.py [-h] -f INPUT_FILE_PATH (-v | -i)

    Preprocessing script.

    optional arguments:
      -h, --help            show this help message and exit
      -f INPUT_FILE_PATH, --file INPUT_FILE_PATH
                            User input argument for the image source file path.
      -v, --video           Input source type indicated as video.
      -i, --image           Input source type indicated as image.
    """
    description = 'Preprocessing script.'
    parser = argparse.ArgumentParser(description=description,
                                     epilog='')

    requiredNamed = parser.add_argument_group('required arguments')
    src_group = requiredNamed.add_mutually_exclusive_group(required=True)
    src_group.add_argument('-f','--file',
                        dest='input_file_path',
                        help='User input argument for the image source file path.')
    src_group.add_argument('-c','--camera',
                        action='store_const',
                        const=0,
                        dest='input_file_path',
                        help='Feed with a webcam.')

    group = requiredNamed.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--video',
                        action='store_const',
                        const="video",
                        dest='source_type',
                        help='Input source type indicated as video.')
    group.add_argument('-i', '--image',
                        action='store_const',
                        const="image",
                        dest='source_type',
                        help='Input source type indicated as image.')

    args = parser.parse_args(sys.argv[1:])

    return args



if __name__ == "__main__":
    """
    Main function for testing
    """
    # parse user input
    args = parse_user_input()

    # if(args.input_file_path)
    # # read the input file
    # input_file_dir = os.path.dirname(os.path.abspath(__file__))
    # input_file_path = os.path.join(input_file_dir, args.input_file_path)
    video_source = try_open_video_file(args.input_file_path)
    # video_source = try_open_video_file(0)
    assert(video_source!=None)

    # decode source and feed preprocessing
    decode_src_and_feed_preprocessing(video_source, src_type=args.source_type)

    cv2.destroyAllWindows()
