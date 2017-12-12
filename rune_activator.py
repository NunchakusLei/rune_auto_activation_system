#!/usr/bin/env python
import cv2
import argparse
import sys, os, numpy as np
import scipy
from common_func import *
from manual_labelling import try_open_video_file
from mnist_deep_estimator import numberRecognizer
from prediction_selection import predExtractor
from preprocess import *


DEBUG = True
recognizer = numberRecognizer()


def activate_rune(src):
    predictions = None
    # grid searching
    pre, gray = grid_searching_preprocess(src.copy())
    rects, boxes = grid_searching(src, pre, gray)
    # handwritten number recognition
    if len(boxes)>1:
        pred_data, human= number_recognizing_preparation(gray, rects, boxes)
        cv2.imshow("cells", human)
        predictions = number_recognizing_prediction(src, pred_data, boxes, recognizer)
    # digit tube searching
    # digit tube recognition
    # prompt light
    return predictions, rects



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

        activate_rune(frame) # perform the algorithm

        cv2.imshow("rune activation visualize", frame)
        key = cv2.waitKey(wait_key_time) & 0xFF
    return



def parse_user_input():
    """
    Parse the command line input argument

    usage: rune_activator.py [-h] (-f INPUT_FILE_PATH | -c) (-v | -i)

    Rune Auto-activation System.

    optional arguments:
      -h, --help            show this help message and exit

    required arguments:
      -f INPUT_FILE_PATH, --file INPUT_FILE_PATH
                            User input argument for the image source file path.
      -c, --camera          Feed with a webcam.
      -v, --video           Input source type indicated as video.
      -i, --image           Input source type indicated as image.
    """
    description = 'Rune Auto-activation System.'
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
