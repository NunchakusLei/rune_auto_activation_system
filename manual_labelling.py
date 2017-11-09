#!/usr/bin/env python
import cv2
import argparse
import sys, os, numpy as np

def get_input(msg):
    """
    Get user input cross python 2 and 3
    """
    if sys.version_info[0] == 3:
        input_content = input(msg)
    elif sys.version_info[0] == 2:
        input_content = raw_input(msg)
    return input_content

def parse_user_input():
    """
    parse the command line input argument
    """
    description = 'Manual labeling script.'
    parser = argparse.ArgumentParser(description=description,
                                     epilog='')

    parser.add_argument('-f','--file',
                        dest='labeling_file_path',
                        help='User input argument for the labeling file path.',
                        required=True)

    args = parser.parse_args(sys.argv[1:])

    return args

def try_open_video_file(labeling_file_path):
    """
    Try to open a video file using OpenCV
    """
    video_source = cv2.VideoCapture(labeling_file_path)
    if video_source.isOpened() == True :
        return video_source
    else:
        return None

class RuneLabel:
    def __init__(self):
        self.is_rune_in_frame = False
        self.grid_boding_box = [(None, None), (None, None)]
        self.handwritten_number_boding_box = []
        self.handwritten_number_position = [] # the center point of boding box

class FrameLabeler:
    def __init__(self, v_name, v_source):
        self.__video_name__ = v_name
        self.__video_source__ = v_source
        self.is_EOF = False
        self.label = []
        self.frame_num = 0

        self.drawing = False
        self.ix, self.iy = None, None
        self.ex, self.ey = None, None
        self.current_frame = None
        self.last_frame = None
        self.mode = "grid"

    def read_next_frame(self):
        ret, frame = self.__video_source__.read()
        if ret==False:
            self.is_EOF = True
        else:
            self.frame_num += 1
            self.drawing = False
            self.ix, self.iy = None, None
            self.ex, self.ey = None, None
            self.last_frame = self.current_frame
            self.current_frame = frame
            self.mode = "grid"

            cv2.imshow(self.__video_name__, self.current_frame)
        return frame

    def start_labeling(self):
        self.read_next_frame()
        cv2.setMouseCallback(self.__video_name__,
                                self.mouse_events_callback)

        while (True):
            if self.is_EOF:
                break
            self.keyboard_callback(cv2.waitKey(0) & 0xFF)()
            # print self.__video_source__.get(11)
            # cv2.imshow(self.__video_name__, self.read_next_frame())

    def mouse_events_callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.mode=="grid":
                    color = (0,255,0)
                else:
                    color = (255,0,255)
                drawed_frame = self.current_frame.copy()
                cv2.rectangle(drawed_frame,
                                (self.ix,self.iy),
                                (x,y),color,1)
                cv2.imshow(self.__video_name__,drawed_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.ex, self.ey = x,y

    def keyboard_callback(self, key):
        command = lambda: self.read_next_frame()
        if key == 27:
            self.is_EOF = True
            # print self.frame_num,"frames been labeled."
            command = lambda: exit()
        elif key == 13:
            command = ""
        elif key == ord("n"):
            command = lambda: self.read_next_frame()
        elif key == ord("c"):
            command = lambda: self.change_mode()
        elif key == ord("1"):
            cv2.putText(self.current_frame,
                        chr(key), (self.ix,self.iy),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (0,0,255))
            command = lambda: cv2.imshow(self.__video_name__, self.current_frame)
        else:
            print "The ord(key) ==", key
        return command

    def change_mode(self, mode=None):
        if mode==None:
            if self.mode=="grid":
                self.mode = "cell"
            else:
                self.mode = "grid"
        else:
            self.mode = mode

if __name__ == "__main__":
    """
    Main function for testing
    """
    # parse user input
    args = parse_user_input()

    # read the file that will be labeled
    labeling_file_dir = os.path.dirname(os.path.abspath(__file__))
    labeling_file_path = os.path.join(labeling_file_dir, args.labeling_file_path)

    video_source = try_open_video_file(labeling_file_path)
    assert(video_source!=None)

    labeler = FrameLabeler("test", video_source)
    labeler.start_labeling()
    # print get_input("What's the label:")


    cv2.destroyAllWindows()
