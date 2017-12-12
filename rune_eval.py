#!/usr/bin/env python
import cv2
import argparse
import cPickle as pickle
import sys, os, numpy as np
from manual_labelling import *
from rune_activator import activate_rune


class runeEvaluator(FrameLabeler):
    """
    A subclass of FrameLabeler. Extend the evaluation function for it
    """
    def get_pred_label_pairs(self, preds_rects, labels_boxes):
        """
        pair structure (pred_index, label_index)
        """
        FP = 0
        pairs = []
        for i in range(len(preds_rects)):
            p_center = preds_rects[i][0]
            for j in range(len(labels_boxes)):
                tl = labels_boxes[j][0]
                br = labels_boxes[j][1]

                if p_center[0] > tl[0] and p_center[0] < br[0] and\
                   p_center[1] > tl[1] and p_center[1] < br[1]:
                    pairs.append((i, j))

        FP = len(preds_rects)-len(pairs)
        TN = len(labels_boxes)-len(pairs)
        return pairs, TN, FP

    def pair_compare(self, pairs, preds, labels):
        TP, TN, FP = 0, 0, 0
        for pair in pairs:
            if int(preds[pair[0]])==int(labels[pair[1]]):
                TP += 1
            else:
                TN += 1
                FP += 1
        return TP, TN, FP

    def eval(self, mode="show", fps=24):
        TP, TN, FP = 0, 0, 0
        for i in range(len(self.label)): # eval thru all frames
            # index = 1018
            a_TP, a_TN, a_FP, visual = self.eval_a_frame(i)
            TP += a_TP
            TN += a_TN
            FP += a_FP

            # dispaly graph for human
            if mode=="show":
                label = self.label[i]
                labeled = label.draw_label(visual)

                if i%fps==0:
                    print "Truth positive: %d" % TP
                    print "Truth negative: %d" % TN
                    print "False positive: %d" % FP
                    if (TN+TP)!=0:
                        print "----- Precision: %f" % (float(TP) / (TN+TP))
                    else:
                        print "----- Precision: NaN"

                cv2.imshow("visualize", labeled)
                cv2.waitKey(1000/fps)

        # dispaly graph for human
        if mode=="show":
            print "=============== Over All Results ==============="
            print "Truth positive: %d" % TP
            print "Truth negative: %d" % TN
            print "False positive: %d" % FP
            if (TN+TP)!=0:
                print "----- Precision: %f" % (float(TP) / (TN+TP))
            else:
                print "----- Precision: NaN"
        return TP, TN, FP

    def eval_a_frame(self, frame_index=None):
        if frame_index is None:
            index = self.frame_num
        else:
            index = frame_index
        TP, TN, FP = 0, 0, 0

        label = self.label[index]
        frame = label.raw_frame.copy()
        pred_labels, pred_rects = activate_rune(frame)

        if label.is_rune_in_frame:
            # print self.label[self.frame_num].raw_frame

            # print label.grid_boding_box
            # print label.handwritten_number_boding_boxes
            # print label.handwritten_number_positions
            # print label.handwritten_number_labels
            #
            # print pred_labels, pred_rects

            pairs, first_TN, first_FP = self.get_pred_label_pairs(pred_rects, label.handwritten_number_boding_boxes)
            TP, TN, FP = self.pair_compare(pairs, pred_labels, label.handwritten_number_labels)
            # print pairs
            TN += first_TN
            FP += first_FP
        else:
            FP += len(pred_rects)

        return TP, TN, FP, frame


def parse_user_input():
    """
    parse the command line input argument
    """
    description = 'Rune Auto-activation System evaluation script.'
    parser = argparse.ArgumentParser(description=description,
                                     epilog='')

    parser.add_argument('-f','--file',
                        dest='file_path',
                        help='User input argument for the testing file path.',
                        required=True)

    args = parser.parse_args(sys.argv[1:])

    return args



if __name__ == "__main__":
    """
    Main function
    """
    # parse user input
    args = parse_user_input()

    video_source = try_open_video_file(args.file_path)
    assert(video_source!=None)

    # evaluator = runeEvaluator("data/Competition2017_buff.mpeg", None)
    evaluator = runeEvaluator(args.file_path, None)
    evaluator.load_labeled_frames_from_file()
    evaluator.eval()
