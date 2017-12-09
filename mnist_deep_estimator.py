"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from mnist_deep import deepnn

import tensorflow as tf
import cv2

class numberRecognizer:
    def __init__(self, model_dir="./model/model.ckpt"):
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])

        # Build the graph for the deep net
        self.y_conv, self.keep_prob = deepnn(self.x)

        # initialize the tensorflow session
        self.sess = tf.Session()

        # Load the trained model
        saver = tf.train.Saver()
        saver.restore(self.sess, model_dir)
        print("Model Loaded from %s." % model_dir)

    def predict(self, predict_data):
        feed_dict = {self.x: predict_data, self.keep_prob:1.0}
        probs = self.sess.run(self.y_conv, feed_dict=feed_dict)
        pred_number = self.sess.run(tf.argmax(probs, axis=1))
        # pred_number = int(pred_number)
        # probs = list(probs.flatten())
        return pred_number, probs




if __name__ == '__main__':
    """
    Main function as a usage example
    """
    img = cv2.imread("data/ROI.png", cv2.IMREAD_GRAYSCALE)
    test_data = (255 - img.reshape((1, img.shape[0] * img.shape[1])) )/ 255.

    recognizer = numberRecognizer()
    print(recognizer.predict(test_data))

    cv2.imshow("img", img)
    cv2.waitKey()

    cv2.destroyAllWindows()
