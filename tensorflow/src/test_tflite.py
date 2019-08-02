# Copyright 2019 The Tensortec Authors. All Rights Reserved.
#
# ==============================================================================

"""Base face detection module.

The job of a FaceDetection is, for an given image, to predict the face location
and scores.

It detect face by performing the following steps:
1) Preprocessing the image, resize it to 300*300.
2) Fill the resized image to the input tensor.
3) Get the output scoes and bounding boxes.

Note that FaceDetection only operate on detections from a single
image at a time, and only return an face with highest scores, so
any logic for applying a FaceDetection to multiple images must be
handled externally.
"""

import tensorflow as tf
import cv2
import numpy as np


class FaceDetection(object):
    """Face Detection to detect the face in an image."""

    def __init__(self,model_path):
        self._interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def __str__(self):
        return 'Face Detection handler:\n\
                    input: {}\n\
                    output: {}'.format(self.input_details, self.output_details)

    def __call__(self,
                 image):
        """Detecting face for an gray image

        Args:
            image: numpy array with shape (image_height, image_width)

        Returns:
            prob: probability of the return box
            sx: start x
            sy: start y
            ex: end x
            ey: end y

        Raises:
            ValueError: if dimension of the iamge is not 2
        """
        if len(image.shape) != 2:
            raise ValueError('dimension of the iamge must 2')

        org_height, org_width = image.shape
        input_data = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))
        input_data = np.reshape(input_data, (1, self.input_shape[2], self.input_shape[1], 1))
        input_data = input_data.astype(np.float32)
        # input_data = input_data/255.0
        # fill data and inference
        self._interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self._interpreter.invoke()

        # get output data
        output_boxes = self._interpreter.get_tensor(self.output_details[0]['index'])
        # ll = self._interpreter.get_tensor(self.output_details[1]['index'])
        # output_classes = self._interpreter.get_tensor(self.output_details[1]['index'])
        # output_scores = self._interpreter.get_tensor(self.output_details[2]['index'])
        # output_nums = self._interpreter.get_tensor(self.output_details[3]['index'])
        #
        # prob = output_scores[0][0]
        # box = output_boxes[0][0]
        #
        # sy = int(org_height * box[0])
        # sx = int(org_width * box[1])
        # ey = int(org_height * box[2])
        # ex = int(org_width * box[3])

        return output_boxes

tflite_path = "tf.tflite"
fd = FaceDetection(tflite_path)
print(fd)
# Read image from path
image_path = "test.jpg"
image= cv2.imread(image_path,0)
image = image/255.0
# print(image)
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


# Predict and get output
output_boxes = fd(image)
print(output_boxes)
# print(score,sx,sy,ex,ey)
#
# cv2.rectangle(image_rgb, (sx,sy), (ex,ey), (255,0,0), 5)