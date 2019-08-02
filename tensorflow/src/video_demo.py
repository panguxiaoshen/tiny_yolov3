#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import random


# return_elements = ["input/input_data:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
return_elements = ["input/input_data:0", "output:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "../hand_detection/hand_object_detection/test/2.mp4"
# video_path      = 0
num_classes     = 1
input_size      = 160
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)


def Contrast_and_Brightness(image, alpha, beta):
    blank = np.zeros_like(image, image.dtype)
    dst = cv2.addWeighted(image, alpha, blank, 1 - alpha, beta)
    return dst

def get_true_hand_box(box,width,height,input_size):
    xmin = box[0] - box[2]/2
    xmax = box[0] + box[2]/2
    ymin = box[1] - box[3]/2
    ymax = box[1] + box[3]/2
    resize_ratio = min(input_size / width, input_size / height)
    dw = (input_size - resize_ratio * width) / 2
    dh = (input_size - resize_ratio * height) / 2
    new_xmin = int((xmin-dw)/resize_ratio)
    new_xmax = int((xmax - dw) / resize_ratio)
    new_ymin = int((ymin - dh) / resize_ratio)
    new_ymax = int((ymax - dh) / resize_ratio)
    new_box = [new_xmin,new_ymin,new_xmax,new_ymax,box[4],box[5]]
    return new_box

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # alpha = random.random()+0.5
            # beta = random.random()+0.5
            # print(alpha,beta)
            # frame = Contrast_and_Brightness(frame, alpha, beta)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        img_width = frame.shape[1]
        img_height = frame.shape[0]
        # if img_width > img_height:
        #     padding_len = int((img_width - img_height) / 2)
        #     frame = cv2.copyMakeBorder(frame, padding_len, padding_len, 0, 0, cv2.BORDER_CONSTANT)
        # else:
        #     padding_len = int((img_height - img_width) / 2)
        #     frame = cv2.copyMakeBorder(frame, 0, 0, padding_len, padding_len, cv2.BORDER_CONSTANT)


        frame_size = frame.shape[:2]
        frame = np.expand_dims(frame,axis=-1)
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        # prev_time = time.time()
        # pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2]], feed_dict={ return_tensors[0]: image_data})
        # pred_bbox = np.concatenate([np.reshape(pred_mbbox, (-1, 5 + num_classes)),np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
        # print(pred_bbox.shape)
        pred_bbox = sess.run([return_tensors[1]],feed_dict={return_tensors[0]: image_data})
        pred_bbox = pred_bbox[0]
        true_box = get_true_hand_box(pred_bbox,img_width,img_height,input_size)
        if(true_box[4]>0.8):
            cv2.rectangle(frame,(true_box[0],true_box[1]),(true_box[2],true_box[3]),(255,0,0),3)
            cv2.imshow('test',frame)
            cv2.waitKey(100)
        # pred_bbox = np.array(pred_bbox).reshape(-1,6)
        # bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.7)
        # bboxes = utils.nms(bboxes, 0.65, method='nms')
        # image = utils.draw_bbox(frame, bboxes)
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("result", image)
        # cv2.waitKey(20)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break




