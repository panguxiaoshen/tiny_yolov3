#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
from core.yolov3_tiny import YOLOV3_TINY
import os
import core.utils as utils
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/bu-0719/yolov3_test_loss=0.3038.ckpt-30"
# ckpt_file = "./checkpoint/yolov3_test_loss=0.3826.ckpt-1"
output_node_names = ["input/input_data", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
output_node_names1 = ["input/input_data", "output_all"]
output_node_names2 = ["input/input_data", "output"]
output_node_names3 = ["input/input_data", "output_all","index"]
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')
    model = YOLOV3_TINY(input_data, trainable=False)

    # sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)
    mbbox = tf.reshape(model.pred_mbbox, (-1, 6))
    lbbox = tf.reshape(model.pred_lbbox, (-1, 6))
    bboxes = tf.concat([mbbox,lbbox],name='output_all',axis=0)
    bboxes = tf.gather(bboxes,tf.nn.top_k(bboxes[:,4],1).indices)
    bboxes =tf.identity(bboxes,'output')
    # scores = bboxes[:, 4]*bboxes[:,5]
    # max_score_index = tf.arg_max(scores,0)
    # max_score_index = tf.identity(max_score_index, 'index')
    # best_box = bboxes[max_score_index,:]
    # best_box = tf.identity(best_box,'output')
    # print(best_box.shape)


    # pred_mbbox = model.pred_mbbox.eval()
    # pred_lbbox = model.pred_lbbox.eval()
    # pred_bbox = np.concatenate([np.reshape(model.pred_mbbox, (-1, 5 + num_classes)),
    #                                     np.reshape(model.pred_lbbox, (-1, 5 + num_classes))], axis=0)
    # bboxes = utils.postprocess_boxes(pred_bbox, (160,160), 160, 0.3)
    # bboxes = utils.nms(bboxes, 0.45, method='nms')
    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                input_graph_def  = sess.graph.as_graph_def(),
                                output_node_names = output_node_names2)

    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())

# converter = tf.lite.TFLiteConverter.from_session(sess,["input/input_data"],["pred_mbbox/concat_2"])
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(pb_file, ["input/input_data"],["output"],
                                                              input_shapes={"input/input_data":[1,160,160,1]})
tflite_model = converter.convert()
open('tf.tflite', "wb").write(tflite_model)
# converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_file, ["input/input_data"],["output"],
#                                                               input_shapes={"input/input_data":[1,160,160,1]} )





