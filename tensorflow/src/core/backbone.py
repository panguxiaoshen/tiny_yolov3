#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32/2), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32/2,  64/2),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64/2,  32/2, 64/2, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64/2, 128/2),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128/2,  64/2, 128/2, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128/2, 256/2),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 256/2, 128/2, 256/2, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256/2, 512/4),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 512/4, 256/4, 512/4, trainable=trainable, name='residual%d' %(i+5))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512/4, 1024/4),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 1024/4, 512/4, 1024/4, trainable=trainable, name='residual%d' %(i+7))

        return route_1, route_2, input_data




