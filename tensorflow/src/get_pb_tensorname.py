# coding:utf-8

import tensorflow as tf
from tensorflow.python.framework import graph_util

tf.reset_default_graph()  # 重置计算图
output_graph_path = './yolov3_coco.pb'
# output_graph_path = './test.pb'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    # 获得默认的图
    graph = tf.get_default_graph()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))

        tensor_name = [tensor.name for tensor in output_graph_def.node]
        print(tensor_name)
        print('---------------------------')
        # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
        summaryWriter = tf.summary.FileWriter('log_graph/', graph)
        num = 0
        for op in graph.get_operations():
            # print出tensor的name和值
            num = num + 1
            print(num, op.name, op.values())
