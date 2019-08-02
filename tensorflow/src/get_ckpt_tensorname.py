from tensorflow.python import pywrap_tensorflow
import os
model_dir = './checkpoint'
checkpoint_path = os.path.join(model_dir, "yolov3_test_loss=66.9507.ckpt-1")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)


