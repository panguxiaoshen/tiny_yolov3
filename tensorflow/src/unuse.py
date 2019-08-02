import numpy as np
import tensorflow as tf
a = np.array([
    [1,2,3,4,5,6],
    [7,8,9,10,11,12],
    [13,14,15,16,17,18]])
# max = np.where(a==np.max(a[:,4]))
# print(a[max[0]])
b = tf.convert_to_tensor(a)
c = b[:,4]*b[:,5]
d = tf.gather(b,tf.nn.top_k(b[:,4],1).indices)
# max = tf.arg_max(c,0)
# print(c[max])
with tf.Session() as sess:
    print(d.eval())