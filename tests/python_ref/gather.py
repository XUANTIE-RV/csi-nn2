#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

# a = tf.Variable([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
# index_a = tf.Variable([0,1,1,2,3])
# b = tf.Variable([1,2,3])
# index_b = tf.Variable([0,1,2,1])
 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.gather(a, index_a)))
#     print(sess.run(tf.gather(b, index_b)))

def gather_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(16, high=128, size=1)))

    indices_cnt = int(np.random.randint(1, high=64, size=1))
    indices = []
    for i in range(0, indices_cnt):
        indices.append(int(np.random.randint(0, high=in_shape[0]+1, size=1)))

    zero_point  = int(np.random.randint(-600, high=600, size=1))
    std         = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    src_in = src_in.astype(np.float32)

    out_calcu = tf.gather(src_in, indices)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_1  = src_in.ravel('C')
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 2 + in_dim + indices_cnt

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])

    para.append(indices_cnt)
    for i in range(0, indices_cnt):
        para.append(indices[i])
    print(para)

    with open("gather_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    gather_f32()
    print("end")
