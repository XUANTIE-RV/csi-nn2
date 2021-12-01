#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


'''
tvm only support 2D input:2-D with shape [batch_size, num_classes]
so, in_dim = 2 and axis = 0
'''
def log_softmax_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(2, high=3, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(32, high=256, size=1)))

    axis_in     = int(np.random.randint(1, high=2, size=1))
    zero_point  = int(np.random.randint(-6, high=6, size=1))
    std         = int(np.random.randint(1, high=3, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    src_in = src_in.astype(np.float32)

    # src_in = [[1.0, 2.0], [2.0, 1.0]]

    out_calcu = tf.nn.log_softmax(src_in)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_1  = src_in.ravel('C')
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 2 + in_dim

    para.append(total_size)
    para.append(axis_in)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    print(para)

    with open("log_softmax_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    log_softmax_f32()
    print("end")
