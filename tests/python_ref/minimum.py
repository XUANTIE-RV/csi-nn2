#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def minimum_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=6, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(8, high=16, size=1)))

    zero_point  = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
 
    src_in_1 = np.random.normal(zero_point, std, in_shape)
    src_in_2 = np.random.normal(zero_point, std, in_shape)
    src_in_1 = src_in_1.astype(np.float32)
    src_in_2 = src_in_2.astype(np.float32)

    out_calcu = tf.minimum(src_in_1,src_in_2)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_1  = src_in_1.flatten()
    src_in_2  = src_in_2.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    print(para)

    with open("minimum_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_2)), *src_in_2)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    minimum_f32()
    print("end")
