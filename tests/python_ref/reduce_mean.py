#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

def reduce_mean_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(2, high=32, size=1))
    in_size_y  = int(np.random.randint(2, high=32, size=1))
    in_channel = int(np.random.randint(2, high=16, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=2, size=1))

    reduce_dim = int(np.random.randint(-1, high=4, size=1))
    reduce_shape = []

    for i in range(0, reduce_dim):
        reduce_shape.append(i)

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x , in_channel))
    # src_in = np.random.randint(-100, 100, (batch, in_size_y, in_size_x , in_channel))
    # src_in = src_in.astype(np.float32)

    if reduce_dim==-1:
        out_calcu = tf.reduce_mean(src_in)
    else:
        out_calcu = tf.reduce_mean(src_in, axis=reduce_dim)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    size_all  = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in.reshape(size_all)

    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(reduce_dim)
    print(para)

    with open("reduce_mean_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    reduce_mean_f32()
    print("end")
