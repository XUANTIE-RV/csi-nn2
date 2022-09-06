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
    batch      = int(np.random.randint(2, high=4, size=1))
    in_size_x  = int(np.random.randint(16, high=17, size=1))
    in_size_y  = int(np.random.randint(32, high=33, size=1))
    in_channel = int(np.random.randint(16, high=17, size=1))


    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(50, high=60, size=1))

    axis_count = int(np.random.randint(1, high=2, size=1))  # must be 1 for anole
    axis_dim = [2, 3]
    axis_shape = random.sample(axis_dim, axis_count)

    keep_dim = int(np.random.randint(0, high=2, size=1))    # o:false   1:true

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    out_calcu = tf.reduce_mean(src_in, axis=axis_shape, keep_dims= True if keep_dim else False)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    size_all  = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in.reshape(size_all)

    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 4 + 2 + axis_count

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(keep_dim)
    para.append(axis_count)
    for i in range(0, axis_count):
        para.append(axis_shape[i])
    print(para)
    print(src_out.shape)


    with open("mean_graph_data_f32.bin", "wb") as fp:
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
