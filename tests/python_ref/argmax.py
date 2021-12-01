#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

def argmax_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_channel = int(np.random.randint(128, high=256, size=1))
    in_size_y  = int(np.random.randint(128, high=256, size=1))
    in_size_x  = int(np.random.randint(128, high=256, size=1))
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in    = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))

    arg_axis  = int(np.random.randint(1, high=4, size=1))
    #arg_shape = random.sample(axis, arg_num)

    out_calcu = tf.argmax(src_in, axis=arg_axis)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5   #len(arg_shape) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(arg_axis)
    print(para)
    print(src_out.shape)

    with open("argmax_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        #data = struct.pack(('%df' % len(arg_shape)), *arg_shape)
        #fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    argmax_f32()
    print("end")
