#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

def reverse_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(2, high=8, size=1))
    in_channel = int(np.random.randint(16, high=128, size=1))
    in_height  = int(np.random.randint(32, high=128, size=1))
    in_width   = int(np.random.randint(32, high=128, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=3, size=1))

    axis = int(np.random.randint(0, high=4, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_channel, in_height, in_width))

    reverse_axis = []
    reverse_axis.append(axis)

    out_calcu = tf.reverse(src_in, axis=reverse_axis)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)
    para.append(reverse_axis[0])

    with open("reverse_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    reverse_f32()
    print("end")
