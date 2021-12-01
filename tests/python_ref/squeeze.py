#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

def squeeze_f32():
    para = []
    dims = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_size_y  = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(100, high=200, size=1))
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    dims.append(batch)
    dims.append(in_size_y)
    dims.append(in_size_x)
    dims.append(1)
    dims.append(1)
    dims.append(1)

    axis = [3,4,5]

    src_in    = np.random.normal(zero_point, std, dims)

    out_calcu = tf.squeeze(src_in, axis = axis)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(dims) + 3 + 1 + len(axis)

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(len(axis))
    for i in range(0, len(axis)):
        para.append(axis[i])
    print(para)

    with open("squeeze_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        # data = struct.pack(('%di' % len(dims)), *dims)
        # fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    squeeze_f32()
    print("end")
