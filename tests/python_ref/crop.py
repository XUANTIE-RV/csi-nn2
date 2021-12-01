#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf


def stride_slice_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(2, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(8, high=32, size=1)))

    axis = int(np.random.randint(1, high=in_dim, size=1))
    slice_count = in_dim - axis

    begin = []
    end = []
    stride = []
    for i in range(0, axis):
        begin.append(0)
        end.append(in_shape[i])
        stride.append(1)

    for i in range(0, slice_count):
        begin.append(int(np.random.randint(0, high=in_shape[axis + i], size=1)))
        end.append(int(np.random.randint(begin[axis+i]+1, high=in_shape[axis+i]+1, size=1)))
        stride.append(int(np.random.randint(1, high=2, size=1)))


    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=40, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    out_calcu = tf.strided_slice(src_in, begin=begin, end=end, strides=stride)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 3 + in_dim + 3 * in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    para.append(axis)
    for i in range(0, in_dim):
        para.append(begin[i])
        para.append(end[i])
        para.append(stride[i])
    para.append(len(src_out_1))

    print(para)
    print(src_out.shape)
    print(begin)
    print(end)
    print(stride)

    with open("crop_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    stride_slice_f32()
    print("end")
