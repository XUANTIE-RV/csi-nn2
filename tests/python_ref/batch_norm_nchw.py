#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
import random


def batch_norm_f32():
    para = []
    dim  = []
    # init the input data and parameters
    dim_count   = int(np.random.randint(4, high=5, size=1))
    for i in range(0, dim_count):
        in_size = int(np.random.randint(1, high=32, size=1))
        dim.append(in_size)

    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point1, std1, size=dim)
    src_in = src_in.astype(np.float32)

    # zero_point2 = int(np.random.randint(0, high=1, size=1))
    # std2        = int(np.random.randint(1, high=3, size=1))
    # mean   = np.random.normal(zero_point2, std2, dim[-1])
    mean = np.mean(src_in, axis = (0,1,2))  # len(mean) = channel_size

    # zero_point3 = int(np.random.randint(1, high=2, size=1))
    # std3        = int(np.random.randint(0, high=2, size=1))
    # var   = np.random.normal(zero_point3, std3, dim[-1])
    var = np.var(src_in, axis = (0,1,2))    # len(var) = channel_size

    zero_point4 = int(np.random.randint(1, high=2, size=1))
    std4        = int(np.random.randint(1, high=2, size=1))
    gamma =  np.random.normal(zero_point4, std4, dim[-1])
    beta  =  np.random.normal(zero_point4, std4, dim[-1])

    value = (1e-05, 1e-04, 1e-03)
    epsi  = random.sample(value, 1)
    out_calcu = tf.nn.batch_normalization(src_in, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=epsi)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_nchw = src_in.transpose(0, 3, 1, 2)
    src_out_nchw = src_out.transpose(0, 3, 1, 2)


    src_in_1  = src_in_nchw.flatten()
    src_out_1 = src_out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(mean) * 4 + len(dim) + 2

    para.append(total_size)
    para.append(len(dim))


    with open("batch_norm_nchw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(dim)), *dim)
        fp.write(data)
        data = struct.pack(('%df' % len(epsi)), *epsi)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(mean)), *mean)
        fp.write(data)
        data = struct.pack(('%df' % len(var)), *var)
        fp.write(data)
        data = struct.pack(('%df' % len(gamma)), *gamma)
        fp.write(data)
        data = struct.pack(('%df' % len(beta)), *beta)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    batch_norm_f32()
    print("end")
