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
    dim_count   = int(np.random.randint(4, high=8, size=1))
    for i in range(0, dim_count):
        in_size = int(np.random.randint(1, high=32, size=1))
        dim.append(in_size)

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point, std, size=dim)
    src_in = src_in.astype(np.float32)

    mean   = np.random.normal(zero_point, std, dim[-1])

    zero_point = int(np.random.randint(1, high=20, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    var   = np.random.normal(zero_point, std, dim[-1])
    gamma =  np.random.normal(zero_point, std, dim[-1])
    beta  =  np.random.normal(zero_point, std, dim[-1])

    value = (1e-05, 1e-04, 1e-03)
    epsi  = random.sample(value, 1)
    out_calcu = tf.nn.batch_normalization(src_in, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=epsi)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + len(mean) * 4 + len(dim) + 2

    para.append(total_size)
    para.append(len(dim))


    with open("batch_norm_data_f32.bin", "wb") as fp:
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
