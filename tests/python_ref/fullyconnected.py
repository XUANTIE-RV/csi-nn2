#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def fullconnected_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(4, high=16, size=1))
    in_size     = int(np.random.randint(64, high=256, size=1))
    out_size    = int(np.random.randint(64, high=256, size=1))

    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))
    zero_point3 = int(np.random.randint(-6, high=6, size=1))
    std3        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, (batch, in_size))
    weight = np.random.normal(zero_point2, std2, (in_size, out_size))
    bias   = np.random.normal(zero_point3, std3, out_size)
    src_in = src_in.astype(np.float32)
    weight = weight.astype(np.float32)
    bias   = bias.astype(np.float32)

    out_calcu = tf.matmul(src_in, weight)
    sess = tf.Session()

    src_out = sess.run(out_calcu)
    src_out = np.add(src_out, bias)

    weight    = np.transpose(weight, [1, 0])
    src_in_1  = src_in.flatten()
    weight_1  = weight.flatten()
    src_out_1 = src_out.flatten()

    total_size = len(src_in_1) + len(src_out_1) + len(bias) + + len(weight_1) + 3

    para.append(total_size)
    para.append(batch)
    para.append(in_size)
    para.append(out_size)

    with open("fullyconnected_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(weight_1)), *weight_1)
        fp.write(data)
        data = struct.pack(('%df' % len(bias)), *bias)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    fullconnected_f32()
    print("end")
