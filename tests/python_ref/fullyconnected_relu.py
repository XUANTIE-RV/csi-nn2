#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def fullconnected_relu_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size     = int(np.random.randint(128, high=512, size=1))
    out_size    = int(np.random.randint(128, high=512, size=1))
    zero_point1 = int(np.random.randint(-60000, high=60000, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-60000, high=60000, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))

    size_all = batch*in_size*out_size

    src_in = np.random.normal(zero_point1, std1, (batch, in_size))
    weight =  np.random.normal(zero_point2, std2, (in_size, out_size))
    zero_point3 = int(np.random.randint(-60000, high=60000, size=1))
    std3        = int(np.random.randint(1, high=20, size=1))
    bias = np.random.normal(zero_point3, std3, size=out_size)

    out_calcu = tf.nn.relu_layer(src_in, weight, biases=bias)
    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = len(src_in_1) + len(src_out_1) + 3
    total_size +=  len(bias)

    para.append(total_size)
    para.append(batch)
    para.append(in_size)
    para.append(out_size)

    with open("fullconnected_relu_data.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(bias)), *bias)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    fullconnected_relu_f32()
    print("end")
