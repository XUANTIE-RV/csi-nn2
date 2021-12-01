#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def lrn_f32():
    para_int = []
    para_float = []
    # init the input data and parameters

    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(16, high=32, size=1))
    in_size_y  = int(np.random.randint(16, high=32, size=1))
    in_channel = int(np.random.randint(15, high=32, size=1))

    depth_radius = int(np.random.randint(1, high=6, size=1))
    bias = float(np.random.uniform(1, high=5, size=1))
    alpha = float(np.random.uniform(1e-5, high=1e-3, size=1))
    beta = float(np.random.uniform(0.5, high=1, size=1))

    zero_point = int(np.random.randint(0, high=1, size=1))
    std        = int(np.random.randint(1, high=2, size=1))

    src_in    = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))
    src_in = src_in.astype(np.float32)


    out_calcu = tf.nn.local_response_normalization(src_in, depth_radius, bias, alpha, beta)
    

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_nhwc = src_in
    out_nhwc    = src_out

    src_in_nchw = np.transpose(src_in, [0, 3, 1, 2])
    out_nchw = np.transpose(src_out, [0, 3, 1, 2])


    src_in_1  = src_in_nchw.flatten()
    src_out_1 = out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 8

    para_int.append(total_size)
    para_int.append(batch)
    para_int.append(in_channel)
    para_int.append(in_size_y)
    para_int.append(in_size_x)
    para_int.append(depth_radius)
    para_float.append(bias)
    para_float.append(alpha)
    para_float.append(beta)
    print(para_int)
    print(para_float)

    with open("lrn_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para_int)), *para_int)
        fp.write(data)
        data = struct.pack(('%df' % len(para_float)), *para_float)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    lrn_f32()
    print("end")
