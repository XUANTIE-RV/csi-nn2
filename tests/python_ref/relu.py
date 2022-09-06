#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def relu_f32(test_type):
    para = []
    # init the input data and parameters
    if test_type == "random":
        batch       = int(np.random.randint(1, high=4, size=1))
        in_size_x   = int(np.random.randint(32, high=64, size=1))
        in_size_y   = int(np.random.randint(32, high=64, size=1))
        in_channel  = int(np.random.randint(1, high=64, size=1))
    elif test_type == "16x3_8_4_2_1":
        batch       = 1
        in_size_x   = 3
        in_size_y   = 3
        in_channel  = 7
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))

    out_calcu = tf.nn.relu(tf.convert_to_tensor(src_in))

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    size_all = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_out_1)) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    print(para)

    with open("relu_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    test_type = sys.argv[1]
    relu_f32(test_type)
    print("end")
