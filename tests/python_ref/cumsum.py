#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def cumsum_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(2, high=8, size=1))
    in_size_x   = int(np.random.randint(64, high=128, size=1))
    in_size_y   = int(np.random.randint(64, high=128, size=1))
    in_channel  = int(np.random.randint(4, high=128, size=1))

    zero_point1 = int(np.random.randint(-30, high=30, size=1))
    std1        = int(np.random.randint(1, high=10, size=1))

    dim         = int(np.random.randint(0, high=4, size=1))
    excl        = int(np.random.randint(0, high=2, size=1))

    size_all = batch*in_size_y*in_size_x*in_channel
    src_in1 = np.random.normal(zero_point1, std1, (batch, in_size_x, in_size_y, in_channel))
    src_in1 = src_in1.astype(np.float32)

    # src_out = np.cumsum(src_in1, axis = dim)
    out_calcu = tf.cumsum(src_in1, axis = dim, exclusive = True if excl else False)

    with tf.Session() as sess:
        src_out = sess.run(out_calcu)

    src_in_1  = src_in1.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_out_1)) + 6

    para.append(total_size)
    para.append(batch)
    para.append(in_size_x)
    para.append(in_size_y)
    para.append(in_channel)
    para.append(dim)
    para.append(excl)
    print(para)

    with open("cumsum_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    cumsum_f32()
    print("end")
