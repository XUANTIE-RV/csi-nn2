#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def rsqrt_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(32, high=64, size=1))
    in_size_y  = int(np.random.randint(32, high=64, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))
    input_min  = int(np.random.randint(1, high=5, size=1))
    input_max  = int(np.random.randint(5, high=40, size=1))

    src_in = np.random.uniform(input_min, input_max, (batch, in_size_y, in_size_x, in_channel))

    src_in = src_in.astype(np.float32)

    out_calcu = tf.rsqrt(tf.convert_to_tensor(src_in))

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

    with open("rsqrt_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    rsqrt_f32()
    print("end")
