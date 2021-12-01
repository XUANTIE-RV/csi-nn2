#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def floor_divide_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_channel  = int(np.random.randint(16, high=64, size=1))

    in_height   = int(np.random.randint(64, high=128, size=1))
    in_width    = int(np.random.randint(64, high=128, size=1))

    zero_point1 = int(np.random.randint(2000, high=6000, size=1))
    std1        = int(np.random.randint(1000, high=1500, size=1))
    zero_point2 = int(np.random.randint(200, high=400, size=1))
    std2        = int(np.random.randint(10, high=20, size=1))


    size_all = batch * in_channel * in_height * in_width

    src_in1 = np.random.normal(zero_point1, std1, (batch, in_channel, in_height, in_width))
    src_in1 = src_in1.astype(np.float32)
    src_in2 = np.random.normal(zero_point2, std2, (batch, in_channel, in_height, in_width))
    src_in2 = src_in2.astype(np.float32)

    # src_out = np.floor_divide(src_in1, src_in2)
    out_calcu = tf.math.floordiv(src_in1, src_in2)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1  = src_in1.reshape(size_all)
    src_in_2  = src_in2.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)

    with open("floor_div_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_2)), *src_in_2)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    floor_divide_f32()
    print("end")
