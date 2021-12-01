#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def concat_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(8, high=32, size=1))
    in_size_y  = int(np.random.randint(8, high=32, size=1))
    in_channel = int(np.random.randint(2, high=16, size=1))
    input_cn   = int(np.random.randint(2, high=5, size=1))
    con_axis   = int(np.random.randint(0, high=4, size=1))

    src_in = []
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_out    = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))
    src_in.append(src_out)

    for i in range(0, input_cn - 1):
        zero_point = int(np.random.randint(-6, high=6, size=1))
        std        = int(np.random.randint(1, high=20, size=1))
        src_in2    = np.random.normal(zero_point, std, (batch, in_channel, in_size_y, in_size_x))
        src_in.append(src_in2)
        src_out    = np.concatenate((src_out, src_in2), axis=con_axis)

    src_in_1 = []
    size_all  = batch*in_size_y*in_size_x*in_channel
    for i in range(0, input_cn):
        src_in_1.append(src_in[i].reshape(size_all))
    src_out_1 = src_out.reshape(size_all * input_cn)

    total_size = (len(src_in_1[0]) * input_cn + len(src_out_1)) + 6

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(input_cn)
    para.append(con_axis)
    print(para)

    with open("concat_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        for i in range(0, input_cn):
            data = struct.pack(('%df' % len(src_in_1[i])), *src_in_1[i])
            fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    concat_f32()
    print("end")
