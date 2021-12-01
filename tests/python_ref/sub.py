#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np


def sub_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(16, high=64, size=1))
    in_size_y   = int(np.random.randint(16, high=64, size=1))
    in_channel  = int(np.random.randint(1, high=64, size=1))
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))

    size_all = batch*in_size_y*in_size_x*in_channel

    src_in1 = np.random.normal(zero_point1, std1, (batch, in_size_y, in_size_x, in_channel))
    src_in1 = src_in1.astype(np.float32)

    src_in2 = np.random.normal(zero_point2, std2, (batch, in_size_y, in_size_x, in_channel))
    src_in1 = src_in1.astype(np.float32)

    src_out = np.subtract(src_in1, src_in2)



    src_in_1  = src_in1.reshape(size_all)
    src_in_2  = src_in2.reshape(size_all)

    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)

    with open("sub_data_f32.bin", "wb") as fp:
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
    sub_f32()
    print("end")
