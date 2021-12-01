#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np


def add_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(2, high=5, size=1))
    in_channel  = int(np.random.randint(8, high=16, size=1))
    in_size_y   = int(np.random.randint(16, high=32, size=1))
    in_size_x   = int(np.random.randint(16, high=32, size=1))
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))
    vector  = 0

    size_all = batch*in_channel*in_size_y*in_size_x

    src_in1 = np.random.normal(zero_point1, std1, (batch, in_channel, in_size_y, in_size_x))
    src_in1 = src_in1.astype(np.float32)

    if(len(sys.argv) == 1):
        src_in2 = np.random.normal(zero_point2, std2, (batch, in_channel, in_size_y, in_size_x))
        src_in2 = src_in2.astype(np.float32)
        size2   = size_all
        src_out = np.add(src_in1, src_in2)
    else:
        if(sys.argv[1] == "vector"):
            vector = 1
            src_in2 = np.random.normal(zero_point2, std2, (batch, in_channel, 1, 1))
            src_in2 = src_in2.astype(np.float32)
            size2   = batch * in_channel
            src_out = np.add(src_in1, src_in2)

    src_in_1  = src_in1.reshape(size_all)
    src_in_2  = src_in2.reshape(size2)

    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(vector)
    print(para)

    with open("add_graph_data_f32.bin", "wb") as fp:
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
    add_f32()
    print("end")
