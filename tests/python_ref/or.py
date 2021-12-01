#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def or_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(5, high=10, size=1)))

    # src_in = np.random.normal(zero_point, std, in_shape)
    src_in_1 = np.random.randint(16, high=32, size=in_shape)
    src_in_1 = src_in_1.astype(np.int32)

    src_in_2 = np.random.randint(16, high=32, size=in_shape)
    src_in_2 = src_in_2.astype(np.int32)

    src_out = src_in_1 | src_in_2

    src_in_1  = src_in_1.flatten()
    src_in_2  = src_in_2.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1) + len(src_in_2)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    print(para)

    with open("or_data_u32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(src_in_2)), *src_in_2)
        fp.write(data)
        data = struct.pack(('%di' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    or_f32()
    print("end")
