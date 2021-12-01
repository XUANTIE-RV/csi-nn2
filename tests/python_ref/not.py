#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def not_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=6, size=1))
    in_shape = []
    in_size = 1
    src_out = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(10, high=30, size=1)))
        in_size *= in_shape[i]


    src_in = np.random.randint(1, 600000, in_shape)


    src_in_1  = src_in.flatten()
    for i in range(0, in_size):
        src_out.append(~(src_in_1[i]))

    src_in = src_in.astype(np.int32)

    src_out = np.array(src_out).astype(np.int32)
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    print(para)

    with open("not_data_u32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%di' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    not_f32()
    print("end")
