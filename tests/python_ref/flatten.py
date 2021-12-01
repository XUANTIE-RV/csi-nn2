#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np


def flatten_f32():
    para = []
    # init the input data and parameters
    in_dims  = int(np.random.randint(1, high=5, size=1))
    in_shape = []
    size_all = 1
    for i in range(0, in_dims):
        in_shape.append(int(np.random.randint(16, high=32, size=1)))
        size_all *= in_shape[i]
    
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, in_shape)
    
    src_out = src_in.flatten()

    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(in_shape) + len(src_out_1)) + 1

    para.append(total_size)
    para.append(in_dims)

    print(in_shape)

    with open("flatten_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(in_shape)), *in_shape)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    flatten_f32()
    print("end")
