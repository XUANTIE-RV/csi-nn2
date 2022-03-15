#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np


def expand_dims_f32():
    para = []
    # init the input data and parameters
    in_dims  = int(np.random.randint(1, high=4, size=1))
    in_axis  = int(np.random.randint(0, high=in_dims, size=1))
    in_shape = []
    size_all = 1
    for i in range(0, in_dims):
        in_shape.append(int(np.random.randint(128, high=256, size=1)))
        size_all *= in_shape[i]
    
    zero_point1 = int(np.random.randint(-60000, high=60000, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point1, std1, in_shape)
    
    src_out = np.expand_dims(src_in, axis=in_axis)

    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(in_shape) + len(src_out_1)) + 2

    para.append(total_size)
    para.append(in_dims)
    para.append(in_axis)
    print(para)
    print(in_shape)

    with open("expand_dims_data_f32.bin", "wb") as fp:
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
    expand_dims_f32()
    print("end")
