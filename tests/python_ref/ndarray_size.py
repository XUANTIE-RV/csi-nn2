#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np

def ndarray_size_f32():
    para = []
    # init the input data and parameters
    in_dim   = int(np.random.randint(1, high=4, size=1))
    in_shape = []
    src_out = 1
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(5, high=16, size=1)))
        src_out *= in_shape[i]

    zero_point  = int(np.random.randint(-600, high=600, size=1))
    std         = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    src_out = np.array([src_out])
    src_in = src_in.astype(np.float32)
    src_out = src_out.astype(np.float32)

    src_in_1  = src_in.flatten()
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + in_dim

    para.append(total_size)
    para.append(in_dim)
    for i in range(0, in_dim):
        para.append(in_shape[i])
    print(para)

    with open("ndarray_size_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    ndarray_size_f32()
    print("end")
