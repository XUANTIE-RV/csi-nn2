#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def stack_f32():
    para = []

    # init the input data and parameters
    in_tensor_num = int(np.random.randint(2, high=5, size=1))
    in_dim   = int(np.random.randint(1, high=5, size=1))
    in_shape = []
    for i in range(0, in_dim):
        in_shape.append(int(np.random.randint(10, high=20, size=1)))
    stack_axis = int(np.random.randint(0, high=in_dim+1, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = []
    for i  in range(0, in_tensor_num):
        src_in.append(np.random.normal(zero_point, std, in_shape))

    src_out    = np.stack(src_in, axis=stack_axis)

    src_in_1 = np.array(src_in).ravel('C')
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 3 + in_dim + 1

    para.append(total_size)
    para.append(in_tensor_num)
    para.append(stack_axis)
    para.append(in_dim+1)
    for i in range(0, in_dim+1):
        para.append(np.shape(src_out)[i])

    with open("stack_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()
    print(para)
    return 0


if __name__ == '__main__':
    stack_f32()
    print("end")
