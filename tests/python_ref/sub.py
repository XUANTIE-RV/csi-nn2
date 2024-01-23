#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np


def sub_f32():
    para = []
    # init the input data and parameters
    batch       = int(np.random.randint(1, high=4, size=1))
    in_size_x   = int(np.random.randint(32, high=64, size=1))
    in_size_y   = int(np.random.randint(32, high=64, size=1))
    in_channel  = int(np.random.randint(1, high=64, size=1))
    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))

    out_shape = [batch, in_size_y, in_size_x, in_channel]

    if(sys.argv[1] == "common"):
        in0_dim = in1_dim = 4
        in0_shape = in1_shape = out_shape
    elif (sys.argv[1] == "a_scalar"):
        in0_dim = 1
        in1_dim = 4
        in0_shape = [1]
        in1_shape = out_shape
    elif (sys.argv[1] == "a_outer"):
        in0_dim = in1_dim = 4
        in0_shape = [out_shape[0], out_shape[1], 1, 1]
        in1_shape = out_shape
    elif (sys.argv[1] == "a_inner"):
        in0_dim = 2
        in1_dim = 4
        in0_shape = [out_shape[2], out_shape[3]]
        in1_shape = out_shape
    elif (sys.argv[1] == "b_scalar"):
        in0_dim = 4
        in1_dim = 1
        in0_shape = out_shape
        in1_shape = [1]
    elif (sys.argv[1] == "b_outer"):
        in0_dim = in1_dim = 4
        in0_shape = out_shape
        in1_shape = [out_shape[0], out_shape[1], 1, 1]
    elif (sys.argv[1] == "b_inner"):
        in0_dim = 4
        in1_dim = 2
        in0_shape = out_shape
        in1_shape = [out_shape[2], out_shape[3]]
    elif (sys.argv[1] == "broadcast_ab"):
        in0_dim = in1_dim = 4
        in0_shape = [out_shape[i] if np.random.choice([0, 1]) else 1 for i in range(len(out_shape))]
        in1_shape = [out_shape[i] if np.random.choice([0, 1]) else 1 for i in range(len(out_shape))]

    size1   = 1
    for i in in0_shape:
        size1 *= i
    size2   = 1
    for i in in1_shape:
        size2 *= i

    src_in1 = np.random.normal(zero_point1, std1, in0_shape)
    src_in1 = src_in1.astype(np.float32)
    src_in2 = np.random.normal(zero_point2, std2, in1_shape)
    src_in2 = src_in2.astype(np.float32)
    src_out = np.subtract(src_in1, src_in2)

    src_in_1  = src_in1.reshape(size1)
    src_in_2  = src_in2.reshape(size2)
    src_out_1 = src_out.reshape(src_out.size)

    total_size = (len(src_in_1) + len(src_in_2) + len(src_out_1)) + 4 + 2 + in0_dim + in1_dim

    para.append(total_size)         # 0
    para.append(src_out.shape[0])   # 1
    para.append(src_out.shape[1])   # 2
    para.append(src_out.shape[2])   # 3
    para.append(src_out.shape[3])   # 4
    para.append(in0_dim)            # 5
    para.append(in1_dim)            # 6
    for i in range(0, in0_dim):
        para.append(in0_shape[i])   # 7 ~ 7+in0_dim-1
    for i in range(0, in1_dim):
        para.append(in1_shape[i])   # 7+in0_dim ~ 7+in0_dim+in1_dim-1
    print(para)

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
