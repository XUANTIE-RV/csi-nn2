#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random

def getpackn(test_dtype, test_vlen):
    if int(test_dtype) == 8:
        return int(test_vlen)/int(test_dtype)/2
    else:
        return int(test_vlen)/int(test_dtype)

def transpose_f32(test_dtype, test_vlen, test_type):
    para = []
    # init the input data and parameters
    input_dim_count = int(np.random.randint(4, high=5, size=1))
    input_shape = []

    if "trans4_0_1_2_3" in test_type:
        input_dim_count    = 4
        layout = np.array([0,1,2,3])
    elif "trans4_0_2_3_1" in test_type:
        input_dim_count    = 4
        layout = np.array([0,2,3,1])
    elif "trans4_0_2_1_3" in test_type:
        input_dim_count    = 4
        layout = np.array([0,2,1,3])
    elif "trans3_0_2_1" in test_type:
        input_dim_count    = 3
        layout = np.array([0,2,1])
    elif "trans4_0_1_3_2" in test_type:
        input_dim_count    = 4
        layout = np.array([0,1,3,2])
    elif "trans5_0_1_3_4_2" in test_type:
        input_dim_count    = 5
        layout = np.array([0,1,3,4,2])
    else:
        layout = np.arange(input_dim_count)
        np.random.shuffle(layout)
    for i in range(0, input_dim_count):
        input_shape.append(int(np.random.randint(12, high=24, size=1)))
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    src_in = np.random.normal(zero_point, std, (input_shape))
    src_out = src_in.transpose(layout)

    output_shape = src_out.shape

    src_in_1   = src_in.flatten()
    src_out_1 = src_out.flatten()



    total_size = (len(src_in_1) + len(src_out_1)) + 1 + input_dim_count*3

    para.append(total_size)
    para.append(input_dim_count)
    for i in range(0, input_dim_count):
        para.append(input_shape[i])
    for i in range(0, input_dim_count):
        para.append(layout[i])
    for i in range(0, input_dim_count):
        para.append(output_shape[i])
    print(para)

    with open("transpose_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    test_dtype = sys.argv[1]
    test_vlen = sys.argv[2]
    test_type = sys.argv[3]
    transpose_f32(test_dtype, test_vlen, test_type)
    print("end")
