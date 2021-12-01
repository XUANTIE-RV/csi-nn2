#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
import random

def matmul_f32():
    para = []
    dim0 = []
    dim1 = []
    # init the input data and parameters
    dim_count   = int(np.random.randint(4, high=6, size=1))
    for i in range(0, dim_count-2):
        in_size = int(np.random.randint(1, high=16, size=1))
        dim0.append(in_size)
        dim1.append(in_size)

    zero_point1 = int(np.random.randint(-6, high=6, size=1))
    std1        = int(np.random.randint(1, high=20, size=1))
    zero_point2 = int(np.random.randint(-6, high=6, size=1))
    std2        = int(np.random.randint(1, high=20, size=1))

    trans_a_flag = False
    trans_b_flag = False

    in_sizei = int(np.random.randint(1, high=32, size=1))
    in_sizek = int(np.random.randint(1, high=32, size=1))
    in_sizej = int(np.random.randint(1, high=32, size=1))

    trans_a = random.choice((0, 1))
    trans_b = random.choice((0, 1))
    if (trans_a == 0 and trans_b == 0):
        dim0.append(in_sizei)
        dim0.append(in_sizek)
        dim1.append(in_sizek)
        dim1.append(in_sizej)
    elif(trans_a == 1 and trans_b == 0):
        dim0.append(in_sizek)
        dim0.append(in_sizei)
        dim1.append(in_sizek)
        dim1.append(in_sizej)
        trans_a_flag = True
    elif (trans_a == 0 and trans_b == 1):
        dim0.append(in_sizei)
        dim0.append(in_sizek)
        dim1.append(in_sizej)
        dim1.append(in_sizek)
        trans_b_flag = True
    else:
        dim0.append(in_sizek)
        dim0.append(in_sizei)
        dim1.append(in_sizej)
        dim1.append(in_sizek)
        trans_a_flag = True
        trans_b_flag = True

    src_in0 = np.random.normal(zero_point1, std1, size=dim0)
    src_in1 = np.random.normal(zero_point2, std2, size=dim1)
    src_in0 = src_in0.astype(np.float32)
    src_in1 = src_in1.astype(np.float32)

    out_calcu = tf.matmul(src_in0, src_in1, transpose_a=trans_a_flag, transpose_b=trans_b_flag)
    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_0  = src_in0.flatten()
    src_in_1  = src_in1.flatten()

    src_out_1 = src_out.flatten()

    total_size = len(src_in_0) + len(src_in_1) + len(src_out_1) + 3 * len(dim0) + 3

    para.append(total_size)
    para.append(trans_a)
    para.append(trans_b)
    para.append(len(dim0))

    with open("matmul_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%di' % len(dim0)), *dim0)
        fp.write(data)
        data = struct.pack(('%di' % len(dim1)), *dim1)
        fp.write(data)
        data = struct.pack(('%di' % len(np.shape(src_out))), *np.shape(src_out))
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_0)), *src_in_0)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    matmul_f32()
    print("end")
