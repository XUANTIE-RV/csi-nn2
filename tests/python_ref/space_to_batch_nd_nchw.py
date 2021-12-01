#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def space_to_batch_nd_f32():
    para = []
    # init the input data and parameters
    in_shape = []
    batch      = int(np.random.randint(1, high=4, size=1))
    in_shape.append(batch)

    block_shape = []
    paddings = []
    spatial_shape = []
    spatial_shape_cnt = int(np.random.randint(2, high=3, size=1))
    for i in range(0, spatial_shape_cnt):
        dim_tmp = int(np.random.randint(16, high=32, size=1))
        spatial_shape.append(dim_tmp)
        in_shape.append(dim_tmp)

        block_size = int(np.random.randint(1, high=4, size=1))   # block_size of each spatial dim
        block_shape.append(block_size)

        pad_start = pad_end = 0
        pad_tmp = dim_tmp - int(dim_tmp / block_size) * block_size
        if(pad_tmp != 0):
            pad_tmp = (int(dim_tmp / block_size) + 1) * block_size - dim_tmp
            pad_start = int(np.random.randint(0, high=pad_tmp + 1, size=1))
            pad_end   = pad_tmp - pad_start
        pad = [pad_start, pad_end]
        paddings.append(pad)


    remain_shape = []
    remain_shape_cnt = int(np.random.randint(1, high=2, size=1))
    for i in range(0, remain_shape_cnt):
        dim_tmp = int(np.random.randint(8, high=16, size=1))
        remain_shape.append(dim_tmp)
        in_shape.append(dim_tmp)


    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)

    out_calcu = tf.space_to_batch_nd(src_in, block_shape, paddings)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1 = np.transpose(src_in, [0, 3, 1, 2])
    src_out_1 = np.transpose(src_out, [0, 3, 1, 2])
    src_in_1  = src_in_1.flatten()
    src_out_1 = src_out_1.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 2 + len(in_shape) + spatial_shape_cnt * 3

    para.append(total_size)
    para.append(spatial_shape_cnt)
    para.append(remain_shape_cnt)
    para.append(in_shape[0])
    para.append(in_shape[3])
    para.append(in_shape[1])
    para.append(in_shape[2])

    for i in range(0, spatial_shape_cnt):
        para.append(block_shape[i])
        para.append(paddings[i][0])    # pad_start[i]
        para.append(paddings[i][1])    # pad_end[i]

    print(para)

    with open("space_to_batch_nd_data_nchw_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    space_to_batch_nd_f32()
    print("end")
