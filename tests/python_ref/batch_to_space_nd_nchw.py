#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def batch_to_space_nd_f32():
    para = []
    # init the input data and parameters
    in_shape = []
    batch_temp = int(np.random.randint(1, high=4, size=1))
    batch = batch_temp
    in_shape.append(batch)

    block_shape = []
    crops = []
    spatial_shape = []
    spatial_shape_cnt = int(np.random.randint(2, high=3, size=1))
    for i in range(0, spatial_shape_cnt):
        dim_tmp = int(np.random.randint(8, high=16, size=1))
        spatial_shape.append(dim_tmp)
        in_shape.append(dim_tmp)

        block_size = int(np.random.randint(1, high=3, size=1))   # block_size of each spatial dim
        block_shape.append(block_size)
        batch *= block_size

        crop_start = int(np.random.randint(0, high=2, size=1))
        crop_end   = int(np.random.randint(0, high=2, size=1))
        crop = [crop_start, crop_end]
        crops.append(crop)

    remain_shape = []
    remain_shape_cnt = int(np.random.randint(1, high=2, size=1))
    for i in range(0, remain_shape_cnt):
        dim_tmp = int(np.random.randint(8, high=16, size=1))
        remain_shape.append(dim_tmp)
        in_shape.append(dim_tmp)

    in_shape[0] = batch     # batch = batch_temp * prod(block_size)


    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, in_shape)
    out_calcu = tf.batch_to_space_nd(src_in, block_shape, crops)
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
        para.append(crops[i][0])    # crop_start[i]
        para.append(crops[i][1])    # crop_end[i]

    print(para)

    with open("batch_to_space_nd_data_nchw_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    batch_to_space_nd_f32()
    print("end")
