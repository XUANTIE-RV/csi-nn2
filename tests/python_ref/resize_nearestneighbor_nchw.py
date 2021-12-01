#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
import random


def resize_nearest_neighbor_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=4, size=1))
    in_size_x  = int(np.random.randint(33, high=34, size=1))
    in_size_y  = int(np.random.randint(35, high=36, size=1))
    in_channel = int(np.random.randint(1, high=64, size=1))
    out_size_x = int(np.random.randint(65, high=66, size=1))
    out_size_y = int(np.random.randint(69, high=70, size=1))
    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    align_corners = random.sample((0, 1), 1)

    if(align_corners[0] == 0):
        alco = False
    else:
        alco = True

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))
    src_in = src_in.astype(np.float32)

    out_calcu = tf.image.resize_nearest_neighbor(tf.convert_to_tensor(src_in), [out_size_y, out_size_x], align_corners=alco)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_nchw = np.transpose(src_in, [0, 3, 1, 2])  #nhwc
    out_nchw   = np.transpose(src_out, [0, 3, 1, 2])


    src_in_1  = src_in_nchw.flatten()
    src_out_1 = out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 7

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(out_size_y)
    para.append(out_size_x)
    para.append(align_corners[0])
    print(para)

    with open("resize_nearestneighbor_nchw_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    resize_nearest_neighbor_f32()
    print("end")
