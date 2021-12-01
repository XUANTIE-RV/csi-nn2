#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf
from torch import tensor

def space_to_depth_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(2, high=8, size=1))
    in_channel = int(np.random.randint(2, high=32, size=1))
    height_temp  = int(np.random.randint(8, high=64, size=1))
    width_temp   = int(np.random.randint(8, high=64, size=1))
    block_size  = int(np.random.randint(2, high=5, size=1))
    
    in_height = height_temp * block_size
    in_width = width_temp * block_size


    zero_point = int(np.random.randint(-600, high=600, size=1))
    std        = int(np.random.randint(1, high=20, size=1))
    
    src_in = np.random.normal(zero_point, std, (batch, in_height, in_width, in_channel))

    out_calcu = tf.space_to_depth(src_in, block_size = block_size)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_nchw = src_in.transpose(0, 3, 1, 2)
    src_out_nchw = src_out.transpose(0, 3, 1, 2)

    src_in_1  = src_in_nchw.flatten()
    src_out_1 = src_out_nchw.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 5

    para.append(total_size)
    para.append(batch)
    para.append(in_channel)
    para.append(in_height)
    para.append(in_width)
    para.append(block_size)

    with open("space_to_depth_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    space_to_depth_f32()
    print("end")
