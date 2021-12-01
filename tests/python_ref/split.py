#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def split_f32():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(2, high=8, size=1))
    in_size_x  = int(np.random.randint(128, high=512, size=1))
    in_size_y  = int(np.random.randint(128, high=512, size=1))
    in_channel = int(np.random.randint(4, high=64, size=1))
    axis_split = int(np.random.randint(0, high=3, size=1))

    num_split  = int(np.random.randint(2, high=4, size=1))
    pad0  = batch -  int(batch / num_split) * num_split
    pad1  = in_size_y -  int(in_size_y / num_split) * num_split
    pad2  = in_size_x -  int(in_size_x / num_split) * num_split
    pad3  = in_channel -  int(in_channel / num_split) * num_split
    batch = batch + pad0
    in_size_y = in_size_y + pad1
    in_size_x = in_size_x + pad2
    in_channel = in_channel + pad3

    zero_point = int(np.random.randint(-60000, high=60000, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, (batch, in_size_y, in_size_x, in_channel))

    out_calcu = tf.split(src_in, num_split, axis=axis_split)

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    src_in_1  = src_in.flatten()
    src_out_1 = []
    for i in range(0, num_split):
        src_out_1.append(src_out[i].flatten())

    total_size = (len(src_in_1) + len(src_out_1[0]) * num_split) + 6

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    para.append(axis_split)
    para.append(num_split)

    with open("split_f32_data.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        for i in range(0, num_split):
            data = struct.pack(('%df' % len(src_out_1[i])), *src_out_1[0])
            fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    split_f32()
    print("end")
