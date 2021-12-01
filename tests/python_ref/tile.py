#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf


def tile_f32():
    para = []
    # init the input data and parameters
    input_dim_count = int(np.random.randint(1, high=5, size=1))
    input_shape = []
    reps_shape = []
    in_size_all = 1
    reps_size_all = 1
    for i in range(0, input_dim_count):
        input_shape.append(int(np.random.randint(16, high=32, size=1)))
        reps_shape.append(int(np.random.randint(1, high=5, size=1)))
        in_size_all *= input_shape[i]
        reps_size_all *= reps_shape[i]

    zero_point = int(np.random.randint(-60, high=60, size=1))
    std        = int(np.random.randint(1, high=20, size=1))

    src_in = np.random.normal(zero_point, std, input_shape)

    out_calcu = tf.tile(src_in, reps_shape)

    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1  = src_in.reshape(in_size_all)
    src_out_1 = src_out.reshape(in_size_all * reps_size_all)

    total_size = (len(src_in_1) + len(src_out_1)) + 1 + input_dim_count + input_dim_count

    para.append(total_size)
    para.append(input_dim_count)
    for i in range(0, input_dim_count):
        para.append(input_shape[i])
    for i in range(0, input_dim_count):
        para.append(reps_shape[i])

    with open("tile_data_f32.bin", "wb") as fp:
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
    tile_f32()
    print("end")
