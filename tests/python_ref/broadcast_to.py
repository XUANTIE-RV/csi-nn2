#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import random
import tensorflow as tf

def broadcast_to_f32():
    para = []
    broadcast_shape = []
    # init the input data and parameters
    broadcast_dimcount  = int(np.random.randint(1, high=7, size=1))

    zero_point = int(np.random.randint(-6, high=6, size=1))
    std        = int(np.random.randint(1, high=2, size=1))

    for i in range(0, broadcast_dimcount):
        broadcast_shape.append(int(np.random.randint(1, high=32, size=1)))

    input_shape_t = []
    for i in range(0, broadcast_dimcount):
        if i != broadcast_dimcount -1 :
            choice_list = [0, 1, broadcast_shape[i]]
        else:
            choice_list = [1, broadcast_shape[i]]
        input_shape_t.append(int(np.random.choice(choice_list, 1)))

    input_shape = []
    for i in input_shape_t:
        if i != 0:
            input_shape.append(i)
        else:
            input_shape.clear()
    input_dimcount = len(input_shape)
    src_in = np.random.normal(zero_point, std, input_shape)

    out_calcu = tf.broadcast_to(src_in, shape=broadcast_shape)
    sess = tf.Session()
    src_out = sess.run(out_calcu)

    src_in_1  = src_in.ravel('C')
    src_out_1 = src_out.flatten()

    total_size = (len(src_in_1) + len(src_out_1)) + 2 + input_dimcount + broadcast_dimcount

    para.append(total_size)
    para.append(input_dimcount)
    para.append(broadcast_dimcount)
    for i in range(0, input_dimcount):
        para.append(input_shape[i])
    for i in range(0, broadcast_dimcount):
        para.append(broadcast_shape[i])
    print(para)
    print(input_shape)
    print(broadcast_shape)

    with open("broadcast_to_data_f32.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%df' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%df' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    broadcast_to_f32()
    print("end")
