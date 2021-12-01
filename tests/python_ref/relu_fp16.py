#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import struct
import numpy as np
import tensorflow as tf

def relu_fp16():
    para = []
    # init the input data and parameters
    batch      = int(np.random.randint(1, high=2, size=1))
    in_size_x  = int(np.random.randint(4, high=16, size=1))
    in_size_y  = int(np.random.randint(4, high=16, size=1))
    in_channel = int(np.random.randint(4, high=8, size=1))

    src_in = np.random.uniform(-1, 2, (batch, in_channel, in_size_y, in_size_x))
    src_in = src_in.astype(np.float16)

    out_calcu = tf.nn.relu(tf.convert_to_tensor(src_in))

    sess = tf.Session()

    src_out = sess.run(out_calcu)

    size_all = batch*in_size_y*in_size_x*in_channel
    src_in_1  = src_in.reshape(size_all)
    src_out_1 = src_out.reshape(size_all)

    total_size = (len(src_in_1) + len(src_out_1)) + 4

    para.append(total_size)
    para.append(batch)
    para.append(in_size_y)
    para.append(in_size_x)
    para.append(in_channel)
    print(para)

    with open("relu_data_fp16.bin", "wb") as fp:
        data = struct.pack(('%di' % len(para)), *para)
        fp.write(data)
        data = struct.pack(('%de' % len(src_in_1)), *src_in_1)
        fp.write(data)
        data = struct.pack(('%de' % len(src_out_1)), *src_out_1)
        fp.write(data)
        fp.close()

    return 0


if __name__ == '__main__':
    relu_fp16()
    print("end")
